"""
Long Context LLM Training with LoRA

This script fine-tunes a long-context model using LoRA (Low-Rank Adaptation) for efficient training on consumer GPUs.

LoRA Benefits:
- Trains only 0.1-1% of model parameters (vs 100% full fine-tuning)
- Fits 7B models on 24GB VRAM (4090) instead of requiring 80GB
- Faster training and lower memory
- Can be merged back into base model or used as adapters

Basic Usage:
    python train_llm_longcontext_lora.py --train-file your_data.txt

Common Options:
    --context-length 4000          # Sequence length (default: 4000)
    --lora-r 16                    # LoRA rank (8-64, higher=more params, default: 16)
    --lora-alpha 32                # LoRA alpha (usually 2x rank, default: 32)
    --use-4bit                     # Use 4-bit quantization (max memory savings)
    --max-examples 1000            # Limit dataset size for testing
    --resume-from checkpoint-500   # Resume from checkpoint

Memory Usage Examples (RTX 4090 24GB):
    Normal (bfloat16):              ~18-20GB VRAM, batch_size=2
    With --use-4bit:                ~12-14GB VRAM, batch_size=2-4
    Context 8000 + 4-bit:           ~16-18GB VRAM, batch_size=1-2

Installation:
    pip install peft bitsandbytes accelerate

Loading Trained Model:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    
    base_model = AutoModelForCausalLM.from_pretrained("SuperAGI/mistral-7B-PoSE-32k")
    model = PeftModel.from_pretrained(base_model, "./longcontext-20241118_123456")
    
    # Or use the merged model (if merge succeeded):
    model = AutoModelForCausalLM.from_pretrained("./longcontext-20241118_123456/merged_model")
"""

import os
import psutil
import datetime
import shutil
import sys
import time
import re
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
# Disable anomaly detection for stability
# torch.autograd.set_detect_anomaly(True)
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from peft import PeftModel, PeftConfig


def load_tokenizer_robust(tokenizer_path: str, model_name: str, trust_remote_code: bool = True):
    """Load tokenizer with fallback for additional_chat_templates 404 and local snapshot retry."""
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    def _try(path: str, local_only: bool = False):
        kwargs = {
            "trust_remote_code": trust_remote_code,
        }
        if hf_token:
            kwargs["token"] = hf_token
        if local_only:
            kwargs["local_files_only"] = True
        return AutoTokenizer.from_pretrained(path, **kwargs)

    # First attempt: direct load from requested tokenizer path.
    try:
        return _try(tokenizer_path)
    except Exception as e1:
        print(f"Warning: Could not load tokenizer from {tokenizer_path}, trying original model {model_name}")

    # Second attempt: original model path.
    try:
        return _try(model_name)
    except Exception as e2:
        err_text = f"{e1}\n{e2}" if 'e1' in locals() else str(e2)
        # Common new-stack issue: additional_chat_templates endpoint may 404.
        if "additional_chat_templates" in err_text:
            try:
                from huggingface_hub import snapshot_download

                print("[INFO] Tokenizer remote template lookup failed; retrying from local snapshot cache...")
                local_dir = snapshot_download(repo_id=model_name, token=hf_token)
                return _try(local_dir, local_only=True)
            except Exception as e3:
                raise RuntimeError(
                    "Tokenizer load failed after local snapshot retry. "
                    f"Initial errors: {err_text}; Snapshot error: {e3}"
                ) from e3

        raise

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-examples', type=int, default=None, help='Limit number of examples for testing')
    parser.add_argument('--context-length', type=int, default=4000, help='Context length for long sequences (default: 4000 — covers ~54%% of examples fully)')
    parser.add_argument('--resume-from', type=str, default=None, help='Resume training from a checkpoint directory (e.g., longcontext-666)')
    parser.add_argument('--train-file', type=str, default='data/training_dataset_xmlish.txt', help='Training text file to use')
    parser.add_argument('--resume-training-state', action='store_true', help='Resume training state (optimizer, scheduler) from checkpoint - default is to only load model weights')
    parser.add_argument('--lora-r', type=int, default=16, help='LoRA rank (default: 16, higher = more parameters but better quality)')
    parser.add_argument('--lora-alpha', type=int, default=32, help='LoRA alpha (default: 32, usually 2x rank)')
    parser.add_argument('--lora-dropout', type=float, default=0.05, help='LoRA dropout (default: 0.05)')
    parser.add_argument('--use-4bit', action='store_true', help='Use 4-bit quantization for even lower memory (may reduce quality)')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate (default: 2e-5; original was 2e-4 which caused fragmentation)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs (default: 3; original was 30)')
    parser.add_argument('--warmup-steps', type=int, default=50, help='Warmup steps (default: 50)')
    parser.add_argument('--model', type=str, default='SuperAGI/mistral-7B-PoSE-32k', 
                       help='Base model to use (default: SuperAGI/mistral-7B-PoSE-32k). Alternatives: mistralai/Mistral-7B-v0.1, NousResearch/Yarn-Mistral-7b-128k')
    args = parser.parse_args()

    model_name = args.model  # Use model from command line
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine model path and output directory
    if args.resume_from:
        if os.path.exists(args.resume_from):
            # Check if it's a checkpoint directory or model directory
            if os.path.basename(args.resume_from).startswith('checkpoint-'):
                # It's a checkpoint, use the checkpoint itself for model loading
                model_path = args.resume_from
                output_dir = os.path.dirname(args.resume_from)
            else:
                # It's a model directory, check if it has model files or is just a checkpoint container
                model_files = ['pytorch_model.bin', 'model.safetensors', 'config.json']
                has_model_files = any(os.path.exists(os.path.join(args.resume_from, f)) for f in model_files)
                
                if has_model_files:
                    # It's a complete model directory
                    model_path = args.resume_from
                    output_dir = args.resume_from
                else:
                    # It's likely just a checkpoint container, use the original base model
                    model_path = model_name
                    output_dir = args.resume_from
                    print(f"Model directory {args.resume_from} doesn't contain model files, using base model {model_name}")
            
            print(f"Model path: {model_path}")
            print(f"Output directory: {output_dir}")
        else:
            print(f"Error: Resume path '{args.resume_from}' does not exist!")
            sys.exit(1)
    else:
        model_path = model_name
        output_dir = f"./longcontext-{timestamp}"
        print(f"Starting new training with model: {model_name}")

    print("Loading tokenizer...")
    # Always use the original model for tokenizer, even when resuming from checkpoint
    tokenizer_path = model_name if args.resume_from else model_path
    tokenizer = load_tokenizer_robust(tokenizer_path, model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use a more conservative context length that won't cause segfaults
    context_length = min(args.context_length, 8000)  # Cap at 8k for stability
    print(f"Using context length: {context_length}")
    
    if context_length > 2000:
        print(f"\n⚠️  WARNING: Long context ({context_length}) may cause training to hang on first step.")
        print(f"   If training freezes, try starting with --context-length 2000 to verify setup works.")
        print(f"   You can increase context length once training is stable.\n")

    # Safety guard: on 24GB cards, long context without quantization often OOMs.
    if torch.cuda.is_available() and context_length >= 3000 and not args.use_4bit:
        print("⚠️  Auto-enabling --use-4bit for long-context training to avoid CUDA OOM.")
        args.use_4bit = True

    print("Loading model...")
    # Load with more memory-efficient settings and optional quantization
    load_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    
    if args.use_4bit:
        # Use 4-bit quantization for maximum memory savings
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        print("Using 4-bit quantization for maximum memory efficiency")
    else:
        # Use bfloat16 for better precision without quantization (use 'dtype' not 'torch_dtype')
        load_kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        print(f"Using {'bfloat16' if torch.cuda.is_available() else 'float32'} precision")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying to fix model configuration...")
        
        # Try to load config and fix it
        # IMPORTANT: For checkpoints, load config from BASE MODEL, not checkpoint
        # Checkpoints don't have model_type in config.json
        from transformers import AutoConfig
        try:
            config_source = model_name  # Always use base model for config
            print(f"Loading config from base model: {config_source}")
            config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
            
            # Fix common config issues
            if hasattr(config, 'head_dim') and config.head_dim is None:
                # Calculate head_dim from hidden_size and num_attention_heads
                if hasattr(config, 'hidden_size') and hasattr(config, 'num_attention_heads'):
                    config.head_dim = config.hidden_size // config.num_attention_heads
                    print(f"Fixed head_dim: {config.head_dim}")
            
            # Try loading checkpoint weights with base model config
            load_kwargs["config"] = config
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
            print("✅ Model loaded with fixed configuration")
        except Exception as e2:
            print(f"Config fix failed: {e2}")
            print("Falling back to simpler model loading...")
            # Last resort: minimal loading
            load_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto" if torch.cuda.is_available() else None
            }
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
            print("✅ Model loaded with minimal configuration")
    
    # Prepare model for LoRA training
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
        print("Model prepared for k-bit training")

    # Disable KV cache during training to reduce activation memory.
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
        print("Disabled model KV cache for training (use_cache=False)")
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,  # Rank - determines how many parameters to add (16 is good balance)
        lora_alpha=args.lora_alpha,  # Scaling factor (usually 2x rank)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Apply to all attention and MLP layers
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Shows how many parameters are actually being trained
    
    print(f"LoRA configuration:")
    print(f"  Rank (r): {args.lora_r}")
    print(f"  Alpha: {args.lora_alpha}")
    print(f"  Dropout: {args.lora_dropout}")
    print(f"  4-bit quantization: {args.use_4bit}")

    # Use the multimodal described data as the training text
    train_file = args.train_file
    print(f"Creating dataset from {train_file}...")
    
    # Validate that the training file exists
    if not os.path.exists(train_file):
        print(f"Error: Training file '{train_file}' does not exist!")
        sys.exit(1)
    
    # Check if file is empty
    file_size = os.path.getsize(train_file)
    if file_size == 0:
        print(f"Error: Training file '{train_file}' is empty!")
        sys.exit(1)
    
    print(f"Training file size: {file_size / 1024 / 1024:.2f} MB")
    
    # Read the file and check format
    with open(train_file, 'r', encoding='utf-8') as f:
        first_lines = [f.readline() for _ in range(5)]
    
    print(f"First line preview: {first_lines[0][:100]}...")
    
    # Load the text data
    try:
        raw_dataset = load_dataset('text', data_files={'train': train_file}, split='train')
        print(f"Dataset loaded: {len(raw_dataset)} lines")
        
        # Check if dataset is empty
        if len(raw_dataset) == 0:
            raise ValueError("Dataset is empty after loading")
            
    except Exception as e:
        print(f"Error loading dataset with default method: {e}")
        print("Trying alternative loading method...")
        
        # Alternative: Read file manually and create dataset
        print("Reading file manually...")
        with open(train_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Filter out empty lines
        lines = [line.strip() for line in lines if line.strip()]
        
        if not lines:
            print(f"Error: No valid lines found in {train_file}")
            sys.exit(1)
        
        print(f"Found {len(lines)} non-empty lines")
        
        # Create dataset from lines
        from datasets import Dataset
        raw_dataset = Dataset.from_dict({'text': lines})
        print(f"Created dataset from {len(lines)} lines")
    
    # -----------------------------------------------------------------------
    # Per-example tokenization strategy.
    #
    # The old approach (group_texts) concatenated everything then sliced into
    # fixed-length chunks, which broke <example> boundaries arbitrarily.  The
    # model therefore never saw a complete <input>→<output> pair — only random
    # fragments — so it learned surface XML structure but not content.
    #
    # New approach: split the file by <example> blocks, tokenize each block
    # individually, and truncate to context_length from the *end* so we always
    # keep the <input> section (the conditioning signal) plus as much <output>
    # as fits.  Multiple short examples are then greedily packed into one
    # sequence (bin-packing) to avoid wasting context capacity.
    # -----------------------------------------------------------------------

    print("Splitting training file into individual <example> blocks...")

    def split_into_examples(path: str) -> list[str]:
        """Split the flat text file into individual <example>…</example> blocks."""
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        # Split on the start tag so each chunk begins with <example
        parts = re.split(r'(?=<example\b)', text)
        blocks = []
        for part in parts:
            part = part.strip()
            if part.startswith('<example'):
                blocks.append(part)
        return blocks

    example_blocks = split_into_examples(train_file)
    print(f"Found {len(example_blocks)} <example> blocks")

    if not example_blocks:
        print("Error: no <example> blocks found in training file — is it the right format?")
        sys.exit(1)

    # Tokenize each example individually (truncation keeps the front = <input> side)
    print("Tokenizing examples individually...")
    all_input_ids = []
    skipped = 0
    for block in example_blocks:
        ids = tokenizer.encode(block, add_special_tokens=False)
        if len(ids) == 0:
            skipped += 1
            continue
        # Truncate to fit; keep the *beginning* (input + first segments)
        ids = ids[:context_length]
        all_input_ids.append(ids)

    print(f"Tokenized {len(all_input_ids)} examples ({skipped} skipped empty), "
          f"{sum(1 for ids in all_input_ids if len(ids) == context_length)} were truncated")

    # Greedily pack short examples into full-length sequences to fill context
    print("Packing examples into context-length sequences...")
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    packed_sequences = []
    current: list[int] = []
    for ids in all_input_ids:
        # Separate examples with EOS token
        needed = len(ids) + (1 if current else 0)
        if current and len(current) + needed > context_length:
            # Pad or store current and start fresh
            packed_sequences.append(current[:])
            current = []
        if current:
            current.append(eos_id)  # separator between packed examples
        current.extend(ids)
    if current:
        packed_sequences.append(current)

    print(f"Packed into {len(packed_sequences)} training sequences")

    # Build final dataset with labels = input_ids (causal LM)
    from datasets import Dataset
    lm_dataset = Dataset.from_dict({
        'input_ids': packed_sequences,
        'attention_mask': [[1] * len(seq) for seq in packed_sequences],
        'labels': [seq[:] for seq in packed_sequences],
    })

    print(f"Final dataset size: {len(lm_dataset)} packed sequences (context_length={context_length})")
    
    # Optionally limit examples for testing
    if args.max_examples is not None:
        original_size = len(lm_dataset)
        if args.max_examples < original_size:
            lm_dataset = lm_dataset.select(range(args.max_examples))
            print(f"Limited dataset from {original_size} to {len(lm_dataset)} examples")
    
    # Split into train/validation
    print("Splitting dataset into train/validation...")
    split_dataset = lm_dataset.train_test_split(test_size=0.01, seed=42)  # Only 1% for eval to prevent hangs
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    print(f"Train examples: {len(train_dataset)}, Eval examples: {len(eval_dataset)}")

    # Use standard data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal language modeling
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,  # Reduce to 1 to avoid hanging
        gradient_accumulation_steps=8,  # Increase to compensate
        save_steps=300,
        save_total_limit=2,
        prediction_loss_only=True,
        bf16=torch.cuda.is_available() and not args.use_4bit,
        fp16=False,
        logging_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        # 8-bit paged optimizer significantly reduces optimizer-state VRAM.
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        max_grad_norm=1.0,
        logging_dir=f"{output_dir}/logs",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="none",
        eval_strategy="steps",
        eval_steps=24501,
        load_best_model_at_end=False,
        dataloader_drop_last=False,
        ddp_find_unused_parameters=False,
        skip_memory_metrics=True,
        ignore_data_skip=True,
        torch_compile=False,
        # Additional debugging settings
        logging_first_step=True,
        logging_nan_inf_filter=False,  # Don't filter, we want to see issues
    )

    print("Starting training...")
    class LossCallback(transformers.TrainerCallback):
        def __init__(self):
            self.step_start_time = None
            
        def on_step_begin(self, args, state, control, **kwargs):
            import time
            self.step_start_time = time.time()
            if state.global_step % 10 == 0:
                print(f"\n[Step {state.global_step}] Starting...")
        
        def on_step_end(self, args, state, control, **kwargs):
            import time
            if self.step_start_time:
                step_time = time.time() - self.step_start_time
                if state.global_step % 10 == 0:
                    print(f"[Step {state.global_step}] Completed in {step_time:.2f}s")
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                if 'loss' in logs:
                    print(f"Step {state.global_step}: train_loss = {logs['loss']:.4f}")
                if 'eval_loss' in logs:
                    print(f"Step {state.global_step}: eval_loss = {logs['eval_loss']:.4f}")

    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all CUDA operations to complete
        print(f"GPU memory before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
        # Test a simple forward pass to make sure model works
        print("\nTesting model with a simple forward pass...")
        try:
            test_input = tokenizer("Hello world", return_tensors="pt", max_length=100, truncation=True)
            test_input = {k: v.to(model.device) for k, v in test_input.items()}
            with torch.no_grad():
                test_output = model(**test_input)
            print("✅ Model forward pass successful")
            del test_input, test_output
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ Model forward pass failed: {e}")
            print("This suggests a model/CUDA issue. Try:")
            print("  1. Restart Python")
            print("  2. Use --use-4bit flag")
            print("  3. Reduce --context-length to 2000")
            raise

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[LossCallback()]
    )
    
    print("\n" + "="*80)
    print("STARTING TRAINING - First step may take 30-60 seconds to initialize...")
    print("="*80 + "\n")
    
    try:
        # Determine if we should resume from checkpoint
        resume_from_checkpoint = None
        if args.resume_from and os.path.exists(args.resume_from) and args.resume_training_state:
            # Only resume training state if explicitly requested
            # Check if the path is directly a checkpoint directory
            if os.path.basename(args.resume_from).startswith('checkpoint-'):
                resume_from_checkpoint = args.resume_from
                print(f"Resuming training state from checkpoint: {resume_from_checkpoint}")
            else:
                # Look for the latest checkpoint in the resume directory
                checkpoints = [d for d in os.listdir(args.resume_from) if d.startswith('checkpoint-') and os.path.isdir(os.path.join(args.resume_from, d))]
                if checkpoints:
                    # Sort by checkpoint number and get the latest
                    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
                    resume_from_checkpoint = os.path.join(args.resume_from, latest_checkpoint)
                    print(f"Resuming training state from checkpoint: {resume_from_checkpoint}")
                else:
                    print(f"No checkpoints found in {args.resume_from}, starting fresh training")
        
        if args.resume_from and not args.resume_training_state:
            print(f"Starting fresh training with model weights from {model_path} (training state will NOT be resumed)")
        
        # Print training configuration before starting
        print(f"Training configuration:")
        print(f"  Total train examples: {len(train_dataset)}")
        print(f"  Total eval examples: {len(eval_dataset)}")
        print(f"  Epochs: {training_args.num_train_epochs}")
        print(f"  Steps per epoch: {len(train_dataset) // training_args.gradient_accumulation_steps}")
        print(f"  Total training steps: {len(train_dataset) * training_args.num_train_epochs // training_args.gradient_accumulation_steps}")
        print(f"  Resume training state: {args.resume_training_state}")
        print(f"  Resume from checkpoint: {resume_from_checkpoint}")
        
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        print(f"Saving LoRA adapters to {output_dir}...")
        # Save LoRA adapters
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Also save a merged model (optional - combines base model + LoRA)
        print(f"Merging LoRA adapters with base model...")
        merged_model_dir = f"{output_dir}/merged_model"
        try:
            # Merge LoRA weights back into the base model
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(merged_model_dir)
            tokenizer.save_pretrained(merged_model_dir)
            print(f"Merged model saved to {merged_model_dir}")
        except Exception as e:
            print(f"Could not merge model (this is OK, LoRA adapters are still saved): {e}")
        
        print("Training complete!")
        print(f"LoRA adapters saved to: {output_dir}")
        print(f"To use the model, load it with:")
        print(f"  from peft import PeftModel")
        print(f"  base_model = AutoModelForCausalLM.from_pretrained('{model_name}')")
        print(f"  model = PeftModel.from_pretrained(base_model, '{output_dir}')")
    except (Exception, KeyboardInterrupt) as e:
        print(f"\nTraining stopped: {type(e).__name__}: {e}")
        # Save what we can, including on Ctrl+C (KeyboardInterrupt)
        if hasattr(trainer, 'state') and trainer.state.global_step > 0:
            print(f"Saving emergency checkpoint at step {trainer.state.global_step}...")
            try:
                trainer.save_model(f"{output_dir}/emergency_checkpoint")
                tokenizer.save_pretrained(f"{output_dir}/emergency_checkpoint")
                print(f"Emergency checkpoint saved to {output_dir}/emergency_checkpoint")
                print(f"Resume with: --resume-from {output_dir}/emergency_checkpoint --resume-training-state")
            except Exception as save_err:
                print(f"Could not save emergency checkpoint: {save_err}")
        else:
            print("No steps completed, nothing to save.")
        raise

if __name__ == "__main__":
    main()
