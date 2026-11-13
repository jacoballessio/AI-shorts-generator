"""
Train LoRA model on YouTuber-style dataset.
"""
import os
import json
from pathlib import Path
from typing import Dict, List
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import wandb


class YouTuberTrainer:
    """Fine-tune LLM on YouTuber-style content."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct",
        output_dir: str = "models/youtuber_lora"
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model_and_tokenizer(self):
        """Load base model with 4-bit quantization."""
        
        print(f"Loading {self.model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load model with 4-bit quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        print("Model loaded!")
        
    def setup_lora(self, rank: int = 64, alpha: int = 128):
        """Configure LoRA for parameter-efficient fine-tuning."""
        
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"\nTrainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
    def prepare_dataset(self, dataset_path: str) -> Dataset:
        """Load and format dataset for training."""
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to instruction format
        formatted_data = []
        
        for example in data:
            # Create instruction-style prompt
            prompt = self._format_training_example(example)
            formatted_data.append({"text": prompt})
        
        # Create HuggingFace dataset
        dataset = Dataset.from_list(formatted_data)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=8192,
                padding="max_length"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def _format_training_example(self, example: Dict) -> str:
        """Format training example as instruction prompt."""
        
        # Extract input
        sources_text = "\n\n".join([
            f"Source {i+1}: {s['title']}\n{s['text']}"
            for i, s in enumerate(example['input']['sources'])
        ])
        
        # Extract output
        segments_text = json.dumps(example['output']['segments'], indent=2)
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a video script generator trained on {example['input']['creator']}'s style. Create engaging video scripts from research sources.

<|start_header_id|>user<|end_header_id|>

Create a {example['input']['duration']}s video script on {example['input']['topic']}.

Sources:
{sources_text}

Generate a JSON script with segments including: timestamp, text, visual descriptions, and source references.

<|start_header_id|>assistant<|end_header_id|>

{segments_text}<|eot_id|>"""
        
        return prompt
    
    def train(
        self,
        dataset_path: str,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        use_wandb: bool = False
    ):
        """Train the model."""
        
        # Initialize wandb if requested
        if use_wandb:
            wandb.init(project="youtuber-training", name=f"lora-{self.model_name.split('/')[-1]}")
        
        # Prepare dataset
        print("\nPreparing dataset...")
        dataset = self.prepare_dataset(dataset_path)
        
        # Split into train/val
        train_test_split = dataset.train_test_split(test_size=0.1)
        train_dataset = train_test_split['train']
        val_dataset = train_test_split['test']
        
        print(f"Training examples: {len(train_dataset)}")
        print(f"Validation examples: {len(val_dataset)}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            report_to="wandb" if use_wandb else "none",
            fp16=True,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print("\nStarting training...")
        trainer.train()
        
        # Save final model
        print("\nSaving final model...")
        trainer.save_model(str(self.output_dir / "final"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final"))
        
        print(f"\nTraining complete! Model saved to {self.output_dir / 'final'}")


if __name__ == "__main__":
    # Example usage
    trainer = YouTuberTrainer(
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
        output_dir="models/veritasium_lora"
    )
    
    # Load model
    # trainer.load_model_and_tokenizer()
    
    # Setup LoRA
    # trainer.setup_lora(rank=64, alpha=128)
    
    # Train
    # trainer.train(
    #     dataset_path="data/training_dataset/youtuber_dataset.json",
    #     num_epochs=3,
    #     batch_size=4,
    #     learning_rate=2e-5,
    #     use_wandb=True
    # )
    
    print("Trainer ready. Uncomment examples to start training.")
