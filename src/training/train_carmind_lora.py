"""
Fine-tune Carmind-2.0 with LoRA on local training data.

Supports:
- A directory containing `*_example.json` files
- A single `*_example.json` file
- A JSON list file (e.g. youtuber_dataset.json)
- A raw `.txt` corpus file (e.g. XML-ish serialized examples)

Example:
  python src/training/train_carmind_lora.py ^
    --data data/training_dataset ^
    --output models/carmind2_lora
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

# Disable HF symlinks warning before any HF imports (Windows without Developer Mode).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from datasets import Dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def load_documents(data_path: Path) -> List[str]:
    """Load training documents from JSON examples or a plain text corpus file."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    if data_path.is_dir():
        files = sorted(data_path.glob("*_example.json"))
        if not files:
            raise ValueError(f"No *_example.json files found in: {data_path}")

        examples: List[Dict[str, Any]] = []
        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                examples.append(json.load(f))
        return [build_full_json_text(example) for example in examples]

    if data_path.suffix.lower() == ".txt":
        text = data_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Text corpus is empty: {data_path}")
        return [text]

    with open(data_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return [build_full_json_text(item) for item in payload]
    if isinstance(payload, dict):
        return [build_full_json_text(payload)]

    raise ValueError(f"Unsupported JSON structure in: {data_path}")


def build_full_json_text(example: Dict[str, Any]) -> str:
    """Serialize an example exactly as JSON text for plain CLM training."""
    return json.dumps(example, ensure_ascii=False, indent=2)


def build_tokenized_dataset(
    documents: List[str],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dataset:
    """
    Build plain causal LM dataset from a concatenated full-JSON token stream.
    Every token is supervised (labels == input_ids), matching next-token CLM.
    """
    all_ids: List[int] = []
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    for text in documents:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_ids.extend(ids)
        all_ids.extend(eos)

    if not all_ids:
        raise ValueError("No tokenized data produced from examples")

    rows: List[Dict[str, List[int]]] = []
    for i in range(0, len(all_ids), max_length):
        chunk = all_ids[i : i + max_length]
        if len(chunk) < 2:
            continue
        rows.append(
            {
                "input_ids": chunk,
                "attention_mask": [1] * len(chunk),
                "labels": chunk.copy(),
            }
        )

    if not rows:
        raise ValueError("Token stream too small after chunking; increase data or lower --max-length")

    return Dataset.from_list(rows)


class SFTDataCollator:
    """Pad pretokenized examples for causal LM training."""

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = []
        attention_masks = []
        labels = []

        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            attention_masks.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def split_dataset(dataset: Dataset, eval_ratio: float, seed: int) -> tuple[Dataset, Dataset | None]:
    """Train/validation split that tolerates very small datasets."""
    if len(dataset) < 2 or eval_ratio <= 0:
        return dataset, None

    split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
    return split["train"], split["test"]


def _patch_hf_symlinks() -> None:
    """
    On Windows without Developer Mode, os.symlink raises WinError 1314.
    HuggingFace Hub's cache system calls _create_symlink internally; patch it
    so it falls back to a plain file copy when symlinks are unavailable.
    """
    try:
        import huggingface_hub.file_download as _fd
        _orig = _fd._create_symlink

        def _safe_symlink(src: str, dst: str, new_blob: bool = False) -> None:
            try:
                _orig(src, dst, new_blob=new_blob)
            except OSError:
                if not os.path.exists(dst):
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                    elif os.path.isdir(src):
                        shutil.copytree(src, dst)

        _fd._create_symlink = _safe_symlink
    except Exception:
        pass  # Non-critical; allow training to continue anyway.


def _safe_snapshot_download(repo_id: str, ignore_patterns: list[str], local_dir: Path) -> str:
    """
    Download a snapshot, writing files directly to local_dir.
    Passes `local_dir_use_symlinks=False` when the installed hub version supports
    it; otherwise relies on the _patch_hf_symlinks() copy-fallback.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {
        "repo_id": repo_id,
        "ignore_patterns": ignore_patterns,
        "local_dir": str(local_dir),
    }
    sig = inspect.signature(snapshot_download)
    if "local_dir_use_symlinks" in sig.parameters:
        kwargs["local_dir_use_symlinks"] = False
    return snapshot_download(**kwargs)


def load_model_with_adapter_fallback(
    model_name: str,
    quant_config: BitsAndBytesConfig,
    force_no_hub_adapter: bool,
    base_cache_dir: Path,
) -> AutoModelForCausalLM:
    """
    Load model in 4-bit, with a fallback for hubs that publish adapter artifacts.

    Some model repos include `adapter_model.safetensors` + `adapter_config.json`,
    which can trigger an internal transformers+quantization crash on Windows in
    certain versions. If that path fails, we retry from a local snapshot that
    excludes adapter artifacts.
    """
    # Patch HF hub symlink creation before any download attempt.
    _patch_hf_symlinks()

    local_dir = base_cache_dir / model_name.replace("/", "--")

    def _load_from(path_or_id: str) -> AutoModelForCausalLM:
        return AutoModelForCausalLM.from_pretrained(
            path_or_id,
            device_map="auto",
            quantization_config=quant_config,
            trust_remote_code=True,
        )

    if force_no_hub_adapter:
        print("Downloading model snapshot without adapter artifacts...")
        local_path = _safe_snapshot_download(
            repo_id=model_name,
            ignore_patterns=["adapter_model.safetensors", "adapter_config.json"],
            local_dir=local_dir,
        )
        return _load_from(local_path)

    try:
        return _load_from(model_name)
    except AttributeError as exc:
        message = str(exc)
        if "is not an nn.Module" not in message:
            raise

        print("Detected adapter-loading crash. Retrying without hub adapter artifacts...")
        local_path = _safe_snapshot_download(
            repo_id=model_name,
            ignore_patterns=["adapter_model.safetensors", "adapter_config.json"],
            local_dir=local_dir,
        )
        return _load_from(local_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Carmind-2.0")
    parser.add_argument("--data", type=Path, default=Path("data/training_dataset"))
    parser.add_argument("--model", type=str, default="Carmwin/Carmind-2.0")
    parser.add_argument("--output", type=Path, default=Path("models/carmind2_lora"))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--force-no-hub-adapter",
        action="store_true",
        help="Always load from a snapshot that excludes adapter artifacts on the Hub.",
    )
    parser.add_argument(
        "--base-cache-dir",
        type=Path,
        default=Path("models/base_models"),
        help="Local directory for downloaded model snapshots (avoids HF cache symlinks).",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    _patch_hf_symlinks()  # Apply before any HF Hub downloads (tokenizer, model, etc.)

    print(f"Loading training documents from: {args.data}")
    documents = load_documents(args.data)
    print(f"Loaded {len(documents)} document(s)")

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading base model in 4-bit")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = load_model_with_adapter_fallback(
        model_name=args.model,
        quant_config=quant_config,
        force_no_hub_adapter=args.force_no_hub_adapter,
        base_cache_dir=args.base_cache_dir,
    )

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable params: {trainable_params:,} / {total_params:,} "
        f"({100.0 * trainable_params / total_params:.2f}%)"
    )

    print("Training objective: full-stream causal LM on concatenated training text")
    dataset = build_tokenized_dataset(documents, tokenizer, args.max_length)
    train_dataset, eval_dataset = split_dataset(dataset, args.eval_ratio, args.seed)
    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset) if eval_dataset is not None else 0}")

    supports_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

    ta_kwargs = {
        "output_dir": str(args.output),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "logging_steps": 5,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 2,
        "eval_steps": 100 if eval_dataset is not None else None,
        "bf16": supports_bf16,
        "fp16": not supports_bf16,
        "report_to": "none",
        "optim": "paged_adamw_8bit",
        "seed": args.seed,
        "data_seed": args.seed,
        "remove_unused_columns": False,
    }

    # transformers has used both names across versions.
    ta_signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in ta_signature.parameters:
        ta_kwargs["evaluation_strategy"] = "steps" if eval_dataset is not None else "no"
    else:
        ta_kwargs["eval_strategy"] = "steps" if eval_dataset is not None else "no"

    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SFTDataCollator(tokenizer),
    )

    print("Starting training...")
    trainer.train()

    final_dir = args.output / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved LoRA adapter and tokenizer to: {final_dir}")


if __name__ == "__main__":
    main()
