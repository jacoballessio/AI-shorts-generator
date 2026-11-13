from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _resolve_base_model(adapter_path: Path, override: str | None) -> str:
    if override:
        return override

    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing adapter config: {config_path}")

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    base = payload.get("base_model_name_or_path")
    if not base:
        raise ValueError(f"base_model_name_or_path not found in {config_path}")
    return str(base)


def _patch_yarn_config(cfg: Any) -> Any:
    rope = getattr(cfg, "rope_scaling", None)
    if isinstance(rope, dict) and rope.get("type") == "yarn" and rope.get("head_dim") is None:
        hidden = getattr(cfg, "hidden_size", None)
        heads = getattr(cfg, "num_attention_heads", None)
        if hidden and heads:
            head_dim = hidden // heads
            rope["head_dim"] = head_dim
            try:
                object.__setattr__(cfg, "rope_scaling", dict(rope, head_dim=head_dim))
            except Exception:
                cfg.rope_scaling = dict(rope, head_dim=head_dim)
            try:
                object.__setattr__(cfg, "head_dim", head_dim)
            except Exception:
                pass
    return cfg


def merge_adapter(
    adapter_path: Path,
    output_path: Path,
    base_model: str,
    trust_remote_code: bool,
    clean_output: bool,
) -> None:
    print(f"Adapter: {adapter_path}")
    print(f"Base model: {base_model}")
    print(f"Output: {output_path}")

    if clean_output and output_path.exists():
        print(f"Cleaning existing output folder: {output_path}")
        shutil.rmtree(output_path)

    # Prefer local tokenizer files from adapter folder to avoid hub metadata edge cases.
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path),
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )

    cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    cfg = _patch_yarn_config(cfg)

    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        config=cfg,
        trust_remote_code=trust_remote_code,
        device_map="auto",
        dtype="auto",
    )

    lora_model = PeftModel.from_pretrained(base_model_obj, str(adapter_path))
    merged_model = lora_model.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print("Merge complete")


def upload_to_hf(
    model_dir: Path,
    repo_id: str,
    private: bool,
    create_repo: bool,
    token: str | None,
) -> None:
    api = HfApi(token=token)
    if create_repo:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    # Use upload_large_folder for better retry logic and resumable uploads on large models
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(model_dir),
        commit_message="Upload merged model",
    )
    print(f"Upload complete: https://huggingface.co/{repo_id}")


def parse_args() -> argparse.Namespace:
    default_adapter = Path("longcontext-20260324_025228/emergency_checkpoint")
    default_output = Path("longcontext-20260324_025228/merged_model")

    parser = argparse.ArgumentParser(description="Merge LoRA checkpoint and optionally upload to Hugging Face.")
    parser.add_argument("--adapter", type=Path, default=default_adapter, help="Path to PEFT adapter checkpoint")
    parser.add_argument("--output", type=Path, default=default_output, help="Where to save merged model")
    parser.add_argument("--base-model", type=str, default=None, help="Override base model id/path")
    parser.add_argument("--repo-id", type=str, default=None, help="Hugging Face repo id like user/model")
    parser.add_argument("--private", action="store_true", help="Create repo as private when --repo-id is used")
    parser.add_argument(
        "--no-create-repo",
        action="store_true",
        help="Do not call create_repo before uploading",
    )
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token (optional)")
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable trust_remote_code when loading base model/tokenizer",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete output folder before merge to avoid stale artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_path = args.adapter.resolve()
    output_path = args.output.resolve()

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    base_model = _resolve_base_model(adapter_path, args.base_model)
    trust_remote_code = not args.no_trust_remote_code

    merge_adapter(
        adapter_path=adapter_path,
        output_path=output_path,
        base_model=base_model,
        trust_remote_code=trust_remote_code,
        clean_output=args.clean_output,
    )

    if args.repo_id:
        token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        upload_to_hf(
            model_dir=output_path,
            repo_id=args.repo_id,
            private=args.private,
            create_repo=not args.no_create_repo,
            token=token,
        )
    else:
        print("Skipping upload because --repo-id was not provided")


if __name__ == "__main__":
    main()
