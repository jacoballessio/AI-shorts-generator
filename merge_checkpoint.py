from __future__ import annotations

from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "SuperAGI/mistral-7B-PoSE-32k"
ADAPTER_PATH = Path("longcontext-20260324_025228/checkpoint-300")
OUTPUT_PATH = Path("longcontext-20260324_025228/merged_model")


def main() -> None:
    print(f"Adapter: {ADAPTER_PATH}")
    print(f"Base model: {BASE_MODEL}")
    print(f"Output: {OUTPUT_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
    )

    lora_model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
    merged_model = lora_model.merge_and_unload()

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(OUTPUT_PATH))
    tokenizer.save_pretrained(str(OUTPUT_PATH))

    print("Merge complete")


if __name__ == "__main__":
    main()
