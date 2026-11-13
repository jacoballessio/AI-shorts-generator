#!/usr/bin/env python3
"""
Run final spacing cleanup on existing example JSON files.

This script reuses the same per-field spacing fix logic from
generate_example_with_carmind2_lora.py so existing JSONs can be cleaned without
regenerating XML-ish output.

Usage examples:
  python fix_existing_json_spacing.py --file new_example.xmlish.json
  python fix_existing_json_spacing.py --dir data/training_dataset --glob "*_example.json"
  python fix_existing_json_spacing.py --file new_example.xmlish.json --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_example_with_carmind2_lora import (
    _final_json_spacing_fix,
)


def _collect_paths(single_file: Path | None, in_dir: Path | None, pattern: str) -> list[Path]:
    if single_file is not None:
        return [single_file]
    if in_dir is None:
        raise ValueError("Provide either --file or --dir")
    return sorted(p for p in in_dir.glob(pattern) if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix spacing artifacts in existing example JSON files")
    parser.add_argument("--file", type=Path, default=None, help="Single JSON file to process")
    parser.add_argument("--dir", type=Path, default=None, help="Directory to scan for JSON files")
    parser.add_argument("--glob", type=str, default="*.json", help="Glob pattern for --dir (default: *.json)")

    parser.add_argument("--dry-run", action="store_true", help="Show stats only; do not modify files")
    parser.add_argument("--backup", action="store_true", help="Write <file>.bak before overwriting")
    args = parser.parse_args()

    if args.file is None and args.dir is None:
        raise SystemExit("Provide either --file or --dir")

    if args.file is not None and not args.file.exists():
        raise SystemExit(f"File not found: {args.file}")

    if args.dir is not None and not args.dir.exists():
        raise SystemExit(f"Directory not found: {args.dir}")

    print("Running deterministic spacing repair (no LLM/model loading)...")

    paths = _collect_paths(args.file, args.dir, args.glob)
    if not paths:
        print("No files matched.")
        return

    total_seen = 0
    total_changed = 0
    total_llm_calls = 0
    files_changed = 0

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as exc:
            print(f"[WARN] Skipping {path}: {exc}")
            continue

        original = json.dumps(obj, ensure_ascii=False, sort_keys=True)
        obj, stats = _final_json_spacing_fix(obj)
        repaired = json.dumps(obj, ensure_ascii=False, sort_keys=True)

        total_seen += stats["fields_seen"]
        total_changed += stats["fields_changed"]
        total_llm_calls += stats["llm_calls"]

        changed = repaired != original
        if changed:
            files_changed += 1

        print(
            f"[INFO] {path.name}: "
            f"fields_seen={stats['fields_seen']} "
            f"fields_changed={stats['fields_changed']} "
            f"llm_calls={stats['llm_calls']}"
        )

        if changed and not args.dry_run:
            if args.backup:
                backup_path = path.with_suffix(path.suffix + ".bak")
                with open(backup_path, "w", encoding="utf-8") as bf:
                    bf.write(json.dumps(json.loads(original), indent=2, ensure_ascii=False))
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)

    print(
        "\nDone: "
        f"files={len(paths)}, files_changed={files_changed}, "
        f"fields_seen={total_seen}, fields_changed={total_changed}, llm_calls={total_llm_calls}"
    )


if __name__ == "__main__":
    main()
