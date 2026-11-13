import re
from datetime import datetime
from pathlib import Path
import argparse

DEFAULT_INPUT_PATH = "data/training_dataset_xmlish.txt"

# Match complete <sources>...</sources> blocks.
SOURCES_BLOCK_PATTERN = re.compile(
    r"(<sources\b[^>]*>)([\s\S]*?)(</sources>)",
    re.IGNORECASE,
)

# Match individual <source ...>...</source> entries inside a sources block.
SOURCE_ENTRY_PATTERN = re.compile(
    r"<source\b[^>]*>[\s\S]*?</source>",
    re.IGNORECASE,
)

# Match Google News RSS links in url attributes.
GOOGLE_RSS_PATTERN = re.compile(
    r"https?://(?:www\.)?news\.google\.com/rss(?:[^\s\"<]*)",
    re.IGNORECASE,
)

URL_ATTR_PATTERN = re.compile(r"\burl\s*=\s*([\"'])(.*?)\1", re.IGNORECASE)


def is_google_rss_source(source_entry: str) -> bool:
    """Return True if a <source> entry points to news.google.com/rss."""
    match = URL_ATTR_PATTERN.search(source_entry)
    if not match:
        return False
    url = match.group(2)
    return bool(GOOGLE_RSS_PATTERN.search(url))


def clean_sources_block(block_match: re.Match, counters: dict) -> str:
    """Remove only Google RSS <source> entries while preserving other sources."""
    opening_tag, inner, closing_tag = block_match.groups()

    entries = SOURCE_ENTRY_PATTERN.findall(inner)
    if not entries:
        return block_match.group(0)

    kept_entries = []
    removed_here = 0

    for entry in entries:
        if is_google_rss_source(entry):
            removed_here += 1
        else:
            kept_entries.append(entry)

    if removed_here == 0:
        return block_match.group(0)

    counters["blocks_touched"] += 1
    counters["sources_removed"] += removed_here

    return f"{opening_tag}{''.join(kept_entries)}{closing_tag}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove news.google.com/rss <source> entries from XML-ish dataset files."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="Input file path (default: data/training_dataset_xmlish.txt)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path. If omitted, updates input file in-place with backup.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    counters = {"blocks_touched": 0, "sources_removed": 0}

    updated_content = SOURCES_BLOCK_PATTERN.sub(
        lambda m: clean_sources_block(m, counters),
        content,
    )

    backup_path = None
    if output_path.resolve() == input_path.resolve():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = input_path.with_name(f"{input_path.stem}.backup_{ts}{input_path.suffix}")
        backup_path.write_text(content, encoding="utf-8")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(updated_content)

    print(
        f"Done. Removed {counters['sources_removed']} source entries "
        f"across {counters['blocks_touched']} <sources> blocks."
    )
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    if backup_path:
        print(f"Backup: {backup_path}")


if __name__ == "__main__":
    main()