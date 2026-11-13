"""Re-evaluate existing training examples to add/update sources.

Reads *_example.json files, runs SourceDetective (Google News RSS always;
NewsAPI as optional supplement when a key is configured), and writes the
discovered sources back into input.sources.

Usage:
    python add_sources.py                          # all files in data/training_dataset/
    python add_sources.py --video FlcFFsj_8Rg      # single video
    python add_sources.py --dir path/to/dir        # custom directory
    python add_sources.py --dry-run                # preview without saving
    python add_sources.py --overwrite              # replace existing sources (default: skip if sources present)

Environment variables:
    NEWS_API_KEY   (optional) newsapi.org API key for supplementary results
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

# Load environment variables from .env file (if present)
from dotenv import load_dotenv
load_dotenv()

from src.data_collection.source_detective import GoogleNewsRSSSearcher

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"  [ERROR] Could not load {path.name}: {exc}")
        return None


def _build_transcript(example: dict) -> str:
    """Concatenate per-segment transcripts into a single string."""
    parts = [
        seg.get("transcript", "").strip()
        for seg in example.get("output", {}).get("segments", [])
    ]
    return " ".join(p for p in parts if p)


def _format_source(s: dict) -> Optional[dict]:
    """Convert a raw SourceDetective dict to the training-example format."""
    url = s.get("url") or (s["content"] if s.get("content", "").startswith("http") else None)
    if not url:
        return None
    out: dict = {
        "type": s.get("type", "url"),
        "url": url,
        "title": s.get("title") or s.get("citation") or "",
        "confidence": round(s.get("confidence", 0.5), 2),
    }
    # Include the NewsAPI description snippet if present
    if s.get("description"):
        out["description"] = s["description"]
    if s.get("source_name"):
        out["source_name"] = s["source_name"]
    if s.get("published_at"):
        out["published_at"] = s["published_at"]
    return out


def _rank_sources(sources: List[dict], title: str, transcript: str, max_sources: int = 5) -> List[dict]:
    """Rank sources by relevance to the video title/transcript and return the top max_sources.

    Explicit URL citations (type='url') are always prioritised.  Within each
    tier, sources are scored by term overlap with the video title (weighted 2×)
    and the transcript, plus the existing confidence value.
    """
    if len(sources) <= max_sources:
        return sources

    import re as _re

    _STOP = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "is", "was", "are", "were", "has", "have", "had",
        "that", "this", "these", "those", "it", "its", "by", "as", "be",
        "from", "not", "he", "she", "we", "they", "who", "how", "what",
        "over", "about", "after", "before", "more", "than", "been", "into",
    }

    def _terms(text: str) -> set:
        return {
            w for w in _re.findall(r'\b[a-z]{3,}\b', text.lower())
            if w not in _STOP
        }

    title_terms = _terms(title)
    transcript_terms = _terms(transcript)

    def _score(s: dict) -> float:
        # Explicit description citations rank above inferred news
        tier = 3.0 if s.get("type") == "url" else 0.0
        src_terms = _terms((s.get("title") or "") + " " + (s.get("source_name") or ""))
        t_overlap = len(src_terms & title_terms) / max(len(title_terms), 1)
        tr_overlap = len(src_terms & transcript_terms) / max(len(transcript_terms), 1)
        return tier + t_overlap * 2.0 + tr_overlap + s.get("confidence", 0.5)

    ranked = sorted(sources, key=_score, reverse=True)
    return ranked[:max_sources]


def _fetch_article_content(url: str, max_chars: int = 3000) -> Optional[str]:
    """Scrape the article body from a URL using newspaper3k, falling back to
    a raw requests fetch. Returns the first *max_chars* characters, or None."""
    try:
        from newspaper import Article
        art = Article(url)
        art.download()
        art.parse()
        text = art.text.strip()
        if text:
            return text[:max_chars]
    except Exception:
        pass
    # Fallback: raw HTTP fetch, strip HTML tags
    try:
        import urllib.request
        import re as _re
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        # Very coarse HTML strip
        text = _re.sub(r"<[^>]+>", " ", raw)
        text = _re.sub(r"\s{2,}", " ", text).strip()
        return text[:max_chars] if text else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_file(
    path: Path,
    detective,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
    scrape_content: bool = True,
    raw_videos_dir: Path = Path("data/raw_videos"),
) -> bool:
    data = _load_json(path)
    if data is None:
        return False

    inp = data.get("input", {})
    existing_sources = inp.get("sources", [])

    if existing_sources and not overwrite:
        print(f"  {path.name}: skipping — already has {len(existing_sources)} source(s) (use --overwrite to replace)")
        return False

    meta = data.get("metadata", {})
    video_id = meta.get("video_id", path.stem.replace("_example", ""))
    upload_date = meta.get("upload_date", "")
    topic = inp.get("topic", "")
    title = inp.get("title", "")

    # Build description: prefer raw metadata (has full YouTube description + URLs)
    description = title
    raw_path = raw_videos_dir / f"{video_id}.json"
    if raw_path.exists():
        raw = _load_json(raw_path)
        if raw:
            description = f"{title} {raw.get('description', '')}".strip()

    transcript = _build_transcript(data)

    print(f"  {path.name}  [topic={topic!r}]")
    sources = detective.detect_sources(
        transcript=transcript,
        description=description,
        labels=[],
        upload_date=upload_date,
        topic=topic,
    )

    formatted = [fs for s in sources if (fs := _format_source(s))]

    # Deduplicate by URL
    seen_urls: set = set()
    unique: List[dict] = []
    for s in formatted:
        if s["url"] not in seen_urls:
            seen_urls.add(s["url"])
            unique.append(s)
    formatted = unique

    # Keep the top 5 most relevant sources
    before_rank = len(formatted)
    formatted = _rank_sources(formatted, title, transcript)
    if len(formatted) < before_rank:
        print(f"    → ranked {before_rank} → kept top {len(formatted)}")
    # Resolve Google News redirect URLs now that we have the final set
    if any("news.google.com" in s["url"] for s in formatted):
        print(f"    [resolve] resolving Google News URLs...")
        GoogleNewsRSSSearcher.resolve_urls(formatted)
        # Re-deduplicate in case two redirects pointed to the same article
        seen2: set = set()
        deduped2: List[dict] = []
        for s in formatted:
            if s["url"] not in seen2:
                seen2.add(s["url"])
                deduped2.append(s)
        formatted = deduped2

    print(f"    → {len(formatted)} source(s) found")
    for s in formatted[:8]:
        print(f"      [{s['type']:12}] {s['url'][:90]}")

    if not formatted and ("news" in topic.lower() or "politic" in topic.lower()):
        print(f"    [WARNING] News & Politics video has 0 sources — consider checking NewsAPI key or expanding date window")

    if dry_run:
        return len(formatted) > 0

    # Fetch article bodies for news/url sources that don't already have text
    if scrape_content:
        for s in formatted:
            if s["type"] in ("news", "url") and not s.get("text"):
                print(f"    [scrape] {s['url'][:80]}")
                body = _fetch_article_content(s["url"])
                if body:
                    s["text"] = body

    data["input"]["sources"] = formatted
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return len(formatted) > 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add/update sources in training example JSON files"
    )
    parser.add_argument(
        "--dir",
        default="data/training_dataset",
        help="Directory containing *_example.json files (default: data/training_dataset)",
    )
    parser.add_argument(
        "--video",
        help="Process a single video ID (e.g. FlcFFsj_8Rg)",
    )
    parser.add_argument(
        "--news-api-key",
        default=os.getenv("NEWS_API_KEY"),
        metavar="KEY",
        help="newsapi.org API key (defaults to NEWS_API_KEY env var)",
    )
    parser.add_argument(
        "--no-scrape",
        action="store_true",
        help="Skip fetching article body text (faster, metadata-only)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing sources (default: skip files that already have sources)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be found without writing any files",
    )
    args = parser.parse_args()

    if not args.news_api_key:
        print("[INFO] No NEWS_API_KEY set — will use Google News RSS (free, unlimited) only.")
        print("       Optionally add a NewsAPI key for supplementary results:")
        print("       PowerShell:  $env:NEWS_API_KEY='your-key-here'\n")

    from src.data_collection.source_detective import SourceDetective
    detective = SourceDetective(provider="none", newsapi_key=args.news_api_key)

    dataset_dir = Path(args.dir)
    raw_videos_dir = Path("data/raw_videos")

    if args.video:
        files = [dataset_dir / f"{args.video}_example.json"]
        if not files[0].exists():
            print(f"[ERROR] File not found: {files[0]}")
            return
    else:
        files = sorted(dataset_dir.glob("*_example.json"))
        print(f"Found {len(files)} example file(s) in {dataset_dir}\n")

    if args.dry_run:
        print("[DRY RUN] No files will be written.\n")

    updated = 0
    for path in files:
        changed = process_file(
            path,
            detective,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            scrape_content=not args.no_scrape,
            raw_videos_dir=raw_videos_dir,
        )
        if changed:
            updated += 1

    noun = "file(s) would be updated" if args.dry_run else "file(s) updated"
    print(f"\n{updated}/{len(files)} {noun}.")


if __name__ == "__main__":
    main()
