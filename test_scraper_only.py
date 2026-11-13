"""
Quick test of YouTube scraper without heavy dependencies.
"""
from src.data_collection.youtube_scraper import YouTubeScraper

def test_scraper():
    print("Testing YouTube Scraper (lightweight)...")
    
    scraper = YouTubeScraper(output_dir="data/test/raw_videos")
    
    # Test with a YouTube Shorts video
    video_url = "https://www.youtube.com/shorts/E2T2_fizSUw"
    
    print(f"Downloading: {video_url}")
    metadata = scraper.scrape_video(video_url)
    
    if metadata:
        print(f"\n==> SUCCESS!")
        print(f"   Title: {metadata['title']}")
        print(f"   Duration: {metadata['duration']}s")
        print(f"   Channel: {metadata.get('channel', 'N/A')}")
        print(f"   Transcript length: {len(metadata.get('transcript', ''))} chars")
        print(f"   Video ID: {metadata['video_id']}")
        return True
    else:
        print("\n==> FAILED")
        return False

if __name__ == "__main__":
    test_scraper()
