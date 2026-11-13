# YouTube Cookie Export Guide

YouTube is blocking automated downloads with HTTP 403 errors. The most reliable way to bypass this is by using your browser's cookies.

## Quick Start

1. **Install a Cookie Export Extension**
   - **Chrome**: [Get cookies.txt](https://chrome.google.com/webstore/detail/get-cookiestxt/bgaddhkoddajcdgocldbbfleckgcbcid)
   - **Firefox**: [cookies.txt](https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/)
   - **Edge**: Same as Chrome (use Chrome Web Store)

2. **Export YouTube Cookies**
   - Go to https://www.youtube.com
   - Make sure you're logged in (if you want access to age-restricted content)
   - Click the extension icon
   - Click "Export" or "Download"
   - Save as `cookies.txt` in this project folder

3. **Run the Pipeline**
   ```bash
   python pipeline_orchestrator.py --video "YOUR_VIDEO_URL"
   ```

The scraper will automatically find and use `cookies.txt` from:
- Current directory: `cookies.txt`
- Current directory: `youtube_cookies.txt`
- Home directory: `~/cookies.txt`
- Data directory: `data/cookies.txt`

## Why This Works

YouTube's 403 errors are part of their SABR (Subscriber-Authenticated Bearer) protection. By using your browser's cookies:
- YouTube sees requests as coming from a logged-in user
- You inherit your account's permissions (age-restricted videos, premium features)
- The requests look identical to normal browser traffic

## Security Notes

⚠️ **IMPORTANT**: `cookies.txt` contains your YouTube session!
- **DO NOT share** this file with anyone
- **DO NOT commit** it to git (already in .gitignore)
- **DO expire** - regenerate every few weeks
- **DO delete** when done if security is a concern

## Fallback Strategies

If you don't want to use cookies, the scraper tries these alternatives (in order):

1. **Cookies** (most reliable) ← Use this!
2. **Android VR Client** - Special mobile client
3. **iOS Client** - Apple device client
4. **Android Music Client** - Music app client
5. **Basic Download** - Standard approach (likely to fail)

## Troubleshooting

### "No cookies file found"
- Make sure `cookies.txt` is in the project root
- Or specify it: `YouTubeScraper(cookies_file="path/to/cookies.txt")`

### Still getting 403 errors
- Your cookies may be expired - export fresh ones
- Make sure you're logged into YouTube when exporting
- Try logging out and back in to YouTube, then re-export

### "Cookie file format error"
- Use the browser extensions listed above
- Make sure it's in Netscape format (standard)
- Don't edit the file manually

## Alternative: pytube / pytubefix

If cookie method doesn't work, you can try alternative libraries:

```bash
pip install pytubefix
```

But yt-dlp with cookies is the gold standard for reliability.
