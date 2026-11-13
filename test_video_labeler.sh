#!/bin/bash
# Helper script to test video_labeler with example.mp4

echo "========================================"
echo "Video Labeler Test"
echo "========================================"
echo ""

if [ ! -f example.mp4 ]; then
    echo "Error: example.mp4 not found!"
    echo "Please add a video file named example.mp4 to this directory."
    exit 1
fi

echo "Found: example.mp4"
echo ""
echo "Choose test mode:"
echo "  1. Quick test (10 frames, ~1 minute)"
echo "  2. Full analysis (with transcription, ~5 minutes)"
echo ""
read -p "Enter choice (1 or 2): " choice

if [ "$choice" = "1" ]; then
    echo ""
    echo "Running quick test..."
    python3 quick_test.py
elif [ "$choice" = "2" ]; then
    echo ""
    echo "Running full analysis..."
    python3 test_example_video.py
else
    echo "Invalid choice!"
    exit 1
fi

echo ""
