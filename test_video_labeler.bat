@echo off
REM Helper script to test video_labeler with example.mp4

echo ========================================
echo Video Labeler Test
echo ========================================
echo.

if not exist example.mp4 (
    echo Error: example.mp4 not found!
    echo Please add a video file named example.mp4 to this directory.
    pause
    exit /b 1
)

echo Found: example.mp4
echo.
echo Choose test mode:
echo   1. Quick test (10 frames, ~1 minute)
echo   2. Full analysis (with transcription, ~5 minutes)
echo.
set /p choice="Enter choice (1 or 2): "

if "%choice%"=="1" (
    echo.
    echo Running quick test...
    python quick_test.py
) else if "%choice%"=="2" (
    echo.
    echo Running full analysis...
    python test_example_video.py
) else (
    echo Invalid choice!
    pause
    exit /b 1
)

echo.
pause
