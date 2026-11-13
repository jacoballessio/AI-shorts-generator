@echo off
REM Installation script for Video Labeler with SmolVLM (Windows)

echo ========================================
echo Video Labeler SmolVLM Installation
echo ========================================
echo.

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.10 or higher from python.org
    pause
    exit /b 1
)

echo Detected Python version:
python --version
echo.

REM Ask about GPU
echo Do you have an NVIDIA GPU? (Y/N)
set /p has_gpu="Enter choice: "

if /i "%has_gpu%"=="Y" goto install_cuda
if /i "%has_gpu%"=="y" goto install_cuda
goto install_cpu

:install_cuda
echo.
echo Installing PyTorch with CUDA support...
echo This will download ~2GB
echo.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
goto continue_install

:install_cpu
echo.
echo Installing PyTorch (CPU only)...
echo Note: This will be slower than GPU!
echo.
pip install torch torchvision
goto continue_install

:continue_install

echo.
echo Installing Transformers and dependencies...
pip install transformers accelerate pillow

echo.
echo Installing MoviePy for video processing...
pip install moviepy

echo.
echo Installing Whisper for transcription...
pip install openai-whisper

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Testing installation...
echo.

python test_smolvlm.py

echo.
echo If all tests passed, you're ready to go!
echo See examples.py for usage examples.
echo See SMOLVLM_SETUP.md for optimization tips.
echo.
pause
