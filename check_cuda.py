"""
Quick diagnostic to check PyTorch CUDA installation
"""

import sys

print("=" * 70)
print("PyTorch CUDA Diagnostic")
print("=" * 70)
print()

try:
    import torch
    print(f"✅ PyTorch installed: {torch.__version__}")
    print()
    
    # Check if CPU-only version
    if "+cpu" in torch.__version__:
        print("❌ PROBLEM: You have CPU-only PyTorch installed!")
        print()
        print("Your PyTorch version ends with '+cpu' which means it was")
        print("built without CUDA support and cannot use your GPU.")
        print()
        print("=" * 70)
        print("SOLUTION: Install CUDA-enabled PyTorch")
        print("=" * 70)
        print()
        print("Run these commands:")
        print()
        print("1. Uninstall current PyTorch:")
        print("   pip uninstall torch torchvision torchaudio")
        print()
        print("2. Install CUDA version:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print()
        print("3. Verify installation:")
        print("   python check_cuda.py")
        print()
        sys.exit(1)
    
    # Check CUDA availability
    print("CUDA Check:")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  torch.version.cuda: {torch.version.cuda}")
        print(f"  torch.cuda.device_count(): {torch.cuda.device_count()}")
        print()
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
            print()
        
        print("=" * 70)
        print("✅ PyTorch CUDA is working correctly!")
        print("=" * 70)
        print()
        print("Your GPU will be used for SmolVLM inference.")
        print()
        
    else:
        print()
        print("❌ CUDA is not available!")
        print()
        print("Possible reasons:")
        print("1. No NVIDIA GPU detected")
        print("2. NVIDIA drivers not installed")
        print("3. CUDA toolkit not installed")
        print("4. Wrong PyTorch version installed")
        print()
        
        # Try to get more info
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
            )
            if result.returncode == 0:
                print("nvidia-smi output:")
                print(result.stdout)
                print()
                print("Your GPU is detected by nvidia-smi but not by PyTorch.")
                print("This usually means PyTorch needs to be reinstalled.")
            else:
                print("nvidia-smi not found or failed.")
                print("This suggests NVIDIA drivers are not installed.")
        except Exception:
            print("Could not run nvidia-smi to check GPU.")
        
        print()
        print("=" * 70)
        print("SOLUTION")
        print("=" * 70)
        print()
        print("1. Verify NVIDIA drivers are installed:")
        print("   Run: nvidia-smi")
        print()
        print("2. Reinstall PyTorch with CUDA support:")
        print("   pip uninstall torch torchvision")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print()
        sys.exit(1)

except ImportError:
    print("❌ PyTorch is not installed!")
    print()
    print("Install PyTorch with CUDA support:")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print()
    sys.exit(1)
