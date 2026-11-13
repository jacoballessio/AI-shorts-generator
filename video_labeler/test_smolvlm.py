"""
Quick test to verify SmolVLM is working correctly.

Run this after installing dependencies to ensure everything is set up.
"""

import sys
import os

def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  CUDA not available - will use CPU (slower)")
            
            # Check if CPU-only PyTorch is installed
            if "+cpu" in torch.__version__:
                print()
                print("❌ You have CPU-only PyTorch installed!")
                print("   For GPU support, reinstall PyTorch with CUDA:")
                print()
                print("   pip uninstall torch torchvision")
                print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
                print()
                return False
    except ImportError:
        print("❌ PyTorch not installed!")
        print("   Install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed!")
        print("   Install: pip install transformers")
        return False
    
    try:
        import PIL
        print(f"✅ Pillow {PIL.__version__}")
    except ImportError:
        print("❌ Pillow not installed!")
        print("   Install: pip install pillow")
        return False
    
    try:
        import moviepy
        print(f"✅ MoviePy {moviepy.__version__}")
    except ImportError:
        print("❌ MoviePy not installed!")
        print("   Install: pip install moviepy")
        return False
    
    print()
    return True


def test_smolvlm_loading():
    """Test loading SmolVLM model."""
    print("Testing SmolVLM loading...")
    print("(This will download ~2GB on first run)")
    print()
    
    try:
        from transformers import AutoProcessor, Idefics3ForConditionalGeneration
        import torch
        
        model_name = "HuggingFaceTB/SmolVLM-Instruct"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("✅ Processor loaded")
        
        print(f"Loading model to {device}...")
        model = Idefics3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            _attn_implementation="eager"
        )
        print("✅ Model loaded successfully!")
        
        # Test inference with dummy image
        print("\nTesting inference...")
        from PIL import Image
        
        # Create a simple test image
        test_image = Image.new('RGB', (224, 224), color='blue')
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What color is this image?"}
            ]
        }]
        
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[test_image], return_tensors="pt")
        
        # Move to device
        input_ids = inputs["input_ids"]
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        print("Generating response...")
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(f"✅ Inference successful!")
        print(f"   Response: {output_text}")
        
        # Clean up
        del model
        del processor
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading SmolVLM: {e}")
        return False


def test_video_labeler():
    """Test video labeler package."""
    print("\nTesting video_labeler package...")
    
    try:
        # Add parent directory to path if we're in the video_labeler folder
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from video_labeler import VideoLabeler, LabelerConfig
        print("✅ video_labeler imports successful")
        
        # Create config
        config = LabelerConfig(
            vlm_provider="local",
            vlm_model="HuggingFaceTB/SmolVLM-Instruct",
            use_whisper=False,  # Skip Whisper for quick test
            generate_summary=False,
            max_frames=1  # Only analyze 1 frame for quick test
        )
        print("✅ Configuration created")
        
        # Note: We're not creating VideoLabeler instance here because
        # it would load the model, and we already tested that above
        print("✅ video_labeler package ready to use")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with video_labeler: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("SmolVLM Installation Test")
    print("=" * 70)
    print()
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed - please install missing packages")
        return
    
    # Test SmolVLM
    print()
    if not test_smolvlm_loading():
        print("\n❌ SmolVLM test failed")
        return
    
    # Test video labeler
    if not test_video_labeler():
        print("\n❌ video_labeler test failed")
        return
    
    # All tests passed!
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print()
    print("SmolVLM is ready to use! 🎉")
    print()
    print("Next steps:")
    print("1. Try the examples: python examples.py")
    print("2. Analyze your first video:")
    print()
    print("   from video_labeler import VideoLabeler")
    print("   labeler = VideoLabeler()")
    print("   analysis = labeler.analyze_video('your_video.mp4')")
    print()
    print("See SMOLVLM_SETUP.md for optimization tips!")


if __name__ == "__main__":
    main()
