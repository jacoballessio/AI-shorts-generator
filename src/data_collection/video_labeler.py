"""
Video labeler - extracts visual descriptions frame-by-frame using SmolVLM.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import cv2
from PIL import Image
import torch
from transformers import AutoProcessor
import numpy as np

# Try to import the correct model class for SmolVLM
try:
    from transformers.models.idefics3.modeling_idefics3 import Idefics3ForConditionalGeneration
    VLM_MODEL_CLASS = Idefics3ForConditionalGeneration
except:
    try:
        from transformers import Idefics2ForConditionalGeneration
        VLM_MODEL_CLASS = Idefics2ForConditionalGeneration
    except:
        from transformers import AutoModelForVision2Seq
        VLM_MODEL_CLASS = AutoModelForVision2Seq


class VideoLabeler:
    """Extract visual descriptions and timestamps from videos."""
    
    def __init__(self, model_name: str = "HuggingFaceTB/SmolVLM-Instruct"):
        """Initialize the video labeler with SmolVLM."""
        
        print(f"Loading {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.model = VLM_MODEL_CLASS.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                _attn_implementation="eager",  # Prevents CUDA errors
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Set pad token if not set
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
                
            print(f"Model loaded on {self.device}")
            
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            print(f"   Trying with CPU fallback...")
            self.device = "cpu"
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.model = VLM_MODEL_CLASS.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                _attn_implementation="eager",
            )
            self.model = self.model.to(self.device)
            
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
                
            print(f"Model loaded on {self.device} (fallback)")
    
    def label_video(self, video_path: str, sample_rate: int = 30) -> List[Dict]:
        """
        Extract visual descriptions from video.
        
        Args:
            video_path: Path to video file
            sample_rate: Sample every N frames (default 30 = ~1 fps for 30fps video)
            
        Returns:
            List of {timestamp, description, frame_number, confidence}
        """
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing {video_path.name}")
        print(f"FPS: {fps}, Total frames: {total_frames}, Duration: {total_frames/fps:.1f}s")
        
        labels = []
        frame_number = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at specified rate
            if frame_number % sample_rate == 0:
                timestamp = frame_number / fps
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                
                # Get description
                description = self._describe_frame(image)
                
                labels.append({
                    'timestamp': timestamp,
                    'frame_number': frame_number,
                    'description': description,
                })
                
                print(f"  [{timestamp:.1f}s] {description}")
            
            frame_number += 1
        
        cap.release()
        return labels
    
    def _describe_frame(self, image: Image.Image) -> str:
        """Generate description for a single frame."""
        
        try:
            # Resize image to reasonable size to avoid memory issues
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            prompt = "Describe this image in detail."
            
            # Use simpler format for SmolVLM
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Prepare inputs
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=text, images=[image], return_tensors="pt")
            
            # Move to device one by one to avoid issues
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Generate with safer parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # Decode
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the answer (remove prompt)
            if "Assistant:" in generated_text:
                description = generated_text.split("Assistant:")[-1].strip()
            elif "\n" in generated_text:
                # Sometimes the answer is after a newline
                description = generated_text.split("\n")[-1].strip()
            else:
                description = generated_text.strip()
            
            return description if description else "No description available"
            
        except Exception as e:
            print(f"    ⚠️ Error describing frame: {e}")
            return f"[Error: {str(e)[:50]}]"
    
    def detect_scene_changes(self, labels: List[Dict], threshold: float = 0.3) -> List[Dict]:
        """
        Detect scene changes based on description similarity.
        
        Args:
            labels: List of frame labels
            threshold: Similarity threshold for scene change
            
        Returns:
            Labels with 'scene_change' flag added
        """
        
        for i in range(1, len(labels)):
            prev_desc = labels[i-1]['description']
            curr_desc = labels[i]['description']
            
            # Simple heuristic: if descriptions are very different, mark as scene change
            similarity = self._calculate_similarity(prev_desc, curr_desc)
            labels[i]['scene_change'] = similarity < threshold
        
        return labels
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple word overlap similarity."""
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def save_labels(self, labels: List[Dict], output_path: str):
        """Save labels to JSON file."""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(labels, f, indent=2)
        
        print(f"Labels saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    labeler = VideoLabeler()
    
    # Label a video (sample every 30 frames = ~1 per second for 30fps video)
    video_path = "data/raw_videos/EXAMPLE.mp4"
    # labels = labeler.label_video(video_path, sample_rate=30)
    
    # Detect scene changes
    # labels = labeler.detect_scene_changes(labels)
    
    # Save
    # labeler.save_labels(labels, "data/labels/EXAMPLE.json")
    
    print("Labeler ready. Uncomment examples to test.")
