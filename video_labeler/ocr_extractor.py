"""
OCR Extractor - Extract text from video frames using OCR
"""

import numpy as np
from PIL import Image
from typing import List, Optional, Tuple
import cv2


class OCRExtractor:
    """Extract text from images using OCR."""
    
    def __init__(self, use_gpu: bool = True):
        """Initialize OCR reader."""
        self.reader = None
        self.use_gpu = use_gpu
        self._initialized = False
    
    def _ensure_reader(self):
        """Lazy initialize OCR reader."""
        if self._initialized:
            return
        
        try:
            import easyocr
            print(f"   Loading EasyOCR (GPU: {self.use_gpu})...")
            self.reader = easyocr.Reader(
                ['en'], 
                gpu=self.use_gpu,
                verbose=False
            )
            self._initialized = True
            print("   ✅ OCR initialized")
        except ImportError:
            print("   ⚠️  EasyOCR not installed. Install with: pip install easyocr")
            self._initialized = False
        except Exception as e:
            print(f"   ⚠️  OCR initialization failed: {str(e)[:50]}")
            self._initialized = False
    
    def extract_text(self, image: Image.Image) -> str:
        """
        Extract text from an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Extracted text as a single string
        """
        if not self._initialized:
            self._ensure_reader()
        
        if not self.reader:
            return ""
        
        try:
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            # Convert RGB to BGR for cv2
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Run OCR
            results = self.reader.readtext(img_array, detail=0)
            
            # Combine all detected text
            text = " ".join(results)
            return text.strip()
            
        except Exception as e:
            print(f"      OCR error: {str(e)[:50]}")
            return ""
    
    def extract_text_with_positions(
        self, 
        image: Image.Image
    ) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """
        Extract text with bounding box positions.
        
        Args:
            image: PIL Image
            
        Returns:
            List of (text, (x1, y1, x2, y2)) tuples
        """
        if not self._initialized:
            self._ensure_reader()
        
        if not self.reader:
            return []
        
        try:
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            # Convert RGB to BGR for cv2
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Run OCR with details
            results = self.reader.readtext(img_array)
            
            # Extract text and bounding boxes
            text_boxes = []
            for bbox, text, confidence in results:
                if confidence > 0.3:  # Filter low confidence
                    # Convert bbox format
                    x1 = int(min(p[0] for p in bbox))
                    y1 = int(min(p[1] for p in bbox))
                    x2 = int(max(p[0] for p in bbox))
                    y2 = int(max(p[1] for p in bbox))
                    text_boxes.append((text, (x1, y1, x2, y2)))
            
            return text_boxes
            
        except Exception as e:
            print(f"      OCR error: {str(e)[:50]}")
            return []
