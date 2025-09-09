#!/usr/bin/env python3
"""
Quick test script to verify deblurring functionality.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import cv2
from deblur import ImageDeblurrer
from blur_detector import BlurDetector
from image_processor import ImageProcessor
from utils import setup_logging, Timer

def quick_test():
    """Quick test of deblurring functionality."""
    print("Quick Deblurring Test")
    print("=" * 40)
    
    # Setup
    setup_logging("INFO")
    processor = ImageProcessor()
    detector = BlurDetector()
    deblurrer = ImageDeblurrer()
    
    # Create a simple test image
    print("\n1. Creating test image...")
    test_image = np.ones((256, 256, 3), dtype=np.uint8) * 255
    
    # Add some text and shapes
    cv2.putText(test_image, "TEST", (50, 128), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.circle(test_image, (128, 128), 50, (255, 0, 0), 2)
    cv2.rectangle(test_image, (180, 180), (230, 230), (0, 255, 0), 2)
    
    # Save original
    processor.save_image(test_image, "test_original.jpg")
    print("  ✓ Original saved: test_original.jpg")
    
    # Apply blur
    print("\n2. Applying blur...")
    blurred = processor.apply_gaussian_blur(test_image, kernel_size=11, sigma=3.0)
    processor.save_image(blurred, "test_blurred.jpg")
    print("  ✓ Blurred saved: test_blurred.jpg")
    
    # Quick blur analysis
    print("\n3. Analyzing blur...")
    with Timer("Blur detection"):
        blur_score = detector.calculate_blur_score(cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY))
    print(f"  Blur score: {blur_score:.2f}")
    
    # Test deblurring methods
    print("\n4. Testing deblurring methods...")
    
    methods = ['cv2', 'wiener']
    for method in methods:
        print(f"\n  Testing {method}...")
        output_path = f"test_{method}_result.jpg"
        
        with Timer(f"  {method} deblurring"):
            success = deblurrer.deblur_image("test_blurred.jpg", output_path, method=method)
        
        if success:
            print(f"    ✓ Result saved: {output_path}")
        else:
            print(f"    ✗ Failed")
    
    print("\n" + "=" * 40)
    print("Quick test complete!")
    print("Check the output files to see results.")

if __name__ == "__main__":
    quick_test()
