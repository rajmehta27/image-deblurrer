#!/usr/bin/env python3
"""
Basic usage example for the Image Deblurrer.

This script demonstrates how to use the ImageDeblurrer class 
to process images programmatically.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from deblur import ImageDeblurrer
from utils import setup_logging, create_output_path


def main():
    """
    Demonstrate basic usage of the image deblurrer.
    """
    # Setup logging
    setup_logging("INFO")
    print("Image Deblurrer - Basic Usage Example")
    print("=" * 40)
    
    # Initialize the deblurrer
    deblurrer = ImageDeblurrer()
    
    # Example image path (you would replace this with actual image paths)
    input_image = "sample_blurred_image.jpg"
    
    if not os.path.exists(input_image):
        print(f"Sample image not found: {input_image}")
        print("To test this example, place a blurred image in the project root")
        print("and update the 'input_image' variable in this script.")
        return
    
    # Create output path
    output_image = create_output_path(input_image)
    
    print(f"Input: {input_image}")
    print(f"Output: {output_image}")
    
    # Method 1: OpenCV-based deblurring (default)
    print("\n1. Testing OpenCV method...")
    success = deblurrer.deblur_image(input_image, output_image, method='cv2')
    
    if success:
        print("✓ OpenCV deblurring completed successfully!")
    else:
        print("✗ OpenCV deblurring failed.")
    
    # Method 2: Deep learning method (if model is available)
    print("\n2. Testing Deep Learning method...")
    dl_output = create_output_path(input_image, suffix="_dl_deblurred")
    success = deblurrer.deblur_image(input_image, dl_output, method='deep_learning')
    
    if success:
        print("✓ Deep learning deblurring completed successfully!")
    else:
        print("✗ Deep learning deblurring failed (likely no model available).")
    
    print("\nExample completed!")
    print("Check the output files to see the results.")


def create_sample_blurred_image():
    """
    Create a sample blurred image for testing (requires a source image).
    """
    import cv2
    import numpy as np
    from image_processor import ImageProcessor
    
    processor = ImageProcessor()
    
    # This would work if you have a source image
    source_image_path = "source_image.jpg"
    
    if os.path.exists(source_image_path):
        # Load source image
        image = processor.load_image(source_image_path)
        
        if image is not None:
            # Apply blur to create test image
            blurred = processor.apply_gaussian_blur(image, kernel_size=15, sigma=3.0)
            
            # Save blurred image
            processor.save_image(blurred, "sample_blurred_image.jpg")
            print("Created sample blurred image: sample_blurred_image.jpg")
            return True
    
    return False


if __name__ == "__main__":
    # Uncomment the line below to create a sample blurred image first
    # create_sample_blurred_image()
    
    main()
