#!/usr/bin/env python3
"""
Enhanced demonstration of the Image Deblurrer with all implemented features.

This script showcases the complete deblurring functionality including:
- Blur detection and analysis
- Multiple deblurring methods
- Automatic method selection
- Performance comparison
"""

import sys
import os
import time
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import cv2
from deblur import ImageDeblurrer
from blur_detector import BlurDetector
from image_processor import ImageProcessor
from utils import setup_logging, Timer, calculate_image_metrics
import matplotlib.pyplot as plt


def create_test_images():
    """Create various types of blurred test images."""
    print("Creating test images...")
    
    processor = ImageProcessor()
    
    # Create a synthetic sharp image with text and patterns
    sharp_image = create_synthetic_image()
    
    # Save the sharp image
    processor.save_image(sharp_image, "test_images/sharp_original.jpg")
    
    # Create different types of blur
    # 1. Gaussian blur
    gaussian_blur = processor.apply_gaussian_blur(sharp_image, kernel_size=15, sigma=5.0)
    processor.save_image(gaussian_blur, "test_images/gaussian_blur.jpg")
    
    # 2. Motion blur
    motion_blur = processor.apply_motion_blur(sharp_image, size=20, angle=45)
    processor.save_image(motion_blur, "test_images/motion_blur.jpg")
    
    # 3. Mild blur
    mild_blur = processor.apply_gaussian_blur(sharp_image, kernel_size=7, sigma=2.0)
    processor.save_image(mild_blur, "test_images/mild_blur.jpg")
    
    print("Test images created in 'test_images/' directory")
    return sharp_image


def create_synthetic_image(size=(512, 512)):
    """Create a synthetic image with various features for testing."""
    image = np.ones((*size, 3), dtype=np.uint8) * 255
    
    # Add text
    cv2.putText(image, "DEBLUR TEST", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    
    # Add lines
    for i in range(0, size[0], 50):
        cv2.line(image, (i, 0), (i, size[1]), (100, 100, 100), 1)
        cv2.line(image, (0, i), (size[0], i), (100, 100, 100), 1)
    
    # Add circles
    cv2.circle(image, (256, 256), 100, (0, 0, 255), 2)
    cv2.circle(image, (256, 256), 50, (0, 255, 0), 2)
    
    # Add rectangles
    cv2.rectangle(image, (350, 350), (450, 450), (255, 0, 0), 2)
    cv2.rectangle(image, (50, 350), (150, 450), (255, 100, 0), -1)
    
    # Add fine details (small text)
    cv2.putText(image, "Fine Detail Test 123", (50, 480), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return image


def analyze_image_blur(image_path):
    """Analyze and display blur characteristics of an image."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {image_path}")
    print('='*60)
    
    processor = ImageProcessor()
    detector = BlurDetector()
    
    # Load image
    image = processor.load_image(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    # Analyze blur
    with Timer("Blur analysis"):
        analysis = detector.analyze_blur(image)
    
    # Display results
    print(f"\nBlur Analysis Results:")
    print(f"  Blur Type:     {analysis['blur_type']}")
    print(f"  Blur Level:    {analysis['blur_level']}")
    print(f"  Blur Score:    {analysis['blur_score']:.2f}")
    print(f"  Focus Measure: {analysis['focus_measure']:.2f}")
    print(f"  Edge Strength: {analysis['edge_strength']:.2f}")
    
    extent = analysis['blur_extent']
    print(f"\nBlur Extent:")
    print(f"  Estimated Kernel Size: {extent['estimated_kernel_size']:.1f}")
    if extent['motion_angle'] is not None:
        print(f"  Motion Angle: {extent['motion_angle']:.1f}°")
    
    freq = analysis['frequency_analysis']
    print(f"\nFrequency Analysis:")
    print(f"  High Freq Ratio: {freq['high_freq_ratio']:.3f}")
    print(f"  Low Freq Ratio:  {freq['low_freq_ratio']:.3f}")
    print(f"  Spectral Slope:  {freq['spectral_slope']:.3f}")
    
    # Get method suggestion
    suggestions = detector.suggest_deblur_method(image)
    print(f"\nRecommended Deblurring:")
    print(f"  Primary Method: {suggestions['primary_method']}")
    print(f"  Confidence:     {suggestions['confidence']:.1%}")
    print(f"  Alternatives:   {', '.join(suggestions['alternative_methods'])}")
    
    return analysis, suggestions


def test_deblurring_methods(image_path, output_dir="results"):
    """Test all deblurring methods on an image."""
    print(f"\n{'='*60}")
    print(f"Testing deblurring methods on: {image_path}")
    print('='*60)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Initialize components
    setup_logging("INFO")
    deblurrer = ImageDeblurrer()
    processor = ImageProcessor()
    
    # Load original image
    original = processor.load_image(image_path)
    if original is None:
        print(f"Failed to load image: {image_path}")
        return
    
    base_name = Path(image_path).stem
    results = {}
    
    # Test each method
    methods = ['cv2', 'wiener', 'deep_learning']
    
    for method in methods:
        print(f"\nTesting {method} method...")
        output_path = f"{output_dir}/{base_name}_{method}.jpg"
        
        with Timer(f"{method} deblurring"):
            success = deblurrer.deblur_image(image_path, output_path, method=method)
        
        if success:
            # Load result and calculate metrics
            result = processor.load_image(output_path)
            if result is not None:
                # Calculate quality metrics if we have a sharp reference
                sharp_path = "test_images/sharp_original.jpg"
                if os.path.exists(sharp_path):
                    sharp = processor.load_image(sharp_path)
                    if sharp is not None and sharp.shape == result.shape:
                        metrics = calculate_image_metrics(sharp, result)
                        results[method] = metrics
                        print(f"  Quality Metrics:")
                        if 'error' not in metrics:
                            print(f"    PSNR: {metrics['psnr']:.2f} dB")
                            print(f"    SSIM: {metrics['ssim']:.4f}")
                            print(f"    MSE:  {metrics['mse']:.2f}")
                        else:
                            print(f"    {metrics['error']}")
                print(f"  ✓ Saved to: {output_path}")
        else:
            print(f"  ✗ Failed")
    
    return results


def compare_results(image_paths, titles=None):
    """Display multiple images side by side for comparison."""
    processor = ImageProcessor()
    
    images = []
    valid_titles = []
    
    for i, path in enumerate(image_paths):
        if os.path.exists(path):
            img = processor.load_image(path)
            if img is not None:
                images.append(img)
                if titles and i < len(titles):
                    valid_titles.append(titles[i])
                else:
                    valid_titles.append(Path(path).stem)
    
    if images:
        processor.display_images(images, valid_titles, figsize=(20, 5))
    else:
        print("No valid images to display")


def main():
    """Main demonstration function."""
    print("\n" + "="*60)
    print("IMAGE DEBLURRER - ENHANCED DEMONSTRATION")
    print("="*60)
    
    # Setup
    setup_logging("INFO")
    
    # Create test images directory
    Path("test_images").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    # Step 1: Create test images
    print("\n1. CREATING TEST IMAGES")
    print("-" * 40)
    sharp_image = create_test_images()
    
    # Step 2: Analyze blur in test images
    print("\n2. BLUR ANALYSIS")
    print("-" * 40)
    
    test_images = [
        "test_images/mild_blur.jpg",
        "test_images/gaussian_blur.jpg",
        "test_images/motion_blur.jpg"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            analyze_image_blur(img_path)
    
    # Step 3: Test deblurring methods
    print("\n3. DEBLURRING TESTS")
    print("-" * 40)
    
    all_results = {}
    for img_path in test_images:
        if os.path.exists(img_path):
            results = test_deblurring_methods(img_path)
            all_results[Path(img_path).stem] = results
    
    # Step 4: Display comparison
    print("\n4. VISUAL COMPARISON")
    print("-" * 40)
    
    # Compare Gaussian blur results
    gaussian_comparison = [
        "test_images/sharp_original.jpg",
        "test_images/gaussian_blur.jpg",
        "results/gaussian_blur_cv2.jpg",
        "results/gaussian_blur_wiener.jpg",
        "results/gaussian_blur_deep_learning.jpg"
    ]
    
    print("\nGaussian Blur Deblurring Comparison:")
    compare_results(
        gaussian_comparison,
        ["Original", "Blurred", "CV2", "Wiener", "Deep Learning"]
    )
    
    # Step 5: Summary
    print("\n5. SUMMARY")
    print("-" * 40)
    
    if all_results:
        print("\nPerformance Summary (PSNR - higher is better):")
        for image_type, methods in all_results.items():
            print(f"\n{image_type}:")
            if methods:
                for method, metrics in methods.items():
                    if 'psnr' in metrics:
                        print(f"  {method:15} PSNR: {metrics['psnr']:.2f} dB")
    
    print("\n" + "="*60)
    print("Demonstration complete!")
    print("Check 'results/' directory for deblurred images")
    print("="*60)


if __name__ == "__main__":
    main()
