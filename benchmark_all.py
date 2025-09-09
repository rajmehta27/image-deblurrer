#!/usr/bin/env python3
"""
Comprehensive Benchmarking Script for All Deblurring Methods

This script benchmarks all available deblurring methods on various test cases.
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import logging
from typing import List, Tuple
import time

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from deblur import ImageDeblurrer
from advanced_deblur import AdvancedDeblurrer
from evaluation import DeblurringBenchmark, ImageQualityEvaluator, create_test_dataset
from image_processor import ImageProcessor
from pretrained_models import SimplifiedDeblurModel
from utils import setup_logging


def create_realistic_test_dataset() -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create a realistic test dataset with various blur types."""
    processor = ImageProcessor()
    dataset = []
    
    # Create base test images
    sizes = [(256, 256), (512, 512)]
    
    for idx, size in enumerate(sizes):
        # Create clean image with various features
        clean = np.ones((*size, 3), dtype=np.uint8) * 255
        
        # Add text
        cv2.putText(clean, f"TEST {idx}", (size[0]//4, size[1]//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        # Add geometric shapes
        cv2.rectangle(clean, (50, 50), (size[0]-50, size[1]-50), (255, 0, 0), 2)
        cv2.circle(clean, (size[0]//2, size[1]//2), min(size)//4, (0, 255, 0), 2)
        
        # Add fine details (grid pattern)
        for i in range(0, size[0], 20):
            cv2.line(clean, (i, 0), (i, size[1]), (200, 200, 200), 1)
            cv2.line(clean, (0, i), (size[0], i), (200, 200, 200), 1)
        
        # Create different blur types
        blur_configs = [
            ('gaussian_mild', processor.apply_gaussian_blur(clean, 5, 1.5)),
            ('gaussian_severe', processor.apply_gaussian_blur(clean, 15, 4.0)),
            ('motion', processor.apply_motion_blur(clean, 15, 45)),
            ('defocus', processor.apply_defocus_blur(clean, 10)),
            ('mixed', processor.apply_gaussian_blur(
                processor.apply_motion_blur(clean, 10, 30), 7, 2.0))
        ]
        
        for blur_type, blurred in blur_configs:
            dataset.append((blurred, clean))
            print(f"Created test image: size={size}, blur={blur_type}")
    
    return dataset


def benchmark_all_methods():
    """Benchmark all available deblurring methods."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE DEBLURRING BENCHMARK")
    print("="*80)
    
    # Initialize components
    deblurrer = ImageDeblurrer()
    advanced = AdvancedDeblurrer()
    simplified = SimplifiedDeblurModel()
    benchmark = DeblurringBenchmark(output_dir="benchmark_results")
    
    # Create test dataset
    print("\nCreating test dataset...")
    test_dataset = create_realistic_test_dataset()
    print(f"Created {len(test_dataset)} test images")
    
    # Define all methods to benchmark
    methods = {
        # Basic CV2 method
        'cv2_basic': (
            lambda img: deblurrer.deblur_cv2(img),
            {}
        ),
        
        # Enhanced CV2 with adaptive parameters
        'cv2_enhanced': (
            lambda img: deblurrer.deblur_cv2_enhanced(img),
            {}
        ),
        
        # Advanced CV2 with multi-scale processing
        'cv2_advanced': (
            lambda img: advanced.deblur_cv2_advanced(img),
            {}
        ),
        
        # Basic Wiener filter
        'wiener_basic': (
            lambda img: deblurrer.deblur_wiener(img, psf_size=15, noise_level=0.01),
            {}
        ),
        
        # Advanced Wiener with PSF estimation
        'wiener_advanced': (
            lambda img: advanced.deblur_wiener_advanced(img, noise_level=0.01),
            {}
        ),
        
        # Simplified deep learning model
        'deep_simplified': (
            lambda img: simplified.deblur(img),
            {}
        ),
        
        # Auto method selection
        'auto_select': (
            lambda img: advanced.deblur_auto(img),
            {}
        )
    }
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    print("-" * 40)
    
    results_df = benchmark.compare_methods(methods, test_dataset)
    
    # Generate report
    report = benchmark.generate_report(results_df)
    print("\n" + report)
    
    # Additional analysis
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    
    # Best method for each blur type
    print("\nBest Methods by Blur Type:")
    print("-" * 40)
    
    # Group results by image characteristics
    for i, (blurred, clean) in enumerate(test_dataset):
        image_results = results_df[results_df['dataset'] == f'image_{i}']
        if not image_results.empty:
            best_psnr = image_results.loc[image_results['psnr'].idxmax()]
            best_ssim = image_results.loc[image_results['ssim'].idxmax()]
            fastest = image_results.loc[image_results['processing_time'].idxmin()]
            
            print(f"\nImage {i}:")
            print(f"  Best PSNR: {best_psnr['method']} ({best_psnr['psnr']:.2f} dB)")
            print(f"  Best SSIM: {best_ssim['method']} ({best_ssim['ssim']:.4f})")
            print(f"  Fastest: {fastest['method']} ({fastest['processing_time']:.3f}s)")
    
    # Memory usage analysis
    print("\n" + "="*40)
    print("RESOURCE USAGE ANALYSIS")
    print("-" * 40)
    
    # Test memory usage for large images
    large_image = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
    
    for method_name, (method_func, _) in list(methods.items())[:3]:  # Test first 3 methods
        print(f"\n{method_name}:")
        
        # Measure memory before
        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process image
        start_time = time.time()
        try:
            _ = method_func(large_image)
            processing_time = time.time() - start_time
            
            # Measure memory after
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_used = mem_after - mem_before
            
            print(f"  Processing time (2048x2048): {processing_time:.2f}s")
            print(f"  Memory used: {mem_used:.2f} MB")
            
        except Exception as e:
            print(f"  Failed: {e}")
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: benchmark_results/")
    
    return results_df


def benchmark_specific_scenario(scenario: str = "motion_blur"):
    """Benchmark methods for specific scenarios."""
    
    print(f"\n" + "="*80)
    print(f"SCENARIO-SPECIFIC BENCHMARK: {scenario.upper()}")
    print("="*80)
    
    processor = ImageProcessor()
    benchmark = DeblurringBenchmark(output_dir=f"benchmark_{scenario}")
    advanced = AdvancedDeblurrer()
    
    # Create scenario-specific test data
    test_images = []
    
    if scenario == "motion_blur":
        # Various motion blur angles and lengths
        base_image = np.ones((512, 512, 3), dtype=np.uint8) * 255
        cv2.putText(base_image, "MOTION TEST", (100, 256), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        for angle in [0, 45, 90, 135]:
            for length in [10, 20, 30]:
                blurred = processor.apply_motion_blur(base_image, length, angle)
                test_images.append((blurred, base_image))
        
    elif scenario == "low_light":
        # Low light conditions with noise
        base_image = np.ones((512, 512, 3), dtype=np.uint8) * 50  # Dark image
        cv2.putText(base_image, "LOW LIGHT", (100, 256), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 100), 3)
        
        # Add noise and blur
        for noise_level in [0.01, 0.05, 0.1]:
            noisy = processor.add_noise(base_image, 'gaussian', noise_level)
            blurred = processor.apply_gaussian_blur(noisy, 7, 2.0)
            test_images.append((blurred, base_image))
    
    elif scenario == "high_resolution":
        # High resolution images
        for size in [(1024, 1024), (2048, 2048)]:
            base_image = np.ones((*size, 3), dtype=np.uint8) * 255
            cv2.putText(base_image, "HIGH RES", (size[0]//4, size[1]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 5)
            
            blurred = processor.apply_gaussian_blur(base_image, 21, 5.0)
            test_images.append((blurred, base_image))
    
    print(f"Created {len(test_images)} test images for {scenario}")
    
    # Define scenario-optimized methods
    if scenario == "motion_blur":
        methods = {
            'wiener_motion': (
                lambda img: advanced.deblur_wiener_advanced(img),
                {}
            ),
            'cv2_motion': (
                lambda img: advanced.deblur_cv2_advanced(img),
                {}
            )
        }
    else:
        methods = {
            'auto': (
                lambda img: advanced.deblur_auto(img),
                {}
            ),
            'cv2_advanced': (
                lambda img: advanced.deblur_cv2_advanced(img),
                {}
            )
        }
    
    # Run benchmark
    results_df = benchmark.compare_methods(methods, test_images)
    report = benchmark.generate_report(results_df)
    
    print("\n" + report)
    
    return results_df


def interactive_benchmark():
    """Interactive benchmarking with user-provided images."""
    
    print("\n" + "="*80)
    print("INTERACTIVE BENCHMARKING")
    print("="*80)
    
    # Check for test images directory
    test_dir = Path("test_images")
    if not test_dir.exists():
        print("\nNo 'test_images' directory found.")
        print("Please create a 'test_images' directory with your test images.")
        return
    
    # Load test images
    test_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    
    if not test_files:
        print("\nNo images found in 'test_images' directory.")
        return
    
    print(f"\nFound {len(test_files)} test images")
    
    # Initialize components
    advanced = AdvancedDeblurrer()
    evaluator = ImageQualityEvaluator()
    
    # Process each image
    for img_path in test_files:
        print(f"\n{'='*40}")
        print(f"Processing: {img_path.name}")
        print('-'*40)
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to load {img_path}")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Analyze blur
        blur_info = advanced.analyze_blur(image)
        print(f"Blur Analysis:")
        print(f"  Type: {blur_info.get('type', 'unknown')}")
        print(f"  Severity: {blur_info.get('severity', 0):.2f}")
        
        # Apply auto deblurring
        print(f"\nApplying auto deblurring...")
        start_time = time.time()
        deblurred = advanced.deblur_auto(image)
        processing_time = time.time() - start_time
        
        print(f"  Processing time: {processing_time:.2f}s")
        
        # Save result
        output_path = f"benchmark_results/{img_path.stem}_deblurred.jpg"
        cv2.imwrite(output_path, cv2.cvtColor(deblurred, cv2.COLOR_RGB2BGR))
        print(f"  Result saved to: {output_path}")
        
        # Quality assessment (no reference)
        quality = advanced.assess_deblur_quality(image, deblurred)
        print(f"\nQuality Assessment:")
        print(f"  Quality score: {quality['quality_score']:.2f}")
        print(f"  Sharpness improvement: {quality['sharpness_improvement']:.2f}x")
        print(f"  Noise level: {quality['noise_level']:.2f}")


def main():
    """Main benchmarking function."""
    
    # Setup logging
    setup_logging("INFO")
    
    print("\n" + "="*80)
    print("IMAGE DEBLURRING SYSTEM - COMPREHENSIVE BENCHMARK")
    print("="*80)
    
    print("\nSelect benchmark mode:")
    print("1. Full benchmark (all methods)")
    print("2. Motion blur scenario")
    print("3. Low light scenario")
    print("4. High resolution scenario")
    print("5. Interactive (your images)")
    print("6. Run all benchmarks")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "1":
        benchmark_all_methods()
    elif choice == "2":
        benchmark_specific_scenario("motion_blur")
    elif choice == "3":
        benchmark_specific_scenario("low_light")
    elif choice == "4":
        benchmark_specific_scenario("high_resolution")
    elif choice == "5":
        interactive_benchmark()
    elif choice == "6":
        # Run all benchmarks
        benchmark_all_methods()
        benchmark_specific_scenario("motion_blur")
        benchmark_specific_scenario("low_light")
        benchmark_specific_scenario("high_resolution")
        interactive_benchmark()
    else:
        print("Invalid choice. Running full benchmark...")
        benchmark_all_methods()
    
    print("\n" + "="*80)
    print("BENCHMARKING COMPLETE!")
    print("="*80)
    print("\nCheck the 'benchmark_results' directory for detailed results.")


if __name__ == "__main__":
    main()
