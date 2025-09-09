#!/usr/bin/env python3
"""
Evaluation and Benchmarking Module for Image Deblurring System

This module provides comprehensive evaluation metrics, benchmarking tools,
and quality assessment for deblurring methods.
"""

import os
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime

# Metrics imports
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from scipy import signal
from scipy.stats import entropy

# Image quality metrics
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    from image_similarity_measures.quality_metrics import rmse, sam, sre, uiq
    ISM_AVAILABLE = True
except ImportError:
    ISM_AVAILABLE = False


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    psnr: float
    ssim: float
    mse: float
    mae: float
    lpips: Optional[float] = None
    niqe: Optional[float] = None
    brisque: Optional[float] = None
    sharpness: Optional[float] = None
    entropy: Optional[float] = None
    laplacian_variance: Optional[float] = None
    edge_strength: Optional[float] = None
    processing_time: Optional[float] = None
    memory_usage: Optional[float] = None


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    method: str
    dataset: str
    metrics: EvaluationMetrics
    parameters: Dict[str, Any]
    timestamp: str


class ImageQualityEvaluator:
    """Comprehensive image quality evaluation."""
    
    def __init__(self, use_deep_metrics: bool = False):
        """
        Initialize evaluator.
        
        Args:
            use_deep_metrics: Whether to use deep learning-based metrics (LPIPS)
        """
        self.logger = logging.getLogger(__name__)
        self.use_deep_metrics = use_deep_metrics and LPIPS_AVAILABLE
        
        if self.use_deep_metrics:
            try:
                self.lpips_model = lpips.LPIPS(net='alex')
            except Exception as e:
                self.logger.warning(f"Failed to load LPIPS model: {e}")
                self.use_deep_metrics = False
    
    def evaluate(self, reference: np.ndarray, deblurred: np.ndarray,
                 compute_all: bool = True) -> EvaluationMetrics:
        """
        Evaluate deblurring quality.
        
        Args:
            reference: Ground truth image
            deblurred: Deblurred image
            compute_all: Whether to compute all available metrics
            
        Returns:
            EvaluationMetrics object
        """
        # Ensure same shape
        if reference.shape != deblurred.shape:
            self.logger.warning("Image shapes don't match, resizing deblurred image")
            deblurred = cv2.resize(deblurred, (reference.shape[1], reference.shape[0]))
        
        # Basic metrics
        metrics = {
            'psnr': self._compute_psnr(reference, deblurred),
            'ssim': self._compute_ssim(reference, deblurred),
            'mse': self._compute_mse(reference, deblurred),
            'mae': self._compute_mae(reference, deblurred)
        }
        
        if compute_all:
            # Advanced metrics
            metrics['sharpness'] = self._compute_sharpness(deblurred)
            metrics['entropy'] = self._compute_entropy(deblurred)
            metrics['laplacian_variance'] = self._compute_laplacian_variance(deblurred)
            metrics['edge_strength'] = self._compute_edge_strength(deblurred)
            
            # Deep metrics
            if self.use_deep_metrics:
                metrics['lpips'] = self._compute_lpips(reference, deblurred)
            
            # No-reference metrics
            metrics['niqe'] = self._compute_niqe(deblurred)
            metrics['brisque'] = self._compute_brisque(deblurred)
        
        return EvaluationMetrics(**metrics)
    
    def _compute_psnr(self, ref: np.ndarray, img: np.ndarray) -> float:
        """Compute Peak Signal-to-Noise Ratio."""
        return psnr(ref, img, data_range=255)
    
    def _compute_ssim(self, ref: np.ndarray, img: np.ndarray) -> float:
        """Compute Structural Similarity Index."""
        # Convert to grayscale if needed
        if len(ref.shape) == 3:
            ref_gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            ref_gray = ref
            img_gray = img
        
        return ssim(ref_gray, img_gray)
    
    def _compute_mse(self, ref: np.ndarray, img: np.ndarray) -> float:
        """Compute Mean Squared Error."""
        return mse(ref, img)
    
    def _compute_mae(self, ref: np.ndarray, img: np.ndarray) -> float:
        """Compute Mean Absolute Error."""
        return np.mean(np.abs(ref.astype(np.float32) - img.astype(np.float32)))
    
    def _compute_sharpness(self, img: np.ndarray) -> float:
        """Compute sharpness using Laplacian variance."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()
    
    def _compute_entropy(self, img: np.ndarray) -> float:
        """Compute image entropy."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        return entropy(hist)
    
    def _compute_laplacian_variance(self, img: np.ndarray) -> float:
        """Compute variance of Laplacian (focus measure)."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()
    
    def _compute_edge_strength(self, img: np.ndarray) -> float:
        """Compute edge strength using Sobel operator."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        return np.mean(edge_magnitude)
    
    def _compute_lpips(self, ref: np.ndarray, img: np.ndarray) -> float:
        """Compute LPIPS (Learned Perceptual Image Patch Similarity)."""
        if not self.use_deep_metrics:
            return None
        
        try:
            import torch
            
            # Prepare images
            ref_tensor = torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            # Normalize to [-1, 1]
            ref_tensor = 2 * ref_tensor - 1
            img_tensor = 2 * img_tensor - 1
            
            # Compute LPIPS
            with torch.no_grad():
                distance = self.lpips_model(ref_tensor, img_tensor)
            
            return distance.item()
        except Exception as e:
            self.logger.error(f"LPIPS computation failed: {e}")
            return None
    
    def _compute_niqe(self, img: np.ndarray) -> float:
        """Compute NIQE (Natural Image Quality Evaluator)."""
        # Simplified NIQE implementation
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        
        # Extract features
        features = []
        
        # Multi-scale feature extraction
        for scale in [0.5, 1.0, 2.0]:
            scaled = cv2.resize(gray, None, fx=scale, fy=scale)
            
            # Local mean and variance
            kernel_size = 7
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
            local_mean = cv2.filter2D(scaled, -1, kernel)
            local_var = cv2.filter2D(scaled ** 2, -1, kernel) - local_mean ** 2
            
            features.append(np.mean(local_var))
            features.append(np.std(local_var))
        
        # Combine features (simplified)
        niqe_score = np.mean(features)
        
        return niqe_score
    
    def _compute_brisque(self, img: np.ndarray) -> float:
        """Compute BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)."""
        # Simplified BRISQUE implementation
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        
        # Normalize
        gray = gray.astype(np.float64)
        gray = (gray - np.mean(gray)) / (np.std(gray) + 1e-7)
        
        # Extract MSCN coefficients
        features = []
        
        # Compute products
        shifts = [(0, 1), (1, 0), (1, 1), (-1, 1)]
        for shift in shifts:
            shifted = np.roll(gray, shift, axis=(0, 1))
            product = gray * shifted
            
            # Fit GGD parameters (simplified)
            alpha = 2.0  # Shape parameter
            sigma = np.sqrt(np.mean(product ** 2))
            
            features.extend([alpha, sigma])
        
        # Combine features (simplified)
        brisque_score = np.mean(features)
        
        return brisque_score


class DeblurringBenchmark:
    """Benchmarking system for deblurring methods."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Initialize benchmark system.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.evaluator = ImageQualityEvaluator(use_deep_metrics=True)
        self.results = []
    
    def benchmark_method(self, method_func, method_name: str,
                         test_images: List[Tuple[np.ndarray, np.ndarray]],
                         parameters: Dict[str, Any] = None) -> List[BenchmarkResult]:
        """
        Benchmark a deblurring method.
        
        Args:
            method_func: Deblurring function
            method_name: Name of the method
            test_images: List of (blurred, ground_truth) image pairs
            parameters: Method parameters
            
        Returns:
            List of benchmark results
        """
        results = []
        
        for i, (blurred, ground_truth) in enumerate(test_images):
            self.logger.info(f"Benchmarking {method_name} on image {i+1}/{len(test_images)}")
            
            # Measure processing time
            start_time = time.time()
            
            try:
                # Apply deblurring
                if parameters:
                    deblurred = method_func(blurred, **parameters)
                else:
                    deblurred = method_func(blurred)
                
                processing_time = time.time() - start_time
                
                # Evaluate quality
                metrics = self.evaluator.evaluate(ground_truth, deblurred)
                metrics.processing_time = processing_time
                
                # Create result
                result = BenchmarkResult(
                    method=method_name,
                    dataset=f"image_{i}",
                    metrics=metrics,
                    parameters=parameters or {},
                    timestamp=datetime.now().isoformat()
                )
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Benchmarking failed for {method_name}: {e}")
        
        self.results.extend(results)
        return results
    
    def compare_methods(self, methods: Dict[str, Any],
                       test_images: List[Tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
        """
        Compare multiple deblurring methods.
        
        Args:
            methods: Dictionary of method_name: (method_func, parameters)
            test_images: Test dataset
            
        Returns:
            Comparison DataFrame
        """
        all_results = []
        
        for method_name, (method_func, params) in methods.items():
            results = self.benchmark_method(
                method_func, method_name, test_images, params
            )
            all_results.extend(results)
        
        # Convert to DataFrame
        df = self._results_to_dataframe(all_results)
        
        # Save results
        self._save_results(df)
        
        return df
    
    def _results_to_dataframe(self, results: List[BenchmarkResult]) -> pd.DataFrame:
        """Convert benchmark results to DataFrame."""
        data = []
        
        for result in results:
            row = {
                'method': result.method,
                'dataset': result.dataset,
                'timestamp': result.timestamp
            }
            
            # Add metrics
            metrics_dict = asdict(result.metrics)
            row.update(metrics_dict)
            
            # Add parameters
            for key, value in result.parameters.items():
                row[f'param_{key}'] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _save_results(self, df: pd.DataFrame):
        """Save benchmark results."""
        # Save as CSV
        csv_path = self.output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as JSON
        json_path = self.output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        df.to_json(json_path, orient='records', indent=2)
        
        self.logger.info(f"Results saved to {csv_path} and {json_path}")
    
    def generate_report(self, df: pd.DataFrame) -> str:
        """
        Generate benchmark report.
        
        Args:
            df: Results DataFrame
            
        Returns:
            Report string
        """
        report = []
        report.append("=" * 80)
        report.append("DEBLURRING BENCHMARK REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics by method
        report.append("SUMMARY BY METHOD:")
        report.append("-" * 40)
        
        metrics = ['psnr', 'ssim', 'mse', 'processing_time']
        summary = df.groupby('method')[metrics].agg(['mean', 'std'])
        
        for method in summary.index:
            report.append(f"\n{method}:")
            for metric in metrics:
                mean = summary.loc[method, (metric, 'mean')]
                std = summary.loc[method, (metric, 'std')]
                report.append(f"  {metric}: {mean:.4f} ± {std:.4f}")
        
        # Best performing method
        report.append("\n" + "=" * 40)
        report.append("BEST PERFORMING METHODS:")
        report.append("-" * 40)
        
        for metric in ['psnr', 'ssim']:
            best_method = df.groupby('method')[metric].mean().idxmax()
            best_score = df.groupby('method')[metric].mean().max()
            report.append(f"  Best {metric.upper()}: {best_method} ({best_score:.4f})")
        
        # Fastest method
        fastest_method = df.groupby('method')['processing_time'].mean().idxmin()
        fastest_time = df.groupby('method')['processing_time'].mean().min()
        report.append(f"  Fastest: {fastest_method} ({fastest_time:.4f}s)")
        
        report_str = "\n".join(report)
        
        # Save report
        report_path = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w') as f:
            f.write(report_str)
        
        return report_str


class RealTimeEvaluator:
    """Real-time quality evaluation during processing."""
    
    def __init__(self, reference_free: bool = True):
        """
        Initialize real-time evaluator.
        
        Args:
            reference_free: Whether to use reference-free metrics only
        """
        self.reference_free = reference_free
        self.history = []
    
    def evaluate_frame(self, frame: np.ndarray, 
                       reference: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate single frame quality.
        
        Args:
            frame: Current frame
            reference: Optional reference frame
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Reference-free metrics
        metrics['sharpness'] = self._compute_sharpness_fast(frame)
        metrics['contrast'] = self._compute_contrast(frame)
        metrics['brightness'] = np.mean(frame)
        
        # Reference-based metrics if available
        if reference is not None and not self.reference_free:
            metrics['psnr'] = psnr(reference, frame, data_range=255)
            gray_ref = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY) if len(reference.shape) == 3 else reference
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
            metrics['ssim'] = ssim(gray_ref, gray_frame)
        
        self.history.append(metrics)
        return metrics
    
    def _compute_sharpness_fast(self, img: np.ndarray) -> float:
        """Fast sharpness computation."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def _compute_contrast(self, img: np.ndarray) -> float:
        """Compute image contrast."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        return gray.std()
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics from evaluation history."""
        if not self.history:
            return {}
        
        df = pd.DataFrame(self.history)
        stats = {}
        
        for col in df.columns:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
        
        return stats


def create_test_dataset(num_images: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create synthetic test dataset for benchmarking.
    
    Args:
        num_images: Number of test images
        
    Returns:
        List of (blurred, ground_truth) pairs
    """
    dataset = []
    
    for i in range(num_images):
        # Create synthetic image
        size = (256, 256, 3)
        image = np.ones(size, dtype=np.uint8) * 255
        
        # Add patterns
        cv2.rectangle(image, (50, 50), (200, 200), (0, 0, 255), 2)
        cv2.circle(image, (128, 128), 50, (0, 255, 0), -1)
        cv2.putText(image, f"Test {i}", (80, 128), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Create blurred version
        kernel_size = 5 + i * 2
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        dataset.append((blurred, image))
    
    return dataset


if __name__ == "__main__":
    # Test evaluation system
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Image Quality Evaluator...")
    evaluator = ImageQualityEvaluator()
    
    # Create test images
    original = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    deblurred = cv2.GaussianBlur(original, (5, 5), 1)
    
    metrics = evaluator.evaluate(original, deblurred)
    print(f"Evaluation metrics: {metrics}")
    
    print("\nTesting Benchmarking System...")
    benchmark = DeblurringBenchmark()
    
    # Create test dataset
    test_data = create_test_dataset(3)
    
    # Define test methods
    def method1(img):
        return cv2.filter2D(img, -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))
    
    def method2(img):
        return cv2.bilateralFilter(img, 9, 75, 75)
    
    methods = {
        'sharpening': (method1, {}),
        'bilateral': (method2, {})
    }
    
    # Run benchmark
    df = benchmark.compare_methods(methods, test_data)
    report = benchmark.generate_report(df)
    
    print("\n" + report)
    print("\nEvaluation and benchmarking system ready!")
