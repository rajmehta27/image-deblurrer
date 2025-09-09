#!/usr/bin/env python3
"""
Image Deblurrer - Main Application

This module provides the main functionality for deblurring images using
various image processing and machine learning techniques.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Any

import cv2
import numpy as np
from PIL import Image

# Import our custom modules
from image_processor import ImageProcessor
from model_loader import ModelLoader
from utils import setup_logging, validate_image_path
from advanced_deblur import AdvancedDeblurrer, PerformanceOptimizer


class ImageDeblurrer:
    """Main class for handling image deblurring operations."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ImageDeblurrer.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.image_processor = ImageProcessor()
        self.model_loader = ModelLoader(config_path)
        self.advanced_deblurrer = AdvancedDeblurrer()
        self.performance_optimizer = PerformanceOptimizer()
        
    def deblur_image(self, input_path: str, output_path: str, method: str = 'auto') -> bool:
        """
        Deblur an image using the specified method.
        
        Args:
            input_path: Path to the input blurred image
            output_path: Path where deblurred image will be saved
            method: Deblurring method ('cv2', 'wiener', 'deep_learning')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate input
            if not validate_image_path(input_path):
                self.logger.error(f"Invalid input image path: {input_path}")
                return False
                
            # Load image
            image = self.image_processor.load_image(input_path)
            if image is None:
                self.logger.error(f"Failed to load image: {input_path}")
                return False
                
            # Automatic method selection if requested
            if method == 'auto':
                method = self.advanced_deblurrer.select_best_method(image)
                self.logger.info(f"Auto-selected method: {method}")
            
            self.logger.info(f"Processing image: {input_path} using method: {method}")
            
            # Apply deblurring based on method
            if method == 'cv2' or method == 'cv2_advanced':
                deblurred = self._deblur_cv2(image)
            elif method == 'wiener':
                deblurred = self._deblur_wiener(image)
            elif method == 'deep_learning':
                deblurred = self._deblur_deep_learning(image)
            else:
                self.logger.error(f"Unknown deblurring method: {method}")
                return False
            
            # Assess quality of deblurring
            quality_metrics = self.advanced_deblurrer.assess_deblur_quality(image, deblurred)
            self.logger.info(f"Deblurring quality score: {quality_metrics['quality_score']:.2f}")
            self.logger.info(f"Sharpness improvement: {quality_metrics['sharpness_improvement']:.2f}x")
                
            # Save result
            success = self.image_processor.save_image(deblurred, output_path)
            if success:
                self.logger.info(f"Deblurred image saved to: {output_path}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error during deblurring: {str(e)}")
            return False
    
    def _deblur_cv2(self, image: np.ndarray) -> np.ndarray:
        """
        Deblur using advanced CV2 techniques.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Deblurred image
        """
        # Use advanced CV2 deblurring
        return self.advanced_deblurrer.deblur_cv2_advanced(image)
    
    def _deblur_cv2_basic(self, image: np.ndarray) -> np.ndarray:
        """
        Deblur using enhanced OpenCV techniques.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Deblurred image
        """
        try:
            # Store original for blending
            original = image.copy()
            
            # 1. Apply bilateral filter to reduce noise while preserving edges
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
            
            # 2. Unsharp masking for initial enhancement
            gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
            unsharp_mask = cv2.addWeighted(denoised, 2.0, gaussian, -1.0, 0)
            
            # 3. Adaptive sharpening based on local variance
            # Calculate local variance to identify edges
            gray = cv2.cvtColor(unsharp_mask, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else unsharp_mask
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = np.abs(laplacian)
            
            # Normalize variance for adaptive sharpening
            variance_norm = variance / (variance.max() + 1e-8)
            
            # 4. Apply different sharpening kernels based on blur detection
            # Standard sharpening kernel
            kernel_sharp = np.array([[-1, -1, -1],
                                    [-1,  9, -1],
                                    [-1, -1, -1]]) / 1.0
            
            # Strong sharpening kernel for heavily blurred regions
            kernel_strong = np.array([[0, -1, 0],
                                     [-1, 5, -1],
                                     [0, -1, 0]]) / 1.0
            
            # Apply sharpening
            sharpened_standard = cv2.filter2D(unsharp_mask, -1, kernel_sharp)
            sharpened_strong = cv2.filter2D(unsharp_mask, -1, kernel_strong)
            
            # Blend based on local variance (edge strength)
            if len(image.shape) == 3:
                variance_norm = np.stack([variance_norm] * 3, axis=-1)
            
            result = sharpened_standard * (1 - variance_norm) + sharpened_strong * variance_norm
            result = result.astype(np.uint8)
            
            # 5. Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
            if len(result.shape) == 3:
                # Convert to LAB color space
                lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                
                # Merge and convert back
                lab = cv2.merge([l, a, b])
                result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                result = clahe.apply(result)
            
            # 6. Final edge enhancement
            edges = cv2.Canny(gray, 50, 150)
            edges = cv2.dilate(edges, None, iterations=1)
            edges = cv2.erode(edges, None, iterations=1)
            
            # Blend with original for natural look (avoid over-sharpening)
            final = cv2.addWeighted(result, 0.8, original, 0.2, 0)
            
            self.logger.info("Enhanced CV2 deblurring completed")
            return final
            
        except Exception as e:
            self.logger.error(f"Enhanced CV2 deblurring failed: {str(e)}")
            # Fallback to simple sharpening
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            return cv2.filter2D(image, -1, kernel)
    
    def _deblur_wiener(self, image: np.ndarray) -> np.ndarray:
        """
        Deblur using advanced Wiener filtering.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Deblurred image
        """
        # Use advanced Wiener deblurring
        return self.advanced_deblurrer.deblur_wiener_advanced(image)
    
    def _deblur_deep_learning(self, image: np.ndarray) -> np.ndarray:
        """
        Deblur using deep learning model.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Deblurred image
        """
        try:
            # Load and use the deep learning model
            model = self.model_loader.load_model()
            if model is None:
                # If no pre-trained model, use enhanced traditional methods
                self.logger.info("No deep learning model available, using hybrid approach")
                return self._hybrid_deblur(image)
            
            # Check if memory optimization is needed
            if not self.performance_optimizer.optimize_memory_usage(image):
                self.logger.warning("Large image detected, using tiled processing")
            
            # Use advanced deep learning deblurring with tiling
            return self.advanced_deblurrer.deblur_deep_learning_advanced(image, model)
            
            # Store original dimensions
            original_shape = image.shape
            
            # Preprocess image for model
            processed_input = self.image_processor.preprocess_for_model(image)
            
            # Get model configuration
            model_info = self.model_loader.get_model_info()
            model_type = model_info.get('model_type')
            
            # Run inference based on model type
            if model_type == 'pytorch':
                result = self._pytorch_inference(processed_input, model)
            elif model_type == 'tensorflow':
                result = self._tensorflow_inference(processed_input, model)
            else:
                self.logger.error(f"Unknown model type: {model_type}")
                return self._hybrid_deblur(image)
            
            # Postprocess the result
            deblurred = self.image_processor.postprocess_from_model(result)
            
            # Resize back to original dimensions if needed
            if deblurred.shape[:2] != original_shape[:2]:
                deblurred = cv2.resize(deblurred, (original_shape[1], original_shape[0]), 
                                      interpolation=cv2.INTER_LANCZOS4)
            
            # Apply post-processing enhancement
            enhanced = self._post_process_deblurred(deblurred, image)
            
            self.logger.info("Deep learning deblurring completed successfully")
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Deep learning deblurring failed: {str(e)}")
            return self._hybrid_deblur(image)
    
    def _pytorch_inference(self, input_tensor: np.ndarray, model: Any) -> np.ndarray:
        """
        Run PyTorch model inference.
        
        Args:
            input_tensor: Preprocessed input
            model: PyTorch model
            
        Returns:
            Model output
        """
        try:
            import torch
            
            # Convert to PyTorch tensor
            if not isinstance(input_tensor, torch.Tensor):
                tensor = torch.from_numpy(input_tensor).float()
            else:
                tensor = input_tensor
            
            # Add batch dimension if needed
            if len(tensor.shape) == 3:
                tensor = tensor.unsqueeze(0)
            
            # Move channels to correct position (NHWC to NCHW)
            if tensor.shape[-1] in [1, 3]:
                tensor = tensor.permute(0, 3, 1, 2)
            
            # Run inference
            with torch.no_grad():
                output = model(tensor)
            
            # Convert back to numpy and correct format
            result = output.cpu().numpy()
            
            # Move channels back (NCHW to NHWC)
            if len(result.shape) == 4 and result.shape[1] in [1, 3]:
                result = np.transpose(result, (0, 2, 3, 1))
            
            return result
            
        except Exception as e:
            self.logger.error(f"PyTorch inference failed: {str(e)}")
            raise
    
    def _tensorflow_inference(self, input_tensor: np.ndarray, model: Any) -> np.ndarray:
        """
        Run TensorFlow model inference.
        
        Args:
            input_tensor: Preprocessed input
            model: TensorFlow model
            
        Returns:
            Model output
        """
        try:
            # TensorFlow expects NHWC format by default
            if len(input_tensor.shape) == 3:
                input_tensor = np.expand_dims(input_tensor, axis=0)
            
            # Run inference
            output = model.predict(input_tensor, verbose=0)
            
            return output
            
        except Exception as e:
            self.logger.error(f"TensorFlow inference failed: {str(e)}")
            raise
    
    def _hybrid_deblur(self, image: np.ndarray) -> np.ndarray:
        """
        Hybrid deblurring approach combining multiple methods.
        
        Args:
            image: Input image
            
        Returns:
            Deblurred image
        """
        try:
            # Analyze blur to determine best approach
            from blur_detector import BlurDetector
            detector = BlurDetector()
            suggestions = detector.suggest_deblur_method(image)
            
            blur_type = suggestions['analysis']['blur_type']
            blur_level = suggestions['analysis']['blur_level']
            
            self.logger.info(f"Detected {blur_level} {blur_type} blur, using hybrid approach")
            
            # Apply appropriate method based on analysis
            if blur_type == 'motion' and blur_level in ['moderate', 'severe']:
                # Use Richardson-Lucy for motion blur
                kernel_size = int(suggestions['parameters'].get('kernel_size', 15))
                angle = suggestions['parameters'].get('angle', 0)
                psf = self._create_motion_kernel(kernel_size, angle)
                deblurred = self.richardson_lucy_deconvolution(image, psf, iterations=25)
                
            elif blur_type == 'gaussian':
                # Use Wiener filter for Gaussian blur
                deblurred = self._deblur_wiener(image)
                
            else:
                # Use enhanced CV2 for other types
                deblurred = self._deblur_cv2(image)
            
            # Apply additional enhancement if blur was severe
            if blur_level == 'severe':
                deblurred = self._enhance_severely_blurred(deblurred)
            
            return deblurred
            
        except Exception as e:
            self.logger.error(f"Hybrid deblurring failed: {str(e)}")
            return self._deblur_cv2(image)
    
    def _post_process_deblurred(self, deblurred: np.ndarray, original: np.ndarray) -> np.ndarray:
        """
        Post-process deblurred image for enhancement.
        
        Args:
            deblurred: Deblurred image
            original: Original blurred image
            
        Returns:
            Enhanced deblurred image
        """
        try:
            # Reduce potential artifacts
            smoothed = cv2.bilateralFilter(deblurred, 5, 50, 50)
            
            # Enhance edges
            edges = cv2.Canny(cv2.cvtColor(smoothed, cv2.COLOR_RGB2GRAY) 
                            if len(smoothed.shape) == 3 else smoothed, 30, 100)
            edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB) if len(smoothed.shape) == 3 else edges
            
            # Blend edges back
            enhanced = cv2.addWeighted(smoothed, 0.95, edges_3channel, 0.05, 0)
            
            # Adjust contrast and brightness
            enhanced = self.image_processor.enhance_contrast(enhanced, alpha=1.1, beta=5)
            
            # Blend with original to preserve natural look
            final = cv2.addWeighted(enhanced, 0.85, original, 0.15, 0)
            
            return final
            
        except Exception as e:
            self.logger.warning(f"Post-processing failed: {str(e)}")
            return deblurred
    
    def _enhance_severely_blurred(self, image: np.ndarray) -> np.ndarray:
        """
        Special enhancement for severely blurred images.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        try:
            # Apply stronger sharpening
            kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # Enhance local contrast
            if len(image.shape) == 3:
                lab = cv2.cvtColor(sharpened, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                enhanced = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            else:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(sharpened)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"Severe blur enhancement failed: {str(e)}")
            return image
    
    def _estimate_blur_kernel(self, image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """
        Estimate the blur kernel (Point Spread Function) from the image.
        
        Args:
            image: Input image (grayscale)
            kernel_size: Size of the kernel to estimate
            
        Returns:
            Estimated blur kernel
        """
        try:
            # Method 1: Estimate based on image gradient analysis
            # Calculate gradients
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Analyze gradient distribution to determine blur type
            mean_gradient = np.mean(gradient_magnitude)
            
            if mean_gradient < 5:  # Heavy blur
                # Create Gaussian kernel for heavy blur
                sigma = kernel_size / 3
                ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
                xx, yy = np.meshgrid(ax, ax)
                kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            elif mean_gradient < 10:  # Moderate blur
                # Create smaller Gaussian kernel
                sigma = kernel_size / 4
                ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
                xx, yy = np.meshgrid(ax, ax)
                kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            else:  # Light blur or motion blur
                # Check for motion blur direction
                angle = self._estimate_motion_blur_angle(image)
                if angle is not None:
                    kernel = self._create_motion_kernel(kernel_size, angle)
                else:
                    # Default to small Gaussian
                    sigma = kernel_size / 5
                    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
                    xx, yy = np.meshgrid(ax, ax)
                    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            
            # Normalize kernel
            kernel = kernel / np.sum(kernel)
            return kernel
            
        except Exception as e:
            self.logger.warning(f"Kernel estimation failed: {str(e)}, using default Gaussian")
            # Default Gaussian kernel
            sigma = kernel_size / 4
            ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
            xx, yy = np.meshgrid(ax, ax)
            kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            return kernel / np.sum(kernel)
    
    def _estimate_motion_blur_angle(self, image: np.ndarray) -> Optional[float]:
        """
        Estimate the angle of motion blur using Fourier analysis.
        
        Args:
            image: Input image (grayscale)
            
        Returns:
            Estimated angle in degrees or None if not motion blur
        """
        try:
            # Apply Fourier Transform
            f_transform = np.fft.fft2(image)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Find dominant direction in frequency domain
            # Motion blur creates a line pattern in frequency domain
            edges = cv2.Canny(magnitude_spectrum.astype(np.uint8), 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None and len(lines) > 0:
                # Get the most prominent line angle
                theta = lines[0][0][1]
                angle = np.degrees(theta)
                return angle
            
            return None
            
        except Exception:
            return None
    
    def _create_motion_kernel(self, size: int, angle: float) -> np.ndarray:
        """
        Create a motion blur kernel.
        
        Args:
            size: Size of the kernel
            angle: Angle of motion in degrees
            
        Returns:
            Motion blur kernel
        """
        kernel = np.zeros((size, size))
        center = size // 2
        angle_rad = np.radians(angle)
        
        # Create line kernel
        for i in range(size):
            offset = i - center
            x = int(center + offset * np.cos(angle_rad))
            y = int(center + offset * np.sin(angle_rad))
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1
        
        # Normalize
        kernel = kernel / (np.sum(kernel) + 1e-8)
        return kernel
    
    def richardson_lucy_deconvolution(self, image: np.ndarray, psf: np.ndarray, 
                                     iterations: int = 30) -> np.ndarray:
        """
        Perform Richardson-Lucy deconvolution.
        
        Args:
            image: Blurred input image
            psf: Point Spread Function (blur kernel)
            iterations: Number of iterations
            
        Returns:
            Deblurred image
        """
        try:
            from skimage import restoration, img_as_float, img_as_ubyte
            
            # Convert to float
            image_float = img_as_float(image)
            
            # Apply Richardson-Lucy deconvolution
            deblurred = restoration.richardson_lucy(image_float, psf, 
                                                   num_iter=iterations, 
                                                   clip=True)
            
            # Convert back to uint8
            result = img_as_ubyte(np.clip(deblurred, 0, 1))
            
            self.logger.info(f"Richardson-Lucy deconvolution completed ({iterations} iterations)")
            return result
            
        except Exception as e:
            self.logger.error(f"Richardson-Lucy deconvolution failed: {str(e)}")
            return image


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Deblur images using various techniques")
    parser.add_argument("input", help="Path to input blurred image")
    parser.add_argument("output", help="Path for output deblurred image")
    parser.add_argument(
        "--method", 
        choices=["cv2", "wiener", "deep_learning"],
        default="cv2",
        help="Deblurring method to use (default: cv2)"
    )
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Initialize deblurrer
    deblurrer = ImageDeblurrer(args.config)
    
    # Process image
    success = deblurrer.deblur_image(args.input, args.output, args.method)
    
    if success:
        logger.info("Image deblurring completed successfully!")
        sys.exit(0)
    else:
        logger.error("Image deblurring failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
