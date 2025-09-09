"""
Image processing utilities for the image deblurrer.

This module contains helper functions and classes for loading, processing,
and saving images.
"""

import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io, restoration, filters


class ImageProcessor:
    """Handles image processing operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load an image from file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image as numpy array or None if failed
        """
        try:
            if not os.path.exists(image_path):
                self.logger.error(f"Image file not found: {image_path}")
                return None
                
            # Try loading with OpenCV first
            image = cv2.imread(image_path)
            if image is not None:
                # Convert BGR to RGB for consistency
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.logger.debug(f"Loaded image with shape: {image.shape}")
                return image
            
            # Fallback to PIL
            pil_image = Image.open(image_path)
            image = np.array(pil_image)
            self.logger.debug(f"Loaded image with PIL, shape: {image.shape}")
            return image
            
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {str(e)}")
            return None
    
    def save_image(self, image: np.ndarray, output_path: str) -> bool:
        """
        Save an image to file.
        
        Args:
            image: Image as numpy array
            output_path: Path where to save the image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:  # Only create directory if there is one
                os.makedirs(output_dir, exist_ok=True)
            
            # Ensure image is in the correct format
            if image.dtype != np.uint8:
                # Normalize to 0-255 range
                image = self._normalize_image(image)
            
            # Save with OpenCV (convert RGB back to BGR)
            if len(image.shape) == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                success = cv2.imwrite(output_path, image_bgr)
            else:
                success = cv2.imwrite(output_path, image)
            
            if success:
                self.logger.debug(f"Image saved to: {output_path}")
                return True
            else:
                self.logger.error(f"Failed to save image to: {output_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving image: {str(e)}")
            return False
    
    def preprocess_for_model(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for deep learning model input.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Normalize to 0-1 range
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        return image
    
    def postprocess_from_model(self, image: np.ndarray) -> np.ndarray:
        """
        Postprocess image from deep learning model output.
        
        Args:
            image: Model output
            
        Returns:
            Postprocessed image
        """
        # Remove batch dimension if present
        if len(image.shape) == 4 and image.shape[0] == 1:
            image = np.squeeze(image, axis=0)
        
        # Convert back to uint8
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = self._normalize_image(image)
        
        return image
    
    def resize_image(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to specified dimensions.
        
        Args:
            image: Input image
            size: Target size (width, height)
            
        Returns:
            Resized image
        """
        return cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
    
    def apply_gaussian_blur(self, image: np.ndarray, kernel_size: int = 15, 
                           sigma: float = 2.0) -> np.ndarray:
        """
        Apply Gaussian blur to simulate image degradation.
        
        Args:
            image: Input image
            kernel_size: Size of the Gaussian kernel
            sigma: Standard deviation for Gaussian kernel
            
        Returns:
            Blurred image
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def apply_motion_blur(self, image: np.ndarray, size: int = 15, 
                         angle: float = 0) -> np.ndarray:
        """
        Apply motion blur to simulate camera shake.
        
        Args:
            image: Input image
            size: Size of the motion blur kernel
            angle: Angle of motion in degrees
            
        Returns:
            Motion blurred image
        """
        # Create motion blur kernel
        kernel = self._create_motion_blur_kernel(size, angle)
        return cv2.filter2D(image, -1, kernel)
    
    def enhance_contrast(self, image: np.ndarray, alpha: float = 1.5, 
                        beta: int = 10) -> np.ndarray:
        """
        Enhance image contrast.
        
        Args:
            image: Input image
            alpha: Contrast control (1.0-3.0)
            beta: Brightness control (0-100)
            
        Returns:
            Contrast enhanced image
        """
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return enhanced
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to 0-255 uint8 range.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        # Clip values to valid range
        image = np.clip(image, 0, 1) if image.max() <= 1.0 else np.clip(image, 0, 255)
        
        # Scale to 0-255 if needed
        if image.max() <= 1.0:
            image = image * 255
        
        return image.astype(np.uint8)
    
    def _create_motion_blur_kernel(self, size: int, angle: float) -> np.ndarray:
        """
        Create a motion blur kernel.
        
        Args:
            size: Size of the kernel
            angle: Angle of motion in degrees
            
        Returns:
            Motion blur kernel
        """
        kernel = np.zeros((size, size))
        
        # Convert angle to radians
        angle_rad = np.deg2rad(angle)
        
        # Calculate the line coordinates
        center = size // 2
        dx = int(np.cos(angle_rad) * center)
        dy = int(np.sin(angle_rad) * center)
        
        # Draw line in the kernel
        cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1, 1)
        
        # Normalize kernel
        kernel = kernel / kernel.sum()
        
        return kernel
    
    def display_images(self, images: list, titles: list = None, 
                      figsize: Tuple[int, int] = (15, 5)):
        """
        Display multiple images side by side.
        
        Args:
            images: List of images to display
            titles: List of titles for each image
            figsize: Figure size for matplotlib
        """
        if titles is None:
            titles = [f"Image {i+1}" for i in range(len(images))]
        
        fig, axes = plt.subplots(1, len(images), figsize=figsize)
        if len(images) == 1:
            axes = [axes]
        
        for i, (img, title) in enumerate(zip(images, titles)):
            axes[i].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            axes[i].set_title(title)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
