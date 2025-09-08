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
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Import our custom modules
from image_processor import ImageProcessor
from model_loader import ModelLoader
from utils import setup_logging, validate_image_path


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
        
    def deblur_image(self, input_path: str, output_path: str, method: str = 'cv2') -> bool:
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
                
            self.logger.info(f"Processing image: {input_path} using method: {method}")
            
            # Apply deblurring based on method
            if method == 'cv2':
                deblurred = self._deblur_cv2(image)
            elif method == 'wiener':
                deblurred = self._deblur_wiener(image)
            elif method == 'deep_learning':
                deblurred = self._deblur_deep_learning(image)
            else:
                self.logger.error(f"Unknown deblurring method: {method}")
                return False
                
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
        Deblur using OpenCV techniques.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Deblurred image
        """
        # Simple sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Additional processing can be added here
        return sharpened
    
    def _deblur_wiener(self, image: np.ndarray) -> np.ndarray:
        """
        Deblur using Wiener filtering.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Deblurred image
        """
        # Placeholder for Wiener filter implementation
        # This would require more sophisticated signal processing
        self.logger.warning("Wiener filtering not fully implemented yet")
        return self._deblur_cv2(image)  # Fallback to CV2 method
    
    def _deblur_deep_learning(self, image: np.ndarray) -> np.ndarray:
        """
        Deblur using deep learning model.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Deblurred image
        """
        # Load and use the deep learning model
        model = self.model_loader.load_model()
        if model is None:
            self.logger.warning("Deep learning model not available, falling back to CV2")
            return self._deblur_cv2(image)
        
        # Preprocess image for model
        processed_input = self.image_processor.preprocess_for_model(image)
        
        # Run inference (placeholder)
        # result = model.predict(processed_input)
        # deblurred = self.image_processor.postprocess_from_model(result)
        
        self.logger.warning("Deep learning inference not fully implemented yet")
        return self._deblur_cv2(image)  # Fallback to CV2 method


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
