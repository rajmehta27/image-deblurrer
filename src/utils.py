"""
Utility functions for the image deblurrer project.

This module contains various helper functions for logging,
file validation, and other common operations.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional


def setup_logging(level: str = "INFO") -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def validate_image_path(image_path: str) -> bool:
    """
    Validate if the given path is a valid image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        True if valid image path, False otherwise
    """
    if not os.path.exists(image_path):
        return False
    
    # Check if it's a file (not directory)
    if not os.path.isfile(image_path):
        return False
    
    # Check file extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    file_ext = Path(image_path).suffix.lower()
    
    return file_ext in valid_extensions


def create_output_path(input_path: str, suffix: str = "_deblurred", 
                      output_dir: Optional[str] = None) -> str:
    """
    Create an appropriate output path for the deblurred image.
    
    Args:
        input_path: Path to the input image
        suffix: Suffix to add to the filename
        output_dir: Optional output directory
        
    Returns:
        Output path for the deblurred image
    """
    input_path_obj = Path(input_path)
    
    if output_dir:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_path = output_dir_path / f"{input_path_obj.stem}{suffix}{input_path_obj.suffix}"
    else:
        output_path = input_path_obj.parent / f"{input_path_obj.stem}{suffix}{input_path_obj.suffix}"
    
    return str(output_path)


def get_supported_formats() -> List[str]:
    """
    Get list of supported image formats.
    
    Returns:
        List of supported file extensions
    """
    return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']


def validate_config_file(config_path: str) -> bool:
    """
    Validate configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        True if valid config file, False otherwise
    """
    if not os.path.exists(config_path):
        return False
    
    if not os.path.isfile(config_path):
        return False
    
    # Check if it's a YAML file
    valid_extensions = {'.yaml', '.yml'}
    file_ext = Path(config_path).suffix.lower()
    
    return file_ext in valid_extensions


def ensure_directory_exists(directory_path: str) -> bool:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        True if directory exists or was created successfully, False otherwise
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in MB
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)  # Convert to MB
    except Exception:
        return 0.0


def print_system_info() -> None:
    """Print system information for debugging purposes."""
    import platform
    
    print("System Information:")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  Architecture: {platform.machine()}")
    
    # Check for GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  CUDA available: Yes (Devices: {torch.cuda.device_count()})")
            for i in range(torch.cuda.device_count()):
                print(f"    Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("  CUDA available: No")
    except ImportError:
        print("  PyTorch: Not installed")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  TensorFlow GPU: Yes (Devices: {len(gpus)})")
        else:
            print("  TensorFlow GPU: No")
    except ImportError:
        print("  TensorFlow: Not installed")


def calculate_image_metrics(original: 'np.ndarray', processed: 'np.ndarray') -> dict:
    """
    Calculate image quality metrics between original and processed images.
    
    Args:
        original: Original image array
        processed: Processed image array
        
    Returns:
        Dictionary containing quality metrics
    """
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        import numpy as np
        
        # Ensure images have same shape
        if original.shape != processed.shape:
            raise ValueError("Images must have the same shape")
        
        # Calculate PSNR
        psnr = peak_signal_noise_ratio(original, processed)
        
        # Calculate SSIM
        if len(original.shape) == 3:  # Color image
            ssim = structural_similarity(original, processed, multichannel=True, channel_axis=-1)
        else:  # Grayscale image
            ssim = structural_similarity(original, processed)
        
        # Calculate MSE
        mse = np.mean((original - processed) ** 2)
        
        return {
            'psnr': psnr,
            'ssim': ssim,
            'mse': mse
        }
        
    except ImportError:
        return {'error': 'scikit-image not available for metrics calculation'}
    except Exception as e:
        return {'error': f'Error calculating metrics: {str(e)}'}


def create_batch_processing_list(input_dir: str, output_dir: str, 
                                batch_size: int = 10) -> List[Tuple[str, str]]:
    """
    Create a list of input-output path pairs for batch processing.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory for output images
        batch_size: Maximum number of files per batch
        
    Returns:
        List of (input_path, output_path) tuples
    """
    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir)
    
    if not input_dir_path.exists():
        return []
    
    # Ensure output directory exists
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = []
    for ext in get_supported_formats():
        image_files.extend(input_dir_path.glob(f"*{ext}"))
        image_files.extend(input_dir_path.glob(f"*{ext.upper()}"))
    
    # Create input-output pairs
    file_pairs = []
    for input_file in image_files[:batch_size]:  # Limit to batch_size
        output_file = output_dir_path / f"{input_file.stem}_deblurred{input_file.suffix}"
        file_pairs.append((str(input_file), str(output_file)))
    
    return file_pairs


def progress_bar(current: int, total: int, width: int = 50) -> str:
    """
    Create a simple progress bar string.
    
    Args:
        current: Current progress value
        total: Total progress value
        width: Width of the progress bar
        
    Returns:
        Progress bar string
    """
    if total == 0:
        return "[" + "=" * width + "]"
    
    percentage = current / total
    filled = int(width * percentage)
    bar = "=" * filled + "-" * (width - filled)
    
    return f"[{bar}] {current}/{total} ({percentage*100:.1f}%)"


class Timer:
    """Simple timer context manager for performance measurement."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"{self.description} took {elapsed:.2f} seconds")
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
