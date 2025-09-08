"""
Blur detection and analysis module.

This module provides functionality to detect and analyze different types of blur
in images, including motion blur, gaussian blur, and out-of-focus blur.
"""

import logging
from typing import Dict, Tuple, Optional, Any
import numpy as np
import cv2
from scipy import signal, ndimage
from skimage import filters, measure


class BlurDetector:
    """Detects and analyzes blur in images."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_blur(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive blur analysis of an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing blur analysis results
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        analysis = {
            'blur_score': self.calculate_blur_score(gray),
            'blur_type': self.detect_blur_type(gray),
            'blur_extent': self.measure_blur_extent(gray),
            'focus_measure': self.calculate_focus_measure(gray),
            'edge_strength': self.calculate_edge_strength(gray),
            'frequency_analysis': self.frequency_domain_analysis(gray),
            'local_blur_map': self.create_blur_map(gray)
        }
        
        # Determine overall blur level
        blur_score = analysis['blur_score']
        if blur_score < 100:
            analysis['blur_level'] = 'severe'
        elif blur_score < 300:
            analysis['blur_level'] = 'moderate'
        elif blur_score < 500:
            analysis['blur_level'] = 'mild'
        else:
            analysis['blur_level'] = 'minimal'
        
        return analysis
    
    def calculate_blur_score(self, image: np.ndarray) -> float:
        """
        Calculate overall blur score using variance of Laplacian.
        Higher values indicate less blur.
        
        Args:
            image: Grayscale image
            
        Returns:
            Blur score (higher = sharper)
        """
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        score = laplacian.var()
        return score
    
    def detect_blur_type(self, image: np.ndarray) -> str:
        """
        Detect the type of blur present in the image.
        
        Args:
            image: Grayscale image
            
        Returns:
            Detected blur type: 'motion', 'gaussian', 'out_of_focus', or 'mixed'
        """
        # Analyze frequency domain
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Motion blur detection (looks for directional patterns)
        edges = cv2.Canny(magnitude_spectrum.astype(np.uint8), 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        has_motion_blur = lines is not None and len(lines) > 0
        
        # Gaussian blur detection (circular pattern in frequency domain)
        center = tuple(np.array(magnitude_spectrum.shape) // 2)
        radius_profile = self._radial_profile(magnitude_spectrum, center)
        gaussian_score = np.std(radius_profile)
        
        # Out-of-focus blur detection (ring pattern)
        circles = cv2.HoughCircles(
            magnitude_spectrum.astype(np.uint8),
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=50
        )
        has_out_of_focus = circles is not None and len(circles) > 0
        
        # Determine blur type
        if has_motion_blur and not has_out_of_focus:
            return 'motion'
        elif has_out_of_focus:
            return 'out_of_focus'
        elif gaussian_score < 10:
            return 'gaussian'
        else:
            return 'mixed'
    
    def measure_blur_extent(self, image: np.ndarray) -> Dict[str, float]:
        """
        Measure the extent and parameters of blur.
        
        Args:
            image: Grayscale image
            
        Returns:
            Dictionary with blur extent measurements
        """
        # Estimate blur kernel size using autocorrelation
        autocorr = signal.correlate2d(image, image, mode='same')
        autocorr_center = autocorr[image.shape[0]//2, image.shape[1]//2]
        autocorr_normalized = autocorr / autocorr_center
        
        # Find where autocorrelation drops to 0.5
        threshold_map = autocorr_normalized > 0.5
        labeled = measure.label(threshold_map)
        center_label = labeled[image.shape[0]//2, image.shape[1]//2]
        center_region = labeled == center_label
        
        # Estimate kernel size from region
        kernel_size_estimate = np.sqrt(np.sum(center_region))
        
        # Estimate motion blur angle if applicable
        blur_type = self.detect_blur_type(image)
        motion_angle = None
        if blur_type == 'motion':
            motion_angle = self._estimate_motion_angle(image)
        
        return {
            'estimated_kernel_size': kernel_size_estimate,
            'motion_angle': motion_angle,
            'blur_radius': kernel_size_estimate / 2
        }
    
    def calculate_focus_measure(self, image: np.ndarray) -> float:
        """
        Calculate focus measure using multiple methods.
        
        Args:
            image: Grayscale image
            
        Returns:
            Focus measure (higher = more focused)
        """
        # Method 1: Gradient-based focus measure
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_measure = np.mean(grad_x**2) + np.mean(grad_y**2)
        
        # Method 2: Tenengrad focus measure
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = np.mean(gx**2 + gy**2)
        
        # Method 3: Normalized variance
        normalized_var = np.var(image) / np.mean(image) if np.mean(image) > 0 else 0
        
        # Combine measures
        focus_measure = gradient_measure * 0.4 + tenengrad * 0.4 + normalized_var * 0.2
        
        return focus_measure
    
    def calculate_edge_strength(self, image: np.ndarray) -> float:
        """
        Calculate average edge strength in the image.
        
        Args:
            image: Grayscale image
            
        Returns:
            Average edge strength
        """
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calculate edge gradient strength
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Only consider strong edges
        strong_edges = gradient_magnitude[edges > 0]
        avg_edge_strength = np.mean(strong_edges) if len(strong_edges) > 0 else 0
        
        return avg_edge_strength * edge_density
    
    def frequency_domain_analysis(self, image: np.ndarray) -> Dict[str, float]:
        """
        Analyze blur in frequency domain.
        
        Args:
            image: Grayscale image
            
        Returns:
            Frequency domain analysis results
        """
        # Compute FFT
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Calculate high frequency content ratio
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Define frequency bands
        low_freq_radius = min(rows, cols) // 8
        high_freq_radius = min(rows, cols) // 4
        
        # Create masks
        y, x = np.ogrid[:rows, :cols]
        center_mask = (x - ccol)**2 + (y - crow)**2 <= low_freq_radius**2
        high_freq_mask = (x - ccol)**2 + (y - crow)**2 > high_freq_radius**2
        
        # Calculate energy in different frequency bands
        total_energy = np.sum(magnitude_spectrum**2)
        low_freq_energy = np.sum(magnitude_spectrum[center_mask]**2)
        high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask]**2)
        
        return {
            'high_freq_ratio': high_freq_energy / total_energy if total_energy > 0 else 0,
            'low_freq_ratio': low_freq_energy / total_energy if total_energy > 0 else 0,
            'frequency_entropy': self._calculate_spectrum_entropy(magnitude_spectrum),
            'spectral_slope': self._calculate_spectral_slope(magnitude_spectrum)
        }
    
    def create_blur_map(self, image: np.ndarray, window_size: int = 32) -> np.ndarray:
        """
        Create a local blur map showing blur levels across the image.
        
        Args:
            image: Grayscale image
            window_size: Size of the analysis window
            
        Returns:
            Blur map (same size as input, values 0-1 where 1 is sharpest)
        """
        rows, cols = image.shape
        blur_map = np.zeros_like(image, dtype=np.float32)
        
        step = window_size // 2
        
        for i in range(0, rows - window_size, step):
            for j in range(0, cols - window_size, step):
                window = image[i:i+window_size, j:j+window_size]
                
                # Calculate local blur score
                local_score = cv2.Laplacian(window, cv2.CV_64F).var()
                
                # Normalize and assign to map
                blur_map[i:i+window_size, j:j+window_size] = local_score
        
        # Normalize blur map to 0-1 range
        if blur_map.max() > blur_map.min():
            blur_map = (blur_map - blur_map.min()) / (blur_map.max() - blur_map.min())
        
        return blur_map
    
    def _radial_profile(self, data: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
        """
        Calculate radial profile of 2D data.
        
        Args:
            data: 2D array
            center: Center point for radial profile
            
        Returns:
            Radial profile array
        """
        y, x = np.indices(data.shape)
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        r = r.astype(np.int32)
        
        tbin = np.bincount(r.ravel(), data.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / nr
        
        return radialprofile
    
    def _estimate_motion_angle(self, image: np.ndarray) -> float:
        """
        Estimate motion blur angle.
        
        Args:
            image: Grayscale image
            
        Returns:
            Estimated angle in degrees
        """
        # Use Radon transform to detect dominant direction
        from skimage.transform import radon
        
        theta = np.linspace(0., 180., 180, endpoint=False)
        sinogram = radon(image, theta=theta, circle=True)
        
        # Find angle with maximum variance (indicates motion direction)
        variances = np.var(sinogram, axis=0)
        motion_angle = theta[np.argmax(variances)]
        
        return motion_angle
    
    def _calculate_spectrum_entropy(self, spectrum: np.ndarray) -> float:
        """
        Calculate entropy of frequency spectrum.
        
        Args:
            spectrum: Frequency spectrum
            
        Returns:
            Spectral entropy
        """
        # Normalize spectrum to probability distribution
        spectrum_normalized = spectrum / np.sum(spectrum)
        spectrum_flat = spectrum_normalized.flatten()
        
        # Remove zeros to avoid log(0)
        spectrum_flat = spectrum_flat[spectrum_flat > 0]
        
        # Calculate entropy
        entropy = -np.sum(spectrum_flat * np.log2(spectrum_flat))
        
        return entropy
    
    def _calculate_spectral_slope(self, spectrum: np.ndarray) -> float:
        """
        Calculate the slope of the frequency spectrum decay.
        
        Args:
            spectrum: Frequency spectrum
            
        Returns:
            Spectral slope
        """
        center = tuple(np.array(spectrum.shape) // 2)
        radial_prof = self._radial_profile(spectrum, center)
        
        # Fit linear regression to log-log plot
        valid_indices = radial_prof > 0
        if np.sum(valid_indices) < 2:
            return 0
        
        x = np.log(np.arange(len(radial_prof))[valid_indices] + 1)
        y = np.log(radial_prof[valid_indices])
        
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        
        return slope
    
    def suggest_deblur_method(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Suggest the best deblurring method based on image analysis.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with suggested method and parameters
        """
        analysis = self.analyze_blur(image)
        
        suggestions = {
            'primary_method': None,
            'alternative_methods': [],
            'parameters': {},
            'confidence': 0.0
        }
        
        blur_type = analysis['blur_type']
        blur_level = analysis['blur_level']
        
        if blur_type == 'motion':
            suggestions['primary_method'] = 'richardson_lucy'
            suggestions['parameters'] = {
                'iterations': 30 if blur_level == 'severe' else 20,
                'kernel_size': int(analysis['blur_extent']['estimated_kernel_size']),
                'angle': analysis['blur_extent']['motion_angle']
            }
            suggestions['alternative_methods'] = ['wiener', 'deep_learning']
            suggestions['confidence'] = 0.8
            
        elif blur_type == 'gaussian':
            suggestions['primary_method'] = 'wiener'
            suggestions['parameters'] = {
                'kernel_size': int(analysis['blur_extent']['estimated_kernel_size']),
                'noise_level': 0.01
            }
            suggestions['alternative_methods'] = ['cv2', 'richardson_lucy']
            suggestions['confidence'] = 0.7
            
        elif blur_type == 'out_of_focus':
            suggestions['primary_method'] = 'deep_learning'
            suggestions['parameters'] = {
                'model': 'unet',
                'preprocessing': 'adaptive'
            }
            suggestions['alternative_methods'] = ['richardson_lucy', 'wiener']
            suggestions['confidence'] = 0.6
            
        else:  # mixed or unknown
            suggestions['primary_method'] = 'cv2'
            suggestions['parameters'] = {
                'adaptive': True,
                'strength': 'auto'
            }
            suggestions['alternative_methods'] = ['wiener', 'deep_learning']
            suggestions['confidence'] = 0.5
        
        suggestions['analysis'] = analysis
        
        return suggestions
