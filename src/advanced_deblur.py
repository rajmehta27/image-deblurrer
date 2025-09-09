"""
Advanced deblurring algorithms with state-of-the-art techniques.

This module provides enhanced deblurring methods with multi-scale processing,
advanced PSF estimation, and optimized deep learning integration.
"""

import logging
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import cv2
from scipy import signal, fftpack, ndimage
from skimage import restoration, filters, measure
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp


class AdvancedDeblurrer:
    """Advanced deblurring with state-of-the-art algorithms."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.use_gpu = self.config.get('use_gpu', True) and PYTORCH_AVAILABLE and torch.cuda.is_available()
        
        if self.use_gpu:
            self.device = torch.device('cuda')
            self.logger.info(f"GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
        elif PYTORCH_AVAILABLE:
            self.device = torch.device('cpu')
        
    # ============================================================================
    # Enhanced CV2 Method with Multi-scale Processing
    # ============================================================================
    
    def deblur_cv2_advanced(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced CV2 deblurring with multi-scale processing and adaptive kernels.
        
        Args:
            image: Input blurred image
            
        Returns:
            Deblurred image
        """
        try:
            # Detect motion direction and blur severity
            motion_direction = self._detect_motion_direction(image)
            blur_severity = self._estimate_blur_severity(image)
            
            # Adaptive kernel size based on blur severity
            kernel_size = self._calculate_adaptive_kernel_size(blur_severity)
            
            # Multi-scale pyramid processing
            pyramid_levels = min(4, int(np.log2(min(image.shape[:2]))) - 2)
            result = self._multiscale_deblur(image, pyramid_levels, kernel_size, motion_direction)
            
            # Advanced sharpening based on local features
            result = self._adaptive_sharpening(result, blur_severity)
            
            # Edge-preserving smoothing
            result = self._edge_preserving_smooth(result)
            
            # Final enhancement
            result = self._enhance_details(result, image)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Advanced CV2 deblurring failed: {str(e)}")
            return image
    
    def _detect_motion_direction(self, image: np.ndarray) -> Optional[float]:
        """Detect motion blur direction using gradient analysis."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Compute gradients
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient orientation histogram
        angles = np.arctan2(gy, gx)
        hist, _ = np.histogram(angles, bins=180, range=(-np.pi, np.pi))
        
        # Find dominant direction
        dominant_bin = np.argmax(hist)
        dominant_angle = (dominant_bin / 180.0) * 2 * np.pi - np.pi
        
        # Check if motion blur is significant
        if hist[dominant_bin] > np.mean(hist) * 2:
            return np.degrees(dominant_angle)
        return None
    
    def _estimate_blur_severity(self, image: np.ndarray) -> float:
        """Estimate blur severity using multiple metrics."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Laplacian variance (lower = more blur)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Gradient magnitude
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(gx**2 + gy**2).mean()
        
        # Frequency domain analysis
        f = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(f_shift)
        
        # High frequency content ratio
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        r = min(rows, cols) // 4
        
        total_energy = np.sum(magnitude_spectrum**2)
        high_freq_mask = np.zeros_like(magnitude_spectrum)
        y, x = np.ogrid[:rows, :cols]
        mask = (x - ccol)**2 + (y - crow)**2 > r**2
        high_freq_energy = np.sum(magnitude_spectrum[mask]**2)
        
        freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        # Combine metrics (normalize to 0-1, where 1 is most blurred)
        blur_score = 1.0 / (1.0 + laplacian_var / 100.0)
        blur_score *= (1.0 - min(gradient_mag / 50.0, 1.0))
        blur_score *= (1.0 - freq_ratio)
        
        return np.clip(blur_score, 0, 1)
    
    def _calculate_adaptive_kernel_size(self, blur_severity: float) -> int:
        """Calculate adaptive kernel size based on blur severity."""
        # Map blur severity to kernel size (3 to 15)
        min_kernel = 3
        max_kernel = 15
        kernel_size = int(min_kernel + (max_kernel - min_kernel) * blur_severity)
        
        # Ensure odd kernel size
        return kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    def _multiscale_deblur(self, image: np.ndarray, levels: int, 
                          kernel_size: int, motion_direction: Optional[float]) -> np.ndarray:
        """Multi-scale pyramid deblurring."""
        # Build Gaussian pyramid
        pyramid = [image]
        for _ in range(levels - 1):
            pyramid.append(cv2.pyrDown(pyramid[-1]))
        
        # Process from coarsest to finest
        for level in range(levels - 1, -1, -1):
            current = pyramid[level]
            
            # Apply appropriate deblurring based on motion detection
            if motion_direction is not None:
                # Directional deblurring for motion blur
                kernel = self._create_directional_kernel(kernel_size, motion_direction)
                deblurred = cv2.filter2D(current, -1, kernel)
            else:
                # Non-directional deblurring
                deblurred = self._apply_smart_sharpening(current, kernel_size)
            
            # Apply bilateral filter to reduce noise
            deblurred = cv2.bilateralFilter(deblurred, 9, 75, 75)
            
            if level > 0:
                # Upscale and blend with next level
                h, w = pyramid[level - 1].shape[:2]
                deblurred = cv2.resize(deblurred, (w, h), interpolation=cv2.INTER_CUBIC)
                
                # Blend with next level using Laplacian pyramid
                next_level = pyramid[level - 1]
                alpha = 0.6  # Blending factor
                pyramid[level - 1] = cv2.addWeighted(next_level, alpha, deblurred, 1 - alpha, 0)
            else:
                pyramid[0] = deblurred
        
        return pyramid[0]
    
    def _create_directional_kernel(self, size: int, angle: float) -> np.ndarray:
        """Create directional kernel for motion blur correction."""
        kernel = np.zeros((size, size))
        center = size // 2
        angle_rad = np.radians(angle)
        
        # Create line kernel in the direction of motion
        for i in range(size):
            offset = i - center
            x = int(center + offset * np.cos(angle_rad))
            y = int(center + offset * np.sin(angle_rad))
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1
        
        # Normalize
        kernel = kernel / (np.sum(kernel) + 1e-10)
        
        # Create inverse kernel for deconvolution
        inverse_kernel = np.zeros_like(kernel)
        inverse_kernel[center, center] = 2
        inverse_kernel -= kernel
        
        return inverse_kernel
    
    def _apply_smart_sharpening(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        """Apply intelligent sharpening based on local features."""
        # Unsharp masking with adaptive strength
        gaussian = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # Calculate local variance for adaptive sharpening
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        local_var = ndimage.generic_filter(gray, np.var, size=5)
        local_var = local_var / (local_var.max() + 1e-10)
        
        # Apply stronger sharpening in high-detail areas
        if len(image.shape) == 3:
            local_var = np.stack([local_var] * 3, axis=-1)
        
        sharpened = image + (image - gaussian) * (1 + local_var)
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def _adaptive_sharpening(self, image: np.ndarray, blur_severity: float) -> np.ndarray:
        """Apply adaptive sharpening based on blur severity."""
        # Stronger sharpening for more severe blur
        strength = 0.5 + 1.5 * blur_severity
        
        # Create adaptive sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1, 8 + strength, -1],
                          [-1, -1, -1]]) / (strength + 1)
        
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Blend with original based on local features
        edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
                         if len(image.shape) == 3 else image, 50, 150)
        edges = cv2.dilate(edges, None, iterations=2)
        
        if len(image.shape) == 3:
            edges = np.stack([edges] * 3, axis=-1) / 255.0
        else:
            edges = edges / 255.0
        
        result = sharpened * edges + image * (1 - edges)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _edge_preserving_smooth(self, image: np.ndarray) -> np.ndarray:
        """Apply edge-preserving smoothing."""
        # Use guided filter for edge preservation
        return cv2.ximgproc.guidedFilter(image, image, 5, 20) if hasattr(cv2, 'ximgproc') else image
    
    def _enhance_details(self, result: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Enhance fine details in the deblurred image."""
        # Extract high-frequency details from result
        blurred_result = cv2.GaussianBlur(result, (5, 5), 1)
        details = result.astype(np.float32) - blurred_result.astype(np.float32)
        
        # Enhance details
        enhanced = result.astype(np.float32) + details * 0.5
        
        # Blend with original to preserve natural look
        final = enhanced * 0.85 + original.astype(np.float32) * 0.15
        
        return np.clip(final, 0, 255).astype(np.uint8)
    
    # ============================================================================
    # Enhanced Wiener Filter with Advanced PSF Estimation
    # ============================================================================
    
    def deblur_wiener_advanced(self, image: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """
        Advanced Wiener filtering with improved PSF estimation.
        
        Args:
            image: Input blurred image
            noise_level: Estimated noise level
            
        Returns:
            Deblurred image
        """
        try:
            # Convert to float
            img_float = image.astype(np.float64) / 255.0
            
            # Estimate optimal PSF
            psf = self._estimate_advanced_psf(img_float)
            
            # Apply Wiener filter with regularization
            if len(img_float.shape) == 3:
                # Process each channel
                result = np.zeros_like(img_float)
                for i in range(3):
                    result[:, :, i] = self._wiener_filter_channel(
                        img_float[:, :, i], psf, noise_level
                    )
            else:
                result = self._wiener_filter_channel(img_float, psf, noise_level)
            
            # Post-process to reduce artifacts
            result = self._reduce_ringing_artifacts(result)
            
            # Convert back to uint8
            result = np.clip(result * 255, 0, 255).astype(np.uint8)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Advanced Wiener filtering failed: {str(e)}")
            return image
    
    def _estimate_advanced_psf(self, image: np.ndarray) -> np.ndarray:
        """Estimate PSF using advanced techniques."""
        # Convert to grayscale if needed
        if len(image.shape) == 2:
            gray = image
        else:
            # Ensure uint8 for color conversion
            if image.dtype != np.uint8:
                img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
                gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
        
        # Estimate PSF size using cepstral analysis
        psf_size = self._estimate_optimal_psf_size(gray)
        
        # Frequency domain analysis
        f = fftpack.fft2(gray)
        f_shift = fftpack.fftshift(f)
        
        # Cepstral domain analysis
        log_spectrum = np.log(np.abs(f_shift) + 1e-10)
        cepstrum = np.real(fftpack.ifft2(log_spectrum))
        
        # Refine PSF estimation
        psf = self._refine_psf(cepstrum, psf_size)
        
        # Apply blind deconvolution for further refinement
        psf = self._blind_psf_estimation(gray, psf)
        
        return psf
    
    def _estimate_optimal_psf_size(self, image: np.ndarray) -> int:
        """Estimate optimal PSF size based on blur characteristics."""
        # Analyze autocorrelation
        autocorr = signal.correlate2d(image, image, mode='same')
        autocorr = autocorr / autocorr.max()
        
        # Find correlation length
        center = tuple(np.array(autocorr.shape) // 2)
        threshold = 0.5
        
        # Search for correlation drop-off
        for r in range(3, 30):
            y, x = np.ogrid[:autocorr.shape[0], :autocorr.shape[1]]
            mask = (x - center[1])**2 + (y - center[0])**2 <= r**2
            if np.mean(autocorr[mask]) < threshold:
                return 2 * r + 1
        
        return 15  # Default size
    
    def _refine_psf(self, cepstrum: np.ndarray, size: int) -> np.ndarray:
        """Refine PSF estimation using cepstral analysis."""
        # Extract PSF from cepstrum
        center = tuple(np.array(cepstrum.shape) // 2)
        half_size = size // 2
        
        # Extract region around center
        psf = cepstrum[
            center[0] - half_size:center[0] + half_size + 1,
            center[1] - half_size:center[1] + half_size + 1
        ]
        
        # Apply window to reduce edge effects
        window = np.outer(np.hanning(size), np.hanning(size))
        psf = psf * window
        
        # Normalize
        psf = psf - psf.min()
        psf = psf / (psf.sum() + 1e-10)
        
        return psf
    
    def _blind_psf_estimation(self, image: np.ndarray, initial_psf: np.ndarray, 
                            iterations: int = 10) -> np.ndarray:
        """Blind PSF estimation using iterative optimization."""
        psf = initial_psf.copy()
        
        for _ in range(iterations):
            # Estimate latent image
            latent = restoration.wiener(image, psf, balance=0.1, clip=False)
            
            # Update PSF estimate
            psf = restoration.wiener(image, latent, balance=0.1, clip=False)
            
            # Constrain PSF
            psf = np.maximum(psf, 0)
            psf = psf / (psf.sum() + 1e-10)
        
        return psf
    
    def _wiener_filter_channel(self, channel: np.ndarray, psf: np.ndarray, 
                              noise_level: float) -> np.ndarray:
        """Apply Wiener filter to a single channel."""
        # Pad image and PSF
        pad_height = psf.shape[0] // 2
        pad_width = psf.shape[1] // 2
        padded = np.pad(channel, ((pad_height, pad_height), (pad_width, pad_width)), 
                       mode='reflect')
        
        # Apply Wiener filter with improved regularization
        result = restoration.wiener(padded, psf, balance=noise_level, clip=False)
        
        # Remove padding
        result = result[pad_height:-pad_height, pad_width:-pad_width]
        
        return result
    
    def _reduce_ringing_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Reduce ringing artifacts from deconvolution."""
        # Apply edge-aware smoothing
        if len(image.shape) == 3:
            for i in range(3):
                image[:, :, i] = ndimage.median_filter(image[:, :, i], size=3)
        else:
            image = ndimage.median_filter(image, size=3)
        
        # Apply soft thresholding to suppress artifacts
        threshold = 0.01
        image = np.where(np.abs(image) < threshold, 0, image)
        
        return image
    
    # ============================================================================
    # Enhanced Deep Learning with Tiling and Post-processing
    # ============================================================================
    
    def deblur_deep_learning_advanced(self, image: np.ndarray, model: Any) -> np.ndarray:
        """
        Advanced deep learning deblurring with tiling for large images.
        
        Args:
            image: Input blurred image
            model: Deep learning model
            
        Returns:
            Deblurred image
        """
        try:
            # Determine optimal tile size based on available memory
            tile_size = self._calculate_optimal_tile_size(image.shape)
            
            # Preprocess image
            normalized = self._preprocess_for_model(image)
            
            # Split into tiles with overlap
            tiles, positions = self._split_into_tiles(normalized, tile_size, overlap=32)
            
            # Process tiles (parallel if possible)
            deblurred_tiles = self._process_tiles(tiles, model)
            
            # Merge tiles with blending
            result = self._merge_tiles_with_blending(
                deblurred_tiles, positions, image.shape, tile_size, overlap=32
            )
            
            # Post-process
            result = self._postprocess_prediction(result, image)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Advanced deep learning deblurring failed: {str(e)}")
            return image
    
    def _calculate_optimal_tile_size(self, image_shape: Tuple[int, ...]) -> int:
        """Calculate optimal tile size based on available memory."""
        if self.use_gpu and PYTORCH_AVAILABLE:
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            available_memory = gpu_memory - torch.cuda.memory_allocated()
            
            # Estimate memory per tile (rough estimate)
            bytes_per_pixel = 4  # float32
            channels = 3 if len(image_shape) == 3 else 1
            
            # Calculate tile size (with safety margin)
            max_pixels = available_memory // (bytes_per_pixel * channels * 4)  # 4x for processing overhead
            tile_size = int(np.sqrt(max_pixels))
            
            # Constrain to reasonable range
            tile_size = min(512, max(128, tile_size))
        else:
            # Default for CPU
            tile_size = 256
        
        # Ensure multiple of 32 for most models
        tile_size = (tile_size // 32) * 32
        
        return tile_size
    
    def _preprocess_for_model(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input."""
        # Normalize to [0, 1]
        normalized = image.astype(np.float32) / 255.0
        
        # Apply CLAHE for better contrast
        if len(normalized.shape) == 3:
            # Convert to uint8 for LAB conversion
            img_uint8 = (normalized * 255).astype(np.uint8)
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            # Merge channels
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            normalized = enhanced.astype(np.float32) / 255.0
        
        return normalized
    
    def _split_into_tiles(self, image: np.ndarray, tile_size: int, 
                         overlap: int = 32) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Split image into overlapping tiles."""
        tiles = []
        positions = []
        
        h, w = image.shape[:2]
        stride = tile_size - overlap
        
        for y in range(0, h - overlap, stride):
            for x in range(0, w - overlap, stride):
                # Calculate tile boundaries
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                
                # Adjust start if we're at the edge
                if y_end == h:
                    y = max(0, h - tile_size)
                if x_end == w:
                    x = max(0, w - tile_size)
                
                # Extract tile
                tile = image[y:y + tile_size, x:x + tile_size]
                
                # Pad if necessary
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    pad_h = tile_size - tile.shape[0]
                    pad_w = tile_size - tile.shape[1]
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                
                tiles.append(tile)
                positions.append((y, x))
        
        return tiles, positions
    
    def _process_tiles(self, tiles: List[np.ndarray], model: Any) -> List[np.ndarray]:
        """Process tiles through the model."""
        deblurred_tiles = []
        
        if self.use_gpu and PYTORCH_AVAILABLE:
            # GPU processing
            for tile in tiles:
                # Convert to tensor
                tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0)
                tensor = tensor.to(self.device)
                
                # Process
                with torch.no_grad():
                    output = model(tensor)
                
                # Convert back
                output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                deblurred_tiles.append(output)
        else:
            # CPU processing or simplified model
            for tile in tiles:
                # Process with fallback to simplified deblurring
                from pretrained_models import SimplifiedDeblurModel
                simplified = SimplifiedDeblurModel()
                output = simplified.deblur((tile * 255).astype(np.uint8)) / 255.0
                deblurred_tiles.append(output)
        
        return deblurred_tiles
    
    def _merge_tiles_with_blending(self, tiles: List[np.ndarray], 
                                  positions: List[Tuple[int, int]], 
                                  output_shape: Tuple[int, ...],
                                  tile_size: int, overlap: int = 32) -> np.ndarray:
        """Merge tiles with overlap blending."""
        h, w = output_shape[:2]
        channels = output_shape[2] if len(output_shape) == 3 else 1
        
        # Initialize output and weight map
        output = np.zeros((h, w, channels), dtype=np.float32)
        weights = np.zeros((h, w, 1), dtype=np.float32)
        
        # Create blending mask
        blend_mask = self._create_blend_mask(tile_size, overlap)
        
        # Merge tiles
        for tile, (y, x) in zip(tiles, positions):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            
            tile_h = y_end - y
            tile_w = x_end - x
            
            # Apply blending mask
            mask = blend_mask[:tile_h, :tile_w]
            if len(tile.shape) == 3:
                mask = np.expand_dims(mask, axis=-1)
            
            # Add to output
            output[y:y_end, x:x_end] += tile[:tile_h, :tile_w] * mask
            weights[y:y_end, x:x_end] += mask[:, :, :1] if len(mask.shape) == 3 else mask[:, :, np.newaxis]
        
        # Normalize by weights
        output = output / (weights + 1e-10)
        
        return output
    
    def _create_blend_mask(self, size: int, overlap: int) -> np.ndarray:
        """Create smooth blending mask for tile merging."""
        mask = np.ones((size, size), dtype=np.float32)
        
        # Create linear ramps for edges
        for i in range(overlap):
            weight = i / overlap
            mask[i, :] *= weight
            mask[-i-1, :] *= weight
            mask[:, i] *= weight
            mask[:, -i-1] *= weight
        
        return mask
    
    def _postprocess_prediction(self, prediction: np.ndarray, 
                               original: np.ndarray) -> np.ndarray:
        """Post-process model prediction."""
        # Ensure valid range
        prediction = np.clip(prediction, 0, 1)
        
        # Reduce artifacts with guided filter
        if hasattr(cv2, 'ximgproc'):
            prediction = cv2.ximgproc.guidedFilter(
                prediction, prediction, radius=5, eps=0.01
            )
        
        # Enhance sharpness
        blurred = cv2.GaussianBlur(prediction, (5, 5), 1)
        sharpened = prediction + 0.5 * (prediction - blurred)
        sharpened = np.clip(sharpened, 0, 1)
        
        # Blend with original for natural look
        alpha = 0.9
        result = alpha * sharpened + (1 - alpha) * (original / 255.0)
        
        # Convert back to uint8
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        return result
    
    # ============================================================================
    # Quality Assessment and Method Selection
    # ============================================================================
    
    def assess_deblur_quality(self, original: np.ndarray, 
                             deblurred: np.ndarray) -> Dict[str, float]:
        """Assess the quality of deblurred image."""
        metrics = {}
        
        # Sharpness metrics
        metrics['sharpness_improvement'] = self._measure_sharpness(deblurred) / (
            self._measure_sharpness(original) + 1e-10
        )
        
        # Noise level
        metrics['noise_level'] = self._estimate_noise_level(deblurred)
        
        # Artifact detection
        metrics['ringing_artifacts'] = self._detect_ringing_artifacts(deblurred)
        
        # Structure preservation
        metrics['structure_similarity'] = self._compute_ssim(original, deblurred)
        
        # Overall quality score
        metrics['quality_score'] = self._compute_quality_score(metrics)
        
        return metrics
    
    def _measure_sharpness(self, image: np.ndarray) -> float:
        """Measure image sharpness."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level in image."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Use median absolute deviation
        h, w = gray.shape
        crop = gray[h//4:3*h//4, w//4:3*w//4]
        sigma = np.median(np.abs(crop - np.median(crop))) / 0.6745
        
        return sigma
    
    def _detect_ringing_artifacts(self, image: np.ndarray) -> float:
        """Detect ringing artifacts in image."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Detect oscillations near edges
        edges = cv2.Canny(gray, 50, 150)
        dilated = cv2.dilate(edges, np.ones((5, 5)), iterations=2)
        
        # Measure variance in edge regions
        edge_regions = gray[dilated > 0]
        if len(edge_regions) > 0:
            variance = np.var(edge_regions)
            return min(variance / 1000.0, 1.0)  # Normalize
        return 0.0
    
    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute structural similarity index."""
        from skimage.metrics import structural_similarity
        
        if len(img1.shape) == 3:
            return structural_similarity(img1, img2, channel_axis=2)
        else:
            return structural_similarity(img1, img2)
    
    def _compute_quality_score(self, metrics: Dict[str, float]) -> float:
        """Compute overall quality score."""
        score = 0.0
        
        # Weight different metrics
        score += metrics['sharpness_improvement'] * 0.3
        score += (1.0 - metrics['noise_level'] / 50.0) * 0.2
        score += (1.0 - metrics['ringing_artifacts']) * 0.2
        score += metrics['structure_similarity'] * 0.3
        
        return np.clip(score, 0, 1)
    
    def select_best_method(self, image: np.ndarray) -> str:
        """Automatically select the best deblurring method."""
        # Analyze image characteristics
        blur_severity = self._estimate_blur_severity(image)
        motion_direction = self._detect_motion_direction(image)
        noise_level = self._estimate_noise_level(image)
        
        # Decision logic
        if blur_severity > 0.7:
            # Severe blur - use deep learning if available
            if self.use_gpu:
                return 'deep_learning'
            else:
                return 'wiener'
        elif motion_direction is not None:
            # Motion blur detected
            return 'cv2_advanced'  # Good for directional blur
        elif noise_level > 10:
            # High noise - Wiener is robust to noise
            return 'wiener'
        else:
            # Mild blur - CV2 is fast and effective
            return 'cv2_advanced'


class PerformanceOptimizer:
    """Optimize performance with parallel processing and GPU acceleration."""
    
    def __init__(self, use_gpu: bool = True, num_workers: int = None):
        self.use_gpu = use_gpu and PYTORCH_AVAILABLE and torch.cuda.is_available()
        self.num_workers = num_workers or mp.cpu_count()
        self.logger = logging.getLogger(__name__)
    
    def parallel_tile_processing(self, tiles: List[np.ndarray], 
                                process_func: callable) -> List[np.ndarray]:
        """Process tiles in parallel."""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(process_func, tiles))
        return results
    
    def optimize_memory_usage(self, image: np.ndarray) -> bool:
        """Check and optimize memory usage."""
        # Estimate memory requirement
        memory_required = image.nbytes * 10  # Rough estimate with overhead
        
        if self.use_gpu:
            # Check GPU memory
            available = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            if memory_required > available:
                self.logger.warning("Insufficient GPU memory, falling back to CPU")
                return False
        
        return True
    
    def batch_process_images(self, images: List[np.ndarray], 
                           process_func: callable) -> List[np.ndarray]:
        """Batch process multiple images efficiently."""
        if self.use_gpu:
            # Process in batches on GPU
            batch_size = 4
            results = []
            
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                batch_results = [process_func(img) for img in batch]
                results.extend(batch_results)
                
                # Clear GPU cache
                if PYTORCH_AVAILABLE:
                    torch.cuda.empty_cache()
            
            return results
        else:
            # Parallel CPU processing
            return self.parallel_tile_processing(images, process_func)
