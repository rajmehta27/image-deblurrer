"""
Dataset generator for creating synthetic training data.

This module generates pairs of sharp and blurred images for training
deblurring models, with various types of blur and degradation.
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
from tqdm import tqdm
import random
from image_processor import ImageProcessor


class DatasetGenerator:
    """Generates synthetic dataset for deblurring model training."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.processor = ImageProcessor()
    
    def generate_dataset(self, num_samples: int = 1000, 
                        output_dir: str = 'data/synthetic',
                        image_size: Tuple[int, int] = (256, 256)):
        """
        Generate synthetic dataset with pairs of sharp and blurred images.
        
        Args:
            num_samples: Number of image pairs to generate
            output_dir: Directory to save the dataset
            image_size: Size of generated images
        """
        output_path = Path(output_dir)
        sharp_dir = output_path / 'sharp'
        blurred_dir = output_path / 'blurred'
        
        # Create directories
        sharp_dir.mkdir(parents=True, exist_ok=True)
        blurred_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Generating {num_samples} synthetic image pairs...")
        
        for i in tqdm(range(num_samples), desc="Generating dataset"):
            # Generate sharp image
            sharp_image = self._generate_sharp_image(image_size)
            
            # Generate blurred version
            blur_type = random.choice(['gaussian', 'motion', 'defocus', 'mixed'])
            blurred_image = self._apply_blur(sharp_image, blur_type)
            
            # Add noise
            if random.random() > 0.5:
                blurred_image = self._add_noise(blurred_image)
            
            # Save images
            sharp_path = sharp_dir / f'image_{i:05d}.jpg'
            blurred_path = blurred_dir / f'image_{i:05d}.jpg'
            
            cv2.imwrite(str(sharp_path), cv2.cvtColor(sharp_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(blurred_path), cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR))
        
        self.logger.info(f"Dataset generated successfully in {output_dir}")
    
    def _generate_sharp_image(self, size: Tuple[int, int]) -> np.ndarray:
        """
        Generate a sharp synthetic image with various patterns.
        
        Args:
            size: Image size (width, height)
            
        Returns:
            Sharp image
        """
        height, width = size
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Randomly choose background color
        bg_color = np.random.randint(200, 256, 3)
        image[:] = bg_color
        
        # Add various elements
        num_elements = random.randint(5, 15)
        
        for _ in range(num_elements):
            element_type = random.choice(['text', 'line', 'circle', 'rectangle', 
                                         'polygon', 'gradient'])
            
            if element_type == 'text':
                self._add_text(image)
            elif element_type == 'line':
                self._add_lines(image)
            elif element_type == 'circle':
                self._add_circles(image)
            elif element_type == 'rectangle':
                self._add_rectangles(image)
            elif element_type == 'polygon':
                self._add_polygon(image)
            elif element_type == 'gradient':
                self._add_gradient(image)
        
        # Add texture
        if random.random() > 0.5:
            image = self._add_texture(image)
        
        return image
    
    def _add_text(self, image: np.ndarray):
        """Add random text to image."""
        texts = ['SAMPLE', 'TEST', 'SHARP', 'CLEAR', 'TEXT', 'DEMO', 
                 'QUALITY', 'FOCUS', 'DETAIL', 'EDGE']
        text = random.choice(texts)
        
        h, w = image.shape[:2]
        x = random.randint(10, w - 100)
        y = random.randint(30, h - 30)
        
        font = random.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX])
        scale = random.uniform(0.5, 2.0)
        thickness = random.randint(1, 3)
        color = tuple(np.random.randint(0, 150, 3).tolist())
        
        cv2.putText(image, text, (x, y), font, scale, color, thickness)
    
    def _add_lines(self, image: np.ndarray):
        """Add random lines to image."""
        h, w = image.shape[:2]
        num_lines = random.randint(1, 5)
        
        for _ in range(num_lines):
            pt1 = (random.randint(0, w), random.randint(0, h))
            pt2 = (random.randint(0, w), random.randint(0, h))
            color = tuple(np.random.randint(0, 200, 3).tolist())
            thickness = random.randint(1, 3)
            
            cv2.line(image, pt1, pt2, color, thickness)
    
    def _add_circles(self, image: np.ndarray):
        """Add random circles to image."""
        h, w = image.shape[:2]
        num_circles = random.randint(1, 4)
        
        for _ in range(num_circles):
            center = (random.randint(20, w-20), random.randint(20, h-20))
            radius = random.randint(10, min(50, w//4, h//4))
            color = tuple(np.random.randint(0, 200, 3).tolist())
            thickness = random.choice([-1, 1, 2, 3])  # -1 for filled
            
            cv2.circle(image, center, radius, color, thickness)
    
    def _add_rectangles(self, image: np.ndarray):
        """Add random rectangles to image."""
        h, w = image.shape[:2]
        num_rects = random.randint(1, 4)
        
        for _ in range(num_rects):
            pt1 = (random.randint(0, w-30), random.randint(0, h-30))
            pt2 = (random.randint(pt1[0]+20, w), random.randint(pt1[1]+20, h))
            color = tuple(np.random.randint(0, 200, 3).tolist())
            thickness = random.choice([-1, 1, 2, 3])
            
            cv2.rectangle(image, pt1, pt2, color, thickness)
    
    def _add_polygon(self, image: np.ndarray):
        """Add random polygon to image."""
        h, w = image.shape[:2]
        num_points = random.randint(3, 8)
        
        points = []
        for _ in range(num_points):
            x = random.randint(10, w-10)
            y = random.randint(10, h-10)
            points.append([x, y])
        
        points = np.array(points, np.int32)
        color = tuple(np.random.randint(0, 200, 3).tolist())
        
        if random.random() > 0.5:
            cv2.fillPoly(image, [points], color)
        else:
            cv2.polylines(image, [points], True, color, 2)
    
    def _add_gradient(self, image: np.ndarray):
        """Add gradient overlay to image."""
        h, w = image.shape[:2]
        
        # Create gradient
        gradient = np.zeros((h, w, 3), dtype=np.uint8)
        
        if random.random() > 0.5:
            # Horizontal gradient
            for i in range(w):
                gradient[:, i] = int(255 * i / w)
        else:
            # Vertical gradient
            for i in range(h):
                gradient[i, :] = int(255 * i / h)
        
        # Random color channel
        channel = random.randint(0, 2)
        gradient[:, :, (channel+1)%3] = 0
        gradient[:, :, (channel+2)%3] = 0
        
        # Blend with image
        alpha = random.uniform(0.1, 0.3)
        image[:] = cv2.addWeighted(image, 1-alpha, gradient, alpha, 0)
    
    def _add_texture(self, image: np.ndarray) -> np.ndarray:
        """Add texture pattern to image."""
        h, w = image.shape[:2]
        
        # Create texture
        texture_type = random.choice(['noise', 'grid', 'dots'])
        
        if texture_type == 'noise':
            noise = np.random.randint(0, 30, (h, w, 3), dtype=np.uint8)
            image = cv2.add(image, noise)
            
        elif texture_type == 'grid':
            step = random.randint(10, 30)
            for i in range(0, w, step):
                cv2.line(image, (i, 0), (i, h), (200, 200, 200), 1)
            for i in range(0, h, step):
                cv2.line(image, (0, i), (w, i), (200, 200, 200), 1)
                
        elif texture_type == 'dots':
            step = random.randint(15, 30)
            for i in range(0, h, step):
                for j in range(0, w, step):
                    cv2.circle(image, (j, i), 1, (180, 180, 180), -1)
        
        return image
    
    def _apply_blur(self, image: np.ndarray, blur_type: str) -> np.ndarray:
        """
        Apply blur to image.
        
        Args:
            image: Input image
            blur_type: Type of blur to apply
            
        Returns:
            Blurred image
        """
        if blur_type == 'gaussian':
            kernel_size = random.choice([5, 7, 9, 11, 13, 15])
            sigma = random.uniform(1.0, 5.0)
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
            
        elif blur_type == 'motion':
            size = random.randint(10, 25)
            angle = random.uniform(0, 180)
            blurred = self.processor.apply_motion_blur(image, size, angle)
            
        elif blur_type == 'defocus':
            # Simulate defocus blur with disk kernel
            radius = random.randint(3, 10)
            kernel = self._create_disk_kernel(radius)
            blurred = cv2.filter2D(image, -1, kernel)
            
        elif blur_type == 'mixed':
            # Apply multiple blur types
            blurred = image.copy()
            
            # First apply mild Gaussian
            if random.random() > 0.5:
                blurred = cv2.GaussianBlur(blurred, (5, 5), 1.5)
            
            # Then apply either motion or defocus
            if random.random() > 0.5:
                size = random.randint(5, 15)
                angle = random.uniform(0, 180)
                blurred = self.processor.apply_motion_blur(blurred, size, angle)
            else:
                radius = random.randint(2, 6)
                kernel = self._create_disk_kernel(radius)
                blurred = cv2.filter2D(blurred, -1, kernel)
        else:
            blurred = image
        
        return blurred
    
    def _create_disk_kernel(self, radius: int) -> np.ndarray:
        """Create disk-shaped kernel for defocus blur."""
        size = 2 * radius + 1
        kernel = np.zeros((size, size))
        center = radius
        
        for i in range(size):
            for j in range(size):
                if (i - center) ** 2 + (j - center) ** 2 <= radius ** 2:
                    kernel[i, j] = 1
        
        kernel = kernel / np.sum(kernel)
        return kernel
    
    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add noise to image."""
        noise_type = random.choice(['gaussian', 'salt_pepper', 'poisson'])
        
        if noise_type == 'gaussian':
            # Gaussian noise
            mean = 0
            std = random.uniform(5, 15)
            noise = np.random.normal(mean, std, image.shape).astype(np.float32)
            noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
        elif noise_type == 'salt_pepper':
            # Salt and pepper noise
            noisy = image.copy()
            prob = random.uniform(0.01, 0.05)
            
            # Salt
            mask = np.random.random(image.shape[:2]) < prob/2
            noisy[mask] = 255
            
            # Pepper
            mask = np.random.random(image.shape[:2]) < prob/2
            noisy[mask] = 0
            
        elif noise_type == 'poisson':
            # Poisson noise
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        else:
            noisy = image
        
        return noisy
    
    def generate_from_existing_images(self, input_dir: str, output_dir: str,
                                     num_augmentations: int = 5):
        """
        Generate dataset from existing sharp images.
        
        Args:
            input_dir: Directory containing sharp images
            output_dir: Directory to save the dataset
            num_augmentations: Number of augmented versions per image
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        sharp_dir = output_path / 'sharp'
        blurred_dir = output_path / 'blurred'
        
        sharp_dir.mkdir(parents=True, exist_ok=True)
        blurred_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(ext))
        
        self.logger.info(f"Found {len(image_files)} images in {input_dir}")
        
        counter = 0
        for img_path in tqdm(image_files, desc="Processing images"):
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize if too large
            h, w = image.shape[:2]
            if h > 512 or w > 512:
                scale = min(512/h, 512/w)
                new_h, new_w = int(h*scale), int(w*scale)
                image = cv2.resize(image, (new_w, new_h))
            
            # Generate augmented versions
            for aug_idx in range(num_augmentations):
                # Apply random augmentations
                augmented = self._augment_image(image)
                
                # Apply blur
                blur_type = random.choice(['gaussian', 'motion', 'defocus', 'mixed'])
                blurred = self._apply_blur(augmented, blur_type)
                
                # Add noise
                if random.random() > 0.5:
                    blurred = self._add_noise(blurred)
                
                # Save images
                sharp_path = sharp_dir / f'image_{counter:05d}.jpg'
                blurred_path = blurred_dir / f'image_{counter:05d}.jpg'
                
                cv2.imwrite(str(sharp_path), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(blurred_path), cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR))
                
                counter += 1
        
        self.logger.info(f"Generated {counter} image pairs in {output_dir}")
    
    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentations to image."""
        augmented = image.copy()
        
        # Random flip
        if random.random() > 0.5:
            augmented = cv2.flip(augmented, 1)  # Horizontal flip
        
        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            h, w = augmented.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented = cv2.warpAffine(augmented, M, (w, h))
        
        # Random brightness/contrast
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)  # Contrast
            beta = random.randint(-20, 20)    # Brightness
            augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)
        
        # Random crop
        if random.random() > 0.3:
            h, w = augmented.shape[:2]
            crop_size = random.uniform(0.8, 0.95)
            new_h, new_w = int(h * crop_size), int(w * crop_size)
            
            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)
            
            augmented = augmented[top:top+new_h, left:left+new_w]
            augmented = cv2.resize(augmented, (w, h))
        
        return augmented


def main():
    """Main function for dataset generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic dataset')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples to generate')
    parser.add_argument('--output-dir', type=str, default='data/synthetic',
                       help='Output directory for dataset')
    parser.add_argument('--image-size', type=int, nargs=2, default=[256, 256],
                       help='Size of generated images')
    parser.add_argument('--from-existing', type=str,
                       help='Generate from existing images directory')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create generator
    generator = DatasetGenerator()
    
    if args.from_existing:
        generator.generate_from_existing_images(
            args.from_existing,
            args.output_dir,
            num_augmentations=5
        )
    else:
        generator.generate_dataset(
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            image_size=tuple(args.image_size)
        )


if __name__ == '__main__':
    main()
