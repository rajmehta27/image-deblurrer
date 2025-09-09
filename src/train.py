"""
Training pipeline for deep learning deblurring models.

This module provides a complete training framework including data loading,
augmentation, loss functions, and training loops for both PyTorch and TensorFlow.
"""

import os
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from datetime import datetime
import time

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from models import ModelFactory
from dataset_generator import DatasetGenerator
from image_processor import ImageProcessor


# ============================================================================
# Loss Functions
# ============================================================================

if PYTORCH_AVAILABLE:
    
    class CombinedLoss(nn.Module):
        """Combined loss function for image deblurring."""
        
        def __init__(self, l1_weight=1.0, l2_weight=0.5, 
                     perceptual_weight=0.1, ssim_weight=0.1):
            super(CombinedLoss, self).__init__()
            self.l1_weight = l1_weight
            self.l2_weight = l2_weight
            self.perceptual_weight = perceptual_weight
            self.ssim_weight = ssim_weight
            
            self.l1_loss = nn.L1Loss()
            self.l2_loss = nn.MSELoss()
            
            # VGG for perceptual loss (if available)
            self.perceptual_loss = None
            try:
                import torchvision.models as models
                vgg = models.vgg16(pretrained=True).features[:16]
                vgg.eval()
                for param in vgg.parameters():
                    param.requires_grad = False
                self.perceptual_loss = vgg
            except:
                pass
        
        def forward(self, pred, target):
            loss = 0
            
            # L1 loss
            if self.l1_weight > 0:
                loss += self.l1_weight * self.l1_loss(pred, target)
            
            # L2 loss
            if self.l2_weight > 0:
                loss += self.l2_weight * self.l2_loss(pred, target)
            
            # Perceptual loss
            if self.perceptual_weight > 0 and self.perceptual_loss is not None:
                pred_features = self.perceptual_loss(pred)
                target_features = self.perceptual_loss(target)
                loss += self.perceptual_weight * self.l2_loss(pred_features, target_features)
            
            # SSIM loss
            if self.ssim_weight > 0:
                ssim_loss = 1 - self.ssim(pred, target)
                loss += self.ssim_weight * ssim_loss
            
            return loss
        
        def ssim(self, x, y):
            """Structural Similarity Index."""
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            mu_x = nn.functional.avg_pool2d(x, 3, 1, padding=1)
            mu_y = nn.functional.avg_pool2d(y, 3, 1, padding=1)
            
            sigma_x = nn.functional.avg_pool2d(x ** 2, 3, 1, padding=1) - mu_x ** 2
            sigma_y = nn.functional.avg_pool2d(y ** 2, 3, 1, padding=1) - mu_y ** 2
            sigma_xy = nn.functional.avg_pool2d(x * y, 3, 1, padding=1) - mu_x * mu_y
            
            SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
            SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
            
            SSIM = SSIM_n / SSIM_d
            return torch.clamp((1 + SSIM) / 2, 0, 1).mean()
    
    
    class DeblurDataset(Dataset):
        """PyTorch dataset for deblurring."""
        
        def __init__(self, blurred_paths: List[str], sharp_paths: List[str], 
                     transform=None):
            self.blurred_paths = blurred_paths
            self.sharp_paths = sharp_paths
            self.transform = transform
            self.processor = ImageProcessor()
        
        def __len__(self):
            return len(self.blurred_paths)
        
        def __getitem__(self, idx):
            # Load images
            blurred = self.processor.load_image(self.blurred_paths[idx])
            sharp = self.processor.load_image(self.sharp_paths[idx])
            
            # Convert to float32 and normalize to [0, 1]
            blurred = blurred.astype(np.float32) / 255.0
            sharp = sharp.astype(np.float32) / 255.0
            
            # Apply transforms if provided
            if self.transform:
                blurred = self.transform(blurred)
                sharp = self.transform(sharp)
            else:
                # Convert to tensor (CHW format)
                blurred = torch.from_numpy(blurred).permute(2, 0, 1)
                sharp = torch.from_numpy(sharp).permute(2, 0, 1)
            
            return blurred, sharp


if TENSORFLOW_AVAILABLE:
    
    def combined_loss_tf(y_true, y_pred):
        """Combined loss function for TensorFlow."""
        # L1 loss
        l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        
        # L2 loss
        l2_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # SSIM loss
        ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
        
        # Combined loss
        total_loss = l1_loss + 0.5 * l2_loss + 0.1 * ssim_loss
        
        return total_loss
    
    
    def create_tf_dataset(blurred_paths: List[str], sharp_paths: List[str], 
                         batch_size: int = 8):
        """Create TensorFlow dataset."""
        
        def load_and_preprocess(blurred_path, sharp_path):
            # Load images
            blurred = tf.io.read_file(blurred_path)
            blurred = tf.image.decode_image(blurred, channels=3)
            
            sharp = tf.io.read_file(sharp_path)
            sharp = tf.image.decode_image(sharp, channels=3)
            
            # Normalize to [0, 1]
            blurred = tf.cast(blurred, tf.float32) / 255.0
            sharp = tf.cast(sharp, tf.float32) / 255.0
            
            # Resize if needed
            blurred = tf.image.resize(blurred, [256, 256])
            sharp = tf.image.resize(sharp, [256, 256])
            
            return blurred, sharp
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((blurred_paths, sharp_paths))
        dataset = dataset.map(load_and_preprocess, 
                             num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


# ============================================================================
# Training Classes
# ============================================================================

class Trainer:
    """Base trainer class for model training."""
    
    def __init__(self, model, framework='pytorch', output_dir='models/trained'):
        self.model = model
        self.framework = framework
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        self.training_history = {
            'loss': [],
            'val_loss': [],
            'metrics': []
        }
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.framework == 'pytorch' and PYTORCH_AVAILABLE:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'loss': loss,
                'timestamp': timestamp
            }
            
            # Save regular checkpoint
            checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            
            # Save best model
            if is_best:
                best_path = self.output_dir / 'best_model.pth'
                torch.save(checkpoint, best_path)
                
        elif self.framework == 'tensorflow' and TENSORFLOW_AVAILABLE:
            # Save regular checkpoint
            checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}'
            self.model.save(checkpoint_path)
            
            # Save best model
            if is_best:
                best_path = self.output_dir / 'best_model'
                self.model.save(best_path)
        
        self.logger.info(f"Saved checkpoint at epoch {epoch}")
    
    def save_training_history(self):
        """Save training history to JSON."""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=4)
    
    def calculate_metrics(self, pred, target):
        """Calculate evaluation metrics."""
        # Calculate PSNR
        mse = np.mean((pred - target) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 100
        
        return {'psnr': psnr, 'mse': mse}


class PyTorchTrainer(Trainer):
    """Trainer for PyTorch models."""
    
    def __init__(self, model, device='cuda'):
        super().__init__(model, framework='pytorch')
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
    def train(self, train_loader, val_loader=None, epochs=100, 
              learning_rate=0.001, scheduler_params=None):
        """Train the model."""
        
        # Setup loss and optimizer
        criterion = CombinedLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        scheduler = None
        if scheduler_params:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **scheduler_params
            )
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            start_time = time.time()
            
            for batch_idx, (blurred, sharp) in enumerate(train_loader):
                blurred = blurred.to(self.device)
                sharp = sharp.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                output = self.model(blurred)
                loss = criterion(output, sharp)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                if batch_idx % 10 == 0:
                    self.logger.info(
                        f'Epoch [{epoch+1}/{epochs}] '
                        f'Batch [{batch_idx}/{len(train_loader)}] '
                        f'Loss: {loss.item():.4f}'
                    )
            
            avg_train_loss = train_loss / train_batches
            self.training_history['loss'].append(avg_train_loss)
            
            # Validation phase
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for blurred, sharp in val_loader:
                        blurred = blurred.to(self.device)
                        sharp = sharp.to(self.device)
                        
                        output = self.model(blurred)
                        loss = criterion(output, sharp)
                        
                        val_loss += loss.item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                self.training_history['val_loss'].append(avg_val_loss)
                
                # Update scheduler
                if scheduler:
                    scheduler.step(avg_val_loss)
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self.save_checkpoint(epoch, avg_val_loss, is_best=True)
                
                self.logger.info(
                    f'Epoch [{epoch+1}/{epochs}] completed in {time.time()-start_time:.2f}s\n'
                    f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}'
                )
            else:
                self.logger.info(
                    f'Epoch [{epoch+1}/{epochs}] completed in {time.time()-start_time:.2f}s\n'
                    f'Train Loss: {avg_train_loss:.4f}'
                )
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, avg_train_loss)
        
        # Save final model and history
        self.save_checkpoint(epochs - 1, avg_train_loss)
        self.save_training_history()
        
        self.logger.info("Training completed!")


class TensorFlowTrainer(Trainer):
    """Trainer for TensorFlow models."""
    
    def __init__(self, model):
        super().__init__(model, framework='tensorflow')
    
    def train(self, train_dataset, val_dataset=None, epochs=100, 
              learning_rate=0.001):
        """Train the model."""
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=combined_loss_tf,
            metrics=['mae', 'mse']
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                str(self.output_dir / 'best_model'),
                save_best_only=True,
                monitor='val_loss' if val_dataset else 'loss'
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if val_dataset else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if val_dataset else 'loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        # Train model
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save history
        self.training_history = history.history
        self.save_training_history()
        
        # Save final model
        self.model.save(str(self.output_dir / 'final_model'))
        
        self.logger.info("Training completed!")


# ============================================================================
# Training Pipeline
# ============================================================================

class TrainingPipeline:
    """Complete training pipeline for deblurring models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model_factory = ModelFactory()
        self.dataset_generator = DatasetGenerator()
    
    def prepare_data(self):
        """Prepare training and validation data."""
        self.logger.info("Preparing training data...")
        
        # Generate synthetic dataset if needed
        if self.config.get('generate_synthetic', True):
            self.dataset_generator.generate_dataset(
                num_samples=self.config.get('num_samples', 1000),
                output_dir=self.config.get('data_dir', 'data/synthetic')
            )
        
        # Get data paths
        data_dir = Path(self.config.get('data_dir', 'data/synthetic'))
        blurred_dir = data_dir / 'blurred'
        sharp_dir = data_dir / 'sharp'
        
        blurred_paths = sorted(list(blurred_dir.glob('*.jpg')))
        sharp_paths = sorted(list(sharp_dir.glob('*.jpg')))
        
        # Split into train and validation
        split_idx = int(len(blurred_paths) * 0.8)
        
        train_data = {
            'blurred': blurred_paths[:split_idx],
            'sharp': sharp_paths[:split_idx]
        }
        
        val_data = {
            'blurred': blurred_paths[split_idx:],
            'sharp': sharp_paths[split_idx:]
        }
        
        return train_data, val_data
    
    def create_model(self):
        """Create the model based on configuration."""
        model_name = self.config.get('model_name', 'enhanced_unet')
        framework = self.config.get('framework', 'pytorch')
        
        model = self.model_factory.create_model(
            model_name=model_name,
            framework=framework,
            **self.config.get('model_params', {})
        )
        
        return model
    
    def run(self):
        """Run the complete training pipeline."""
        self.logger.info("Starting training pipeline...")
        
        # Prepare data
        train_data, val_data = self.prepare_data()
        
        # Create model
        model = self.create_model()
        
        # Create data loaders based on framework
        framework = self.config.get('framework', 'pytorch')
        batch_size = self.config.get('batch_size', 8)
        
        if framework == 'pytorch' and PYTORCH_AVAILABLE:
            # Create PyTorch data loaders
            train_dataset = DeblurDataset(
                [str(p) for p in train_data['blurred']],
                [str(p) for p in train_data['sharp']]
            )
            val_dataset = DeblurDataset(
                [str(p) for p in val_data['blurred']],
                [str(p) for p in val_data['sharp']]
            )
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size,
                shuffle=True,
                num_workers=2
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2
            )
            
            # Create trainer and train
            trainer = PyTorchTrainer(model)
            trainer.train(
                train_loader,
                val_loader,
                epochs=self.config.get('epochs', 100),
                learning_rate=self.config.get('learning_rate', 0.001)
            )
            
        elif framework == 'tensorflow' and TENSORFLOW_AVAILABLE:
            # Create TensorFlow datasets
            train_dataset = create_tf_dataset(
                [str(p) for p in train_data['blurred']],
                [str(p) for p in train_data['sharp']],
                batch_size=batch_size
            )
            val_dataset = create_tf_dataset(
                [str(p) for p in val_data['blurred']],
                [str(p) for p in val_data['sharp']],
                batch_size=batch_size
            )
            
            # Create trainer and train
            trainer = TensorFlowTrainer(model)
            trainer.train(
                train_dataset,
                val_dataset,
                epochs=self.config.get('epochs', 100),
                learning_rate=self.config.get('learning_rate', 0.001)
            )
        else:
            raise ValueError(f"Framework {framework} not available")
        
        self.logger.info("Training pipeline completed!")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train deblurring model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--model', type=str, default='enhanced_unet',
                       choices=['enhanced_unet', 'residual_deblur', 'lightweight'])
    parser.add_argument('--framework', type=str, default='pytorch',
                       choices=['pytorch', 'tensorflow'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--data-dir', type=str, default='data/synthetic')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = {
        'model_name': args.model,
        'framework': args.framework,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'data_dir': args.data_dir,
        'generate_synthetic': True,
        'num_samples': 100  # For quick testing
    }
    
    # Run training
    pipeline = TrainingPipeline(config)
    pipeline.run()


if __name__ == '__main__':
    main()
