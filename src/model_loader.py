"""
Model loading utilities for deep learning-based image deblurring.

This module handles loading and managing machine learning models
for image deblurring tasks.
"""

import logging
import os
import yaml
from typing import Optional, Dict, Any

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class ModelLoader:
    """Handles loading and managing machine learning models."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ModelLoader.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.model = None
        self.model_type = None
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            'model': {
                'type': 'pytorch',  # 'pytorch' or 'tensorflow'
                'path': None,
                'architecture': 'unet',
                'input_size': [256, 256],
                'channels': 3
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    # Merge with default config
                    default_config.update(user_config)
                    self.logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
                
        return default_config
    
    def load_model(self) -> Optional[Any]:
        """
        Load the machine learning model.
        
        Returns:
            Loaded model or None if failed
        """
        model_config = self.config.get('model', {})
        model_type = model_config.get('type', 'pytorch')
        model_path = model_config.get('path')
        
        if not model_path or not os.path.exists(model_path):
            self.logger.warning(f"Model path not found: {model_path}")
            return None
        
        try:
            if model_type == 'pytorch' and PYTORCH_AVAILABLE:
                return self._load_pytorch_model(model_path, model_config)
            elif model_type == 'tensorflow' and TENSORFLOW_AVAILABLE:
                return self._load_tensorflow_model(model_path, model_config)
            else:
                self.logger.error(f"Unsupported model type or framework not available: {model_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return None
    
    def _load_pytorch_model(self, model_path: str, config: Dict[str, Any]) -> Optional[Any]:
        """
        Load a PyTorch model.
        
        Args:
            model_path: Path to the model file
            config: Model configuration
            
        Returns:
            Loaded PyTorch model
        """
        if not PYTORCH_AVAILABLE:
            self.logger.error("PyTorch is not available")
            return None
        
        try:
            # Load model architecture
            architecture = config.get('architecture', 'unet')
            if architecture == 'unet':
                model = self._create_unet_model(config)
            else:
                self.logger.error(f"Unsupported architecture: {architecture}")
                return None
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            self.model = model
            self.model_type = 'pytorch'
            
            self.logger.info(f"Successfully loaded PyTorch model from {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading PyTorch model: {e}")
            return None
    
    def _load_tensorflow_model(self, model_path: str, config: Dict[str, Any]) -> Optional[Any]:
        """
        Load a TensorFlow model.
        
        Args:
            model_path: Path to the model file
            config: Model configuration
            
        Returns:
            Loaded TensorFlow model
        """
        if not TENSORFLOW_AVAILABLE:
            self.logger.error("TensorFlow is not available")
            return None
        
        try:
            model = tf.keras.models.load_model(model_path)
            self.model = model
            self.model_type = 'tensorflow'
            
            self.logger.info(f"Successfully loaded TensorFlow model from {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading TensorFlow model: {e}")
            return None
    
    def _create_unet_model(self, config: Dict[str, Any]) -> Optional['torch.nn.Module']:
        """
        Create a U-Net model architecture.
        
        Args:
            config: Model configuration
            
        Returns:
            U-Net model
        """
        if not PYTORCH_AVAILABLE:
            return None
        
        input_channels = config.get('channels', 3)
        output_channels = config.get('channels', 3)
        
        # Simple U-Net implementation
        class SimpleUNet(nn.Module):
            def __init__(self, in_channels, out_channels):
                super(SimpleUNet, self).__init__()
                
                # Encoder
                self.enc1 = self._conv_block(in_channels, 64)
                self.enc2 = self._conv_block(64, 128)
                self.enc3 = self._conv_block(128, 256)
                self.enc4 = self._conv_block(256, 512)
                
                # Bottleneck
                self.bottleneck = self._conv_block(512, 1024)
                
                # Decoder
                self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
                self.dec4 = self._conv_block(1024, 512)
                
                self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
                self.dec3 = self._conv_block(512, 256)
                
                self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
                self.dec2 = self._conv_block(256, 128)
                
                self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
                self.dec1 = self._conv_block(128, 64)
                
                # Output
                self.final = nn.Conv2d(64, out_channels, 1)
                
            def _conv_block(self, in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.ReLU(inplace=True)
                )
            
            def forward(self, x):
                # Encoder
                e1 = self.enc1(x)
                e2 = self.enc2(nn.MaxPool2d(2)(e1))
                e3 = self.enc3(nn.MaxPool2d(2)(e2))
                e4 = self.enc4(nn.MaxPool2d(2)(e3))
                
                # Bottleneck
                b = self.bottleneck(nn.MaxPool2d(2)(e4))
                
                # Decoder
                d4 = self.up4(b)
                d4 = torch.cat([d4, e4], dim=1)
                d4 = self.dec4(d4)
                
                d3 = self.up3(d4)
                d3 = torch.cat([d3, e3], dim=1)
                d3 = self.dec3(d3)
                
                d2 = self.up2(d3)
                d2 = torch.cat([d2, e2], dim=1)
                d2 = self.dec2(d2)
                
                d1 = self.up1(d2)
                d1 = torch.cat([d1, e1], dim=1)
                d1 = self.dec1(d1)
                
                return torch.sigmoid(self.final(d1))
        
        return SimpleUNet(input_channels, output_channels)
    
    def predict(self, input_image):
        """
        Run inference on an input image.
        
        Args:
            input_image: Preprocessed input image
            
        Returns:
            Model prediction
        """
        if self.model is None:
            self.logger.error("No model loaded")
            return None
        
        try:
            if self.model_type == 'pytorch':
                with torch.no_grad():
                    if isinstance(input_image, torch.Tensor):
                        tensor_input = input_image
                    else:
                        tensor_input = torch.from_numpy(input_image).float()
                    
                    # Add batch dimension if needed
                    if len(tensor_input.shape) == 3:
                        tensor_input = tensor_input.unsqueeze(0)
                    
                    output = self.model(tensor_input)
                    return output.cpu().numpy()
                    
            elif self.model_type == 'tensorflow':
                return self.model.predict(input_image)
                
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        return {
            'model_type': self.model_type,
            'model_loaded': self.model is not None,
            'config': self.config.get('model', {})
        }
