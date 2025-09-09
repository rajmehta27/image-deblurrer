"""
Deep learning models for image deblurring.

This module contains various neural network architectures optimized for
image deblurring tasks, including U-Net variants, ResNet-based models,
and attention mechanisms.
"""

import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np

# Try importing deep learning frameworks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


# ============================================================================
# PyTorch Models
# ============================================================================

if PYTORCH_AVAILABLE:
    
    class ConvBlock(nn.Module):
        """Convolutional block with batch normalization and activation."""
        
        def __init__(self, in_channels: int, out_channels: int, 
                     kernel_size: int = 3, stride: int = 1, 
                     padding: int = 1, use_bn: bool = True):
            super(ConvBlock, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, 
                                 kernel_size, stride, padding)
            self.use_bn = use_bn
            if use_bn:
                self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
        
        def forward(self, x):
            x = self.conv(x)
            if self.use_bn:
                x = self.bn(x)
            x = self.relu(x)
            return x
    
    
    class EnhancedUNet(nn.Module):
        """Enhanced U-Net architecture for image deblurring."""
        
        def __init__(self, in_channels: int = 3, out_channels: int = 3):
            super(EnhancedUNet, self).__init__()
            
            # Encoder
            self.enc1 = nn.Sequential(
                ConvBlock(in_channels, 64),
                ConvBlock(64, 64)
            )
            self.pool1 = nn.MaxPool2d(2)
            
            self.enc2 = nn.Sequential(
                ConvBlock(64, 128),
                ConvBlock(128, 128)
            )
            self.pool2 = nn.MaxPool2d(2)
            
            self.enc3 = nn.Sequential(
                ConvBlock(128, 256),
                ConvBlock(256, 256)
            )
            self.pool3 = nn.MaxPool2d(2)
            
            self.enc4 = nn.Sequential(
                ConvBlock(256, 512),
                ConvBlock(512, 512)
            )
            self.pool4 = nn.MaxPool2d(2)
            
            # Bottleneck with attention
            self.bottleneck = nn.Sequential(
                ConvBlock(512, 1024),
                ConvBlock(1024, 1024),
                ConvBlock(1024, 512)
            )
            
            # Decoder
            self.up4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
            self.dec4 = nn.Sequential(
                ConvBlock(1024, 512),
                ConvBlock(512, 256)
            )
            
            self.up3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
            self.dec3 = nn.Sequential(
                ConvBlock(512, 256),
                ConvBlock(256, 128)
            )
            
            self.up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
            self.dec2 = nn.Sequential(
                ConvBlock(256, 128),
                ConvBlock(128, 64)
            )
            
            self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
            self.dec1 = nn.Sequential(
                ConvBlock(128, 64),
                ConvBlock(64, 64)
            )
            
            # Output layer
            self.final = nn.Conv2d(64, out_channels, 1)
            
            # Residual connection
            self.residual = nn.Conv2d(in_channels, out_channels, 1)
        
        def forward(self, x):
            # Store input for residual connection
            residual = self.residual(x)
            
            # Encoder
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            e3 = self.enc3(self.pool2(e2))
            e4 = self.enc4(self.pool3(e3))
            
            # Bottleneck
            b = self.bottleneck(self.pool4(e4))
            
            # Decoder with skip connections
            d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
            d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            
            # Output with residual connection
            out = self.final(d1)
            out = torch.sigmoid(out + residual)
            
            return out
    
    
    class AttentionBlock(nn.Module):
        """Self-attention block for feature refinement."""
        
        def __init__(self, channels: int):
            super(AttentionBlock, self).__init__()
            self.channels = channels
            
            self.query = nn.Conv2d(channels, channels // 8, 1)
            self.key = nn.Conv2d(channels, channels // 8, 1)
            self.value = nn.Conv2d(channels, channels, 1)
            self.gamma = nn.Parameter(torch.zeros(1))
        
        def forward(self, x):
            batch_size, C, H, W = x.size()
            
            # Generate query, key, and value
            query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
            key = self.key(x).view(batch_size, -1, H * W)
            value = self.value(x).view(batch_size, C, H * W)
            
            # Calculate attention
            attention = F.softmax(torch.bmm(query, key), dim=-1)
            out = torch.bmm(value, attention.permute(0, 2, 1))
            out = out.view(batch_size, C, H, W)
            
            # Apply attention with learnable weight
            out = self.gamma * out + x
            
            return out
    
    
    class ResidualDeblurNet(nn.Module):
        """ResNet-based architecture for image deblurring."""
        
        def __init__(self, in_channels: int = 3, out_channels: int = 3, 
                     num_blocks: int = 16):
            super(ResidualDeblurNet, self).__init__()
            
            # Initial convolution
            self.conv_in = ConvBlock(in_channels, 64, kernel_size=9, padding=4)
            
            # Residual blocks
            self.res_blocks = nn.ModuleList([
                self._make_residual_block(64) for _ in range(num_blocks)
            ])
            
            # Attention blocks at intervals
            self.attention_blocks = nn.ModuleList([
                AttentionBlock(64) if i % 4 == 3 else nn.Identity()
                for i in range(num_blocks)
            ])
            
            # Output convolution
            self.conv_out = nn.Sequential(
                ConvBlock(64, 64, kernel_size=3, padding=1),
                nn.Conv2d(64, out_channels, kernel_size=9, padding=4)
            )
        
        def _make_residual_block(self, channels: int):
            """Create a residual block."""
            return nn.Sequential(
                ConvBlock(channels, channels, use_bn=True),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels)
            )
        
        def forward(self, x):
            # Initial feature extraction
            feat = self.conv_in(x)
            
            # Residual blocks with attention
            for res_block, att_block in zip(self.res_blocks, self.attention_blocks):
                residual = feat
                feat = res_block(feat)
                feat = feat + residual  # Residual connection
                feat = att_block(feat)  # Attention (if not Identity)
            
            # Output
            out = self.conv_out(feat)
            out = torch.tanh(out) * 0.5 + 0.5  # Normalize to [0, 1]
            
            return out
    
    
    class LightweightDeblurNet(nn.Module):
        """Lightweight network for fast deblurring."""
        
        def __init__(self, in_channels: int = 3, out_channels: int = 3):
            super(LightweightDeblurNet, self).__init__()
            
            self.encoder = nn.Sequential(
                # Downsample
                ConvBlock(in_channels, 32, stride=2),
                ConvBlock(32, 64, stride=2),
                ConvBlock(64, 128, stride=2),
            )
            
            self.processor = nn.Sequential(
                # Process features
                ConvBlock(128, 128),
                ConvBlock(128, 128),
                ConvBlock(128, 128),
            )
            
            self.decoder = nn.Sequential(
                # Upsample
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            # Encode
            encoded = self.encoder(x)
            
            # Process
            processed = self.processor(encoded)
            
            # Decode
            decoded = self.decoder(processed)
            
            return decoded


# ============================================================================
# TensorFlow/Keras Models
# ============================================================================

if TENSORFLOW_AVAILABLE:
    
    def conv_block(x, filters, kernel_size=3, strides=1, 
                   padding='same', use_bn=True, activation='relu'):
        """Convolutional block for TensorFlow/Keras."""
        x = layers.Conv2D(filters, kernel_size, strides=strides, 
                          padding=padding)(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        if activation:
            x = layers.Activation(activation)(x)
        return x
    
    
    def create_enhanced_unet_tf(input_shape=(256, 256, 3)):
        """Create Enhanced U-Net model in TensorFlow/Keras."""
        inputs = keras.Input(shape=input_shape)
        
        # Encoder
        conv1 = conv_block(inputs, 64)
        conv1 = conv_block(conv1, 64)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = conv_block(pool1, 128)
        conv2 = conv_block(conv2, 128)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = conv_block(pool2, 256)
        conv3 = conv_block(conv3, 256)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = conv_block(pool3, 512)
        conv4 = conv_block(conv4, 512)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
        
        # Bottleneck
        conv5 = conv_block(pool4, 1024)
        conv5 = conv_block(conv5, 1024)
        
        # Decoder
        up6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), 
                                     padding='same')(conv5)
        merge6 = layers.concatenate([conv4, up6], axis=3)
        conv6 = conv_block(merge6, 512)
        conv6 = conv_block(conv6, 512)
        
        up7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), 
                                     padding='same')(conv6)
        merge7 = layers.concatenate([conv3, up7], axis=3)
        conv7 = conv_block(merge7, 256)
        conv7 = conv_block(conv7, 256)
        
        up8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), 
                                     padding='same')(conv7)
        merge8 = layers.concatenate([conv2, up8], axis=3)
        conv8 = conv_block(merge8, 128)
        conv8 = conv_block(conv8, 128)
        
        up9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), 
                                     padding='same')(conv8)
        merge9 = layers.concatenate([conv1, up9], axis=3)
        conv9 = conv_block(merge9, 64)
        conv9 = conv_block(conv9, 64)
        
        # Output
        outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(conv9)
        
        # Add residual connection
        outputs = layers.Add()([inputs, outputs])
        outputs = layers.Activation('sigmoid')(outputs)
        
        model = keras.Model(inputs=inputs, outputs=outputs, 
                           name='enhanced_unet')
        return model
    
    
    def create_residual_deblur_tf(input_shape=(256, 256, 3), num_blocks=8):
        """Create Residual Deblur Network in TensorFlow/Keras."""
        inputs = keras.Input(shape=input_shape)
        
        # Initial convolution
        x = conv_block(inputs, 64, kernel_size=9, padding='same')
        
        # Residual blocks
        for i in range(num_blocks):
            residual = x
            x = conv_block(x, 64, kernel_size=3)
            x = conv_block(x, 64, kernel_size=3, activation=None)
            x = layers.Add()([x, residual])
            x = layers.Activation('relu')(x)
            
            # Add attention every 4 blocks
            if (i + 1) % 4 == 0:
                x = attention_block_tf(x, 64)
        
        # Output convolution
        x = conv_block(x, 64, kernel_size=3)
        outputs = layers.Conv2D(3, kernel_size=9, padding='same', 
                               activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, 
                           name='residual_deblur')
        return model
    
    
    def attention_block_tf(x, channels):
        """Self-attention block for TensorFlow/Keras."""
        # Simplified attention mechanism
        # Just apply channel attention for simplicity
        
        # Global average pooling
        gap = layers.GlobalAveragePooling2D()(x)
        gap = layers.Reshape((1, 1, channels))(gap)
        
        # Channel attention
        attention = layers.Dense(channels // 8, activation='relu')(gap)
        attention = layers.Dense(channels, activation='sigmoid')(attention)
        
        # Apply attention
        out = layers.Multiply()([x, attention])
        
        return out
    
    
    def create_lightweight_deblur_tf(input_shape=(256, 256, 3)):
        """Create Lightweight Deblur Network in TensorFlow/Keras."""
        inputs = keras.Input(shape=input_shape)
        
        # Encoder
        x = conv_block(inputs, 32, strides=2)
        x = conv_block(x, 64, strides=2)
        x = conv_block(x, 128, strides=2)
        
        # Processor
        x = conv_block(x, 128)
        x = conv_block(x, 128)
        x = conv_block(x, 128)
        
        # Decoder
        x = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(32, 4, strides=2, padding='same')(x)
        x = layers.Activation('relu')(x)
        outputs = layers.Conv2DTranspose(3, 4, strides=2, padding='same', 
                                        activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, 
                           name='lightweight_deblur')
        return model


# ============================================================================
# Model Factory
# ============================================================================

class ModelFactory:
    """Factory class for creating deep learning models."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_model(self, model_name: str, framework: str = 'pytorch',
                    input_shape: Optional[Tuple[int, ...]] = None,
                    **kwargs) -> Any:
        """
        Create a deep learning model.
        
        Args:
            model_name: Name of the model architecture
            framework: 'pytorch' or 'tensorflow'
            input_shape: Input shape for the model
            **kwargs: Additional model parameters
            
        Returns:
            Model instance
        """
        if framework == 'pytorch' and PYTORCH_AVAILABLE:
            return self._create_pytorch_model(model_name, **kwargs)
        elif framework == 'tensorflow' and TENSORFLOW_AVAILABLE:
            return self._create_tensorflow_model(model_name, input_shape, **kwargs)
        else:
            raise ValueError(f"Framework {framework} not available or not supported")
    
    def _create_pytorch_model(self, model_name: str, **kwargs) -> nn.Module:
        """Create PyTorch model."""
        models = {
            'enhanced_unet': EnhancedUNet,
            'residual_deblur': ResidualDeblurNet,
            'lightweight': LightweightDeblurNet,
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown PyTorch model: {model_name}")
        
        model_class = models[model_name]
        return model_class(**kwargs)
    
    def _create_tensorflow_model(self, model_name: str, 
                                input_shape: Optional[Tuple[int, ...]] = None,
                                **kwargs) -> keras.Model:
        """Create TensorFlow model."""
        if input_shape is None:
            input_shape = (256, 256, 3)
        
        models = {
            'enhanced_unet': create_enhanced_unet_tf,
            'residual_deblur': create_residual_deblur_tf,
            'lightweight': create_lightweight_deblur_tf,
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown TensorFlow model: {model_name}")
        
        model_fn = models[model_name]
        return model_fn(input_shape, **kwargs)
    
    def get_available_models(self, framework: Optional[str] = None) -> Dict[str, list]:
        """Get list of available models."""
        models = {
            'pytorch': ['enhanced_unet', 'residual_deblur', 'lightweight'] if PYTORCH_AVAILABLE else [],
            'tensorflow': ['enhanced_unet', 'residual_deblur', 'lightweight'] if TENSORFLOW_AVAILABLE else []
        }
        
        if framework:
            return models.get(framework, [])
        return models
