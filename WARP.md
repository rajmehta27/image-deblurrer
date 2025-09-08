# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Commands

### Development Setup
```bash
# Create and activate virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Core Usage Commands
```bash
# Basic image deblurring using OpenCV
python src/deblur.py input_image.jpg output_image.jpg

# Use specific deblurring method
python src/deblur.py input_image.jpg output_image.jpg --method deep_learning

# Use custom configuration and debug logging
python src/deblur.py input_image.jpg output_image.jpg --config config.yaml --log-level DEBUG
```

### Development Commands
```bash
# Run tests (when test suite exists)
pytest tests/

# Code formatting
black src/

# Linting
flake8 src/

# Type checking
mypy src/

# Run example script
python examples/basic_usage.py
```

## Architecture Overview

This is a modular image deblurring application with a plugin-like architecture supporting multiple deblurring techniques:

### Core Components

**ImageDeblurrer (`src/deblur.py`)**
- Main orchestrator class that coordinates deblurring operations
- Supports three deblurring methods: OpenCV (`cv2`), Wiener filtering (`wiener`), and deep learning (`deep_learning`)
- Currently, Wiener and deep learning methods fall back to OpenCV implementation

**ImageProcessor (`src/image_processor.py`)**
- Handles all image I/O operations and basic processing
- Supports multiple image formats (JPEG, PNG, BMP, TIFF, WebP)
- Provides preprocessing/postprocessing for ML models
- Includes utility functions for blur simulation and contrast enhancement

**ModelLoader (`src/model_loader.py`)**
- Manages machine learning model lifecycle
- Supports both PyTorch and TensorFlow models
- Includes a basic U-Net architecture implementation
- Handles model configuration via YAML files

**Utils (`src/utils.py`)**
- Common utilities: logging setup, path validation, batch processing
- Image quality metrics calculation (PSNR, SSIM, MSE)
- Performance measurement tools and system info reporting

### Configuration System

The application uses a comprehensive YAML-based configuration system (`config.yaml`) with sections for:
- Model settings (type, path, architecture, device)
- Image processing parameters
- Method-specific settings (OpenCV, Wiener, deep learning)
- Output formatting and logging configuration
- Performance and validation settings

### Deblurring Methods

1. **OpenCV Method**: Uses traditional image processing with sharpening kernels and optional bilateral filtering
2. **Wiener Filter**: Planned statistical restoration approach (currently incomplete)
3. **Deep Learning**: U-Net based neural network approach (requires trained models)

### Current Implementation Status

- OpenCV deblurring: ✅ Fully implemented
- Deep learning infrastructure: ✅ Framework complete, needs trained models
- Wiener filtering: ⚠️ Placeholder implementation
- Model training pipeline: ❌ Not yet implemented

### Key Design Patterns

- **Strategy Pattern**: Different deblurring methods are interchangeable
- **Factory Pattern**: ModelLoader creates appropriate model instances
- **Configuration-driven**: Behavior controlled via YAML configuration
- **Modular Architecture**: Clear separation of concerns between components

### Dependencies Architecture

The codebase has optional dependencies:
- Core: numpy, opencv-python, Pillow, matplotlib, scikit-image, scipy
- Deep Learning: tensorflow (optional), torch + torchvision (optional)
- Development: pytest, black, flake8, mypy
- Configuration: pyyaml, click, tqdm

### Error Handling

The application follows a consistent error handling pattern:
- Graceful degradation (fallback to OpenCV when other methods fail)
- Comprehensive logging at multiple levels
- Input validation for images and configuration files
- Exception handling with user-friendly error messages

## Working with This Codebase

### Adding New Deblurring Methods
1. Add method choice to `deblur.py` argument parser
2. Implement private method following `_deblur_cv2` pattern
3. Add method-specific configuration section to `config.yaml`
4. Update documentation and examples

### Model Integration
- Place trained models in project root or specify path in config
- Supported formats: `.pth` (PyTorch), `.h5`/`.pb` (TensorFlow)
- Update `config.yaml` model section with correct path and architecture

### Extending Image Processing
- Add new processing methods to `ImageProcessor` class
- Maintain consistency with existing numpy array handling
- Include appropriate error handling and logging

### Configuration Changes
- All configurable parameters should be added to `config.yaml`
- Provide sensible defaults in `ModelLoader._load_config()`
- Update README.md configuration section when adding new options
