# Image Deblurrer

A Python-based image deblurring tool that uses various techniques including traditional image processing methods and deep learning approaches to restore clarity to blurred images.

## Features

- **Multiple Deblurring Methods**:
  - OpenCV-based image sharpening
  - Wiener filtering (planned)
  - Deep learning models (U-Net architecture)

- **Flexible Input/Output**:
  - Support for common image formats (JPEG, PNG, BMP, TIFF, WebP)
  - Command-line interface for batch processing
  - Configurable processing parameters

- **Extensible Architecture**:
  - Modular design for easy addition of new algorithms
  - Support for both PyTorch and TensorFlow models
  - Configuration-based model loading

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd image-deblurrer
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Basic Usage

Deblur a single image using the default OpenCV method:

```bash
python src/deblur.py input_image.jpg output_image.jpg
```

### Advanced Usage

Specify a different deblurring method:

```bash
python src/deblur.py input_image.jpg output_image.jpg --method deep_learning
```

Use a custom configuration file:

```bash
python src/deblur.py input_image.jpg output_image.jpg --config config.yaml --log-level DEBUG
```

## Available Methods

### 1. OpenCV Method (`--method cv2`)
- **Description**: Uses traditional image processing techniques with sharpening kernels
- **Pros**: Fast, no model required, works well for mild blur
- **Cons**: Limited effectiveness on severe blur or complex degradation
- **Best for**: Quick processing, mild blur correction

### 2. Wiener Filtering (`--method wiener`)
- **Description**: Statistical approach to image restoration (implementation in progress)
- **Pros**: Good theoretical foundation, handles noise well
- **Cons**: Requires knowledge of blur kernel
- **Best for**: Known blur patterns, images with noise

### 3. Deep Learning (`--method deep_learning`)
- **Description**: Uses trained neural networks (U-Net architecture) for deblurring
- **Pros**: Can handle complex blur patterns, learns from data
- **Cons**: Requires trained model, computationally intensive
- **Best for**: Complex blur, motion blur, unknown degradation

## Configuration

Create a `config.yaml` file to customize the deblurring process:

```yaml
model:
  type: pytorch              # 'pytorch' or 'tensorflow'
  path: models/deblur_model.pth
  architecture: unet
  input_size: [256, 256]
  channels: 3

processing:
  resize_input: true
  target_size: [512, 512]
  enhance_contrast: true
  contrast_alpha: 1.2
  contrast_beta: 10
```

## Project Structure

```
image-deblurrer/
├── src/                    # Source code
│   ├── __init__.py        # Package initialization
│   ├── deblur.py          # Main application script
│   ├── image_processor.py # Image processing utilities
│   ├── model_loader.py    # ML model management
│   └── utils.py           # Helper functions
├── tests/                 # Unit tests
├── data/                  # Sample data and test images
├── models/                # Trained models
├── examples/              # Usage examples
├── docs/                  # Additional documentation
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Development Setup

1. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run tests**:
   ```bash
   pytest tests/
   ```

3. **Code formatting**:
   ```bash
   black src/
   flake8 src/
   ```

4. **Type checking**:
   ```bash
   mypy src/
   ```

## Usage Examples

### Python API

```python
from src import ImageDeblurrer, setup_logging

# Setup logging
setup_logging("INFO")

# Initialize deblurrer
deblurrer = ImageDeblurrer("config.yaml")

# Process single image
success = deblurrer.deblur_image("input.jpg", "output.jpg", method="cv2")

if success:
    print("Image deblurred successfully!")
```

### Batch Processing

```python
from src.utils import create_batch_processing_list
from src import ImageDeblurrer

# Create batch processing list
file_pairs = create_batch_processing_list("input_folder/", "output_folder/")

# Process all images
deblurrer = ImageDeblurrer()
for input_path, output_path in file_pairs:
    deblurrer.deblur_image(input_path, output_path)
```

## Model Training

To train your own deep learning models:

1. **Prepare training data**: Create pairs of blurred and sharp images
2. **Use the provided training scripts** (coming soon)
3. **Save trained models** in the `models/` directory
4. **Update configuration** to point to your model

## Performance Tips

- **GPU Acceleration**: Install CUDA-enabled PyTorch/TensorFlow for faster processing
- **Image Resizing**: Consider resizing large images for faster processing
- **Batch Processing**: Process multiple images together for better efficiency
- **Model Selection**: Choose appropriate method based on your specific use case

## Troubleshooting

### Common Issues

1. **Module Import Errors**:
   - Ensure you're in the project root directory
   - Check that all dependencies are installed

2. **CUDA Out of Memory**:
   - Reduce batch size or image resolution
   - Use CPU instead of GPU for processing

3. **Poor Deblurring Results**:
   - Try different methods
   - Adjust configuration parameters
   - Ensure input image quality is sufficient

### Debug Mode

Enable debug logging for detailed information:

```bash
python src/deblur.py input.jpg output.jpg --log-level DEBUG
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV community for image processing tools
- PyTorch and TensorFlow teams for deep learning frameworks
- Research papers on image deblurring and restoration

## Roadmap

- [ ] Complete Wiener filter implementation
- [ ] Add more deep learning architectures
- [ ] Implement model training pipeline
- [ ] Add GUI interface
- [ ] Support for video deblurring
- [ ] Real-time processing capabilities
- [ ] Mobile app integration

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in the `docs/` folder
- Review the examples in the `examples/` folder

---

**Note**: This is a research and development project. Results may vary depending on the type and severity of blur in your images.
