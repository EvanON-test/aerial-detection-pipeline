# Aerial Detection Pipeline

A real-time aerial object detection and tracking system optimized for Jetson Nano 2GB hardware using state-of-the-art computer vision models.

## Features

- Real-time aerial object detection using YOLOv8
- Multi-object tracking with DeepSORT
- Threat assessment and classification
- Optimized for edge deployment on Jetson Nano
- Configurable detection parameters
- Event logging and visualization

## Project Structure

```
├── src/
│   ├── detection/          # Detection module
│   ├── tracking/           # Multi-object tracking
│   ├── config/             # Configuration management
│   ├── models/             # Data models and interfaces
│   └── utils/              # Utility functions
├── models/                 # Pre-trained model files
├── config/                 # Configuration files
└── logs/                   # Log files
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure the system:
```bash
cp config/default_config.yaml config/config.yaml
# Edit config/config.yaml as needed
```

## Configuration

The system uses YAML configuration files for easy parameter adjustment. Key configuration sections:

- **camera**: Camera settings and capture parameters
- **model**: Model selection and inference parameters
- **tracker**: Multi-object tracking configuration
- **assessment**: Threat assessment rules and thresholds
- **logging**: Event logging configuration
- **system**: System performance and optimization settings

## Requirements

- Python 3.8+
- OpenCV 4.5+
- ONNX Runtime
- PyTorch (for fallback models)
- Jetson Nano 2GB (recommended) or compatible hardware

## License

This project is licensed under the MIT License.