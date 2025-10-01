"""
Aerial Object Detection & Tracking Pipeline.

This package provides a comprehensive system for real-time aerial object detection
and tracking, designed for security and surveillance applications. The system
integrates computer vision, machine learning, and threat assessment capabilities
to identify and track aerial objects such as drones and aircraft.

Main Components:
- **config**: Configuration management system
- **models**: Data models and interfaces
- **detection**: Object detection using AI models
- **tracking**: Multi-object tracking across frames
- **utils**: Utility functions and helpers

The system is designed to run on various hardware platforms including:
- Raspberry Pi 4 with Camera Module 2
- Desktop systems with USB/IP cameras
- Edge computing devices with GPU acceleration

Key Features:
- Real-time object detection using ONNX models
- Multi-object tracking with trajectory analysis
- Threat assessment and alert generation
- Configurable detection parameters
- Performance monitoring and optimization
- Modular architecture for easy extension

Usage:
    from src.config import ConfigurationManager
    from src.models import Detection, TrackedObject
    
    # Initialize system
    config = ConfigurationManager("config/config.yaml")
    # ... start detection pipeline

Version: 1.0.0
Author: Aerial Detection Team
License: MIT
"""

# Package version
__version__ = "1.0.0"

# Main package imports for convenience
from .config.config_manager import ConfigurationManager
from .models.data_models import (
    Detection, TrackedObject, ThreatAssessment, Alert,
    TrackState, ThreatLevel, SystemStatus
)

# Package metadata
__all__ = [
    "ConfigurationManager",
    "Detection", 
    "TrackedObject", 
    "ThreatAssessment", 
    "Alert",
    "TrackState", 
    "ThreatLevel", 
    "SystemStatus"
]