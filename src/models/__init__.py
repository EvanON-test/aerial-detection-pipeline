"""
Data models and interfaces for the aerial detection system.

This module defines the core data structures and interfaces used throughout
the system. It provides type-safe, validated data models and abstract
interfaces that enable modular, testable architecture.

Components:
- **data_models**: Core data structures with validation
- **interfaces**: Abstract base classes defining system contracts

Data Models:
- **Detection**: Raw detection results from inference models
- **TrackedObject**: Objects tracked across multiple frames
- **ThreatAssessment**: Security analysis and threat scoring
- **Alert**: Threat notifications and alerts
- **Configuration Models**: Type-safe configuration containers
- **System Models**: Status and monitoring information

Interfaces:
- **FrameCaptureInterface**: Camera and video input management
- **ModelInferenceInterface**: AI model loading and inference
- **TrackerInterface**: Multi-object tracking systems
- **ThreatAssessmentInterface**: Security analysis engines
- **VisualizationInterface**: Display and rendering systems
- **EventLoggerInterface**: Logging and audit systems
- **ConfigurationInterface**: Configuration management

Design Principles:
- Immutable data structures where appropriate
- Comprehensive validation with meaningful error messages
- Type hints for static analysis and IDE support
- Clear separation of concerns between components
- Extensible architecture for future enhancements

Usage:
    from src.models import Detection, TrackedObject, ThreatLevel
    from src.models.interfaces import FrameCaptureInterface
    
    # Create detection from model output
    detection = Detection(
        bbox=(100, 100, 200, 200),
        confidence=0.85,
        class_id=0,
        class_name="drone",
        timestamp=datetime.now(),
        frame_id=1
    )
"""

from .data_models import (
    Detection, TrackedObject, ThreatAssessment, Alert,
    CameraConfig, ModelConfig, TrackerConfig, AssessmentConfig,
    TrackState, ThreatLevel, SystemStatus
)

from .interfaces import (
    FrameCaptureInterface, ModelInferenceInterface, TrackerInterface,
    ThreatAssessmentInterface, VisualizationInterface, EventLoggerInterface,
    ConfigurationInterface
)

__all__ = [
    # Data models
    "Detection", "TrackedObject", "ThreatAssessment", "Alert",
    "CameraConfig", "ModelConfig", "TrackerConfig", "AssessmentConfig",
    "TrackState", "ThreatLevel", "SystemStatus",
    
    # Interfaces
    "FrameCaptureInterface", "ModelInferenceInterface", "TrackerInterface",
    "ThreatAssessmentInterface", "VisualizationInterface", "EventLoggerInterface",
    "ConfigurationInterface"
]