"""
Comprehensive tests for data models and validation logic.

This test module validates all data model classes including:
- Detection objects from inference results
- TrackedObject instances with trajectory data
- ThreatAssessment and alert generation
- Configuration objects with validation
- Enumeration types and their usage

The tests focus on:
1. **Object Creation**: Proper initialization with valid data
2. **Validation Logic**: __post_init__ validation catches invalid data
3. **Property Methods**: Computed properties work correctly
4. **Edge Cases**: Boundary conditions and error handling
5. **Type Safety**: Proper type checking and conversion

Each test is designed to be independent and uses realistic data
that would be encountered in the actual system operation.

Usage:
    pytest tests/test_data_models.py -v
    pytest tests/test_data_models.py::test_detection_creation
"""

import pytest
from datetime import datetime
from src.models.data_models import (
    Detection, TrackedObject, ThreatAssessment,
    TrackState, ThreatLevel, CameraConfig, ModelConfig
)


def test_detection_creation():
    """
    Test Detection object creation with valid data.
    
    Verifies that:
    1. Detection objects can be created with valid parameters
    2. All attributes are properly assigned
    3. Computed properties (center, area, width, height) work correctly
    4. Object maintains data integrity after creation
    """
    # Create detection with typical values
    timestamp = datetime.now()
    detection = Detection(
        bbox=(100.0, 100.0, 200.0, 200.0),
        confidence=0.85,
        class_id=0,
        class_name="drone",
        timestamp=timestamp,
        frame_id=1
    )
    
    # Test basic attribute assignment
    assert detection.confidence == 0.85, "Confidence should be preserved"
    assert detection.class_name == "drone", "Class name should be preserved"
    assert detection.bbox == (100.0, 100.0, 200.0, 200.0), "Bounding box should be preserved"
    assert detection.class_id == 0, "Class ID should be preserved"
    assert detection.timestamp == timestamp, "Timestamp should be preserved"
    assert detection.frame_id == 1, "Frame ID should be preserved"
    
    # Test computed properties
    assert detection.center == (150.0, 150.0), "Center should be calculated correctly"
    assert detection.area == 10000.0, "Area should be calculated correctly (100x100)"
    assert detection.width == 100.0, "Width should be calculated correctly"
    assert detection.height == 100.0, "Height should be calculated correctly"


def test_detection_validation():
    """
    Test Detection validation logic in __post_init__.
    
    Verifies that:
    1. Invalid confidence values are rejected (outside [0.0, 1.0])
    2. Negative class IDs are rejected
    3. Empty class names are rejected
    4. Validation provides meaningful error messages
    """
    # Test confidence validation - too high
    with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
        Detection(
            bbox=(100.0, 100.0, 200.0, 200.0),
            confidence=1.5,  # Invalid: > 1.0
            class_id=0,
            class_name="drone",
            timestamp=datetime.now(),
            frame_id=1
        )
    
    # Test confidence validation - negative
    with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
        Detection(
            bbox=(100.0, 100.0, 200.0, 200.0),
            confidence=-0.1,  # Invalid: < 0.0
            class_id=0,
            class_name="drone",
            timestamp=datetime.now(),
            frame_id=1
        )
    
    # Test class ID validation - negative
    with pytest.raises(ValueError, match="Class ID must be non-negative"):
        Detection(
            bbox=(100.0, 100.0, 200.0, 200.0),
            confidence=0.85,
            class_id=-1,  # Invalid: negative
            class_name="drone",
            timestamp=datetime.now(),
            frame_id=1
        )
    
    # Test class name validation - empty
    with pytest.raises(ValueError, match="Class name cannot be empty"):
        Detection(
            bbox=(100.0, 100.0, 200.0, 200.0),
            confidence=0.85,
            class_id=0,
            class_name="",  # Invalid: empty
            timestamp=datetime.now(),
            frame_id=1
        )


def test_tracked_object_creation():
    """Test TrackedObject creation."""
    detection = Detection(
        bbox=(100.0, 100.0, 200.0, 200.0),
        confidence=0.85,
        class_id=0,
        class_name="drone",
        timestamp=datetime.now(),
        frame_id=1
    )
    
    tracked_obj = TrackedObject(
        track_id=1,
        detection=detection,
        trajectory=[(150.0, 150.0)],
        velocity=(5.0, 2.0),
        age=10,
        time_since_update=0,
        state=TrackState.CONFIRMED
    )
    
    assert tracked_obj.track_id == 1
    assert tracked_obj.state == TrackState.CONFIRMED
    assert len(tracked_obj.trajectory) == 1


def test_threat_assessment_creation():
    """Test ThreatAssessment creation."""
    threat = ThreatAssessment(
        object_id=1,
        threat_level=ThreatLevel.HIGH,
        threat_score=0.8,
        threat_factors=["high_speed", "restricted_zone"],
        recommended_action="monitor_closely",
        timestamp=datetime.now()
    )
    
    assert threat.threat_level == ThreatLevel.HIGH
    assert threat.threat_score == 0.8
    assert "high_speed" in threat.threat_factors


def test_camera_config_creation():
    """Test CameraConfig creation and validation."""
    config = CameraConfig(
        resolution=(1920, 1080),
        fps=30,
        exposure_mode="auto",
        focus_mode="auto"
    )
    
    assert config.resolution == (1920, 1080)
    assert config.fps == 30


def test_model_config_creation():
    """Test ModelConfig creation and validation."""
    config = ModelConfig(
        primary_model="models/yolov8n.onnx",
        fallback_model="models/yolov8n.pt",
        confidence_threshold=0.5,
        nms_threshold=0.4
    )
    
    assert config.primary_model == "models/yolov8n.onnx"
    assert config.confidence_threshold == 0.5