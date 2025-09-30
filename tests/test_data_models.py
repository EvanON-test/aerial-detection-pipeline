"""
Tests for data models and basic functionality.
"""

import pytest
from datetime import datetime
from src.models.data_models import (
    Detection, TrackedObject, ThreatAssessment,
    TrackState, ThreatLevel, CameraConfig, ModelConfig
)


def test_detection_creation():
    """Test Detection object creation and validation."""
    detection = Detection(
        bbox=(100.0, 100.0, 200.0, 200.0),
        confidence=0.85,
        class_id=0,
        class_name="drone",
        timestamp=datetime.now(),
        frame_id=1
    )
    
    assert detection.confidence == 0.85
    assert detection.class_name == "drone"
    assert detection.bbox == (100.0, 100.0, 200.0, 200.0)


def test_detection_validation():
    """Test Detection validation."""
    with pytest.raises(ValueError):
        Detection(
            bbox=(100.0, 100.0, 200.0, 200.0),
            confidence=1.5,  # Invalid confidence > 1.0
            class_id=0,
            class_name="drone",
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