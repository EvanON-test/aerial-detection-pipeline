"""
Core data models for the aerial object detection and tracking system.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Tuple, Optional, Any, Dict


class TrackState(Enum):
    """Tracking state enumeration."""
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    DELETED = "deleted"


class ThreatLevel(Enum):
    """Threat level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Detection:
    """Detection object representing a detected aerial object."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    timestamp: datetime
    frame_id: int
    
    def __post_init__(self):
        """Validate detection data."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.class_id < 0:
            raise ValueError("Class ID must be non-negative")


@dataclass
class TrackedObject:
    """Tracked object with trajectory and state information."""
    track_id: int
    detection: Detection
    trajectory: List[Tuple[float, float]]  # Historical positions
    velocity: Tuple[float, float]  # Speed and direction
    age: int  # Frames since first detection
    time_since_update: int  # Frames since last detection
    state: TrackState
    
    def __post_init__(self):
        """Validate tracked object data."""
        if self.track_id < 0:
            raise ValueError("Track ID must be non-negative")
        if self.age < 0:
            raise ValueError("Age must be non-negative")
        if self.time_since_update < 0:
            raise ValueError("Time since update must be non-negative")


@dataclass
class ThreatAssessment:
    """Threat assessment for a tracked object."""
    object_id: int
    threat_level: ThreatLevel
    threat_score: float  # 0.0 to 1.0
    threat_factors: List[str]  # Reasons for threat level
    recommended_action: str
    timestamp: datetime
    
    def __post_init__(self):
        """Validate threat assessment data."""
        if not (0.0 <= self.threat_score <= 1.0):
            raise ValueError("Threat score must be between 0.0 and 1.0")
        if self.object_id < 0:
            raise ValueError("Object ID must be non-negative")


@dataclass
class CameraConfig:
    """Camera configuration parameters."""
    resolution: Tuple[int, int]
    fps: int
    exposure_mode: str = "auto"
    focus_mode: str = "auto"
    brightness: int = 50
    contrast: int = 0
    
    def __post_init__(self):
        """Validate camera configuration."""
        if self.fps <= 0:
            raise ValueError("FPS must be positive")
        if any(dim <= 0 for dim in self.resolution):
            raise ValueError("Resolution dimensions must be positive")


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    primary_model: str
    fallback_model: str
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_detections: int = 100
    input_size: Tuple[int, int] = (640, 640)
    
    def __post_init__(self):
        """Validate model configuration."""
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        if not (0.0 <= self.nms_threshold <= 1.0):
            raise ValueError("NMS threshold must be between 0.0 and 1.0")
        if self.max_detections <= 0:
            raise ValueError("Max detections must be positive")


@dataclass
class TrackerConfig:
    """Tracker configuration parameters."""
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    feature_extractor: str = "mobilenet"
    max_tracks: int = 100
    
    def __post_init__(self):
        """Validate tracker configuration."""
        if self.max_age <= 0:
            raise ValueError("Max age must be positive")
        if self.min_hits <= 0:
            raise ValueError("Min hits must be positive")
        if not (0.0 <= self.iou_threshold <= 1.0):
            raise ValueError("IoU threshold must be between 0.0 and 1.0")


@dataclass
class AssessmentConfig:
    """Threat assessment configuration parameters."""
    threat_rules: Dict[str, Any]
    alert_thresholds: Dict[ThreatLevel, float]
    restricted_zones: List[Tuple[float, float, float, float]]  # x1, y1, x2, y2
    priority_classes: List[str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.priority_classes is None:
            self.priority_classes = ["drone", "aircraft"]
        if not self.alert_thresholds:
            self.alert_thresholds = {
                ThreatLevel.LOW: 0.3,
                ThreatLevel.MEDIUM: 0.5,
                ThreatLevel.HIGH: 0.7,
                ThreatLevel.CRITICAL: 0.9
            }


@dataclass
class Alert:
    """Alert object for threat notifications."""
    alert_id: str
    object_id: int
    threat_assessment: ThreatAssessment
    message: str
    timestamp: datetime
    acknowledged: bool = False
    
    def __post_init__(self):
        """Validate alert data."""
        if not self.alert_id:
            raise ValueError("Alert ID cannot be empty")
        if self.object_id < 0:
            raise ValueError("Object ID must be non-negative")


@dataclass
class SystemStatus:
    """System status information."""
    fps: float
    memory_usage: float  # Percentage
    gpu_memory_usage: float  # Percentage
    cpu_usage: float  # Percentage
    active_tracks: int
    total_detections: int
    timestamp: datetime
    errors: List[str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.errors is None:
            self.errors = []