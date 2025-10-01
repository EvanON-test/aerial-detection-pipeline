"""
Core data models for the aerial object detection and tracking system.

This module defines the fundamental data structures used throughout the aerial
detection pipeline. All models use dataclasses for efficient memory usage and
automatic generation of common methods (__init__, __repr__, __eq__, etc.).

The data models are organized into several categories:

1. **Detection Models**: Raw detection results from inference models
2. **Tracking Models**: Objects with temporal information and trajectories  
3. **Assessment Models**: Threat analysis and alert generation
4. **Configuration Models**: Type-safe configuration containers
5. **System Models**: Status and monitoring information

Key Design Principles:
- Immutable data structures where possible
- Comprehensive validation in __post_init__ methods
- Type hints for all fields to enable static analysis
- Clear separation between raw detections and tracked objects
- Extensible threat assessment framework

Example Usage:
    # Create a detection from model output
    detection = Detection(
        bbox=(100, 100, 200, 200),
        confidence=0.85,
        class_id=0,
        class_name="drone",
        timestamp=datetime.now(),
        frame_id=1
    )
    
    # Create a tracked object with trajectory
    tracked_obj = TrackedObject(
        track_id=1,
        detection=detection,
        trajectory=[(150, 150)],
        velocity=(5.0, 2.0),
        age=10,
        time_since_update=0,
        state=TrackState.CONFIRMED
    )
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Tuple, Optional, Any, Dict


class TrackState(Enum):
    """
    Tracking state enumeration for multi-object tracking lifecycle.
    
    Defines the possible states of a tracked object throughout its lifecycle:
    
    - TENTATIVE: Newly detected object, not yet confirmed as a valid track
    - CONFIRMED: Stable track with sufficient detection history
    - DELETED: Track marked for removal due to extended absence
    
    State transitions typically follow: TENTATIVE -> CONFIRMED -> DELETED
    """
    TENTATIVE = "tentative"  # New track, needs more detections to confirm
    CONFIRMED = "confirmed"  # Stable track with sufficient history
    DELETED = "deleted"      # Track lost, marked for cleanup


class ThreatLevel(Enum):
    """
    Threat level enumeration for security assessment.
    
    Defines escalating threat levels used by the threat assessment engine:
    
    - LOW: Minimal threat, routine monitoring
    - MEDIUM: Elevated concern, increased monitoring
    - HIGH: Significant threat, immediate attention required
    - CRITICAL: Imminent danger, emergency response needed
    
    These levels correspond to different alert thresholds and response protocols.
    """
    LOW = "low"          # Routine monitoring, no immediate action
    MEDIUM = "medium"    # Increased monitoring, log events
    HIGH = "high"        # Immediate attention, notify operators
    CRITICAL = "critical" # Emergency response, automatic alerts


@dataclass
class Detection:
    """
    Detection object representing a detected aerial object from inference.
    
    Represents a single object detection from the inference model, containing
    spatial information (bounding box), classification results (class and confidence),
    and temporal context (timestamp and frame ID).
    
    Attributes:
        bbox: Bounding box coordinates as (x1, y1, x2, y2) in image coordinates
        confidence: Detection confidence score from 0.0 to 1.0
        class_id: Numeric class identifier from the model
        class_name: Human-readable class name (e.g., "drone", "aircraft")
        timestamp: When the detection occurred
        frame_id: Sequential frame number for temporal tracking
    
    Note:
        Bounding box coordinates should be in the original image coordinate system.
        The confidence score represents the model's certainty about the detection.
    """
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 in image coordinates
    confidence: float                        # Model confidence [0.0, 1.0]
    class_id: int                           # Numeric class identifier
    class_name: str                         # Human-readable class name
    timestamp: datetime                     # Detection timestamp
    frame_id: int                          # Sequential frame number
    
    def __post_init__(self):
        """
        Validate detection data after initialization.
        
        Ensures that confidence scores are within valid range and class IDs
        are non-negative. This prevents invalid detections from propagating
        through the system and causing downstream errors.
        
        Raises:
            ValueError: If confidence is outside [0.0, 1.0] or class_id is negative
        """
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if self.class_id < 0:
            raise ValueError(f"Class ID must be non-negative, got {self.class_id}")
        if not self.class_name.strip():
            raise ValueError("Class name cannot be empty")
    
    @property
    def center(self) -> Tuple[float, float]:
        """
        Calculate the center point of the bounding box.
        
        Returns:
            Tuple of (center_x, center_y) coordinates
        """
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    
    @property
    def area(self) -> float:
        """
        Calculate the area of the bounding box.
        
        Returns:
            Area in square pixels
        """
        x1, y1, x2, y2 = self.bbox
        return max(0.0, (x2 - x1) * (y2 - y1))
    
    @property
    def width(self) -> float:
        """Get bounding box width."""
        return max(0.0, self.bbox[2] - self.bbox[0])
    
    @property
    def height(self) -> float:
        """Get bounding box height."""
        return max(0.0, self.bbox[3] - self.bbox[1])


@dataclass
class TrackedObject:
    """
    Tracked object with trajectory and state information.
    
    Represents an object that has been tracked across multiple frames, containing
    the current detection, historical trajectory, motion information, and tracking
    state. This is the primary data structure used by the multi-object tracking
    system to maintain object identity over time.
    
    Attributes:
        track_id: Unique identifier for this track
        detection: Most recent detection for this object
        trajectory: Historical center positions [(x, y), ...]
        velocity: Current velocity as (vx, vy) in pixels/frame
        age: Total number of frames since track creation
        time_since_update: Frames since last successful detection match
        state: Current tracking state (TENTATIVE, CONFIRMED, DELETED)
    
    The trajectory maintains a history of object positions for motion analysis
    and prediction. Velocity is calculated from recent trajectory points and
    used for motion-based prediction and threat assessment.
    """
    track_id: int                           # Unique track identifier
    detection: Detection                    # Current detection
    trajectory: List[Tuple[float, float]]   # Historical center positions
    velocity: Tuple[float, float]           # Current velocity (vx, vy)
    age: int                               # Total frames since creation
    time_since_update: int                 # Frames since last detection
    state: TrackState                      # Current tracking state
    
    def __post_init__(self):
        """
        Validate tracked object data after initialization.
        
        Ensures all numeric values are within valid ranges and the trajectory
        contains at least the current position. This prevents invalid tracking
        data from causing downstream processing errors.
        
        Raises:
            ValueError: If any validation check fails
        """
        if self.track_id < 0:
            raise ValueError(f"Track ID must be non-negative, got {self.track_id}")
        if self.age < 0:
            raise ValueError(f"Age must be non-negative, got {self.age}")
        if self.time_since_update < 0:
            raise ValueError(f"Time since update must be non-negative, got {self.time_since_update}")
        if not self.trajectory:
            raise ValueError("Trajectory cannot be empty")
    
    @property
    def current_position(self) -> Tuple[float, float]:
        """
        Get the current position of the tracked object.
        
        Returns:
            Current (x, y) position, typically the most recent trajectory point
        """
        return self.trajectory[-1] if self.trajectory else self.detection.center
    
    @property
    def speed(self) -> float:
        """
        Calculate the current speed (magnitude of velocity vector).
        
        Returns:
            Speed in pixels per frame
        """
        vx, vy = self.velocity
        return (vx ** 2 + vy ** 2) ** 0.5
    
    @property
    def direction(self) -> float:
        """
        Calculate the current direction of movement in radians.
        
        Returns:
            Direction angle in radians (0 = right, π/2 = up)
        """
        import math
        vx, vy = self.velocity
        return math.atan2(vy, vx)
    
    def predict_position(self, frames_ahead: int = 1) -> Tuple[float, float]:
        """
        Predict future position based on current velocity.
        
        Args:
            frames_ahead: Number of frames to predict ahead
            
        Returns:
            Predicted (x, y) position
        """
        current_x, current_y = self.current_position
        vx, vy = self.velocity
        return (current_x + vx * frames_ahead, current_y + vy * frames_ahead)
    
    def get_trajectory_length(self) -> float:
        """
        Calculate the total distance traveled along the trajectory.
        
        Returns:
            Total distance in pixels
        """
        if len(self.trajectory) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(self.trajectory)):
            x1, y1 = self.trajectory[i-1]
            x2, y2 = self.trajectory[i]
            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            total_distance += distance
        
        return total_distance


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