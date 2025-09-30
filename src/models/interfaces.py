"""
Base interfaces for the aerial object detection and tracking system.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from .data_models import (
    Detection, TrackedObject, ThreatAssessment, Alert,
    CameraConfig, ModelConfig, TrackerConfig, AssessmentConfig,
    SystemStatus
)


class FrameCaptureInterface(ABC):
    """Interface for frame capture management."""
    
    @abstractmethod
    def start_capture(self) -> None:
        """Start the camera capture process."""
        pass
    
    @abstractmethod
    def get_frame(self) -> Tuple[bool, np.ndarray]:
        """Get the next frame from the camera.
        
        Returns:
            Tuple of (success, frame) where success is bool and frame is numpy array
        """
        pass
    
    @abstractmethod
    def adjust_resolution(self, target_fps: float) -> None:
        """Adjust camera resolution based on target FPS."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up camera resources."""
        pass
    
    @abstractmethod
    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information and current settings."""
        pass


class ModelInferenceInterface(ABC):
    """Interface for model inference engine."""
    
    @abstractmethod
    def load_model(self, model_path: str, optimize: bool = True) -> None:
        """Load a model for inference.
        
        Args:
            model_path: Path to the model file
            optimize: Whether to optimize the model (e.g., with TensorRT)
        """
        pass
    
    @abstractmethod
    def infer(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of Detection objects
        """
        pass
    
    @abstractmethod
    def switch_model(self, model_name: str) -> None:
        """Switch to a different loaded model."""
        pass
    
    @abstractmethod
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for the current model."""
        pass
    
    @abstractmethod
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for model input."""
        pass
    
    @abstractmethod
    def postprocess_detections(self, raw_output: Any) -> List[Detection]:
        """Postprocess raw model output to Detection objects."""
        pass


class TrackerInterface(ABC):
    """Interface for multi-object tracking."""
    
    @abstractmethod
    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """Update tracker with new detections.
        
        Args:
            detections: List of Detection objects from current frame
            
        Returns:
            List of TrackedObject objects
        """
        pass
    
    @abstractmethod
    def get_trajectories(self) -> Dict[int, List[Tuple[float, float]]]:
        """Get trajectory history for all tracked objects.
        
        Returns:
            Dictionary mapping track_id to list of (x, y) positions
        """
        pass
    
    @abstractmethod
    def cleanup_lost_tracks(self) -> None:
        """Remove tracks that have been lost for too long."""
        pass
    
    @abstractmethod
    def get_track_by_id(self, track_id: int) -> Optional[TrackedObject]:
        """Get a specific track by ID."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the tracker state."""
        pass


class ThreatAssessmentInterface(ABC):
    """Interface for threat assessment engine."""
    
    @abstractmethod
    def assess_objects(self, tracked_objects: List[TrackedObject]) -> List[ThreatAssessment]:
        """Assess threat level for tracked objects.
        
        Args:
            tracked_objects: List of TrackedObject instances
            
        Returns:
            List of ThreatAssessment objects
        """
        pass
    
    @abstractmethod
    def update_threat_rules(self, rules: Dict[str, Any]) -> None:
        """Update threat assessment rules."""
        pass
    
    @abstractmethod
    def get_alert_queue(self) -> List[Alert]:
        """Get pending alerts."""
        pass
    
    @abstractmethod
    def check_restricted_zones(self, tracked_object: TrackedObject) -> bool:
        """Check if object is in a restricted zone."""
        pass
    
    @abstractmethod
    def analyze_behavior(self, tracked_object: TrackedObject) -> Dict[str, float]:
        """Analyze object behavior patterns."""
        pass


class VisualizationInterface(ABC):
    """Interface for visualization and display."""
    
    @abstractmethod
    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detection bounding boxes on frame."""
        pass
    
    @abstractmethod
    def draw_tracks(self, frame: np.ndarray, tracked_objects: List[TrackedObject]) -> np.ndarray:
        """Draw tracking information on frame."""
        pass
    
    @abstractmethod
    def draw_threats(self, frame: np.ndarray, threat_assessments: List[ThreatAssessment]) -> np.ndarray:
        """Draw threat indicators on frame."""
        pass
    
    @abstractmethod
    def draw_system_info(self, frame: np.ndarray, system_status: SystemStatus) -> np.ndarray:
        """Draw system information overlay."""
        pass
    
    @abstractmethod
    def create_dashboard(self, tracked_objects: List[TrackedObject], 
                        threat_assessments: List[ThreatAssessment]) -> np.ndarray:
        """Create a dashboard view with statistics."""
        pass


class EventLoggerInterface(ABC):
    """Interface for event logging."""
    
    @abstractmethod
    def log_detection(self, detection: Detection) -> None:
        """Log a detection event."""
        pass
    
    @abstractmethod
    def log_tracking(self, tracked_object: TrackedObject) -> None:
        """Log a tracking event."""
        pass
    
    @abstractmethod
    def log_threat(self, threat_assessment: ThreatAssessment) -> None:
        """Log a threat assessment."""
        pass
    
    @abstractmethod
    def log_alert(self, alert: Alert) -> None:
        """Log an alert."""
        pass
    
    @abstractmethod
    def log_system_status(self, status: SystemStatus) -> None:
        """Log system status."""
        pass
    
    @abstractmethod
    def get_logs(self, start_time: Optional[str] = None, 
                end_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve logs within time range."""
        pass


class ConfigurationInterface(ABC):
    """Interface for configuration management."""
    
    @abstractmethod
    def get_config(self, section: str) -> Dict[str, Any]:
        """Get configuration for a specific section."""
        pass
    
    @abstractmethod
    def update_config(self, section: str, params: Dict[str, Any]) -> None:
        """Update configuration parameters."""
        pass
    
    @abstractmethod
    def reload_config(self) -> None:
        """Reload configuration from file."""
        pass
    
    @abstractmethod
    def save_config(self) -> None:
        """Save current configuration to file."""
        pass
    
    @abstractmethod
    def validate_config(self) -> List[str]:
        """Validate configuration and return any errors."""
        pass