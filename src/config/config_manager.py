"""
Configuration management system with YAML/JSON support.
"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..models.interfaces import ConfigurationInterface
from ..models.data_models import (
    CameraConfig, ModelConfig, TrackerConfig, AssessmentConfig, ThreatLevel
)


class ConfigurationManager(ConfigurationInterface):
    """Configuration manager supporting YAML and JSON formats."""
    
    def __init__(self, config_path: str):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        self.config_path = Path(config_path)
        self.config_data: Dict[str, Any] = {}
        self._last_modified: Optional[float] = None
        
        # Default configuration structure
        self._default_config = {
            "camera": {
                "resolution": [1920, 1080],
                "fps": 30,
                "exposure_mode": "auto",
                "focus_mode": "auto",
                "brightness": 50,
                "contrast": 0
            },
            "model": {
                "primary_model": "models/yolov8n.onnx",
                "fallback_model": "models/yolov8n.pt",
                "confidence_threshold": 0.5,
                "nms_threshold": 0.4,
                "max_detections": 100,
                "input_size": [640, 640]
            },
            "tracker": {
                "max_age": 30,
                "min_hits": 3,
                "iou_threshold": 0.3,
                "feature_extractor": "mobilenet",
                "max_tracks": 100
            },
            "assessment": {
                "threat_rules": {
                    "size_weight": 0.3,
                    "speed_weight": 0.4,
                    "proximity_weight": 0.3,
                    "formation_bonus": 0.2
                },
                "alert_thresholds": {
                    "low": 0.3,
                    "medium": 0.5,
                    "high": 0.7,
                    "critical": 0.9
                },
                "restricted_zones": [],
                "priority_classes": ["drone", "aircraft"]
            },
            "logging": {
                "level": "INFO",
                "file_path": "logs/aerial_detection.log",
                "max_file_size": "10MB",
                "backup_count": 5,
                "log_detections": True,
                "log_tracking": True,
                "log_threats": True
            },
            "system": {
                "max_memory_usage": 0.8,
                "target_fps": 15,
                "auto_optimize": True,
                "fallback_cpu": True,
                "performance_monitoring": True
            }
        }
        
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file."""
        if not self.config_path.exists():
            self._create_default_config()
            return
        
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
                    self.config_data = yaml.safe_load(f)
                else:
                    self.config_data = json.load(f)
            
            self._last_modified = self.config_path.stat().st_mtime
            self._merge_with_defaults()
            
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using default configuration")
            self.config_data = self._default_config.copy()
    
    def _create_default_config(self) -> None:
        """Create default configuration file."""
        self.config_data = self._default_config.copy()
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_config()
    
    def _merge_with_defaults(self) -> None:
        """Merge loaded config with defaults to ensure all keys exist."""
        def merge_dict(default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
            result = default.copy()
            for key, value in loaded.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dict(result[key], value)
                else:
                    result[key] = value
            return result
        
        self.config_data = merge_dict(self._default_config, self.config_data)
    
    def get_config(self, section: str) -> Dict[str, Any]:
        """Get configuration for a specific section.
        
        Args:
            section: Configuration section name
            
        Returns:
            Dictionary containing section configuration
        """
        self._check_file_changes()
        return self.config_data.get(section, {}).copy()
    
    def update_config(self, section: str, params: Dict[str, Any]) -> None:
        """Update configuration parameters.
        
        Args:
            section: Configuration section name
            params: Parameters to update
        """
        if section not in self.config_data:
            self.config_data[section] = {}
        
        self.config_data[section].update(params)
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self._load_config()
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
                    yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(self.config_data, f, indent=2)
            
            self._last_modified = self.config_path.stat().st_mtime
            
        except Exception as e:
            raise RuntimeError(f"Failed to save configuration: {e}")
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any errors.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        try:
            # Validate camera config
            camera_config = self.get_camera_config()
        except Exception as e:
            errors.append(f"Camera config error: {e}")
        
        try:
            # Validate model config
            model_config = self.get_model_config()
        except Exception as e:
            errors.append(f"Model config error: {e}")
        
        try:
            # Validate tracker config
            tracker_config = self.get_tracker_config()
        except Exception as e:
            errors.append(f"Tracker config error: {e}")
        
        try:
            # Validate assessment config
            assessment_config = self.get_assessment_config()
        except Exception as e:
            errors.append(f"Assessment config error: {e}")
        
        return errors
    
    def get_camera_config(self) -> CameraConfig:
        """Get camera configuration as CameraConfig object."""
        config = self.get_config("camera")
        return CameraConfig(
            resolution=tuple(config["resolution"]),
            fps=config["fps"],
            exposure_mode=config["exposure_mode"],
            focus_mode=config["focus_mode"],
            brightness=config["brightness"],
            contrast=config["contrast"]
        )
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration as ModelConfig object."""
        config = self.get_config("model")
        return ModelConfig(
            primary_model=config["primary_model"],
            fallback_model=config["fallback_model"],
            confidence_threshold=config["confidence_threshold"],
            nms_threshold=config["nms_threshold"],
            max_detections=config["max_detections"],
            input_size=tuple(config["input_size"])
        )
    
    def get_tracker_config(self) -> TrackerConfig:
        """Get tracker configuration as TrackerConfig object."""
        config = self.get_config("tracker")
        return TrackerConfig(
            max_age=config["max_age"],
            min_hits=config["min_hits"],
            iou_threshold=config["iou_threshold"],
            feature_extractor=config["feature_extractor"],
            max_tracks=config["max_tracks"]
        )
    
    def get_assessment_config(self) -> AssessmentConfig:
        """Get assessment configuration as AssessmentConfig object."""
        config = self.get_config("assessment")
        
        # Convert string keys to ThreatLevel enum
        alert_thresholds = {}
        for level_str, threshold in config["alert_thresholds"].items():
            try:
                level = ThreatLevel(level_str.lower())
                alert_thresholds[level] = threshold
            except ValueError:
                continue
        
        return AssessmentConfig(
            threat_rules=config["threat_rules"],
            alert_thresholds=alert_thresholds,
            restricted_zones=[tuple(zone) for zone in config["restricted_zones"]],
            priority_classes=config["priority_classes"]
        )
    
    def _check_file_changes(self) -> None:
        """Check if config file has been modified and reload if necessary."""
        if not self.config_path.exists():
            return
        
        current_mtime = self.config_path.stat().st_mtime
        if self._last_modified is None or current_mtime > self._last_modified:
            self.reload_config()
    
    def export_config(self, export_path: str, format: str = "yaml") -> None:
        """Export current configuration to a file.
        
        Args:
            export_path: Path to export file
            format: Export format ("yaml" or "json")
        """
        export_file = Path(export_path)
        export_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_file, 'w') as f:
            if format.lower() == "yaml":
                yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
            else:
                json.dump(self.config_data, f, indent=2)
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get complete configuration dictionary."""
        self._check_file_changes()
        return self.config_data.copy()
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self.config_data = self._default_config.copy()
        self.save_config()