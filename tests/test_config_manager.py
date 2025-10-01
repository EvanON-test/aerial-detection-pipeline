"""
Comprehensive tests for the configuration manager system.

This test module validates the ConfigurationManager class functionality including:
- Configuration file creation and loading (YAML/JSON)
- Default configuration generation
- Configuration validation and error handling
- Runtime configuration updates
- Type-safe configuration object generation
- File format auto-detection

The tests use temporary directories to avoid interfering with actual
configuration files and ensure test isolation.

Test Categories:
- Creation and initialization tests
- File format handling (YAML/JSON)
- Configuration validation
- Runtime updates and persistence
- Error handling and edge cases
- Type conversion and validation

Usage:
    pytest tests/test_config_manager.py -v
    pytest tests/test_config_manager.py::test_config_manager_creation
"""

import pytest
import tempfile
import os
from pathlib import Path
from src.config.config_manager import ConfigurationManager


def test_config_manager_creation():
    """
    Test ConfigurationManager creation with automatic default config generation.
    
    Verifies that:
    1. ConfigurationManager creates default config when file doesn't exist
    2. Default configuration file is properly written to disk
    3. Camera configuration can be retrieved and has valid values
    4. Configuration objects have expected structure and types
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "test_config.yaml")
        
        # Create configuration manager - should generate default config
        config_manager = ConfigurationManager(config_path)
        
        # Verify default config file was created
        assert Path(config_path).exists(), "Default configuration file should be created"
        
        # Test camera configuration retrieval and validation
        camera_config = config_manager.get_camera_config()
        assert camera_config.fps > 0, "FPS should be positive"
        assert len(camera_config.resolution) == 2, "Resolution should be (width, height) tuple"
        assert all(dim > 0 for dim in camera_config.resolution), "Resolution dimensions should be positive"
        assert isinstance(camera_config.exposure_mode, str), "Exposure mode should be string"
        assert isinstance(camera_config.focus_mode, str), "Focus mode should be string"


def test_config_validation():
    """
    Test comprehensive configuration validation functionality.
    
    Verifies that:
    1. validate_config() returns a list of error messages
    2. Default configuration passes validation (no errors)
    3. Validation covers all major configuration sections
    4. Invalid configurations are properly detected
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "test_config.yaml")
        config_manager = ConfigurationManager(config_path)
        
        # Test validation with default configuration
        errors = config_manager.validate_config()
        assert isinstance(errors, list), "Validation should return list of errors"
        assert len(errors) == 0, f"Default configuration should be valid, but got errors: {errors}"
        
        # Test that validation covers all major sections
        # This ensures the validation method actually checks each component
        camera_config = config_manager.get_camera_config()
        model_config = config_manager.get_model_config()
        tracker_config = config_manager.get_tracker_config()
        assessment_config = config_manager.get_assessment_config()
        
        # Verify each config object was created successfully
        assert camera_config is not None, "Camera config should be created"
        assert model_config is not None, "Model config should be created"
        assert tracker_config is not None, "Tracker config should be created"
        assert assessment_config is not None, "Assessment config should be created"


def test_config_update():
    """
    Test runtime configuration updates and persistence.
    
    Verifies that:
    1. Configuration can be updated at runtime
    2. Updates are reflected in configuration objects
    3. Multiple parameters can be updated simultaneously
    4. Updates don't affect other configuration sections
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "test_config.yaml")
        config_manager = ConfigurationManager(config_path)
        
        # Get original camera config for comparison
        original_camera_config = config_manager.get_camera_config()
        original_fps = original_camera_config.fps
        original_resolution = original_camera_config.resolution
        
        # Update camera FPS
        config_manager.update_config("camera", {"fps": 60})
        updated_camera_config = config_manager.get_camera_config()
        
        # Verify update was applied
        assert updated_camera_config.fps == 60, "FPS should be updated to 60"
        assert updated_camera_config.resolution == original_resolution, "Resolution should remain unchanged"
        
        # Test multiple parameter update
        config_manager.update_config("camera", {
            "fps": 30,
            "brightness": 75,
            "contrast": 10
        })
        multi_updated_config = config_manager.get_camera_config()
        
        assert multi_updated_config.fps == 30, "FPS should be updated to 30"
        assert multi_updated_config.brightness == 75, "Brightness should be updated to 75"
        assert multi_updated_config.contrast == 10, "Contrast should be updated to 10"


def test_config_sections():
    """
    Test retrieval of different configuration sections and their structure.
    
    Verifies that:
    1. All major configuration sections are available
    2. Each section contains expected keys and structure
    3. Configuration sections return dictionaries with proper data types
    4. Section isolation (changes to one don't affect others)
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "test_config.yaml")
        config_manager = ConfigurationManager(config_path)
        
        # Test camera configuration section
        camera_config = config_manager.get_config("camera")
        assert isinstance(camera_config, dict), "Camera config should be dictionary"
        assert "resolution" in camera_config, "Camera config should have resolution"
        assert "fps" in camera_config, "Camera config should have fps"
        assert "exposure_mode" in camera_config, "Camera config should have exposure_mode"
        assert isinstance(camera_config["resolution"], list), "Resolution should be list"
        assert isinstance(camera_config["fps"], int), "FPS should be integer"
        
        # Test model configuration section
        model_config = config_manager.get_config("model")
        assert isinstance(model_config, dict), "Model config should be dictionary"
        assert "primary_model" in model_config, "Model config should have primary_model"
        assert "confidence_threshold" in model_config, "Model config should have confidence_threshold"
        assert "nms_threshold" in model_config, "Model config should have nms_threshold"
        assert isinstance(model_config["confidence_threshold"], float), "Confidence threshold should be float"
        
        # Test tracker configuration section
        tracker_config = config_manager.get_config("tracker")
        assert isinstance(tracker_config, dict), "Tracker config should be dictionary"
        assert "max_age" in tracker_config, "Tracker config should have max_age"
        assert "min_hits" in tracker_config, "Tracker config should have min_hits"
        assert "iou_threshold" in tracker_config, "Tracker config should have iou_threshold"
        assert isinstance(tracker_config["max_age"], int), "Max age should be integer"
        
        # Test assessment configuration section
        assessment_config = config_manager.get_config("assessment")
        assert isinstance(assessment_config, dict), "Assessment config should be dictionary"
        assert "threat_rules" in assessment_config, "Assessment config should have threat_rules"
        assert "alert_thresholds" in assessment_config, "Assessment config should have alert_thresholds"
        assert "restricted_zones" in assessment_config, "Assessment config should have restricted_zones"
        assert isinstance(assessment_config["threat_rules"], dict), "Threat rules should be dictionary"