"""
Tests for configuration manager.
"""

import pytest
import tempfile
import os
from pathlib import Path
from src.config.config_manager import ConfigurationManager


def test_config_manager_creation():
    """Test ConfigurationManager creation with default config."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "test_config.yaml")
        config_manager = ConfigurationManager(config_path)
        
        # Should create default config
        assert Path(config_path).exists()
        
        # Test getting camera config
        camera_config = config_manager.get_camera_config()
        assert camera_config.fps > 0
        assert len(camera_config.resolution) == 2


def test_config_validation():
    """Test configuration validation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "test_config.yaml")
        config_manager = ConfigurationManager(config_path)
        
        errors = config_manager.validate_config()
        assert isinstance(errors, list)


def test_config_update():
    """Test configuration updates."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "test_config.yaml")
        config_manager = ConfigurationManager(config_path)
        
        # Update camera config
        config_manager.update_config("camera", {"fps": 60})
        camera_config = config_manager.get_camera_config()
        assert camera_config.fps == 60


def test_config_sections():
    """Test getting different config sections."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "test_config.yaml")
        config_manager = ConfigurationManager(config_path)
        
        # Test all main sections exist
        camera_config = config_manager.get_config("camera")
        model_config = config_manager.get_config("model")
        tracker_config = config_manager.get_config("tracker")
        assessment_config = config_manager.get_config("assessment")
        
        assert "resolution" in camera_config
        assert "primary_model" in model_config
        assert "max_age" in tracker_config
        assert "threat_rules" in assessment_config