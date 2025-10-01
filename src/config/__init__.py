"""
Configuration management module.

This module provides comprehensive configuration management for the aerial
detection system. It supports multiple file formats (YAML, JSON), automatic
default generation, runtime updates, and validation.

Key Components:
- **ConfigurationManager**: Main configuration management class
- **Default configurations**: Sensible defaults for all system components
- **Validation**: Comprehensive configuration validation
- **Hot-reload**: Runtime configuration updates without restart

The configuration system manages settings for:
- Camera parameters (resolution, FPS, exposure settings)
- Model inference (confidence thresholds, NMS parameters)
- Tracking (age limits, IoU thresholds, feature extraction)
- Threat assessment (rules, thresholds, restricted zones)
- System performance (memory limits, optimization settings)
- Logging (levels, file paths, rotation settings)

Usage:
    from src.config import ConfigurationManager
    
    config_manager = ConfigurationManager("config/system.yaml")
    camera_config = config_manager.get_camera_config()
    model_config = config_manager.get_model_config()

The module automatically handles:
- File format detection based on extension
- Missing configuration file creation
- Configuration merging with defaults
- Type conversion and validation
- Error reporting with helpful messages
"""

from .config_manager import ConfigurationManager

__all__ = ["ConfigurationManager"]