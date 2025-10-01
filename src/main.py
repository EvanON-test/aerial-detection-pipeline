"""
Main entry point for the Aerial Detection Pipeline.

This module serves as the primary entry point for the aerial object detection and tracking
system. It handles command-line argument parsing, configuration management initialization,
and orchestrates the startup of the detection pipeline.

The system is designed to:
- Load and validate configuration from YAML/JSON files
- Initialize camera capture, model inference, tracking, and threat assessment components
- Provide a unified interface for running the complete detection pipeline

Usage:
    python -m src.main --config config/config.yaml
    python -m src.main --validate-config --config config/custom.yaml

Author: Aerial Detection Team
Version: 1.0.0
"""

import sys
import argparse
from pathlib import Path

# Add project root to path for imports - ensures modules can be found regardless of execution context
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.config_manager import ConfigurationManager


def main():
    """
    Main application entry point.
    
    Parses command-line arguments, initializes the configuration manager,
    and starts the aerial detection pipeline. Handles graceful error reporting
    and provides configuration validation functionality.
    
    Command-line Arguments:
        --config: Path to configuration file (default: config/config.yaml)
        --validate-config: Validate configuration and exit without running pipeline
    
    Returns:
        None
        
    Raises:
        SystemExit: On configuration errors or validation failures
    """
    # Set up command-line argument parser with detailed help
    parser = argparse.ArgumentParser(
        description="Aerial Detection Pipeline - Real-time aerial object detection and tracking system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run with default config
  %(prog)s --config custom.yaml              # Run with custom config
  %(prog)s --validate-config                 # Validate default config
  %(prog)s --validate-config --config test.yaml  # Validate specific config
        """
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        metavar="PATH",
        help="Path to configuration file (YAML or JSON format)"
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration file and exit without starting pipeline"
    )
    
    args = parser.parse_args()
    
    # Initialize configuration manager with error handling
    try:
        print(f"Loading configuration from: {args.config}")
        config_manager = ConfigurationManager(args.config)
        print("✓ Configuration loaded successfully")
        
        # Handle configuration validation mode
        if args.validate_config:
            print("Validating configuration...")
            errors = config_manager.validate_config()
            if errors:
                print("❌ Configuration validation failed:")
                for error in errors:
                    print(f"  • {error}")
                sys.exit(1)
            else:
                print("✓ Configuration is valid")
                sys.exit(0)
        
        # TODO: Initialize and start the detection pipeline
        # This will be implemented in subsequent development phases:
        # 1. Initialize camera capture system
        # 2. Load and configure inference models
        # 3. Set up multi-object tracking
        # 4. Initialize threat assessment engine
        # 5. Start main processing loop
        
        print("🚁 Aerial Detection Pipeline")
        print("📋 Pipeline initialization will be implemented in subsequent tasks")
        print("🔧 Current status: Configuration system ready")
        
    except FileNotFoundError as e:
        print(f"❌ Configuration file not found: {e}")
        print("💡 Tip: Check the file path or use --config to specify a different file")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        print("💡 Tip: Use --validate-config to check your configuration")
        sys.exit(1)


if __name__ == "__main__":
    main()