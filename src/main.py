"""
Main entry point for the Aerial Object Detection & Tracking Pipeline.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.config_manager import ConfigurationManager


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Aerial Object Detection & Tracking Pipeline"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit"
    )
    
    args = parser.parse_args()
    
    # Initialize configuration manager
    try:
        config_manager = ConfigurationManager(args.config)
        print(f"Configuration loaded from: {args.config}")
        
        if args.validate_config:
            errors = config_manager.validate_config()
            if errors:
                print("Configuration validation errors:")
                for error in errors:
                    print(f"  - {error}")
                sys.exit(1)
            else:
                print("Configuration is valid.")
                sys.exit(0)
        
        # TODO: Initialize and start the detection pipeline
        print("Aerial Object Detection & Tracking Pipeline")
        print("Pipeline initialization will be implemented in subsequent tasks.")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()