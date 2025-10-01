#!/bin/bash

# Development environment setup script for Aerial Object Detection System
# This script sets up a virtual environment with all necessary dependencies

echo "Setting up development environment for Aerial Object Detection System..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Development environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate the environment, run:"
echo "  deactivate"
echo ""
echo "To run tests:"
echo "  python -m pytest tests/ -v"
echo ""
echo "Note: OpenCV installation may take several minutes on macOS."