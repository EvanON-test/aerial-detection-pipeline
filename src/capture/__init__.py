"""
Camera capture module for aerial object detection system.

This module provides camera capture functionality optimized for Jetson Nano
and Raspberry Pi Camera Module 2. It includes threaded frame capture,
adaptive resolution scaling, and performance monitoring.
"""

from .frame_capture_manager import FrameCaptureManager

__all__ = ['FrameCaptureManager']