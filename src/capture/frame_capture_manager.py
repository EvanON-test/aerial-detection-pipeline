"""
Frame capture manager optimized for Jetson Nano and Pi Camera Module 2.

This module implements the FrameCaptureManager class that provides efficient
camera capture with threading, adaptive resolution scaling, and performance
monitoring specifically optimized for edge devices.

Key Features:
- Threaded frame capture to prevent blocking
- Adaptive resolution scaling based on system performance
- Pi Camera Module 2 optimization
- Configurable buffer management
- Performance monitoring and statistics
- Graceful error handling and recovery

The implementation prioritizes real-time performance while maintaining
frame quality suitable for aerial object detection.
"""

import threading
import time
import queue
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
import logging
from datetime import datetime

from ..models.interfaces import FrameCaptureInterface
from ..models.data_models import CameraConfig


class FrameCaptureManager(FrameCaptureInterface):
    """
    Frame capture manager optimized for Jetson Nano and Pi Camera Module 2.
    
    Provides efficient camera capture with threading, adaptive resolution scaling,
    and performance monitoring. Designed specifically for edge deployment with
    limited computational resources.
    
    Features:
    - Non-blocking threaded frame capture
    - Adaptive resolution scaling for performance
    - Pi Camera Module 2 CSI interface support
    - USB camera fallback support
    - Configurable frame buffer management
    - Real-time performance monitoring
    - Automatic error recovery
    
    Example:
        config = CameraConfig(resolution=(1920, 1080), fps=30)
        capture_manager = FrameCaptureManager(config)
        capture_manager.start_capture()
        
        success, frame = capture_manager.get_frame()
        if success:
            # Process frame
            pass
            
        capture_manager.cleanup()
    """
    
    def __init__(self, camera_config: CameraConfig):
        """
        Initialize the frame capture manager.
        
        Args:
            camera_config: Camera configuration parameters
        """
        self.config = camera_config
        self.logger = logging.getLogger(__name__)
        
        # Camera and capture state
        self.camera = None
        self.capture_thread = None
        self.frame_queue = queue.Queue(maxsize=5)  # Buffer for 5 frames
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Control flags
        self.is_capturing = False
        self.should_stop = False
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        self.last_fps_calculation = time.time()
        self.current_fps = 0.0
        self.dropped_frames = 0
        
        # Adaptive resolution parameters
        self.original_resolution = camera_config.resolution
        self.current_resolution = camera_config.resolution
        self.min_resolution = (640, 480)
        self.resolution_scale_factor = 0.8
        self.fps_threshold_low = 10.0  # Scale down if FPS drops below this
        self.fps_threshold_high = 25.0  # Scale up if FPS is above this
        
        # Camera detection and initialization
        self.camera_type = None  # Will be set during initialization
        self.device_id = None
        
    def _detect_camera_type(self) -> Tuple[str, Optional[int]]:
        """
        Detect available camera type and device ID.
        
        Attempts to detect Pi Camera Module 2 via CSI interface first,
        then falls back to USB cameras.
        
        Returns:
            Tuple of (camera_type, device_id) where camera_type is
            'csi', 'usb', or None if no camera found
        """
        # Try to detect Pi Camera Module 2 (CSI interface)
        try:
            # On Jetson Nano, CSI cameras are typically accessed via GStreamer
            # or directly through /dev/video0
            test_cap = cv2.VideoCapture(0)
            if test_cap.isOpened():
                # Check if this looks like a CSI camera
                ret, frame = test_cap.read()
                test_cap.release()
                if ret and frame is not None:
                    self.logger.info("Detected camera on device 0 (likely CSI)")
                    return 'csi', 0
        except Exception as e:
            self.logger.debug(f"CSI camera detection failed: {e}")
        
        # Try USB cameras (device IDs 0-3)
        for device_id in range(4):
            try:
                test_cap = cv2.VideoCapture(device_id)
                if test_cap.isOpened():
                    ret, frame = test_cap.read()
                    test_cap.release()
                    if ret and frame is not None:
                        self.logger.info(f"Detected USB camera on device {device_id}")
                        return 'usb', device_id
            except Exception as e:
                self.logger.debug(f"USB camera detection failed for device {device_id}: {e}")
        
        self.logger.error("No camera detected")
        return None, None
    
    def _initialize_camera(self) -> bool:
        """
        Initialize the camera with optimal settings for Jetson Nano.
        
        Returns:
            True if camera initialized successfully, False otherwise
        """
        self.camera_type, self.device_id = self._detect_camera_type()
        
        if self.camera_type is None:
            self.logger.error("No camera detected")
            return False
        
        try:
            if self.camera_type == 'csi':
                # For Pi Camera Module 2 on Jetson Nano, use GStreamer pipeline
                # This provides better performance than direct OpenCV access
                gst_pipeline = (
                    f"nvarguscamerasrc sensor-id=0 ! "
                    f"video/x-raw(memory:NVMM), "
                    f"width={self.current_resolution[0]}, "
                    f"height={self.current_resolution[1]}, "
                    f"framerate={self.config.fps}/1, format=NV12 ! "
                    f"nvvidconv ! video/x-raw, format=BGRx ! "
                    f"videoconvert ! video/x-raw, format=BGR ! "
                    f"appsink drop=1"
                )
                self.camera = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                self.logger.info("Initialized CSI camera with GStreamer pipeline")
            else:
                # USB camera initialization
                self.camera = cv2.VideoCapture(self.device_id)
                self.logger.info(f"Initialized USB camera on device {self.device_id}")
            
            if not self.camera.isOpened():
                self.logger.error("Failed to open camera")
                return False
            
            # Set camera properties
            self._configure_camera_properties()
            
            # Verify camera is working
            ret, test_frame = self.camera.read()
            if not ret or test_frame is None:
                self.logger.error("Camera opened but cannot read frames")
                return False
            
            self.logger.info(f"Camera initialized successfully: {self.current_resolution} @ {self.config.fps} FPS")
            return True
            
        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            return False
    
    def _configure_camera_properties(self):
        """Configure camera properties for optimal performance."""
        try:
            if self.camera_type == 'usb':
                # USB camera properties
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.current_resolution[0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.current_resolution[1])
                self.camera.set(cv2.CAP_PROP_FPS, self.config.fps)
                self.camera.set(cv2.CAP_PROP_BRIGHTNESS, self.config.brightness)
                self.camera.set(cv2.CAP_PROP_CONTRAST, self.config.contrast)
                
                # Optimize for low latency
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
            # Set auto-exposure and auto-focus if supported
            if hasattr(cv2, 'CAP_PROP_AUTO_EXPOSURE'):
                if self.config.exposure_mode == "auto":
                    self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                else:
                    self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            
            self.logger.info("Camera properties configured")
            
        except Exception as e:
            self.logger.warning(f"Some camera properties could not be set: {e}")
    
    def _capture_loop(self):
        """
        Main capture loop running in separate thread.
        
        Continuously captures frames and manages the frame buffer.
        Implements adaptive resolution scaling based on performance.
        """
        self.logger.info("Starting capture loop")
        frame_time_buffer = []
        last_adaptive_check = time.time()
        
        while not self.should_stop:
            try:
                frame_start_time = time.time()
                
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    self.logger.warning("Failed to capture frame")
                    self.dropped_frames += 1
                    time.sleep(0.01)  # Brief pause before retry
                    continue
                
                # Update frame buffer for FPS calculation
                frame_time_buffer.append(time.time())
                if len(frame_time_buffer) > 30:  # Keep last 30 frame times
                    frame_time_buffer.pop(0)
                
                # Calculate current FPS
                if len(frame_time_buffer) > 1:
                    time_span = frame_time_buffer[-1] - frame_time_buffer[0]
                    if time_span > 0:
                        self.current_fps = (len(frame_time_buffer) - 1) / time_span
                
                # Store frame with thread safety
                with self.frame_lock:
                    self.current_frame = frame.copy()
                    self.frame_count += 1
                
                # Try to add to queue (non-blocking)
                try:
                    self.frame_queue.put_nowait((True, frame, time.time()))
                except queue.Full:
                    # Queue is full, drop oldest frame
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait((True, frame, time.time()))
                        self.dropped_frames += 1
                    except queue.Empty:
                        pass
                
                # Adaptive resolution scaling check (every 2 seconds)
                current_time = time.time()
                if current_time - last_adaptive_check > 2.0:
                    self._check_adaptive_scaling()
                    last_adaptive_check = current_time
                
                # Frame rate limiting
                frame_duration = time.time() - frame_start_time
                target_duration = 1.0 / self.config.fps
                if frame_duration < target_duration:
                    time.sleep(target_duration - frame_duration)
                
            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)  # Brief pause before retry
        
        self.logger.info("Capture loop stopped")
    
    def _check_adaptive_scaling(self):
        """
        Check if resolution scaling is needed based on current performance.
        
        Scales resolution down if FPS is too low, or up if performance allows.
        """
        if self.current_fps < self.fps_threshold_low and self.current_resolution != self.min_resolution:
            # Scale down resolution
            new_width = int(self.current_resolution[0] * self.resolution_scale_factor)
            new_height = int(self.current_resolution[1] * self.resolution_scale_factor)
            
            # Ensure minimum resolution
            new_width = max(new_width, self.min_resolution[0])
            new_height = max(new_height, self.min_resolution[1])
            
            if (new_width, new_height) != self.current_resolution:
                self.logger.info(f"Scaling down resolution: {self.current_resolution} -> ({new_width}, {new_height})")
                self._update_resolution((new_width, new_height))
        
        elif (self.current_fps > self.fps_threshold_high and 
              self.current_resolution != self.original_resolution):
            # Scale up resolution
            new_width = int(self.current_resolution[0] / self.resolution_scale_factor)
            new_height = int(self.current_resolution[1] / self.resolution_scale_factor)
            
            # Don't exceed original resolution
            new_width = min(new_width, self.original_resolution[0])
            new_height = min(new_height, self.original_resolution[1])
            
            if (new_width, new_height) != self.current_resolution:
                self.logger.info(f"Scaling up resolution: {self.current_resolution} -> ({new_width}, {new_height})")
                self._update_resolution((new_width, new_height))
    
    def _update_resolution(self, new_resolution: Tuple[int, int]):
        """
        Update camera resolution dynamically.
        
        Args:
            new_resolution: New (width, height) resolution
        """
        try:
            if self.camera_type == 'usb':
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, new_resolution[0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, new_resolution[1])
            
            self.current_resolution = new_resolution
            self.logger.info(f"Resolution updated to {new_resolution}")
            
        except Exception as e:
            self.logger.error(f"Failed to update resolution: {e}")
    
    def start_capture(self) -> None:
        """
        Start the camera capture process.
        
        Initializes the camera hardware and begins threaded frame capture.
        
        Raises:
            RuntimeError: If camera cannot be initialized
        """
        if self.is_capturing:
            self.logger.warning("Capture already started")
            return
        
        if not self._initialize_camera():
            raise RuntimeError("Failed to initialize camera")
        
        # Reset state
        self.should_stop = False
        self.frame_count = 0
        self.dropped_frames = 0
        self.start_time = time.time()
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        self.is_capturing = True
        self.logger.info("Camera capture started")
    
    def get_frame(self) -> Tuple[bool, np.ndarray]:
        """
        Get the next frame from the camera.
        
        Returns the most recent frame from the camera buffer. Non-blocking
        operation that returns immediately.
        
        Returns:
            Tuple of (success, frame) where:
            - success (bool): True if frame was captured successfully
            - frame (np.ndarray): Frame data as BGR image array, or None if failed
        """
        if not self.is_capturing:
            return False, None
        
        # Try to get frame from queue first (most recent)
        try:
            success, frame, timestamp = self.frame_queue.get_nowait()
            return success, frame
        except queue.Empty:
            pass
        
        # Fall back to current frame
        with self.frame_lock:
            if self.current_frame is not None:
                return True, self.current_frame.copy()
        
        return False, None
    
    def adjust_resolution(self, target_fps: float) -> None:
        """
        Adjust camera resolution based on target FPS requirements.
        
        Dynamically adjusts camera resolution to maintain target frame rate.
        This is crucial for performance optimization on resource-constrained devices.
        
        Args:
            target_fps (float): Desired frames per second
        """
        if not self.is_capturing:
            self.logger.warning("Cannot adjust resolution - capture not started")
            return
        
        self.logger.info(f"Adjusting resolution for target FPS: {target_fps}")
        
        # Update FPS thresholds based on target
        self.fps_threshold_low = target_fps * 0.7  # 70% of target
        self.fps_threshold_high = target_fps * 0.9  # 90% of target
        
        # Immediate check for resolution adjustment
        self._check_adaptive_scaling()
    
    def cleanup(self) -> None:
        """
        Clean up camera resources and release hardware.
        
        Properly releases camera hardware, closes connections, and frees
        system resources. Safe to call multiple times.
        """
        self.logger.info("Cleaning up camera resources")
        
        # Stop capture thread
        if self.is_capturing:
            self.should_stop = True
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2.0)
                if self.capture_thread.is_alive():
                    self.logger.warning("Capture thread did not stop gracefully")
        
        # Release camera
        if self.camera:
            try:
                self.camera.release()
                self.logger.info("Camera released")
            except Exception as e:
                self.logger.error(f"Error releasing camera: {e}")
            finally:
                self.camera = None
        
        # Clear frame buffer
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        # Reset state
        self.is_capturing = False
        self.current_frame = None
        
        self.logger.info("Camera cleanup completed")
    
    def get_camera_info(self) -> Dict[str, Any]:
        """
        Get camera information and current settings.
        
        Returns comprehensive information about the camera including
        capabilities, current settings, and performance statistics.
        
        Returns:
            Dictionary containing camera information
        """
        info = {
            'resolution': self.current_resolution,
            'original_resolution': self.original_resolution,
            'fps': self.config.fps,
            'current_fps': self.current_fps,
            'format': 'BGR',
            'device_name': f"{self.camera_type}_{self.device_id}" if self.device_id is not None else "unknown",
            'camera_type': self.camera_type,
            'device_id': self.device_id,
            'is_capturing': self.is_capturing,
            'frame_count': self.frame_count,
            'dropped_frames': self.dropped_frames,
            'status': 'active' if self.is_capturing else 'inactive'
        }
        
        # Add performance statistics if capturing
        if self.is_capturing and self.start_time:
            runtime = time.time() - self.start_time
            info.update({
                'runtime_seconds': runtime,
                'average_fps': self.frame_count / runtime if runtime > 0 else 0,
                'drop_rate': self.dropped_frames / max(1, self.frame_count + self.dropped_frames)
            })
        
        # Add camera capabilities if available
        if self.camera and self.camera.isOpened():
            try:
                capabilities = {
                    'max_width': int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'max_height': int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'max_fps': int(self.camera.get(cv2.CAP_PROP_FPS)),
                }
                info['capabilities'] = capabilities
            except Exception as e:
                self.logger.debug(f"Could not get camera capabilities: {e}")
        
        return info
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()