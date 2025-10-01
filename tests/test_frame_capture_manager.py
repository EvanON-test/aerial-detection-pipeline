"""
Unit tests for FrameCaptureManager class.

Tests camera interface functionality, frame management, adaptive resolution
scaling, and error handling for the aerial object detection system.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import threading
import time
import queue
from datetime import datetime

from src.capture.frame_capture_manager import FrameCaptureManager
from src.models.data_models import CameraConfig


class TestFrameCaptureManager(unittest.TestCase):
    """Test cases for FrameCaptureManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.camera_config = CameraConfig(
            resolution=(1920, 1080),
            fps=30,
            exposure_mode="auto",
            focus_mode="auto",
            brightness=50,
            contrast=0
        )
        self.capture_manager = FrameCaptureManager(self.camera_config)
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self.capture_manager, 'cleanup'):
            self.capture_manager.cleanup()
    
    def test_initialization(self):
        """Test FrameCaptureManager initialization."""
        self.assertEqual(self.capture_manager.config, self.camera_config)
        self.assertIsNone(self.capture_manager.camera)
        self.assertFalse(self.capture_manager.is_capturing)
        self.assertEqual(self.capture_manager.frame_count, 0)
        self.assertEqual(self.capture_manager.dropped_frames, 0)
        self.assertEqual(self.capture_manager.current_resolution, (1920, 1080))
        self.assertEqual(self.capture_manager.original_resolution, (1920, 1080))
    
    @patch('cv2.VideoCapture')
    def test_detect_camera_type_csi(self, mock_video_capture):
        """Test CSI camera detection."""
        # Mock successful CSI camera detection
        mock_camera = Mock()
        mock_camera.isOpened.return_value = True
        mock_camera.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_camera
        
        camera_type, device_id = self.capture_manager._detect_camera_type()
        
        self.assertEqual(camera_type, 'csi')
        self.assertEqual(device_id, 0)
        mock_video_capture.assert_called_with(0)
    
    @patch('cv2.VideoCapture')
    def test_detect_camera_type_usb(self, mock_video_capture):
        """Test USB camera detection."""
        # Mock failed CSI, successful USB camera detection
        def mock_video_capture_side_effect(device_id):
            mock_camera = Mock()
            if device_id == 0:
                # First call (CSI detection) fails
                mock_camera.isOpened.return_value = False
                return mock_camera
            elif device_id == 1:
                # Second call (USB detection) succeeds
                mock_camera.isOpened.return_value = True
                mock_camera.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
                return mock_camera
            else:
                mock_camera.isOpened.return_value = False
                return mock_camera
        
        mock_video_capture.side_effect = mock_video_capture_side_effect
        
        camera_type, device_id = self.capture_manager._detect_camera_type()
        
        self.assertEqual(camera_type, 'usb')
        self.assertEqual(device_id, 1)
    
    @patch('cv2.VideoCapture')
    def test_detect_camera_type_none(self, mock_video_capture):
        """Test no camera detection."""
        # Mock no camera found
        mock_camera = Mock()
        mock_camera.isOpened.return_value = False
        mock_video_capture.return_value = mock_camera
        
        camera_type, device_id = self.capture_manager._detect_camera_type()
        
        self.assertIsNone(camera_type)
        self.assertIsNone(device_id)
    
    @patch('cv2.VideoCapture')
    def test_initialize_camera_success(self, mock_video_capture):
        """Test successful camera initialization."""
        # Mock successful camera initialization
        mock_camera = Mock()
        mock_camera.isOpened.return_value = True
        mock_camera.read.return_value = (True, np.zeros((1080, 1920, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_camera
        
        with patch.object(self.capture_manager, '_detect_camera_type', return_value=('usb', 0)):
            with patch.object(self.capture_manager, '_configure_camera_properties'):
                result = self.capture_manager._initialize_camera()
        
        self.assertTrue(result)
        self.assertEqual(self.capture_manager.camera_type, 'usb')
        self.assertEqual(self.capture_manager.device_id, 0)
        self.assertIsNotNone(self.capture_manager.camera)
    
    @patch('cv2.VideoCapture')
    def test_initialize_camera_failure(self, mock_video_capture):
        """Test camera initialization failure."""
        with patch.object(self.capture_manager, '_detect_camera_type', return_value=(None, None)):
            result = self.capture_manager._initialize_camera()
        
        self.assertFalse(result)
        self.assertIsNone(self.capture_manager.camera_type)
        self.assertIsNone(self.capture_manager.device_id)
    
    def test_configure_camera_properties_usb(self):
        """Test USB camera property configuration."""
        # Mock USB camera
        mock_camera = Mock()
        self.capture_manager.camera = mock_camera
        self.capture_manager.camera_type = 'usb'
        
        self.capture_manager._configure_camera_properties()
        
        # Verify camera properties were set
        mock_camera.set.assert_any_call(3, 1920)  # CAP_PROP_FRAME_WIDTH
        mock_camera.set.assert_any_call(4, 1080)  # CAP_PROP_FRAME_HEIGHT
        mock_camera.set.assert_any_call(5, 30)    # CAP_PROP_FPS
        mock_camera.set.assert_any_call(10, 50)   # CAP_PROP_BRIGHTNESS
        mock_camera.set.assert_any_call(11, 0)    # CAP_PROP_CONTRAST
    
    @patch('cv2.VideoCapture')
    def test_start_capture_success(self, mock_video_capture):
        """Test successful capture start."""
        # Mock successful camera initialization
        mock_camera = Mock()
        mock_camera.isOpened.return_value = True
        mock_camera.read.return_value = (True, np.zeros((1080, 1920, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_camera
        
        with patch.object(self.capture_manager, '_detect_camera_type', return_value=('usb', 0)):
            with patch.object(self.capture_manager, '_configure_camera_properties'):
                self.capture_manager.start_capture()
        
        self.assertTrue(self.capture_manager.is_capturing)
        self.assertIsNotNone(self.capture_manager.capture_thread)
        self.assertTrue(self.capture_manager.capture_thread.is_alive())
        
        # Clean up
        self.capture_manager.cleanup()
    
    def test_start_capture_failure(self):
        """Test capture start failure."""
        with patch.object(self.capture_manager, '_initialize_camera', return_value=False):
            with self.assertRaises(RuntimeError):
                self.capture_manager.start_capture()
        
        self.assertFalse(self.capture_manager.is_capturing)
    
    def test_get_frame_not_capturing(self):
        """Test get_frame when not capturing."""
        success, frame = self.capture_manager.get_frame()
        
        self.assertFalse(success)
        self.assertIsNone(frame)
    
    def test_get_frame_with_current_frame(self):
        """Test get_frame with current frame available."""
        # Set up capturing state
        self.capture_manager.is_capturing = True
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.capture_manager.current_frame = test_frame
        
        success, frame = self.capture_manager.get_frame()
        
        self.assertTrue(success)
        self.assertIsNotNone(frame)
        np.testing.assert_array_equal(frame, test_frame)
    
    def test_get_frame_from_queue(self):
        """Test get_frame from queue."""
        # Set up capturing state
        self.capture_manager.is_capturing = True
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add frame to queue
        self.capture_manager.frame_queue.put_nowait((True, test_frame, time.time()))
        
        success, frame = self.capture_manager.get_frame()
        
        self.assertTrue(success)
        self.assertIsNotNone(frame)
        np.testing.assert_array_equal(frame, test_frame)
    
    def test_adjust_resolution(self):
        """Test resolution adjustment."""
        self.capture_manager.is_capturing = True
        
        with patch.object(self.capture_manager, '_check_adaptive_scaling') as mock_check:
            self.capture_manager.adjust_resolution(15.0)
        
        # Verify thresholds were updated
        self.assertEqual(self.capture_manager.fps_threshold_low, 10.5)  # 15 * 0.7
        self.assertEqual(self.capture_manager.fps_threshold_high, 13.5)  # 15 * 0.9
        mock_check.assert_called_once()
    
    def test_adjust_resolution_not_capturing(self):
        """Test resolution adjustment when not capturing."""
        with patch.object(self.capture_manager, '_check_adaptive_scaling') as mock_check:
            self.capture_manager.adjust_resolution(15.0)
        
        mock_check.assert_not_called()
    
    def test_check_adaptive_scaling_down(self):
        """Test adaptive scaling down when FPS is low."""
        self.capture_manager.current_fps = 8.0  # Below threshold
        self.capture_manager.fps_threshold_low = 10.0
        self.capture_manager.current_resolution = (1920, 1080)
        self.capture_manager.min_resolution = (640, 480)
        
        with patch.object(self.capture_manager, '_update_resolution') as mock_update:
            self.capture_manager._check_adaptive_scaling()
        
        # Should scale down
        expected_width = int(1920 * 0.8)  # 1536
        expected_height = int(1080 * 0.8)  # 864
        mock_update.assert_called_once_with((expected_width, expected_height))
    
    def test_check_adaptive_scaling_up(self):
        """Test adaptive scaling up when FPS is high."""
        self.capture_manager.current_fps = 28.0  # Above threshold
        self.capture_manager.fps_threshold_high = 25.0
        self.capture_manager.current_resolution = (1536, 864)  # Scaled down
        self.capture_manager.original_resolution = (1920, 1080)
        
        with patch.object(self.capture_manager, '_update_resolution') as mock_update:
            self.capture_manager._check_adaptive_scaling()
        
        # Should scale up
        expected_width = int(1536 / 0.8)  # 1920
        expected_height = int(864 / 0.8)   # 1080
        mock_update.assert_called_once_with((expected_width, expected_height))
    
    def test_check_adaptive_scaling_no_change(self):
        """Test adaptive scaling with no change needed."""
        self.capture_manager.current_fps = 20.0  # Within acceptable range
        self.capture_manager.fps_threshold_low = 10.0
        self.capture_manager.fps_threshold_high = 25.0
        
        with patch.object(self.capture_manager, '_update_resolution') as mock_update:
            self.capture_manager._check_adaptive_scaling()
        
        mock_update.assert_not_called()
    
    def test_update_resolution_usb(self):
        """Test resolution update for USB camera."""
        mock_camera = Mock()
        self.capture_manager.camera = mock_camera
        self.capture_manager.camera_type = 'usb'
        
        self.capture_manager._update_resolution((1280, 720))
        
        mock_camera.set.assert_any_call(3, 1280)  # CAP_PROP_FRAME_WIDTH
        mock_camera.set.assert_any_call(4, 720)   # CAP_PROP_FRAME_HEIGHT
        self.assertEqual(self.capture_manager.current_resolution, (1280, 720))
    
    def test_cleanup(self):
        """Test cleanup functionality."""
        # Set up some state
        mock_camera = Mock()
        self.capture_manager.is_capturing = True
        self.capture_manager.should_stop = False
        self.capture_manager.camera = mock_camera
        self.capture_manager.current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some frames to queue
        self.capture_manager.frame_queue.put_nowait((True, np.zeros((480, 640, 3)), time.time()))
        
        self.capture_manager.cleanup()
        
        # Verify cleanup
        self.assertFalse(self.capture_manager.is_capturing)
        self.assertIsNone(self.capture_manager.camera)
        self.assertIsNone(self.capture_manager.current_frame)
        self.assertTrue(self.capture_manager.frame_queue.empty())
        mock_camera.release.assert_called_once()
    
    def test_get_camera_info_inactive(self):
        """Test camera info when inactive."""
        info = self.capture_manager.get_camera_info()
        
        expected_keys = [
            'resolution', 'original_resolution', 'fps', 'current_fps',
            'format', 'device_name', 'camera_type', 'device_id',
            'is_capturing', 'frame_count', 'dropped_frames', 'status'
        ]
        
        for key in expected_keys:
            self.assertIn(key, info)
        
        self.assertEqual(info['status'], 'inactive')
        self.assertFalse(info['is_capturing'])
        self.assertEqual(info['format'], 'BGR')
    
    def test_get_camera_info_active(self):
        """Test camera info when active."""
        # Set up active state
        self.capture_manager.is_capturing = True
        self.capture_manager.start_time = time.time() - 10  # 10 seconds ago
        self.capture_manager.frame_count = 300
        self.capture_manager.dropped_frames = 5
        self.capture_manager.current_fps = 25.0
        self.capture_manager.camera_type = 'usb'
        self.capture_manager.device_id = 0
        
        info = self.capture_manager.get_camera_info()
        
        self.assertEqual(info['status'], 'active')
        self.assertTrue(info['is_capturing'])
        self.assertEqual(info['frame_count'], 300)
        self.assertEqual(info['dropped_frames'], 5)
        self.assertEqual(info['current_fps'], 25.0)
        self.assertIn('runtime_seconds', info)
        self.assertIn('average_fps', info)
        self.assertIn('drop_rate', info)
    
    def test_capture_loop_integration(self):
        """Integration test for capture loop functionality."""
        # Mock camera for integration test
        mock_camera = Mock()
        test_frames = [
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.ones((480, 640, 3), dtype=np.uint8) * 128,
            np.ones((480, 640, 3), dtype=np.uint8) * 255
        ]
        
        frame_iter = iter(test_frames * 10)  # Repeat frames
        mock_camera.read.side_effect = lambda: (True, next(frame_iter))
        
        self.capture_manager.camera = mock_camera
        self.capture_manager.is_capturing = True
        self.capture_manager.should_stop = False
        
        # Start capture loop in thread
        capture_thread = threading.Thread(target=self.capture_manager._capture_loop, daemon=True)
        capture_thread.start()
        
        # Let it run for a short time
        time.sleep(0.5)
        
        # Stop capture
        self.capture_manager.should_stop = True
        capture_thread.join(timeout=2.0)
        
        # Verify frames were captured
        self.assertGreater(self.capture_manager.frame_count, 0)
        self.assertIsNotNone(self.capture_manager.current_frame)
    
    def test_thread_safety(self):
        """Test thread safety of frame access."""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.capture_manager.current_frame = test_frame
        self.capture_manager.is_capturing = True
        
        # Simulate concurrent access
        results = []
        
        def access_frame():
            success, frame = self.capture_manager.get_frame()
            results.append((success, frame is not None))
        
        threads = [threading.Thread(target=access_frame) for _ in range(10)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All accesses should succeed
        self.assertEqual(len(results), 10)
        for success, has_frame in results:
            self.assertTrue(success)
            self.assertTrue(has_frame)


class TestCameraConfigValidation(unittest.TestCase):
    """Test camera configuration validation."""
    
    def test_valid_config(self):
        """Test valid camera configuration."""
        config = CameraConfig(
            resolution=(1920, 1080),
            fps=30,
            exposure_mode="auto",
            focus_mode="auto"
        )
        
        self.assertEqual(config.resolution, (1920, 1080))
        self.assertEqual(config.fps, 30)
        self.assertEqual(config.exposure_mode, "auto")
        self.assertEqual(config.focus_mode, "auto")
    
    def test_invalid_fps(self):
        """Test invalid FPS configuration."""
        with self.assertRaises(ValueError):
            CameraConfig(
                resolution=(1920, 1080),
                fps=0  # Invalid FPS
            )
    
    def test_invalid_resolution(self):
        """Test invalid resolution configuration."""
        with self.assertRaises(ValueError):
            CameraConfig(
                resolution=(0, 1080),  # Invalid width
                fps=30
            )
        
        with self.assertRaises(ValueError):
            CameraConfig(
                resolution=(1920, 0),  # Invalid height
                fps=30
            )


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    unittest.main()