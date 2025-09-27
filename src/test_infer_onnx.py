#!/usr/bin/env python3
"""
Raspberry Pi 4 ONNX Inference Test with Camera Module 2
Optimized for performance and reliability on RPi4 hardware
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
import argparse
import sys
import os
from pathlib import Path
import threading
import queue
import signal
from contextlib import contextmanager

class RaspberryPiONNXInference:
    def __init__(self, model_path="models/yolov8n-sim.onnx", imgsz=416, 
                 camera_width=640, camera_height=480, camera_fps=30, 
                 display_scale=1.0, use_threading=True):
        """
        Initialize the Raspberry Pi ONNX inference system
        
        Args:
            model_path: Path to ONNX model file
            imgsz: Input size for model inference
            camera_width: Camera capture width
            camera_height: Camera capture height  
            camera_fps: Camera framerate
            display_scale: Scale factor for display window
            use_threading: Whether to use threaded frame capture
        """
        self.model_path = model_path
        self.imgsz = imgsz
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.display_scale = display_scale
        self.use_threading = use_threading
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_history = []
        self.inference_times = []
        
        # Threading components
        self.frame_queue = queue.Queue(maxsize=2) if use_threading else None
        self.capture_thread = None
        self.running = False
        
        # Initialize components
        self.sess = None
        self.cap = None
        self.input_name = None
        
    def setup_model(self):
        """Load and configure the ONNX model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        print(f"Loading ONNX model: {self.model_path}")
        
        # Optimize for RPi4: Use CPU with optimizations
        providers = [
            ('CPUExecutionProvider', {
                'enable_cpu_mem_arena': True,
                'cpu_mem_arena_cfg': '{"initial_chunk_size_bytes": 1048576, "max_dead_bytes_per_chunk": 134217728, "memory_limit_bytes": 268435456}',
                'enable_memory_pattern': True,
                'enable_memory_arena_shrinkage': True,
            })
        ]
        
        try:
            self.sess = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.sess.get_inputs()[0].name
            input_shape = self.sess.get_inputs()[0].shape
            print(f"Model loaded successfully. Input shape: {input_shape}")
            print(f"Input name: {self.input_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    def setup_camera(self):
        """Initialize camera with multiple fallback options"""
        camera_configs = [
            # Pi Camera Module 2 with libcamera (preferred for newer Pi OS)
            {
                'pipeline': f"libcamerasrc ! video/x-raw,width={self.camera_width},height={self.camera_height},framerate={self.camera_fps}/1,format=BGR ! videoconvert ! appsink drop=1 max-buffers=1",
                'backend': cv2.CAP_GSTREAMER,
                'name': 'Pi Camera (libcamera)'
            },
            # Pi Camera with legacy driver
            {
                'pipeline': f"v4l2src device=/dev/video0 ! video/x-raw,width={self.camera_width},height={self.camera_height},framerate={self.camera_fps}/1 ! videoconvert ! appsink drop=1 max-buffers=1",
                'backend': cv2.CAP_GSTREAMER,
                'name': 'Pi Camera (v4l2)'
            },
            # Direct V4L2 access
            {
                'pipeline': 0,
                'backend': cv2.CAP_V4L2,
                'name': 'Direct V4L2'
            },
            # Fallback to any available camera
            {
                'pipeline': 0,
                'backend': cv2.CAP_ANY,
                'name': 'Default camera'
            }
        ]
        
        for config in camera_configs:
            try:
                print(f"Trying {config['name']}...")
                self.cap = cv2.VideoCapture(config['pipeline'], config['backend'])
                
                if self.cap.isOpened():
                    # Test if we can actually read frames
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print(f"Successfully initialized {config['name']}")
                        
                        # Configure camera properties if using direct access
                        if isinstance(config['pipeline'], int):
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
                            self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        # Verify final settings
                        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                        
                        print(f"Camera settings: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
                        return
                    else:
                        self.cap.release()
                        
            except Exception as e:
                print(f"Failed to initialize {config['name']}: {e}")
                if self.cap:
                    self.cap.release()
                continue
        
        raise RuntimeError("Could not initialize any camera configuration")
    
    def preprocess_frame(self, bgr_frame):
        """Optimized preprocessing for RPi4"""
        # Resize efficiently
        if bgr_frame.shape[:2] != (self.imgsz, self.imgsz):
            resized = cv2.resize(bgr_frame, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        else:
            resized = bgr_frame
            
        # Convert BGR to RGB and normalize in one step
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Transpose and normalize efficiently
        normalized = rgb_frame.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
        
        return input_tensor
    
    def capture_frames(self):
        """Threaded frame capture to improve performance"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    # Drop oldest frame if queue is full
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame)
                    except queue.Empty:
                        pass
            else:
                time.sleep(0.01)  # Brief pause if read fails
    
    def get_frame(self):
        """Get frame either from thread queue or direct capture"""
        if self.use_threading:
            try:
                return True, self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                return False, None
        else:
            return self.cap.read()
    
    def update_statistics(self, inference_time):
        """Update performance statistics"""
        self.frame_count += 1
        self.inference_times.append(inference_time)
        
        # Calculate FPS over recent frames
        current_time = time.time()
        if len(self.inference_times) >= 10:
            recent_avg_inference = np.mean(self.inference_times[-10:])
            display_fps = 1.0 / recent_avg_inference if recent_avg_inference > 0 else 0
            self.fps_history.append(display_fps)
            
            if len(self.fps_history) > 30:  # Keep last 30 FPS measurements
                self.fps_history.pop(0)
        
        # Print periodic statistics
        if self.frame_count % 100 == 0:
            elapsed = current_time - self.start_time
            avg_fps = self.frame_count / elapsed
            avg_inference = np.mean(self.inference_times) * 1000  # ms
            print(f"Stats - Frames: {self.frame_count}, Avg FPS: {avg_fps:.1f}, Avg Inference: {avg_inference:.1f}ms")
    
    def draw_overlay(self, frame, inference_time):
        """Draw performance overlay on frame"""
        if len(self.fps_history) > 0:
            current_fps = self.fps_history[-1]
            avg_fps = np.mean(self.fps_history)
        else:
            current_fps = 1.0 / inference_time if inference_time > 0 else 0
            avg_fps = current_fps
        
        # Scale text size based on display scale
        font_scale = 0.7 * self.display_scale
        thickness = max(1, int(2 * self.display_scale))
        
        # Draw semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw performance text
        cv2.putText(frame, f"FPS: {current_fps:.1f} (avg: {avg_fps:.1f})", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        cv2.putText(frame, f"Inference: {inference_time*1000:.1f}ms", 
                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        cv2.putText(frame, f"Frames: {self.frame_count}", 
                   (20, 95), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def run(self):
        """Main inference loop"""
        print("Starting inference loop...")
        print("Press 'q', 'ESC', or Ctrl+C to quit")
        
        self.running = True
        
        # Start capture thread if using threading
        if self.use_threading:
            self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
            self.capture_thread.start()
        
        try:
            while self.running:
                # Get frame
                ret, frame = self.get_frame()
                if not ret or frame is None:
                    print("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Preprocess frame
                input_tensor = self.preprocess_frame(frame)
                
                # Run inference
                start_time = time.time()
                try:
                    outputs = self.sess.run(None, {self.input_name: input_tensor})
                    inference_time = time.time() - start_time
                except Exception as e:
                    print(f"Inference error: {e}")
                    continue
                
                # Update statistics
                self.update_statistics(inference_time)
                
                # Prepare display frame
                if self.display_scale != 1.0:
                    display_height = int(frame.shape[0] * self.display_scale)
                    display_width = int(frame.shape[1] * self.display_scale)
                    display_frame = cv2.resize(frame, (display_width, display_height))
                else:
                    display_frame = frame
                
                # Draw overlay
                display_frame = self.draw_overlay(display_frame, inference_time)
                
                # Show frame
                cv2.imshow("Raspberry Pi ONNX Inference", display_frame)
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Print final statistics
        if self.frame_count > 0:
            elapsed = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed
            avg_inference = np.mean(self.inference_times) * 1000
            print(f"\nFinal Statistics:")
            print(f"Total frames: {self.frame_count}")
            print(f"Total time: {elapsed:.1f}s")
            print(f"Average FPS: {avg_fps:.1f}")
            print(f"Average inference time: {avg_inference:.1f}ms")


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\nReceived interrupt signal...")
    sys.exit(0)


def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description="Raspberry Pi 4 ONNX Inference Test")
    parser.add_argument("--model", default="models/yolov8n-sim.onnx", 
                       help="Path to ONNX model file")
    parser.add_argument("--imgsz", type=int, default=416, 
                       help="Input image size for model")
    parser.add_argument("--camera-width", type=int, default=640, 
                       help="Camera capture width")
    parser.add_argument("--camera-height", type=int, default=480, 
                       help="Camera capture height")
    parser.add_argument("--camera-fps", type=int, default=30, 
                       help="Camera framerate")
    parser.add_argument("--display-scale", type=float, default=1.0, 
                       help="Display window scale factor")
    parser.add_argument("--no-threading", action="store_true", 
                       help="Disable threaded frame capture")
    
    args = parser.parse_args()
    
    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Create inference system
        inference_system = RaspberryPiONNXInference(
            model_path=args.model,
            imgsz=args.imgsz,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_fps=args.camera_fps,
            display_scale=args.display_scale,
            use_threading=not args.no_threading
        )
        
        # Setup components
        inference_system.setup_model()
        inference_system.setup_camera()
        
        # Run inference
        inference_system.run()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

