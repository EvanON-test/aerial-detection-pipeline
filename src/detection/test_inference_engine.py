"""
Test script for ModelInferenceEngine functionality.

This script provides basic testing and validation of the ModelInferenceEngine
implementation, including model loading, inference, and performance monitoring.
"""

import sys
import logging
import numpy as np
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.data_models import ModelConfig
from src.detection.model_inference_engine import ModelInferenceEngine
from src.detection.performance_benchmark import PerformanceBenchmark

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_frame(width: int = 640, height: int = 640) -> np.ndarray:
    """Create a synthetic test frame."""
    # Create random frame
    frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    # Add some rectangles to simulate objects
    import cv2
    cv2.rectangle(frame, (100, 100), (200, 150), (255, 0, 0), -1)
    cv2.rectangle(frame, (300, 200), (400, 250), (0, 255, 0), -1)
    
    return frame


def test_model_inference_engine():
    """Test basic ModelInferenceEngine functionality."""
    logger.info("Testing ModelInferenceEngine...")
    
    # Create test configuration
    config = ModelConfig(
        primary_model="models/yolov8n.onnx",  # This may not exist
        fallback_model="models/yolov8n.onnx",
        confidence_threshold=0.5,
        nms_threshold=0.4,
        max_detections=100,
        input_size=(640, 640)
    )
    
    # Initialize engine
    engine = ModelInferenceEngine(config)
    
    # Test without loading a model (should fail gracefully)
    test_frame = create_test_frame()
    
    try:
        detections = engine.infer(test_frame)
        logger.error("Expected RuntimeError for no loaded model")
    except RuntimeError as e:
        logger.info(f"Correctly caught error: {e}")
    
    # Test preprocessing
    preprocessed = engine.preprocess_frame(test_frame)
    logger.info(f"Preprocessed frame shape: {preprocessed.shape}")
    
    # Test postprocessing with dummy data
    dummy_output = np.random.rand(1, 25200, 85)  # YOLOv8 format
    detections = engine.postprocess_detections(dummy_output)
    logger.info(f"Postprocessed {len(detections)} detections from dummy data")
    
    # Test performance stats (should be empty)
    stats = engine.get_performance_stats()
    logger.info(f"Performance stats: {stats}")
    
    # Cleanup
    engine.cleanup()
    
    logger.info("ModelInferenceEngine basic tests completed")


def test_performance_benchmark():
    """Test PerformanceBenchmark functionality."""
    logger.info("Testing PerformanceBenchmark...")
    
    # Initialize benchmark
    benchmark = PerformanceBenchmark()
    
    # Create test frames
    test_frames = benchmark.create_test_frames(count=10)
    logger.info(f"Created {len(test_frames)} test frames")
    
    # Test system info
    logger.info(f"System info: {benchmark.system_info}")
    
    # Test resource monitor
    from src.detection.performance_benchmark import ResourceMonitor
    monitor = ResourceMonitor()
    
    monitor.start_monitoring(interval=0.1)
    import time
    time.sleep(1.0)  # Monitor for 1 second
    monitor.stop_monitoring()
    
    stats = monitor.get_stats()
    logger.info(f"Resource monitoring stats: {stats}")
    
    logger.info("PerformanceBenchmark basic tests completed")


def main():
    """Run all tests."""
    logger.info("Starting ModelInferenceEngine tests...")
    
    try:
        test_model_inference_engine()
        test_performance_benchmark()
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())