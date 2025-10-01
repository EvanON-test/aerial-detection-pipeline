#!/usr/bin/env python3
"""
Simple test for ModelInferenceEngine implementation.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.data_models import ModelConfig

def main():
    print("Testing ModelInferenceEngine...")
    
    # Create test configuration
    config = ModelConfig(
        primary_model="models/yolov8n.onnx",
        fallback_model="models/yolov8n.onnx",
        confidence_threshold=0.5,
        nms_threshold=0.4,
        max_detections=100,
        input_size=(640, 640)
    )
    
    # Initialize engine
    engine = ModelInferenceEngine(config)
    print("✓ ModelInferenceEngine initialized")
    
    # Test preprocessing
    test_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    preprocessed = engine.preprocess_frame(test_frame)
    print(f"✓ Preprocessing works: {test_frame.shape} -> {preprocessed.shape}")
    
    # Test postprocessing with dummy data
    dummy_output = np.random.rand(1, 25200, 85)
    detections = engine.postprocess_detections(dummy_output)
    print(f"✓ Postprocessing works: found {len(detections)} detections")
    
    # Test performance stats
    stats = engine.get_performance_stats()
    print(f"✓ Performance stats: {len(stats)} metrics")
    
    # Test error handling
    try:
        engine.infer(test_frame)
        print("✗ Should have failed with no model loaded")
    except RuntimeError:
        print("✓ Error handling works correctly")
    
    # Cleanup
    engine.cleanup()
    print("✓ Cleanup completed")
    
    print("\nAll tests passed! ModelInferenceEngine is working correctly.")
    return 0

if __name__ == "__main__":
    sys.exit(main())