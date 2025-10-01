#!/usr/bin/env python3
"""
Integration test for the complete ModelInferenceEngine implementation.
"""

import os
import sys
from pathlib import Path

# Set up Python path to import from src
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Now we can import with absolute imports
import numpy as np
from models.data_models import ModelConfig
from models.interfaces import ModelInferenceInterface

def test_interface_compliance():
    """Test that ModelInferenceEngine implements the required interface."""
    print("Testing interface compliance...")
    
    # Import the engine
    from detection.model_inference_engine import ModelInferenceEngine
    
    # Check if it's a subclass of the interface
    assert issubclass(ModelInferenceEngine, ModelInferenceInterface)
    print("✓ ModelInferenceEngine implements ModelInferenceInterface")
    
    # Create instance
    config = ModelConfig(
        primary_model="models/yolov8n.onnx",
        fallback_model="models/yolov8n.onnx",
        confidence_threshold=0.5
    )
    
    engine = ModelInferenceEngine(config)
    print("✓ ModelInferenceEngine can be instantiated")
    
    # Check required methods exist
    required_methods = [
        'load_model', 'infer', 'switch_model', 'get_performance_stats',
        'preprocess_frame', 'postprocess_detections'
    ]
    
    for method in required_methods:
        assert hasattr(engine, method), f"Missing method: {method}"
        assert callable(getattr(engine, method)), f"Method not callable: {method}"
    
    print(f"✓ All required methods present: {', '.join(required_methods)}")
    
    # Test basic functionality
    test_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    # Test preprocessing
    preprocessed = engine.preprocess_frame(test_frame)
    assert preprocessed.shape == (1, 3, 640, 640), f"Unexpected preprocessed shape: {preprocessed.shape}"
    print("✓ Preprocessing works correctly")
    
    # Test postprocessing
    dummy_output = np.random.rand(1, 25200, 85)
    detections = engine.postprocess_detections(dummy_output)
    print(f"✓ Postprocessing works: {len(detections)} detections")
    
    # Test performance stats
    stats = engine.get_performance_stats()
    assert isinstance(stats, dict), "Performance stats should be a dictionary"
    print("✓ Performance stats accessible")
    
    # Test error handling for inference without loaded model
    try:
        engine.infer(test_frame)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        print("✓ Proper error handling for missing model")
    
    # Cleanup
    engine.cleanup()
    print("✓ Cleanup completed")
    
    return True

def test_benchmark_import():
    """Test that PerformanceBenchmark can be imported and used."""
    print("\nTesting PerformanceBenchmark...")
    
    from detection.performance_benchmark import PerformanceBenchmark, BenchmarkResult
    
    # Create benchmark instance
    benchmark = PerformanceBenchmark()
    print("✓ PerformanceBenchmark can be instantiated")
    
    # Test system info
    system_info = benchmark.system_info
    assert hasattr(system_info, 'cpu_model')
    assert hasattr(system_info, 'total_memory')
    print(f"✓ System info available: {system_info.cpu_model}")
    
    # Test frame creation
    test_frames = benchmark.create_test_frames(count=5)
    assert len(test_frames) == 5
    assert test_frames[0].shape == (640, 640, 3)
    print("✓ Test frame creation works")
    
    return True

def main():
    """Run integration tests."""
    print("Running ModelInferenceEngine integration tests...\n")
    
    try:
        # Test interface compliance
        test_interface_compliance()
        
        # Test benchmark functionality
        test_benchmark_import()
        
        print(f"\n🎉 All integration tests passed!")
        print("ModelInferenceEngine is ready for use!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())