#!/usr/bin/env python3
"""
Example usage of the ModelInferenceEngine for aerial object detection.

This example demonstrates how to use the ModelInferenceEngine with a real
ONNX model for aerial object detection, including TensorRT optimization
and performance monitoring.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to Python path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def create_sample_frame():
    """Create a sample aerial frame for testing."""
    # Create a realistic aerial scene
    frame = np.random.randint(50, 200, (720, 1280, 3), dtype=np.uint8)
    
    # Add some sky gradient (lighter at top)
    for y in range(frame.shape[0]):
        brightness = int(255 * (1 - y / frame.shape[0]) * 0.3)
        frame[y, :, :] = np.clip(frame[y, :, :] + brightness, 0, 255)
    
    return frame

def main():
    """Demonstrate ModelInferenceEngine usage."""
    print("ModelInferenceEngine Usage Example")
    print("=" * 40)
    
    # Import after path setup
    from models.data_models import ModelConfig
    
    # Check if we have a real model file
    model_path = "models/yolov8n.onnx"
    if not Path(model_path).exists():
        print(f"⚠️  Model file not found: {model_path}")
        print("This example requires a YOLOv8 ONNX model.")
        print("You can download one from: https://github.com/ultralytics/ultralytics")
        print("\nFor now, we'll demonstrate the interface without actual inference.")
        
        # Create a dummy model path for demonstration
        model_path = "dummy_model.onnx"
    
    # Create configuration
    config = ModelConfig(
        primary_model=model_path,
        fallback_model=model_path,
        confidence_threshold=0.5,
        nms_threshold=0.4,
        max_detections=50,
        input_size=(640, 640)
    )
    
    print(f"Configuration:")
    print(f"  Model: {config.primary_model}")
    print(f"  Confidence threshold: {config.confidence_threshold}")
    print(f"  Input size: {config.input_size}")
    
    # Initialize the inference engine
    try:
        # Import here to avoid issues if dependencies are missing
        from detection.model_inference_engine import ModelInferenceEngine
        
        engine = ModelInferenceEngine(config)
        print(f"\n✓ ModelInferenceEngine initialized")
        
        # Create a sample frame
        frame = create_sample_frame()
        print(f"✓ Created sample frame: {frame.shape}")
        
        # Demonstrate preprocessing
        preprocessed = engine.preprocess_frame(frame)
        print(f"✓ Preprocessed frame: {frame.shape} -> {preprocessed.shape}")
        
        # Demonstrate postprocessing with dummy data
        dummy_output = np.random.rand(1, 25200, 85)  # YOLOv8 output format
        detections = engine.postprocess_detections(dummy_output)
        print(f"✓ Postprocessed dummy output: {len(detections)} detections")
        
        # Show detection details
        if detections:
            print(f"\nSample detections:")
            for i, det in enumerate(detections[:3]):  # Show first 3
                print(f"  {i+1}. {det.class_name} (conf: {det.confidence:.3f}) "
                      f"bbox: ({det.bbox[0]:.1f}, {det.bbox[1]:.1f}, "
                      f"{det.bbox[2]:.1f}, {det.bbox[3]:.1f})")
        
        # Get performance stats
        stats = engine.get_performance_stats()
        print(f"\n✓ Performance stats available: {len(stats)} metrics")
        
        # Demonstrate error handling
        try:
            # This should fail since no real model is loaded
            detections = engine.infer(frame)
            print(f"✓ Inference successful: {len(detections)} detections")
        except RuntimeError as e:
            print(f"✓ Expected error (no model loaded): {str(e)[:50]}...")
        
        # Cleanup
        engine.cleanup()
        print(f"✓ Engine cleanup completed")
        
        print(f"\n🎉 ModelInferenceEngine demonstration completed successfully!")
        
        # Show what would happen with a real model
        print(f"\nWith a real ONNX model, you would:")
        print(f"1. engine.load_model('{model_path}', optimize=True)")
        print(f"2. detections = engine.infer(frame)")
        print(f"3. Process detections for tracking and threat assessment")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed.")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())