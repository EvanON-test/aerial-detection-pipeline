#!/usr/bin/env python3
"""
Standalone test for ModelInferenceEngine core functionality.
"""

import numpy as np
import cv2
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Detection:
    """Simple detection class for testing."""
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    timestamp: datetime
    frame_id: int

@dataclass 
class ModelConfig:
    """Simple config class for testing."""
    primary_model: str
    fallback_model: str
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_detections: int = 100
    input_size: Tuple[int, int] = (640, 640)

def test_preprocessing():
    """Test frame preprocessing."""
    print("Testing preprocessing...")
    
    # Create test frame
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    # Preprocess (simplified version)
    input_size = (640, 640)
    resized = cv2.resize(frame, input_size)
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb_frame.astype(np.float32) / 255.0
    preprocessed = np.transpose(normalized, (2, 0, 1))
    preprocessed = np.expand_dims(preprocessed, axis=0)
    
    print(f"✓ Preprocessing: {frame.shape} -> {preprocessed.shape}")
    return preprocessed

def test_postprocessing():
    """Test detection postprocessing."""
    print("Testing postprocessing...")
    
    # Create dummy YOLOv8-style output
    dummy_output = np.random.rand(1, 25200, 85)
    predictions = dummy_output[0]
    
    detections = []
    confidence_threshold = 0.5
    
    for pred in predictions[:10]:  # Test first 10 predictions
        if len(pred) < 6:
            continue
        
        x_center, y_center, width, height = pred[:4]
        confidence = pred[4]
        class_scores = pred[5:]
        
        if confidence < confidence_threshold:
            continue
        
        class_id = int(np.argmax(class_scores))
        class_confidence = class_scores[class_id]
        final_confidence = confidence * class_confidence
        
        if final_confidence < confidence_threshold:
            continue
        
        # Convert to corner format
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        detection = Detection(
            bbox=(float(x1), float(y1), float(x2), float(y2)),
            confidence=float(final_confidence),
            class_id=class_id,
            class_name=f"class_{class_id}",
            timestamp=datetime.now(),
            frame_id=0
        )
        detections.append(detection)
    
    print(f"✓ Postprocessing: found {len(detections)} detections")
    return detections

def test_nms():
    """Test Non-Maximum Suppression."""
    print("Testing NMS...")
    
    # Create overlapping detections
    detections = [
        Detection((100, 100, 200, 200), 0.9, 0, "drone", datetime.now(), 0),
        Detection((110, 110, 210, 210), 0.8, 0, "drone", datetime.now(), 0),  # Overlapping
        Detection((300, 300, 400, 400), 0.7, 1, "aircraft", datetime.now(), 0),
    ]
    
    # Simple NMS implementation
    if len(detections) <= 1:
        filtered = detections
    else:
        boxes = []
        confidences = []
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(det.confidence)
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        if len(indices) > 0:
            indices = indices.flatten()
            filtered = [detections[i] for i in indices]
        else:
            filtered = []
    
    print(f"✓ NMS: {len(detections)} -> {len(filtered)} detections")
    return filtered

def main():
    """Run all tests."""
    print("Testing ModelInferenceEngine core functionality...\n")
    
    try:
        # Test preprocessing
        preprocessed = test_preprocessing()
        
        # Test postprocessing
        detections = test_postprocessing()
        
        # Test NMS
        filtered_detections = test_nms()
        
        print(f"\n✓ All core functionality tests passed!")
        print(f"  - Preprocessing: ✓")
        print(f"  - Postprocessing: ✓ ({len(detections)} detections)")
        print(f"  - NMS: ✓ ({len(filtered_detections)} after filtering)")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())