"""
Utility functions and helpers for the aerial detection system.

This module provides common utility functions, helper classes, and shared
functionality used across the detection pipeline. It includes mathematical
operations, image processing utilities, performance monitoring, and
system-specific optimizations.

Planned Components:
- **Image Processing**: Frame manipulation, resizing, color conversion
- **Mathematical Utilities**: Geometric calculations, distance metrics
- **Performance Monitoring**: FPS calculation, memory usage, profiling
- **Hardware Detection**: System capabilities, optimization selection
- **Logging Utilities**: Structured logging, performance metrics
- **Visualization Helpers**: Drawing utilities, overlay generation
- **File Management**: Model loading, configuration handling
- **System Optimization**: Platform-specific performance tuning

Key Utilities:
- Bounding box operations (IoU, area, overlap)
- Coordinate transformations and projections
- Performance benchmarking and profiling
- Memory management and optimization
- Hardware capability detection
- Logging and debugging helpers

Common Functions (planned):
- calculate_iou(box1, box2): Intersection over Union calculation
- resize_frame(frame, target_size): Optimized frame resizing
- draw_detections(frame, detections): Visualization overlay
- monitor_performance(): System performance tracking
- optimize_for_hardware(): Platform-specific optimizations

Usage (planned):
    from src.utils import calculate_iou, resize_frame
    from src.utils.performance import PerformanceMonitor
    
    iou = calculate_iou(detection1.bbox, detection2.bbox)
    resized = resize_frame(frame, (640, 480))
    
    monitor = PerformanceMonitor()
    with monitor.time_operation("inference"):
        results = model.infer(frame)

Note: This module is currently under development. Utility functions
will be implemented as needed by other system components.
"""

# Module is under development
# Utilities will be added as needed by other components

__all__ = []