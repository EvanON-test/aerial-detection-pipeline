"""
Detection module for aerial object detection and tracking system.

This module provides the core inference engine and performance benchmarking
utilities for running object detection models optimized for edge deployment.

Key Components:
- ModelInferenceEngine: High-performance inference with TensorRT optimization
- PerformanceBenchmark: Comprehensive benchmarking and model comparison tools

The module is designed for deployment on resource-constrained hardware like
Jetson Nano while maintaining high performance and reliability.

Features:
- ONNX and TensorRT model support with automatic conversion
- TensorRT engine caching for faster startup times
- GPU memory monitoring with CPU fallback
- Performance benchmarking and optimization
- Dynamic model switching for different operational modes
- Preprocessing and postprocessing pipelines optimized for aerial objects

Usage:
    from src.detection import ModelInferenceEngine
    from src.models.data_models import ModelConfig
    
    config = ModelConfig(
        primary_model="models/yolov8n.onnx",
        fallback_model="models/yolov8n-cpu.onnx",
        confidence_threshold=0.5
    )
    
    engine = ModelInferenceEngine(config)
    engine.load_model("models/yolov8n.onnx", optimize=True)
    
    detections = engine.infer(frame)
    stats = engine.get_performance_stats()
"""

from .model_inference_engine import ModelInferenceEngine
from .performance_benchmark import PerformanceBenchmark, BenchmarkResult, run_quick_benchmark

__all__ = [
    'ModelInferenceEngine',
    'PerformanceBenchmark', 
    'BenchmarkResult',
    'run_quick_benchmark'
]