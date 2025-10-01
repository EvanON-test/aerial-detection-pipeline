"""
Model Inference Engine with TensorRT optimization for aerial object detection.

This module provides a high-performance inference engine optimized for edge deployment
on Jetson Nano hardware. It supports multiple model formats (ONNX, TensorRT) with
automatic optimization and fallback mechanisms for robust operation under resource
constraints.

Key Features:
- ONNX and TensorRT model support with automatic conversion
- TensorRT engine caching for faster startup times
- GPU memory monitoring with CPU fallback
- Performance benchmarking and optimization
- Dynamic model switching for different operational modes
- Preprocessing and postprocessing pipelines optimized for aerial objects

The engine is designed to maximize performance on Jetson Nano 2GB while maintaining
reliability through comprehensive error handling and resource management.

Example Usage:
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

import os
import time
import logging
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from datetime import datetime

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNXRuntime not available. ONNX model support disabled.")

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logging.warning("TensorRT not available. TensorRT optimization disabled.")

import cv2

from ..models.interfaces import ModelInferenceInterface
from ..models.data_models import Detection, ModelConfig


class ModelInferenceEngine(ModelInferenceInterface):
    """
    High-performance model inference engine with TensorRT optimization.
    
    Provides unified interface for running inference on ONNX and TensorRT models
    with automatic optimization, caching, and fallback mechanisms. Designed for
    edge deployment on resource-constrained hardware like Jetson Nano.
    
    Attributes:
        config: Model configuration parameters
        current_model: Currently loaded model identifier
        models: Dictionary of loaded models
        performance_stats: Performance metrics for each model
        tensorrt_cache_dir: Directory for TensorRT engine caching
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the model inference engine.
        
        Args:
            config: Model configuration containing paths, thresholds, and settings
        """
        self.config = config
        self.current_model: Optional[str] = None
        self.models: Dict[str, Any] = {}
        self.performance_stats: Dict[str, Dict[str, float]] = {}
        self.tensorrt_cache_dir = Path("cache/tensorrt_engines")
        self.tensorrt_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.inference_times: List[float] = []
        self.preprocessing_times: List[float] = []
        self.postprocessing_times: List[float] = []
        
        # GPU memory monitoring
        self.gpu_memory_limit = self._get_gpu_memory_limit()
        
        self.logger.info(f"ModelInferenceEngine initialized with config: {config}")
        self.logger.info(f"ONNX available: {ONNX_AVAILABLE}, TensorRT available: {TENSORRT_AVAILABLE}")
    
    def load_model(self, model_path: str, optimize: bool = True) -> None:
        """
        Load a model for inference with optional TensorRT optimization.
        
        Supports ONNX models with automatic TensorRT conversion and caching.
        Falls back to CPU execution if GPU memory is insufficient.
        
        Args:
            model_path: Path to the model file (.onnx or .trt)
            optimize: Whether to optimize with TensorRT (if available)
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_name = os.path.basename(model_path)
        self.logger.info(f"Loading model: {model_name}")
        
        try:
            # Check if TensorRT engine exists in cache
            if optimize and TENSORRT_AVAILABLE and model_path.endswith('.onnx'):
                trt_engine_path = self._get_tensorrt_cache_path(model_path)
                if os.path.exists(trt_engine_path):
                    self.logger.info(f"Loading cached TensorRT engine: {trt_engine_path}")
                    model = self._load_tensorrt_engine(trt_engine_path)
                else:
                    self.logger.info("Creating TensorRT engine from ONNX model")
                    model = self._create_tensorrt_engine(model_path, trt_engine_path)
                    if model is None:
                        self.logger.warning("TensorRT engine creation failed, falling back to ONNX")
                        model = self._load_onnx_model(model_path, use_gpu=True)
            elif model_path.endswith('.onnx'):
                model = self._load_onnx_model(model_path, use_gpu=True)
            elif model_path.endswith('.trt'):
                model = self._load_tensorrt_engine(model_path)
            else:
                raise ValueError(f"Unsupported model format: {model_path}")
            
            if model is not None:
                self.models[model_name] = model
                self.current_model = model_name
                self.performance_stats[model_name] = {
                    'load_time': time.time(),
                    'inference_count': 0,
                    'avg_inference_time': 0.0,
                    'avg_preprocessing_time': 0.0,
                    'avg_postprocessing_time': 0.0
                }
                self.logger.info(f"Successfully loaded model: {model_name}")
            else:
                raise RuntimeError(f"Failed to load model: {model_path}")
                
        except Exception as e:
            self.logger.error(f"Error loading model {model_path}: {str(e)}")
            # Try fallback model if primary model fails
            if hasattr(self.config, 'fallback_model') and model_path != self.config.fallback_model:
                self.logger.info(f"Attempting to load fallback model: {self.config.fallback_model}")
                self.load_model(self.config.fallback_model, optimize=False)
            else:
                raise
    
    def infer(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a frame and return detections.
        
        Performs complete inference pipeline including preprocessing,
        model inference, and postprocessing with performance tracking.
        
        Args:
            frame: Input frame as BGR numpy array
            
        Returns:
            List of Detection objects
            
        Raises:
            RuntimeError: If no model is loaded or inference fails
        """
        if self.current_model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Preprocessing
        preprocess_start = time.time()
        preprocessed_frame = self.preprocess_frame(frame)
        preprocess_time = time.time() - preprocess_start
        self.preprocessing_times.append(preprocess_time)
        
        # Inference
        inference_start = time.time()
        try:
            model = self.models[self.current_model]
            if isinstance(model, dict) and 'type' in model:
                if model['type'] == 'onnx':
                    raw_output = self._run_onnx_inference(model, preprocessed_frame)
                elif model['type'] == 'tensorrt':
                    raw_output = self._run_tensorrt_inference(model, preprocessed_frame)
                else:
                    raise ValueError(f"Unknown model type: {model['type']}")
            else:
                raise ValueError("Invalid model format")
        except Exception as e:
            self.logger.error(f"Inference failed: {str(e)}")
            # Try CPU fallback if GPU inference fails
            if hasattr(self.config, 'fallback_model'):
                self.logger.info("Attempting CPU fallback")
                return self._cpu_fallback_inference(frame)
            else:
                raise RuntimeError(f"Inference failed: {str(e)}")
        
        inference_time = time.time() - inference_start
        self.inference_times.append(inference_time)
        
        # Postprocessing
        postprocess_start = time.time()
        detections = self.postprocess_detections(raw_output)
        postprocess_time = time.time() - postprocess_start
        self.postprocessing_times.append(postprocess_time)
        
        # Update performance stats
        total_time = time.time() - start_time
        self._update_performance_stats(preprocess_time, inference_time, postprocess_time)
        
        self.logger.debug(f"Inference completed in {total_time:.3f}s, found {len(detections)} detections")
        return detections
    
    def switch_model(self, model_name: str) -> None:
        """
        Switch to a different loaded model.
        
        Args:
            model_name: Name of the model to switch to
            
        Raises:
            ValueError: If model is not loaded
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded. Available models: {list(self.models.keys())}")
        
        self.current_model = model_name
        self.logger.info(f"Switched to model: {model_name}")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics for the current model.
        
        Returns:
            Dictionary containing performance metrics
        """
        if self.current_model is None:
            return {}
        
        stats = self.performance_stats.get(self.current_model, {}).copy()
        
        # Add recent performance metrics
        if self.inference_times:
            stats['recent_avg_inference_time'] = np.mean(self.inference_times[-100:])
            stats['recent_max_inference_time'] = np.max(self.inference_times[-100:])
            stats['recent_min_inference_time'] = np.min(self.inference_times[-100:])
        
        if self.preprocessing_times:
            stats['recent_avg_preprocessing_time'] = np.mean(self.preprocessing_times[-100:])
        
        if self.postprocessing_times:
            stats['recent_avg_postprocessing_time'] = np.mean(self.postprocessing_times[-100:])
        
        # Calculate FPS
        if 'recent_avg_inference_time' in stats and stats['recent_avg_inference_time'] > 0:
            total_time = (stats.get('recent_avg_preprocessing_time', 0) + 
                         stats['recent_avg_inference_time'] + 
                         stats.get('recent_avg_postprocessing_time', 0))
            stats['estimated_fps'] = 1.0 / total_time if total_time > 0 else 0.0
        
        return stats
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for model input.
        
        Performs standard preprocessing including resizing, normalization,
        and format conversion optimized for aerial object detection models.
        
        Args:
            frame: Input frame as BGR numpy array
            
        Returns:
            Preprocessed frame ready for model inference
        """
        # Resize to model input size
        input_size = getattr(self.config, 'input_size', (640, 640))
        resized = cv2.resize(frame, input_size)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb_frame.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to NCHW format
        preprocessed = np.transpose(normalized, (2, 0, 1))  # HWC to CHW
        preprocessed = np.expand_dims(preprocessed, axis=0)  # Add batch dimension
        
        return preprocessed
    
    def postprocess_detections(self, raw_output: Any) -> List[Detection]:
        """
        Postprocess raw model output to Detection objects.
        
        Converts raw model predictions to structured Detection objects with
        confidence filtering, NMS, and coordinate conversion.
        
        Args:
            raw_output: Raw model output (format depends on model type)
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        try:
            # Handle different output formats
            if isinstance(raw_output, (list, tuple)):
                predictions = raw_output[0] if len(raw_output) > 0 else None
            else:
                predictions = raw_output
            
            if predictions is None:
                return detections
            
            # Convert to numpy array if needed
            if hasattr(predictions, 'numpy'):
                predictions = predictions.numpy()
            elif not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)
            
            # Handle different prediction formats (YOLOv8 style)
            if len(predictions.shape) == 3:
                predictions = predictions[0]  # Remove batch dimension
            
            if len(predictions.shape) == 2 and predictions.shape[1] >= 6:
                # YOLOv8 format: [x_center, y_center, width, height, confidence, class_scores...]
                for pred in predictions:
                    if len(pred) < 6:
                        continue
                    
                    x_center, y_center, width, height = pred[:4]
                    confidence = pred[4]
                    class_scores = pred[5:]
                    
                    # Filter by confidence threshold
                    if confidence < self.config.confidence_threshold:
                        continue
                    
                    # Get class with highest score
                    class_id = int(np.argmax(class_scores))
                    class_confidence = class_scores[class_id]
                    
                    # Combined confidence
                    final_confidence = confidence * class_confidence
                    
                    if final_confidence < self.config.confidence_threshold:
                        continue
                    
                    # Convert center format to corner format
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    
                    # Scale coordinates back to original image size
                    input_size = getattr(self.config, 'input_size', (640, 640))
                    # Note: This assumes square input. For production, store original dimensions
                    
                    detection = Detection(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=float(final_confidence),
                        class_id=class_id,
                        class_name=self._get_class_name(class_id),
                        timestamp=datetime.now(),
                        frame_id=0  # Will be set by caller
                    )
                    detections.append(detection)
            
            # Apply Non-Maximum Suppression
            detections = self._apply_nms(detections)
            
            # Limit number of detections
            max_detections = getattr(self.config, 'max_detections', 100)
            if len(detections) > max_detections:
                # Sort by confidence and keep top detections
                detections.sort(key=lambda d: d.confidence, reverse=True)
                detections = detections[:max_detections]
            
        except Exception as e:
            self.logger.error(f"Error in postprocessing: {str(e)}")
            # Return empty list on error rather than crashing
        
        return detections
    
    def _get_gpu_memory_limit(self) -> float:
        """Get GPU memory limit in GB."""
        try:
            if TENSORRT_AVAILABLE:
                cuda.init()
                device = cuda.Device(0)
                context = device.make_context()
                free_mem, total_mem = cuda.mem_get_info()
                context.pop()
                return total_mem / (1024**3)  # Convert to GB
        except:
            pass
        return 2.0  # Default for Jetson Nano 2GB
    
    def _get_tensorrt_cache_path(self, onnx_path: str) -> str:
        """Generate TensorRT engine cache path based on ONNX model."""
        # Create hash of model file for unique cache name
        with open(onnx_path, 'rb') as f:
            model_hash = hashlib.md5(f.read()).hexdigest()[:8]
        
        model_name = os.path.splitext(os.path.basename(onnx_path))[0]
        cache_name = f"{model_name}_{model_hash}.trt"
        return str(self.tensorrt_cache_dir / cache_name)
    
    def _create_tensorrt_engine(self, onnx_path: str, engine_path: str) -> Optional[Dict[str, Any]]:
        """Create TensorRT engine from ONNX model."""
        if not TENSORRT_AVAILABLE:
            return None
        
        try:
            self.logger.info(f"Creating TensorRT engine from {onnx_path}")
            
            # TensorRT logger
            trt_logger = trt.Logger(trt.Logger.WARNING)
            
            # Create builder and network
            builder = trt.Builder(trt_logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, trt_logger)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    self.logger.error("Failed to parse ONNX model")
                    return None
            
            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = min(1 << 30, int(self.gpu_memory_limit * 0.5 * 1024**3))  # Use half available GPU memory
            
            # Enable FP16 precision if supported
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                self.logger.info("Enabled FP16 precision")
            
            # Build engine
            self.logger.info("Building TensorRT engine (this may take several minutes)...")
            engine = builder.build_engine(network, config)
            
            if engine is None:
                self.logger.error("Failed to build TensorRT engine")
                return None
            
            # Save engine to cache
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            
            self.logger.info(f"TensorRT engine saved to {engine_path}")
            
            # Create execution context
            context = engine.create_execution_context()
            
            return {
                'type': 'tensorrt',
                'engine': engine,
                'context': context,
                'input_shape': self._get_tensorrt_input_shape(engine),
                'output_shapes': self._get_tensorrt_output_shapes(engine)
            }
            
        except Exception as e:
            self.logger.error(f"TensorRT engine creation failed: {str(e)}")
            return None
    
    def _load_tensorrt_engine(self, engine_path: str) -> Optional[Dict[str, Any]]:
        """Load TensorRT engine from file."""
        if not TENSORRT_AVAILABLE:
            return None
        
        try:
            trt_logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(trt_logger)
            
            with open(engine_path, 'rb') as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            
            if engine is None:
                return None
            
            context = engine.create_execution_context()
            
            return {
                'type': 'tensorrt',
                'engine': engine,
                'context': context,
                'input_shape': self._get_tensorrt_input_shape(engine),
                'output_shapes': self._get_tensorrt_output_shapes(engine)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load TensorRT engine: {str(e)}")
            return None
    
    def _load_onnx_model(self, model_path: str, use_gpu: bool = True) -> Optional[Dict[str, Any]]:
        """Load ONNX model with ONNXRuntime."""
        if not ONNX_AVAILABLE:
            return None
        
        try:
            providers = []
            if use_gpu and ort.get_available_providers():
                # Try CUDA provider first, then CPU
                available_providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available_providers:
                    providers.append('CUDAExecutionProvider')
                providers.append('CPUExecutionProvider')
            else:
                providers = ['CPUExecutionProvider']
            
            session = ort.InferenceSession(model_path, providers=providers)
            
            return {
                'type': 'onnx',
                'session': session,
                'input_name': session.get_inputs()[0].name,
                'output_names': [output.name for output in session.get_outputs()]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load ONNX model: {str(e)}")
            return None
    
    def _run_onnx_inference(self, model: Dict[str, Any], input_data: np.ndarray) -> Any:
        """Run inference with ONNX model."""
        session = model['session']
        input_name = model['input_name']
        
        return session.run(None, {input_name: input_data})
    
    def _run_tensorrt_inference(self, model: Dict[str, Any], input_data: np.ndarray) -> Any:
        """Run inference with TensorRT engine."""
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")
        
        engine = model['engine']
        context = model['context']
        
        # Allocate GPU memory
        input_shape = model['input_shape']
        output_shapes = model['output_shapes']
        
        # Allocate device memory
        d_input = cuda.mem_alloc(input_data.nbytes)
        d_outputs = []
        h_outputs = []
        
        for output_shape in output_shapes:
            output_size = np.prod(output_shape) * np.dtype(np.float32).itemsize
            d_output = cuda.mem_alloc(output_size)
            h_output = np.empty(output_shape, dtype=np.float32)
            d_outputs.append(d_output)
            h_outputs.append(h_output)
        
        # Copy input to device
        cuda.memcpy_htod(d_input, input_data)
        
        # Run inference
        bindings = [int(d_input)] + [int(d_output) for d_output in d_outputs]
        context.execute_v2(bindings)
        
        # Copy outputs back to host
        for i, (d_output, h_output) in enumerate(zip(d_outputs, h_outputs)):
            cuda.memcpy_dtoh(h_output, d_output)
        
        return h_outputs
    
    def _get_tensorrt_input_shape(self, engine) -> Tuple[int, ...]:
        """Get input shape from TensorRT engine."""
        return tuple(engine.get_binding_shape(0))
    
    def _get_tensorrt_output_shapes(self, engine) -> List[Tuple[int, ...]]:
        """Get output shapes from TensorRT engine."""
        shapes = []
        for i in range(1, engine.num_bindings):  # Skip input binding
            shapes.append(tuple(engine.get_binding_shape(i)))
        return shapes
    
    def _cpu_fallback_inference(self, frame: np.ndarray) -> List[Detection]:
        """Fallback to CPU inference when GPU fails."""
        try:
            if hasattr(self.config, 'fallback_model') and os.path.exists(self.config.fallback_model):
                # Load CPU model if not already loaded
                fallback_name = os.path.basename(self.config.fallback_model)
                if fallback_name not in self.models:
                    cpu_model = self._load_onnx_model(self.config.fallback_model, use_gpu=False)
                    if cpu_model:
                        self.models[fallback_name] = cpu_model
                
                if fallback_name in self.models:
                    original_model = self.current_model
                    self.current_model = fallback_name
                    
                    # Run inference with CPU model
                    preprocessed = self.preprocess_frame(frame)
                    raw_output = self._run_onnx_inference(self.models[fallback_name], preprocessed)
                    detections = self.postprocess_detections(raw_output)
                    
                    # Restore original model
                    self.current_model = original_model
                    
                    self.logger.warning("Used CPU fallback for inference")
                    return detections
        except Exception as e:
            self.logger.error(f"CPU fallback failed: {str(e)}")
        
        return []  # Return empty list if all fallbacks fail
    
    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply Non-Maximum Suppression to detections."""
        if len(detections) <= 1:
            return detections
        
        # Convert to format expected by cv2.dnn.NMSBoxes
        boxes = []
        confidences = []
        class_ids = []
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            boxes.append([x1, y1, x2 - x1, y2 - y1])  # Convert to x, y, w, h
            confidences.append(det.confidence)
            class_ids.append(det.class_id)
        
        # Apply NMS
        nms_threshold = getattr(self.config, 'nms_threshold', 0.4)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.config.confidence_threshold, nms_threshold)
        
        # Return filtered detections
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        else:
            return []
    
    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID."""
        # Default aerial object classes - should be loaded from config in production
        class_names = {
            0: "drone",
            1: "aircraft",
            2: "helicopter",
            3: "bird",
            4: "balloon",
            5: "unknown"
        }
        return class_names.get(class_id, "unknown")
    
    def _update_performance_stats(self, preprocess_time: float, inference_time: float, postprocess_time: float) -> None:
        """Update performance statistics."""
        if self.current_model in self.performance_stats:
            stats = self.performance_stats[self.current_model]
            stats['inference_count'] += 1
            
            # Update running averages
            count = stats['inference_count']
            stats['avg_preprocessing_time'] = ((stats['avg_preprocessing_time'] * (count - 1)) + preprocess_time) / count
            stats['avg_inference_time'] = ((stats['avg_inference_time'] * (count - 1)) + inference_time) / count
            stats['avg_postprocessing_time'] = ((stats['avg_postprocessing_time'] * (count - 1)) + postprocess_time) / count
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("Cleaning up ModelInferenceEngine resources")
        
        # Clear models
        self.models.clear()
        self.current_model = None
        
        # Clear performance data
        self.inference_times.clear()
        self.preprocessing_times.clear()
        self.postprocessing_times.clear()
        
        self.logger.info("ModelInferenceEngine cleanup completed")