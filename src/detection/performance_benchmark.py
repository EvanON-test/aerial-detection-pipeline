"""
Performance benchmarking utilities for model inference comparison.

This module provides comprehensive benchmarking tools for evaluating and comparing
the performance of different models and optimization strategies on Jetson Nano hardware.
It measures inference speed, memory usage, accuracy, and system resource utilization.

Key Features:
- Automated benchmarking of multiple models
- Memory usage monitoring (RAM and GPU)
- FPS and latency measurements
- Accuracy evaluation with test datasets
- Resource utilization tracking
- Comparative analysis and reporting
- Hardware-specific optimizations testing

The benchmarking suite is designed to help optimize model selection and configuration
for specific deployment scenarios and performance requirements.

Example Usage:
    benchmark = PerformanceBenchmark()
    
    # Add models to compare
    benchmark.add_model("yolov8n.onnx", optimize=True)
    benchmark.add_model("yolov8s.onnx", optimize=True)
    
    # Run benchmark
    results = benchmark.run_benchmark(test_frames, duration=60)
    
    # Generate report
    benchmark.generate_report(results, "benchmark_report.json")
"""

import time
import json
import logging
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

try:
    import GPUtil
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False
    logging.warning("GPUtil not available. GPU monitoring disabled.")

import cv2

from .model_inference_engine import ModelInferenceEngine
from ..models.data_models import ModelConfig, Detection


@dataclass
class BenchmarkResult:
    """Results from a single model benchmark run."""
    model_name: str
    model_path: str
    optimization_enabled: bool
    
    # Performance metrics
    avg_inference_time: float
    min_inference_time: float
    max_inference_time: float
    std_inference_time: float
    avg_fps: float
    
    # Resource usage
    avg_cpu_usage: float
    max_cpu_usage: float
    avg_memory_usage: float
    max_memory_usage: float
    
    # Detection metrics
    total_detections: int
    avg_detections_per_frame: float
    avg_confidence: float
    
    # System metrics
    total_frames: int
    benchmark_duration: float
    timestamp: str
    
    # Error tracking
    failed_inferences: int
    error_rate: float
    
    # GPU metrics (with defaults)
    avg_gpu_usage: float = 0.0
    max_gpu_usage: float = 0.0
    avg_gpu_memory: float = 0.0
    max_gpu_memory: float = 0.0


@dataclass
class SystemInfo:
    """System information for benchmark context."""
    cpu_model: str
    cpu_cores: int
    total_memory: float  # GB
    gpu_model: str = "Unknown"
    gpu_memory: float = 0.0  # GB
    jetson_model: str = "Unknown"
    cuda_version: str = "Unknown"
    tensorrt_version: str = "Unknown"


class ResourceMonitor:
    """Monitor system resource usage during benchmarking."""
    
    def __init__(self):
        self.monitoring = False
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        self.monitor_thread = None
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self, interval: float = 0.5):
        """Start resource monitoring in background thread."""
        self.monitoring = True
        self.cpu_usage.clear()
        self.memory_usage.clear()
        self.gpu_usage.clear()
        self.gpu_memory.clear()
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        self.logger.info("Resource monitoring stopped")
    
    def get_stats(self) -> Dict[str, float]:
        """Get resource usage statistics."""
        stats = {}
        
        if self.cpu_usage:
            stats['avg_cpu_usage'] = np.mean(self.cpu_usage)
            stats['max_cpu_usage'] = np.max(self.cpu_usage)
        
        if self.memory_usage:
            stats['avg_memory_usage'] = np.mean(self.memory_usage)
            stats['max_memory_usage'] = np.max(self.memory_usage)
        
        if self.gpu_usage:
            stats['avg_gpu_usage'] = np.mean(self.gpu_usage)
            stats['max_gpu_usage'] = np.max(self.gpu_usage)
        
        if self.gpu_memory:
            stats['avg_gpu_memory'] = np.mean(self.gpu_memory)
            stats['max_gpu_memory'] = np.max(self.gpu_memory)
        
        return stats
    
    def _monitor_loop(self, interval: float):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                # CPU and memory usage
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_info = psutil.virtual_memory()
                
                self.cpu_usage.append(cpu_percent)
                self.memory_usage.append(memory_info.percent)
                
                # GPU usage (if available)
                if GPU_MONITORING_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]  # Assume single GPU
                            self.gpu_usage.append(gpu.load * 100)
                            self.gpu_memory.append(gpu.memoryUtil * 100)
                    except Exception as e:
                        self.logger.debug(f"GPU monitoring error: {e}")
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                break


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking suite for model inference.
    
    Provides automated testing and comparison of different models, optimization
    strategies, and configurations to help optimize deployment performance.
    """
    
    def __init__(self, cache_dir: str = "cache/benchmark"):
        """
        Initialize the performance benchmark suite.
        
        Args:
            cache_dir: Directory for storing benchmark results and cached data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.models_to_test: List[Tuple[str, bool]] = []  # (model_path, optimize)
        self.resource_monitor = ResourceMonitor()
        
        # System information
        self.system_info = self._get_system_info()
        
        self.logger.info("PerformanceBenchmark initialized")
        self.logger.info(f"System: {self.system_info.cpu_model}, {self.system_info.total_memory:.1f}GB RAM")
    
    def add_model(self, model_path: str, optimize: bool = True):
        """
        Add a model to the benchmark suite.
        
        Args:
            model_path: Path to the model file
            optimize: Whether to enable TensorRT optimization
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.models_to_test.append((model_path, optimize))
        self.logger.info(f"Added model to benchmark: {model_path} (optimize={optimize})")
    
    def run_benchmark(self, 
                     test_frames: List[np.ndarray], 
                     duration: float = 60.0,
                     warmup_frames: int = 10) -> List[BenchmarkResult]:
        """
        Run comprehensive benchmark on all added models.
        
        Args:
            test_frames: List of test frames for inference
            duration: Benchmark duration in seconds
            warmup_frames: Number of warmup frames before measurement
            
        Returns:
            List of BenchmarkResult objects
        """
        if not self.models_to_test:
            raise ValueError("No models added for benchmarking")
        
        if not test_frames:
            raise ValueError("No test frames provided")
        
        results = []
        
        for model_path, optimize in self.models_to_test:
            self.logger.info(f"Benchmarking model: {model_path} (optimize={optimize})")
            
            try:
                result = self._benchmark_single_model(
                    model_path, optimize, test_frames, duration, warmup_frames
                )
                results.append(result)
                
                self.logger.info(f"Completed benchmark for {model_path}: "
                               f"{result.avg_fps:.1f} FPS, {result.avg_inference_time*1000:.1f}ms avg")
                
            except Exception as e:
                self.logger.error(f"Benchmark failed for {model_path}: {str(e)}")
                continue
        
        return results
    
    def _benchmark_single_model(self, 
                               model_path: str, 
                               optimize: bool,
                               test_frames: List[np.ndarray],
                               duration: float,
                               warmup_frames: int) -> BenchmarkResult:
        """Benchmark a single model."""
        model_name = Path(model_path).name
        
        # Create model configuration
        config = ModelConfig(
            primary_model=model_path,
            fallback_model=model_path,  # Use same model as fallback
            confidence_threshold=0.5,
            nms_threshold=0.4,
            max_detections=100
        )
        
        # Initialize inference engine
        engine = ModelInferenceEngine(config)
        engine.load_model(model_path, optimize=optimize)
        
        # Warmup
        self.logger.info(f"Warming up with {warmup_frames} frames...")
        for i in range(min(warmup_frames, len(test_frames))):
            frame = test_frames[i % len(test_frames)]
            try:
                engine.infer(frame)
            except Exception as e:
                self.logger.warning(f"Warmup frame {i} failed: {e}")
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        # Benchmark variables
        inference_times = []
        all_detections = []
        failed_inferences = 0
        frame_count = 0
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                frame = test_frames[frame_count % len(test_frames)]
                
                inference_start = time.time()
                try:
                    detections = engine.infer(frame)
                    inference_time = time.time() - inference_start
                    
                    inference_times.append(inference_time)
                    all_detections.extend(detections)
                    
                except Exception as e:
                    failed_inferences += 1
                    self.logger.debug(f"Inference failed: {e}")
                
                frame_count += 1
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.001)
        
        finally:
            # Stop resource monitoring
            self.resource_monitor.stop_monitoring()
            
            # Cleanup
            engine.cleanup()
        
        actual_duration = time.time() - start_time
        
        # Calculate metrics
        if inference_times:
            avg_inference_time = np.mean(inference_times)
            min_inference_time = np.min(inference_times)
            max_inference_time = np.max(inference_times)
            std_inference_time = np.std(inference_times)
            avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0
        else:
            avg_inference_time = min_inference_time = max_inference_time = std_inference_time = avg_fps = 0.0
        
        # Detection metrics
        total_detections = len(all_detections)
        avg_detections_per_frame = total_detections / frame_count if frame_count > 0 else 0.0
        avg_confidence = np.mean([d.confidence for d in all_detections]) if all_detections else 0.0
        
        # Resource usage
        resource_stats = self.resource_monitor.get_stats()
        
        # Error rate
        error_rate = failed_inferences / frame_count if frame_count > 0 else 0.0
        
        return BenchmarkResult(
            model_name=model_name,
            model_path=model_path,
            optimization_enabled=optimize,
            avg_inference_time=avg_inference_time,
            min_inference_time=min_inference_time,
            max_inference_time=max_inference_time,
            std_inference_time=std_inference_time,
            avg_fps=avg_fps,
            avg_cpu_usage=resource_stats.get('avg_cpu_usage', 0.0),
            max_cpu_usage=resource_stats.get('max_cpu_usage', 0.0),
            avg_memory_usage=resource_stats.get('avg_memory_usage', 0.0),
            max_memory_usage=resource_stats.get('max_memory_usage', 0.0),
            avg_gpu_usage=resource_stats.get('avg_gpu_usage', 0.0),
            max_gpu_usage=resource_stats.get('max_gpu_usage', 0.0),
            avg_gpu_memory=resource_stats.get('avg_gpu_memory', 0.0),
            max_gpu_memory=resource_stats.get('max_gpu_memory', 0.0),
            total_detections=total_detections,
            avg_detections_per_frame=avg_detections_per_frame,
            avg_confidence=avg_confidence,
            total_frames=frame_count,
            benchmark_duration=actual_duration,
            timestamp=datetime.now().isoformat(),
            failed_inferences=failed_inferences,
            error_rate=error_rate
        )
    
    def generate_report(self, results: List[BenchmarkResult], output_path: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive benchmark report.
        
        Args:
            results: List of benchmark results
            output_path: Optional path to save report JSON
            
        Returns:
            Dictionary containing the complete report
        """
        if not results:
            raise ValueError("No benchmark results provided")
        
        # Sort results by average FPS (descending)
        sorted_results = sorted(results, key=lambda r: r.avg_fps, reverse=True)
        
        report = {
            'benchmark_info': {
                'timestamp': datetime.now().isoformat(),
                'system_info': asdict(self.system_info),
                'total_models_tested': len(results)
            },
            'results': [asdict(result) for result in sorted_results],
            'summary': self._generate_summary(sorted_results),
            'recommendations': self._generate_recommendations(sorted_results)
        }
        
        # Save to file if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Benchmark report saved to {output_path}")
        
        return report
    
    def _generate_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate summary statistics from benchmark results."""
        if not results:
            return {}
        
        # Performance metrics
        fps_values = [r.avg_fps for r in results]
        inference_times = [r.avg_inference_time * 1000 for r in results]  # Convert to ms
        
        # Resource usage
        cpu_usage = [r.avg_cpu_usage for r in results]
        memory_usage = [r.avg_memory_usage for r in results]
        
        # Detection metrics
        detection_counts = [r.avg_detections_per_frame for r in results]
        confidence_scores = [r.avg_confidence for r in results]
        
        summary = {
            'performance': {
                'best_fps': max(fps_values),
                'worst_fps': min(fps_values),
                'avg_fps': np.mean(fps_values),
                'best_inference_time_ms': min(inference_times),
                'worst_inference_time_ms': max(inference_times),
                'avg_inference_time_ms': np.mean(inference_times)
            },
            'resource_usage': {
                'avg_cpu_usage': np.mean(cpu_usage),
                'max_cpu_usage': max(cpu_usage),
                'avg_memory_usage': np.mean(memory_usage),
                'max_memory_usage': max(memory_usage)
            },
            'detection_quality': {
                'avg_detections_per_frame': np.mean(detection_counts),
                'avg_confidence': np.mean(confidence_scores)
            },
            'optimization_impact': self._analyze_optimization_impact(results)
        }
        
        return summary
    
    def _analyze_optimization_impact(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze the impact of TensorRT optimization."""
        optimized = [r for r in results if r.optimization_enabled]
        unoptimized = [r for r in results if not r.optimization_enabled]
        
        if not optimized or not unoptimized:
            return {'analysis': 'Insufficient data for optimization comparison'}
        
        opt_fps = np.mean([r.avg_fps for r in optimized])
        unopt_fps = np.mean([r.avg_fps for r in unoptimized])
        
        opt_inference = np.mean([r.avg_inference_time for r in optimized])
        unopt_inference = np.mean([r.avg_inference_time for r in unoptimized])
        
        fps_improvement = ((opt_fps - unopt_fps) / unopt_fps * 100) if unopt_fps > 0 else 0
        inference_improvement = ((unopt_inference - opt_inference) / unopt_inference * 100) if unopt_inference > 0 else 0
        
        return {
            'optimized_avg_fps': opt_fps,
            'unoptimized_avg_fps': unopt_fps,
            'fps_improvement_percent': fps_improvement,
            'inference_time_improvement_percent': inference_improvement,
            'recommendation': 'Enable TensorRT optimization' if fps_improvement > 10 else 'Optimization impact minimal'
        }
    
    def _generate_recommendations(self, results: List[BenchmarkResult]) -> List[str]:
        """Generate recommendations based on benchmark results."""
        recommendations = []
        
        if not results:
            return recommendations
        
        best_result = results[0]  # Already sorted by FPS
        
        # Performance recommendations
        if best_result.avg_fps < 15:
            recommendations.append("Consider using a smaller model or reducing input resolution for better performance")
        
        if best_result.max_memory_usage > 80:
            recommendations.append("High memory usage detected. Consider model optimization or memory management")
        
        if best_result.error_rate > 0.05:
            recommendations.append("High error rate detected. Check model compatibility and system stability")
        
        # Optimization recommendations
        optimized_results = [r for r in results if r.optimization_enabled]
        if optimized_results and len(optimized_results) < len(results):
            recommendations.append("TensorRT optimization shows performance benefits. Enable for all models")
        
        # Model selection recommendations
        if len(results) > 1:
            recommendations.append(f"Best performing model: {best_result.model_name} "
                                 f"({best_result.avg_fps:.1f} FPS)")
        
        return recommendations
    
    def _get_system_info(self) -> SystemInfo:
        """Get system information for benchmark context."""
        try:
            # CPU information
            cpu_info = "Unknown"
            cpu_cores = psutil.cpu_count()
            
            # Try to get CPU model from /proc/cpuinfo (Linux)
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            cpu_info = line.split(':')[1].strip()
                            break
            except:
                pass
            
            # Memory information
            memory_info = psutil.virtual_memory()
            total_memory = memory_info.total / (1024**3)  # Convert to GB
            
            # GPU information
            gpu_model = "Unknown"
            gpu_memory = 0.0
            
            if GPU_MONITORING_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        gpu_model = gpu.name
                        gpu_memory = gpu.memoryTotal / 1024  # Convert MB to GB
                except:
                    pass
            
            # Jetson-specific information
            jetson_model = "Unknown"
            try:
                with open('/proc/device-tree/model', 'r') as f:
                    model_info = f.read().strip()
                    if 'jetson' in model_info.lower():
                        jetson_model = model_info
            except:
                pass
            
            return SystemInfo(
                cpu_model=cpu_info,
                cpu_cores=cpu_cores,
                total_memory=total_memory,
                gpu_model=gpu_model,
                gpu_memory=gpu_memory,
                jetson_model=jetson_model
            )
            
        except Exception as e:
            self.logger.error(f"Error getting system info: {e}")
            return SystemInfo(
                cpu_model="Unknown",
                cpu_cores=1,
                total_memory=0.0
            )
    
    def create_test_frames(self, width: int = 640, height: int = 640, count: int = 100) -> List[np.ndarray]:
        """
        Create synthetic test frames for benchmarking.
        
        Args:
            width: Frame width
            height: Frame height
            count: Number of frames to generate
            
        Returns:
            List of synthetic test frames
        """
        frames = []
        
        for i in range(count):
            # Create random frame with some structure
            frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            
            # Add some geometric shapes to simulate objects
            if i % 10 == 0:  # Every 10th frame has objects
                # Add rectangles to simulate aerial objects
                cv2.rectangle(frame, (50, 50), (100, 80), (255, 0, 0), -1)
                cv2.rectangle(frame, (200, 150), (250, 180), (0, 255, 0), -1)
            
            frames.append(frame)
        
        self.logger.info(f"Created {count} synthetic test frames ({width}x{height})")
        return frames


def run_quick_benchmark(model_paths: List[str], output_path: str = "benchmark_results.json") -> Dict[str, Any]:
    """
    Run a quick benchmark on provided models.
    
    Args:
        model_paths: List of model file paths to benchmark
        output_path: Path to save benchmark results
        
    Returns:
        Benchmark report dictionary
    """
    benchmark = PerformanceBenchmark()
    
    # Add models
    for model_path in model_paths:
        if Path(model_path).exists():
            benchmark.add_model(model_path, optimize=True)
            benchmark.add_model(model_path, optimize=False)  # Also test without optimization
    
    # Create test frames
    test_frames = benchmark.create_test_frames(count=50)
    
    # Run benchmark
    results = benchmark.run_benchmark(test_frames, duration=30.0)
    
    # Generate report
    report = benchmark.generate_report(results, output_path)
    
    return report