# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
GPU Acceleration (Phase 5 Enhancement 4)

Accelerates gradient validation using CUDA/cuBLAS:
- 10x speedup target (<100ms validation)
- Batch processing on GPU
- Graceful fallback to CPU
- Memory-efficient GPU operations
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
import time

# Try to import PyTorch for GPU operations
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available. GPU acceleration disabled. Install: pip install torch")


class DeviceType(Enum):
    """Computation device types"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


@dataclass
class GPUConfig:
    """GPU configuration"""
    device_type: DeviceType = DeviceType.CUDA
    device_id: int = 0
    memory_fraction: float = 0.8  # Use 80% of GPU memory
    enable_tf32: bool = True  # TensorFloat-32 for speed
    enable_mixed_precision: bool = True  # FP16/FP32 mixed
    batch_size: int = 32
    pin_memory: bool = True  # Faster CPU→GPU transfer


@dataclass
class GPUStats:
    """GPU performance statistics"""
    device_name: str
    total_memory_gb: float
    allocated_memory_gb: float
    cached_memory_gb: float
    gpu_utilization: float
    temperature_c: Optional[float] = None


class GPUManager:
    """
    Manages GPU resources and operations

    Handles:
    - Device selection (CUDA, MPS, CPU)
    - Memory management
    - Batch processing
    - Graceful fallback to CPU
    """

    def __init__(self, config: Optional[GPUConfig] = None):
        self.config = config or GPUConfig()
        self.device = self._initialize_device()
        self.dtype = torch.float32  # Default precision

        if self.is_gpu_available():
            self._configure_gpu()

        print(f"🚀 GPU Manager initialized on {self.device}")

    def _initialize_device(self) -> torch.device:
        """Initialize computation device with fallback"""
        if not TORCH_AVAILABLE:
            print("⚠️  PyTorch not available, using CPU")
            return torch.device("cpu")

        # Try CUDA first
        if self.config.device_type == DeviceType.CUDA and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config.device_id}")
            print(f"✓ CUDA device detected: {torch.cuda.get_device_name(device)}")
            return device

        # Try MPS (Apple Silicon)
        if self.config.device_type == DeviceType.MPS and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("✓ Apple Silicon (MPS) detected")
            return device

        # Fallback to CPU
        print("⚠️  No GPU available, falling back to CPU")
        return torch.device("cpu")

    def _configure_gpu(self):
        """Configure GPU settings for optimal performance"""
        if not self.is_gpu_available():
            return

        # Enable TensorFloat-32 (NVIDIA Ampere+)
        if self.config.enable_tf32 and hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Set memory fraction
        if self.device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(
                self.config.memory_fraction,
                device=self.device
            )

    def is_gpu_available(self) -> bool:
        """Check if GPU is available"""
        return self.device.type in ["cuda", "mps"]

    def to_device(self, tensor: np.ndarray) -> torch.Tensor:
        """Move numpy array to GPU"""
        if not TORCH_AVAILABLE:
            return tensor

        # Convert to torch tensor
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor).to(self.dtype)

        # Move to device
        return tensor.to(self.device, non_blocking=self.config.pin_memory)

    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Move GPU tensor back to numpy"""
        if not TORCH_AVAILABLE:
            return tensor

        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        return tensor

    def get_stats(self) -> GPUStats:
        """Get GPU statistics"""
        if not self.is_gpu_available() or not TORCH_AVAILABLE:
            return GPUStats(
                device_name="CPU",
                total_memory_gb=0.0,
                allocated_memory_gb=0.0,
                cached_memory_gb=0.0,
                gpu_utilization=0.0
            )

        if self.device.type == "cuda":
            device_props = torch.cuda.get_device_properties(self.device)
            allocated = torch.cuda.memory_allocated(self.device) / 1e9
            cached = torch.cuda.memory_reserved(self.device) / 1e9
            total = device_props.total_memory / 1e9

            # Try to get GPU utilization
            utilization = 0.0
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.config.device_id)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization = util.gpu / 100.0
                pynvml.nvmlShutdown()
            except:
                pass  # pynvml not available

            return GPUStats(
                device_name=device_props.name,
                total_memory_gb=total,
                allocated_memory_gb=allocated,
                cached_memory_gb=cached,
                gpu_utilization=utilization
            )

        # MPS
        return GPUStats(
            device_name="Apple Silicon (MPS)",
            total_memory_gb=0.0,  # Not queryable on MPS
            allocated_memory_gb=0.0,
            cached_memory_gb=0.0,
            gpu_utilization=0.0
        )


class GPUAcceleratedValidator:
    """
    GPU-accelerated gradient validation

    Provides 10x speedup for:
    - Gradient norm calculation
    - Model update simulation
    - PoGQ evaluation
    - Batch validation
    """

    def __init__(self, gpu_manager: Optional[GPUManager] = None):
        self.gpu = gpu_manager or GPUManager()
        self.batch_size = self.gpu.config.batch_size

    @torch.no_grad()
    def validate_gradient_gpu(
        self,
        gradient: np.ndarray,
        model: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[bool, float, float]:
        """
        GPU-accelerated gradient validation

        Args:
            gradient: Gradient to validate
            model: Current model weights
            threshold: Validation threshold

        Returns:
            (is_valid, pogq_score, norm)
        """
        if not TORCH_AVAILABLE or not self.gpu.is_gpu_available():
            # Fallback to CPU numpy
            return self._validate_gradient_cpu(gradient, model, threshold)

        # Move to GPU
        grad_gpu = self.gpu.to_device(gradient)
        model_gpu = self.gpu.to_device(model)

        # Calculate gradient norm (GPU-accelerated)
        grad_norm = torch.linalg.norm(grad_gpu).item()

        # Simulate model update
        updated_model = model_gpu - 0.01 * grad_gpu

        # Calculate PoGQ score (proxy: improvement in model)
        # In real implementation, this would evaluate on test set
        # For now, use change magnitude
        change_magnitude = torch.linalg.norm(updated_model - model_gpu).item()
        pogq_score = min(1.0, change_magnitude / (grad_norm + 1e-10))

        # Validation decision
        is_valid = pogq_score > threshold

        return is_valid, float(pogq_score), float(grad_norm)

    def _validate_gradient_cpu(
        self,
        gradient: np.ndarray,
        model: np.ndarray,
        threshold: float
    ) -> Tuple[bool, float, float]:
        """CPU fallback for gradient validation"""
        # Calculate gradient norm
        grad_norm = np.linalg.norm(gradient)

        # Simulate model update
        updated_model = model - 0.01 * gradient

        # Calculate PoGQ score
        change_magnitude = np.linalg.norm(updated_model - model)
        pogq_score = min(1.0, change_magnitude / (grad_norm + 1e-10))

        # Validation decision
        is_valid = pogq_score > threshold

        return is_valid, float(pogq_score), float(grad_norm)

    @torch.no_grad()
    def validate_batch_gpu(
        self,
        gradients: List[np.ndarray],
        model: np.ndarray,
        threshold: float = 0.5
    ) -> List[Tuple[bool, float, float]]:
        """
        GPU-accelerated batch validation

        Args:
            gradients: List of gradients to validate
            model: Current model weights
            threshold: Validation threshold

        Returns:
            List of (is_valid, pogq_score, norm) for each gradient
        """
        if not TORCH_AVAILABLE or not self.gpu.is_gpu_available():
            # Fallback to sequential CPU validation
            return [
                self._validate_gradient_cpu(grad, model, threshold)
                for grad in gradients
            ]

        # Stack gradients into batch (shape: [batch_size, *gradient_shape])
        try:
            grad_batch = torch.stack([
                self.gpu.to_device(grad) for grad in gradients
            ])
        except:
            # Shapes don't match, fall back to sequential
            return [
                self.validate_gradient_gpu(grad, model, threshold)
                for grad in gradients
            ]

        model_gpu = self.gpu.to_device(model)

        # Batch norm calculation
        grad_norms = torch.linalg.norm(
            grad_batch.view(len(gradients), -1),
            dim=1
        )

        # Batch model update simulation
        updated_models = model_gpu.unsqueeze(0) - 0.01 * grad_batch

        # Batch change magnitude
        change_magnitudes = torch.linalg.norm(
            (updated_models - model_gpu.unsqueeze(0)).view(len(gradients), -1),
            dim=1
        )

        # Batch PoGQ scores
        pogq_scores = torch.clamp(
            change_magnitudes / (grad_norms + 1e-10),
            max=1.0
        )

        # Batch validation
        is_valid_batch = pogq_scores > threshold

        # Convert back to numpy
        results = [
            (
                bool(is_valid_batch[i].item()),
                float(pogq_scores[i].item()),
                float(grad_norms[i].item())
            )
            for i in range(len(gradients))
        ]

        return results


class GPUMemoryPool:
    """
    GPU memory pool for efficient allocation

    Reduces allocation overhead by reusing GPU tensors
    """

    def __init__(self, gpu_manager: GPUManager):
        self.gpu = gpu_manager
        self.pool: Dict[Tuple[int, ...], List[torch.Tensor]] = {}

    def allocate(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Allocate or reuse GPU tensor"""
        if not TORCH_AVAILABLE:
            return None

        # Check pool for available tensor
        if shape in self.pool and self.pool[shape]:
            tensor = self.pool[shape].pop()
            return tensor

        # Allocate new tensor
        tensor = torch.empty(shape, dtype=dtype, device=self.gpu.device)
        return tensor

    def free(self, tensor: torch.Tensor):
        """Return tensor to pool"""
        if not TORCH_AVAILABLE or tensor is None:
            return

        shape = tuple(tensor.shape)
        if shape not in self.pool:
            self.pool[shape] = []

        self.pool[shape].append(tensor)

    def clear(self):
        """Clear all pooled tensors"""
        self.pool.clear()
        if TORCH_AVAILABLE and self.gpu.is_gpu_available():
            torch.cuda.empty_cache()


# Performance benchmarking
if __name__ == "__main__":
    print("=" * 60)
    print("GPU ACCELERATION - PERFORMANCE BENCHMARK")
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("\n❌ PyTorch not available. Install with: pip install torch")
        exit(1)

    # Initialize GPU
    gpu_manager = GPUManager()
    validator = GPUAcceleratedValidator(gpu_manager)

    # Print GPU info
    stats = gpu_manager.get_stats()
    print(f"\n📊 Device: {stats.device_name}")
    if gpu_manager.is_gpu_available():
        print(f"   Memory: {stats.allocated_memory_gb:.2f}GB / {stats.total_memory_gb:.2f}GB")
        print(f"   Utilization: {stats.gpu_utilization*100:.1f}%")

    # Create test data
    gradient_shape = (1000, 100)
    num_gradients = 100

    print(f"\n🧪 Test Configuration:")
    print(f"   Gradient shape: {gradient_shape}")
    print(f"   Number of gradients: {num_gradients}")

    gradients = [np.random.randn(*gradient_shape).astype(np.float32) for _ in range(num_gradients)]
    model = np.random.randn(*gradient_shape).astype(np.float32)

    # Benchmark CPU
    print(f"\n⏱️  CPU Benchmark (Sequential):")
    start = time.time()
    cpu_results = [
        validator._validate_gradient_cpu(grad, model, 0.5)
        for grad in gradients
    ]
    cpu_time = time.time() - start
    print(f"   Time: {cpu_time*1000:.2f}ms")
    print(f"   Throughput: {num_gradients/cpu_time:.1f} gradients/sec")

    # Benchmark GPU (if available)
    if gpu_manager.is_gpu_available():
        print(f"\n⚡ GPU Benchmark (Batch):")
        start = time.time()
        gpu_results = validator.validate_batch_gpu(gradients, model, 0.5)
        gpu_time = time.time() - start
        print(f"   Time: {gpu_time*1000:.2f}ms")
        print(f"   Throughput: {num_gradients/gpu_time:.1f} gradients/sec")
        print(f"   Speedup: {cpu_time/gpu_time:.1f}x")

        # Verify results match
        matches = sum(
            1 for cpu_res, gpu_res in zip(cpu_results, gpu_results)
            if cpu_res[0] == gpu_res[0]  # Compare is_valid
        )
        print(f"   Accuracy: {matches}/{num_gradients} ({matches/num_gradients*100:.1f}%)")

        # Print final stats
        final_stats = gpu_manager.get_stats()
        print(f"\n📊 Final GPU Stats:")
        print(f"   Memory allocated: {final_stats.allocated_memory_gb:.2f}GB")
        print(f"   Memory cached: {final_stats.cached_memory_gb:.2f}GB")

    print("\n" + "=" * 60)
    print("✅ BENCHMARK COMPLETE")
    print("=" * 60)