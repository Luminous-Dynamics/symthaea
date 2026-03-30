# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Performance Layer for Hybrid ZeroTrustML

Implements:
- Gradient compression (zstd, lz4)
- Batch validation for efficiency
- Redis caching for frequently accessed data
- Performance monitoring and optimization
"""

import asyncio
import time
import hashlib
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np

# Compression imports
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    print("Warning: zstandard not available. Install with: pip install zstandard")

try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False
    print("Warning: lz4 not available. Install with: pip install lz4")

# Redis imports
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: redis not available. Install with: pip install redis")


@dataclass
class CompressionStats:
    """Statistics for compression operations"""
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time: float
    decompression_time: float
    algorithm: str

    def __str__(self) -> str:
        return (f"{self.algorithm}: {self.compression_ratio:.2f}x "
                f"({self.original_size} → {self.compressed_size} bytes, "
                f"compress: {self.compression_time*1000:.2f}ms, "
                f"decompress: {self.decompression_time*1000:.2f}ms)")


class GradientCompressor:
    """Compresses gradients for efficient network transmission"""

    def __init__(self, algorithm: str = "zstd", compression_level: int = 3):
        """
        Initialize compressor

        Args:
            algorithm: 'zstd' or 'lz4'
            compression_level: 1-22 for zstd, 0-16 for lz4 (higher = better compression)
        """
        self.algorithm = algorithm.lower()
        self.compression_level = compression_level

        if self.algorithm == "zstd" and not ZSTD_AVAILABLE:
            raise RuntimeError("zstandard library required for zstd compression")
        if self.algorithm == "lz4" and not LZ4_AVAILABLE:
            raise RuntimeError("lz4 library required for lz4 compression")

        # Initialize compressor
        if self.algorithm == "zstd":
            self.compressor = zstd.ZstdCompressor(level=compression_level)
            self.decompressor = zstd.ZstdDecompressor()
        elif self.algorithm == "lz4":
            # lz4 uses compression_level directly in compress()
            pass
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def compress_gradient(self, gradient: np.ndarray) -> Tuple[bytes, CompressionStats]:
        """
        Compress gradient array

        Returns:
            (compressed_data, stats)
        """
        # Convert to bytes
        gradient_bytes = gradient.tobytes()
        original_size = len(gradient_bytes)

        # Compress
        start = time.time()
        if self.algorithm == "zstd":
            compressed = self.compressor.compress(gradient_bytes)
        elif self.algorithm == "lz4":
            compressed = lz4.frame.compress(
                gradient_bytes,
                compression_level=self.compression_level
            )
        compression_time = time.time() - start

        # Decompress (for timing)
        start = time.time()
        if self.algorithm == "zstd":
            _ = self.decompressor.decompress(compressed)
        elif self.algorithm == "lz4":
            _ = lz4.frame.decompress(compressed)
        decompression_time = time.time() - start

        compressed_size = len(compressed)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0

        stats = CompressionStats(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            compression_time=compression_time,
            decompression_time=decompression_time,
            algorithm=self.algorithm
        )

        return compressed, stats

    def decompress_gradient(self, compressed: bytes, shape: Tuple, dtype=np.float32) -> np.ndarray:
        """Decompress gradient array"""
        if self.algorithm == "zstd":
            gradient_bytes = self.decompressor.decompress(compressed)
        elif self.algorithm == "lz4":
            gradient_bytes = lz4.frame.decompress(compressed)

        # Reconstruct array
        gradient = np.frombuffer(gradient_bytes, dtype=dtype)
        gradient = gradient.reshape(shape)

        return gradient


class BatchValidator:
    """Validates multiple gradients in batches for efficiency"""

    def __init__(self, batch_size: int = 32, max_wait_time: float = 0.1):
        """
        Initialize batch validator

        Args:
            batch_size: Maximum gradients per batch
            max_wait_time: Maximum time to wait for batch to fill (seconds)
        """
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_gradients: List[Tuple[np.ndarray, Dict]] = []
        self.results: Dict[str, Any] = {}
        self.lock = asyncio.Lock()

    async def validate_gradient(self, gradient: np.ndarray, metadata: Dict, validator_func) -> bool:
        """
        Add gradient to batch and wait for validation

        Args:
            gradient: Gradient array
            metadata: Gradient metadata
            validator_func: Async function to validate gradients

        Returns:
            Validation result
        """
        # Generate unique ID for this gradient
        gradient_id = hashlib.sha256(gradient.tobytes()).hexdigest()[:16]

        async with self.lock:
            self.pending_gradients.append((gradient, metadata))

            # If batch is full, process immediately
            if len(self.pending_gradients) >= self.batch_size:
                await self._process_batch(validator_func)

        # Wait for result (with timeout)
        start = time.time()
        while gradient_id not in self.results:
            if time.time() - start > self.max_wait_time * 10:
                # Timeout - process batch now
                async with self.lock:
                    await self._process_batch(validator_func)
                break

            await asyncio.sleep(0.01)

        # Retrieve result
        result = self.results.pop(gradient_id, False)
        return result

    async def _process_batch(self, validator_func):
        """Process current batch of gradients"""
        if not self.pending_gradients:
            return

        batch = self.pending_gradients
        self.pending_gradients = []

        # Validate all gradients in parallel
        tasks = [
            validator_func(grad, meta)
            for grad, meta in batch
        ]
        results = await asyncio.gather(*tasks)

        # Store results
        for (grad, _), result in zip(batch, results):
            grad_id = hashlib.sha256(grad.tobytes()).hexdigest()[:16]
            self.results[grad_id] = result


class RedisCache:
    """Redis-based caching for gradients and validation results"""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        """Initialize Redis cache"""
        if not REDIS_AVAILABLE:
            raise RuntimeError("redis library required for caching")

        self.host = host
        self.port = port
        self.db = db
        self.redis: Optional[redis.Redis] = None
        self.connected = False

    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis = await redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=False  # Binary mode
            )
            # Test connection
            await self.redis.ping()
            self.connected = True
            print(f"✓ Connected to Redis at {self.host}:{self.port}")
        except Exception as e:
            print(f"✗ Redis connection failed: {e}")
            self.connected = False

    async def cache_gradient(
        self,
        gradient_id: str,
        gradient: np.ndarray,
        ttl: int = 3600
    ):
        """
        Cache gradient

        Args:
            gradient_id: Unique gradient identifier
            gradient: Gradient array
            ttl: Time to live in seconds
        """
        if not self.connected:
            return

        try:
            # Store as bytes
            await self.redis.setex(
                f"gradient:{gradient_id}",
                ttl,
                gradient.tobytes()
            )

            # Store metadata
            await self.redis.setex(
                f"gradient_meta:{gradient_id}",
                ttl,
                str(gradient.shape) + "," + str(gradient.dtype)
            )
        except Exception as e:
            # TODO: Replace print with proper logging - see CLEANUP_LOG.md
            print(f"Cache write error: {e}")

    async def get_gradient(self, gradient_id: str) -> Optional[np.ndarray]:
        """Retrieve cached gradient"""
        if not self.connected:
            return None

        try:
            # Get bytes and metadata
            gradient_bytes = await self.redis.get(f"gradient:{gradient_id}")
            meta_str = await self.redis.get(f"gradient_meta:{gradient_id}")

            if gradient_bytes is None or meta_str is None:
                return None

            # Parse metadata
            meta_str = meta_str.decode()
            shape_str, dtype_str = meta_str.rsplit(",", 1)
            shape = eval(shape_str)  # Safe since we control the data
            dtype = np.dtype(dtype_str)

            # Reconstruct array
            gradient = np.frombuffer(gradient_bytes, dtype=dtype)
            gradient = gradient.reshape(shape)

            return gradient
        except Exception as e:
            # TODO: Replace print with proper logging - see CLEANUP_LOG.md
            print(f"Cache read error: {e}")
            return None

    async def cache_validation_result(
        self,
        gradient_id: str,
        is_valid: bool,
        ttl: int = 3600
    ):
        """Cache validation result"""
        if not self.connected:
            return

        try:
            await self.redis.setex(
                f"validation:{gradient_id}",
                ttl,
                "1" if is_valid else "0"
            )
        except Exception as e:
            # TODO: Replace print with proper logging - see CLEANUP_LOG.md
            print(f"Cache write error: {e}")

    async def get_validation_result(self, gradient_id: str) -> Optional[bool]:
        """Get cached validation result"""
        if not self.connected:
            return None

        try:
            result = await self.redis.get(f"validation:{gradient_id}")
            if result is None:
                return None
            return result.decode() == "1"
        except Exception as e:
            # TODO: Replace print with proper logging - see CLEANUP_LOG.md
            print(f"Cache read error: {e}")
            return None

    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
            self.connected = False


class PerformanceMonitor:
    """Monitors system performance metrics"""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {
            'validation_time': [],
            'compression_ratio': [],
            'network_latency': [],
            'cache_hit_rate': [],
        }
        self.cache_hits = 0
        self.cache_misses = 0

    def record_validation_time(self, time_ms: float):
        """Record validation time"""
        self.metrics['validation_time'].append(time_ms)

    def record_compression_ratio(self, ratio: float):
        """Record compression ratio"""
        self.metrics['compression_ratio'].append(ratio)

    def record_network_latency(self, latency_ms: float):
        """Record network latency"""
        self.metrics['network_latency'].append(latency_ms)

    def record_cache_hit(self):
        """Record cache hit"""
        self.cache_hits += 1

    def record_cache_miss(self):
        """Record cache miss"""
        self.cache_misses += 1

    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics"""
        stats = {}

        for metric_name, values in self.metrics.items():
            if values:
                stats[metric_name] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values),
                }

        stats['cache'] = {
            'hit_rate': self.get_cache_hit_rate(),
            'hits': self.cache_hits,
            'misses': self.cache_misses
        }

        return stats

    def print_summary(self):
        """Print performance summary"""
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)

        stats = self.get_statistics()

        for metric_name, metric_stats in stats.items():
            if metric_name == 'cache':
                print(f"\nCache Performance:")
                print(f"  Hit rate: {metric_stats['hit_rate']*100:.1f}%")
                print(f"  Hits: {metric_stats['hits']}")
                print(f"  Misses: {metric_stats['misses']}")
            else:
                print(f"\n{metric_name.replace('_', ' ').title()}:")
                print(f"  Mean: {metric_stats['mean']:.2f}")
                print(f"  Median: {metric_stats['median']:.2f}")
                print(f"  Range: {metric_stats['min']:.2f} - {metric_stats['max']:.2f}")

        print("\n" + "="*60)


# Example usage
if __name__ == "__main__":
    print("Performance Layer - Testing Components\n")

    # Test gradient compression
    if ZSTD_AVAILABLE or LZ4_AVAILABLE:
        print("1. Testing Gradient Compression")
        print("-" * 40)

        # Create test gradient
        gradient = np.random.randn(1000, 100).astype(np.float32)
        print(f"Original gradient: {gradient.shape}, {gradient.nbytes} bytes")

        # Test zstd
        if ZSTD_AVAILABLE:
            compressor = GradientCompressor(algorithm="zstd", compression_level=3)
            compressed, stats = compressor.compress_gradient(gradient)
            print(f"\nZSTD: {stats}")

            # Verify decompression
            decompressed = compressor.decompress_gradient(
                compressed,
                gradient.shape,
                gradient.dtype
            )
            assert np.allclose(gradient, decompressed), "Decompression mismatch!"
            print("  ✓ Decompression verified")

        # Test lz4
        if LZ4_AVAILABLE:
            compressor = GradientCompressor(algorithm="lz4", compression_level=3)
            compressed, stats = compressor.compress_gradient(gradient)
            print(f"\nLZ4: {stats}")

            decompressed = compressor.decompress_gradient(
                compressed,
                gradient.shape,
                gradient.dtype
            )
            assert np.allclose(gradient, decompressed), "Decompression mismatch!"
            print("  ✓ Decompression verified")

    # Test performance monitoring
    print("\n\n2. Testing Performance Monitor")
    print("-" * 40)

    monitor = PerformanceMonitor()

    # Simulate some metrics
    for _ in range(100):
        monitor.record_validation_time(np.random.uniform(0.5, 2.0))
        monitor.record_compression_ratio(np.random.uniform(2.0, 5.0))
        monitor.record_network_latency(np.random.uniform(10, 50))

        if np.random.random() < 0.7:  # 70% cache hit rate
            monitor.record_cache_hit()
        else:
            monitor.record_cache_miss()

    monitor.print_summary()

    print("\n✓ Performance Layer initialized successfully")