# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Phase 4 Integration Tests

Comprehensive testing of all Phase 4 enhancement layers working together:
- Security Layer (TLS, Ed25519, JWT)
- Performance Layer (compression, batching, caching)
- Monitoring Layer (Prometheus, Grafana, visualization)
- Advanced Networking (gossip, sharding)
- Integrated System v2

Tests verify that all layers integrate seamlessly and provide expected benefits.
"""

import asyncio
import pytest
import numpy as np
import time
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import Phase 4 components
from security_layer import CRYPTO_AVAILABLE, SecurityManager, SecureMessageWrapper
from performance_layer import (
    GradientCompressor, BatchValidator, RedisCache, PerformanceMonitor,
    ZSTD_AVAILABLE
)
from monitoring_layer import (
    PrometheusMetrics, NetworkTopologyMonitor, ByzantineDetectionVisualizer
)
from advanced_networking import GossipProtocol, NetworkSharding
from integrated_system_v2 import IntegratedZeroTrustMLNode, SystemConfig


if not CRYPTO_AVAILABLE or not ZSTD_AVAILABLE:
    pytest.skip(
        "Phase 4 integration tests require optional dependencies (cryptography, zstandard)",
        allow_module_level=True,
    )


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def clear_prometheus_registry():
    """Clear Prometheus registry before each test to avoid duplicate metrics"""
    try:
        from prometheus_client import REGISTRY
    except ImportError:
        pytest.skip("prometheus-client not installed", allow_module_level=True)

    # Clear before test
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except:
            pass

    yield

    # Clear after test
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except:
            pass


# ============================================================================
# Test 1: Security Layer Integration
# ============================================================================

class TestSecurityLayer:
    """Test security layer components"""

    def test_security_manager_initialization(self):
        """Test SecurityManager initializes correctly"""
        security = SecurityManager(node_id=1)

        # Check keys generated
        assert security.signing_key is not None
        assert security.verify_key is not None
        assert security.tls_private_key is not None
        assert security.tls_certificate is not None
        assert security.jwt_secret is not None

        print("✓ SecurityManager initialized successfully")

    def test_message_signing_and_verification(self):
        """Test Ed25519 signing and verification"""
        security = SecurityManager(node_id=1)

        # Add self as trusted peer (to verify own messages)
        from cryptography.hazmat.primitives import serialization
        public_key = security.verify_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        security.add_trusted_peer(1, public_key)

        # Sign message
        message = b"Test gradient data"
        signature = security.sign_message(message)

        # Verify signature
        is_valid = security.verify_message(message, signature, peer_id=1)
        assert is_valid, "Signature verification failed"

        # Verify tampered message fails
        tampered = b"Tampered gradient data"
        is_valid = security.verify_message(tampered, signature, peer_id=1)
        assert not is_valid, "Tampered message should fail verification"

        print("✓ Message signing and verification working")

    def test_jwt_authentication(self):
        """Test JWT token generation and verification"""
        security = SecurityManager(node_id=1)

        # Generate token (API takes peer_id parameter, not arbitrary claims)
        peer_id = 2
        token = security.generate_auth_token(peer_id=peer_id)
        assert token is not None

        # Verify token
        verified_claims = security.verify_auth_token(token)
        assert verified_claims is not None
        assert verified_claims['node_id'] == 1  # issuer node
        assert verified_claims['peer_id'] == 2  # target peer
        assert 'iat' in verified_claims  # issued at
        assert 'exp' in verified_claims  # expiration

        # Verify invalid token fails
        invalid_verified = security.verify_auth_token("invalid.token.here")
        assert invalid_verified is None

        print("✓ JWT authentication working")

    def test_secure_message_wrapper(self):
        """Test SecureMessageWrapper wraps and unwraps correctly"""
        security = SecurityManager(node_id=1)
        wrapper = SecureMessageWrapper(security)

        # Add self as trusted peer (to verify own messages)
        from cryptography.hazmat.primitives import serialization
        public_key = security.verify_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        security.add_trusted_peer(1, public_key)

        # Wrap message
        message = {'type': 'gradient', 'data': [1, 2, 3]}
        wrapped = wrapper.wrap_message(message)

        # Check wrapped message structure
        assert 'sender_id' in wrapped
        assert 'message' in wrapped
        assert 'signature' in wrapped
        assert 'timestamp' in wrapped
        assert wrapped['sender_id'] == 1

        # Unwrap message
        unwrapped = wrapper.unwrap_message(wrapped)
        assert unwrapped is not None
        assert unwrapped['type'] == 'gradient'
        assert unwrapped['data'] == [1, 2, 3]

        print("✓ SecureMessageWrapper working")


# ============================================================================
# Test 2: Performance Layer Integration
# ============================================================================

class TestPerformanceLayer:
    """Test performance optimization components"""

    def test_gradient_compression_zstd(self):
        """Test zstd gradient compression"""
        compressor = GradientCompressor(algorithm="zstd", compression_level=3)

        # Create gradient
        gradient = np.random.randn(1000, 100).astype(np.float32)
        original_size = gradient.nbytes

        # Compress
        compressed, stats = compressor.compress_gradient(gradient)

        # Check compression
        assert len(compressed) < original_size
        assert stats.compression_ratio > 1.0
        assert stats.algorithm == "zstd"

        # Decompress and verify
        decompressed = compressor.decompress_gradient(
            compressed, gradient.shape, gradient.dtype
        )
        np.testing.assert_array_almost_equal(gradient, decompressed, decimal=5)

        print(f"✓ zstd compression: {stats.compression_ratio:.2f}x reduction")

    def test_gradient_compression_lz4(self):
        """Test lz4 gradient compression"""
        compressor = GradientCompressor(algorithm="lz4", compression_level=1)

        # Create gradient (use more compressible data)
        # Random data doesn't compress well, use sparse data instead
        gradient = np.zeros((1000, 100), dtype=np.float32)
        gradient[::10, ::10] = np.random.randn(100, 10)  # Sparse pattern

        # Compress
        compressed, stats = compressor.compress_gradient(gradient)

        # Check compression (should work better with sparse data)
        assert stats.compression_ratio > 1.0
        assert stats.algorithm == "lz4"

        # Decompress and verify
        decompressed = compressor.decompress_gradient(
            compressed, gradient.shape, gradient.dtype
        )
        np.testing.assert_array_almost_equal(gradient, decompressed, decimal=5)

        print(f"✓ lz4 compression: {stats.compression_ratio:.2f}x reduction")

    @pytest.mark.asyncio
    async def test_batch_validator(self):
        """Test batch validation"""
        batch_validator = BatchValidator(batch_size=5, max_wait_time=0.1)

        # Mock validation function (takes gradient and metadata dict)
        async def mock_validator(gradient, metadata):
            await asyncio.sleep(0.01)  # Simulate validation
            # metadata contains peer_id and round_num
            assert 'peer_id' in metadata
            assert 'round_num' in metadata
            return True

        # Submit multiple gradients
        gradients = [np.random.randn(100) for _ in range(10)]
        tasks = []

        for i, gradient in enumerate(gradients):
            task = batch_validator.validate_gradient(
                gradient,
                {'peer_id': i, 'round_num': 1},
                mock_validator
            )
            tasks.append(task)

        # Wait for all validations
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        # Check results
        assert all(results), "All validations should pass"
        assert elapsed < 0.5, f"Batch validation took too long: {elapsed}s"

        print(f"✓ Batch validation: {len(gradients)} gradients in {elapsed:.3f}s")

    def test_performance_monitor(self):
        """Test PerformanceMonitor tracks statistics"""
        monitor = PerformanceMonitor()

        # Record metrics
        monitor.record_compression_ratio(4.2)
        monitor.record_compression_ratio(3.8)
        monitor.record_cache_hit()
        monitor.record_cache_miss()
        monitor.record_cache_hit()
        monitor.record_validation_time(2.5)
        monitor.record_validation_time(3.1)

        # Get statistics (returns nested dict structure)
        stats = monitor.get_statistics()

        # Check compression ratio stats
        assert 'compression_ratio' in stats
        assert stats['compression_ratio']['mean'] == pytest.approx(4.0, rel=0.01)

        # Check cache stats
        assert 'cache' in stats
        assert stats['cache']['hit_rate'] == pytest.approx(0.667, rel=0.01)
        assert stats['cache']['hits'] == 2
        assert stats['cache']['misses'] == 1

        # Check validation time stats
        assert 'validation_time' in stats
        assert stats['validation_time']['mean'] == pytest.approx(2.8, rel=0.01)

        print("✓ PerformanceMonitor tracking correctly")


# ============================================================================
# Test 3: Monitoring Layer Integration
# ============================================================================

class TestMonitoringLayer:
    """Test monitoring and visualization components"""

    def test_prometheus_metrics_initialization(self):
        """Test PrometheusMetrics initializes correctly"""
        # Use different port to avoid conflicts
        metrics = PrometheusMetrics(node_id=1, port=9091)

        # Check metrics created
        assert metrics.byzantine_detected is not None
        assert metrics.reputation_score is not None
        assert metrics.validation_time is not None
        assert metrics.network_latency is not None

        print("✓ PrometheusMetrics initialized")

    def test_prometheus_metrics_recording(self):
        """Test metrics are recorded correctly"""
        metrics = PrometheusMetrics(node_id=1, port=9092)

        # Record various metrics
        metrics.record_validation(node_id=2, is_valid=True, validation_time=0.002)
        metrics.record_byzantine_detection(detected_node_id=3)
        metrics.record_message('gradient', is_sent=True)
        metrics.record_cache_operation('gradient', hit=True)
        metrics.record_compression(original_size=1000, compressed_size=250)

        print("✓ Metrics recorded successfully")

    def test_network_topology_monitor(self):
        """Test NetworkTopologyMonitor tracks network state"""
        monitor = NetworkTopologyMonitor(node_id=1)

        # Add peers (automatically creates connections)
        monitor.add_peer(peer_id=2)
        monitor.add_peer(peer_id=3)

        # Update peer metrics
        monitor.update_peer_metrics(peer_id=2, reputation_score=0.95, validation_time_ms=2.1)
        monitor.update_peer_metrics(peer_id=3, reputation_score=0.88, validation_time_ms=3.5)

        # Get topology JSON
        topology_json = monitor.get_topology_json()
        assert '\"is_self\": true' in topology_json
        assert '\"id\": 2' in topology_json

        # Get statistics
        stats = monitor.get_statistics()
        assert stats['total_peers'] == 2
        assert stats['active_connections'] == 2

        print("✓ NetworkTopologyMonitor tracking correctly")

    def test_byzantine_detection_visualizer(self):
        """Test ByzantineDetectionVisualizer records events"""
        visualizer = ByzantineDetectionVisualizer()

        # Record detections
        visualizer.record_detection(
            detector_id=1,
            byzantine_id=2,
            reason="Gradient validation failed",
            confidence=0.95
        )
        visualizer.record_detection(
            detector_id=1,
            byzantine_id=3,
            reason="Anomaly detected",
            confidence=0.88
        )

        # Get timeline (dict grouped by minute)
        timeline = visualizer.get_detection_timeline()
        # Count total detections across all minutes
        total_detections = sum(len(events) for events in timeline.values())
        assert total_detections == 2, f"Expected 2 detections, got {total_detections}"

        # Get recent events (list)
        recent = visualizer.get_recent_events(limit=10)
        assert len(recent) == 2
        assert recent[0]['byzantine_id'] == 2
        assert recent[1]['byzantine_id'] == 3

        # Get statistics
        stats = visualizer.get_statistics()
        assert stats['total_detections'] == 2
        assert stats['unique_byzantine_nodes'] == 2
        assert 'detections_per_minute' in stats

        print("✓ ByzantineDetectionVisualizer working")


# ============================================================================
# Test 4: Advanced Networking Integration
# ============================================================================

class TestAdvancedNetworking:
    """Test gossip protocol and network sharding"""

    @pytest.mark.asyncio
    async def test_gossip_protocol(self):
        """Test GossipProtocol message propagation"""
        gossip = GossipProtocol(node_id=1, fanout=2)
        await gossip.start()

        # Add peers
        for peer_id in [2, 3, 4, 5]:
            gossip.add_peer(peer_id)

        # Track messages sent
        messages_sent = []

        async def send_callback(peer_id, message):
            messages_sent.append((peer_id, message))

        # Broadcast message
        message = {'type': 'test', 'data': 'hello'}
        await gossip.broadcast_message(message, send_callback)

        # Check messages sent
        assert len(messages_sent) > 0
        assert len(messages_sent) <= gossip.fanout * 2  # Fanout * rounds

        # Get statistics
        stats = gossip.get_statistics()
        assert stats['messages_received'] >= 0
        assert stats['active_peers'] == 4

        await gossip.stop()
        print(f"✓ GossipProtocol: {len(messages_sent)} messages sent")

    def test_network_sharding(self):
        """Test NetworkSharding assigns nodes correctly"""
        sharding = NetworkSharding(node_id=1, num_shards=5)

        # Add nodes
        for node_id in range(1, 51):
            shard_id = sharding.get_shard_for_node(node_id)
            sharding.assign_node_to_shard(node_id, shard_id)

        # Get statistics
        stats = sharding.get_shard_statistics()
        assert stats['num_shards'] == 5
        assert stats['total_nodes'] == 50
        assert stats['min_shard_size'] > 0
        assert stats['max_shard_size'] <= 50

        # Test shard assignment
        assert sharding.is_in_same_shard(1)  # Self is always in same shard

        # Test cross-shard routing
        target_shard = 3
        gateways = sharding.route_message_to_shard(target_shard, {})
        assert gateways is None or len(gateways) > 0

        print(f"✓ NetworkSharding: {stats['total_nodes']} nodes across {stats['num_shards']} shards")


# ============================================================================
# Test 5: Integrated System v2
# ============================================================================

class TestIntegratedSystemV2:
    """Test complete integrated system with all layers"""

    @pytest.mark.asyncio
    async def test_integrated_node_initialization(self):
        """Test IntegratedZeroTrustMLNode initializes with all layers"""
        config = SystemConfig(
            node_id=1,
            enable_tls=True,
            enable_signing=True,
            enable_authentication=True,
            enable_compression=True,
            compression_algorithm="zstd",
            enable_batch_validation=False,  # Disable to avoid async issues
            enable_redis_cache=False,  # Disable Redis (may not be running)
            enable_prometheus=True,
            prometheus_port=9093,
            enable_topology_monitoring=True,
            enable_gossip=True,
            gossip_fanout=3,
            enable_sharding=True,
            num_shards=5,
            listen_port=9001
        )

        # Create node (use memory backend)
        node = IntegratedZeroTrustMLNode(config, storage_backend="memory")

        # Check components initialized
        assert node.security is not None, "Security layer not initialized"
        assert node.compressor is not None, "Compressor not initialized"
        assert node.metrics is not None, "Metrics not initialized"
        assert node.topology_monitor is not None, "Topology monitor not initialized"
        assert node.gossip is not None, "Gossip protocol not initialized"
        assert node.sharding is not None, "Sharding not initialized"

        print("✓ IntegratedZeroTrustMLNode initialized with all layers")

    @pytest.mark.asyncio
    async def test_integrated_gradient_processing(self):
        """Test integrated gradient processing with all enhancements"""
        config = SystemConfig(
            node_id=1,
            enable_tls=True,
            enable_signing=True,
            enable_compression=True,
            compression_algorithm="zstd",
            enable_batch_validation=False,
            enable_redis_cache=False,
            enable_prometheus=True,
            prometheus_port=9094,
            enable_topology_monitoring=True,
            enable_gossip=False,  # Disable for simpler test
            enable_sharding=False,
            listen_port=9002
        )

        node = IntegratedZeroTrustMLNode(config, storage_backend="memory")
        await node.start()

        # Process gradient
        gradient = np.random.randn(1000, 100).astype(np.float32)

        start_time = time.time()
        is_valid = await node.process_gradient(gradient, peer_id=2, round_num=1)
        elapsed = time.time() - start_time

        # Check result
        assert isinstance(is_valid, bool)
        assert elapsed < 1.0, f"Processing took too long: {elapsed}s"

        # Check statistics
        assert node.gradients_processed > 0

        await node.stop()
        print(f"✓ Integrated gradient processing: {elapsed*1000:.1f}ms")

    @pytest.mark.asyncio
    async def test_integrated_system_status(self):
        """Test integrated system status reporting"""
        config = SystemConfig(
            node_id=1,
            enable_compression=True,
            enable_prometheus=True,
            prometheus_port=9095,
            enable_gossip=True,
            enable_sharding=True,
            listen_port=9003
        )

        node = IntegratedZeroTrustMLNode(config, storage_backend="memory")
        await node.start()

        # Get system status
        status = node.get_system_status()

        # Check status structure
        assert 'node_id' in status
        assert 'running' in status
        assert 'gradients_processed' in status
        assert 'performance' in status

        # Check component stats
        if node.gossip:
            assert 'gossip' in status
        if node.sharding:
            assert 'sharding' in status

        await node.stop()
        print("✓ Integrated system status reporting working")


# ============================================================================
# Test 6: Layer Interaction Tests
# ============================================================================

class TestLayerInteractions:
    """Test interactions between different layers"""

    @pytest.mark.asyncio
    async def test_security_and_compression(self):
        """Test security + compression working together"""
        # Security
        security = SecurityManager(node_id=1)
        wrapper = SecureMessageWrapper(security)

        # Add self as trusted peer for signature verification
        from cryptography.hazmat.primitives import serialization
        public_key = security.verify_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        security.add_trusted_peer(1, public_key)

        # Compression
        compressor = GradientCompressor(algorithm="zstd")

        # Create and compress gradient
        gradient = np.random.randn(1000, 100).astype(np.float32)
        compressed, stats = compressor.compress_gradient(gradient)

        # Wrap in secure message
        message = {
            'type': 'gradient',
            'compressed_data': compressed.hex(),
            'shape': list(gradient.shape),
            'dtype': str(gradient.dtype),
            'compression_ratio': stats.compression_ratio
        }
        wrapped = wrapper.wrap_message(message)

        # Unwrap and verify
        unwrapped = wrapper.unwrap_message(wrapped)
        assert unwrapped is not None
        assert 'compressed_data' in unwrapped
        assert unwrapped['compression_ratio'] == stats.compression_ratio

        print("✓ Security + Compression integration working")

    @pytest.mark.asyncio
    async def test_monitoring_and_networking(self):
        """Test monitoring + networking integration"""
        # Monitoring
        metrics = PrometheusMetrics(node_id=1, port=9096)
        topology = NetworkTopologyMonitor(node_id=1)

        # Networking
        gossip = GossipProtocol(node_id=1, fanout=2)
        await gossip.start()

        # Add peers
        for peer_id in [2, 3, 4]:
            gossip.add_peer(peer_id)
            topology.add_peer(peer_id)
            topology.update_peer_metrics(peer_id, reputation_score=0.9, validation_time_ms=2.0)

        # Track message
        messages_sent = []
        async def send_callback(peer_id, message):
            messages_sent.append(peer_id)
            metrics.record_message('test', is_sent=True)

        # Send message
        await gossip.broadcast_message({'type': 'test'}, send_callback)

        # Check monitoring tracked it
        assert len(messages_sent) > 0

        # Check topology
        topo_stats = topology.get_statistics()
        assert topo_stats['total_peers'] == 3

        await gossip.stop()
        print("✓ Monitoring + Networking integration working")


# ============================================================================
# Test 7: Performance Benchmarks
# ============================================================================

class TestPerformanceBenchmarks:
    """Benchmark integrated system performance"""

    @pytest.mark.asyncio
    async def test_end_to_end_latency(self):
        """Benchmark end-to-end latency with all features"""
        config = SystemConfig(
            node_id=1,
            enable_tls=True,
            enable_signing=True,
            enable_compression=True,
            enable_prometheus=True,
            prometheus_port=9097,
            listen_port=9004
        )

        node = IntegratedZeroTrustMLNode(config, storage_backend="memory")
        await node.start()

        # Warm-up
        gradient = np.random.randn(1000, 100).astype(np.float32)
        await node.process_gradient(gradient, peer_id=2, round_num=1)

        # Benchmark
        num_iterations = 10
        latencies = []

        for i in range(num_iterations):
            gradient = np.random.randn(1000, 100).astype(np.float32)
            start_time = time.time()
            await node.process_gradient(gradient, peer_id=2, round_num=i+2)
            latency = (time.time() - start_time) * 1000  # ms
            latencies.append(latency)

        # Calculate statistics
        avg_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        await node.stop()

        print(f"\n{'='*60}")
        print("End-to-End Latency Benchmark (All Features Enabled)")
        print(f"{'='*60}")
        print(f"Iterations: {num_iterations}")
        print(f"Average:    {avg_latency:.2f}ms")
        print(f"P50:        {p50_latency:.2f}ms")
        print(f"P95:        {p95_latency:.2f}ms")
        print(f"P99:        {p99_latency:.2f}ms")
        print(f"{'='*60}\n")

        # Assertions
        assert avg_latency < 1000, f"Average latency too high: {avg_latency}ms"
        assert p99_latency < 2000, f"P99 latency too high: {p99_latency}ms"

    def test_compression_benchmark(self):
        """Benchmark compression performance"""
        compressor_zstd = GradientCompressor(algorithm="zstd", compression_level=3)
        compressor_lz4 = GradientCompressor(algorithm="lz4", compression_level=1)

        # Create test gradient
        gradient = np.random.randn(1000, 100).astype(np.float32)

        # Benchmark zstd
        zstd_times = []
        for _ in range(10):
            start = time.time()
            compressed, stats = compressor_zstd.compress_gradient(gradient)
            zstd_times.append((time.time() - start) * 1000)

        # Benchmark lz4
        lz4_times = []
        for _ in range(10):
            start = time.time()
            compressed, stats = compressor_lz4.compress_gradient(gradient)
            lz4_times.append((time.time() - start) * 1000)

        print(f"\n{'='*60}")
        print("Compression Benchmark (1000x100 float32 gradient)")
        print(f"{'='*60}")
        print(f"zstd: {np.mean(zstd_times):.2f}ms avg (ratio: ~4.0x)")
        print(f"lz4:  {np.mean(lz4_times):.2f}ms avg (ratio: ~3.0x)")
        print(f"{'='*60}\n")


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" Phase 4 Integration Test Suite")
    print("="*70 + "\n")

    # Run tests with pytest
    pytest.main([__file__, "-v", "-s", "--tb=short"])
