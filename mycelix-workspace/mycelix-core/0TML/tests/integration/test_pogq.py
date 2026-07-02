#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Comprehensive Integration Tests for PoGQ-Enhanced Byzantine FL System
Tests the complete unified system with all components integrated

Test Coverage:
1. PoGQ proof generation and verification
2. Byzantine detection with trust weighting
3. Holochain reputation integration (simulated)
4. Next-gen algorithm selection and application
5. Performance optimizations (GPU, caching, parallelization)
6. Privacy preservation with Rényi DP
7. Monitoring and observability
8. End-to-end FL rounds
"""

import unittest
import asyncio
import numpy as np
import time
import torch
from typing import Dict, List
import hashlib
import json
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os

# Import all components to test
from unified_fl_system import UnifiedFLSystem, UnifiedConfig
from pogq_system import ProofOfGoodQuality, PoGQProof
from byzantine_fl_with_pogq import ByzantineFLWithPoGQ
from holochain_reputation_bridge import ReputationBridge, ReputationEntry
from next_gen_fl_algorithms import ClientUpdate
from integrated_nextgen_fl import SystemHeterogeneity
from performance_optimizations import OptimizedFLSystem
from monitoring_observability import MonitoringSystem
from privacy_preservation import AdvancedPrivacyPreservation


class TestPoGQIntegration(unittest.TestCase):
    """Test PoGQ proof generation and verification"""
    
    def setUp(self):
        """Initialize test environment"""
        self.pogq = ProofOfGoodQuality(quality_threshold=0.3)
        self.gradient_dim = 100
        
    def test_proof_generation(self):
        """Test PoGQ proof generation for various client types"""
        # Honest high-quality client
        proof_honest = self.pogq.generate_proof(
            client_id="honest_alice",
            round_number=1,
            gradient=np.ones(self.gradient_dim),
            loss_before=2.0,
            loss_after=1.0,
            dataset_size=1000,
            computation_time=10.0
        )
        
        self.assertIsInstance(proof_honest, PoGQProof)
        self.assertGreater(proof_honest.quality_score, 0.5)
        self.assertTrue(proof_honest.is_valid)
        
        # Byzantine client (fake improvement)
        proof_byzantine = self.pogq.generate_proof(
            client_id="byzantine_eve",
            round_number=1,
            gradient=np.random.randn(self.gradient_dim) * 10,
            loss_before=1.0,
            loss_after=0.99,  # Fake minimal improvement
            dataset_size=100,
            computation_time=0.1
        )
        
        self.assertLess(proof_byzantine.quality_score, 0.3)
        
    def test_trust_weight_calculation(self):
        """Test trust weight calculation with PoGQ and reputation"""
        proof = self.pogq.generate_proof(
            client_id="test_client",
            round_number=1,
            gradient=np.ones(self.gradient_dim),
            loss_before=2.0,
            loss_after=1.2,
            dataset_size=1000,
            computation_time=10.0
        )
        
        # Test with different historical reputations
        trust_high = self.pogq.calculate_trust_weight(proof, historical_reputation=0.9)
        trust_low = self.pogq.calculate_trust_weight(proof, historical_reputation=0.2)
        
        self.assertGreater(trust_high, trust_low)
        self.assertTrue(0 <= trust_high <= 1)
        self.assertTrue(0 <= trust_low <= 1)
        
    def test_quality_verification(self):
        """Test quality verification with statistical tests"""
        # Generate gradients with known patterns
        gradient_consistent = np.sin(np.arange(self.gradient_dim) * 0.1)
        gradient_random = np.random.randn(self.gradient_dim) * 10
        
        proof_consistent = self.pogq.generate_proof(
            client_id="consistent",
            round_number=1,
            gradient=gradient_consistent,
            loss_before=2.0,
            loss_after=1.0,
            dataset_size=1000,
            computation_time=10.0
        )
        
        proof_random = self.pogq.generate_proof(
            client_id="random",
            round_number=1,
            gradient=gradient_random,
            loss_before=2.0,
            loss_after=1.9,
            dataset_size=1000,
            computation_time=10.0
        )
        
        # Consistent gradient should have higher quality
        self.assertGreater(proof_consistent.quality_score, proof_random.quality_score)


class TestByzantineDetectionWithPoGQ(unittest.TestCase):
    """Test Byzantine detection enhanced with PoGQ trust weighting"""
    
    def setUp(self):
        """Initialize test environment"""
        self.detector = ByzantineFLWithPoGQ(
            aggregation_method='auto',
            byzantine_threshold=0.3,
            detection_ensemble=True,
            quality_threshold=0.3
        )
        self.gradient_dim = 100
        
    async def test_trust_weighted_detection(self):
        """Test Byzantine detection with trust weighting from PoGQ"""
        # Create mixed client updates
        client_updates = {}
        quality_scores = {}
        reputation_scores = {}
        
        # Honest clients
        for i in range(7):
            client_id = f"honest_{i}"
            client_updates[client_id] = np.ones(self.gradient_dim) + \
                                       np.random.randn(self.gradient_dim) * 0.1
            quality_scores[client_id] = 0.8 + np.random.random() * 0.15
            reputation_scores[client_id] = 0.7 + np.random.random() * 0.2
        
        # Byzantine clients
        for i in range(3):
            client_id = f"byzantine_{i}"
            client_updates[client_id] = np.random.randn(self.gradient_dim) * 10
            quality_scores[client_id] = 0.1 + np.random.random() * 0.2
            reputation_scores[client_id] = 0.2 + np.random.random() * 0.2
        
        # Run detection
        result = await self.detector.aggregate_with_pogq(
            client_updates,
            quality_scores,
            reputation_scores
        )
        
        self.assertIn('aggregated_gradient', result)
        self.assertIn('byzantine_clients', result)
        self.assertIn('trust_weights', result)
        
        # Check that Byzantine clients were detected
        byzantine_detected = result['byzantine_clients']
        for byz_id in ["byzantine_0", "byzantine_1", "byzantine_2"]:
            self.assertIn(byz_id, byzantine_detected)
        
    def test_ensemble_detection_methods(self):
        """Test ensemble of Byzantine detection methods"""
        # Create gradients with different Byzantine patterns
        gradients = []
        
        # Normal gradients
        for _ in range(6):
            gradients.append(np.ones(self.gradient_dim) + 
                           np.random.randn(self.gradient_dim) * 0.1)
        
        # Outlier Byzantine
        gradients.append(np.ones(self.gradient_dim) * 100)
        
        # Sign-flipping Byzantine
        gradients.append(-np.ones(self.gradient_dim))
        
        # Random Byzantine
        gradients.append(np.random.randn(self.gradient_dim) * 10)
        
        # Detect Byzantine clients
        byzantine_mask = self.detector._detect_byzantine_ensemble(gradients)
        
        # Last 3 should be detected as Byzantine
        self.assertTrue(byzantine_mask[6])
        self.assertTrue(byzantine_mask[7])
        self.assertTrue(byzantine_mask[8])


class TestHolochainIntegration(unittest.TestCase):
    """Test Holochain reputation bridge integration (with mocking)"""
    
    @patch('holochain_reputation_bridge.websockets.connect')
    async def test_reputation_retrieval(self, mock_connect):
        """Test getting reputation from Holochain"""
        # Mock WebSocket connection
        mock_ws = AsyncMock()
        mock_connect.return_value = mock_ws
        
        # Mock Holochain response
        mock_ws.recv.return_value = json.dumps({
            'id': 1,
            'result': {
                'agent_id': 'hCAk123',
                'client_id': 'test_client',
                'reputation_score': 0.85,
                'quality_history': [0.8, 0.82, 0.85],
                'total_rounds': 10,
                'byzantine_flags': 0,
                'last_update': time.time(),
                'proof_hash': 'abc123'
            }
        })
        
        bridge = ReputationBridge()
        
        # Test get reputation
        reputation = await bridge.get_client_reputation('test_client')
        self.assertEqual(reputation, 0.85)
        
    @patch('holochain_reputation_bridge.websockets.connect')
    async def test_reputation_update(self, mock_connect):
        """Test updating reputation in Holochain"""
        mock_ws = AsyncMock()
        mock_connect.return_value = mock_ws
        mock_ws.recv.return_value = json.dumps({'id': 1, 'result': True})
        
        bridge = ReputationBridge()
        
        # Test update from FL round
        client_updates = {
            'client_1': {
                'quality_score': 0.9,
                'was_byzantine': False,
                'gradient_hash': 'hash1'
            },
            'client_2': {
                'quality_score': 0.2,
                'was_byzantine': True,
                'gradient_hash': 'hash2'
            }
        }
        
        await bridge.update_from_fl_round(1, client_updates)
        
        # Check that updates were queued
        self.assertEqual(len(bridge.local_updates), 2)


class TestNextGenAlgorithms(unittest.TestCase):
    """Test next-generation FL algorithms integration"""
    
    def setUp(self):
        """Initialize test environment"""
        self.config = UnifiedConfig(
            nextgen_algorithm='auto',
            dataset_type='MNIST'
        )
        self.system = UnifiedFLSystem(self.config)
        self.gradient_dim = 100
        
    def test_heterogeneity_measurement(self):
        """Test system heterogeneity measurement"""
        # Create heterogeneous client updates
        updates = []
        for i in range(10):
            if i < 3:
                # Fast clients
                num_steps = 100
                num_samples = 1000
            elif i < 7:
                # Medium clients
                num_steps = 50
                num_samples = 500
            else:
                # Stragglers
                num_steps = 10
                num_samples = 100
            
            updates.append(ClientUpdate(
                gradient=np.random.randn(self.gradient_dim),
                num_steps=num_steps,
                num_samples=num_samples,
                client_id=f"client_{i}",
                metadata={}
            ))
        
        heterogeneity = self.system.nextgen_fl.measure_heterogeneity(updates)
        
        self.assertIsInstance(heterogeneity, SystemHeterogeneity)
        self.assertGreater(heterogeneity.system_heterogeneity, 0)
        self.assertGreater(heterogeneity.straggler_rate, 0)
        
    def test_algorithm_selection(self):
        """Test automatic algorithm selection based on heterogeneity"""
        # High straggler scenario
        het_stragglers = SystemHeterogeneity(
            system_heterogeneity=0.3,
            data_heterogeneity=0.2,
            straggler_rate=0.6,
            avg_steps_variance=0.5,
            communication_quality=0.6
        )
        
        selected = self.system.nextgen_fl.select_nextgen_algorithm(
            het_stragglers, num_rounds_completed=5
        )
        self.assertEqual(selected, 'fedprox')  # FedProx for stragglers
        
        # High data heterogeneity scenario
        het_data = SystemHeterogeneity(
            system_heterogeneity=0.2,
            data_heterogeneity=0.8,
            straggler_rate=0.1,
            avg_steps_variance=0.2,
            communication_quality=0.8
        )
        
        selected = self.system.nextgen_fl.select_nextgen_algorithm(
            het_data, num_rounds_completed=50
        )
        self.assertEqual(selected, 'scaffold')  # Scaffold for data heterogeneity


class TestPerformanceOptimizations(unittest.TestCase):
    """Test performance optimization components"""
    
    def setUp(self):
        """Initialize test environment"""
        self.optimizer = OptimizedFLSystem(
            use_gpu=torch.cuda.is_available(),
            use_cache=True,
            use_parallel=True,
            cache_size_mb=100
        )
        self.gradient_dim = 1000
        
    def test_gpu_acceleration(self):
        """Test GPU acceleration for aggregation"""
        if not torch.cuda.is_available():
            self.skipTest("GPU not available")
        
        # Generate test gradients
        gradients = [np.random.randn(self.gradient_dim).astype(np.float32)
                    for _ in range(50)]
        
        # Time CPU vs GPU
        start = time.time()
        result_cpu = np.median(gradients, axis=0)
        cpu_time = time.time() - start
        
        start = time.time()
        result_gpu = self.optimizer.gpu_accelerator.accelerated_median(gradients)
        gpu_time = time.time() - start
        
        # GPU should be faster for large operations
        self.assertLess(gpu_time, cpu_time * 2)  # At least not slower
        
        # Results should be similar
        np.testing.assert_allclose(result_cpu, result_gpu, rtol=1e-5)
        
    def test_caching_system(self):
        """Test intelligent caching"""
        gradients = [np.random.randn(self.gradient_dim) for _ in range(10)]
        
        # First aggregation (cache miss)
        start = time.time()
        result1 = self.optimizer.optimized_aggregate(gradients, method='median')
        miss_time = time.time() - start
        
        # Second aggregation (cache hit)
        start = time.time()
        result2 = self.optimizer.optimized_aggregate(gradients, method='median')
        hit_time = time.time() - start
        
        # Cache hit should be much faster
        self.assertLess(hit_time, miss_time / 10)
        
        # Results should be identical
        np.testing.assert_array_equal(result1, result2)
        
        # Check cache stats
        stats = self.optimizer.cache.get_stats()
        self.assertGreater(stats['hit_rate'], 0)
        
    async def test_parallel_processing(self):
        """Test parallel client processing"""
        num_clients = 20
        client_data = {
            f"client_{i}": np.random.randn(self.gradient_dim)
            for i in range(num_clients)
        }
        
        # Simple processing function
        def process_func(gradient):
            # Simulate some computation
            return np.linalg.norm(gradient)
        
        start = time.time()
        results = await self.optimizer.parallel_processor.process_clients_async(
            client_data, process_func, batch_size=5
        )
        parallel_time = time.time() - start
        
        self.assertEqual(len(results), num_clients)
        
        # Compare with sequential
        start = time.time()
        sequential_results = {
            cid: process_func(grad) 
            for cid, grad in client_data.items()
        }
        sequential_time = time.time() - start
        
        # Parallel should be faster for many clients
        if num_clients > 10:
            self.assertLess(parallel_time, sequential_time * 1.5)


class TestPrivacyPreservation(unittest.TestCase):
    """Test privacy preservation with Rényi DP"""
    
    def setUp(self):
        """Initialize test environment"""
        self.privacy = AdvancedPrivacyPreservation(
            mechanism='renyi_dp',
            privacy_budget=10.0,
            noise_scale=0.1
        )
        self.gradient_dim = 100
        
    def test_renyi_dp_accounting(self):
        """Test Rényi differential privacy accounting"""
        gradient = np.ones(self.gradient_dim)
        
        # Apply privacy multiple times
        total_spent = 0
        for round_num in range(10):
            private_grad, epsilon_spent = self.privacy.apply_privacy(
                gradient, round_num
            )
            total_spent += epsilon_spent
            
            # Check that noise was added
            self.assertFalse(np.allclose(gradient, private_grad))
        
        # Check privacy accounting
        self.assertLess(total_spent, self.privacy.privacy_budget)
        self.assertGreater(total_spent, 0)
        
    def test_privacy_amplification(self):
        """Test privacy amplification by sampling"""
        gradient = np.ones(self.gradient_dim)
        
        # With subsampling
        private_grad1, epsilon1 = self.privacy.apply_privacy(
            gradient, round_number=1, sampling_rate=0.1
        )
        
        # Without subsampling
        private_grad2, epsilon2 = self.privacy.apply_privacy(
            gradient, round_number=2, sampling_rate=1.0
        )
        
        # Subsampling should give better privacy (lower epsilon)
        self.assertLess(epsilon1, epsilon2)


class TestMonitoringObservability(unittest.TestCase):
    """Test monitoring and observability system"""
    
    def setUp(self):
        """Initialize test environment"""
        self.monitoring = MonitoringSystem(
            dashboard_port=8081,  # Different port to avoid conflicts
            prometheus_port=9091
        )
        
    def test_metrics_recording(self):
        """Test recording FL round metrics"""
        metrics = {
            'round': 1,
            'num_clients': 100,
            'byzantine_clients': ['client_1', 'client_2'],
            'round_time': 5.2,
            'aggregation_time': 1.3,
            'aggregation_method': 'krum',
            'gradient_norm': 2.5,
            'accuracy': 0.85,
            'privacy_budget_remaining': 8.5,
            'byzantine_ratio': 0.02
        }
        
        self.monitoring.record_round(metrics)
        
        # Check that metrics were recorded
        summary = self.monitoring.metrics_collector.get_metrics_summary()
        self.assertEqual(summary['rounds_total'], 1)
        self.assertEqual(summary['model_accuracy'], 0.85)
        
    def test_alert_system(self):
        """Test alert triggering"""
        # Trigger high Byzantine ratio alert
        metrics = {
            'byzantine_ratio': 0.5,  # Above threshold
            'num_clients': 100,
            'accuracy': 0.8,
            'privacy_budget_remaining': 5.0
        }
        
        self.monitoring.record_round(metrics)
        
        # Check that alert was fired
        alerts = list(self.monitoring.metrics_collector.alert_history)
        self.assertTrue(any(
            alert['name'] == 'high_byzantine_ratio' 
            for alert in alerts
        ))
        
    def test_health_checks(self):
        """Test system health monitoring"""
        health = self.monitoring.health_monitor.run_all_checks()
        
        self.assertIsInstance(health, list)
        self.assertTrue(len(health) > 0)
        
        # Check health check structure
        for check in health:
            self.assertIn('name', check.__dict__)
            self.assertIn('status', check.__dict__)
            self.assertIn('message', check.__dict__)


class TestUnifiedSystem(unittest.TestCase):
    """Test complete unified FL system"""
    
    def setUp(self):
        """Initialize test environment"""
        self.config = UnifiedConfig(
            num_rounds=5,
            byzantine_threshold=0.3,
            quality_threshold=0.3,
            holochain_enabled=False,  # Disable for testing
            nextgen_algorithm='auto',
            use_gpu=False,  # Disable for CI
            privacy_enabled=True,
            monitoring_enabled=True,
            dashboard_port=8082,
            prometheus_port=9092
        )
        self.system = UnifiedFLSystem(self.config)
        
    async def test_complete_fl_round(self):
        """Test complete FL round with all components"""
        # Simulate client updates
        client_updates = self.system.simulate_client_training(num_clients=15)
        
        # Process round
        metrics = await self.system.process_fl_round(client_updates)
        
        # Verify all steps completed
        self.assertIn('avg_quality_score', metrics)
        self.assertIn('avg_reputation', metrics)
        self.assertIn('byzantine_clients', metrics)
        self.assertIn('nextgen_algorithm', metrics)
        self.assertIn('gradient_norm', metrics)
        self.assertIn('privacy_spent', metrics)
        self.assertIn('round_time', metrics)
        self.assertTrue(metrics['success'])
        
        # Check that model was updated
        self.assertIsNotNone(self.system.global_model)
        
    async def test_byzantine_resilience(self):
        """Test system resilience to Byzantine attacks"""
        # Create updates with 40% Byzantine
        num_clients = 20
        gradient_dim = 100
        client_updates = {}
        
        # Honest clients
        for i in range(12):
            client_id = f"honest_{i}"
            client_updates[client_id] = {
                'gradient': np.ones(gradient_dim) + np.random.randn(gradient_dim) * 0.1,
                'loss_before': 2.0,
                'loss_after': 1.0,
                'dataset_size': 1000,
                'computation_time': 10.0,
                'num_steps': 100
            }
        
        # Byzantine clients (various attacks)
        for i in range(8):
            client_id = f"byzantine_{i}"
            if i < 3:
                # Random noise attack
                gradient = np.random.randn(gradient_dim) * 10
            elif i < 6:
                # Sign-flipping attack
                gradient = -np.ones(gradient_dim) * 5
            else:
                # Gradient scaling attack
                gradient = np.ones(gradient_dim) * 100
            
            client_updates[client_id] = {
                'gradient': gradient,
                'loss_before': 1.0,
                'loss_after': 0.99,
                'dataset_size': 100,
                'computation_time': 0.1,
                'num_steps': 10
            }
        
        # Process round
        metrics = await self.system.process_fl_round(client_updates)
        
        # System should detect most Byzantine clients
        self.assertGreater(metrics['byzantine_clients'], 5)
        self.assertTrue(metrics['success'])
        
        # Aggregated gradient should be close to honest average
        honest_gradients = [
            client_updates[f"honest_{i}"]['gradient']
            for i in range(12)
        ]
        honest_avg = np.mean(honest_gradients, axis=0)
        
        # Check distance to honest average
        distance = np.linalg.norm(self.system.global_model - honest_avg)
        self.assertLess(distance, 50)  # Should filter out Byzantine influence
        
    async def test_adaptive_algorithm_selection(self):
        """Test adaptive selection of next-gen algorithms"""
        rounds_data = []
        
        for round_num in range(3):
            # Vary heterogeneity across rounds
            if round_num == 0:
                # High straggler rate
                num_clients = 15
                num_stragglers = 8
            elif round_num == 1:
                # High system heterogeneity
                num_clients = 20
                num_stragglers = 2
            else:
                # Normal conditions
                num_clients = 10
                num_stragglers = 1
            
            client_updates = {}
            for i in range(num_clients):
                client_id = f"client_{i}_r{round_num}"
                
                if i < num_stragglers:
                    # Straggler
                    num_steps = 10
                else:
                    # Normal client
                    num_steps = 100
                
                client_updates[client_id] = {
                    'gradient': np.random.randn(100),
                    'loss_before': 2.0,
                    'loss_after': 1.5,
                    'dataset_size': 1000,
                    'computation_time': num_steps * 0.1,
                    'num_steps': num_steps
                }
            
            metrics = await self.system.process_fl_round(client_updates)
            rounds_data.append(metrics)
        
        # Check that different algorithms were selected
        algorithms_used = [m['nextgen_algorithm'] for m in rounds_data]
        
        # Should use FedProx for high straggler rate
        self.assertEqual(algorithms_used[0], 'fedprox')
        
        # Should adapt to different conditions
        self.assertTrue(len(set(algorithms_used)) >= 2)


class TestEndToEndIntegration(unittest.TestCase):
    """End-to-end integration tests"""
    
    async def test_complete_training_session(self):
        """Test complete FL training session"""
        config = UnifiedConfig(
            num_rounds=5,
            target_accuracy=0.8,
            byzantine_threshold=0.2,
            privacy_enabled=True,
            monitoring_enabled=False  # Disable for faster test
        )
        
        system = UnifiedFLSystem(config)
        
        # Run training
        await system.run_training(num_rounds=3)
        
        # Verify training completed
        self.assertEqual(system.current_round, 3)
        self.assertEqual(len(system.accuracy_history), 3)
        self.assertEqual(len(system.round_metrics_history), 3)
        
        # Check final metrics
        final_metrics = system.round_metrics_history[-1]
        self.assertTrue(final_metrics['success'])
        self.assertIn('privacy_budget_remaining', final_metrics)
        
        # Check that privacy budget was consumed
        initial_budget = config.privacy_budget
        remaining = final_metrics['privacy_budget_remaining']
        self.assertLess(remaining, initial_budget)


def run_all_integration_tests():
    """Run all integration tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPoGQIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestByzantineDetectionWithPoGQ))
    suite.addTests(loader.loadTestsFromTestCase(TestHolochainIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestNextGenAlgorithms))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceOptimizations))
    suite.addTests(loader.loadTestsFromTestCase(TestPrivacyPreservation))
    suite.addTests(loader.loadTestsFromTestCase(TestMonitoringObservability))
    suite.addTests(loader.loadTestsFromTestCase(TestUnifiedSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*80)
    print(" Integration Test Summary")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ All integration tests passed!")
        print("\nThe unified FL system with PoGQ integration is working correctly:")
        print("  • PoGQ quality proofs generated and verified")
        print("  • Byzantine detection with trust weighting operational")
        print("  • Next-gen algorithms adapting to heterogeneity")
        print("  • Performance optimizations improving speed")
        print("  • Privacy preservation maintaining DP guarantees")
        print("  • Monitoring system tracking all metrics")
        print("  • Complete FL rounds executing successfully")
    else:
        print("\n❌ Some tests failed. Please review the output above.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run async tests properly
    import sys
    
    # For async test methods
    async def run_async_tests():
        """Helper to run async test methods"""
        # Run specific async tests
        test_byzantine = TestByzantineDetectionWithPoGQ()
        test_byzantine.setUp()
        await test_byzantine.test_trust_weighted_detection()
        
        test_holochain = TestHolochainIntegration()
        await test_holochain.test_reputation_retrieval()
        await test_holochain.test_reputation_update()
        
        test_perf = TestPerformanceOptimizations()
        test_perf.setUp()
        await test_perf.test_parallel_processing()
        
        test_unified = TestUnifiedSystem()
        test_unified.setUp()
        await test_unified.test_complete_fl_round()
        await test_unified.test_byzantine_resilience()
        await test_unified.test_adaptive_algorithm_selection()
        
        test_e2e = TestEndToEndIntegration()
        await test_e2e.test_complete_training_session()
    
    # Run regular tests
    success = run_all_integration_tests()
    
    # Run async tests
    print("\nRunning async integration tests...")
    asyncio.run(run_async_tests())
    
    sys.exit(0 if success else 1)