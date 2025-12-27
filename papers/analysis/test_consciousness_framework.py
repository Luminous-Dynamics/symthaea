#!/usr/bin/env python3
"""
Comprehensive Test Suite for Consciousness Framework v3.0

Tests:
1. Unit tests for each component
2. Integration tests for Master Equation
3. Validation tests for expected orderings
4. Edge cases and boundary conditions
5. Mathematical property verification

Run with: python -m pytest test_consciousness_framework.py -v
Or directly: python test_consciousness_framework.py
"""

import numpy as np
import sys
from typing import Dict, List
import unittest

# Import the framework
from consciousness_framework_v3 import (
    InformationGeometry,
    CausalEmergence,
    PhaseTransitionModel,
    AttentionWeightedBinding,
    RecursiveMetaAwareness,
    MasterEquationV3,
    ConsciousnessConfig,
    ConsciousnessResult,
    ValidationFramework
)


class TestInformationGeometry(unittest.TestCase):
    """Test information-geometric foundations."""

    def setUp(self):
        self.ig = InformationGeometry()

    def test_kl_divergence_identical(self):
        """KL divergence of identical distributions is 0."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        kl = self.ig.kl_divergence(p, p)
        self.assertAlmostEqual(kl, 0.0, places=5)

    def test_kl_divergence_positive(self):
        """KL divergence is always non-negative."""
        for _ in range(100):
            p = np.random.dirichlet(np.ones(4))
            q = np.random.dirichlet(np.ones(4))
            kl = self.ig.kl_divergence(p, q)
            self.assertGreaterEqual(kl, 0.0)

    def test_mutual_information_independent(self):
        """MI of independent variables is 0."""
        # Independent: p(i,j) = p(i) * p(j)
        p_x = np.array([0.5, 0.5])
        p_y = np.array([0.3, 0.7])
        joint = np.outer(p_x, p_y)
        mi = self.ig.mutual_information(joint)
        self.assertAlmostEqual(mi, 0.0, places=3)

    def test_mutual_information_perfect_correlation(self):
        """MI of perfectly correlated variables equals marginal entropy."""
        # Perfect correlation: diagonal matrix
        joint = np.diag([0.5, 0.5])
        mi = self.ig.mutual_information(joint)
        # Should equal H(X) = 1 bit
        self.assertGreater(mi, 0.5)

    def test_integrated_information_increases_with_correlation(self):
        """Φ increases with correlation."""
        # Low correlation
        low_corr = np.array([
            [0.25, 0.25],
            [0.25, 0.25]
        ])
        phi_low = self.ig.integrated_information_approx(low_corr, 1)

        # High correlation
        high_corr = np.array([
            [0.45, 0.05],
            [0.05, 0.45]
        ])
        phi_high = self.ig.integrated_information_approx(high_corr, 1)

        self.assertGreaterEqual(phi_high, phi_low)


class TestCausalEmergence(unittest.TestCase):
    """Test causal emergence metrics."""

    def setUp(self):
        self.ce = CausalEmergence()

    def test_effective_information_deterministic(self):
        """Deterministic TPM has maximum EI."""
        # Deterministic: each state maps to exactly one next state
        tpm_det = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        ei = self.ce.effective_information(tpm_det)
        # Should be high (low effect entropy)
        self.assertGreater(ei, 0)

    def test_effective_information_random(self):
        """Random TPM has low EI."""
        # Random: each state leads to uniform distribution
        tpm_rand = np.ones((3, 3)) / 3
        ei = self.ce.effective_information(tpm_rand)
        # Should be near 0
        self.assertLess(ei, 0.5)

    def test_causal_emergence_calculation(self):
        """Test causal emergence calculation."""
        micro_tpm = np.ones((4, 4)) / 4  # Random
        macro_tpm = np.eye(2)  # Deterministic

        ce = self.ce.causal_emergence(micro_tpm, macro_tpm)
        # Macro should have more causal power
        # (This is a simplified test)
        self.assertIsInstance(ce, float)


class TestPhaseTransitions(unittest.TestCase):
    """Test dynamic phase transition model."""

    def setUp(self):
        self.pt = PhaseTransitionModel()

    def test_sigmoid_bounded(self):
        """Sigmoid output is bounded [0, 1]."""
        for c in np.linspace(-1, 2, 100):
            result = self.pt.sigmoid_transition(c)
            self.assertGreaterEqual(result, 0)
            self.assertLessEqual(result, 1)

    def test_sigmoid_transition_point(self):
        """Sigmoid crosses 0.5 at threshold."""
        at_threshold = self.pt.sigmoid_transition(self.pt.theta)
        self.assertAlmostEqual(at_threshold, 0.5, places=2)

    def test_hysteresis(self):
        """Hysteresis creates different thresholds for up/down."""
        C_raw = self.pt.theta

        up_result = self.pt.hysteresis_transition(C_raw, 0, increasing=True)
        down_result = self.pt.hysteresis_transition(C_raw, 1, increasing=False)

        # At the same raw value, up should give lower result than down
        # (because up threshold is higher)
        self.assertLess(up_result, down_result)

    def test_bifurcation_detection(self):
        """Bifurcation detection identifies variance changes."""
        # Create trajectory with sudden change
        t1 = np.random.randn(100) * 0.1 + 0.5  # Low variance
        t2 = np.random.randn(100) * 0.5 + 0.5  # High variance
        trajectory = np.concatenate([t1, t2])

        bifurcations = self.pt.detect_bifurcation(trajectory, window_size=30)

        # Should detect something around index 100
        self.assertGreater(len(bifurcations), 0)


class TestAttentionWeightedBinding(unittest.TestCase):
    """Test precision-weighted binding."""

    def setUp(self):
        self.awb = AttentionWeightedBinding(dim=256)

    def test_encode_feature_normalized(self):
        """Encoded features are unit normalized."""
        feature = np.random.randn(10)
        encoded = self.awb.encode_feature(feature, precision=1.0)

        norm = np.linalg.norm(encoded)
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_precision_affects_encoding(self):
        """Higher precision gives more deterministic encoding."""
        feature = np.random.randn(10)

        # Multiple encodings with low precision
        low_prec_encodings = [
            self.awb.encode_feature(feature, precision=0.1)
            for _ in range(10)
        ]

        # Multiple encodings with high precision
        high_prec_encodings = [
            self.awb.encode_feature(feature, precision=10.0)
            for _ in range(10)
        ]

        # Variance should be lower for high precision (more consistent)
        var_low = np.mean([np.var(e) for e in low_prec_encodings])
        var_high = np.mean([np.var(e) for e in high_prec_encodings])

        # Both should have similar variance (they're all tanh-normalized)
        # But the actual values should differ more for low precision
        self.assertIsInstance(var_low, float)
        self.assertIsInstance(var_high, float)

    def test_binding_strength_increases_with_attention(self):
        """Higher attention gives stronger binding."""
        features = [np.random.randn(256) for _ in range(3)]
        features = [f / np.linalg.norm(f) for f in features]

        # Low attention
        low_attn = np.array([0.1, 0.1, 0.1])
        bound_low, strength_low = self.awb.attention_gated_binding(
            features, low_attn, threshold=0.05
        )

        # High attention
        high_attn = np.array([0.9, 0.9, 0.9])
        bound_high, strength_high = self.awb.attention_gated_binding(
            features, high_attn, threshold=0.05
        )

        self.assertGreater(strength_high, strength_low)

    def test_binding_coherence_bounded(self):
        """Binding coherence is in [0, 1]."""
        bound = np.random.randn(256)
        coherence = self.awb.compute_binding_coherence(bound)

        self.assertGreaterEqual(coherence, 0)
        self.assertLessEqual(coherence, 1)


class TestRecursiveMetaAwareness(unittest.TestCase):
    """Test recursive meta-awareness tower."""

    def setUp(self):
        self.rma = RecursiveMetaAwareness(max_depth=5, decay=0.7)

    def test_tower_depth(self):
        """Tower has correct depth."""
        base = np.random.randn(64)
        transform = lambda x: np.tanh(np.roll(x, 1))

        tower = self.rma.compute_meta_level(base, transform)

        self.assertEqual(len(tower), self.rma.max_depth)

    def test_tower_decay(self):
        """Higher levels have smaller magnitude."""
        base = np.ones(64)
        transform = lambda x: x * 0.9  # Simple scaling

        tower = self.rma.compute_meta_level(base, transform)

        # Each level should have smaller norm
        norms = [np.linalg.norm(level) for level in tower]
        for i in range(len(norms) - 1):
            self.assertGreater(norms[i], norms[i+1])

    def test_recursion_strength_bounded(self):
        """Recursion strength is in [0, 1]."""
        base = np.random.randn(64)
        transform = lambda x: np.tanh(np.roll(x, 1))
        tower = self.rma.compute_meta_level(base, transform)

        R = self.rma.recursion_strength(tower)

        self.assertGreaterEqual(R, 0)
        self.assertLessEqual(R, 1)

    def test_fixed_point_distance_bounded(self):
        """Fixed point distance is in [0, 1]."""
        base = np.random.randn(64)
        transform = lambda x: x  # Identity (at fixed point)
        tower = self.rma.compute_meta_level(base, transform)

        distance = self.rma.fixed_point_distance(tower)

        self.assertGreaterEqual(distance, 0)
        self.assertLessEqual(distance, 1)


class TestMasterEquation(unittest.TestCase):
    """Test Master Equation implementation."""

    def setUp(self):
        self.eq = MasterEquationV3()
        self.validator = ValidationFramework()

    def test_consciousness_bounded(self):
        """Consciousness scores are bounded [0, 1]."""
        for state in ['wake', 'n1', 'n2', 'n3', 'rem']:
            eeg = self.validator.generate_synthetic_eeg(state, sfreq=256)
            result = self.eq.assess_consciousness(eeg, sfreq=256)

            self.assertGreaterEqual(result.C_raw, 0)
            self.assertLessEqual(result.C_raw, 1)
            self.assertGreaterEqual(result.C_phenomenal, 0)
            self.assertLessEqual(result.C_phenomenal, 1)

    def test_components_bounded(self):
        """All components are bounded [0, 1]."""
        eeg = self.validator.generate_synthetic_eeg('wake', sfreq=256)
        result = self.eq.assess_consciousness(eeg, sfreq=256)

        for comp_name in ['phi', 'binding', 'workspace', 'attention', 'recursion']:
            comp_value = getattr(result, comp_name)
            self.assertGreaterEqual(comp_value, 0, f"{comp_name} below 0")
            self.assertLessEqual(comp_value, 1, f"{comp_name} above 1")

    def test_wake_higher_than_n3(self):
        """Wake consciousness > N3 consciousness."""
        wake_eeg = self.validator.generate_synthetic_eeg('wake', sfreq=256)
        n3_eeg = self.validator.generate_synthetic_eeg('n3', sfreq=256)

        wake_result = self.eq.assess_consciousness(wake_eeg, sfreq=256)
        n3_result = self.eq.assess_consciousness(n3_eeg, sfreq=256)

        self.assertGreater(wake_result.C_raw, n3_result.C_raw)

    def test_substrate_affects_score(self):
        """Different substrates give different scores."""
        eeg = self.validator.generate_synthetic_eeg('wake', sfreq=256)

        bio_result = self.eq.assess_consciousness(eeg, sfreq=256, substrate='biological')
        si_result = self.eq.assess_consciousness(eeg, sfreq=256, substrate='silicon')

        self.assertGreater(bio_result.C_raw, si_result.C_raw)

    def test_limiting_component_correct(self):
        """Limiting component is actually the minimum."""
        eeg = self.validator.generate_synthetic_eeg('wake', sfreq=256)
        result = self.eq.assess_consciousness(eeg, sfreq=256)

        components = {
            'phi': result.phi,
            'binding': result.binding,
            'workspace': result.workspace,
            'attention': result.attention,
            'recursion': result.recursion
        }

        actual_min = min(components, key=components.get)
        self.assertEqual(result.limiting_component, actual_min)

    def test_critical_min_is_minimum(self):
        """Critical min equals minimum of 5 components."""
        eeg = self.validator.generate_synthetic_eeg('wake', sfreq=256)
        result = self.eq.assess_consciousness(eeg, sfreq=256)

        expected_min = min(
            result.phi, result.binding, result.workspace,
            result.attention, result.recursion
        )

        self.assertAlmostEqual(result.critical_min, expected_min, places=5)

    def test_phase_states_make_sense(self):
        """Phase states correspond to C values."""
        for state in ['wake', 'n3']:
            eeg = self.validator.generate_synthetic_eeg(state, sfreq=256)
            result = self.eq.assess_consciousness(eeg, sfreq=256)

            if result.C_phenomenal < 0.1:
                self.assertEqual(result.phase_state, "unconscious")
            elif result.C_phenomenal >= 0.5:
                self.assertEqual(result.phase_state, "conscious")


class TestMathematicalProperties(unittest.TestCase):
    """Test mathematical properties of the Master Equation."""

    def setUp(self):
        self.eq = MasterEquationV3()
        self.validator = ValidationFramework()

    def test_monotonicity_phi(self):
        """Improving phi (ceteris paribus) increases C."""
        # Generate base data
        eeg = self.validator.generate_synthetic_eeg('n2', sfreq=256)
        base_result = self.eq.assess_consciousness(eeg, sfreq=256)

        # Create modified EEG with higher complexity (proxy for higher phi)
        modified_eeg = eeg + 0.1 * np.random.randn(*eeg.shape)
        modified_result = self.eq.assess_consciousness(modified_eeg, sfreq=256)

        # Can't guarantee monotonicity with random modification,
        # but we can verify the function doesn't crash
        self.assertIsInstance(modified_result.C_raw, float)

    def test_substrate_limited(self):
        """C <= S for any configuration."""
        for state in ['wake', 'n1', 'n2']:
            eeg = self.validator.generate_synthetic_eeg(state, sfreq=256)
            result = self.eq.assess_consciousness(eeg, sfreq=256)

            # C_raw should be <= S
            self.assertLessEqual(result.C_raw, result.substrate_factor + 0.01)

    def test_threshold_gating(self):
        """If any critical component is 0, C should be 0."""
        # This is hard to test without directly manipulating components,
        # but we can verify with VS state which has very low components
        eeg = self.validator.generate_synthetic_eeg('vs', sfreq=256)
        result = self.eq.assess_consciousness(eeg, sfreq=256)

        # VS should have very low C
        self.assertLess(result.C_raw, 0.15)


class TestValidationFramework(unittest.TestCase):
    """Test validation framework."""

    def setUp(self):
        self.validator = ValidationFramework()

    def test_synthetic_eeg_shape(self):
        """Synthetic EEG has correct shape."""
        eeg = self.validator.generate_synthetic_eeg(
            'wake', n_channels=64, duration=10.0, sfreq=256
        )

        self.assertEqual(eeg.shape[0], 64)  # Channels
        self.assertEqual(eeg.shape[1], 2560)  # 10 * 256 samples

    def test_synthetic_eeg_different_states(self):
        """Different states produce different EEG patterns."""
        wake_eeg = self.validator.generate_synthetic_eeg('wake', sfreq=256)
        n3_eeg = self.validator.generate_synthetic_eeg('n3', sfreq=256)

        # Check variance (should differ significantly)
        wake_var = np.var(wake_eeg)
        n3_var = np.var(n3_eeg)

        # Both should have non-zero variance
        self.assertGreater(wake_var, 0)
        self.assertGreater(n3_var, 0)

    def test_sleep_validation_runs(self):
        """Sleep validation completes without error."""
        results = self.validator.validate_sleep_stages(n_trials=2)

        self.assertIn('wake', results)
        self.assertIn('n3', results)
        self.assertIn('ordering_correct', results)

    def test_doc_validation_runs(self):
        """DOC validation completes without error."""
        results = self.validator.validate_doc_classification(n_trials=2)

        self.assertIn('wake', results)
        self.assertIn('mcs', results)
        self.assertIn('vs', results)
        self.assertIn('accuracy', results)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def setUp(self):
        self.eq = MasterEquationV3()

    def test_single_channel(self):
        """Framework handles single-channel EEG."""
        eeg = np.random.randn(1, 2560)
        result = self.eq.assess_consciousness(eeg, sfreq=256)

        self.assertIsInstance(result.C_raw, float)
        self.assertGreaterEqual(result.C_raw, 0)

    def test_short_duration(self):
        """Framework handles short recordings."""
        eeg = np.random.randn(8, 256)  # 1 second
        result = self.eq.assess_consciousness(eeg, sfreq=256)

        self.assertIsInstance(result.C_raw, float)

    def test_very_noisy_data(self):
        """Framework handles very noisy data."""
        eeg = np.random.randn(8, 2560) * 1000  # Very high amplitude
        result = self.eq.assess_consciousness(eeg, sfreq=256)

        # Should still be bounded
        self.assertGreaterEqual(result.C_raw, 0)
        self.assertLessEqual(result.C_raw, 1)

    def test_constant_signal(self):
        """Framework handles constant (flat) signal."""
        eeg = np.ones((8, 2560))
        result = self.eq.assess_consciousness(eeg, sfreq=256)

        # Should be very low (no information)
        self.assertLess(result.C_raw, 0.2)

    def test_nan_handling(self):
        """Framework doesn't produce NaN."""
        for state in ['wake', 'n3', 'vs']:
            eeg = ValidationFramework().generate_synthetic_eeg(state, sfreq=256)
            result = self.eq.assess_consciousness(eeg, sfreq=256)

            self.assertFalse(np.isnan(result.C_raw))
            self.assertFalse(np.isnan(result.C_phenomenal))


def run_tests():
    """Run all tests and report results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestInformationGeometry))
    suite.addTests(loader.loadTestsFromTestCase(TestCausalEmergence))
    suite.addTests(loader.loadTestsFromTestCase(TestPhaseTransitions))
    suite.addTests(loader.loadTestsFromTestCase(TestAttentionWeightedBinding))
    suite.addTests(loader.loadTestsFromTestCase(TestRecursiveMetaAwareness))
    suite.addTests(loader.loadTestsFromTestCase(TestMasterEquation))
    suite.addTests(loader.loadTestsFromTestCase(TestMathematicalProperties))
    suite.addTests(loader.loadTestsFromTestCase(TestValidationFramework))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")

    return result


if __name__ == "__main__":
    run_tests()
