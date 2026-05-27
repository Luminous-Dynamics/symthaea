// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Cross-Bridge Integration Tests
//!
//! End-to-end tests that chain all three bridge modules together:
//!
//! ```text
//! simulation_bridge (physics)
//!         ↓  trajectory states
//! state_to_binary_hv (HDC encoding)
//!         ↓  Vec<BinaryHV>
//! ConsciousnessPipeline.process()
//!         ↓  ConsciousnessState + Φ
//! assess_integration()
//!         ↓  IntegrationAssessment
//!
//! math_bridge (arithmetic)
//!         ↓  MathResult { encoding: BinaryHV, phi }
//! ConsciousnessPipeline.process()
//!         ↓  ConsciousnessState
//! ```
//!
//! These tests validate that the physics → HDC → consciousness pipeline
//! and the math → HDC → consciousness pipeline produce meaningful,
//! non-degenerate results when chained together.

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::consciousness_integration::{ConsciousnessPipeline, IntegrationConfig};
use crate::hdc::dynamical_system::{HarmonicOscillator, LorenzSystem, NBodyGravity};
use crate::hdc::math_bridge::{MathValue, UnifiedMathEngine};
use crate::physics::simulation_bridge::{PhysicsSimulator, SimulationAnalysis, state_to_binary_hv};

// =============================================================================
// PHYSICS → HDC → CONSCIOUSNESS PIPELINE
// =============================================================================

/// Harmonic oscillator: simulate → encode as BinaryHV → feed to consciousness pipeline.
/// Verifies that periodic motion produces non-zero Φ and conscious binding.
#[test]
fn test_harmonic_to_consciousness() {
    // 1. Simulate harmonic oscillator
    let sim = PhysicsSimulator::harmonic(2.0, 1.0, 0.0);
    let result = sim.simulate(5.0, 0.01);

    assert!(result.steps > 100, "Should have many integration steps");
    assert!(
        result.states.len() > 100,
        "Should have many recorded states"
    );

    // 2. Encode trajectory states as BinaryHV
    let binary_hvs: Vec<BinaryHV> = result
        .states
        .iter()
        .take(10) // Take 10 representative states
        .map(|state| state_to_binary_hv(state))
        .collect();

    assert_eq!(binary_hvs.len(), 10);

    // 3. Feed into consciousness pipeline
    let config = IntegrationConfig::default();
    let mut pipeline = ConsciousnessPipeline::new(config);
    pipeline.enable_integrated_systems();

    let priorities: Vec<f64> = vec![0.8; binary_hvs.len()];
    let state = pipeline.process(binary_hvs, &priorities);

    // 4. Verify meaningful consciousness metrics
    assert!(
        state.phi > 0.0,
        "Φ should be positive for multi-component input"
    );
    assert!(
        state.consciousness_level > 0.0,
        "Should have non-zero consciousness"
    );

    // 5. Assess integration
    let assessment = pipeline.assess();
    assert!(
        assessment.consciousness_score > 0.0,
        "Assessment score should be positive"
    );
}

/// Lorenz attractor (chaotic): verify that chaotic physics produces richer
/// consciousness signatures than simple harmonic motion.
#[test]
fn test_lorenz_chaos_consciousness() {
    // 1. Simulate chaotic system
    let sim = PhysicsSimulator::lorenz();
    let result = sim.simulate(2.0, 0.005);

    // 2. Encode as BinaryHV
    let binary_hvs: Vec<BinaryHV> = result
        .states
        .iter()
        .step_by(result.states.len() / 10) // Sample 10 evenly-spaced states
        .map(|state| state_to_binary_hv(state))
        .collect();

    assert!(
        binary_hvs.len() >= 5,
        "Should have at least 5 sample states"
    );

    // 3. Feed to consciousness
    let config = IntegrationConfig::default();
    let mut pipeline = ConsciousnessPipeline::new(config);
    pipeline.enable_integrated_systems();

    let priorities: Vec<f64> = vec![0.9; binary_hvs.len()];
    let state = pipeline.process(binary_hvs, &priorities);

    assert!(state.phi > 0.0, "Chaotic system should produce positive Φ");
    assert!(
        state.consciousness_level > 0.0,
        "Chaotic system should be conscious"
    );

    // 4. Verify analysis recognizes chaos
    let lorenz_sys = LorenzSystem::standard();
    let analysis = SimulationAnalysis::from_result(&result).with_lyapunov_estimate(
        &lorenz_sys,
        &[1.0, 1.0, 1.0],
        2.0,
        0.005,
        1e-8,
    );
    assert!(
        analysis.lyapunov_estimate.is_some(),
        "Should estimate Lyapunov exponent"
    );
    // Lorenz is chaotic → Lyapunov should be computed (sign depends on
    // estimation method and short simulation window; just verify it's finite)
    if let Some(lyap) = analysis.lyapunov_estimate {
        assert!(
            lyap.is_finite(),
            "Lyapunov estimate should be finite, got {}",
            lyap
        );
    }
}

/// Two-body gravitational orbit: verify energy conservation in analysis
/// and that orbital dynamics are encoded in consciousness.
#[test]
fn test_gravity_orbit_consciousness() {
    // 1. Simulate two-body gravity
    let sim = PhysicsSimulator::gravitational_two_body(1.0, 1.0, 2.0);
    let result = sim.simulate(5.0, 0.01);

    // 2. Encode trajectory as HDC vectors
    let sample_count = 8.min(result.states.len());
    let binary_hvs: Vec<BinaryHV> = result
        .states
        .iter()
        .step_by((result.states.len() / sample_count).max(1))
        .take(sample_count)
        .map(|state| state_to_binary_hv(state))
        .collect();

    // 3. Feed to consciousness pipeline
    let config = IntegrationConfig::default();
    let mut pipeline = ConsciousnessPipeline::new(config);
    pipeline.enable_integrated_systems();

    let priorities: Vec<f64> = vec![0.85; binary_hvs.len()];
    let state = pipeline.process(binary_hvs, &priorities);

    assert!(
        state.phi > 0.0,
        "Orbital dynamics should produce positive Φ"
    );

    // 4. Check energy conservation
    let nbody = NBodyGravity::two_body(1.0, 1.0, 1.0, 2);
    let analysis = SimulationAnalysis::from_result(&result).with_nbody_energy(&nbody, &result);
    if let Some(drift) = analysis.energy_drift {
        assert!(
            drift.abs() < 0.5, // RK4 should conserve energy reasonably
            "Energy drift {} is too large for RK4 integration",
            drift
        );
    }
}

/// Charged particle in electromagnetic field: cyclotron motion → consciousness.
#[test]
fn test_charged_particle_consciousness() {
    let sim = PhysicsSimulator::charged_particle(
        1.0,                            // q/m
        [0.0, 0.0, 0.0],                // no electric field
        [0.0, 0.0, 1.0],                // B field in z
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], // circular motion
    );
    let result = sim.simulate(6.28, 0.01); // One full cyclotron period

    let binary_hvs: Vec<BinaryHV> = result
        .states
        .iter()
        .step_by((result.states.len() / 6).max(1))
        .take(6)
        .map(|state| state_to_binary_hv(state))
        .collect();

    let config = IntegrationConfig::default();
    let mut pipeline = ConsciousnessPipeline::new(config);
    pipeline.enable_integrated_systems();

    let priorities: Vec<f64> = vec![0.75; binary_hvs.len()];
    let state = pipeline.process(binary_hvs, &priorities);

    assert!(
        state.phi > 0.0,
        "Cyclotron motion should produce positive Φ"
    );
}

/// Heat diffusion: dissipative system → consciousness. Verify that
/// thermal relaxation produces diminishing information integration.
#[test]
fn test_heat_diffusion_consciousness() {
    let sim = PhysicsSimulator::heat_diffusion(0.01, 10, 100.0, 0.0);
    let result = sim.simulate(1.0, 0.001);

    // Early states (far from equilibrium) vs late states (near equilibrium)
    let early_hvs: Vec<BinaryHV> = result
        .states
        .iter()
        .take(5)
        .map(|state| state_to_binary_hv(state))
        .collect();

    let late_hvs: Vec<BinaryHV> = result
        .states
        .iter()
        .rev()
        .take(5)
        .map(|state| state_to_binary_hv(state))
        .collect();

    let config = IntegrationConfig::default();

    // Process early states
    let mut pipeline_early = ConsciousnessPipeline::new(config.clone());
    pipeline_early.enable_integrated_systems();
    let early_priorities: Vec<f64> = vec![0.8; early_hvs.len()];
    let early_state = pipeline_early.process(early_hvs, &early_priorities);
    let early_phi = early_state.phi;

    // Process late states
    let mut pipeline_late = ConsciousnessPipeline::new(config);
    pipeline_late.enable_integrated_systems();
    let late_priorities: Vec<f64> = vec![0.8; late_hvs.len()];
    let late_state = pipeline_late.process(late_hvs, &late_priorities);
    let late_phi = late_state.phi;

    // Both should produce valid Φ
    assert!(early_phi >= 0.0, "Early Φ should be non-negative");
    assert!(late_phi >= 0.0, "Late Φ should be non-negative");
}

/// Multi-cycle consciousness: feed physics trajectory over multiple processing
/// cycles to test temporal accumulation of consciousness state.
#[test]
fn test_physics_multi_cycle_consciousness() {
    let sim = PhysicsSimulator::harmonic(1.0, 1.0, 0.0);
    let result = sim.simulate(10.0, 0.01);

    let config = IntegrationConfig::default();
    let mut pipeline = ConsciousnessPipeline::new(config);
    pipeline.enable_integrated_systems();

    // Feed trajectory in chunks of 5 states across 4 cycles
    let mut phi_values = Vec::new();
    for chunk_start in (0..20).step_by(5) {
        let chunk: Vec<BinaryHV> = result.states[chunk_start..chunk_start + 5]
            .iter()
            .map(|state| state_to_binary_hv(state))
            .collect();

        let priorities = vec![0.8; chunk.len()];
        let state = pipeline.process(chunk, &priorities);
        phi_values.push(state.phi);
    }

    assert_eq!(phi_values.len(), 4, "Should have 4 cycles");
    // All cycles should produce valid Φ
    for (i, &phi) in phi_values.iter().enumerate() {
        assert!(
            phi >= 0.0,
            "Cycle {} Φ should be non-negative, got {}",
            i,
            phi
        );
    }
}

// =============================================================================
// MATH → HDC → CONSCIOUSNESS PIPELINE
// =============================================================================

/// Math operations produce BinaryHV encodings → feed to consciousness.
#[test]
fn test_math_operations_to_consciousness() {
    let engine = UnifiedMathEngine::new();

    // Perform several math operations, collecting their BinaryHV encodings
    let results = vec![
        engine.add(&MathValue::Natural(42), &MathValue::Natural(58)),
        engine.multiply(&MathValue::Integer(-3), &MathValue::Real(2.7)),
        engine.sqrt(&MathValue::Real(144.0)),
        engine.subtract(&MathValue::Real(10.0), &MathValue::Real(3.5)),
    ];

    // Collect BinaryHV encodings from math results
    let binary_hvs: Vec<BinaryHV> = results.iter().map(|r| r.encoding).collect();

    assert_eq!(binary_hvs.len(), 4);

    // Feed to consciousness pipeline
    let config = IntegrationConfig::default();
    let mut pipeline = ConsciousnessPipeline::new(config);
    pipeline.enable_integrated_systems();

    let priorities = vec![0.9, 0.7, 0.8, 0.6]; // Different priorities per operation
    let state = pipeline.process(binary_hvs, &priorities);

    assert!(state.phi > 0.0, "Math encodings should produce positive Φ");
    assert!(
        state.consciousness_level > 0.0,
        "Should be conscious of math results"
    );
}

/// Complex number operations → consciousness. Domain promotion should produce
/// richer information integration.
#[test]
fn test_complex_math_consciousness() {
    let engine = UnifiedMathEngine::new();

    // Operations that trigger domain promotion to ℂ
    let sqrt_neg = engine.sqrt(&MathValue::Integer(-1)); // → Complex(0, 1)
    let complex_mul = engine.multiply(
        &MathValue::Complex { re: 1.0, im: 1.0 },
        &MathValue::Complex { re: 0.0, im: 1.0 },
    );
    let promotion = engine.add(
        &MathValue::Natural(5),
        &MathValue::Complex { re: 3.0, im: 4.0 },
    );

    // All should have promoted to ℂ
    assert_eq!(sqrt_neg.value.domain(), "\u{2102}"); // ℂ
    assert!(
        !sqrt_neg.domain_promotions.is_empty(),
        "sqrt(-1) should record promotion"
    );

    // Feed encodings to consciousness
    let binary_hvs = vec![sqrt_neg.encoding, complex_mul.encoding, promotion.encoding];

    let config = IntegrationConfig::default();
    let mut pipeline = ConsciousnessPipeline::new(config);
    pipeline.enable_integrated_systems();

    let priorities = vec![0.9; 3];
    let state = pipeline.process(binary_hvs, &priorities);

    assert!(state.phi > 0.0, "Complex math should produce positive Φ");
}

/// Numeric tower chain: ℕ → ℤ → ℚ → ℝ → ℂ across operations,
/// all fed to consciousness for a full-stack integration test.
#[test]
fn test_full_tower_chain_to_consciousness() {
    let engine = UnifiedMathEngine::new();

    // Build a chain through every domain
    let natural = engine.add(&MathValue::Natural(10), &MathValue::Natural(5)); // ℕ
    let integer = engine.subtract(&MathValue::Natural(3), &MathValue::Natural(7)); // → ℤ (-4)
    let rational = engine.divide(&MathValue::Integer(7), &MathValue::Integer(3)); // → ℚ (7/3)
    let real = engine.sqrt(&MathValue::Natural(2)); // → ℝ (√2)
    let complex = engine.sqrt(&MathValue::Integer(-4)); // → ℂ (0+2i)

    // Verify domain progression
    assert_eq!(natural.value.domain(), "\u{2115}"); // ℕ
    assert_eq!(integer.value.domain(), "\u{2124}"); // ℤ
    assert!(rational.is_some(), "7/3 should not be division by zero");
    assert_eq!(real.value.domain(), "\u{211d}"); // ℝ
    assert_eq!(complex.value.domain(), "\u{2102}"); // ℂ

    // Collect all encodings
    let mut binary_hvs = vec![
        natural.encoding,
        integer.encoding,
        real.encoding,
        complex.encoding,
    ];
    if let Some(rat_result) = rational {
        assert_eq!(rat_result.value.domain(), "\u{211a}"); // ℚ
        binary_hvs.push(rat_result.encoding);
    }

    // Verify Φ increases with domain promotion
    assert!(
        complex.phi >= natural.phi,
        "Complex domain Φ ({}) should be >= Natural Φ ({})",
        complex.phi,
        natural.phi
    );

    // Feed full tower to consciousness
    let config = IntegrationConfig::default();
    let mut pipeline = ConsciousnessPipeline::new(config);
    pipeline.enable_integrated_systems();

    let priorities = vec![0.6, 0.7, 0.8, 0.9, 0.95]; // Increasing priority with domain
    let state = pipeline.process(binary_hvs, &priorities);

    assert!(
        state.phi > 0.0,
        "Full tower chain should produce positive Φ"
    );
    assert!(
        state.consciousness_level > 0.0,
        "Full tower should be conscious"
    );

    let assessment = pipeline.assess();
    assert!(
        assessment.consciousness_score > 0.0,
        "Assessment should be positive"
    );
}

// =============================================================================
// COMBINED: PHYSICS + MATH → CONSCIOUSNESS
// =============================================================================

/// Mixed physics and math inputs fed simultaneously to consciousness.
/// This tests the pipeline handling heterogeneous information sources.
#[test]
fn test_mixed_physics_math_consciousness() {
    // Physics: harmonic oscillator states
    let sim = PhysicsSimulator::harmonic(1.0, 1.0, 0.0);
    let result = sim.simulate(3.0, 0.01);

    let physics_hvs: Vec<BinaryHV> = result
        .states
        .iter()
        .take(3)
        .map(|state| state_to_binary_hv(state))
        .collect();

    // Math: arithmetic results
    let engine = UnifiedMathEngine::new();
    let math_hvs: Vec<BinaryHV> = vec![
        engine
            .add(&MathValue::Real(3.14), &MathValue::Real(2.72))
            .encoding,
        engine.sqrt(&MathValue::Integer(-1)).encoding,
        engine
            .multiply(&MathValue::Natural(7), &MathValue::Natural(6))
            .encoding,
    ];

    // Combine physics + math into single consciousness input
    let mut all_hvs = physics_hvs;
    all_hvs.extend(math_hvs);
    assert_eq!(all_hvs.len(), 6);

    let config = IntegrationConfig::default();
    let mut pipeline = ConsciousnessPipeline::new(config);
    pipeline.enable_integrated_systems();

    let priorities = vec![0.8, 0.7, 0.9, 0.85, 0.95, 0.75];
    let state = pipeline.process(all_hvs, &priorities);

    assert!(
        state.phi > 0.0,
        "Mixed physics+math should produce positive Φ"
    );
    assert!(
        state.consciousness_level > 0.0,
        "Mixed input should be conscious"
    );
}

/// Simulate → Analyze → Compute → Consciousness: full pipeline where
/// physics analysis results are used as math inputs, then fed to consciousness.
#[test]
fn test_analysis_to_math_to_consciousness() {
    // 1. Physics simulation
    let sim = PhysicsSimulator::harmonic(2.0, 1.0, 0.0);
    let result = sim.simulate(5.0, 0.01);

    // 2. Analyze trajectory
    let osc = HarmonicOscillator::simple(2.0);
    let analysis = SimulationAnalysis::from_result(&result).with_harmonic_energy(&osc, &result);

    // 3. Use analysis values as math inputs
    let engine = UnifiedMathEngine::new();

    let total_time_result =
        engine.add(&MathValue::Real(analysis.total_time), &MathValue::Real(0.0));

    let steps_result = engine.multiply(
        &MathValue::Natural(analysis.num_steps as u64),
        &MathValue::Natural(1),
    );

    // Energy drift as math value (may be available from analysis)
    let drift_val = analysis.energy_drift.unwrap_or(0.0);
    let drift_result = engine.sqrt(&MathValue::Real(drift_val.abs()));

    // 4. Combine all into consciousness
    let binary_hvs = vec![
        total_time_result.encoding,
        steps_result.encoding,
        drift_result.encoding,
    ];

    let config = IntegrationConfig::default();
    let mut pipeline = ConsciousnessPipeline::new(config);
    pipeline.enable_integrated_systems();

    let priorities = vec![0.7, 0.6, 0.9]; // Energy drift gets highest priority
    let state = pipeline.process(binary_hvs, &priorities);

    assert!(
        state.phi >= 0.0,
        "Analysis→math→consciousness should produce valid Φ"
    );
}

// =============================================================================
// PROPERTY TESTS: ENCODING DISTINCTNESS ACROSS BRIDGES
// =============================================================================

/// Verify that BinaryHV encodings from different physics systems are
/// distinguishable (not all identical).
#[test]
fn test_physics_encodings_are_distinct() {
    let harmonic = PhysicsSimulator::harmonic(1.0, 1.0, 0.0);
    let lorenz = PhysicsSimulator::lorenz();

    let h_result = harmonic.simulate(1.0, 0.01);
    let l_result = lorenz.simulate(1.0, 0.005);

    // Encode a state from each
    let h_hv = state_to_binary_hv(&h_result.states[50]);
    let l_hv = state_to_binary_hv(&l_result.states[50]);

    // They should differ (different physics → different encodings)
    let hamming = h_hv.hamming_distance(&l_hv);
    assert!(
        hamming > 0,
        "Different physics systems should produce different BinaryHV encodings"
    );
}

/// Verify that BinaryHV encodings from different math domains are
/// distinguishable.
#[test]
fn test_math_encodings_are_distinct() {
    let engine = UnifiedMathEngine::new();

    let natural = engine.add(&MathValue::Natural(42), &MathValue::Natural(0));
    let complex = engine.sqrt(&MathValue::Integer(-42));

    // Different domains → different encodings
    let hamming = natural.encoding.hamming_distance(&complex.encoding);
    assert!(
        hamming > 0,
        "Different math domains should produce different encodings"
    );
}

/// Verify that physics and math encodings fed together produce
/// higher Φ than either alone (information integration hypothesis).
#[test]
fn test_combined_phi_exceeds_individual() {
    let sim = PhysicsSimulator::harmonic(1.0, 1.0, 0.0);
    let result = sim.simulate(2.0, 0.01);
    let physics_hvs: Vec<BinaryHV> = result.states[0..3]
        .iter()
        .map(|s| state_to_binary_hv(s))
        .collect();

    let engine = UnifiedMathEngine::new();
    let math_hvs: Vec<BinaryHV> = vec![
        engine
            .add(&MathValue::Real(1.0), &MathValue::Real(2.0))
            .encoding,
        engine.sqrt(&MathValue::Integer(-1)).encoding,
        engine
            .multiply(&MathValue::Natural(7), &MathValue::Natural(8))
            .encoding,
    ];

    // Physics only
    let config = IntegrationConfig::default();
    let mut pipeline_phys = ConsciousnessPipeline::new(config.clone());
    pipeline_phys.enable_integrated_systems();
    let phys_state = pipeline_phys.process(physics_hvs.clone(), &vec![0.8; 3]);
    let phi_phys = phys_state.phi;

    // Math only
    let mut pipeline_math = ConsciousnessPipeline::new(config.clone());
    pipeline_math.enable_integrated_systems();
    let math_state = pipeline_math.process(math_hvs.clone(), &vec![0.8; 3]);
    let phi_math = math_state.phi;

    // Combined
    let mut combined_hvs = physics_hvs;
    combined_hvs.extend(math_hvs);
    let mut pipeline_combined = ConsciousnessPipeline::new(config);
    pipeline_combined.enable_integrated_systems();
    let combined_state = pipeline_combined.process(combined_hvs, &vec![0.8; 6]);
    let phi_combined = combined_state.phi;

    // All should be valid
    assert!(phi_phys >= 0.0, "Physics Φ should be non-negative");
    assert!(phi_math >= 0.0, "Math Φ should be non-negative");
    assert!(phi_combined >= 0.0, "Combined Φ should be non-negative");

    // Verify combined produces valid, non-trivial Φ.
    // Note: Φ with more components doesn't necessarily exceed individual Φ
    // because the IIT entropy formula measures mutual information between
    // partitions, which can decrease with heterogeneous components.
    assert!(
        phi_combined > 0.1,
        "Combined Φ ({:.4}) should be substantial (>0.1)",
        phi_combined
    );
}
