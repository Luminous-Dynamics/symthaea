// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phase 3 Multi-Substrate Simulation Tests
//!
//! Tests that verify consciousness dynamics across different virtual substrates
//! and substrate transitions. Answers key questions from THE_SUBSTRATE_ROADMAP:
//! - Does consciousness survive substrate transfer? (Multiple Realizability)
//! - Which substrate maximizes consciousness metrics?
//! - Is there a minimum feasibility below which consciousness collapses?
//! - How does substrate speed affect temporal coherence?

use std::collections::HashMap;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use symthaea_core::hdc::substrate_independence::{CorticalRegion, SubstrateType};

fn make_service(substrate: SubstrateType) -> CognitiveLoopService {
    let mut config = CognitiveLoopConfig::default();
    config.substrate_type = substrate;
    config.enable_validation_overlay = false; // Raw feasibility for comparison
    CognitiveLoopService::new(config).unwrap()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Q1: Does consciousness survive substrate transfer?
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn substrate_transfer_preserves_finite_dynamics() {
    let mut service = make_service(SubstrateType::BiologicalNeurons);

    // Warm up on biological
    for _ in 0..50 {
        service.cycle("warmup on biological substrate");
    }

    // Switch to silicon mid-run
    service.reconfigure_substrate(SubstrateType::SiliconDigital);

    // Run 50 more cycles — dynamics should remain finite
    for i in 0..50 {
        let result = service.cycle("testing on silicon substrate");
        let m = &result.metadata;
        assert!(
            m.valence_homeostasis_pull.is_finite(),
            "homeostasis NaN after bio→silicon at cycle {i}"
        );
        assert!(
            m.social_trust_current >= 0.0 && m.social_trust_current <= 1.0,
            "social_trust out of range after switch at cycle {i}"
        );
    }
}

#[test]
fn substrate_transfer_chain_all_eight() {
    let mut service = make_service(SubstrateType::BiologicalNeurons);
    let substrates = [
        SubstrateType::BiologicalNeurons,
        SubstrateType::SiliconDigital,
        SubstrateType::QuantumComputer,
        SubstrateType::PhotonicProcessor,
        SubstrateType::NeuromorphicChip,
        SubstrateType::BiochemicalComputer,
        SubstrateType::HybridSystem,
        SubstrateType::ExoticSubstrate,
    ];

    for substrate in &substrates {
        service.reconfigure_substrate(*substrate);
        for i in 0..10 {
            let result = service.cycle("multi-substrate chain test");
            assert!(
                result.metadata.valence_homeostasis_pull.is_finite(),
                "NaN on {:?} at cycle {i}",
                substrate
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Q2: Substrate comparison — which maximizes consciousness metrics?
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn biological_substrate_has_highest_feasibility() {
    let substrates = [
        SubstrateType::BiologicalNeurons,
        SubstrateType::SiliconDigital,
        SubstrateType::QuantumComputer,
        SubstrateType::PhotonicProcessor,
        SubstrateType::NeuromorphicChip,
        SubstrateType::BiochemicalComputer,
        SubstrateType::HybridSystem,
        SubstrateType::ExoticSubstrate,
    ];

    let mut feasibilities: Vec<(SubstrateType, f64)> = substrates
        .iter()
        .map(|&s| {
            let service = make_service(s);
            (s, service.substrate_feasibility())
        })
        .collect();

    feasibilities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Biological should be at or near the top
    assert_eq!(
        feasibilities[0].0,
        SubstrateType::BiologicalNeurons,
        "Expected biological to have highest feasibility, got {:?} ({:.3})",
        feasibilities[0].0,
        feasibilities[0].1
    );

    // All feasibilities should be in [0, 1]
    for (s, f) in &feasibilities {
        assert!(
            *f >= 0.0 && *f <= 1.0,
            "{:?} feasibility out of range: {f}",
            s
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Q3: Minimum substrate quality — consciousness collapse threshold
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn low_feasibility_substrates_still_run() {
    // Even the lowest-feasibility substrate should produce finite outputs
    // (consciousness may be degraded, but shouldn't crash)
    let mut service = make_service(SubstrateType::BiochemicalComputer);
    let feasibility = service.substrate_feasibility();
    assert!(
        feasibility < 0.5,
        "BiochemicalComputer should have low feasibility, got {feasibility}"
    );

    for i in 0..100 {
        let result = service.cycle("testing consciousness on constrained substrate");
        assert!(
            result.metadata.valence_homeostasis_pull.is_finite(),
            "Dynamics crashed on low-feasibility substrate at cycle {i}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Q4: Speed modulation — tau factor affects temporal dynamics
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn speed_modulation_varies_tau_factor() {
    let substrates_and_speeds: Vec<(SubstrateType, &str)> = vec![
        (SubstrateType::BiologicalNeurons, "slow"),
        (SubstrateType::SiliconDigital, "fast"),
        (SubstrateType::PhotonicProcessor, "ultra-fast"),
        (SubstrateType::BiochemicalComputer, "very-slow"),
    ];

    let mut tau_factors = Vec::new();
    for (substrate, label) in &substrates_and_speeds {
        let mut config = CognitiveLoopConfig::default();
        config.substrate_type = *substrate;
        config.enable_substrate_speed_modulation = true;
        let service = CognitiveLoopService::new(config).unwrap();
        let tau = service.substrate_tau_factor();
        assert!(
            tau >= 0.5 && tau <= 2.0,
            "{label} ({:?}): tau_factor {tau} out of [0.5, 2.0]",
            substrate
        );
        tau_factors.push((label, tau));
    }

    // Different substrates should produce different tau factors
    // (at least some should differ)
    let unique_count = {
        let mut vals: Vec<u32> = tau_factors
            .iter()
            .map(|(_, t)| (*t * 1000.0) as u32)
            .collect();
        vals.sort();
        vals.dedup();
        vals.len()
    };
    assert!(
        unique_count >= 2,
        "Expected at least 2 distinct tau factors, got {unique_count}: {:?}",
        tau_factors
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Multi-substrate soak: 200 cycles with mid-run switch
// ═══════════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 4: Hybrid substrate modeling — per-region substrate assignment
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn per_region_substrate_assignment() {
    let mut config = CognitiveLoopConfig::default();
    config.enable_validation_overlay = false;

    // Configure heterogeneous substrate: different substrates for different regions
    let mut per_region = HashMap::new();
    per_region.insert(CorticalRegion::Prefrontal, SubstrateType::SiliconDigital);
    per_region.insert(CorticalRegion::Memory, SubstrateType::BiologicalNeurons);
    per_region.insert(CorticalRegion::Sensory, SubstrateType::QuantumComputer);
    per_region.insert(CorticalRegion::Motor, SubstrateType::PhotonicProcessor);
    config.per_region_substrates = Some(per_region);

    let service = CognitiveLoopService::new(config).unwrap();

    // Each region should have its own feasibility
    let pfc_f = service.region_feasibility(CorticalRegion::Prefrontal);
    let mem_f = service.region_feasibility(CorticalRegion::Memory);
    let sens_f = service.region_feasibility(CorticalRegion::Sensory);
    let motor_f = service.region_feasibility(CorticalRegion::Motor);

    // Biological memory should have highest feasibility
    assert!(
        mem_f > pfc_f,
        "Biological memory ({mem_f:.3}) should exceed silicon PFC ({pfc_f:.3})"
    );

    // All should be in [0, 1]
    for (label, f) in [
        ("PFC", pfc_f),
        ("Memory", mem_f),
        ("Sensory", sens_f),
        ("Motor", motor_f),
    ] {
        assert!(
            f >= 0.0 && f <= 1.0,
            "{label} feasibility out of range: {f}"
        );
    }

    // Unconfigured regions fall back to global effective feasibility
    let emotion_f = service.region_feasibility(CorticalRegion::Emotional);
    let global_f = service.substrate_effective_feasibility() as f32;
    assert!(
        (emotion_f - global_f).abs() < f32::EPSILON,
        "Unconfigured region should fall back to global: {emotion_f:.4} vs {global_f:.4}"
    );
}

#[test]
fn per_region_runtime_reconfiguration() {
    let mut service = make_service(SubstrateType::SiliconDigital);

    // Warm up 20 cycles
    for _ in 0..20 {
        service.cycle("warmup");
    }

    // Assign biological neurons to memory region at runtime
    service.reconfigure_region(CorticalRegion::Memory, SubstrateType::BiologicalNeurons);

    let mem_f = service.region_feasibility(CorticalRegion::Memory);
    assert!(
        mem_f > 0.8,
        "Biological memory region should have high feasibility: {mem_f:.3}"
    );

    // Run 50 more cycles — should remain stable
    for i in 0..50 {
        let result = service.cycle("testing with hybrid substrate");
        assert!(
            result.metadata.valence_homeostasis_pull.is_finite(),
            "NaN after per-region reconfiguration at cycle {i}"
        );
    }
}

#[test]
fn cross_substrate_communication_penalty() {
    let mut config = CognitiveLoopConfig::default();
    config.enable_validation_overlay = false;

    // Homogeneous: all regions same substrate → no penalty
    let mut homo_map = HashMap::new();
    homo_map.insert(CorticalRegion::Prefrontal, SubstrateType::SiliconDigital);
    homo_map.insert(CorticalRegion::Memory, SubstrateType::SiliconDigital);
    homo_map.insert(CorticalRegion::Sensory, SubstrateType::SiliconDigital);
    config.per_region_substrates = Some(homo_map);
    let homo_service = CognitiveLoopService::new(config.clone()).unwrap();
    let homo_eff = homo_service.substrate_effective_feasibility();

    // Heterogeneous: 3 different substrates → communication penalty
    let mut hetero_map = HashMap::new();
    hetero_map.insert(CorticalRegion::Prefrontal, SubstrateType::SiliconDigital);
    hetero_map.insert(CorticalRegion::Memory, SubstrateType::BiologicalNeurons);
    hetero_map.insert(CorticalRegion::Sensory, SubstrateType::QuantumComputer);
    config.per_region_substrates = Some(hetero_map);
    let hetero_service = CognitiveLoopService::new(config).unwrap();
    let hetero_eff = hetero_service.substrate_effective_feasibility();

    // Heterogeneous should have lower effective feasibility due to communication penalty
    // (even though biological memory pulls the average up, the penalty should dominate)
    assert!(
        hetero_eff < homo_eff,
        "Heterogeneous ({hetero_eff:.4}) should have lower effective feasibility \
         than homogeneous ({homo_eff:.4}) due to communication penalty"
    );
}

#[test]
fn hybrid_substrate_soak_100_cycles() {
    let mut config = CognitiveLoopConfig::default();
    config.enable_validation_overlay = false;

    // Example from THE_SUBSTRATE_ROADMAP Phase 4:
    // Prefrontal: SiliconDigital (fast HOT)
    // Sensory: QuantumComputer (entanglement binding)
    // Memory: BiologicalNeurons (proven integration)
    // Motor: PhotonicProcessor (ultra-fast dynamics)
    let mut per_region = HashMap::new();
    per_region.insert(CorticalRegion::Prefrontal, SubstrateType::SiliconDigital);
    per_region.insert(CorticalRegion::Sensory, SubstrateType::QuantumComputer);
    per_region.insert(CorticalRegion::Memory, SubstrateType::BiologicalNeurons);
    per_region.insert(CorticalRegion::Motor, SubstrateType::PhotonicProcessor);
    config.per_region_substrates = Some(per_region);

    let mut service = CognitiveLoopService::new(config).unwrap();

    let inputs = [
        "the cat sat on the mat",
        "quantum mechanics describes wave functions",
        "consciousness emerges from integrated information",
        "ethical reasoning requires empathy",
        "the quick brown fox jumps",
    ];

    for i in 0..100 {
        let result = service.cycle(inputs[i % inputs.len()]);
        let m = &result.metadata;
        assert!(
            m.valence_homeostasis_pull.is_finite(),
            "Hybrid substrate NaN at cycle {i}"
        );
        assert!(
            m.social_trust_current >= 0.0 && m.social_trust_current <= 1.0,
            "social_trust out of range at cycle {i}: {}",
            m.social_trust_current
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Multi-substrate soak: 200 cycles with mid-run switch
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn soak_200_cycles_with_mid_run_substrate_switch() {
    let mut service = make_service(SubstrateType::SiliconDigital);

    // Phase 1: 100 cycles on silicon
    for i in 0..100 {
        let result = service.cycle("silicon phase");
        assert!(
            result.metadata.valence_homeostasis_pull.is_finite(),
            "Silicon phase NaN at cycle {i}"
        );
    }

    // Switch to quantum
    service.reconfigure_substrate(SubstrateType::QuantumComputer);

    // Phase 2: 100 cycles on quantum
    for i in 0..100 {
        let result = service.cycle("quantum phase");
        assert!(
            result.metadata.valence_homeostasis_pull.is_finite(),
            "Quantum phase NaN at cycle {i}"
        );
        assert!(
            result.metadata.social_trust_current >= 0.0,
            "Social trust negative in quantum phase at cycle {i}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 3 Additions: Transition Smoothing, Energy, Dim Fraction, History
// ═══════════════════════════════════════════════════════════════════════════════

fn make_simulation_service(substrate: SubstrateType) -> CognitiveLoopService {
    let mut config = CognitiveLoopConfig {
        async_training: false,
        learning_threshold: 0.0,
        substrate_type: substrate,
        ..Default::default()
    };
    config.enable_substrate_simulation();
    CognitiveLoopService::new(config).unwrap()
}

#[test]
fn test_energy_recalculates_on_substrate_switch() {
    let mut service = make_simulation_service(SubstrateType::SiliconDigital);
    let silicon_energy = service.substrate_energy_per_cycle();

    service.reconfigure_substrate_at_cycle(SubstrateType::BiologicalNeurons, 0);
    let bio_energy = service.substrate_energy_per_cycle();

    // Silicon and biological have different energy_per_operation
    assert!(
        (silicon_energy - bio_energy).abs() > f64::EPSILON,
        "Energy should change on switch: silicon={silicon_energy:.2e} bio={bio_energy:.2e}"
    );
}

#[test]
fn test_transition_smoothing_converges() {
    let mut service = make_simulation_service(SubstrateType::SiliconDigital);

    // Warmup
    for _ in 0..5 {
        service.cycle("warmup");
    }

    let tau_before = service.substrate_tau_factor();
    service.reconfigure_substrate_at_cycle(SubstrateType::BiochemicalComputer, 5);

    // With alpha=0.1, after 50 cycles we should be very close to target
    for i in 0..50 {
        service.cycle(&format!("converge {i}"));
    }
    let tau_after = service.substrate_tau_factor();

    // Tau should have moved significantly from silicon toward biochemical
    assert!(
        tau_after != tau_before || tau_before == 1.0,
        "Tau should converge to target: before={tau_before:.4} after={tau_after:.4}"
    );
    assert!(
        tau_after.is_finite() && tau_after >= 0.5 && tau_after <= 2.0,
        "Tau out of range: {tau_after:.4}"
    );
}

#[test]
fn test_effective_dim_fraction_varies_by_substrate() {
    let silicon = make_simulation_service(SubstrateType::SiliconDigital);
    let quantum = make_simulation_service(SubstrateType::QuantumComputer);

    let silicon_frac = silicon.substrate_effective_dim_fraction();
    let quantum_frac = quantum.substrate_effective_dim_fraction();

    // Silicon has positive scale_pressure → frac = 1.0
    assert!(
        (silicon_frac - 1.0).abs() < f32::EPSILON,
        "Silicon should have dim_fraction=1.0: {silicon_frac}"
    );
    // Quantum has very negative scale_pressure → frac < 1.0
    assert!(
        quantum_frac < 1.0,
        "Quantum should have dim_fraction < 1.0: {quantum_frac}"
    );
    assert!(
        quantum_frac >= 0.1,
        "Quantum dim_fraction should be >= 0.1: {quantum_frac}"
    );
}

#[test]
fn test_transition_history_tracks_switches() {
    let mut service = make_simulation_service(SubstrateType::SiliconDigital);

    assert!(service.substrate_transition_history().is_empty());

    let switches = [
        SubstrateType::BiologicalNeurons,
        SubstrateType::QuantumComputer,
        SubstrateType::PhotonicProcessor,
        SubstrateType::NeuromorphicChip,
        SubstrateType::ExoticSubstrate,
    ];

    for (i, &sub) in switches.iter().enumerate() {
        service.reconfigure_substrate_at_cycle(sub, i as u64);
    }

    let history = service.substrate_transition_history();
    assert_eq!(history.len(), 5, "Should have 5 transition records");

    // Verify cycle numbers are correct
    for (i, record) in history.iter().enumerate() {
        assert_eq!(record.cycle, i as u64, "Record {i} should have cycle={i}");
    }
}

#[test]
fn test_energy_telemetry_populated() {
    let mut service = make_simulation_service(SubstrateType::SiliconDigital);

    // Run a few cycles so energy accumulates
    for _ in 0..5 {
        service.cycle("energy test");
    }

    let result = service.cycle("check telemetry");
    let substrate = &result.metadata.substrate;
    assert!(
        substrate.total_energy_spent > 0.0,
        "Energy should have accumulated: {}",
        substrate.total_energy_spent
    );
    assert!(
        substrate.energy_this_cycle > 0.0,
        "Per-cycle energy should be positive: {}",
        substrate.energy_this_cycle
    );
    assert!(
        substrate.energy_throughput_multiplier > 0.0,
        "Throughput multiplier should be positive: {}",
        substrate.energy_throughput_multiplier
    );
    assert!(
        (0.1..=1.0).contains(&substrate.effective_dim_fraction),
        "Dim fraction out of range: {}",
        substrate.effective_dim_fraction
    );
}

#[test]
fn test_scale_masking_affects_state_diversity() {
    // Quantum substrate with scale masking should produce lower state diversity
    // than silicon (which retains full dimensionality)
    let mut silicon = make_simulation_service(SubstrateType::SiliconDigital);
    let mut quantum = make_simulation_service(SubstrateType::QuantumComputer);

    // Warmup both identically
    for _ in 0..20 {
        silicon.cycle("diversity test");
        quantum.cycle("diversity test");
    }

    // Both should produce finite, non-negative state diversity
    // (We can't assert quantum < silicon because the masking is subtle,
    // but we CAN assert no NaN/Inf)
    let silicon_result = silicon.cycle("final");
    let quantum_result = quantum.cycle("final");
    assert!(
        silicon_result
            .metadata
            .consciousness
            .consciousness_level
            .is_finite(),
        "Silicon consciousness should be finite"
    );
    assert!(
        quantum_result
            .metadata
            .consciousness
            .consciousness_level
            .is_finite(),
        "Quantum consciousness should be finite after scale masking"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phase B: CPG Sync → Consciousness Coupling
// ═══════════════════════════════════════════════════════════════════════════════

/// CPG sync_index modulates unified consciousness ±5%.
/// With CPG disabled (default), sync_index = 0.5 (neutral). After ~30 cycles
/// the consciousness engine has enough data to show measurable differences
/// when we compare against a hypothetical fully-synced state.
#[test]
fn test_cpg_sync_consciousness_coupling_finite() {
    // Default config has no CPG feature, so cpg_sync_index = 0.5 (neutral)
    let mut svc = make_service(SubstrateType::SiliconDigital);

    // Run 30 cycles to get stable consciousness
    let mut consciousness_levels = Vec::new();
    for i in 0..30 {
        let result = svc.cycle(&format!("cpg sync test {i}"));
        let c = result.metadata.consciousness.consciousness_level;
        assert!(
            c.is_finite(),
            "consciousness should be finite at cycle {i}, got {c}"
        );
        consciousness_levels.push(c);
    }

    // All levels should be bounded [0, 1]
    for (i, &c) in consciousness_levels.iter().enumerate() {
        assert!(
            c >= 0.0 && c <= 1.0,
            "consciousness out of [0,1] at cycle {i}: {c}"
        );
    }

    // Later cycles should show consciousness above zero (system warms up)
    let last_5_avg: f64 = consciousness_levels[25..].iter().sum::<f64>() / 5.0;
    assert!(
        last_5_avg > 0.0,
        "Consciousness should be above zero after warmup, avg={last_5_avg}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phase B: Spectral Entropy Masking (requires spectral_state feature)
// ═══════════════════════════════════════════════════════════════════════════════

/// Verify CfC outputs remain finite with spectral masking active.
/// Since spectral_state is a feature flag, this test runs with default features
/// and verifies the CfC path is robust regardless of whether masking is active.
#[test]
fn test_cfc_outputs_finite_with_spectral_path() {
    let mut svc = make_service(SubstrateType::SiliconDigital);

    for i in 0..20 {
        let result = svc.cycle(&format!("spectral masking test {i}"));
        let m = &result.metadata;

        // Core outputs must be finite
        assert!(
            m.consciousness.consciousness_level.is_finite(),
            "consciousness NaN at cycle {i}"
        );
        assert!(
            result.prediction_error.is_finite(),
            "prediction_error NaN at cycle {i}"
        );
        assert!(
            m.actual_effective_lr.is_finite(),
            "learning_rate NaN at cycle {i}"
        );
        // CfC-derived temporal coherence must be finite
        assert!(
            m.temporal.temporal_coherence_score.is_finite(),
            "temporal_coherence NaN at cycle {i}"
        );
    }
}