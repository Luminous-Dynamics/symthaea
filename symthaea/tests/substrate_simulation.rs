//! Phase 3 Multi-Substrate Simulation Tests
//!
//! Tests that verify consciousness dynamics across different virtual substrates
//! and substrate transitions. Answers key questions from THE_SUBSTRATE_ROADMAP:
//! - Does consciousness survive substrate transfer? (Multiple Realizability)
//! - Which substrate maximizes consciousness metrics?
//! - Is there a minimum feasibility below which consciousness collapses?
//! - How does substrate speed affect temporal coherence?

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use symthaea_core::hdc::substrate_independence::SubstrateType;

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
        let mut vals: Vec<u32> = tau_factors.iter().map(|(_, t)| (*t * 1000.0) as u32).collect();
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
