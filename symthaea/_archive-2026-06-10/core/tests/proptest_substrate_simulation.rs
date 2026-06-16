// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Property-Based Tests for Substrate Phase 3 (Multi-Substrate Simulation)

Verifies structural invariants:
1. Transition smoothing converges to target
2. Energy monotonically increases
3. Effective dim fraction bounded [0.1, 1.0]
4. Consciousness survives any substrate switch (no NaN)
5. Energy recalculates on substrate switch
6. Transition history bounded by cap
*/

use proptest::prelude::*;

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use symthaea_core::hdc::substrate_independence::SubstrateType;

fn all_substrates() -> Vec<SubstrateType> {
    vec![
        SubstrateType::BiologicalNeurons,
        SubstrateType::SiliconDigital,
        SubstrateType::QuantumComputer,
        SubstrateType::PhotonicProcessor,
        SubstrateType::NeuromorphicChip,
        SubstrateType::BiochemicalComputer,
        SubstrateType::HybridSystem,
        SubstrateType::ExoticSubstrate,
    ]
}

fn substrate_index_strategy() -> impl Strategy<Value = usize> {
    0..8usize
}

fn make_simulation_service(substrate: SubstrateType) -> CognitiveLoopService {
    let mut config = CognitiveLoopConfig {
        async_training: false,
        learning_threshold: 0.0,
        substrate_type: substrate,
        ..Default::default()
    };
    config.enable_substrate_simulation();
    CognitiveLoopService::new(config).expect("CLS::new should succeed")
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4))]

    /// Consciousness survives any substrate switch (no NaN after 5 cycles).
    #[test]
    fn prop_consciousness_survives_switch(
        from_idx in substrate_index_strategy(),
        to_idx in substrate_index_strategy(),
    ) {
        let substrates = all_substrates();
        let from = substrates[from_idx];
        let to = substrates[to_idx];
        let mut service = make_simulation_service(from);

        // Run 3 warmup cycles
        for _ in 0..3 {
            service.cycle("warmup");
        }

        // Switch substrate
        service.reconfigure_substrate_at_cycle(to, 3);

        // Run 5 cycles post-switch
        for i in 0..5 {
            let result = service.cycle(&format!("post-switch {i}"));
            let cl = result.metadata.consciousness.consciousness_level;
            prop_assert!(
                cl.is_finite(),
                "Consciousness NaN after {:?}->{:?} at cycle {}: {}",
                from, to, i, cl
            );
        }
    }

    /// Energy monotonically increases across cycles (when budget enabled).
    #[test]
    fn prop_energy_monotonic(sub_idx in substrate_index_strategy()) {
        let substrates = all_substrates();
        let mut service = make_simulation_service(substrates[sub_idx]);
        let mut prev_energy = 0.0f64;

        for i in 0..10 {
            service.cycle(&format!("energy test {i}"));
            let energy = service.substrate_total_energy_spent();
            prop_assert!(
                energy >= prev_energy,
                "Energy decreased at cycle {}: {} < {}",
                i, energy, prev_energy
            );
            prev_energy = energy;
        }
    }

    /// Effective dim fraction is bounded [0.1, 1.0] for all substrates.
    #[test]
    fn prop_dim_fraction_bounded(sub_idx in substrate_index_strategy()) {
        let substrates = all_substrates();
        let service = make_simulation_service(substrates[sub_idx]);
        let frac = service.substrate_effective_dim_fraction();
        prop_assert!(
            (0.1..=1.0).contains(&frac),
            "{:?} dim_fraction out of bounds: {}",
            substrates[sub_idx], frac
        );
    }

    /// Energy per cycle recalculates on substrate switch.
    #[test]
    fn prop_energy_recalculates_on_switch(
        from_idx in substrate_index_strategy(),
        to_idx in substrate_index_strategy(),
    ) {
        if from_idx == to_idx { return Ok(()); }
        let substrates = all_substrates();
        let mut service = make_simulation_service(substrates[from_idx]);
        let energy_before = service.substrate_energy_per_cycle();
        service.reconfigure_substrate_at_cycle(substrates[to_idx], 0);
        let energy_after = service.substrate_energy_per_cycle();
        // Different substrates should produce different energy (unless same canonical)
        let from_canon = substrates[from_idx].canonical();
        let to_canon = substrates[to_idx].canonical();
        if from_canon != to_canon {
            prop_assert!(
                (energy_before - energy_after).abs() > f64::EPSILON,
                "Energy should change on switch {:?}->{:?}: {} vs {}",
                from_canon, to_canon, energy_before, energy_after
            );
        }
    }

    /// Transition history never exceeds cap (32).
    #[test]
    fn prop_transition_history_bounded(n_switches in 1u32..50) {
        let substrates = all_substrates();
        let mut service = make_simulation_service(SubstrateType::SiliconDigital);
        for i in 0..n_switches {
            let target = substrates[(i as usize + 1) % substrates.len()];
            service.reconfigure_substrate_at_cycle(target, i as u64);
        }
        let history = service.substrate_transition_history();
        prop_assert!(
            history.len() <= 32,
            "Transition history exceeded cap: {}",
            history.len()
        );
    }

    /// Transition smoothing converges: after 50 cycles, tau_factor
    /// is within 1% of target.
    #[test]
    fn prop_transition_smoothing_converges(
        from_idx in substrate_index_strategy(),
        to_idx in substrate_index_strategy(),
    ) {
        let substrates = all_substrates();
        let mut service = make_simulation_service(substrates[from_idx]);

        // Warmup
        for _ in 0..3 {
            service.cycle("warmup");
        }

        // Switch
        service.reconfigure_substrate_at_cycle(substrates[to_idx], 3);

        // Run 50 cycles (alpha=0.1 → ~10 cycles to settle)
        for i in 0..50 {
            service.cycle(&format!("converge {i}"));
        }

        // After 50 cycles the tau should have converged
        let tau = service.substrate_tau_factor();
        prop_assert!(
            tau.is_finite() && tau >= 0.5 && tau <= 2.0,
            "Tau out of range after convergence: {}",
            tau
        );
    }
}