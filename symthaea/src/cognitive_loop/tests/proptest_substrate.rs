// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Property tests for substrate independence.
//!
//! Validates that substrate switching, feasibility computation, and
//! consciousness scaling maintain invariants across random configurations.

use proptest::prelude::*;

use symthaea_core::hdc::substrate_independence::SubstrateType;

use crate::cognitive_loop::config::CognitiveLoopConfig;
use crate::cognitive_loop::substrate_manager::SubstrateManager;

/// Strategy: random SubstrateType from all 9 variants.
fn substrate_strategy() -> impl Strategy<Value = SubstrateType> {
    prop_oneof![
        Just(SubstrateType::BiologicalNeurons),
        Just(SubstrateType::SiliconDigital),
        Just(SubstrateType::QuantumComputer),
        Just(SubstrateType::PhotonicProcessor),
        Just(SubstrateType::NeuromorphicChip),
        Just(SubstrateType::BiochemicalComputer),
        Just(SubstrateType::HybridSystem),
        Just(SubstrateType::ExoticSubstrate),
        Just(SubstrateType::SpacecraftComputer),
    ]
}

proptest! {
    // Property 1: Feasibility is always finite and bounded [0, 1] for any substrate
    #[test]
    fn prop_feasibility_bounded(substrate in substrate_strategy()) {
        let mut config = CognitiveLoopConfig::default();
        config.substrate_type = substrate;
        let mgr = SubstrateManager::new(&config);
        prop_assert!(mgr.feasibility.is_finite(), "feasibility not finite for {:?}", substrate);
        prop_assert!(mgr.feasibility >= 0.0 && mgr.feasibility <= 1.0,
            "feasibility out of [0,1]: {} for {:?}", mgr.feasibility, substrate);
    }

    // Property 2: Effective feasibility <= raw feasibility (validation overlay only reduces)
    #[test]
    fn prop_effective_leq_raw(substrate in substrate_strategy()) {
        let mut config = CognitiveLoopConfig::default();
        config.substrate_type = substrate;
        config.enable_validation_overlay = true;
        let mgr = SubstrateManager::new(&config);
        prop_assert!(mgr.effective_feasibility <= mgr.feasibility + 1e-10,
            "effective ({}) > raw ({}) for {:?}",
            mgr.effective_feasibility, mgr.feasibility, substrate);
    }

    // Property 3: Tau factor is always in [0.5, 2.0]
    #[test]
    fn prop_tau_factor_bounded(substrate in substrate_strategy()) {
        let mut config = CognitiveLoopConfig::default();
        config.substrate_type = substrate;
        config.enable_substrate_speed_modulation = true;
        let mgr = SubstrateManager::new(&config);
        prop_assert!(mgr.tau_factor >= 0.5 && mgr.tau_factor <= 2.0,
            "tau_factor out of [0.5, 2.0]: {} for {:?}", mgr.tau_factor, substrate);
    }

    // Property 4: Substrate switching preserves finiteness
    #[test]
    fn prop_switch_preserves_finite(
        from in substrate_strategy(),
        to in substrate_strategy(),
    ) {
        let mut config = CognitiveLoopConfig::default();
        config.substrate_type = from;
        config.enable_validation_overlay = true;
        let mut mgr = SubstrateManager::new(&config);
        mgr.reconfigure_substrate(&mut config, to);
        prop_assert!(mgr.feasibility.is_finite());
        prop_assert!(mgr.effective_feasibility.is_finite());
        prop_assert!(mgr.tau_factor.is_finite());
        prop_assert!(mgr.scale_pressure.is_finite());
    }

    // Property 5: Skepticism floor bounds minimum effective feasibility
    #[test]
    fn prop_skepticism_floor_bounds(
        substrate in substrate_strategy(),
        floor in 0.0f64..=1.0,
    ) {
        let mut config = CognitiveLoopConfig::default();
        config.substrate_type = substrate;
        config.enable_validation_overlay = true;
        config.validation_skepticism_floor = floor;
        let mgr = SubstrateManager::new(&config);
        // effective >= feasibility * floor (within floating point tolerance)
        let min_expected = mgr.feasibility * floor - 1e-10;
        prop_assert!(mgr.effective_feasibility >= min_expected,
            "effective ({}) < floor bound ({}) for {:?} with floor={}",
            mgr.effective_feasibility, min_expected, substrate, floor);
    }

    // Property 6: Sequential switches don't accumulate drift
    #[test]
    fn prop_sequential_switches_no_drift(
        substrates in prop::collection::vec(substrate_strategy(), 5..20),
    ) {
        let mut config = CognitiveLoopConfig::default();
        config.substrate_type = substrates[0];
        config.enable_validation_overlay = true;
        let mut mgr = SubstrateManager::new(&config);
        for &s in &substrates[1..] {
            mgr.reconfigure_substrate(&mut config, s);
            prop_assert!(mgr.feasibility.is_finite());
            prop_assert!(mgr.effective_feasibility.is_finite());
            prop_assert!(mgr.tau_factor.is_finite());
            prop_assert!(mgr.feasibility >= 0.0 && mgr.feasibility <= 1.0);
            prop_assert!(mgr.effective_feasibility >= 0.0 && mgr.effective_feasibility <= 1.0);
        }
        // Final state should match direct construction with last substrate
        let last = *substrates.last().unwrap();
        config.substrate_type = last;
        let direct = SubstrateManager::new(&config);
        let diff = (mgr.feasibility - direct.feasibility).abs();
        prop_assert!(diff < 1e-10,
            "drift after {} switches: switched={} vs direct={} (diff={})",
            substrates.len(), mgr.feasibility, direct.feasibility, diff);
    }
}
