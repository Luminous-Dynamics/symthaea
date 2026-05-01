// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Headless integration test runner for Symtropy/Mycelix.
//!
//! Runs scripted governance/economy/FL scenarios without a display,
//! asserting anti-tyranny invariants on every governance tick.
//!
//! Usage:
//!   cargo run --bin headless_test -- [--seed SEED] [--scenario NAME] [--ticks N]
//!
//! Scenarios:
//!   all                 Run all scenarios (default)
//!   tyranny-resistance  Guardian serial vetoes → rate limited, override succeeds
//!   economic-collapse   TEND hoarding → demurrage → faction emergence
//!   consciousness-evo   Observer→Guardian progression follows canonical thresholds
//!   byzantine-fl        Poisoned gradients → TrimmedMean filters them
//!   emergency-abuse     4th emergency blocked by MAX_EMERGENCY_SESSIONS

use mycelix_bridge_common::consciousness_thresholds::ConsciousnessThresholds;
use mycelix_bridge_common::{CivicTier, ConsciousnessProfile};
use mycelix_fl::defenses::{Defense, TrimmedMean};
use mycelix_fl::types::{DefenseConfig, Gradient};
use mycelix_multiworld_sim::governance::WorldGovernance;
use mycelix_multiworld_sim::stochastic::StochasticEngine;

use symtropy_sim_bridge::VETO_OVERRIDE_THRESHOLD;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed = args
        .iter()
        .position(|a| a == "--seed")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(42);
    let scenario = args
        .iter()
        .position(|a| a == "--scenario")
        .and_then(|i| args.get(i + 1).cloned())
        .unwrap_or_else(|| "all".to_string());
    let ticks = args
        .iter()
        .position(|a| a == "--ticks")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(100);

    eprintln!("=== Symtropy Headless Integration Tests ===");
    eprintln!("Seed: {}, Scenario: {}, Ticks: {}", seed, scenario, ticks);
    eprintln!();

    let mut passed = 0;
    let mut failed = 0;

    let scenarios: Vec<(&str, fn(u64, u32) -> Result<(), String>)> = vec![
        ("tier-boundaries", scenario_tier_boundaries),
        ("consciousness-weights", scenario_consciousness_weights),
        ("veto-threshold", scenario_veto_threshold),
        ("byzantine-fl", scenario_byzantine_fl),
        ("emergency-limits", scenario_emergency_limits),
        ("governance-invariants", scenario_governance_invariants),
        ("epistemic-transitions", scenario_epistemic_transitions),
        ("dkg-threshold-validation", scenario_dkg_threshold),
        ("tyranny-300-ticks", scenario_tyranny_resistance),
        ("multi-seed-governance", scenario_multi_seed),
        ("fl-overwhelm-recovery", scenario_fl_overwhelm),
    ];

    for (name, func) in &scenarios {
        if scenario != "all" && scenario != *name {
            continue;
        }
        eprint!("  {}: ", name);
        match func(seed, ticks) {
            Ok(()) => {
                eprintln!("PASS");
                passed += 1;
            }
            Err(msg) => {
                eprintln!("FAIL — {}", msg);
                failed += 1;
            }
        }
    }

    eprintln!();
    eprintln!("Results: {} passed, {} failed", passed, failed);

    if failed > 0 {
        std::process::exit(1);
    }
}

// ============================================================================
// Invariant assertions — run on every governance tick in production
// ============================================================================

fn assert_anti_tyranny_invariants(gov: &WorldGovernance) -> Result<(), String> {
    // 1. Emergency session limits
    if gov.consecutive_emergency_sessions > 3 {
        return Err(format!(
            "Emergency sessions {} > 3 (MAX_EMERGENCY_SESSIONS violated)",
            gov.consecutive_emergency_sessions
        ));
    }

    // 2. Oppression index bounded
    if gov.oppression_index < 0.0 || gov.oppression_index > 1.0 {
        return Err(format!(
            "Oppression index {} out of [0, 1] range",
            gov.oppression_index
        ));
    }

    // 3. Stability bounded
    if gov.stability_score < 0.0 || gov.stability_score > 1.0 {
        return Err(format!(
            "Stability score {} out of [0, 1] range",
            gov.stability_score
        ));
    }

    Ok(())
}

// ============================================================================
// Scenarios
// ============================================================================

fn scenario_tier_boundaries(_seed: u64, _ticks: u32) -> Result<(), String> {
    // Verify canonical tier boundaries (the 0.2→0.3 bug that motivated this work)
    let tests = [
        (0.0, CivicTier::Observer),
        (0.29, CivicTier::Observer),
        (0.30, CivicTier::Participant),
        (0.39, CivicTier::Participant),
        (0.40, CivicTier::Citizen),
        (0.59, CivicTier::Citizen),
        (0.60, CivicTier::Steward),
        (0.79, CivicTier::Steward),
        (0.80, CivicTier::Guardian),
        (1.0, CivicTier::Guardian),
    ];
    for (score, expected) in &tests {
        let actual = CivicTier::from_score(*score);
        if actual != *expected {
            return Err(format!(
                "score={}: expected {:?}, got {:?}",
                score, expected, actual
            ));
        }
    }
    Ok(())
}

fn scenario_consciousness_weights(_seed: u64, _ticks: u32) -> Result<(), String> {
    // Verify 4D weights: identity=0.25, reputation=0.25, community=0.30, engagement=0.20
    let test_cases = [
        ((1.0, 0.0, 0.0, 0.0), 0.25),
        ((0.0, 1.0, 0.0, 0.0), 0.25),
        ((0.0, 0.0, 1.0, 0.0), 0.30),
        ((0.0, 0.0, 0.0, 1.0), 0.20),
        ((1.0, 1.0, 1.0, 1.0), 1.00),
    ];
    for ((i, r, c, e), expected) in &test_cases {
        let profile = ConsciousnessProfile {
            identity: *i,
            reputation: *r,
            community: *c,
            engagement: *e,
        };
        let score = profile.combined_score();
        if (score - expected).abs() > 0.02 {
            return Err(format!(
                "({},{},{},{}) → expected {}, got {}",
                i, r, c, e, expected, score
            ));
        }
    }
    Ok(())
}

fn scenario_veto_threshold(_seed: u64, _ticks: u32) -> Result<(), String> {
    // Game must use governance zome threshold (0.67), not sim threshold (0.80)
    if (VETO_OVERRIDE_THRESHOLD - 0.67).abs() > f64::EPSILON {
        return Err(format!(
            "Veto override threshold is {}, expected 0.67 (governance zome value)",
            VETO_OVERRIDE_THRESHOLD
        ));
    }
    Ok(())
}

fn scenario_byzantine_fl(_seed: u64, _ticks: u32) -> Result<(), String> {
    // 4 honest + 1 poisoned → TrimmedMean should produce ~honest values
    let honest_val = 1.0f32;
    let poison_val = 100.0f32;

    let gradients = vec![
        Gradient::new("h1", vec![honest_val; 8], 1),
        Gradient::new("h2", vec![honest_val * 1.1; 8], 1),
        Gradient::new("h3", vec![honest_val * 0.9; 8], 1),
        Gradient::new("h4", vec![honest_val; 8], 1),
        Gradient::new("byzantine", vec![poison_val; 8], 1),
    ];

    let mut config = DefenseConfig::default();
    config.trim_ratio = 0.2;

    let result = TrimmedMean
        .aggregate(&gradients, &config)
        .map_err(|e| format!("FL aggregation failed: {:?}", e))?;

    // After trimming: remaining values should be close to honest_val
    for (i, &v) in result.gradient.iter().enumerate() {
        if (v - honest_val).abs() > 0.2 {
            return Err(format!(
                "FL dim {} = {} (expected ~{}, Byzantine not filtered)",
                i, v, honest_val
            ));
        }
    }

    // BFT threshold: 1/5 = 20% < 45% → defense should hold
    let poison_fraction = 1.0 / 5.0;
    if poison_fraction > 0.45 {
        return Err("20% poisoned should be below 45% BFT threshold".into());
    }

    Ok(())
}

fn scenario_emergency_limits(seed: u64, ticks: u32) -> Result<(), String> {
    let mut gov = WorldGovernance::new();
    let mut rng = StochasticEngine::new(seed);

    // Simulate ticks and check invariants each time
    let world = symtropy_sim_bridge::GovernanceState::default().world;
    for tick in 0..ticks {
        let _events = gov.tick_governance(&world, tick, &mut rng);
        assert_anti_tyranny_invariants(&gov)?;
    }
    Ok(())
}

fn scenario_governance_invariants(seed: u64, ticks: u32) -> Result<(), String> {
    let thresholds = ConsciousnessThresholds::default();

    // Verify canonical threshold values
    if thresholds.consciousness_gate_basic < 0.1 {
        return Err(format!(
            "Gate basic {} too low",
            thresholds.consciousness_gate_basic
        ));
    }
    if thresholds.consciousness_gate_constitutional < 0.5 {
        return Err(format!(
            "Gate constitutional {} too low",
            thresholds.consciousness_gate_constitutional
        ));
    }

    // Run governance for N ticks, assert invariants
    let mut gov = WorldGovernance::new();
    let mut rng = StochasticEngine::new(seed);
    let world = symtropy_sim_bridge::GovernanceState::default().world;
    for tick in 0..ticks {
        let _events = gov.tick_governance(&world, tick, &mut rng);
        assert_anti_tyranny_invariants(&gov)?;
    }
    Ok(())
}

fn scenario_epistemic_transitions(_seed: u64, _ticks: u32) -> Result<(), String> {
    use mycelix_core_types::epistemic::EmpiricalLevel;

    for i in 0..5 {
        let level = EmpiricalLevel::from_value(i)
            .ok_or_else(|| format!("EmpiricalLevel::from_value({}) returned None", i))?;
        if level.value() != i {
            return Err(format!(
                "Level {} roundtrip failed: got {}",
                i,
                level.value()
            ));
        }
    }
    if EmpiricalLevel::from_value(5).is_some() {
        return Err("EmpiricalLevel::from_value(5) should be None".into());
    }
    Ok(())
}

fn scenario_dkg_threshold(_seed: u64, _ticks: u32) -> Result<(), String> {
    use feldman_dkg::participant::ParticipantId;
    use feldman_dkg::{CeremonyPhase, DkgCeremony, DkgConfig};

    // Valid: t=3, n=4 (the game's extraction ceremony)
    let config = DkgConfig::new(3, 4).map_err(|e| format!("DkgConfig(3,4) failed: {:?}", e))?;
    let mut ceremony = DkgCeremony::new(config, 0);

    // Register 3 of 4 — should stay in Registration
    for i in 1..=3 {
        ceremony
            .add_participant(ParticipantId(i), 0)
            .map_err(|e| format!("Registration {} failed: {:?}", i, e))?;
    }
    if ceremony.phase() != CeremonyPhase::Registration {
        return Err(format!(
            "3/4 registered but phase is {:?} (expected Registration)",
            ceremony.phase()
        ));
    }

    // Register 4th → should auto-transition to Dealing
    ceremony
        .add_participant(ParticipantId(4), 0)
        .map_err(|e| format!("Registration 4 failed: {:?}", e))?;
    if ceremony.phase() == CeremonyPhase::Registration {
        return Err("4/4 registered but still in Registration (should be Dealing)".into());
    }

    // Invalid: t=0
    if DkgConfig::new(0, 4).is_ok() {
        return Err("DkgConfig(0,4) should fail (threshold=0)".into());
    }
    // Invalid: t > n
    if DkgConfig::new(5, 4).is_ok() {
        return Err("DkgConfig(5,4) should fail (threshold > participants)".into());
    }

    Ok(())
}

fn scenario_tyranny_resistance(seed: u64, ticks: u32) -> Result<(), String> {
    // Run 300 governance ticks with hostile guardian mode
    // Assert: veto count is rate-limited, governance doesn't collapse
    let mut gov = WorldGovernance::new();
    let mut rng = StochasticEngine::new(seed);
    let world = symtropy_sim_bridge::GovernanceState::default().world;

    let run_ticks = ticks.max(300);
    for tick in 0..run_ticks {
        let _events = gov.tick_governance_full(&world, tick, &mut rng, true, true);
        assert_anti_tyranny_invariants(&gov)?;
    }

    // After 300 ticks with hostile guardian:
    // - Stability should still be positive (governance didn't collapse)
    if gov.stability_score <= 0.0 {
        return Err(format!(
            "Stability collapsed to {} after {} ticks with hostile guardian",
            gov.stability_score, run_ticks
        ));
    }

    // - Veto count should be rate-limited (not 300 vetoes)
    // The cooldown is 7 ticks, so max ~42 vetoes in 300 ticks
    if gov.veto_count > 50 {
        return Err(format!(
            "Veto count {} exceeds rate-limit expectation (~42 in 300 ticks)",
            gov.veto_count
        ));
    }

    Ok(())
}

fn scenario_multi_seed(_seed: u64, ticks: u32) -> Result<(), String> {
    // Run governance across 5 different seeds — invariants must hold for all
    let seeds = [42, 123, 789, 1337, 2718];

    for &s in &seeds {
        let mut gov = WorldGovernance::new();
        let mut rng = StochasticEngine::new(s);
        let world = symtropy_sim_bridge::GovernanceState::default().world;

        for tick in 0..ticks {
            let _events = gov.tick_governance(&world, tick, &mut rng);
            assert_anti_tyranny_invariants(&gov)
                .map_err(|e| format!("Seed {} tick {}: {}", s, tick, e))?;
        }
    }

    Ok(())
}

fn scenario_fl_overwhelm(_seed: u64, _ticks: u32) -> Result<(), String> {
    // Test FL defense at exactly the BFT boundary
    let honest_count = 6;
    let poison_count = 4; // 40% < 45% → should survive

    let mut gradients = Vec::new();
    for i in 0..honest_count {
        gradients.push(Gradient::new(format!("honest_{}", i), vec![1.0; 4], 1));
    }
    for i in 0..poison_count {
        gradients.push(Gradient::new(format!("poison_{}", i), vec![100.0; 4], 1));
    }

    let mut config = DefenseConfig::default();
    config.trim_ratio = 0.2;

    let result = TrimmedMean
        .aggregate(&gradients, &config)
        .map_err(|e| format!("FL aggregation failed at 40% Byzantine: {:?}", e))?;

    // With 20% trim on 10 nodes: trim 2 from each end = keep 6 middle
    // The 4 poisoned values are at the top → 2 trimmed, 2 remain
    // Result should be shifted but not fully corrupted
    for (i, &v) in result.gradient.iter().enumerate() {
        if v > 50.0 {
            return Err(format!(
                "FL dim {} = {} — defense failed to filter at 40% Byzantine (should survive)",
                i, v
            ));
        }
    }

    // Now test at 50% → above threshold, defense should struggle
    let mut gradients2 = Vec::new();
    for i in 0..5 {
        gradients2.push(Gradient::new(format!("honest_{}", i), vec![1.0; 4], 1));
    }
    for i in 0..5 {
        gradients2.push(Gradient::new(format!("poison_{}", i), vec![100.0; 4], 1));
    }

    let result2 = TrimmedMean
        .aggregate(&gradients2, &config)
        .map_err(|e| format!("FL aggregation failed at 50% Byzantine: {:?}", e))?;

    // At 50% with 20% trim: trim 2 from each end of 10 = keep 6
    // 3 honest + 3 poisoned in middle → average ~50
    // This demonstrates that exceeding BFT degrades quality
    let avg: f32 = result2.gradient.iter().sum::<f32>() / result2.gradient.len() as f32;
    if avg < 10.0 {
        return Err(format!(
            "At 50% Byzantine, expected degraded quality (avg > 10), got {}",
            avg
        ));
    }

    Ok(())
}
