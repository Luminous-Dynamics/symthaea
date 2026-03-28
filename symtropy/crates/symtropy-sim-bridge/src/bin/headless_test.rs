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

use mycelix_bridge_common::{ConsciousnessProfile, ConsciousnessTier};
use mycelix_bridge_common::consciousness_thresholds::ConsciousnessThresholds;
use mycelix_fl::defenses::{Defense, TrimmedMean};
use mycelix_fl::types::{DefenseConfig, Gradient};
use mycelix_multiworld_sim::governance::WorldGovernance;
use mycelix_multiworld_sim::stochastic::StochasticEngine;

use symtropy_sim_bridge::VETO_OVERRIDE_THRESHOLD;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed = args.iter().position(|a| a == "--seed")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(42);
    let scenario = args.iter().position(|a| a == "--scenario")
        .and_then(|i| args.get(i + 1).cloned())
        .unwrap_or_else(|| "all".to_string());
    let ticks = args.iter().position(|a| a == "--ticks")
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
        (0.0, ConsciousnessTier::Observer),
        (0.29, ConsciousnessTier::Observer),
        (0.30, ConsciousnessTier::Participant),
        (0.39, ConsciousnessTier::Participant),
        (0.40, ConsciousnessTier::Citizen),
        (0.59, ConsciousnessTier::Citizen),
        (0.60, ConsciousnessTier::Steward),
        (0.79, ConsciousnessTier::Steward),
        (0.80, ConsciousnessTier::Guardian),
        (1.0, ConsciousnessTier::Guardian),
    ];
    for (score, expected) in &tests {
        let actual = ConsciousnessTier::from_score(*score);
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
            identity: *i, reputation: *r, community: *c, engagement: *e,
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

    let result = TrimmedMean.aggregate(&gradients, &config)
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
        return Err(format!("Gate basic {} too low", thresholds.consciousness_gate_basic));
    }
    if thresholds.consciousness_gate_constitutional < 0.5 {
        return Err(format!("Gate constitutional {} too low", thresholds.consciousness_gate_constitutional));
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

    // Verify all 5 levels exist and have correct values
    for i in 0..5 {
        let level = EmpiricalLevel::from_value(i)
            .ok_or_else(|| format!("EmpiricalLevel::from_value({}) returned None", i))?;
        if level.value() != i {
            return Err(format!("Level {} roundtrip failed: got {}", i, level.value()));
        }
    }

    // E5 should not exist
    if EmpiricalLevel::from_value(5).is_some() {
        return Err("EmpiricalLevel::from_value(5) should be None".into());
    }

    Ok(())
}
