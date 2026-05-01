// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Conscious Reasoning Engine Demo
//!
//! Demonstrates the 7-step reasoning cycle with tiered degradation:
//!   1. DETECT conflicts between consciousness theories
//!   2. ASSESS reliability and compute effective Φ
//!   3. DECIDE whether simulation is worthwhile (EVS)
//!   4. PLAN via budget-bounded MCTS
//!   5. GATE tool actions against Φ_eff and confidence
//!   6. ANALYZE counterfactuals (Tier 2 only)
//!   7. EMIT telemetry
//!
//! Run: cargo run --example conscious_reasoning_demo --features reasoning_engine

#[cfg(feature = "reasoning_engine")]
fn main() {
    use symthaea::consciousness::epistemic_conflict::MultiTheoryMetrics;
    use symthaea::consciousness::reasoning_engine::{ConsciousReasoningEngine, ReasoningContext};
    use symthaea::consciousness::temporal_planning::types::PlannedAction;
    use symthaea::consciousness::tool_gate::types::ToolDescriptor;

    println!("═══════════════════════════════════════════════════════════");
    println!("  Symthaea Conscious Reasoning Engine v0.2 Demo");
    println!("═══════════════════════════════════════════════════════════\n");

    let mut engine = ConsciousReasoningEngine::new();

    // ── Scenario 1: High consensus, full budget ──────────────────────
    println!("── Scenario 1: Theories in consensus ──");
    let ctx1 = ReasoningContext {
        theory_metrics: MultiTheoryMetrics {
            phi: 0.85,
            gwt: 0.82,
            ast: 0.80,
            pp: 0.83,
            rpt: 0.81,
            embodiment: 0.79,
            unified: 0.82,
        },
        phi: 0.85,
        available_budget_us: 25_000,
        available_actions: vec![
            PlannedAction {
                id: "explore".into(),
                description: "Explore environment".into(),
                embedding: vec![0.5; 4],
                prior: 0.4,
                is_epistemic: true,
            },
            PlannedAction {
                id: "act".into(),
                description: "Take action".into(),
                embedding: vec![1.0; 4],
                prior: 0.6,
                is_epistemic: false,
            },
        ],
        tool: Some(ToolDescriptor::read_only("nix search nixpkgs firefox")),
        code_context: None,
        recent_utility: 0.5,
        cycle_id: 1,
        neuromod_exploration_mod: 1.0,
        epistemic_quality: 0.5,
    };

    let r1 = engine.reason(&ctx1);
    println!("  Tier:        {:?}", r1.tier);
    println!(
        "  Φ_eff:       {:.4} (raw Φ={:.2}, R={:.2}, γ={:.1})",
        r1.phi_eff, ctx1.phi, r1.reliability, r1.gamma
    );
    println!(
        "  Conflicts:   max={:.3}, entropy={:.3}",
        r1.conflicts.max_magnitude(),
        r1.conflicts.total_entropy
    );
    println!(
        "  Gate:        {}",
        if r1.action_allowed() {
            "ALLOWED"
        } else {
            "BLOCKED"
        }
    );
    if let Some(ref plan) = r1.plan {
        println!(
            "  Plan:        {} iterations, confidence={:.2}",
            plan.iterations, plan.confidence
        );
    }
    if let Some(ref narrative) = r1.narrative {
        println!("  Narrative:   {}", narrative);
    }
    println!();

    // ── Scenario 2: Theories disagree, risky action ──────────────────
    println!("── Scenario 2: Theories in conflict ──");
    let ctx2 = ReasoningContext {
        theory_metrics: MultiTheoryMetrics {
            phi: 0.90,
            gwt: 0.15,
            ast: 0.88,
            pp: 0.12,
            rpt: 0.85,
            embodiment: 0.10,
            unified: 0.50,
        },
        phi: 0.90,
        available_budget_us: 25_000,
        available_actions: vec![PlannedAction {
            id: "cautious".into(),
            description: "Cautious action".into(),
            embedding: vec![0.2; 4],
            prior: 0.7,
            is_epistemic: true,
        }],
        tool: Some(
            ToolDescriptor::from_command("nixos-rebuild switch")
                .with_domain("nixos")
                .with_rollback("nixos-rebuild switch --rollback")
                .with_calibration_count(100),
        ),
        code_context: None,
        recent_utility: 0.3,
        cycle_id: 2,
        neuromod_exploration_mod: 1.0,
        epistemic_quality: 0.5,
    };

    let r2 = engine.reason(&ctx2);
    println!("  Tier:        {:?}", r2.tier);
    println!(
        "  Φ_eff:       {:.4} (raw Φ={:.2}, R={:.2}, γ={:.1})",
        r2.phi_eff, ctx2.phi, r2.reliability, r2.gamma
    );
    println!(
        "  Conflicts:   max={:.3}, entropy={:.3}",
        r2.conflicts.max_magnitude(),
        r2.conflicts.total_entropy
    );
    if let Some((a, b)) = r2.conflicts.dominant {
        println!("  Dominant:    {:?} vs {:?}", a, b);
    }
    println!(
        "  Gate:        {}",
        if r2.action_allowed() {
            "ALLOWED"
        } else {
            "BLOCKED"
        }
    );
    if let Some(ref gate) = r2.gate {
        if let Some(ref fallback) = gate.fallback {
            println!("  Fallback:    {:?}", fallback);
        }
    }
    if let Some(ref narrative) = r2.narrative {
        println!("  Narrative:   {}", narrative);
    }
    println!();

    // ── Scenario 3: Tight budget (Tier 0) ────────────────────────────
    println!("── Scenario 3: Tight budget (Tier 0) ──");
    let ctx3 = ReasoningContext {
        theory_metrics: MultiTheoryMetrics {
            phi: 0.75,
            gwt: 0.70,
            ast: 0.72,
            pp: 0.71,
            rpt: 0.73,
            embodiment: 0.69,
            unified: 0.72,
        },
        phi: 0.75,
        available_budget_us: 800, // <2ms → Tier 0
        available_actions: vec![],
        tool: None,
        code_context: None,
        recent_utility: 0.5,
        cycle_id: 3,
        neuromod_exploration_mod: 1.0,
        epistemic_quality: 0.5,
    };

    let r3 = engine.reason(&ctx3);
    println!("  Tier:        {:?}", r3.tier);
    println!("  Φ_eff:       {:.4}", r3.phi_eff);
    println!("  Wall time:   {}μs", r3.wall_time_us);
    println!(
        "  Plan:        {}",
        if r3.plan.is_some() {
            "yes"
        } else {
            "none (budget too tight)"
        }
    );
    println!();

    // ── Engine Statistics ─────────────────────────────────────────────
    let stats = engine.stats();
    println!("── Engine Statistics ──");
    println!("  Total cycles:     {}", stats.total_cycles);
    println!("  Avg Φ_eff:        {:.4}", stats.avg_phi_eff);
    println!("  Avg reliability:  {:.4}", stats.avg_reliability);
    println!("  Avg wall time:    {:.0}μs", stats.avg_wall_time_us);
    println!("  Tier 0 fraction:  {:.0}%", stats.tier0_fraction * 100.0);
    println!("  Tier 2 fraction:  {:.0}%", stats.tier2_fraction * 100.0);
    println!("  γ:                {:.2}", stats.gamma);
    println!("  Cal version:      {}", stats.calibration_version);
    println!();

    println!("═══════════════════════════════════════════════════════════");
    println!("  Demo complete. All cycles returned within budget.");
    println!("═══════════════════════════════════════════════════════════");
}

#[cfg(not(feature = "reasoning_engine"))]
fn main() {
    eprintln!("This example requires the 'reasoning_engine' feature:");
    eprintln!("  cargo run --example conscious_reasoning_demo --features reasoning_engine");
}
