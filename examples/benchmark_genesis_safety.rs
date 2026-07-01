// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Genesis Mission Challenge 26: Safety Agents
//!
//! Demonstrates:
//! - Declining consciousness triggers level escalation
//! - NRC-style safety levels (Green → Yellow → Orange → Red)
//! - Audit report generation (markdown + JSON)

fn main() {
    println!("=== Genesis Mission Challenge 26: Safety Agents ===\n");

    use symthaea::safety::agent::{SafetyAgent, SafetyMetrics};
    use symthaea::safety::audit::SafetyAuditReport;
    use symthaea::safety::gate::{consciousness_gate, safety_gate};

    let mut agent = SafetyAgent::new();

    // Simulate declining consciousness
    println!("--- Consciousness Decline Simulation ---");
    let scenarios = [
        (0.85, 0.10, 0.80, "Healthy"),
        (0.75, 0.15, 0.70, "Slight decline"),
        (0.55, 0.30, 0.50, "Moderate decline"),
        (0.30, 0.60, 0.25, "Severe decline"),
        (0.10, 0.90, 0.10, "Critical"),
    ];

    for (i, (consciousness, pred_err, coherence, label)) in scenarios.iter().enumerate() {
        let metrics = SafetyMetrics {
            cycle: i,
            consciousness_level: *consciousness,
            prediction_error: *pred_err,
            temporal_coherence: *coherence,
            integrity_critical: false,
        };
        let assessment = agent.assess(metrics);
        println!(
            "  Cycle {}: {} → {} ({})",
            i,
            label,
            assessment.level.label(),
            assessment.reasons.join("; ")
        );
    }

    // Safety gate checks
    println!("\n--- Safety Gate Checks ---");
    let level = agent.current_level();
    let safe_op = safety_gate(level, false);
    let risky_op = safety_gate(level, true);
    println!("  Current level: {}", level.label());
    println!("  Safe operation: {:?}", safe_op);
    println!("  Risky operation: {:?}", risky_op);

    // Consciousness gate
    let gate_result = consciousness_gate(0.1, 0.5);
    println!(
        "  Consciousness gate (0.1 vs 0.5 threshold): {:?}",
        gate_result
    );

    // Generate audit report
    println!("\n--- Audit Report (Markdown) ---");
    let report = SafetyAuditReport::from_assessments(agent.history());
    println!("{}", report.to_markdown());

    println!("PASS: Safety Agents operational");
}
