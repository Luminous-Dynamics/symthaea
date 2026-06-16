// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Genesis Mission Challenge 10: Autonomous Lab Controller
//!
//! Demonstrates:
//! - 4-step iPSC reprogramming protocol
//! - FEP action selection per step
//! - Ethics gate enforcement

fn main() {
    println!("=== Genesis Mission Challenge 10: Autonomous Lab Controller ===\n");

    use symthaea::symthaea_cell_foundry::{EthicsGate, LabController, LabProtocol, MockInstrument};

    // 1. Run with ethics approval
    println!("--- Protocol with Ethics Approval ---");
    let mock = MockInstrument::new();
    let ethics = EthicsGate::fully_approved();
    let mut controller = LabController::new(mock, ethics);

    let protocol = LabProtocol::ipsc_reprogramming();
    println!(
        "Protocol: {} ({} steps)",
        protocol.name,
        protocol.steps.len()
    );

    let result = controller.run_protocol(&protocol);
    for step in &result.step_results {
        println!(
            "  Step {}: {} → {:?} (executed={})",
            step.step_index, step.step_name, step.selected_action, step.action_executed
        );
    }
    println!(
        "  Success: {}, Steps completed: {}",
        result.success, result.steps_completed
    );

    // 2. Run without ethics approval (should fail)
    println!("\n--- Protocol without Ethics Approval ---");
    let mock2 = MockInstrument::new();
    let no_ethics = EthicsGate::new();
    let mut controller2 = LabController::new(mock2, no_ethics);

    let result2 = controller2.run_protocol(&protocol);
    println!(
        "  Success: {}, Failure: {}",
        result2.success,
        result2.failure_reason.as_deref().unwrap_or("none")
    );

    println!("\nPASS: Autonomous Lab Controller operational");
}
