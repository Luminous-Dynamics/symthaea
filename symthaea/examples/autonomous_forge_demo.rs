// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Autonomous Forge Demo: WASM Code Mutation
//!
//! Demonstrates the 'Asymmetric Swarm' and 'Autonomous Forge':
//! 1. Simulate receiving a pre-compiled .wasm optimization from the Swarm.
//! 2. Autonomous verification in the WasmSandbox.
//! 3. Hot-swapping the 'DNA' (Verified status) after successful test.

use symthaea::Symthaea;
use symthaea::action::PolicyBundle;
use tracing::Level;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    println!("\n🔨 Symthaea v0.7.0: The Autonomous Forge\n");

    // 1. Initialize Symthaea
    let mut sym = Symthaea::new(1024, 64).await?;
    sym.executor = symthaea::action::SimpleExecutor::with_real_commands();

    // Setup Policy
    let mut policy = PolicyBundle::restrictive();
    policy.capabilities.min_phi = 0.1;

    // 2. RECEIVE: Simulate receiving a .wasm optimization path from a peer
    // In a real swarm, the payload would be the actual bytes or a path to them.
    println!("[SWARM] Inbound optimization: 'optimized_similarity.wasm'...");
    let payload = b"optimizations/ssm_similarity_v2.wasm";

    // 3. THE FORGE: Process the incoming mutation
    println!("\n[FORGE] Commencing autonomous verification cycle...");
    sym.receive_swarm_message("optimization", payload).await?;

    // 4. VERIFY: Audit the telemetry to see the sandbox execution
    println!("\n[AUDIT] Checking Forge Telemetry...");
    for record in sym.executor.telemetry() {
        if let symthaea::action::ActionIR::WasmSandbox {
            module_path,
            function_name,
            ..
        } = &record.action
        {
            println!(
                "   -> Sandbox Executed: {}::{}()",
                module_path.display(),
                function_name
            );
        }
    }

    println!("\n[RESULT] AUTONOMOUS FORGE VERIFIED: Mutation verified and hot-loaded.");

    Ok(())
}
