// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Autonomous Forge Demo: WASM Code Mutation
//!
//! Demonstrates the current 'Asymmetric Swarm' / 'Autonomous Forge' experiment:
//! 1. Simulate receiving a pre-compiled `.wasm` optimization path from the Swarm.
//! 2. Ask the current pipeline to attempt local WASM verification.
//! 3. Audit the executor telemetry and report only the result actually observed.
//!
//! This demo does **not** establish artifact provenance, Forge authorization,
//! promotion to a verified state, or hot-loading. Those require stronger typed
//! evidence and authority boundaries before they can be claimed.

use symthaea::Symthaea;
use tracing::Level;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    println!("\n🔨 Symthaea v0.7.0: The Autonomous Forge\n");

    // 1. Initialize Symthaea.
    //
    // This intentionally exercises the existing real-execution path. It is an
    // experimental demo, not evidence that the Forge boundary is hardened.
    let mut sym = Symthaea::new(1024, 64).await?;
    sym.executor = symthaea::action::SimpleExecutor::with_real_commands();

    // 2. RECEIVE: Simulate receiving a .wasm optimization path from a peer.
    // In a real swarm, artifact bytes/identity/provenance need a typed boundary;
    // this legacy demo still passes a path-shaped payload into receive_swarm_message.
    println!("[SWARM] Inbound optimization: 'optimized_similarity.wasm'...");
    let payload = b"optimizations/ssm_similarity_v2.wasm";

    // 3. THE FORGE: Process the incoming candidate through the current pipeline.
    println!("\n[FORGE] Commencing experimental verification attempt...");
    sym.receive_swarm_message("optimization", payload).await?;

    // 4. OBSERVE: Audit telemetry without promoting the observation into a
    // stronger claim such as "artifact verified" or "hot-loaded".
    println!("\n[AUDIT] Checking Forge telemetry...");
    let mut wasm_attempts = 0usize;
    let mut protocol_successes = 0usize;

    for record in sym.executor.telemetry() {
        if let symthaea::action::ActionIR::WasmSandbox {
            module_path,
            function_name,
            ..
        } = &record.action
        {
            wasm_attempts += 1;
            println!(
                "   -> WASM action observed: {}::{}()",
                module_path.display(),
                function_name
            );

            match &record.outcome {
                symthaea::action::ActionOutcome::WasmResult { output, logs } => {
                    let protocol_success = output.first().copied() == Some(1);
                    if protocol_success {
                        protocol_successes += 1;
                    }
                    println!(
                        "      protocol_result={} logs={}",
                        if protocol_success { "success" } else { "failure" },
                        logs.join("; ")
                    );
                }
                other => {
                    println!("      unexpected outcome: {other:?}");
                }
            }
        }
    }

    if wasm_attempts == 0 {
        println!("\n[RESULT] No WASM execution attempt was observed in executor telemetry.");
    } else {
        println!(
            "\n[RESULT] Forge observation: {wasm_attempts} WASM attempt(s); {protocol_successes} returned the current protocol success marker."
        );
    }

    println!(
        "[BOUNDARY] No promotion or hot-load is claimed: artifact provenance, explicit Forge authority, sandbox containment, and claim-scoped verification remain separate requirements."
    );

    Ok(())
}
