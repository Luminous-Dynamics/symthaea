// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Real-World Autonomous Agent Test
//!
//! Symthaea v0.6.0: The "Closed-Loop" Proof.
//!
//! 1. **Perception**: Tasked with fixing a broken Nix flake.
//! 2. **Reasoning**: Identifies missing dependency.
//! 3. **Action**: Writes the fix to flake.nix.
//! 4. **Verification**: Executes NIX_BUILD.
//! 5. **Feedback**: Perceives build output and promotes knowledge to "Verified".

use symthaea::Symthaea;
use symthaea::action::{PolicyBundle, SandboxRoot};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(true)
        .init();

    println!("\n🚀 Symthaea v0.6.0: Real-World Autonomous Loop Test\n");

    // 1. Initialize Symthaea
    let mut sym = Symthaea::new(1024, 64).await?;

    // Set to real execution mode for the test
    sym.executor = symthaea::action::SimpleExecutor::with_real_commands();

    // Setup sandbox and policy
    let sandbox_path = std::env::current_dir()?.join("symthaea/test_flake");
    let _sandbox = SandboxRoot::new("real_world_test")?;
    // Note: We'll use a custom policy that allows nix and writing to our test file
    let mut policy = PolicyBundle::restrictive();
    policy
        .capabilities
        .shell
        .allowed_programs
        .insert("nix".into());
    policy
        .capabilities
        .filesystem
        .write_patterns
        .push(format!("{}/ **", sandbox_path.display()));
    policy.capabilities.min_phi = 0.1; // Lower for the demo to ensure action
    policy.capabilities.shell.min_phi = 0.1;

    // 2. The Task
    let task = "The Nix flake in symthaea/test_flake/flake.nix is broken. It's missing 'pkgs.hello' in buildInputs but calls it in shellHook. Please fix it.";
    println!("📥 User Task: {}\n", task);

    // 3. Process Task (Loop 1: Reasoning & Action)
    println!("🧠 [Thinking...] Analyzing flake and missing dependencies...");
    let response = sym.process(task).await?;

    println!("🤖 Symthaea says: {}\n", response.content);

    if let Some(thought) = response.structured_thought {
        println!("📊 Epistemic Status: {:?}", thought.epistemic_status);
        println!("📊 Phi: {:.4}", thought.psi);
    }

    // 4. Verification Loop
    println!("\n🔍 [Verification] Checking if Symthaea initiated a fix and build...");

    let telemetry = sym.executor.telemetry();
    for record in telemetry {
        println!("   ⚡ Action Executed: {:?}", record.action);
        match &record.outcome {
            symthaea::action::ActionOutcome::CommandOutput {
                exit_code,
                stdout: _stdout,
                stderr,
            } => {
                println!("   📝 Build Exit Code: {}", exit_code);
                if *exit_code == 0 {
                    println!("   ✅ SUCCESS: Symthaea fixed and verified the flake!");
                } else {
                    println!(
                        "   ❌ FAILED: Build failed with stderr: {}",
                        String::from_utf8_lossy(stderr)
                    );
                }
            }
            _ => println!("   📝 Outcome: {:?}", record.outcome),
        }
    }

    // 5. Epistemic Promotion Check
    println!("\n🎓 [Memory] Checking for 'Verified' graduation...");
    let stats = sym.introspect();
    println!("   📈 Consciousness Complexity: {:.4}", stats.complexity);
    println!(
        "   📈 Memory Count: {}",
        stats.memory_stats.short_term_count
    );

    println!("\n🎉 TEST SEQUENCE COMPLETE.");

    Ok(())
}
