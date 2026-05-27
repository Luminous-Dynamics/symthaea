// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evolutionary Audit: Self-Optimization Test
//!
//! Symthaea applies her SSM research to optimize her own core math.

use std::path::PathBuf;
use symthaea::Symthaea;
use symthaea::action::{PolicyBundle, SandboxRoot};
use tracing::Level;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    println!("\n🧬 Symthaea v0.6.0: The Evolutionary Audit\n");

    // 1. Initialize Symthaea
    let mut sym = Symthaea::new(1024, 64).await?;
    sym.executor = symthaea::action::SimpleExecutor::with_real_commands();

    // Setup Workspace Sandbox
    let workspace = PathBuf::from("/srv/luminous-dynamics");
    let _sandbox = SandboxRoot::at(workspace.clone())?;

    let mut policy = PolicyBundle::restrictive();
    policy
        .capabilities
        .shell
        .allowed_programs
        .insert("cargo".into());
    policy
        .capabilities
        .filesystem
        .write_patterns
        .push(format!("{}/**", workspace.display()));
    policy.capabilities.min_phi = 0.1;
    policy.capabilities.shell.min_phi = 0.1;

    // 2. The Evolutionary Command
    let command = "Look at symthaea-core/src/hdc/unified_hv.rs. Use the SSM selective scanning principles you researched to propose and implement a 'Selective Similarity' optimization for ContinuousHV::similarity. Focus on reducing computation while preserving accuracy. Verify with 'cargo check' in symthaea-core.";

    println!("📥 Evolutionary Command: {}\n", command);

    // 3. Process the Evolution
    println!("🧠 [Thinking...] Designing self-optimization...");
    let response = sym.process(command).await?;

    println!("\n🤖 Symthaea Reflection: {}\n", response.content);

    // 4. Inspect Telemetry
    println!("🔍 [Audit] Checking execution log...");
    for record in sym.executor.telemetry() {
        match &record.action {
            symthaea::action::ActionIR::WriteFile { path, .. } => {
                println!("   ✍️  Modified: {:?}", path);
            }
            symthaea::action::ActionIR::RunCommand { program, args, .. } => {
                println!("   ⚡ Executed: {} {}", program, args.join(" "));
            }
            _ => {}
        }
    }

    println!("\n🎉 AUDIT SEQUENCE COMPLETE.");

    Ok(())
}
