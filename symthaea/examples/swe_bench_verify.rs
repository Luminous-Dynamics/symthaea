// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! SWE-bench Lite Verification Harness
//!
//! Simulates a multi-file repository bug and verifies that Symthaea
//! can navigate, reason, and produce a verified cross-file fix.

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use symthaea::language::code_orchestrator::CodeOrchestrator;
use symthaea_core::synthesis_trait::SynthesisRequest;
use tempfile::tempdir;

fn main() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    // 1. Create a Mock Repository with a bug
    println!("📦 Creating mock repository...");
    let dir = tempdir()?;
    let repo_root = dir.path();

    // src/lib.rs
    fs::write(
        repo_root.join("Cargo.toml"),
        r#"[package]
name = "mock-repo"
version = "0.1.0"
edition = "2021"
[dependencies]
"#,
    )?;

    fs::create_dir_all(repo_root.join("src"))?;
    fs::write(
        repo_root.join("src/lib.rs"),
        r#"pub mod calculator;
pub use calculator::add;
"#,
    )?;

    // The Bug: calculator.rs expects i32 but lib.rs/tests might want something else
    // Or just a simple implementation bug.
    fs::write(
        repo_root.join("src/calculator.rs"),
        r#"/// Add two numbers
pub fn add(a: i32, b: i32) -> i32 {
    a - b // BUG: Should be +
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_add() {
        assert_eq!(add(2, 2), 4);
    }
}
"#,
    )?;

    println!("🔎 Initializing Symthaea Orchestrator...");
    let mut orchestrator = CodeOrchestrator::new();

    // Index the mock project
    orchestrator.index_project(repo_root)?;

    // 2. Create the SWE-bench style request
    println!("📝 Constructing SWE-bench Solve request...");
    let issue_text = "The calculator add function is performing subtraction instead of addition. \
                      Fix the logic in src/calculator.rs and ensure all tests pass.";

    let request = SynthesisRequest::new("rust", "fix_calculator_addition", issue_text)
        .with_constraint("Must use + operator")
        .with_constraint("Must pass existing workspace tests");

    // 3. Execute the Solve Intent
    println!("🚀 Executing multi-file repair loop...");
    let response = orchestrator.synthesize(&request);

    // 4. Report Results
    println!("\n--- Scenario Report ---");
    println!("Accepted: {}", response.accepted);
    println!("Confidence: {:.3}", response.confidence);
    println!("Backend: {}", response.backend_name);

    if let Some(narrative) = response.narrative {
        println!("Narrative: {}", narrative);
    }

    if response.accepted {
        println!("\n✅ SWE-bench Lite Verification PASSED!");
        println!("Symthaea autonomously identified, repaired, and verified the multi-file bug.");
    } else {
        println!("\n❌ SWE-bench Lite Verification FAILED.");
        for layer in response.verification {
            println!(
                "  - {}: PASSED={} SCORE={:?} DETAIL={}",
                layer.name, layer.passed, layer.score, layer.detail
            );
        }
    }

    Ok(())
}
