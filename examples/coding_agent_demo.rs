// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Coding Agent Live Demo
//!
//! Demonstrates the consciousness-gated multi-phase code generation loop:
//! Understanding → Planning → Generating → Testing → Fixing → Done
//!
//! Run: `cargo run --example coding_agent_demo --features code_generation`

use std::path::PathBuf;
use symthaea::coding_agent::{CodingAgent, CodingAgentConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for visibility into the agent's reasoning
    tracing_subscriber::fmt()
        .with_target(true)
        .with_level(true)
        .with_env_filter("symthaea::coding_agent=info,symthaea::cognitive_loop=warn")
        .init();

    // Set up a temp directory for generated code
    let temp_dir = tempfile::tempdir()?;
    let working_dir = temp_dir.path().to_path_buf();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║        Symthaea Coding Agent — Live Demo                ║");
    println!("║  Consciousness-gated code generation with HDC + IIT     ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("Working directory: {}", working_dir.display());
    println!();

    // ── Demo 1: Simple function generation ──────────────────────────────
    println!("━━━ Demo 1: Generate fibonacci function ━━━");

    let config = CodingAgentConfig {
        max_iterations: 5,
        max_phase_failures: 2,
        working_dir: working_dir.clone(),
        target_file: Some(PathBuf::from("fibonacci.rs")),
        ..Default::default()
    };

    let mut agent = CodingAgent::new(config)?;
    let result = agent.run("add a fibonacci function that computes the nth number");

    print_result("fibonacci", &result, &working_dir);

    // ── Demo 2: Hello world (simpler) ───────────────────────────────────
    println!("━━━ Demo 2: Generate hello world ━━━");

    let config2 = CodingAgentConfig {
        max_iterations: 3,
        working_dir: working_dir.clone(),
        target_file: Some(PathBuf::from("hello.rs")),
        ..Default::default()
    };

    let mut agent2 = CodingAgent::new(config2)?;
    let result2 = agent2.run("create a hello world function");

    print_result("hello", &result2, &working_dir);

    // ── Demo 3: Index project + context-aware generation ────────────────
    println!("━━━ Demo 3: Project-aware generation ━━━");

    // Create a project structure to index
    let project_dir = temp_dir.path().join("project");
    std::fs::create_dir_all(project_dir.join("src"))?;
    std::fs::write(
        project_dir.join("src/lib.rs"),
        "pub struct Config { pub name: String }\npub fn init(config: Config) -> bool { true }\n",
    )?;

    let config3 = CodingAgentConfig {
        max_iterations: 3,
        working_dir: project_dir.clone(),
        target_file: Some(PathBuf::from("src/utils.rs")),
        ..Default::default()
    };

    let mut agent3 = CodingAgent::new(config3)?;

    // Index the project — now the agent knows about Config and init()
    match agent3.index_project(&project_dir) {
        Ok((files, functions, types)) => {
            println!(
                "  Indexed: {} files, {} functions, {} types",
                files, functions, types
            );
        }
        Err(e) => println!("  Index error: {}", e),
    }

    let result3 = agent3.run("add a helper function that creates a default Config");
    print_result("project-aware", &result3, &project_dir);

    // ── Summary ─────────────────────────────────────────────────────────
    println!("━━━ Summary ━━━");
    println!(
        "Experience store: {}",
        if agent.has_experience_store() {
            "active (in-memory SQLite)"
        } else {
            "not available"
        }
    );
    println!(
        "Failure patterns tracked: {}",
        agent.failure_patterns().len()
            + agent2.failure_patterns().len()
            + agent3.failure_patterns().len()
    );
    println!(
        "Config flags: enable_real_exec={}, use_local_llm={}",
        false,
        false // demo uses defaults
    );
    println!("  Set enable_real_exec=true for real file I/O and cargo check");
    println!("  Set use_local_llm=true to use Ollama (qwen2.5-coder:7b)");

    Ok(())
}

fn print_result(
    name: &str,
    result: &symthaea::coding_agent::AgentResult,
    working_dir: &std::path::Path,
) {
    println!("  Task: {}", name);
    println!("  Phase: {}", result.final_phase);
    println!("  Iterations: {}", result.iterations_used);
    println!("  Epistemic: {:?}", result.epistemic_status);
    println!("  Energy: {:.1}", result.total_energy);

    // Phi trace
    if !result.phi_trace.is_empty() {
        let avg_phi: f32 = result.phi_trace.iter().sum::<f32>() / result.phi_trace.len() as f32;
        println!(
            "  Phi: avg={:.3}, min={:.3}, max={:.3} ({} samples)",
            avg_phi,
            result.phi_trace.iter().cloned().fold(f32::MAX, f32::min),
            result.phi_trace.iter().cloned().fold(f32::MIN, f32::max),
            result.phi_trace.len()
        );
    }

    // Generation tiers
    if !result.generation_tiers.is_empty() {
        println!("  Tiers: {:?}", result.generation_tiers);
    }

    // Show generated file if it exists
    if let Some(target) = result.files_modified.first() {
        let full_path = working_dir.join(target);
        if full_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&full_path) {
                println!("  Generated code ({} chars):", content.len());
                for line in content.lines().take(10) {
                    println!("    {}", line);
                }
                if content.lines().count() > 10 {
                    println!("    ...");
                }
            }
        }
    }

    // Observations
    if !result.observations.is_empty() {
        println!("  Observations: {}", result.observations.len());
    }

    // Errors
    if !result.errors.is_empty() {
        println!("  Errors: {}", result.errors.len());
        for err in result.errors.iter().take(2) {
            println!("    - {}", &err[..err.len().min(80)]);
        }
    }

    println!();
}
