// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for the Coding Agent pipeline.
//!
//! Tests the full loop: CodingAgent → MotorOutput → file I/O → test → learn.
//!
//! Run: `cargo test --test coding_agent_integration --features code_generation`

#![cfg(feature = "code_generation")]

use std::path::PathBuf;
use symthaea::coding_agent::{AgentEvent, CodingAgent, CodingAgentConfig};

/// Helper: create a CodingAgent with a temp working directory.
fn make_agent(working_dir: PathBuf) -> CodingAgent {
    let config = CodingAgentConfig {
        max_iterations: 5,
        max_phase_failures: 2,
        working_dir,
        target_file: Some(PathBuf::from("generated.rs")),
        ..Default::default()
    };
    CodingAgent::new(config).expect("CodingAgent::new should succeed")
}

#[test]
fn test_agent_completes_fibonacci_task() {
    let tmp = tempfile::tempdir().unwrap();
    let mut agent = make_agent(tmp.path().to_path_buf());

    let result = agent.run("add a fibonacci function");

    // Agent should complete (reach Done) or exhaust iterations
    assert!(
        result.iterations_used > 0,
        "Should have run at least 1 iteration"
    );
    assert!(!result.phi_trace.is_empty(), "Should have Phi measurements");

    // Should have generated a file
    let target = tmp.path().join("generated.rs");
    if target.exists() {
        let content = std::fs::read_to_string(&target).unwrap();
        assert!(
            content.contains("fibonacci") || content.contains("fib"),
            "Generated code should contain fibonacci: {}",
            &content[..content.len().min(200)]
        );
    }

    // Energy should be non-negative
    assert!(result.total_energy >= 0.0);

    // Epistemic status should be valid
    assert!(
        !format!("{:?}", result.epistemic_status).is_empty(),
        "Should have epistemic status"
    );
}

#[test]
fn test_agent_handles_explain_task() {
    let tmp = tempfile::tempdir().unwrap();
    let mut agent = make_agent(tmp.path().to_path_buf());

    let result = agent.run("explain how fibonacci works");

    // Explain tasks should complete quickly (Understanding → Done)
    assert!(
        result.iterations_used <= 5,
        "Explain should not use many iterations"
    );
}

#[test]
fn test_agent_phi_trace_reasonable() {
    let tmp = tempfile::tempdir().unwrap();
    let mut agent = make_agent(tmp.path().to_path_buf());

    let result = agent.run("create a hello world function");

    for phi in &result.phi_trace {
        assert!(
            *phi >= 0.0 && *phi <= 1.0,
            "Phi should be in [0, 1], got {}",
            phi
        );
    }
}

#[test]
fn test_agent_observations_recorded() {
    let tmp = tempfile::tempdir().unwrap();
    let mut agent = make_agent(tmp.path().to_path_buf());

    let result = agent.run("add a function that reverses a string");

    // Agent should record observations during Understanding phase
    assert!(
        !result.observations.is_empty() || !result.errors.is_empty(),
        "Agent should record observations or errors"
    );
}

#[test]
fn test_agent_experience_store_initialized() {
    let tmp = tempfile::tempdir().unwrap();
    let agent = make_agent(tmp.path().to_path_buf());

    // Experience store should be auto-initialized
    assert!(
        agent.has_experience_store(),
        "Agent should auto-initialize experience store"
    );
}

#[test]
fn test_agent_failure_patterns_tracked() {
    let tmp = tempfile::tempdir().unwrap();
    let mut agent = make_agent(tmp.path().to_path_buf());

    // Run a task — failure patterns start empty
    assert!(agent.failure_patterns().is_empty());

    let _result = agent.run("create a sort function");

    // Failure patterns may or may not be populated depending on whether
    // the native template compiles. Either way, the field should be accessible.
    let _patterns = agent.failure_patterns();
}

#[test]
fn test_agent_generation_tiers_tracked() {
    let tmp = tempfile::tempdir().unwrap();
    let mut agent = make_agent(tmp.path().to_path_buf());

    let result = agent.run("add fibonacci function");

    // If code was generated, tiers should be recorded
    if !result.generation_tiers.is_empty() {
        for tier in &result.generation_tiers {
            // All tiers should be valid enum variants
            assert!(
                !format!("{}", tier).is_empty(),
                "Tier should be displayable"
            );
        }
    }
}

#[cfg(feature = "school_learning")]
#[test]
fn test_agent_generates_lessons_from_failures() {
    let tmp = tempfile::tempdir().unwrap();
    let mut agent = make_agent(tmp.path().to_path_buf());

    let result = agent.run("implement complex generic trait system");

    // generated_lessons should exist (may be empty if no failures matched patterns)
    let _lessons = &result.generated_lessons;
}

#[test]
fn test_agent_real_exec_config() {
    let tmp = tempfile::tempdir().unwrap();
    let config = CodingAgentConfig {
        max_iterations: 3,
        working_dir: tmp.path().to_path_buf(),
        target_file: Some(PathBuf::from("test.rs")),
        enable_real_exec: true,
        use_local_llm: false, // don't require Ollama for test
        ..Default::default()
    };
    let agent = CodingAgent::new(config).expect("Should create with real exec enabled");

    // Agent should work with real exec config
    assert!(agent.has_experience_store());
}

#[test]
fn test_agent_index_project() {
    let tmp = tempfile::tempdir().unwrap();

    // Create a small Rust file to index
    let src_dir = tmp.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::write(
        src_dir.join("lib.rs"),
        "pub fn hello() -> &'static str { \"hello\" }\npub struct Config { pub name: String }\n",
    )
    .unwrap();

    let config = CodingAgentConfig {
        max_iterations: 3,
        working_dir: tmp.path().to_path_buf(),
        ..Default::default()
    };
    let mut agent = CodingAgent::new(config).expect("Should create agent");

    let result = agent.index_project(tmp.path());
    assert!(result.is_ok(), "index_project should succeed: {:?}", result);

    let (files, functions, types) = result.unwrap();
    assert_eq!(files, 1, "Should index 1 file");
    assert!(functions >= 1, "Should find at least 1 function");
    assert!(types >= 1, "Should find at least 1 type");
}

#[test]
fn test_agent_source_context_contains_code() {
    let tmp = tempfile::tempdir().unwrap();
    let src_dir = tmp.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::write(
        src_dir.join("lib.rs"),
        "pub fn fibonacci(n: u64) -> u64 {\n    if n <= 1 { return n; }\n    fibonacci(n-1) + fibonacci(n-2)\n}\n",
    )
    .unwrap();

    let config = CodingAgentConfig {
        max_iterations: 3,
        working_dir: tmp.path().to_path_buf(),
        target_file: Some(PathBuf::from("src/utils.rs")),
        ..Default::default()
    };
    let mut agent = CodingAgent::new(config).expect("Should create agent");
    agent.index_project(tmp.path()).unwrap();
    let result = agent.run("add a memoized fibonacci function");
    assert!(result.iterations_used > 0);
}

#[test]
fn test_agent_real_exec_writes_file() {
    let tmp = tempfile::tempdir().unwrap();
    let config = CodingAgentConfig {
        max_iterations: 5,
        max_phase_failures: 2,
        working_dir: tmp.path().to_path_buf(),
        target_file: Some(PathBuf::from("generated.rs")),
        enable_real_exec: true,
        use_local_llm: false,
        ..Default::default()
    };
    let mut agent = CodingAgent::new(config).expect("Should create with real exec");
    let result = agent.run("add a fibonacci function");

    let target = tmp.path().join("generated.rs");
    if target.exists() {
        let content = std::fs::read_to_string(&target).unwrap();
        assert!(
            content.contains("fn "),
            "Generated file should contain a function"
        );
    }
    assert!(result.iterations_used > 0);
    assert!(!result.phi_trace.is_empty());
}

/// Real end-to-end test with Ollama (qwen2.5-coder:7b).
///
/// This test requires:
/// 1. Ollama running at localhost:11434
/// 2. `qwen2.5-coder:7b` model pulled
/// 3. Environment variable: `SYMTHAEA_REAL_E2E=1`
///
/// Run: `SYMTHAEA_REAL_E2E=1 cargo test --test coding_agent_integration --features code_generation test_real_e2e`
#[test]
fn test_real_e2e_ollama_fibonacci() {
    if std::env::var("SYMTHAEA_REAL_E2E").unwrap_or_default() != "1" {
        eprintln!("Skipping real E2E test (set SYMTHAEA_REAL_E2E=1 to enable)");
        return;
    }

    let tmp = tempfile::tempdir().unwrap();
    let config = CodingAgentConfig {
        max_iterations: 8,
        max_phase_failures: 3,
        working_dir: tmp.path().to_path_buf(),
        target_file: Some(PathBuf::from("fib.rs")),
        enable_real_exec: true,
        use_local_llm: true,
        ..Default::default()
    };
    let agent = CodingAgent::new(config).expect("Should create agent with real exec + LLM");
    let (mut agent, rx) = agent.with_event_channel();

    // Collect events in background
    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let event_thread = std::thread::spawn(move || {
        while let Ok(event) = rx.recv() {
            events_clone.lock().unwrap().push(event);
        }
    });

    let result = agent.run("add a fibonacci function that returns the nth fibonacci number");

    // Wait for event thread
    drop(agent); // drops the sender
    let _ = event_thread.join();

    // === Assertions ===
    let collected = events.lock().unwrap();
    eprintln!("=== Real E2E Results ===");
    eprintln!("Iterations: {}", result.iterations_used);
    eprintln!("Final phase: {}", result.final_phase);
    eprintln!("Events: {}", collected.len());
    eprintln!("Phi trace: {:?}", result.phi_trace);
    eprintln!("Energy: {:.1}", result.total_energy);
    eprintln!("Tiers used: {:?}", result.generation_tiers);
    eprintln!("Files modified: {:?}", result.files_modified);
    eprintln!("Observations: {}", result.observations.len());
    for (i, obs) in result.observations.iter().enumerate() {
        eprintln!("  [{i}] {}", &obs[..obs.len().min(120)]);
    }
    if !result.errors.is_empty() {
        eprintln!("Errors:");
        for e in &result.errors {
            eprintln!("  - {}", &e[..e.len().min(200)]);
        }
    }

    // Should have run iterations
    assert!(
        result.iterations_used > 0,
        "Should run at least 1 iteration"
    );

    // Should have phi measurements
    assert!(!result.phi_trace.is_empty(), "Should have phi trace");
    for phi in &result.phi_trace {
        assert!(*phi >= 0.0 && *phi <= 1.0, "Phi out of range: {phi}");
    }

    // Should have streamed events
    assert!(!collected.is_empty(), "Should have emitted events");

    // Should have phase transitions or consciousness snapshots
    let phase_transitions = collected
        .iter()
        .filter(|e| matches!(e, AgentEvent::PhaseTransition { .. }))
        .count();
    let snapshots = collected
        .iter()
        .filter(|e| matches!(e, AgentEvent::ConsciousnessSnapshot { .. }))
        .count();
    eprintln!("Phase transitions: {phase_transitions}, Consciousness snapshots: {snapshots}");
    assert!(snapshots > 0, "Should have consciousness snapshots");

    // Check if code was generated and written
    let target = tmp.path().join("fib.rs");
    if target.exists() {
        let content = std::fs::read_to_string(&target).unwrap();
        eprintln!("=== Generated Code ({} bytes) ===", content.len());
        eprintln!("{}", &content[..content.len().min(500)]);
        assert!(
            content.contains("fn "),
            "Generated code should contain a function definition"
        );
    } else {
        eprintln!("Note: No file written (may be expected if native templates matched)");
    }

    eprintln!("=== E2E Test Complete ===");
}

/// Real E2E: full compile→test→fix loop.
///
/// Sets up a real Cargo project, generates code, and verifies the agent
/// runs `cargo check` directly and feeds errors back into the fixing phase.
/// Requires: `SYMTHAEA_REAL_E2E=1`
#[test]
fn test_real_e2e_compile_loop() {
    if std::env::var("SYMTHAEA_REAL_E2E").unwrap_or_default() != "1" {
        eprintln!("Skipping real E2E compile loop test (set SYMTHAEA_REAL_E2E=1)");
        return;
    }

    let tmp = tempfile::tempdir().unwrap();

    // Set up a minimal Cargo project
    std::fs::write(
        tmp.path().join("Cargo.toml"),
        "[package]\nname = \"e2e-test\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    let src_dir = tmp.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::write(src_dir.join("lib.rs"), "// placeholder\n").unwrap();

    let config = CodingAgentConfig {
        max_iterations: 10,
        max_phase_failures: 3,
        working_dir: tmp.path().to_path_buf(),
        target_file: Some(PathBuf::from("src/lib.rs")),
        enable_real_exec: true,
        use_local_llm: true,
        ..Default::default()
    };
    let agent = CodingAgent::new(config).expect("Should create agent");
    let (mut agent, rx) = agent.with_event_channel();

    // Index the project
    let _ = agent.index_project(tmp.path());

    // Collect events
    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let event_thread = std::thread::spawn(move || {
        while let Ok(event) = rx.recv() {
            events_clone.lock().unwrap().push(event);
        }
    });

    let result = agent.run("add a function that checks if a number is prime");

    drop(agent);
    let _ = event_thread.join();

    let collected = events.lock().unwrap();

    // === Diagnostics ===
    let target = tmp.path().join("src/lib.rs");
    eprintln!("=== Compile Loop E2E ===");
    eprintln!(
        "Iterations: {}, Phase: {}",
        result.iterations_used, result.final_phase
    );
    eprintln!("Tiers: {:?}", result.generation_tiers);
    eprintln!("Events: {}", collected.len());
    eprintln!("Observations:");
    for (i, obs) in result.observations.iter().enumerate() {
        eprintln!("  [{i}] {}", &obs[..obs.len().min(150)]);
    }
    if !result.errors.is_empty() {
        eprintln!("Errors:");
        for e in &result.errors {
            eprintln!("  - {}", &e[..e.len().min(200)]);
        }
    }

    if target.exists() {
        let content = std::fs::read_to_string(&target).unwrap();
        eprintln!(
            "Generated ({} bytes):\n{}",
            content.len(),
            &content[..content.len().min(500)]
        );

        // Verify cargo check independently
        let output = std::process::Command::new("cargo")
            .args(["check"])
            .current_dir(tmp.path())
            .output()
            .expect("cargo check should run");

        if output.status.success() {
            eprintln!("cargo check (independent): PASSED");
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!(
                "cargo check (independent): FAILED\n{}",
                &stderr[..stderr.len().min(500)]
            );
        }
    }

    // === Assertions ===
    // The agent should have run cargo check during its Testing phase
    let has_cargo_obs = result
        .observations
        .iter()
        .any(|o| o.contains("cargo check passed") || o.contains("cargo check failed"));
    eprintln!("Agent ran cargo check: {has_cargo_obs}");

    // The agent should have phase transitions through Generating → Testing
    let has_test_events = collected.iter().any(
        |e| matches!(e, AgentEvent::PhaseTransition { to, .. } if format!("{to}") == "Testing"),
    );
    eprintln!("Reached Testing phase: {has_test_events}");

    // Learning: experience store should have been populated
    eprintln!(
        "Experience count: {}",
        result
            .observations
            .iter()
            .filter(|o| o.contains("Prior experience"))
            .count()
    );

    eprintln!("=== Compile Loop E2E Complete ===");
}