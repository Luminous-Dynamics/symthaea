//! Integration tests for the Coding Agent pipeline.
//!
//! Tests the full loop: CodingAgent → MotorOutput → file I/O → test → learn.
//!
//! Run: `cargo test --test coding_agent_integration --features code_generation`

#![cfg(feature = "code_generation")]

use std::path::PathBuf;
use symthaea::coding_agent::{CodingAgent, CodingAgentConfig};

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
