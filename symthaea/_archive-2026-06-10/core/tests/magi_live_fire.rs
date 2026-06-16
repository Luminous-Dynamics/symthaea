#![cfg(any(feature = "magi_loop", feature = "full_consciousness"))]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # MAGI Loop Live Fire Integration Test
//!
//! This test suite proves that the MAGI Loop can **touch the world**.
//!
//! Unlike unit tests that mock the OS, these tests:
//! - Spawn real child processes
//! - Create real files on the filesystem
//! - Capture real exit codes
//! - Parse real command output
//!
//! ## The Crossing Criterion
//!
//! > "The system that can close this loop end-to-end, even imperfectly,
//! > will have crossed a line that cannot be uncrossed."
//!
//! This test file is that crossing.

use std::path::PathBuf;
use std::time::Duration;
use tempfile::TempDir;

use symthaea::consciousness::recursive_improvement::{
    // Active Inference Bridge
    ActiveInferenceBridge,
    ActiveInferenceBridgeConfig,
    // Core MAGI types
    OutcomeCategory,
    // Resolution System
    resolution::{ExitCodeResolver, Resolver, ResourceStateResolver},
};

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 1: THE BASIC CROSSING - File Creation
// ═══════════════════════════════════════════════════════════════════════════════

/// The simplest possible world-touching test:
/// 1. Predict that `touch evidence.txt` will succeed
/// 2. Execute the command
/// 3. Verify the file exists
/// 4. Feed success back to calibration
#[test]
fn test_live_fire_file_creation() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let evidence_path = temp_dir.path().join("evidence.txt");
    let mut bridge = ActiveInferenceBridge::new(ActiveInferenceBridgeConfig {
        min_data_points: 1,
        ..Default::default()
    });

    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 1: PREDICT
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n═══ STEP 1: PREDICT ═══");
    println!("Predicting: 'touch evidence.txt' will succeed (exit 0) and file will exist");

    let confidence = 0.85;

    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 2: ACT (Execute the command)
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n═══ STEP 2: ACT ═══");
    println!("Executing: touch {:?}", evidence_path);

    // Actually execute the command
    let exit_resolver = ExitCodeResolver::new("touch", vec![0])
        .with_args(vec![evidence_path.to_string_lossy().to_string()]);

    let exit_result = exit_resolver.execute(Duration::from_secs(10));

    println!("Exit code result: {:?}", exit_result.outcome);
    println!("Raw output: {:?}", exit_result.raw_output);

    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 3: OBSERVE (Check the world state)
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n═══ STEP 3: OBSERVE ═══");

    // Check if file exists using ResourceStateResolver
    let resource_resolver = ResourceStateResolver::exists(evidence_path.to_string_lossy());
    let resource_result = resource_resolver.execute();

    println!("File exists check: {:?}", resource_result.outcome);

    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 4: RESOLVE (Determine overall success)
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n═══ STEP 4: RESOLVE ═══");

    let overall_success = matches!(exit_result.outcome, OutcomeCategory::Success)
        && matches!(resource_result.outcome, OutcomeCategory::Success);

    println!("Overall success: {}", overall_success);

    // ═══════════════════════════════════════════════════════════════════════════
    // STEP 5: UPDATE (Feed back to Active Inference Bridge)
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n═══ STEP 5: UPDATE ═══");

    bridge.observe_resolution(confidence, overall_success);

    let stats = bridge.statistics();
    println!("Bridge statistics after observation:");
    println!("  Total observations: {}", stats.total_observations);
    println!(
        "  Average prediction error: {:?}",
        stats.average_prediction_error
    );
    println!("  Average outcome: {:?}", stats.average_outcome);

    // ═══════════════════════════════════════════════════════════════════════════
    // ASSERTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    println!("\n═══ ASSERTIONS ═══");

    // The command should have succeeded
    assert!(
        matches!(exit_result.outcome, OutcomeCategory::Success),
        "touch command should succeed"
    );

    // The file should exist
    assert!(
        evidence_path.exists(),
        "evidence.txt should exist after touch"
    );

    // The resource check should confirm existence
    assert!(
        matches!(resource_result.outcome, OutcomeCategory::Success),
        "ResourceStateResolver should confirm file exists"
    );

    // The bridge should have recorded the observation
    assert_eq!(
        stats.total_observations, 1,
        "Bridge should have 1 observation"
    );

    println!("\n✓ LIVE FIRE TEST PASSED: File creation loop closed successfully");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 2: THE FAILURE PATH - Predicting Failure
// ═══════════════════════════════════════════════════════════════════════════════

/// Test that the system correctly handles predicted and actual failure:
/// 1. Predict that accessing a non-existent path will fail
/// 2. Verify the prediction was correct (file doesn't exist)
/// 3. This tests calibration for "confident about failure"
#[test]
fn test_live_fire_predicted_failure() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let nonexistent_path = temp_dir.path().join("definitely_does_not_exist_12345.txt");
    let mut bridge = ActiveInferenceBridge::new(ActiveInferenceBridgeConfig {
        min_data_points: 1,
        ..Default::default()
    });

    println!("\n═══ PREDICTED FAILURE TEST ═══");
    println!("Predicting: File {:?} does NOT exist", nonexistent_path);

    // Predict with high confidence that file doesn't exist
    let confidence = 0.95;

    // Check using ResourceStateResolver for non-existence
    let resolver = ResourceStateResolver::not_exists(nonexistent_path.to_string_lossy());
    let result = resolver.execute();

    println!("Result: {:?}", result.outcome);

    // The prediction should be correct (file doesn't exist = success for not_exists check)
    let prediction_correct = matches!(result.outcome, OutcomeCategory::Success);

    bridge.observe_resolution(confidence, prediction_correct);

    let stats = bridge.statistics();
    println!("Prediction correct: {}", prediction_correct);
    println!("Bridge observations: {}", stats.total_observations);

    assert!(
        prediction_correct,
        "Prediction about non-existence should be correct"
    );
    assert_eq!(stats.total_observations, 1);

    println!("\n✓ PREDICTED FAILURE TEST PASSED");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 3: CALIBRATION DRIFT - Multiple Predictions
// ═══════════════════════════════════════════════════════════════════════════════

/// Test that calibration actually updates over multiple predictions:
/// 1. Make several predictions with varying confidence
/// 2. Execute real commands
/// 3. Verify the Brier score and ECE reflect actual performance
#[test]
fn test_live_fire_calibration_drift() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let mut bridge = ActiveInferenceBridge::new(ActiveInferenceBridgeConfig {
        min_data_points: 3,
        ..Default::default()
    });

    println!("\n═══ CALIBRATION DRIFT TEST ═══");

    // Scenario: Make predictions about file creation with varying confidence
    let predictions = vec![
        ("file1.txt", 0.9),
        ("file2.txt", 0.8),
        ("file3.txt", 0.7),
        ("file4.txt", 0.6),
        ("file5.txt", 0.95),
    ];

    for (filename, confidence) in &predictions {
        let file_path = temp_dir.path().join(filename);

        println!(
            "\nPredicting: touch {} will succeed (confidence: {})",
            filename, confidence
        );

        // Execute touch command
        let resolver = ExitCodeResolver::new("touch", vec![0])
            .with_args(vec![file_path.to_string_lossy().to_string()]);

        let result = resolver.execute(Duration::from_secs(5));
        let success = matches!(result.outcome, OutcomeCategory::Success);

        println!("  Result: {}", if success { "SUCCESS" } else { "FAILURE" });

        // Feed to bridge
        bridge.observe_resolution(*confidence, success);
    }

    // Check final statistics
    let stats = bridge.statistics();

    println!("\n═══ FINAL CALIBRATION STATE ═══");
    println!("Total observations: {}", stats.total_observations);
    println!(
        "Average prediction error: {:?}",
        stats.average_prediction_error
    );
    println!("Average outcome: {:?}", stats.average_outcome);
    println!("Modulation Index: {:?}", stats.modulation_index);
    println!("Coupling Quality: {:?}", stats.coupling_quality);

    // Assertions
    assert_eq!(stats.total_observations, 5, "Should have 5 observations");

    // Average outcome should be high (all succeeded)
    if let Some(avg_outcome) = stats.average_outcome {
        assert!(
            avg_outcome > 0.8,
            "Average outcome should be high (all succeeded)"
        );
    }

    // Average prediction error should be low (predictions were accurate)
    if let Some(avg_error) = stats.average_prediction_error {
        assert!(avg_error < 0.3, "Average prediction error should be low");
    }

    println!("\n✓ CALIBRATION DRIFT TEST PASSED");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 4: THE OVERCONFIDENCE TRAP
// ═══════════════════════════════════════════════════════════════════════════════

/// Test that overconfident wrong predictions increase uncertainty:
/// 1. Predict with HIGH confidence that a command will FAIL
/// 2. The command actually SUCCEEDS
/// 3. This should increase prediction error (we were wrong AND confident)
#[test]
fn test_live_fire_overconfidence_penalty() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let test_file = temp_dir.path().join("overconfidence_test.txt");
    let mut bridge = ActiveInferenceBridge::new(ActiveInferenceBridgeConfig {
        min_data_points: 1,
        ..Default::default()
    });

    println!("\n═══ OVERCONFIDENCE TRAP TEST ═══");

    // Deliberately predict WRONG with HIGH confidence
    // We predict 'touch' will FAIL with 0.9 confidence
    // But 'touch' always succeeds
    let wrong_confidence = 0.9;

    println!(
        "Predicting: touch will FAIL (confidence in failure: {})",
        wrong_confidence
    );
    println!("(This is deliberately wrong - touch always succeeds)");

    let resolver = ExitCodeResolver::new("touch", vec![0])
        .with_args(vec![test_file.to_string_lossy().to_string()]);

    let result = resolver.execute(Duration::from_secs(5));
    let actually_succeeded = matches!(result.outcome, OutcomeCategory::Success);

    println!(
        "Actual result: {}",
        if actually_succeeded {
            "SUCCESS"
        } else {
            "FAILURE"
        }
    );

    // We predicted failure, so "prediction correct" = !actually_succeeded
    let prediction_correct = !actually_succeeded; // We were WRONG

    bridge.observe_resolution(wrong_confidence, prediction_correct);

    let stats = bridge.statistics();

    println!("\nAfter overconfident wrong prediction:");
    println!(
        "  Prediction correct: {} (we predicted failure, got success)",
        prediction_correct
    );
    println!(
        "  Average prediction error: {:?}",
        stats.average_prediction_error
    );

    // The prediction error should be HIGH because we were wrong with high confidence
    // Brier score for wrong prediction at 0.9 confidence: (0.9 - 0)^2 = 0.81
    if let Some(avg_error) = stats.average_prediction_error {
        assert!(
            avg_error > 0.7,
            "Prediction error should be high after confident wrong prediction"
        );
    }

    println!("\n✓ OVERCONFIDENCE TRAP TEST PASSED");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 5: DIRECTORY OPERATIONS - More Complex Commands
// ═══════════════════════════════════════════════════════════════════════════════

/// Test more complex operations: directory creation and nested files
#[test]
fn test_live_fire_directory_operations() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let subdir = temp_dir.path().join("nested/deep/directory");
    let nested_file = subdir.join("deep_file.txt");
    let mut bridge = ActiveInferenceBridge::new(ActiveInferenceBridgeConfig {
        min_data_points: 1,
        ..Default::default()
    });

    println!("\n═══ DIRECTORY OPERATIONS TEST ═══");

    // Step 1: Create nested directory with mkdir -p
    println!("\nStep 1: mkdir -p {:?}", subdir);

    let mkdir_resolver = ExitCodeResolver::new("mkdir", vec![0])
        .with_args(vec!["-p".to_string(), subdir.to_string_lossy().to_string()]);

    let mkdir_result = mkdir_resolver.execute(Duration::from_secs(5));
    let mkdir_success = matches!(mkdir_result.outcome, OutcomeCategory::Success);

    println!("mkdir result: {:?}", mkdir_result.outcome);
    bridge.observe_resolution(0.85, mkdir_success);

    // Step 2: Create file in nested directory
    println!("\nStep 2: touch {:?}", nested_file);

    let touch_resolver = ExitCodeResolver::new("touch", vec![0])
        .with_args(vec![nested_file.to_string_lossy().to_string()]);

    let touch_result = touch_resolver.execute(Duration::from_secs(5));
    let touch_success = matches!(touch_result.outcome, OutcomeCategory::Success);

    println!("touch result: {:?}", touch_result.outcome);
    bridge.observe_resolution(0.80, touch_success);

    // Step 3: Verify the entire path exists
    println!("\nStep 3: Verify path exists");

    let exists_resolver = ResourceStateResolver::exists(nested_file.to_string_lossy());
    let exists_result = exists_resolver.execute();
    let exists_success = matches!(exists_result.outcome, OutcomeCategory::Success);

    println!("exists result: {:?}", exists_result.outcome);
    bridge.observe_resolution(0.90, exists_success);

    // Final stats
    let stats = bridge.statistics();
    println!("\n═══ FINAL STATE ═══");
    println!("Total observations: {}", stats.total_observations);
    println!("Coupling quality: {:?}", stats.coupling_quality);

    assert!(mkdir_success, "mkdir -p should succeed");
    assert!(touch_success, "touch should succeed");
    assert!(nested_file.exists(), "Nested file should exist");
    assert_eq!(stats.total_observations, 3);

    println!("\n✓ DIRECTORY OPERATIONS TEST PASSED");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 6: THE FULL MAGI LOOP - End to End
// ═══════════════════════════════════════════════════════════════════════════════

/// The complete MAGI loop test:
/// 1. PREDICT: Make a world-grounded prediction
/// 2. RESOLVE: Check against reality via resolvers
/// 3. SELECT: (Implicit - we chose to execute)
/// 4. OBSERVE: Capture the outcome
/// 5. ATTRIBUTE: (Track the domain and confidence)
/// 6. UPDATE: Feed back to calibration and signals
#[test]
fn test_magi_full_loop_closure() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let base_path = temp_dir.path().to_path_buf();
    let mut bridge = ActiveInferenceBridge::new(ActiveInferenceBridgeConfig {
        min_data_points: 3,
        ..Default::default()
    });

    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              THE MAGI LOOP - FULL CLOSURE TEST                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    // Compute all paths upfront to avoid borrow issues
    let marker_path = base_path.join("marker.txt");
    let evidence_path = base_path.join("evidence");
    let evidence_marker_path = evidence_path.join("marker.txt");

    // Test scenarios with pre-computed paths
    let test_scenarios: Vec<(&str, &str, Vec<String>, f64)> = vec![
        (
            "Create marker file",
            "touch",
            vec![marker_path.to_string_lossy().to_string()],
            0.90,
        ),
        (
            "Write to file",
            "sh",
            vec![
                "-c".to_string(),
                format!("echo 'MAGI WAS HERE' > {}", marker_path.display()),
            ],
            0.85,
        ),
        (
            "Verify content",
            "sh",
            vec![
                "-c".to_string(),
                format!("grep 'MAGI' {}", marker_path.display()),
            ],
            0.80,
        ),
        (
            "Create subdirectory",
            "mkdir",
            vec![evidence_path.to_string_lossy().to_string()],
            0.88,
        ),
        (
            "Move file",
            "mv",
            vec![
                marker_path.to_string_lossy().to_string(),
                evidence_path.to_string_lossy().to_string(),
            ],
            0.82,
        ),
    ];

    let mut loop_iterations = 0;
    let mut successful_predictions = 0;

    for (description, command, args, confidence) in &test_scenarios {
        loop_iterations += 1;

        println!("\n┌─────────────────────────────────────────────────────────────────┐");
        println!(
            "│ LOOP ITERATION {} - {}                    ",
            loop_iterations, description
        );
        println!("└─────────────────────────────────────────────────────────────────┘");

        // STEP 1: PREDICT
        println!("\n  [1. PREDICT]");
        println!("      Command: {} {:?}", command, args);
        println!("      Confidence: {}", confidence);

        // STEP 2-4: EXECUTE AND OBSERVE
        println!("\n  [2-4. EXECUTE → OBSERVE]");

        let resolver = ExitCodeResolver::new(*command, vec![0])
            .with_args(args.clone())
            .with_working_dir(base_path.to_string_lossy().to_string());

        let result = resolver.execute(Duration::from_secs(10));
        let success = matches!(result.outcome, OutcomeCategory::Success);

        println!("      Outcome: {:?}", result.outcome);
        if let Some(ref output) = result.raw_output {
            if !output.is_empty() && output.len() < 100 {
                println!("      Output: {}", output.trim());
            }
        }

        // STEP 5: ATTRIBUTE
        println!("\n  [5. ATTRIBUTE]");
        println!("      Prediction correct: {}", success);
        if !success {
            println!("      Reason: {:?}", result.reason);
        }

        // STEP 6: UPDATE
        println!("\n  [6. UPDATE]");
        bridge.observe_resolution(*confidence, success);

        let stats = bridge.statistics();
        println!("      Total observations: {}", stats.total_observations);
        println!("      Coupling quality: {:?}", stats.coupling_quality);

        if success {
            successful_predictions += 1;
        }
    }

    // Final summary
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                      MAGI LOOP SUMMARY                          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let final_stats = bridge.statistics();

    println!("\n  Loop iterations completed: {}", loop_iterations);
    println!(
        "  Successful predictions: {}/{}",
        successful_predictions, loop_iterations
    );
    println!(
        "  Accuracy: {:.1}%",
        (successful_predictions as f64 / loop_iterations as f64) * 100.0
    );
    println!("\n  Bridge Statistics:");
    println!("    Total observations: {}", final_stats.total_observations);
    println!(
        "    Average prediction error: {:?}",
        final_stats.average_prediction_error
    );
    println!("    Average outcome: {:?}", final_stats.average_outcome);
    println!("    Modulation Index: {:?}", final_stats.modulation_index);
    println!("    Coupling Quality: {:?}", final_stats.coupling_quality);

    // Assertions
    assert_eq!(final_stats.total_observations, loop_iterations);
    assert!(
        successful_predictions >= 4,
        "At least 4/5 predictions should succeed"
    );

    // Verify the world state changed
    println!("\n  World State Verification:");
    println!("    evidence/ directory exists: {}", evidence_path.exists());
    println!(
        "    marker.txt moved to evidence/: {}",
        evidence_marker_path.exists()
    );

    // The ultimate assertion: the file we created and moved still exists
    // This proves the loop touched the world AND the world retained the change
    assert!(
        evidence_marker_path.exists(),
        "The moved file must exist - proving world state was modified"
    );

    println!("\n  ╔════════════════════════════════════════════════════════════╗");
    println!("  ║  ✓ MAGI LOOP CLOSURE VERIFIED                              ║");
    println!("  ║  The system predicted, acted, observed, and updated.      ║");
    println!("  ║  The world was touched. The loop is closed.               ║");
    println!("  ╚════════════════════════════════════════════════════════════╝");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 7: SIGNAL GENERATION UNDER LOAD
// ═══════════════════════════════════════════════════════════════════════════════

/// Test that PrincipledSignals change appropriately after many observations
#[test]
fn test_live_fire_signal_evolution() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let mut bridge = ActiveInferenceBridge::new(ActiveInferenceBridgeConfig {
        min_data_points: 5,
        ..Default::default()
    });

    println!("\n═══ SIGNAL EVOLUTION TEST ═══");

    // Generate many observations with high success rate
    for i in 0..20 {
        let file_path = temp_dir.path().join(format!("signal_test_{}.txt", i));

        let resolver = ExitCodeResolver::new("touch", vec![0])
            .with_args(vec![file_path.to_string_lossy().to_string()]);

        let result = resolver.execute(Duration::from_secs(2));
        let success = matches!(result.outcome, OutcomeCategory::Success);

        // High confidence predictions that mostly succeed
        let confidence = 0.85 + (i as f64 * 0.005); // Gradually increasing confidence
        bridge.observe_resolution(confidence, success);
    }

    // Check final statistics
    let final_stats = bridge.statistics();

    println!("\nAfter 20 observations:");
    println!("  Total observations: {}", final_stats.total_observations);
    println!(
        "  Average prediction error: {:?}",
        final_stats.average_prediction_error
    );
    println!("  Average outcome: {:?}", final_stats.average_outcome);
    println!("  Coupling quality: {:?}", final_stats.coupling_quality);

    // With many successful high-confidence predictions:
    // - Prediction error should be LOW
    // - Average outcome should be HIGH
    // - Coupling should be moving toward meaningful

    if let Some(avg_error) = final_stats.average_prediction_error {
        assert!(
            avg_error < 0.2,
            "Prediction error should be low after many successes"
        );
    }

    if let Some(avg_outcome) = final_stats.average_outcome {
        assert!(
            avg_outcome > 0.9,
            "Average outcome should be high (all succeeded)"
        );
    }

    println!("\n✓ SIGNAL EVOLUTION TEST PASSED");
}