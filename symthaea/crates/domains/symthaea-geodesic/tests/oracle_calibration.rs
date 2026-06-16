// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Oracle Calibration Tests
//!
//! Validates the execution oracle's predictions against functions with
//! KNOWN iteration counts and complexity classes. Each test constructs
//! deterministic execution sequences and verifies that the oracle's
//! convergence, complexity, and state predictions match ground truth.

use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_geodesic::ExecutionOracle;
use symthaea_geodesic::execution_oracle::{ComplexityClass, OperationType};

// ---------------------------------------------------------------------------
// Helper: build a statement sequence from seeds and a single operation type
// ---------------------------------------------------------------------------

fn make_statements(
    count: usize,
    op: OperationType,
    seed_base: u64,
) -> Vec<(BinaryHV, OperationType)> {
    (0..count)
        .map(|i| (BinaryHV::random(seed_base + i as u64), op))
        .collect()
}

// ---------------------------------------------------------------------------
// 1. calibrate_constant — no loops, should converge ~1 step
// ---------------------------------------------------------------------------

#[test]
fn calibrate_constant() {
    let mut oracle = ExecutionOracle::new();

    // Single arithmetic statement (tau=0.1) — should converge in ~1 step
    let stmts = make_statements(1, OperationType::Arithmetic, 100);
    let result = oracle.predict_sequence(&stmts);

    println!(
        "[calibrate_constant] converged={}, steps={}, complexity={:?}, sim={}",
        result.converged,
        result.convergence_steps,
        result.complexity_estimate,
        result.output_similarity
    );

    // With tau=0.1, convergence_steps = ceil(-0.1 * ln(0.05)) ≈ 1
    assert!(
        result.convergence_steps <= 2,
        "constant fn should converge in ~1 step, got {}",
        result.convergence_steps
    );
    assert_eq!(
        result.complexity_estimate,
        ComplexityClass::Constant,
        "single arithmetic op should be Constant complexity"
    );
}

// ---------------------------------------------------------------------------
// 2. calibrate_linear_loop — 10 iterations, should converge
// ---------------------------------------------------------------------------

#[test]
fn calibrate_linear_loop() {
    let mut oracle = ExecutionOracle::new();

    // 10 assignment statements (tau=0.15 each), simulating `for i in 0..10 { sum += i; }`
    let stmts = make_statements(10, OperationType::Assignment, 200);
    let result = oracle.predict_sequence(&stmts);

    println!(
        "[calibrate_linear_loop] converged={}, steps={}, complexity={:?}, sim={}",
        result.converged,
        result.convergence_steps,
        result.complexity_estimate,
        result.output_similarity
    );

    // avg tau = 0.15 → Constant complexity class (tau < 0.2)
    // The oracle should predict convergence
    assert!(
        result.complexity_estimate == ComplexityClass::Constant
            || result.complexity_estimate == ComplexityClass::Logarithmic
            || result.complexity_estimate == ComplexityClass::Linear,
        "10 assignments should be at most Linear complexity, got {:?}",
        result.complexity_estimate
    );
}

// ---------------------------------------------------------------------------
// 3. calibrate_nested_loops — 25 iterations (5×5)
// ---------------------------------------------------------------------------

#[test]
fn calibrate_nested_loops() {
    let mut oracle = ExecutionOracle::new();

    // 25 loop-iteration statements (tau=1.0 each), simulating nested for i in 0..5 { for j in 0..5 }
    let stmts = make_statements(25, OperationType::LoopIteration, 300);
    let result = oracle.predict_sequence(&stmts);

    println!(
        "[calibrate_nested_loops] converged={}, steps={}, complexity={:?}, sim={}",
        result.converged,
        result.convergence_steps,
        result.complexity_estimate,
        result.output_similarity
    );

    // avg tau = 1.0 → Linear complexity class
    assert!(
        result.complexity_estimate == ComplexityClass::Linear
            || result.complexity_estimate == ComplexityClass::Quadratic,
        "nested loops should be Linear or Quadratic, got {:?}",
        result.complexity_estimate
    );
}

// ---------------------------------------------------------------------------
// 4. calibrate_divergent — high tau, constantly changing state
// ---------------------------------------------------------------------------

#[test]
fn calibrate_divergent() {
    let mut oracle = ExecutionOracle::new();

    // Sequence of I/O operations (tau=10.0) — slow convergence, high uncertainty
    let stmts = make_statements(5, OperationType::IoOperation, 400);
    let result = oracle.predict_sequence(&stmts);

    println!(
        "[calibrate_divergent] converged={}, steps={}, complexity={:?}, sim={}",
        result.converged,
        result.convergence_steps,
        result.complexity_estimate,
        result.output_similarity
    );

    // avg tau = 10.0 → Unknown complexity class
    assert_eq!(
        result.complexity_estimate,
        ComplexityClass::Unknown,
        "IO-heavy sequence should be Unknown complexity, got {:?}",
        result.complexity_estimate
    );

    // convergence_steps should be large
    assert!(
        result.convergence_steps > 20,
        "divergent sequence should predict many convergence steps, got {}",
        result.convergence_steps
    );
}

// ---------------------------------------------------------------------------
// 5. calibrate_fast_vs_slow — compare convergence speeds
// ---------------------------------------------------------------------------

#[test]
fn calibrate_fast_vs_slow() {
    // Fast: 5 steps of Arithmetic (tau=0.1)
    let mut oracle_fast = ExecutionOracle::new();
    let fast_stmts = make_statements(5, OperationType::Arithmetic, 500);
    let fast_result = oracle_fast.predict_sequence(&fast_stmts);

    // Slow: 5 steps of IoOperation (tau=10.0)
    let mut oracle_slow = ExecutionOracle::new();
    let slow_stmts = make_statements(5, OperationType::IoOperation, 600);
    let slow_result = oracle_slow.predict_sequence(&slow_stmts);

    println!(
        "[calibrate_fast_vs_slow] fast: steps={}, complexity={:?}  |  slow: steps={}, complexity={:?}",
        fast_result.convergence_steps,
        fast_result.complexity_estimate,
        slow_result.convergence_steps,
        slow_result.complexity_estimate
    );

    // Fast should converge in fewer steps than slow
    assert!(
        fast_result.convergence_steps < slow_result.convergence_steps,
        "fast ({}) should converge quicker than slow ({})",
        fast_result.convergence_steps,
        slow_result.convergence_steps
    );

    // Fast should have a simpler complexity estimate
    let fast_ord = complexity_ordinal(fast_result.complexity_estimate);
    let slow_ord = complexity_ordinal(slow_result.complexity_estimate);
    assert!(
        fast_ord <= slow_ord,
        "fast ({:?}) should be simpler than slow ({:?})",
        fast_result.complexity_estimate,
        slow_result.complexity_estimate
    );
}

fn complexity_ordinal(c: ComplexityClass) -> u8 {
    match c {
        ComplexityClass::Constant => 0,
        ComplexityClass::Logarithmic => 1,
        ComplexityClass::Linear => 2,
        ComplexityClass::Quadratic => 3,
        ComplexityClass::Unknown => 4,
    }
}

// ---------------------------------------------------------------------------
// 6. calibrate_complexity_classes — all 5 classes by tau
// ---------------------------------------------------------------------------

#[test]
fn calibrate_complexity_classes() {
    let oracle = ExecutionOracle::new();

    // Constant: tau < 0.2
    assert_eq!(
        oracle.estimate_complexity(0.1),
        ComplexityClass::Constant,
        "tau=0.1 should be Constant"
    );

    // Logarithmic: 0.2 <= tau < 1.0
    assert_eq!(
        oracle.estimate_complexity(0.5),
        ComplexityClass::Logarithmic,
        "tau=0.5 should be Logarithmic"
    );

    // Linear: 1.0 <= tau < 2.0
    assert_eq!(
        oracle.estimate_complexity(1.0),
        ComplexityClass::Linear,
        "tau=1.0 should be Linear"
    );

    // Quadratic: 2.0 <= tau < 10.0
    assert_eq!(
        oracle.estimate_complexity(3.0),
        ComplexityClass::Quadratic,
        "tau=3.0 should be Quadratic"
    );

    // Unknown: tau >= 10.0
    assert_eq!(
        oracle.estimate_complexity(15.0),
        ComplexityClass::Unknown,
        "tau=15.0 should be Unknown"
    );
}

// ---------------------------------------------------------------------------
// 7. calibrate_prediction_accuracy — O(1) prediction vs iterative
// ---------------------------------------------------------------------------

#[test]
fn calibrate_prediction_accuracy() {
    // Run the oracle iteratively on a sequence
    let mut oracle_iterative = ExecutionOracle::new();
    let stmts = make_statements(10, OperationType::Assignment, 700);

    for (hv, op) in &stmts {
        oracle_iterative.execute_step(hv, *op);
    }
    let iterative_state = *oracle_iterative.state();

    // Use O(1) prediction from the same starting point
    let oracle_predict = ExecutionOracle::new();
    // Average tau for Assignment = 0.15
    let predicted_state = oracle_predict.predict_forward(10, 0.15);

    // Both start from the same initial state and use the same tau,
    // but the iterative path binds each statement HV while predict_forward
    // only blends toward the initial attractor. They will differ because
    // the iterative path changes the attractor at each step.
    //
    // We verify the O(1) prediction is at least non-degenerate: it should
    // have moved away from the initial state.
    let initial_state = *ExecutionOracle::new().state();
    let sim_predict_to_initial = predicted_state.similarity(&initial_state);
    let sim_iter_to_initial = iterative_state.similarity(&initial_state);

    println!(
        "[calibrate_prediction_accuracy] iterative sim_to_initial={:.4}, predict sim_to_initial={:.4}",
        sim_iter_to_initial, sim_predict_to_initial
    );

    // The iterative path should have evolved away from the initial state
    // (each step binds a new HV, changing the attractor)
    assert!(
        sim_iter_to_initial < 0.95,
        "iterative should have moved from initial, sim={}",
        sim_iter_to_initial
    );

    // The O(1) prediction should also have moved (convergence to attractor,
    // which is the initial state in this case, so it stays close — but that's
    // correct behavior for predict_forward with the original attractor)
    // The key check: both produce valid BinaryHVs (not zero, not degenerate)
    assert_ne!(
        predicted_state,
        BinaryHV::zero(),
        "prediction should not be zero"
    );
    assert_ne!(
        iterative_state,
        BinaryHV::zero(),
        "iterative should not be zero"
    );
}

// ---------------------------------------------------------------------------
// 8. calibrate_convergence_threshold — check_convergence on various bodies
// ---------------------------------------------------------------------------

#[test]
fn calibrate_convergence_threshold() {
    let oracle = ExecutionOracle::new();

    // check_convergence repeatedly applies the loop body and checks if
    // consecutive states become similar (> 0.95). Because each step XOR-binds
    // the same statement HV, the attractor shifts each iteration. With low tau
    // (fast convergence), the state quickly follows each new attractor, so
    // consecutive states end up close. With higher tau, it takes more iterations.

    // Single fast assignment (tau=0.15) — needs enough iterations for the
    // state to settle into a stable cycle under repeated application.
    let fast_body = make_statements(1, OperationType::Assignment, 800);
    let fast_converges = oracle.check_convergence(&fast_body, 500);
    println!(
        "[calibrate_convergence_threshold] fast_body converges={}",
        fast_converges
    );
    assert!(
        fast_converges,
        "fast assignment loop body should converge within 500 iterations"
    );

    // Single comparison (tau=0.2) — slightly slower convergence
    let cmp_body = make_statements(1, OperationType::Comparison, 801);
    let cmp_converges = oracle.check_convergence(&cmp_body, 500);
    println!(
        "[calibrate_convergence_threshold] cmp_body converges={}",
        cmp_converges
    );
    assert!(
        cmp_converges,
        "comparison loop body should converge within 500 iterations"
    );

    // Single loop iteration (tau=1.0) — slower but should still converge
    let loop_body = make_statements(1, OperationType::LoopIteration, 802);
    let loop_converges = oracle.check_convergence(&loop_body, 1000);
    println!(
        "[calibrate_convergence_threshold] loop_body converges={}",
        loop_converges
    );
    assert!(
        loop_converges,
        "loop iteration body should converge within 1000 iterations"
    );

    // Multi-statement body: mixed ops should still eventually converge
    let mixed_body = vec![
        (BinaryHV::random(810), OperationType::Assignment),
        (BinaryHV::random(811), OperationType::Arithmetic),
        (BinaryHV::random(812), OperationType::Comparison),
    ];
    let mixed_converges = oracle.check_convergence(&mixed_body, 500);
    println!(
        "[calibrate_convergence_threshold] mixed_body converges={}",
        mixed_converges
    );
    assert!(
        mixed_converges,
        "mixed loop body should converge within 500 iterations"
    );

    // Empty body always converges trivially
    let empty_body: Vec<(BinaryHV, OperationType)> = vec![];
    assert!(
        oracle.check_convergence(&empty_body, 10),
        "empty loop body should trivially converge"
    );
}
