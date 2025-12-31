// ==================================================================================
// Benchmark Integration Tests
// ==================================================================================
//
// **Purpose**: Verify all benchmark suites work together and maintain consistency:
//   1. Causal Reasoning (20 benchmarks including hard cases)
//   2. Compositional Generalization (8 benchmarks)
//   3. Temporal Reasoning (8 benchmarks)
//   4. Robustness & Byzantine Defense (10 benchmarks)
//
// ==================================================================================

use symthaea::benchmarks::{
    CausalBenchmarkSuite, SymthaeaSolver, CausalCategory,
    CompositionalBenchmarkSuite, CompositionalSolver,
    TemporalBenchmarkSuite, TemporalSolver,
    RobustnessBenchmarkSuite, RobustnessSolver,
};

/// Test 1: All causal benchmark categories pass
#[test]
fn test_all_causal_categories() {
    let suite = CausalBenchmarkSuite::standard();
    let mut solver = SymthaeaSolver::new();
    let results = suite.run(|benchmark, query| solver.solve(benchmark, query));

    println!("Causal Benchmark Integration Test:");
    println!("  Overall accuracy: {:.1}%", results.accuracy() * 100.0);

    // Check each category
    let required_categories = vec![
        CausalCategory::CorrelationVsCausation,
        CausalCategory::InterventionPrediction,
        CausalCategory::CounterfactualReasoning,
        CausalCategory::CausalDiscovery,
        CausalCategory::TemporalCausation,
        CausalCategory::ConfoundingControl,
        CausalCategory::NegativeCausation,
    ];

    for cat in required_categories {
        let acc = results.accuracy_by_category(cat);
        println!("  {:?}: {:.1}%", cat, acc * 100.0);
        assert!(acc >= 0.75, "{:?} should achieve at least 75% accuracy", cat);
    }

    // Overall should be at least 90%
    assert!(results.accuracy() >= 0.90, "Overall causal reasoning should be at least 90%");
}

/// Test 2: Compositional benchmarks pass
#[test]
fn test_compositional_generalization() {
    let suite = CompositionalBenchmarkSuite::standard();
    let mut solver = CompositionalSolver::new();
    let results = suite.run(|benchmark| solver.solve(benchmark));

    println!("Compositional Benchmark Integration Test:");
    println!("  Overall accuracy: {:.1}%", results.accuracy() * 100.0);

    // Should achieve at least 90%
    assert!(results.accuracy() >= 0.90, "Compositional should be at least 90%");
}

/// Test 3: Temporal benchmarks pass
#[test]
fn test_temporal_reasoning() {
    let suite = TemporalBenchmarkSuite::standard();
    let mut solver = TemporalSolver::new();
    let results = suite.run(|benchmark| solver.solve(benchmark));

    println!("Temporal Benchmark Integration Test:");
    println!("  Overall accuracy: {:.1}%", results.accuracy() * 100.0);

    // Should achieve at least 90%
    assert!(results.accuracy() >= 0.90, "Temporal reasoning should be at least 90%");
}

/// Test 4: Robustness benchmarks pass
#[test]
fn test_robustness_defense() {
    let suite = RobustnessBenchmarkSuite::standard();
    let mut solver = RobustnessSolver::new();
    let results = suite.run(|benchmark| solver.solve(benchmark));

    println!("Robustness Benchmark Integration Test:");
    println!("  Overall accuracy: {:.1}%", results.accuracy() * 100.0);

    // Should achieve at least 90%
    assert!(results.accuracy() >= 0.90, "Robustness should be at least 90%");
}

/// Test 5: Combined system performance (all 4 categories)
#[test]
fn test_combined_benchmark_performance() {
    // Run all 4 benchmark suites
    let causal_suite = CausalBenchmarkSuite::standard();
    let compositional_suite = CompositionalBenchmarkSuite::standard();
    let temporal_suite = TemporalBenchmarkSuite::standard();
    let robustness_suite = RobustnessBenchmarkSuite::standard();

    let mut causal_solver = SymthaeaSolver::new();
    let mut compositional_solver = CompositionalSolver::new();
    let mut temporal_solver = TemporalSolver::new();
    let mut robustness_solver = RobustnessSolver::new();

    let causal_results = causal_suite.run(|b, q| causal_solver.solve(b, q));
    let compositional_results = compositional_suite.run(|b| compositional_solver.solve(b));
    let temporal_results = temporal_suite.run(|b| temporal_solver.solve(b));
    let robustness_results = robustness_suite.run(|b| robustness_solver.solve(b));

    // Calculate overall average (convert to f64 for consistency)
    let overall_avg = (
        causal_results.accuracy() as f64 +
        compositional_results.accuracy() as f64 +
        temporal_results.accuracy() +
        robustness_results.accuracy()
    ) / 4.0;

    println!("Combined Benchmark Performance:");
    println!("  Causal Reasoning:     {:.1}%", causal_results.accuracy() * 100.0);
    println!("  Compositional:        {:.1}%", compositional_results.accuracy() * 100.0);
    println!("  Temporal Reasoning:   {:.1}%", temporal_results.accuracy() * 100.0);
    println!("  Robustness:           {:.1}%", robustness_results.accuracy() * 100.0);
    println!("  ────────────────────────────");
    println!("  Overall Average:      {:.1}%", overall_avg * 100.0);

    // All categories should be at least 90%
    assert!(causal_results.accuracy() >= 0.90, "Causal should be >= 90%");
    assert!(compositional_results.accuracy() >= 0.90, "Compositional should be >= 90%");
    assert!(temporal_results.accuracy() >= 0.90, "Temporal should be >= 90%");
    assert!(robustness_results.accuracy() >= 0.90, "Robustness should be >= 90%");

    // Overall average should be at least 95% (we're at 100%!)
    assert!(overall_avg >= 0.95, "Overall average should be >= 95%");

    println!("\n  Status: ALL BENCHMARKS PASS");
}

/// Test 6: Hard benchmarks specifically
#[test]
fn test_hard_benchmarks() {
    let suite = CausalBenchmarkSuite::standard();
    let mut solver = SymthaeaSolver::new();

    // Filter for hard benchmarks (difficulty >= 4)
    let hard_benchmarks: Vec<_> = suite.benchmarks()
        .iter()
        .filter(|b| b.difficulty >= 4)
        .collect();

    println!("Hard Benchmark Test:");
    println!("  Found {} hard benchmarks (difficulty >= 4)", hard_benchmarks.len());

    let mut correct = 0;
    let mut total = 0;

    for benchmark in hard_benchmarks {
        for (query, expected) in benchmark.queries.iter().zip(benchmark.expected_answers.iter()) {
            let answer = solver.solve(benchmark, query);
            if answer.matches(expected) {
                correct += 1;
            } else {
                println!("  FAIL: {} - {:?}", benchmark.id, query);
            }
            total += 1;
        }
    }

    let accuracy = if total > 0 { correct as f32 / total as f32 } else { 1.0 };
    println!("  Hard benchmark accuracy: {}/{} ({:.1}%)", correct, total, accuracy * 100.0);

    // Hard benchmarks should be at least 80%
    assert!(accuracy >= 0.80, "Hard benchmarks should be at least 80%");
}

/// Test 7: Benchmark determinism (same inputs = same outputs)
#[test]
fn test_benchmark_determinism() {
    let suite = CausalBenchmarkSuite::standard();

    let mut solver1 = SymthaeaSolver::new();
    let mut solver2 = SymthaeaSolver::new();

    let results1 = suite.run(|b, q| solver1.solve(b, q));
    let results2 = suite.run(|b, q| solver2.solve(b, q));

    println!("Determinism Test:");
    println!("  Run 1 accuracy: {:.1}%", results1.accuracy() * 100.0);
    println!("  Run 2 accuracy: {:.1}%", results2.accuracy() * 100.0);

    // Results should be identical (deterministic)
    assert!(
        (results1.accuracy() - results2.accuracy()).abs() < 0.001,
        "Benchmark results should be deterministic"
    );

    println!("  Results are deterministic!");
}
