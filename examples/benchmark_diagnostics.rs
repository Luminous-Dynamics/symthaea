// Diagnostic script to analyze all benchmark results

use symthaea::benchmarks::{
    CausalBenchmarkSuite, SymthaeaSolver, CausalCategory,
    CompositionalBenchmarkSuite, CompositionalSolver,
    TemporalBenchmarkSuite, TemporalSolver,
    RobustnessBenchmarkSuite, RobustnessSolver,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          Symthaea Causal Benchmark Diagnostics               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let suite = CausalBenchmarkSuite::standard();
    let mut solver = SymthaeaSolver::new();

    let results = suite.run(|benchmark, query| solver.solve(benchmark, query));

    // Print detailed summary
    println!("{}", results.summary());

    // Analyze failures
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    Failure Analysis                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let failures: Vec<_> = results.results.iter().filter(|r| !r.correct).collect();

    if failures.is_empty() {
        println!("No failures! Symthaea achieved 100% accuracy.\n");
    } else {
        println!("Failed {} of {} benchmarks:\n", failures.len(), results.results.len());

        for (i, failure) in failures.iter().enumerate() {
            println!("{}. Benchmark: {}", i + 1, failure.benchmark_id);
            println!("   Category: {:?}", failure.category);
            println!("   Difficulty: {}", failure.difficulty);
            println!("   Query: {}", failure.query);
            println!("   Expected: {}", failure.expected);
            println!("   Got: {}", failure.actual);
            println!();
        }
    }

    // Category breakdown
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║               Accuracy by Category                           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let categories = [
        CausalCategory::CorrelationVsCausation,
        CausalCategory::InterventionPrediction,
        CausalCategory::CounterfactualReasoning,
        CausalCategory::CausalDiscovery,
        CausalCategory::TemporalCausation,
        CausalCategory::ConfoundingControl,
        CausalCategory::NegativeCausation,
    ];

    for category in &categories {
        let acc = results.accuracy_by_category(*category);
        let bar_len = (acc * 20.0) as usize;
        let bar = "█".repeat(bar_len) + &"░".repeat(20 - bar_len);
        println!("  {:25} {} {:.1}%", format!("{:?}:", category), bar, acc * 100.0);
    }

    // Recommendations
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║              Improvement Recommendations                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Identify weakest categories
    let mut category_scores: Vec<_> = categories.iter()
        .map(|c| (*c, results.accuracy_by_category(*c)))
        .collect();
    category_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    println!("Weakest categories (prioritize these for improvement):\n");
    for (category, score) in category_scores.iter().take(3) {
        println!("  • {:?}: {:.1}%", category, score * 100.0);
        match category {
            CausalCategory::CounterfactualReasoning => {
                println!("    → Need structural equation models for what-if queries");
                println!("    → Implement twin network method for counterfactuals");
            }
            CausalCategory::CausalDiscovery => {
                println!("    → Implement PC algorithm for constraint-based discovery");
                println!("    → Add FCI for handling latent confounders");
            }
            CausalCategory::TemporalCausation => {
                println!("    → Leverage LTC's continuous-time dynamics");
                println!("    → Implement Granger causality tests");
            }
            CausalCategory::InterventionPrediction => {
                println!("    → Implement backdoor adjustment formula");
                println!("    → Add frontdoor criterion for indirect effects");
            }
            CausalCategory::ConfoundingControl => {
                println!("    → Implement propensity score matching");
                println!("    → Add instrumental variable estimation");
            }
            _ => {
                println!("    → Strengthen graph traversal algorithms");
            }
        }
        println!();
    }

    // Solver statistics
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                  Solver Statistics                           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let stats = solver.stats();
    println!("  Queries solved: {}", stats.queries_solved);
    println!("  Causal detections: {}", stats.causal_detections);
    println!("  Interventions computed: {}", stats.interventions_computed);
    println!("  Counterfactuals evaluated: {}", stats.counterfactuals_evaluated);

    // ================================================================
    // Compositional Benchmarks
    // ================================================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║        Compositional Generalization Benchmarks               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let comp_suite = CompositionalBenchmarkSuite::standard();
    let mut comp_solver = CompositionalSolver::new();

    let comp_results = comp_suite.run(|benchmark| comp_solver.solve(benchmark));

    println!("{}", comp_results.summary());

    // ================================================================
    // Temporal Benchmarks
    // ================================================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║           Temporal Reasoning Benchmarks                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let temporal_suite = TemporalBenchmarkSuite::standard();
    let mut temporal_solver = TemporalSolver::new();

    let temporal_results = temporal_suite.run(|benchmark| temporal_solver.solve(benchmark));

    println!("{}", temporal_results.summary());

    // ================================================================
    // Robustness Benchmarks
    // ================================================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║           Robustness & Defense Benchmarks                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let robust_suite = RobustnessBenchmarkSuite::standard();
    let mut robust_solver = RobustnessSolver::new();

    let robust_results = robust_suite.run(|benchmark| robust_solver.solve(benchmark));

    println!("{}", robust_results.summary());

    // Overall summary
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    OVERALL SUMMARY                           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let causal_acc = results.accuracy() as f64;
    let comp_acc = comp_results.accuracy() as f64;
    let temporal_acc = temporal_results.accuracy();
    let robust_acc = robust_results.accuracy();
    let overall = (causal_acc + comp_acc + temporal_acc + robust_acc) / 4.0;

    println!("  Causal Reasoning:     {:.1}%", causal_acc * 100.0);
    println!("  Compositional:        {:.1}%", comp_acc * 100.0);
    println!("  Temporal Reasoning:   {:.1}%", temporal_acc * 100.0);
    println!("  Robustness:           {:.1}%", robust_acc * 100.0);
    println!("  ────────────────────────────");
    println!("  Overall Average:      {:.1}%", overall * 100.0);
    println!();

    if overall >= 0.9 {
        println!("  Status: EXCELLENT - Symthaea demonstrates strong AI capabilities!");
    } else if overall >= 0.7 {
        println!("  Status: GOOD - Solid performance with room for improvement");
    } else {
        println!("  Status: NEEDS WORK - Focus on failing categories");
    }
}
