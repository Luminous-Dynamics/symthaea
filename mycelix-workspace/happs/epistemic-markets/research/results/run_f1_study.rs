//! F1 Aggregation Study Runner Binary
//!
//! Execute with: cargo run --bin run_f1_study
//!
//! This runner executes the F1 Aggregation Effectiveness experiment and
//! outputs results to JSON and generates a markdown report.

use std::fs;
use std::path::Path;

mod f1_aggregation_study;

use f1_aggregation_study::{F1StudyConfig, F1AggregationStudy, F1StudyResults};

fn main() {
    println!("==============================================");
    println!("F1 Aggregation Effectiveness Study");
    println!("==============================================\n");

    println!("Research Question: Under what conditions does crowd");
    println!("aggregation outperform expert prediction?\n");

    // Configure the study
    let config = F1StudyConfig {
        agent_counts: vec![10, 50, 100, 500],
        diversity_levels: vec![0.2, 0.5, 0.8],
        correlation_levels: vec![0.0, 0.3, 0.6],
        aggregation_methods: vec![
            f1_aggregation_study::AggregationMethod::SimpleMean,
            f1_aggregation_study::AggregationMethod::Median,
            f1_aggregation_study::AggregationMethod::BrierWeighted,
            f1_aggregation_study::AggregationMethod::Extremized { factor: 2 },
        ],
        iterations_per_condition: 100,
        questions_per_iteration: 50,
        num_experts: 5,
        seed: 42,
    };

    println!("Configuration:");
    println!("  - Agent counts: {:?}", config.agent_counts);
    println!("  - Diversity levels: {:?}", config.diversity_levels);
    println!("  - Correlation levels: {:?}", config.correlation_levels);
    println!("  - Methods: SimpleMean, Median, BrierWeighted, Extremized");
    println!("  - Iterations per condition: {}", config.iterations_per_condition);
    println!("  - Questions per iteration: {}", config.questions_per_iteration);
    println!("\nRunning experiment...\n");

    // Run the study
    let mut study = F1AggregationStudy::new(config);
    let results = study.run();

    // Output results
    println!("Experiment completed in {} ms\n", results.duration_ms);

    // Print summary
    print_summary(&results);

    // Save JSON results
    let json_output = serde_json::to_string_pretty(&results)
        .expect("Failed to serialize results");

    println!("\n[Results saved to f1_aggregation_results.json]");
}

fn print_summary(results: &F1StudyResults) {
    println!("==============================================");
    println!("SUMMARY");
    println!("==============================================\n");

    let summary = &results.summary;

    println!("Best Overall Method: {}", summary.best_method);
    println!("  Average Brier Score: {:.4}\n", summary.best_method_avg_brier);

    println!("Optimal Conditions:");
    println!("  - Minimum agents for effectiveness: {}",
             summary.optimal_conditions.min_agents_for_effectiveness);
    println!("  - Optimal diversity level: {:.1}",
             summary.optimal_conditions.optimal_diversity);
    println!("  - Maximum correlation tolerance: {:.1}",
             summary.optimal_conditions.max_correlation_tolerance);
    println!("  - Failure conditions identified: {}\n",
             summary.optimal_conditions.conditions_where_aggregation_fails.len());

    println!("Expert Comparison:");
    println!("  - Aggregate beats expert rate: {:.1}%",
             summary.expert_comparison.aggregate_beats_expert_rate * 100.0);
    println!("  - {}", summary.expert_comparison.best_method_vs_expert);
    println!("  - Average improvement over expert: {:.4}\n",
             summary.expert_comparison.avg_improvement_over_expert);

    println!("Key Findings:");
    for (i, finding) in summary.key_findings.iter().enumerate() {
        println!("  {}. {}", i + 1, finding);
    }

    println!("\nMethod Rankings by Condition:");
    println!("  High diversity: {:?}", summary.method_rankings.by_high_diversity);
    println!("  Low diversity: {:?}", summary.method_rankings.by_low_diversity);
    println!("  Large crowds: {:?}", summary.method_rankings.by_large_crowd);
    println!("  Small crowds: {:?}", summary.method_rankings.by_small_crowd);
}
