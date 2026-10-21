use std::error::Error;

use symthaea_core::hdc::pattern_interaction::{
    run_pattern_interaction_suite, PatternInteractionConfig,
};

fn main() -> Result<(), Box<dyn Error>> {
    let config = PatternInteractionConfig::default();
    let report = run_pattern_interaction_suite(0x4c45_5649_4e, &config)?;

    println!("experiment,seed,dimension,prototypes,baseline");
    println!(
        "pattern_interaction,{},{},{},{}",
        report.seed, report.dimension, report.prototype_count, report.random_chance_baseline
    );
    println!();
    println!("metric,value");
    println!("role_unbinding_similarity,{:.6}", report.role_unbinding_similarity);
    println!("bundling_ablation_similarity,{:.6}", report.bundling_ablation_similarity);
    println!("binding_ablation_advantage,{:.6}", report.binding_ablation_advantage);
    println!(
        "antipattern_cancellation_abs_similarity,{:.6}",
        report.antipattern_cancellation_abs_similarity
    );
    println!("sequence_order_similarity,{:.6}", report.sequence_order_similarity);
    println!("basis_equivariance_exact,{}", report.basis_equivariance_exact);

    println!();
    println!("noise_fraction,nearest_top1,nearest_margin,cleanup_top1,cleanup_margin");
    for result in &report.noise_results {
        println!(
            "{:.3},{:.6},{:.6},{:.6},{:.6}",
            result.noise_fraction,
            result.top1_accuracy,
            result.mean_similarity_margin,
            result.cleanup_top1_accuracy,
            result.cleanup_mean_similarity_margin
        );
    }

    println!();
    println!("bundle_size,mean_member_similarity,mean_max_distractor_similarity,member_recall_at_1");
    for result in &report.bundle_results {
        println!(
            "{},{:.6},{:.6},{:.6}",
            result.bundle_size,
            result.mean_member_similarity,
            result.mean_max_distractor_similarity,
            result.member_recall_at_1
        );
    }

    println!();
    println!("associations,target_similarity,max_distractor,recall_at_1,cleanup_target_similarity,cleanup_max_distractor,cleanup_recall_at_1");
    for result in &report.associative_capacity_results {
        println!(
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            result.association_count,
            result.mean_target_similarity,
            result.mean_max_distractor_similarity,
            result.recall_at_1,
            result.cleanup_mean_target_similarity,
            result.cleanup_mean_max_distractor_similarity,
            result.cleanup_recall_at_1
        );
    }

    Ok(())
}
