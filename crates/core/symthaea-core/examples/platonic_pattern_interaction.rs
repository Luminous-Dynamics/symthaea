use std::error::Error;
use std::mem::size_of;
use std::time::Instant;

use symthaea_core::hdc::hrr_baseline::run_hrr_associative_capacity;
use symthaea_core::hdc::pattern_interaction::{
    run_pattern_interaction_suite, PatternInteractionConfig,
};

fn k95(
    observations: impl IntoIterator<Item = (usize, f64)>,
) -> Option<usize> {
    let mut observations: Vec<(usize, f64)> = observations.into_iter().collect();
    observations.sort_by_key(|(capacity, _)| *capacity);

    let mut last_qualified = None;
    for (capacity, lower_recall_bound) in observations {
        // K95 is a prefix capacity target: once a tested capacity fails, a
        // later noisy point must not "requalify" the larger memory.
        if lower_recall_bound < 0.95 {
            break;
        }
        last_qualified = Some(capacity);
    }
    last_qualified
}

fn format_k95(value: Option<usize>) -> String {
    value.map_or_else(|| "NA".to_owned(), |capacity| capacity.to_string())
}

fn main() -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let stress_mode = std::env::args().skip(1).any(|arg| arg == "--stress");
    let config = if stress_mode {
        PatternInteractionConfig {
            dimension: 4_096,
            prototype_count: 256,
            trials: 128,
            noise_fractions: vec![0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
            bundle_sizes: vec![1, 7, 31, 63, 127, 255],
        }
    } else {
        PatternInteractionConfig::default()
    };
    let report = run_pattern_interaction_suite(0x4c45_5649_4e, &config)?;
    let hrr_report = run_hrr_associative_capacity(
        0x4c45_5649_4e,
        config.dimension,
        config.prototype_count,
        config.trials,
        &config.bundle_sizes,
    )?;
    let matched_bytes_hrr_dimension = (config.dimension / 8).max(1);
    let matched_bytes_hrr_report = run_hrr_associative_capacity(
        0x4c45_5649_4e,
        matched_bytes_hrr_dimension,
        config.prototype_count,
        config.trials,
        &config.bundle_sizes,
    )?;
    let elapsed_ms = started.elapsed().as_millis();
    let estimated_prototype_payload_bytes =
        report.dimension * report.prototype_count * size_of::<i8>();
    let estimated_hrr_candidate_payload_bytes =
        hrr_report.dimension * hrr_report.candidate_count * size_of::<f64>();
    let estimated_matched_bytes_hrr_payload_bytes =
        matched_bytes_hrr_report.dimension
            * matched_bytes_hrr_report.candidate_count
            * size_of::<f64>();
    let map_raw_k95 = k95(
        report
            .associative_capacity_results
            .iter()
            .map(|result| (result.association_count, result.recall_at_1_ci95.0)),
    );
    let map_cleanup_k95 = k95(
        report
            .associative_capacity_results
            .iter()
            .map(|result| (result.association_count, result.cleanup_recall_at_1_ci95.0)),
    );
    let hrr_same_dimension_k95 = k95(
        config
            .bundle_sizes
            .iter()
            .copied()
            .zip(&hrr_report.results)
            .map(|(capacity, result)| (capacity, result.raw_recall_ci95.0)),
    );
    let hrr_matched_bytes_k95 = k95(
        config
            .bundle_sizes
            .iter()
            .copied()
            .zip(&matched_bytes_hrr_report.results)
            .map(|(capacity, result)| (capacity, result.raw_recall_ci95.0)),
    );

    println!("experiment,mode,seed,dimension,prototypes,trials,random_baseline");
    println!(
        "pattern_interaction,{},{},{},{},{},{}",
        if stress_mode { "stress" } else { "smoke" },
        report.seed,
        report.dimension,
        report.prototype_count,
        config.trials,
        report.random_chance_baseline
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
    println!("run_elapsed_ms,{elapsed_ms}");
    println!("pattern_storage_encoding,bipolar_i8");
    println!("estimated_map_candidate_payload_bytes,{estimated_prototype_payload_bytes}");
    println!("hrr_storage_encoding,f64");
    println!("estimated_hrr_same_dimension_payload_bytes,{estimated_hrr_candidate_payload_bytes}");
    println!("matched_bytes_hrr_dimension,{matched_bytes_hrr_dimension}");
    println!("estimated_matched_bytes_hrr_payload_bytes,{estimated_matched_bytes_hrr_payload_bytes}");
    println!("storage_estimate_excludes,Vec metadata and transient working allocations");
    println!("capacity_comparisons,same_dimension_and_matched_payload_bytes");
    println!("k95_criterion,lower_95_percent_recall_bound_at_least_0.95");
    println!("k95_tested_grid_only,true");
    println!("k95_metric,mode,dimension,prototypes,trials,estimated_candidate_payload_bytes,max_tested_k95");
    println!(
        "max_tested_k95,map_raw,{},{},{},{},{}",
        report.dimension,
        report.prototype_count,
        config.trials,
        estimated_prototype_payload_bytes,
        format_k95(map_raw_k95)
    );
    println!(
        "max_tested_k95,map_cleanup,{},{},{},{},{}",
        report.dimension,
        report.prototype_count,
        config.trials,
        estimated_prototype_payload_bytes,
        format_k95(map_cleanup_k95)
    );
    println!(
        "max_tested_k95,hrr_same_dimension,{},{},{},{},{}",
        hrr_report.dimension,
        hrr_report.candidate_count,
        config.trials,
        estimated_hrr_candidate_payload_bytes,
        format_k95(hrr_same_dimension_k95)
    );
    println!(
        "max_tested_k95,hrr_matched_bytes,{},{},{},{},{}",
        matched_bytes_hrr_report.dimension,
        matched_bytes_hrr_report.candidate_count,
        config.trials,
        estimated_matched_bytes_hrr_payload_bytes,
        format_k95(hrr_matched_bytes_k95)
    );

    println!();
    println!(
        "noise_fraction,nearest_top1,nearest_ci_low,nearest_ci_high,nearest_margin,cleanup_top1,cleanup_ci_low,cleanup_ci_high,cleanup_margin"
    );
    for result in &report.noise_results {
        println!(
            "{:.3},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            result.noise_fraction,
            result.top1_accuracy,
            result.top1_accuracy_ci95.0,
            result.top1_accuracy_ci95.1,
            result.mean_similarity_margin,
            result.cleanup_top1_accuracy,
            result.cleanup_top1_accuracy_ci95.0,
            result.cleanup_top1_accuracy_ci95.1,
            result.cleanup_mean_similarity_margin
        );
    }

    println!();
    println!(
        "bundle_size,mean_member_similarity,mean_max_distractor_similarity,member_recall_at_1,recall_ci_low,recall_ci_high"
    );
    for result in &report.bundle_results {
        println!(
            "{},{:.6},{:.6},{:.6},{:.6},{:.6}",
            result.bundle_size,
            result.mean_member_similarity,
            result.mean_max_distractor_similarity,
            result.member_recall_at_1,
            result.member_recall_at_1_ci95.0,
            result.member_recall_at_1_ci95.1
        );
    }

    println!();
    println!(
        "associations,map_raw_recall,map_ci_low,map_ci_high,map_cleanup_recall,map_cleanup_ci_low,map_cleanup_ci_high,hrr_same_dim_recall,hrr_same_dim_ci_low,hrr_same_dim_ci_high,hrr_same_dim_target_sim,hrr_same_dim_distractor_sim,hrr_matched_bytes_recall,hrr_matched_bytes_ci_low,hrr_matched_bytes_ci_high,hrr_matched_bytes_target_sim,hrr_matched_bytes_distractor_sim"
    );
    for ((map_result, hrr_result), matched_bytes_hrr_result) in report
        .associative_capacity_results
        .iter()
        .zip(&hrr_report.results)
        .zip(&matched_bytes_hrr_report.results)
    {
        println!(
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            map_result.association_count,
            map_result.recall_at_1,
            map_result.recall_at_1_ci95.0,
            map_result.recall_at_1_ci95.1,
            map_result.cleanup_recall_at_1,
            map_result.cleanup_recall_at_1_ci95.0,
            map_result.cleanup_recall_at_1_ci95.1,
            hrr_result.raw_recall_at_1,
            hrr_result.raw_recall_ci95.0,
            hrr_result.raw_recall_ci95.1,
            hrr_result.mean_target_similarity,
            hrr_result.mean_max_distractor_similarity,
            matched_bytes_hrr_result.raw_recall_at_1,
            matched_bytes_hrr_result.raw_recall_ci95.0,
            matched_bytes_hrr_result.raw_recall_ci95.1,
            matched_bytes_hrr_result.mean_target_similarity,
            matched_bytes_hrr_result.mean_max_distractor_similarity
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::k95;

    #[test]
    fn k95_reports_largest_tested_capacity_meeting_lower_bound() {
        assert_eq!(
            k95([(1, 1.0), (7, 0.98), (31, 0.95), (63, 0.949)]),
            Some(31)
        );
    }

    #[test]
    fn k95_is_na_when_no_tested_capacity_meets_the_recall_bound() {
        assert_eq!(k95([(1, 0.90), (7, 0.949)]), None);
    }

    #[test]
    fn k95_does_not_requalify_after_a_capacity_point_fails() {
        assert_eq!(k95([(31, 0.99), (1, 1.0), (7, 0.94)]), Some(1));
    }
}
