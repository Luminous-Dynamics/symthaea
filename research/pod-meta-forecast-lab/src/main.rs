use pod_meta_forecast_lab::{ForecastMethod, SelectionPolicy, run_benchmark};

fn main() {
    let report = run_benchmark(1729);
    println!(
        "record_type\tseed\tmethod\tpartition\tobservations\tbrier\tlog_loss\tece_10\tmean_predicted\tobserved_rate"
    );
    for metric in &report.forecast_metrics {
        println!(
            "FORECAST\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            report.seed,
            metric.method.as_str(),
            metric.partition.as_str(),
            metric.observations,
            metric.brier_score,
            metric.log_loss,
            metric.ece_10,
            metric.mean_predicted_probability,
            metric.observed_success_rate
        );
    }
    println!(
        "record_type\tseed\tmethod\tpartition\tbin\tobservations\tmean_predicted\tobserved_rate"
    );
    for bin in &report.reliability_bins {
        println!(
            "RELIABILITY\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            report.seed,
            bin.method.as_str(),
            bin.partition.as_str(),
            bin.bin,
            bin.observations,
            bin.mean_predicted_probability
                .map(|v| format!("{v:.6}"))
                .unwrap_or_else(|| "-".to_owned()),
            bin.observed_success_rate
                .map(|v| format!("{v:.6}"))
                .unwrap_or_else(|| "-".to_owned())
        );
    }
    println!(
        "record_type\tseed\tregime\tpolicy\tepisodes\tverified_proofs\tno_verified_proof\tsuccess_rate\ttotal_spent\ttotal_budget\tverified_proofs_per_work_unit\twork_units_per_verified_proof\taverage_work_units_per_episode\tbudget_violations\tmaximum_episode_spend\toperations"
    );
    for metric in &report.selection_metrics {
        let work_per_proof = metric
            .work_units_per_verified_proof()
            .map(|v| format!("{v:.6}"))
            .unwrap_or_else(|| "-".to_owned());
        println!(
            "SELECTION\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{}\t{}\t{:.6}\t{}\t{:.6}\t{}\t{}\t{}",
            report.seed,
            metric.regime.as_str(),
            metric.policy.as_str(),
            metric.episodes,
            metric.verified_proofs,
            metric.no_verified_proof,
            metric.success_rate(),
            metric.total_spent,
            metric.total_budget,
            metric.verified_proofs_per_work_unit(),
            work_per_proof,
            metric.average_work_units_per_episode(),
            metric.budget_violations,
            metric.maximum_episode_spend,
            metric.total_operations
        );
    }
    assert_eq!(report.forecast_metrics.len(), ForecastMethod::ALL.len() * 3);
    assert_eq!(
        report.selection_metrics.len(),
        SelectionPolicy::ALL.len() * 5
    );
    assert_eq!(report.training_observations, 2048);
    assert_eq!(report.evaluation_observations, 1280);
}
