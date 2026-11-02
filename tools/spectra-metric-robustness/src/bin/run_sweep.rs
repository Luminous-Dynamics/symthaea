use std::error::Error;

use spectra_metric_robustness::{
    SimulationDesign, StressWorld, evaluate, simulate,
};

fn main() -> Result<(), Box<dyn Error>> {
    let seed_count = std::env::args()
        .nth(1)
        .map(|value| value.parse::<u64>())
        .transpose()?
        .unwrap_or(256);
    if seed_count == 0 {
        return Err("seed count must be positive".into());
    }

    println!(
        "world,seed,total_trials,observed_trials,missing_trials,d_prime,criterion_c,auc,average_precision,prevalence,variance_ratio,signal_skew,noise_skew,signal_excess_kurtosis,noise_excess_kurtosis,lag1_residual_correlation,early_late_d_prime_delta,early_late_criterion_delta,participant_d_prime_sd,signal_missing_rate,noise_missing_rate,auxiliary_auc,disposition,flags"
    );

    for world in StressWorld::ALL {
        for seed_index in 0..seed_count {
            let seed = 0x5EED_2026_u64.wrapping_add(seed_index);
            let design = SimulationDesign {
                participants: 24,
                sessions_per_participant: 4,
                trials_per_session: 32,
                seed,
            };
            let bundle = simulate(world, design)?;
            let report = evaluate(&bundle.candidate, &bundle.evaluator)?;
            let flags = report
                .flags
                .iter()
                .map(|flag| flag.as_str())
                .collect::<Vec<_>>()
                .join(";");
            let delta_d = report
                .early_late_d_prime_delta
                .map(|value| format!("{value:.8}"))
                .unwrap_or_default();
            let delta_c = report
                .early_late_criterion_delta
                .map(|value| format!("{value:.8}"))
                .unwrap_or_default();
            let participant_sd = report
                .participant_d_prime_sd
                .map(|value| format!("{value:.8}"))
                .unwrap_or_default();

            println!(
                "{},{},{},{},{},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{},{},{},{:.8},{:.8},{:.8},{},{}",
                report.world.as_str(),
                report.seed,
                report.total_trials,
                report.observed_trials,
                report.missing_trials,
                report.signal_detection.d_prime,
                report.signal_detection.criterion_c,
                report.roc_auc,
                report.average_precision,
                report.positive_prevalence,
                report.signal_noise_variance_ratio,
                report.signal_skewness,
                report.noise_skewness,
                report.signal_excess_kurtosis,
                report.noise_excess_kurtosis,
                report.within_session_lag_one_residual_correlation,
                delta_d,
                delta_c,
                participant_sd,
                report.signal_missing_rate,
                report.noise_missing_rate,
                report.auxiliary_channel_auc,
                report.disposition.as_str(),
                flags
            );
        }
    }

    Ok(())
}
