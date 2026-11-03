use std::collections::BTreeMap;
use std::error::Error;

use spectra_metric_robustness::{
    DiagnosticFlag, MetricDisposition, SimulationDesign, StressWorld, evaluate, simulate,
};

#[derive(Debug, Default)]
struct CampaignAggregate {
    campaigns: u64,
    any_flag: u64,
    assumption_concern: u64,
    no_configured_concern: u64,
    not_evaluable: u64,
    flag_counts: [u64; DiagnosticFlag::ALL.len()],
}

impl CampaignAggregate {
    fn observe(&mut self, flags: &[DiagnosticFlag], disposition: MetricDisposition) {
        self.campaigns += 1;
        if !flags.is_empty() {
            self.any_flag += 1;
        }
        match disposition {
            MetricDisposition::NotEvaluable => self.not_evaluable += 1,
            MetricDisposition::AssumptionConcern => self.assumption_concern += 1,
            MetricDisposition::NoConfiguredConcern => self.no_configured_concern += 1,
        }
        for (index, flag) in DiagnosticFlag::ALL.iter().enumerate() {
            if flags.contains(flag) {
                self.flag_counts[index] += 1;
            }
        }
    }

    fn rate(count: u64, total: u64) -> f64 {
        if total == 0 {
            0.0
        } else {
            count as f64 / total as f64
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let seed_count = std::env::args()
        .nth(1)
        .map(|value| value.parse::<u64>())
        .transpose()?
        .unwrap_or(256);
    if seed_count == 0 {
        return Err("seed count must be positive".into());
    }

    let mut summaries = BTreeMap::<StressWorld, CampaignAggregate>::new();

    println!(
        "world,seed,total_trials,observed_trials,missing_trials,d_prime,criterion_c,auc,average_precision,prevalence,variance_ratio,signal_skew,noise_skew,signal_excess_kurtosis,noise_excess_kurtosis,lag1_residual_correlation,early_late_d_prime_delta,early_late_criterion_delta,participant_d_prime_sd,signal_missing_rate,noise_missing_rate,session_missingness_range,auxiliary_auc,disposition,flags"
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

            summaries
                .entry(report.world)
                .or_default()
                .observe(&report.flags, report.disposition);

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
                "{},{},{},{},{},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{},{},{},{:.8},{:.8},{:.8},{:.8},{},{}",
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
                report.session_missingness_range,
                report.auxiliary_channel_auc,
                report.disposition.as_str(),
                flags
            );
        }
    }

    // This second report makes the diagnostic battery's aggregate false-alert
    // rate visible. Per-flag rates alone do not control the chance that at
    // least one of nine diagnostics fires. Counts/rates are descriptive only;
    // they are not confidence intervals or hypothesis-test results.
    eprintln!("# Per-world diagnostic campaign summary (descriptive counts/rates; no inferential intervals)");
    let mut header = vec![
        "world".to_string(),
        "campaigns".to_string(),
        "any_flag_count".to_string(),
        "any_flag_rate".to_string(),
        "assumption_concern_count".to_string(),
        "assumption_concern_rate".to_string(),
        "not_evaluable_count".to_string(),
        "not_evaluable_rate".to_string(),
        "no_configured_concern_count".to_string(),
        "no_configured_concern_rate".to_string(),
    ];
    for flag in DiagnosticFlag::ALL {
        header.push(format!("{}_count", flag.as_str()));
        header.push(format!("{}_rate", flag.as_str()));
    }
    eprintln!("{}", header.join(","));

    for world in StressWorld::ALL {
        let Some(summary) = summaries.get(&world) else {
            continue;
        };
        let n = summary.campaigns;
        let mut fields = vec![
            world.as_str().to_string(),
            n.to_string(),
            summary.any_flag.to_string(),
            format!("{:.6}", CampaignAggregate::rate(summary.any_flag, n)),
            summary.assumption_concern.to_string(),
            format!("{:.6}", CampaignAggregate::rate(summary.assumption_concern, n)),
            summary.not_evaluable.to_string(),
            format!("{:.6}", CampaignAggregate::rate(summary.not_evaluable, n)),
            summary.no_configured_concern.to_string(),
            format!("{:.6}", CampaignAggregate::rate(summary.no_configured_concern, n)),
        ];
        for count in summary.flag_counts {
            fields.push(count.to_string());
            fields.push(format!("{:.6}", CampaignAggregate::rate(count, n)));
        }
        eprintln!("{}", fields.join(","));
    }

    Ok(())
}
