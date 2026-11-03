use std::collections::BTreeMap;
use std::error::Error;

use spectra_metric_robustness::{
    DiagnosticFlag, MetricDisposition, SimulationDesign, StressWorld, evaluate,
    simulate_with_severity,
};

const SEVERITIES: [f64; 6] = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0];
const STRESS_WORLDS: [StressWorld; 10] = [
    StressWorld::UnequalVariance,
    StressWorld::HeavyTailed,
    StressWorld::ContaminatedMixture,
    StressWorld::SerialDependence,
    StressWorld::SensitivityDrift,
    StressWorld::CriterionDrift,
    StressWorld::ParticipantHeterogeneity,
    StressWorld::MissingAtRandomBySession,
    StressWorld::OutcomeDependentMissingness,
    StressWorld::CovariateLeakage,
];

#[derive(Debug, Default)]
struct CampaignSummary {
    campaigns: u64,
    any_flag: u64,
    assumption_concern: u64,
    not_evaluable: u64,
    no_configured_concern: u64,
    flag_counts: [u64; DiagnosticFlag::ALL.len()],
}

impl CampaignSummary {
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
        if total == 0 { 0.0 } else { count as f64 / total as f64 }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let seed_count = std::env::args()
        .nth(1)
        .map(|value| value.parse::<u64>())
        .transpose()?
        .unwrap_or(128);
    if seed_count == 0 {
        return Err("seed count must be positive".into());
    }

    let mut summaries = BTreeMap::<(StressWorld, u16), CampaignSummary>::new();

    println!(
        "world,severity,seed,total_trials,observed_trials,missing_trials,d_prime,criterion_c,auc,average_precision,variance_ratio,lag1_residual_correlation,early_late_d_prime_delta,early_late_criterion_delta,participant_d_prime_sd,signal_missing_rate,noise_missing_rate,session_missingness_range,auxiliary_auc,disposition,flags"
    );

    for world in STRESS_WORLDS {
        for severity in SEVERITIES {
            let severity_key = (severity * 1000.0).round() as u16;
            for seed_index in 0..seed_count {
                let seed = 0xD05E_2026_u64.wrapping_add(seed_index);
                let design = SimulationDesign {
                    participants: 24,
                    sessions_per_participant: 4,
                    trials_per_session: 32,
                    seed,
                };
                let bundle = simulate_with_severity(world, design, severity)?;
                let report = evaluate(&bundle.candidate, &bundle.evaluator)?;
                let flags = report.flags
                    .iter()
                    .map(|flag| flag.as_str())
                    .collect::<Vec<_>>()
                    .join(";");

                summaries.entry((world, severity_key))
                    .or_default()
                    .observe(&report.flags, report.disposition);

                let delta_d = report.early_late_d_prime_delta
                    .map(|value| format!("{value:.8}")).unwrap_or_default();
                let delta_c = report.early_late_criterion_delta
                    .map(|value| format!("{value:.8}")).unwrap_or_default();
                let participant_sd = report.participant_d_prime_sd
                    .map(|value| format!("{value:.8}")).unwrap_or_default();

                println!(
                    "{},{:.2},{},{},{},{},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{},{},{},{:.8},{:.8},{:.8},{:.8},{},{}",
                    report.world.as_str(),
                    report.severity,
                    report.seed,
                    report.total_trials,
                    report.observed_trials,
                    report.missing_trials,
                    report.signal_detection.d_prime,
                    report.signal_detection.criterion_c,
                    report.roc_auc,
                    report.average_precision,
                    report.signal_noise_variance_ratio,
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
    }

    eprintln!("# Severity dose-response summary: descriptive campaign counts/rates only");
    let mut header = vec![
        "world".to_string(), "severity".to_string(), "campaigns".to_string(),
        "any_flag_count".to_string(), "any_flag_rate".to_string(),
        "assumption_concern_count".to_string(), "assumption_concern_rate".to_string(),
        "not_evaluable_count".to_string(), "not_evaluable_rate".to_string(),
        "no_configured_concern_count".to_string(), "no_configured_concern_rate".to_string(),
    ];
    for flag in DiagnosticFlag::ALL {
        header.push(format!("{}_count", flag.as_str()));
        header.push(format!("{}_rate", flag.as_str()));
    }
    eprintln!("{}", header.join(","));

    for world in STRESS_WORLDS {
        for severity in SEVERITIES {
            let key = (world, (severity * 1000.0).round() as u16);
            let Some(summary) = summaries.get(&key) else { continue; };
            let n = summary.campaigns;
            let mut fields = vec![
                world.as_str().to_string(),
                format!("{severity:.2}"),
                n.to_string(),
                summary.any_flag.to_string(),
                format!("{:.6}", CampaignSummary::rate(summary.any_flag, n)),
                summary.assumption_concern.to_string(),
                format!("{:.6}", CampaignSummary::rate(summary.assumption_concern, n)),
                summary.not_evaluable.to_string(),
                format!("{:.6}", CampaignSummary::rate(summary.not_evaluable, n)),
                summary.no_configured_concern.to_string(),
                format!("{:.6}", CampaignSummary::rate(summary.no_configured_concern, n)),
            ];
            for count in summary.flag_counts {
                fields.push(count.to_string());
                fields.push(format!("{:.6}", CampaignSummary::rate(count, n)));
            }
            eprintln!("{}", fields.join(","));
        }
    }

    Ok(())
}
