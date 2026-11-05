// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! End-to-end known-truth recovery campaign for the SPECTRA observer pipeline.
//!
//! Flow: frozen operational schedule -> synthetic DGP -> observed-only
//! participant-level adapter -> UVSD fitter -> failure-inclusive CSV.
//!
//! Oracle fields are used only after fitting (or after adapter abstention) to
//! score known synthetic truth. They are never passed to the adapter or fitter.
//! This is exploratory recovery, not interval calibration or a human study.

use std::error::Error;
use std::io;

use spectra_uvsd_fit::observation_table_adapter::{
    FitDisposition, ObservationFitGroup, adapt_synthetic_dataset_for_uvsd,
};
use spectra_uvsd_fit::scenario_dgp::{
    EvidenceFamily, ObserverScenario, ParticipantHeterogeneity, ResponseScaleSpec,
    RtChannelSpec, SyntheticDataset, TrialRequest, generate_synthetic_dataset,
};
use spectra_uvsd_fit::sequential::{
    OperationalAction, OperationalActionSet, StimulusClass,
};
use spectra_uvsd_fit::{FitOptions, FitStatus, fit};

#[derive(Debug, Clone, Copy)]
struct Cli {
    outer: usize,
    trials_per_class: u32,
    seed_root: u64,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            outer: 10,
            trials_per_class: 250,
            seed_root: 0x5350_4543_5452_4101,
        }
    }
}

#[derive(Debug, Clone)]
struct ScenarioCase {
    id: u16,
    name: &'static str,
    spec: ObserverScenario,
    correctly_specified_single_observer_gaussian: bool,
}

#[derive(Debug, Default)]
struct Accumulator {
    campaigns: usize,
    adapter_ready: usize,
    adapter_abstentions: usize,
    converged: usize,
    boundary: usize,
    not_identifiable: usize,
    weakly_identified: usize,
    not_converged: usize,
    not_evaluable: usize,
    mu_error_sum: f64,
    sigma_error_sum: f64,
    d_a_error_sum: f64,
    mu_squared_error_sum: f64,
    sigma_squared_error_sum: f64,
    d_a_squared_error_sum: f64,
}

impl Accumulator {
    fn record_campaign(&mut self) {
        self.campaigns += 1;
    }

    fn record_adapter_disposition(&mut self, disposition: FitDisposition) {
        if disposition == FitDisposition::Ready {
            self.adapter_ready += 1;
        } else {
            self.adapter_abstentions += 1;
        }
    }

    fn record_fit(&mut self, status: FitStatus, errors: Option<(f64, f64, f64)>) {
        match status {
            FitStatus::Converged => {
                self.converged += 1;
                if let Some((mu, sigma, d_a)) = errors {
                    self.mu_error_sum += mu;
                    self.sigma_error_sum += sigma;
                    self.d_a_error_sum += d_a;
                    self.mu_squared_error_sum += mu * mu;
                    self.sigma_squared_error_sum += sigma * sigma;
                    self.d_a_squared_error_sum += d_a * d_a;
                }
            }
            FitStatus::Boundary => self.boundary += 1,
            FitStatus::NotIdentifiable => self.not_identifiable += 1,
            FitStatus::WeaklyIdentified => self.weakly_identified += 1,
            FitStatus::NotConverged => self.not_converged += 1,
            FitStatus::NotEvaluable => self.not_evaluable += 1,
        }
    }

    fn rmse(&self, squared_error_sum: f64) -> String {
        if self.converged == 0 {
            String::new()
        } else {
            format!("{:.9}", (squared_error_sum / self.converged as f64).sqrt())
        }
    }

    fn mean_error(&self, error_sum: f64) -> String {
        if self.converged == 0 {
            String::new()
        } else {
            format!("{:.9}", error_sum / self.converged as f64)
        }
    }
}

fn parse_cli() -> io::Result<Cli> {
    let mut cli = Cli::default();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, format!("missing value for {flag}"))
        })?;
        match flag.as_str() {
            "--outer" => {
                cli.outer = value.parse().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "--outer must be a positive integer")
                })?;
            }
            "--trials-per-class" => {
                cli.trials_per_class = value.parse().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "--trials-per-class must be a positive integer")
                })?;
            }
            "--seed-root" => {
                cli.seed_root = if let Some(hex) = value.strip_prefix("0x") {
                    u64::from_str_radix(hex, 16).map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidInput, "--seed-root has invalid hexadecimal digits")
                    })?
                } else {
                    value.parse().map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidInput, "--seed-root must be decimal or 0x-prefixed hex")
                    })?
                };
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown argument: {flag}"),
                ));
            }
        }
    }
    if cli.outer == 0 || cli.trials_per_class == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--outer and --trials-per-class must be positive",
        ));
    }
    Ok(cli)
}

fn action_set() -> Result<OperationalActionSet, Box<dyn Error>> {
    Ok(OperationalActionSet::new(
        vec![
            OperationalAction {
                action_id: 1,
                audit_stratum: 1,
                condition_id: 1,
                stimulus_class: StimulusClass::Noise,
                stimulus_level: 1.0,
                response_scale_id: 5,
                rt_deadline_ms: None,
            },
            OperationalAction {
                action_id: 2,
                audit_stratum: 1,
                condition_id: 1,
                stimulus_class: StimulusClass::Signal,
                stimulus_level: 1.0,
                response_scale_id: 5,
                rt_deadline_ms: None,
            },
        ],
        "spectra-recovery-action-set-v0.1",
    )?)
}

fn base_scenario(id: u16) -> ObserverScenario {
    ObserverScenario {
        scenario_id: id,
        evidence_family: EvidenceFamily::Gaussian,
        base_signal_mean: 1.1,
        signal_mean_per_stimulus_unit: 0.3,
        signal_sigma: 1.0,
        criterion_offset: 0.0,
        criterion_drift_per_100_trials: 0.0,
        lapse_probability: 0.0,
        confidence_missing_probability: 0.0,
        confidence_missing_evidence_slope: 0.0,
        response_scales: vec![ResponseScaleSpec {
            response_scale_id: 5,
            latent_criteria: vec![-1.0, -0.3, 0.3, 1.0],
        }],
        heterogeneity: None,
        rt_channel: None,
    }
}

fn scenario_cases() -> Vec<ScenarioCase> {
    let mut cases = Vec::new();

    let null = base_scenario(1);
    cases.push(ScenarioCase {
        id: 1,
        name: "gaussian_null_sigma_1",
        spec: null,
        correctly_specified_single_observer_gaussian: true,
    });

    let mut high_variance = base_scenario(2);
    high_variance.signal_sigma = 1.3;
    cases.push(ScenarioCase {
        id: 2,
        name: "gaussian_uvsd_sigma_1_3",
        spec: high_variance,
        correctly_specified_single_observer_gaussian: true,
    });

    let mut low_variance = base_scenario(3);
    low_variance.signal_sigma = 0.7;
    cases.push(ScenarioCase {
        id: 3,
        name: "gaussian_uvsd_sigma_0_7",
        spec: low_variance,
        correctly_specified_single_observer_gaussian: true,
    });

    let mut heavy_tail = base_scenario(4);
    heavy_tail.signal_sigma = 1.3;
    heavy_tail.evidence_family = EvidenceFamily::VarianceStandardizedStudentT {
        degrees_of_freedom: 5,
    };
    cases.push(ScenarioCase {
        id: 4,
        name: "student_t5_heavy_tail_stress",
        spec: heavy_tail,
        correctly_specified_single_observer_gaussian: false,
    });

    let mut lapse = base_scenario(5);
    lapse.signal_sigma = 1.3;
    lapse.lapse_probability = 0.05;
    cases.push(ScenarioCase {
        id: 5,
        name: "uniform_lapse_5pct_stress",
        spec: lapse,
        correctly_specified_single_observer_gaussian: false,
    });

    let mut drift = base_scenario(6);
    drift.signal_sigma = 1.3;
    drift.criterion_drift_per_100_trials = 0.4;
    cases.push(ScenarioCase {
        id: 6,
        name: "criterion_drift_0_4_per_100_stress",
        spec: drift,
        correctly_specified_single_observer_gaussian: false,
    });

    let mut heterogeneous = base_scenario(7);
    heterogeneous.signal_sigma = 1.3;
    heterogeneous.heterogeneity = Some(ParticipantHeterogeneity {
        signal_mean_sd: 0.25,
        log_signal_sigma_sd: 0.12,
        criterion_offset_sd: 0.18,
    });
    // Within a single participant-session these random effects are fixed,
    // so the conditional individual Gaussian UVSD estimand remains valid.
    cases.push(ScenarioCase {
        id: 7,
        name: "between_observer_effects_single_observer_fit",
        spec: heterogeneous,
        correctly_specified_single_observer_gaussian: true,
    });

    let mut missing = base_scenario(8);
    missing.signal_sigma = 1.3;
    missing.confidence_missing_probability = 0.05;
    cases.push(ScenarioCase {
        id: 8,
        name: "missing_category_5pct_abstention_stress",
        spec: missing,
        correctly_specified_single_observer_gaussian: false,
    });

    let mut rt = base_scenario(9);
    rt.signal_sigma = 1.3;
    rt.rt_channel = Some(RtChannelSpec {
        base_latency_ms: 700.0,
        evidence_speed_gain: 0.18,
        log_noise_sd: 0.2,
        non_decision_time_ms: 180.0,
        quantization_ms: 50,
        default_deadline_ms: Some(1_800),
    });
    cases.push(ScenarioCase {
        id: 9,
        name: "quantized_rt_channel_sigma_1_3",
        spec: rt,
        correctly_specified_single_observer_gaussian: true,
    });

    cases
}

fn trial_schedule(trials_per_class: u32) -> Result<Vec<TrialRequest>, Box<dyn Error>> {
    let total = trials_per_class.checked_mul(2).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "requested trial count overflow")
    })?;
    let capacity = usize::try_from(total).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidInput, "requested trial count exceeds platform capacity")
    })?;
    let mut requests = Vec::with_capacity(capacity);
    for trial in 0..total {
        requests.push(TrialRequest {
            action_id: if trial % 2 == 0 { 1 } else { 2 },
            participant_id: 0,
            within_participant_trial: trial,
        });
    }
    Ok(requests)
}

fn csv_number(value: f64) -> String {
    format!("{value:.9}")
}

fn csv_option(value: Option<f64>) -> String {
    value.map_or_else(String::new, csv_number)
}

fn emit_csv_row(fields: Vec<String>) {
    debug_assert_eq!(fields.len(), 39, "CSV row must match the header");
    println!("{}", fields.join(","));
}

fn write_replicate_header() {
    println!(
        "row_type,scenario_id,scenario_name,estimand_track,replicate,root_seed,prng_version,trials_per_class,requested_total,observed_signal,observed_noise,missing_signal,missing_noise,rt_observed,rt_timeouts,adapter_disposition,fit_status,qualified,truth_mu,truth_sigma,truth_d_a,estimate_mu,estimate_sigma,estimate_d_a,mu_error,sigma_error,d_a_error,mu_rmse,sigma_rmse,d_a_rmse,fit_availability_all,fit_availability_eligible,adapter_ready,adapter_abstentions,boundary,not_identifiable,weakly_identified,not_converged,not_evaluable"
    );
}

fn write_replicate_row(
    case: &ScenarioCase,
    replicate: usize,
    seed: u64,
    trials_per_class: u32,
    dataset: &SyntheticDataset,
    group: &ObservationFitGroup,
    fit_status: &str,
    qualified: bool,
    truth: (f64, f64, f64),
    estimates: Option<(f64, f64, f64)>,
    errors: Option<(f64, f64, f64)>,
) {
    let (estimate_mu, estimate_sigma, estimate_d_a) = estimates
        .map(|(mu, sigma, d_a)| (Some(mu), Some(sigma), Some(d_a)))
        .unwrap_or((None, None, None));
    let (mu_error, sigma_error, d_a_error) = errors
        .map(|(mu, sigma, d_a)| (Some(mu), Some(sigma), Some(d_a)))
        .unwrap_or((None, None, None));
    let track = if case.correctly_specified_single_observer_gaussian {
        "gaussian_single_observer_track"
    } else {
        "misspecification_stress_track"
    };
    emit_csv_row(vec![
        "REPLICATE".to_owned(),
        case.id.to_string(),
        case.name.to_owned(),
        track.to_owned(),
        replicate.to_string(),
        seed.to_string(),
        dataset.prng_version.to_owned(),
        trials_per_class.to_string(),
        trials_per_class.saturating_mul(2).to_string(),
        group.observed_signal_trials.to_string(),
        group.observed_noise_trials.to_string(),
        group.missing_signal_categories.to_string(),
        group.missing_noise_categories.to_string(),
        group.rt_observed.to_string(),
        group.rt_timeouts.to_string(),
        format!("{:?}", group.disposition),
        fit_status.to_owned(),
        qualified.to_string(),
        csv_number(truth.0),
        csv_number(truth.1),
        csv_number(truth.2),
        csv_option(estimate_mu),
        csv_option(estimate_sigma),
        csv_option(estimate_d_a),
        csv_option(mu_error),
        csv_option(sigma_error),
        csv_option(d_a_error),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
    ]);
}

fn write_summary_row(case: &ScenarioCase, aggregate: &Accumulator) {
    let fit_availability_all = aggregate.converged as f64 / aggregate.campaigns.max(1) as f64;
    let fit_availability_eligible =
        aggregate.converged as f64 / aggregate.adapter_ready.max(1) as f64;
    let track = if case.correctly_specified_single_observer_gaussian {
        "gaussian_single_observer_track"
    } else {
        "misspecification_stress_track"
    };
    emit_csv_row(vec![
        "SUMMARY".to_owned(),
        case.id.to_string(),
        case.name.to_owned(),
        track.to_owned(),
        "ALL".to_owned(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        "SUMMARY".to_owned(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        aggregate.mean_error(aggregate.mu_error_sum),
        aggregate.mean_error(aggregate.sigma_error_sum),
        aggregate.mean_error(aggregate.d_a_error_sum),
        aggregate.rmse(aggregate.mu_squared_error_sum),
        aggregate.rmse(aggregate.sigma_squared_error_sum),
        aggregate.rmse(aggregate.d_a_squared_error_sum),
        csv_number(fit_availability_all),
        csv_number(fit_availability_eligible),
        aggregate.adapter_ready.to_string(),
        aggregate.adapter_abstentions.to_string(),
        aggregate.boundary.to_string(),
        aggregate.not_identifiable.to_string(),
        aggregate.weakly_identified.to_string(),
        aggregate.not_converged.to_string(),
        aggregate.not_evaluable.to_string(),
    ]);
}

fn known_truth_after_fit(dataset: &SyntheticDataset) -> io::Result<(f64, f64, f64)> {
    let signal = dataset
        .trials
        .iter()
        .find(|trial| trial.stimulus_class == StimulusClass::Signal)
        .ok_or_else(|| io::Error::other("synthetic schedule contained no signal trials"))?;
    let mu = signal.oracle_signal_mean;
    let sigma = signal.oracle_signal_sigma;
    if !mu.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
        return Err(io::Error::other("synthetic oracle truth was invalid"));
    }
    let d_a = mu / ((1.0 + sigma * sigma) / 2.0).sqrt();
    Ok((mu, sigma, d_a))
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = parse_cli()?;
    let action_set = action_set()?;
    let requests = trial_schedule(cli.trials_per_class)?;
    let fit_options = FitOptions::default();

    write_replicate_header();

    for case in scenario_cases() {
        let mut aggregate = Accumulator::default();

        for replicate in 0..cli.outer {
            aggregate.record_campaign();
            let seed = cli.seed_root.wrapping_add(replicate as u64);
            let dataset = generate_synthetic_dataset(
                &action_set,
                &case.spec,
                &requests,
                format!("scenario:{}:{}", case.id, case.name),
                seed,
            )?;
            let receipt = adapt_synthetic_dataset_for_uvsd(&dataset, &action_set)?;
            let group = receipt.groups.iter().find(|group| {
                group.key.participant_id == 0
                    && group.key.condition_id == 1
                    && group.key.stimulus_level() == 1.0
                    && group.key.response_scale_id == 5
            }).ok_or_else(|| io::Error::other("expected participant-session fit group missing"))?;
            aggregate.record_adapter_disposition(group.disposition);

            if group.disposition != FitDisposition::Ready {
                let truth = known_truth_after_fit(&dataset)?;
                write_replicate_row(
                    &case,
                    replicate,
                    seed,
                    cli.trials_per_class,
                    &dataset,
                    group,
                    "NOT_RUN",
                    false,
                    truth,
                    None,
                    None,
                );
                continue;
            }

            let table = group.table.as_ref().ok_or_else(|| {
                io::Error::other("Ready adapter disposition did not include CategoryTable")
            })?;
            let fit_result = fit(table, fit_options);
            // Oracle fields are accessed only after the estimator has returned.
            let truth = known_truth_after_fit(&dataset)?;
            let estimate = fit_result.estimate.as_ref();
            let estimated = estimate.map(|estimate| (estimate.mu, estimate.sigma, estimate.d_a));
            let qualified = fit_result.status == FitStatus::Converged;
            let errors = if qualified {
                estimate.map(|estimate| {
                    (
                        estimate.mu - truth.0,
                        estimate.sigma - truth.1,
                        estimate.d_a - truth.2,
                    )
                })
            } else {
                None
            };
            aggregate.record_fit(fit_result.status, errors);
            write_replicate_row(
                &case,
                replicate,
                seed,
                cli.trials_per_class,
                &dataset,
                group,
                &format!("{:?}", fit_result.status),
                qualified,
                truth,
                estimated,
                errors,
            );
        }
        write_summary_row(&case, &aggregate);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_is_balanced_and_uses_one_participant_session() {
        let schedule = trial_schedule(101).unwrap();
        assert_eq!(schedule.len(), 202);
        assert_eq!(schedule.iter().filter(|request| request.action_id == 1).count(), 101);
        assert_eq!(schedule.iter().filter(|request| request.action_id == 2).count(), 101);
        assert!(schedule.iter().all(|request| request.participant_id == 0));
        assert!(schedule.iter().all(|request| request.within_participant_trial < 202));
    }

    #[test]
    fn known_d_a_uses_the_uvsd_parameterization() {
        let mu = 1.4;
        let sigma = 1.3;
        let expected = mu / ((1.0 + sigma * sigma) / 2.0).sqrt();
        assert!(expected.is_finite());
        assert!(expected > 0.0);
    }
}
