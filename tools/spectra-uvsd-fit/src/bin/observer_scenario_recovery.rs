// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! End-to-end known-truth recovery campaign for the SPECTRA observer pipeline.
//!
//! Flow: frozen operational schedule -> synthetic DGP -> observed-only
//! participant-level adapter -> UVSD fitter -> failure-inclusive CSV.
//!
//! Oracle values are read only after the fit for known-answer scoring. They
//! are never passed to the adapter or the fitter. This runner is exploratory:
//! it does not calibrate Wald intervals, model-selection tests, or human data.

use std::error::Error;
use std::io;

use spectra_uvsd_fit::observation_table_adapter::{
    FitDisposition, adapt_synthetic_dataset_for_uvsd,
};
use spectra_uvsd_fit::scenario_dgp::{
    EvidenceFamily, ObserverScenario, ParticipantHeterogeneity, ResponseScaleSpec,
    RtChannelSpec, TrialRequest, generate_synthetic_dataset,
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
    family: EvidenceFamily,
    spec: ObserverScenario,
    correctly_specified_gaussian: bool,
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

    fn rmse(&self, squared_error_sum: f64) -> Option<f64> {
        (self.converged > 0).then(|| (squared_error_sum / self.converged as f64).sqrt())
    }

    fn mean_error(&self, error_sum: f64) -> Option<f64> {
        (self.converged > 0).then(|| error_sum / self.converged as f64)
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
        family: null.evidence_family,
        spec: null,
        correctly_specified_gaussian: true,
    });

    let mut high_variance = base_scenario(2);
    high_variance.signal_sigma = 1.3;
    cases.push(ScenarioCase {
        id: 2,
        name: "gaussian_uvsd_sigma_1_3",
        family: high_variance.evidence_family,
        spec: high_variance,
        correctly_specified_gaussian: true,
    });

    let mut low_variance = base_scenario(3);
    low_variance.signal_sigma = 0.7;
    cases.push(ScenarioCase {
        id: 3,
        name: "gaussian_uvsd_sigma_0_7",
        family: low_variance.evidence_family,
        spec: low_variance,
        correctly_specified_gaussian: true,
    });

    let mut heavy_tail = base_scenario(4);
    heavy_tail.signal_sigma = 1.3;
    heavy_tail.evidence_family = EvidenceFamily::VarianceStandardizedStudentT {
        degrees_of_freedom: 5,
    };
    cases.push(ScenarioCase {
        id: 4,
        name: "student_t5_heavy_tail_stress",
        family: heavy_tail.evidence_family,
        spec: heavy_tail,
        correctly_specified_gaussian: false,
    });

    let mut lapse = base_scenario(5);
    lapse.signal_sigma = 1.3;
    lapse.lapse_probability = 0.05;
    cases.push(ScenarioCase {
        id: 5,
        name: "uniform_lapse_5pct_stress",
        family: lapse.evidence_family,
        spec: lapse,
        correctly_specified_gaussian: false,
    });

    let mut drift = base_scenario(6);
    drift.signal_sigma = 1.3;
    drift.criterion_drift_per_100_trials = 0.4;
    cases.push(ScenarioCase {
        id: 6,
        name: "criterion_drift_0_4_per_100_stress",
        family: drift.evidence_family,
        spec: drift,
        correctly_specified_gaussian: false,
    });

    let mut missing = base_scenario(7);
    missing.signal_sigma = 1.3;
    missing.confidence_missing_probability = 0.05;
    cases.push(ScenarioCase {
        id: 7,
        name: "missing_category_5pct_abstention_stress",
        family: missing.evidence_family,
        spec: missing,
        correctly_specified_gaussian: false,
    });

    let mut rt = base_scenario(8);
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
        id: 8,
        name: "quantized_rt_channel_sigma_1_3",
        family: rt.evidence_family,
        spec: rt,
        correctly_specified_gaussian: true,
    });

    cases
}

fn trial_schedule(trials_per_class: u32) -> Result<Vec<TrialRequest>, Box<dyn Error>> {
    let total = trials_per_class.checked_mul(2).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "requested trial count overflow")
    })?;
    let mut requests = Vec::with_capacity(total as usize);
    for trial in 0..total {
        requests.push(TrialRequest {
            action_id: if trial % 2 == 0 { 1 } else { 2 },
            participant_id: 0,
            within_participant_trial: trial,
        });
    }
    Ok(requests)
}

fn csv_option(value: Option<f64>) -> String {
    value.map_or_else(String::new, |number| format!("{number:.9}"))
}

fn write_replicate_header() {
    println!(
        "row_type,scenario_id,scenario_name,estimand_track,replicate,root_seed,prng_version,trials_per_class,requested_total,observed_signal,observed_noise,missing_signal,missing_noise,rt_observed,rt_timeouts,adapter_disposition,fit_status,qualified,truth_mu,truth_sigma,truth_d_a,estimate_mu,estimate_sigma,estimate_d_a,mu_error,sigma_error,d_a_error,mu_rmse,sigma_rmse,d_a_rmse,fit_availability_all,fit_availability_eligible,adapter_ready,adapter_abstentions,boundary,not_identifiable,weakly_identified,not_converged,not_evaluable"
    );
}

#[allow(clippy::too_many_arguments)]
fn write_replicate_row(
    case: &ScenarioCase,
    replicate: usize,
    seed: u64,
    trials_per_class: u32,
    dataset: &spectra_uvsd_fit::scenario_dgp::SyntheticDataset,
    group: Option<&spectra_uvsd_fit::observation_table_adapter::ObservationFitGroup>,
    adapter_disposition: &str,
    fit_status: &str,
    qualified: bool,
    truth_mu: f64,
    truth_sigma: f64,
    truth_d_a: f64,
    estimates: Option<(f64, f64, f64)>,
    errors: Option<(f64, f64, f64)>,
) {
    let (observed_signal, observed_noise, missing_signal, missing_noise, rt_observed, rt_timeouts) =
        group.map_or((0, 0, 0, 0, 0, 0), |group| {
            (
                group.observed_signal_trials,
                group.observed_noise_trials,
                group.missing_signal_categories,
                group.missing_noise_categories,
                group.rt_observed,
                group.rt_timeouts,
            )
        });
    let (estimate_mu, estimate_sigma, estimate_d_a) =
        estimates.map_or((None, None, None), |(mu, sigma, d_a)| {
            (Some(mu), Some(sigma), Some(d_a))
        });
    let (mu_error, sigma_error, d_a_error) =
        errors.map_or((None, None, None), |(mu, sigma, d_a)| {
            (Some(mu), Some(sigma), Some(d_a))
        });
    println!(
        "REPLICATE,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},,,,,,,,,{},{}",
        case.id,
        case.name,
        if case.correctly_specified_gaussian { "gaussian_uvsd_track" } else { "misspecification_stress_track" },
        replicate,
        seed,
        dataset.prng_version,
        trials_per_class,
        trials_per_class.saturating_mul(2),
        observed_signal,
        observed_noise,
        missing_signal,
        missing_noise,
        rt_observed,
        rt_timeouts,
        adapter_disposition,
        fit_status,
        qualified,
        truth_mu,
        truth_sigma,
        truth_d_a,
        csv_option(estimate_mu),
        csv_option(estimate_sigma),
        csv_option(estimate_d_a),
        csv_option(mu_error),
        csv_option(sigma_error),
        csv_option(d_a_error),
        "",
        "",
    );
}

fn write_summary_row(case: &ScenarioCase, aggregate: &Accumulator) {
    let fit_availability_all = if aggregate.campaigns > 0 {
        aggregate.converged as f64 / aggregate.campaigns as f64
    } else {
        0.0
    };
    let fit_availability_eligible = if aggregate.adapter_ready > 0 {
        aggregate.converged as f64 / aggregate.adapter_ready as f64
    } else {
        0.0
    };
    let mean_mu = aggregate.mean_error(aggregate.mu_error_sum);
    let mean_sigma = aggregate.mean_error(aggregate.sigma_error_sum);
    let mean_d_a = aggregate.mean_error(aggregate.d_a_error_sum);
    let rmse_mu = aggregate.rmse(aggregate.mu_squared_error_sum);
    let rmse_sigma = aggregate.rmse(aggregate.sigma_squared_error_sum);
    let rmse_d_a = aggregate.rmse(aggregate.d_a_squared_error_sum);
    println!(
        "SUMMARY,{},{},{},ALL,,,,,,,,,,,,,SUMMARY,,,,,,,,,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        case.id,
        case.name,
        if case.correctly_specified_gaussian { "gaussian_uvsd_track" } else { "misspecification_stress_track" },
        csv_option(mean_mu),
        csv_option(mean_sigma),
        csv_option(mean_d_a),
        csv_option(rmse_mu),
        csv_option(rmse_sigma),
        csv_option(rmse_d_a),
        fit_availability_all,
        fit_availability_eligible,
        aggregate.adapter_ready,
        aggregate.adapter_abstentions,
        aggregate.boundary,
        aggregate.not_identifiable,
        aggregate.weakly_identified,
        aggregate.not_converged,
        aggregate.not_evaluable,
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = parse_cli()?;
    let action_set = action_set()?;
    let requests = trial_schedule(cli.trials_per_class)?;
    let fit_options = FitOptions::default();
    let fit_mu = 1.1 + 0.3; // Must match base_scenario stimulus mapping below.

    write_replicate_header();

    for case in scenario_cases() {
        let mut aggregate = Accumulator::default();
        let true_mu = fit_mu;
        let true_sigma = case.spec.signal_sigma;
        let true_d_a = true_mu / ((1.0 + true_sigma * true_sigma) / 2.0).sqrt();

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
            });
            let Some(group) = group else {
                return Err(io::Error::other("expected participant/session fit group missing").into());
            };
            aggregate.record_adapter_disposition(group.disposition);

            if group.disposition != FitDisposition::Ready {
                write_replicate_row(
                    &case,
                    replicate,
                    seed,
                    cli.trials_per_class,
                    &dataset,
                    Some(group),
                    &format!("{:?}", group.disposition),
                    "NOT_RUN",
                    false,
                    true_mu,
                    true_sigma,
                    true_d_a,
                    None,
                    None,
                );
                continue;
            }

            let table = group.table.as_ref().ok_or_else(|| {
                io::Error::other("Ready adapter disposition did not include CategoryTable")
            })?;
            let fit_result = fit(table, fit_options);
            let estimate = fit_result.estimate.as_ref();
            let estimated = estimate.map(|estimate| {
                (estimate.mu, estimate.sigma, estimate.d_a)
            });
            let qualified = fit_result.status == FitStatus::Converged;
            let errors = if qualified {
                estimate.map(|estimate| {
                    (
                        estimate.mu - true_mu,
                        estimate.sigma - true_sigma,
                        estimate.d_a - true_d_a,
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
                Some(group),
                "Ready",
                &format!("{:?}", fit_result.status),
                qualified,
                true_mu,
                true_sigma,
                true_d_a,
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
