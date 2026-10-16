//! Deterministic synthetic fixtures for parameter recovery and kinetic-model discrimination.
//!
//! This module is a falsification harness, not an enzyme predictor. Its numerical outputs only
//! establish that the toy implementation behaves as specified on known synthetic truth. In
//! particular, the finite-difference sensitivity score is not a profile-likelihood analysis.
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::fs;
use std::path::Path;

const RECOVERY_SUBSTRATES: &[f64] = &[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0];
const RECOVERY_NOISE: &[f64] = &[0.01, -0.02, 0.01, -0.015, 0.005, -0.01, 0.02, -0.005];
const INHIBITION_SUBSTRATES: &[f64] = &[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
const INHIBITION_NOISE: &[f64] = &[0.01, -0.02, 0.01, -0.015, 0.005, -0.01, 0.02, -0.005, 0.015];
const EXISTING_SUBSTRATES: &[f64] = &[0.1, 0.25, 0.5, 1.0];
const CANDIDATE_SUBSTRATES: &[f64] = &[2.0, 5.0, 10.0, 20.0, 50.0, 100.0];
const OBSERVATION_SD: f64 = 2.0;
const RELATIVE_RECOVERY_LIMIT: f64 = 0.02;

#[derive(Clone, Copy, Debug)]
struct Observation {
    substrate: f64,
    rate: f64,
}

#[derive(Clone, Copy, Debug)]
struct MichaelisMentenFit {
    vmax: f64,
    km: f64,
    rmse: f64,
    sse: f64,
}

#[derive(Clone, Copy, Debug)]
struct SubstrateInhibitionFit {
    vmax: f64,
    km: f64,
    ki: f64,
    rmse: f64,
    sse: f64,
}

fn michaelis_menten(substrate: f64, vmax: f64, km: f64) -> f64 {
    vmax * substrate / (km + substrate)
}

fn substrate_inhibition(substrate: f64, vmax: f64, km: f64, ki: f64) -> f64 {
    vmax * substrate / (km + substrate + substrate * substrate / ki)
}

fn build_recovery_observations() -> Vec<Observation> {
    RECOVERY_SUBSTRATES
        .iter()
        .zip(RECOVERY_NOISE.iter())
        .map(|(&substrate, &noise)| Observation {
            substrate,
            rate: michaelis_menten(substrate, 100.0, 2.0) + noise,
        })
        .collect()
}

fn fit_michaelis_menten_grid(observations: &[Observation]) -> MichaelisMentenFit {
    let mut best = MichaelisMentenFit {
        vmax: f64::NAN,
        km: f64::NAN,
        rmse: f64::INFINITY,
        sse: f64::INFINITY,
    };

    // Integer-indexed grids avoid accumulated floating-point loop drift and include (100, 2).
    for vmax_index in 0..=80 {
        let vmax = 80.0 + vmax_index as f64 * 0.5;
        for km_index in 0..=180 {
            let km = 0.5 + km_index as f64 * 0.025;
            let squared_error = observations
                .iter()
                .map(|obs| {
                    let residual = obs.rate - michaelis_menten(obs.substrate, vmax, km);
                    residual * residual
                })
                .sum::<f64>();
            let rmse = (squared_error / observations.len() as f64).sqrt();
            if squared_error < best.sse {
                best = MichaelisMentenFit {
                    vmax,
                    km,
                    rmse,
                    sse: squared_error,
                };
            }
        }
    }
    best
}

fn build_inhibition_observations() -> Vec<Observation> {
    INHIBITION_SUBSTRATES
        .iter()
        .zip(INHIBITION_NOISE.iter())
        .map(|(&substrate, &noise)| Observation {
            substrate,
            rate: substrate_inhibition(substrate, 100.0, 2.0, 20.0) + noise,
        })
        .collect()
}

fn fit_substrate_inhibition_grid(observations: &[Observation]) -> SubstrateInhibitionFit {
    let mut best = SubstrateInhibitionFit {
        vmax: f64::NAN,
        km: f64::NAN,
        ki: f64::NAN,
        rmse: f64::INFINITY,
        sse: f64::INFINITY,
    };

    // The fixture truth (100, 2, 20) is included in these fixed grids.
    for vmax_index in 0..=80 {
        let vmax = 80.0 + vmax_index as f64 * 0.5;
        for km_index in 0..=180 {
            let km = 0.5 + km_index as f64 * 0.025;
            for ki_index in 0..=75 {
                let ki = 5.0 + ki_index as f64;
                let squared_error = observations
                    .iter()
                    .map(|obs| {
                        let residual =
                            obs.rate - substrate_inhibition(obs.substrate, vmax, km, ki);
                        residual * residual
                    })
                    .sum::<f64>();
                if squared_error < best.sse {
                    best = SubstrateInhibitionFit {
                        vmax,
                        km,
                        ki,
                        rmse: (squared_error / observations.len() as f64).sqrt(),
                        sse: squared_error,
                    };
                }
            }
        }
    }
    best
}

fn bic(sse: f64, observation_count: usize, parameter_count: usize) -> f64 {
    let n = observation_count as f64;
    n * (sse / n).ln() + parameter_count as f64 * n.ln()
}

fn relative_error(estimated: f64, truth: f64) -> f64 {
    (estimated - truth).abs() / truth.abs()
}

fn build_report() -> Result<Value> {
    let observations = build_recovery_observations();
    let fit = fit_michaelis_menten_grid(&observations);
    let vmax_relative_error = relative_error(fit.vmax, 100.0);
    let km_relative_error = relative_error(fit.km, 2.0);

    ensure!(
        vmax_relative_error <= RELATIVE_RECOVERY_LIMIT,
        "synthetic Vmax recovery relative error {:.4} exceeds {:.2}%",
        vmax_relative_error,
        RELATIVE_RECOVERY_LIMIT * 100.0
    );
    ensure!(
        km_relative_error <= RELATIVE_RECOVERY_LIMIT,
        "synthetic Km recovery relative error {:.4} exceeds {:.2}%",
        km_relative_error,
        RELATIVE_RECOVERY_LIMIT * 100.0
    );

    // Model-family selection on two intentionally separable synthetic truths. BIC is used here
    // only as a fixture criterion; this does not establish real-data selection calibration.
    let mm_truth_mm_fit = fit_michaelis_menten_grid(&observations);
    let mm_truth_inhibition_fit = fit_substrate_inhibition_grid(&observations);
    let mm_truth_delta_bic = bic(mm_truth_mm_fit.sse, observations.len(), 2)
        - bic(mm_truth_inhibition_fit.sse, observations.len(), 3);
    ensure!(
        mm_truth_delta_bic < -10.0,
        "MM-truth fixture should favor Michaelis-Menten by delta BIC < -10, got {mm_truth_delta_bic}"
    );

    let inhibition_observations = build_inhibition_observations();
    let inhibition_truth_mm_fit = fit_michaelis_menten_grid(&inhibition_observations);
    let inhibition_truth_inhibition_fit =
        fit_substrate_inhibition_grid(&inhibition_observations);
    let inhibition_truth_delta_bic =
        bic(inhibition_truth_mm_fit.sse, inhibition_observations.len(), 2)
            - bic(
                inhibition_truth_inhibition_fit.sse,
                inhibition_observations.len(),
                3,
            );
    ensure!(
        inhibition_truth_delta_bic > 10.0,
        "substrate-inhibition truth fixture should favor substrate inhibition by delta BIC > 10, got {inhibition_truth_delta_bic}"
    );

    // Model discrimination uses two frozen hypotheses, not fitted or externally validated models.
    let mm_vmax = 100.0;
    let mm_km = 2.0;
    let inhibition_vmax = 100.0;
    let inhibition_km = 2.0;
    let inhibition_ki = 20.0;
    let mut candidate_scores = Vec::new();

    for &substrate in CANDIDATE_SUBSTRATES {
        let mm_rate = michaelis_menten(substrate, mm_vmax, mm_km);
        let inhibition_rate =
            substrate_inhibition(substrate, inhibition_vmax, inhibition_km, inhibition_ki);
        let standardized_separation = (mm_rate - inhibition_rate).abs() / OBSERVATION_SD;
        candidate_scores.push(json!({
            "substrate": substrate,
            "michaelis_menten_rate": mm_rate,
            "substrate_inhibition_rate": inhibition_rate,
            "measurement_sd_assumption": OBSERVATION_SD,
            "standardized_separation": standardized_separation,
            "squared_separation_score": standardized_separation * standardized_separation
        }));
    }

    let selected = candidate_scores
        .iter()
        .max_by(|left, right| {
            left["squared_separation_score"]
                .as_f64()
                .unwrap_or(f64::NEG_INFINITY)
                .total_cmp(
                    &right["squared_separation_score"]
                        .as_f64()
                        .unwrap_or(f64::NEG_INFINITY),
                )
        })
        .context("candidate condition list must not be empty")?;
    let selected_substrate = selected["substrate"]
        .as_f64()
        .context("selected substrate is not numeric")?;
    let selected_score = selected["squared_separation_score"]
        .as_f64()
        .context("selected score is not numeric")?;
    let random_mean_score = candidate_scores
        .iter()
        .map(|item| item["squared_separation_score"].as_f64().unwrap_or(0.0))
        .sum::<f64>()
        / candidate_scores.len() as f64;

    ensure!(
        selected_substrate == 100.0,
        "synthetic discrimination fixture expected 100.0 to maximize fixed-hypothesis separation, got {selected_substrate}"
    );
    ensure!(
        selected_score > random_mean_score,
        "selected design did not outperform the uniform-random expected discrimination score"
    );

    // Local finite-difference sensitivity is an intentionally modest design proxy, not a
    // practical-identifiability guarantee. Compare candidates not included in existing conditions.
    let km_step = 0.05;
    let mut sensitivity_scores = Vec::new();
    for &substrate in CANDIDATE_SUBSTRATES {
        ensure!(
            !EXISTING_SUBSTRATES.contains(&substrate),
            "candidate condition {substrate} duplicates an existing condition"
        );
        let upper = michaelis_menten(substrate, 100.0, 2.0 + km_step);
        let lower = michaelis_menten(substrate, 100.0, 2.0 - km_step);
        let sensitivity = ((upper - lower) / (2.0 * km_step)).abs();
        let normalized_sensitivity = sensitivity / OBSERVATION_SD;
        sensitivity_scores.push(json!({
            "substrate": substrate,
            "absolute_local_d_rate_d_km": sensitivity,
            "measurement_sd_assumption": OBSERVATION_SD,
            "normalized_local_sensitivity": normalized_sensitivity
        }));
    }
    let sensitivity_selected = sensitivity_scores
        .iter()
        .max_by(|left, right| {
            left["normalized_local_sensitivity"]
                .as_f64()
                .unwrap_or(f64::NEG_INFINITY)
                .total_cmp(
                    &right["normalized_local_sensitivity"]
                        .as_f64()
                        .unwrap_or(f64::NEG_INFINITY),
                )
        })
        .context("candidate condition list must not be empty")?;
    let sensitivity_selected_substrate = sensitivity_selected["substrate"]
        .as_f64()
        .context("selected sensitivity substrate is not numeric")?;

    ensure!(
        sensitivity_selected_substrate == 2.0,
        "synthetic local-sensitivity fixture expected substrate 2.0 to be most informative for Km, got {sensitivity_selected_substrate}"
    );

    Ok(json!({
        "schema_version": 1,
        "benchmark_id": "enzyme-model-falsification-synthetic-v0",
        "status": "synthetic_fixture_passed",
        "scope": "Deterministic software fixture only; not a biochemical validation, real-assay reproduction, calibrated uncertainty estimate, profile-likelihood analysis, or prediction benchmark.",
        "parameter_recovery": {
            "truth_model": "michaelis_menten",
            "truth": {"vmax": 100.0, "km": 2.0},
            "observation_count": observations.len(),
            "substrate_conditions": RECOVERY_SUBSTRATES,
            "deterministic_residuals_added_to_truth": RECOVERY_NOISE,
            "fit_method": "exhaustive_grid_search",
            "grid": {
                "vmax_min": 80.0, "vmax_max": 120.0, "vmax_step": 0.5,
                "km_min": 0.5, "km_max": 5.0, "km_step": 0.025
            },
            "estimated": {"vmax": fit.vmax, "km": fit.km, "rmse": fit.rmse},
            "relative_error": {
                "vmax": vmax_relative_error,
                "km": km_relative_error
            },
            "prespecified_max_relative_error": RELATIVE_RECOVERY_LIMIT,
            "fixture_gate_passed": true
        },
        "model_selection_falsification": {
            "method": "Bayesian Information Criterion (BIC), lower is preferred",
            "delta_definition": "BIC(Michaelis-Menten) - BIC(substrate inhibition); negative favors Michaelis-Menten, positive favors substrate inhibition",
            "decisive_fixture_gate": "absolute delta BIC > 10 in both known-truth separable scenarios",
            "michaelis_menten_truth": {
                "truth": {"vmax": 100.0, "km": 2.0},
                "observation_count": observations.len(),
                "michaelis_menten_fit": {"vmax": mm_truth_mm_fit.vmax, "km": mm_truth_mm_fit.km, "rmse": mm_truth_mm_fit.rmse, "bic": bic(mm_truth_mm_fit.sse, observations.len(), 2)},
                "substrate_inhibition_fit": {"vmax": mm_truth_inhibition_fit.vmax, "km": mm_truth_inhibition_fit.km, "ki": mm_truth_inhibition_fit.ki, "rmse": mm_truth_inhibition_fit.rmse, "bic": bic(mm_truth_inhibition_fit.sse, observations.len(), 3)},
                "delta_bic_mm_minus_inhibition": mm_truth_delta_bic,
                "selected_model": "michaelis_menten"
            },
            "substrate_inhibition_truth": {
                "truth": {"vmax": 100.0, "km": 2.0, "ki": 20.0},
                "observation_count": inhibition_observations.len(),
                "michaelis_menten_fit": {"vmax": inhibition_truth_mm_fit.vmax, "km": inhibition_truth_mm_fit.km, "rmse": inhibition_truth_mm_fit.rmse, "bic": bic(inhibition_truth_mm_fit.sse, inhibition_observations.len(), 2)},
                "substrate_inhibition_fit": {"vmax": inhibition_truth_inhibition_fit.vmax, "km": inhibition_truth_inhibition_fit.km, "ki": inhibition_truth_inhibition_fit.ki, "rmse": inhibition_truth_inhibition_fit.rmse, "bic": bic(inhibition_truth_inhibition_fit.sse, inhibition_observations.len(), 3)},
                "delta_bic_mm_minus_inhibition": inhibition_truth_delta_bic,
                "selected_model": "substrate_inhibition"
            },
            "fixture_gate_passed": true,
            "warning": "Two deliberately separable in-sample synthetic cases; not model-selection calibration or evidence for a real enzyme mechanism."
        },
        "model_discrimination_design": {
            "objective": "maximize squared prediction separation normalized by a fixed assumed measurement SD",
            "existing_conditions": EXISTING_SUBSTRATES,
            "candidate_conditions": candidate_scores.iter().map(|item| item["substrate"].clone()).collect::<Vec<_>>(),
            "fixed_hypotheses": [
                {"model": "michaelis_menten", "vmax": mm_vmax, "km": mm_km},
                {"model": "substrate_inhibition", "vmax": inhibition_vmax, "km": inhibition_km, "ki": inhibition_ki}
            ],
            "measurement_sd_assumption": OBSERVATION_SD,
            "scores": candidate_scores,
            "selected_condition": selected_substrate,
            "selected_squared_separation_score": selected_score,
            "uniform_random_expected_score": random_mean_score,
            "gain_over_uniform_random_expected_score": selected_score / random_mean_score
        },
        "km_design_proxy": {
            "objective": "maximize local finite-difference absolute sensitivity of the Michaelis-Menten rate to Km, normalized by an assumed measurement SD",
            "km_truth": 2.0,
            "finite_difference_step": km_step,
            "scores": sensitivity_scores,
            "selected_condition": sensitivity_selected_substrate,
            "warning": "Local sensitivity is not a profile likelihood, confidence interval, global identifiability proof, or guarantee of parameter recovery."
        },
        "next_scientific_gates": [
            "replicate across preregistered noise levels and seeds",
            "include non-identifiable and misspecified synthetic systems",
            "evaluate interval coverage and model-selection calibration",
            "compare active design with random and local-sensitivity baselines at matched measurement budgets",
            "independently replay a rights-cleared real assay before any real-data claim"
        ]
    }))
}

pub fn run(output: &Path) -> Result<()> {
    let report = build_report()?;
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create report directory {}", parent.display()))?;
        }
    }
    let bytes = serde_json::to_vec_pretty(&report).context("serialize falsification report")?;
    fs::write(output, &bytes)
        .with_context(|| format!("write synthetic falsification report {}", output.display()))?;
    println!(
        "Synthetic enzyme-model falsification fixture passed: output={}, bytes={}, scope=synthetic-only",
        output.display(),
        bytes.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_michaelis_menten_parameters_are_recovered_within_gate() {
        let observations = build_recovery_observations();
        let fit = fit_michaelis_menten_grid(&observations);
        assert!(relative_error(fit.vmax, 100.0) <= RELATIVE_RECOVERY_LIMIT);
        assert!(relative_error(fit.km, 2.0) <= RELATIVE_RECOVERY_LIMIT);
        assert!(fit.rmse.is_finite());
    }

    #[test]
    fn information_criterion_selects_each_generating_family_on_separable_truth_fixtures() {
        let report = build_report().unwrap();
        assert_eq!(
            report["model_selection_falsification"]["michaelis_menten_truth"]["selected_model"],
            "michaelis_menten"
        );
        assert_eq!(
            report["model_selection_falsification"]["substrate_inhibition_truth"]["selected_model"],
            "substrate_inhibition"
        );
        assert!(
            report["model_selection_falsification"]["michaelis_menten_truth"]["delta_bic_mm_minus_inhibition"]
                .as_f64().unwrap() < -10.0
        );
        assert!(
            report["model_selection_falsification"]["substrate_inhibition_truth"]["delta_bic_mm_minus_inhibition"]
                .as_f64().unwrap() > 10.0
        );
    }

    #[test]
    fn discrimination_design_selects_condition_with_maximum_fixed_hypothesis_separation() {
        let report = build_report().unwrap();
        assert_eq!(
            report["model_discrimination_design"]["selected_condition"]
                .as_f64()
                .unwrap(),
            100.0
        );
        let selected = report["model_discrimination_design"]["selected_squared_separation_score"]
            .as_f64()
            .unwrap();
        let random_mean = report["model_discrimination_design"]["uniform_random_expected_score"]
            .as_f64()
            .unwrap();
        assert!(selected > random_mean);
    }

    #[test]
    fn local_km_sensitivity_proxy_selects_the_expected_candidate_without_claiming_identifiability() {
        let report = build_report().unwrap();
        assert_eq!(report["km_design_proxy"]["selected_condition"].as_f64().unwrap(), 2.0);
        assert!(report["km_design_proxy"]["warning"].as_str().unwrap().contains("not a profile likelihood"));
    }

    #[test]
    fn report_explicitly_limits_claims_to_synthetic_fixture() {
        let report = build_report().unwrap();
        assert_eq!(report["status"], "synthetic_fixture_passed");
        assert!(report["scope"].as_str().unwrap().contains("not a biochemical validation"));
        assert!(report["next_scientific_gates"].as_array().unwrap().len() >= 4);
    }
}
