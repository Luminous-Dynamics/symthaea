//! Paired synthetic benchmark for budgeted enzyme experiment design.
//!
//! The benchmark compares four policies under the same frozen initial observations and assay
//! budget. It separates fixed-hypothesis model discrimination from local parameter sensitivity.
//! All outcomes are exploratory synthetic evidence, not a real-assay validation or a claim that
//! any strategy is universally optimal.

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[derive(Clone, Copy, Debug)]
struct Observation {
    substrate: f64,
    rate: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ModelKind {
    MichaelisMenten,
    SubstrateInhibition,
}

impl ModelKind {
    fn label(self) -> &'static str {
        match self {
            Self::MichaelisMenten => "michaelis_menten",
            Self::SubstrateInhibition => "substrate_inhibition",
        }
    }

    fn parameter_count(self) -> usize {
        match self {
            Self::MichaelisMenten => 2,
            Self::SubstrateInhibition => 3,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ModelFit {
    kind: ModelKind,
    vmax: f64,
    km: f64,
    ki: Option<f64>,
    sse: f64,
    rmse: f64,
}

#[derive(Clone, Copy, Debug)]
struct Scenario {
    label: &'static str,
    truth: ModelKind,
    vmax: f64,
    km: f64,
    ki: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Strategy {
    ModelDiscrimination,
    KmSensitivity,
    UniformRandom,
    PrecommittedLogGrid,
}

impl Strategy {
    fn all() -> [Self; 4] {
        [
            Self::ModelDiscrimination,
            Self::KmSensitivity,
            Self::UniformRandom,
            Self::PrecommittedLogGrid,
        ]
    }

    fn label(self) -> &'static str {
        match self {
            Self::ModelDiscrimination => "adaptive_model_discrimination",
            Self::KmSensitivity => "adaptive_km_sensitivity",
            Self::UniformRandom => "uniform_random",
            Self::PrecommittedLogGrid => "precommitted_log_grid",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Selection {
    substrate: f64,
    score: f64,
    objective: &'static str,
}

const INITIAL_SUBSTRATES: &[f64] = &[0.5, 1.0, 2.0];
const CANDIDATE_SUBSTRATES: &[f64] = &[
    0.1, 0.2, 0.35, 0.7, 1.4, 2.8, 5.6, 11.2, 22.4, 44.8, 89.6, 179.2,
];
const NOISE_SDS: &[f64] = &[0.5, 3.0];
const SEEDS: &[u64] = &[1, 2, 3, 4];
const ADDED_MEASUREMENT_BUDGET: usize = 4;

fn scenarios() -> [Scenario; 2] {
    [
        Scenario {
            label: "michaelis_menten_truth",
            truth: ModelKind::MichaelisMenten,
            vmax: 100.0,
            km: 2.0,
            ki: 20.0,
        },
        Scenario {
            label: "substrate_inhibition_truth",
            truth: ModelKind::SubstrateInhibition,
            vmax: 100.0,
            km: 2.0,
            ki: 20.0,
        },
    ]
}

fn michaelis_menten(substrate: f64, vmax: f64, km: f64) -> f64 {
    vmax * substrate / (km + substrate)
}

fn substrate_inhibition(substrate: f64, vmax: f64, km: f64, ki: f64) -> f64 {
    vmax * substrate / (km + substrate + substrate * substrate / ki)
}

fn predict(kind: ModelKind, substrate: f64, vmax: f64, km: f64, ki: f64) -> f64 {
    match kind {
        ModelKind::MichaelisMenten => michaelis_menten(substrate, vmax, km),
        ModelKind::SubstrateInhibition => substrate_inhibition(substrate, vmax, km, ki),
    }
}

/// SplitMix64-derived deterministic bounded-uniform noise. Identical seed/condition inputs yield
/// identical synthetic measurements across policies, enabling paired comparisons at matched budget.
fn synthetic_noise(seed: u64, condition_index: usize, sigma: f64) -> f64 {
    let mut z = seed
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add((condition_index as u64).wrapping_mul(0xBF58476D1CE4E5B9));
    z = z.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    let unit = ((z >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64));
    (2.0 * unit - 1.0) * 3.0_f64.sqrt() * sigma
}

fn simulated_rate(scenario: Scenario, substrate: f64, seed: u64, noise_index: usize, sigma: f64) -> f64 {
    predict(
        scenario.truth,
        substrate,
        scenario.vmax,
        scenario.km,
        scenario.ki,
    ) + synthetic_noise(seed, noise_index, sigma)
}

fn sse(observations: &[Observation], kind: ModelKind, vmax: f64, km: f64, ki: f64) -> f64 {
    observations
        .iter()
        .map(|obs| {
            let residual = obs.rate - predict(kind, obs.substrate, vmax, km, ki);
            residual * residual
        })
        .sum()
}

fn fit_grid(observations: &[Observation], kind: ModelKind) -> ModelFit {
    let mut best = ModelFit {
        kind,
        vmax: f64::NAN,
        km: f64::NAN,
        ki: None,
        sse: f64::INFINITY,
        rmse: f64::INFINITY,
    };

    for vmax_index in 0..=40 {
        let vmax = 60.0 + vmax_index as f64 * 2.0;
        for km_index in 1..=40 {
            let km = km_index as f64 * 0.25;
            if kind == ModelKind::MichaelisMenten {
                let squared_error = sse(observations, kind, vmax, km, 20.0);
                if squared_error < best.sse {
                    best = ModelFit {
                        kind,
                        vmax,
                        km,
                        ki: None,
                        sse: squared_error,
                        rmse: (squared_error / observations.len() as f64).sqrt(),
                    };
                }
            } else {
                for ki_index in 1..=20 {
                    let ki = ki_index as f64 * 5.0;
                    let squared_error = sse(observations, kind, vmax, km, ki);
                    if squared_error < best.sse {
                        best = ModelFit {
                            kind,
                            vmax,
                            km,
                            ki: Some(ki),
                            sse: squared_error,
                            rmse: (squared_error / observations.len() as f64).sqrt(),
                        };
                    }
                }
            }
        }
    }
    best
}

fn bic(fit: ModelFit, n_observations: usize) -> f64 {
    let n = n_observations as f64;
    n * (fit.sse / n).max(1e-12).ln() + fit.kind.parameter_count() as f64 * n.ln()
}

fn all_fits(observations: &[Observation]) -> (ModelFit, ModelFit) {
    (
        fit_grid(observations, ModelKind::MichaelisMenten),
        fit_grid(observations, ModelKind::SubstrateInhibition),
    )
}

fn select_condition(
    strategy: Strategy,
    available: &[f64],
    mm: ModelFit,
    inhibition: ModelFit,
    sigma: f64,
    seed: u64,
    step: usize,
) -> Result<Selection> {
    ensure!(!available.is_empty(), "candidate substrate pool is exhausted");

    if strategy == Strategy::UniformRandom {
        // Deterministic pseudo-random selection. Candidate indices are paired by seed/step and
        // do not depend on the simulated truth or observed rate.
        let mut z = seed
            .wrapping_add((step as u64).wrapping_mul(0x9E3779B97F4A7C15))
            .wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 30)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        let index = (z as usize) % available.len();
        return Ok(Selection {
            substrate: available[index],
            score: 0.0,
            objective: "deterministic_uniform_random_control",
        });
    }

    if strategy == Strategy::PrecommittedLogGrid {
        // Select the most widely spaced remaining condition relative to the fixed candidate
        // ordering; this sequence is declared before observations are generated.
        const SEQUENCE: [usize; 12] = [0, 4, 8, 11, 2, 6, 10, 1, 5, 9, 3, 7];
        let target_index = SEQUENCE[step % SEQUENCE.len()];
        let preferred = CANDIDATE_SUBSTRATES[target_index];
        let chosen = available
            .iter()
            .copied()
            .min_by(|a, b| (*a - preferred).abs().total_cmp(&(*b - preferred).abs()))
            .context("candidate pool must contain the precommitted condition")?;
        return Ok(Selection {
            substrate: chosen,
            score: 0.0,
            objective: "precommitted_log_spaced_sequence",
        });
    }

    let mut scored = Vec::with_capacity(available.len());
    for &substrate in available {
        let score = match strategy {
            Strategy::ModelDiscrimination => {
                // Objective A: distinguish the two currently fitted model hypotheses.
                (predict(
                    ModelKind::MichaelisMenten,
                    substrate,
                    mm.vmax,
                    mm.km,
                    20.0,
                ) - predict(
                    ModelKind::SubstrateInhibition,
                    substrate,
                    inhibition.vmax,
                    inhibition.km,
                    inhibition.ki.unwrap_or(20.0),
                ))
                .abs()
                    / sigma
            }
            Strategy::KmSensitivity => {
                // Objective B: estimate Km well conditional on the Michaelis-Menten family.
                // A finite-difference local sensitivity is a proxy, not a profile likelihood.
                let step_size = (mm.km * 0.05).clamp(0.01, 0.2);
                let upper = michaelis_menten(substrate, mm.vmax, mm.km + step_size);
                let lower_km = (mm.km - step_size).max(0.001);
                let lower = michaelis_menten(substrate, mm.vmax, lower_km);
                ((upper - lower) / (mm.km + step_size - lower_km)).abs() / sigma
            }
            Strategy::UniformRandom | Strategy::PrecommittedLogGrid => unreachable!(),
        };
        scored.push((substrate, score));
    }

    // Ties resolve to the lower substrate concentration because the candidate list is frozen
    // in ascending order and max_by is given a deterministic total-order comparison.
    let (substrate, score) = scored
        .into_iter()
        .max_by(|(left_s, left_score), (right_s, right_score)| {
            left_score
                .total_cmp(right_score)
                .then_with(|| right_s.total_cmp(left_s))
        })
        .context("candidate score list must not be empty")?;

    Ok(Selection {
        substrate,
        score,
        objective: match strategy {
            Strategy::ModelDiscrimination => "fitted_model_prediction_gap_over_assumed_noise_sd",
            Strategy::KmSensitivity => "local_abs_d_rate_d_km_over_assumed_noise_sd",
            Strategy::UniformRandom => unreachable!(),
            Strategy::PrecommittedLogGrid => unreachable!(),
        },
    })
}

fn run_campaign(
    scenario: Scenario,
    strategy: Strategy,
    sigma: f64,
    seed: u64,
) -> Result<Value> {
    let mut observations = Vec::new();
    let mut history = Vec::new();
    for (index, &substrate) in INITIAL_SUBSTRATES.iter().enumerate() {
        observations.push(Observation {
            substrate,
            rate: simulated_rate(scenario, substrate, seed, 100 + index, sigma),
        });
    }

    let mut remaining: Vec<f64> = CANDIDATE_SUBSTRATES
        .iter()
        .copied()
        .filter(|candidate| !INITIAL_SUBSTRATES.contains(candidate))
        .collect();

    for step in 0..ADDED_MEASUREMENT_BUDGET {
        let (mm, inhibition) = all_fits(&observations);
        let selection = select_condition(
            strategy,
            &remaining,
            mm,
            inhibition,
            sigma,
            seed,
            step,
        )?;

        // Use the same seed + condition-index observation for all policies. Thus, if two policies
        // choose the same condition within a campaign, they receive the same noisy outcome.
        let condition_index = CANDIDATE_SUBSTRATES
            .iter()
            .position(|candidate| *candidate == selection.substrate)
            .context("selected substrate must belong to frozen candidate list")?;
        let measured_rate = simulated_rate(
            scenario,
            selection.substrate,
            seed,
            condition_index,
            sigma,
        );
        history.push(json!({
            "step": step + 1,
            "substrate": selection.substrate,
            "objective": selection.objective,
            "selection_score": selection.score,
            "measured_rate": measured_rate,
            "available_candidate_count_before_selection": remaining.len()
        }));
        observations.push(Observation {
            substrate: selection.substrate,
            rate: measured_rate,
        });
        remaining.retain(|candidate| *candidate != selection.substrate);
    }

    let (mm, inhibition) = all_fits(&observations);
    let delta_bic = bic(mm, observations.len()) - bic(inhibition, observations.len());
    let selected_model = if delta_bic < 0.0 {
        ModelKind::MichaelisMenten
    } else {
        ModelKind::SubstrateInhibition
    };
    let chosen_fit = if selected_model == ModelKind::MichaelisMenten {
        mm
    } else {
        inhibition
    };

    let selected_model_correct = selected_model == scenario.truth;
    let relative_vmax_error = (chosen_fit.vmax - scenario.vmax).abs() / scenario.vmax;
    let relative_km_error = (chosen_fit.km - scenario.km).abs() / scenario.km;
    let relative_ki_error = match (scenario.truth, chosen_fit.ki) {
        (ModelKind::SubstrateInhibition, Some(ki)) => Some((ki - scenario.ki).abs() / scenario.ki),
        _ => None,
    };

    Ok(json!({
        "scenario": scenario.label,
        "truth_model": scenario.truth.label(),
        "strategy": strategy.label(),
        "seed": seed,
        "measurement_sd": sigma,
        "initial_observation_count": INITIAL_SUBSTRATES.len(),
        "added_measurement_budget": ADDED_MEASUREMENT_BUDGET,
        "final_observation_count": observations.len(),
        "initial_substrates": INITIAL_SUBSTRATES,
        "selected_substrates": history.iter().map(|item| item["substrate"].clone()).collect::<Vec<_>>(),
        "selection_history": history,
        "final_selection": {
            "delta_bic_mm_minus_inhibition": delta_bic,
            "selected_model": selected_model.label(),
            "selected_model_correct": selected_model_correct,
            "warning": "BIC selection is candidate-relative and assumes a declared observation likelihood; this score is exploratory under the synthetic noise model."
        },
        "fitted_candidates": {
            "michaelis_menten": {
                "vmax": mm.vmax,
                "km": mm.km,
                "rmse": mm.rmse,
                "bic": bic(mm, observations.len())
            },
            "substrate_inhibition": {
                "vmax": inhibition.vmax,
                "km": inhibition.km,
                "ki": inhibition.ki,
                "rmse": inhibition.rmse,
                "bic": bic(inhibition, observations.len())
            }
        },
        "selected_fit_relative_error": {
            "vmax": relative_vmax_error,
            "km": relative_km_error,
            "ki": relative_ki_error
        },
        "all_observations": observations.iter().map(|obs| json!({
            "substrate": obs.substrate,
            "rate": obs.rate
        })).collect::<Vec<_>>()
    }))
}

fn build_report() -> Result<Value> {
    let mut campaigns = Vec::new();
    let mut by_strategy: BTreeMap<&'static str, Vec<Value>> = BTreeMap::new();
    for scenario in scenarios() {
        for &sigma in NOISE_SDS {
            for &seed in SEEDS {
                for strategy in Strategy::all() {
                    let campaign = run_campaign(scenario, strategy, sigma, seed)?;
                    by_strategy
                        .entry(strategy.label())
                        .or_default()
                        .push(campaign.clone());
                    campaigns.push(campaign);
                }
            }
        }
    }

    let summary: Vec<Value> = Strategy::all()
        .iter()
        .map(|strategy| {
            let cases = by_strategy.get(strategy.label()).expect("strategy always has cases");
            let count = cases.len();
            let correct = cases.iter()
                .filter(|case| case["final_selection"]["selected_model_correct"].as_bool() == Some(true))
                .count();
            // Parameter errors are aggregated only when the chosen model family matches the
            // known synthetic generator. Wrong-family parameters are not treated as comparable
            // parameter estimates merely because their JSON fields share a name.
            let matched_cases: Vec<&Value> = cases.iter()
                .filter(|case| case["final_selection"]["selected_model_correct"].as_bool() == Some(true))
                .collect();
            let mean_vmax_error = if matched_cases.is_empty() {
                Value::Null
            } else {
                json!(matched_cases.iter()
                    .filter_map(|case| case["selected_fit_relative_error"]["vmax"].as_f64())
                    .sum::<f64>() / matched_cases.len() as f64)
            };
            let mean_km_error = if matched_cases.is_empty() {
                Value::Null
            } else {
                json!(matched_cases.iter()
                    .filter_map(|case| case["selected_fit_relative_error"]["km"].as_f64())
                    .sum::<f64>() / matched_cases.len() as f64)
            };
            let mean_ki_error_cases: Vec<f64> = matched_cases.iter()
                .filter_map(|case| case["selected_fit_relative_error"]["ki"].as_f64())
                .collect();
            let mean_ki_error = if mean_ki_error_cases.is_empty() {
                Value::Null
            } else {
                json!(mean_ki_error_cases.iter().sum::<f64>() / mean_ki_error_cases.len() as f64)
            };
            json!({
                "strategy": strategy.label(),
                "campaign_count": count,
                "matched_budget_per_campaign": {
                    "initial_observations": INITIAL_SUBSTRATES.len(),
                    "additional_measurements": ADDED_MEASUREMENT_BUDGET,
                    "total_observations": INITIAL_SUBSTRATES.len() + ADDED_MEASUREMENT_BUDGET
                },
                "correct_final_model_count": correct,
                "final_model_selection_accuracy": correct as f64 / count as f64,
                "parameter_error_case_count_with_correct_family": matched_cases.len(),
                "mean_relative_vmax_error_on_correctly_selected_family": mean_vmax_error,
                "mean_relative_km_error_on_correctly_selected_family": mean_km_error,
                "ki_error_case_count_on_correctly_selected_inhibition_family": mean_ki_error_cases.len(),
                "mean_relative_ki_error_on_correctly_selected_inhibition_family": mean_ki_error
            })
        })
        .collect();

    let expected_campaigns = scenarios().len() * NOISE_SDS.len() * SEEDS.len() * Strategy::all().len();
    ensure!(campaigns.len() == expected_campaigns, "campaign accounting mismatch");
    ensure!(
        campaigns.iter().all(|campaign| {
            campaign["selection_history"].as_array().map(|history| history.len()) == Some(ADDED_MEASUREMENT_BUDGET)
                && campaign["all_observations"].as_array().map(|rows| rows.len()) == Some(INITIAL_SUBSTRATES.len() + ADDED_MEASUREMENT_BUDGET)
        }),
        "campaign budgets or observation accounting mismatch"
    );

    Ok(json!({
        "schema_version": 1,
        "benchmark_id": "enzyme-budgeted-experiment-design-v0",
        "status": "exploratory_synthetic_campaigns_generated",
        "scope": "Paired synthetic design benchmark. It is not biochemical validation, a calibrated uncertainty estimator, or evidence of prospective laboratory utility.",
        "design": {
            "scenario_count": scenarios().len(),
            "scenarios": scenarios().iter().map(|s| json!({
                "label": s.label,
                "truth_model": s.truth.label(),
                "true_vmax": s.vmax,
                "true_km": s.km,
                "true_ki": if s.truth == ModelKind::SubstrateInhibition { json!(s.ki) } else { Value::Null }
            })).collect::<Vec<_>>(),
            "noise_levels": NOISE_SDS,
            "seeds": SEEDS,
            "strategy_count": Strategy::all().len(),
            "strategies": Strategy::all().iter().map(|s| s.label()).collect::<Vec<_>>(),
            "campaign_count": expected_campaigns,
            "initial_substrates": INITIAL_SUBSTRATES,
            "candidate_substrates": CANDIDATE_SUBSTRATES,
            "added_measurement_budget": ADDED_MEASUREMENT_BUDGET,
            "noise_model": "deterministic bounded uniform noise with declared SD; not assumed to represent real assay errors",
            "fit_method": "coarse exhaustive grid for deterministic comparator; not a production estimator",
            "fit_grid": {
                "vmax_min": 60.0, "vmax_max": 140.0, "vmax_step": 2.0,
                "km_min": 0.25, "km_max": 10.0, "km_step": 0.25,
                "ki_min": 5.0, "ki_max": 100.0, "ki_step": 5.0
            }
        },
        "objective_separation": {
            "model_discrimination": "adaptively select substrate maximizing the prediction gap between currently fitted Michaelis-Menten and substrate-inhibition hypotheses",
            "parameter_identifiability_proxy": "adaptively select substrate maximizing finite-difference |d rate / d Km| for the fitted Michaelis-Menten candidate",
            "controls": ["uniform_random", "precommitted_log_grid"],
            "warning": "The local Km sensitivity score is conditional on the Michaelis-Menten family, and prediction-gap maximization is not a full information-gain calculation. These objectives are benchmarked separately by design."
        },
        "summary_by_strategy": summary,
        "campaigns": campaigns,
        "next_gates": [
            "run at least 1000 independently seeded scenario instances per noise regime before any headline performance claim",
            "include parameter variation, weak-identifiability cases, candidate-set misspecification, and justified heteroscedastic/correlated error models",
            "compare policy utility across paired scenarios and report confidence intervals, not only averages",
            "add profile-likelihood interval width as a parameter-identifiability endpoint",
            "evaluate experiment choice under an explicit assay-cost/feasibility budget",
            "validate on independently reproduced, rights-cleared real assay series before any prospective-utility claim"
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
    let bytes = serde_json::to_vec_pretty(&report).context("serialize experiment-design report")?;
    fs::write(output, &bytes)
        .with_context(|| format!("write experiment-design report {}", output.display()))?;
    println!(
        "Synthetic enzyme experiment-design benchmark generated: output={}, bytes={}, campaigns={}, scope=exploratory-synthetic-only",
        output.display(),
        bytes.len(),
        report["design"]["campaign_count"].as_u64().unwrap_or(0)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinetic_equations_have_expected_values() {
        assert!((michaelis_menten(2.0, 100.0, 2.0) - 50.0).abs() < 1e-12);
        assert!((substrate_inhibition(2.0, 100.0, 2.0, 20.0) - 200.0 / 4.2).abs() < 1e-12);
    }

    #[test]
    fn strategies_are_distinct_and_budget_is_fixed() {
        let labels: Vec<_> = Strategy::all().iter().map(|strategy| strategy.label()).collect();
        assert_eq!(labels.len(), 4);
        assert_eq!(labels.iter().collect::<std::collections::BTreeSet<_>>().len(), 4);
        assert_eq!(ADDED_MEASUREMENT_BUDGET, 4);
        assert!(CANDIDATE_SUBSTRATES.iter().all(|s| !INITIAL_SUBSTRATES.contains(s)));
    }

    #[test]
    fn discrimination_and_parameter_sensitivity_choose_different_objectives() {
        let mm = ModelFit {
            kind: ModelKind::MichaelisMenten,
            vmax: 100.0,
            km: 2.0,
            ki: None,
            sse: 1.0,
            rmse: 0.5,
        };
        let inhibition = ModelFit {
            kind: ModelKind::SubstrateInhibition,
            vmax: 100.0,
            km: 2.0,
            ki: Some(20.0),
            sse: 1.0,
            rmse: 0.5,
        };
        let available: Vec<f64> = CANDIDATE_SUBSTRATES.iter().copied()
            .filter(|substrate| !INITIAL_SUBSTRATES.contains(substrate))
            .collect();
        let discrimination = select_condition(
            Strategy::ModelDiscrimination, &available, mm, inhibition, 1.0, 1, 0
        ).unwrap();
        let sensitivity = select_condition(
            Strategy::KmSensitivity, &available, mm, inhibition, 1.0, 1, 0
        ).unwrap();

        assert!((discrimination.substrate - 179.2).abs() < 1e-12);
        assert!((sensitivity.substrate - 2.8).abs() < 1e-12);
        assert_ne!(discrimination.substrate, sensitivity.substrate);
    }

    #[test]
    fn deterministic_noise_replays_exactly() {
        assert_eq!(synthetic_noise(3, 8, 0.5), synthetic_noise(3, 8, 0.5));
        assert_ne!(synthetic_noise(3, 8, 0.5), synthetic_noise(3, 9, 0.5));
    }
}
