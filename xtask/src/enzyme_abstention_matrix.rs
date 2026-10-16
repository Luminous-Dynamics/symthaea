//! Deterministic synthetic stress matrix for kinetic model selection and abstention.
//!
//! This is an exploratory software benchmark, not a validated statistical procedure. The
//! thresholds are preregistered policy fixtures; they require calibration against real assay
//! residuals and independent studies before any scientific decision-making use.

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModelKind {
    MichaelisMenten,
    SubstrateInhibition,
}

impl ModelKind {
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
struct Decision {
    selected: &'static str,
    reason: &'static str,
    delta_bic_mm_minus_inhibition: f64,
    fitted_prediction_gap_in_noise_sd: f64,
    best_candidate_rmse_in_noise_sd: f64,
}

#[derive(Clone, Copy, Debug)]
struct Scenario {
    stratum: &'static str,
    truth_family: &'static str,
    expected_decision: &'static str,
    vmax: f64,
    km: f64,
    ki: f64,
    hill_exponent: f64,
    substrate_set: &'static [f64],
}

const WIDE_SUBSTRATES: &[f64] = &[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0];
const LOW_SUBSTRATES: &[f64] = &[0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0];
const NOISE_SDS: &[f64] = &[0.25, 1.0, 5.0];
const SEEDS: &[u64] = &[1, 2, 3, 4, 5, 6, 7, 8];

const DECISIVE_DELTA_BIC: f64 = 10.0;
const MAX_NORMALIZED_RMSE_FOR_ADEQUACY: f64 = 3.0;
const MIN_PREDICTION_GAP_IN_NOISE_SDS: f64 = 1.0;

fn michaelis_menten(substrate: f64, vmax: f64, km: f64) -> f64 {
    vmax * substrate / (km + substrate)
}

fn substrate_inhibition(substrate: f64, vmax: f64, km: f64, ki: f64) -> f64 {
    vmax * substrate / (km + substrate + substrate * substrate / ki)
}

fn hill_rate(substrate: f64, vmax: f64, half_saturation: f64, exponent: f64) -> f64 {
    let s = substrate.powf(exponent);
    vmax * s / (half_saturation.powf(exponent) + s)
}

/// Stable, deterministic pseudo-noise. Uniform noise is scaled to have the requested standard
/// deviation; it is not intended as a biological error model.
fn synthetic_noise(seed: u64, index: usize, sigma: f64) -> f64 {
    let mut z = seed
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add((index as u64).wrapping_mul(0xBF58476D1CE4E5B9));
    z = z.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    let unit = ((z >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64));
    (2.0 * unit - 1.0) * 3.0_f64.sqrt() * sigma
}

fn scenarios() -> [Scenario; 4] {
    [
        Scenario {
            stratum: "well_separated_michaelis_menten",
            truth_family: "michaelis_menten",
            expected_decision: "michaelis_menten",
            vmax: 100.0,
            km: 2.0,
            ki: 20.0,
            hill_exponent: 1.0,
            substrate_set: WIDE_SUBSTRATES,
        },
        Scenario {
            stratum: "well_separated_substrate_inhibition",
            truth_family: "substrate_inhibition",
            expected_decision: "substrate_inhibition",
            vmax: 100.0,
            km: 2.0,
            ki: 20.0,
            hill_exponent: 1.0,
            substrate_set: WIDE_SUBSTRATES,
        },
        Scenario {
            stratum: "observationally_ambiguous_low_substrate",
            // Generate a curve that approaches Michaelis-Menten over the observed substrate
            // range. The system's target is to preserve ambiguity, not infer the hidden label.
            truth_family: "substrate_inhibition_weakly_identified",
            expected_decision: "abstain",
            vmax: 100.0,
            km: 2.0,
            ki: 10_000.0,
            hill_exponent: 1.0,
            substrate_set: LOW_SUBSTRATES,
        },
        Scenario {
            stratum: "misspecified_hill_truth",
            truth_family: "hill",
            // Neither candidate family is the generating model. Abstention is the intended
            // outcome whenever the candidate pair is demonstrably inadequate.
            expected_decision: "abstain",
            vmax: 100.0,
            km: 4.0,
            ki: 20.0,
            hill_exponent: 1.6,
            substrate_set: WIDE_SUBSTRATES,
        },
    ]
}

fn observations_for(scenario: Scenario, seed: u64, sigma: f64) -> Vec<Observation> {
    scenario
        .substrate_set
        .iter()
        .enumerate()
        .map(|(index, &substrate)| {
            let signal = match scenario.truth_family {
                "michaelis_menten" => michaelis_menten(substrate, scenario.vmax, scenario.km),
                "substrate_inhibition" | "substrate_inhibition_weakly_identified" => {
                    substrate_inhibition(substrate, scenario.vmax, scenario.km, scenario.ki)
                }
                "hill" => hill_rate(
                    substrate,
                    scenario.vmax,
                    scenario.km,
                    scenario.hill_exponent,
                ),
                _ => unreachable!("all synthetic strata have an explicit generator"),
            };
            Observation {
                substrate,
                rate: signal + synthetic_noise(seed, index, sigma),
            }
        })
        .collect()
}

fn residual_sse(observations: &[Observation], kind: ModelKind, vmax: f64, km: f64, ki: f64) -> f64 {
    observations
        .iter()
        .map(|obs| {
            let estimate = match kind {
                ModelKind::MichaelisMenten => michaelis_menten(obs.substrate, vmax, km),
                ModelKind::SubstrateInhibition => substrate_inhibition(obs.substrate, vmax, km, ki),
            };
            let residual = obs.rate - estimate;
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
            match kind {
                ModelKind::MichaelisMenten => {
                    let sse = residual_sse(observations, kind, vmax, km, 20.0);
                    if sse < best.sse {
                        best = ModelFit {
                            kind,
                            vmax,
                            km,
                            ki: None,
                            sse,
                            rmse: (sse / observations.len() as f64).sqrt(),
                        };
                    }
                }
                ModelKind::SubstrateInhibition => {
                    for ki_index in 1..=20 {
                        let ki = ki_index as f64 * 5.0;
                        let sse = residual_sse(observations, kind, vmax, km, ki);
                        if sse < best.sse {
                            best = ModelFit {
                                kind,
                                vmax,
                                km,
                                ki: Some(ki),
                                sse,
                                rmse: (sse / observations.len() as f64).sqrt(),
                            };
                        }
                    }
                }
            }
        }
    }
    best
}

fn bic(fit: ModelFit, observation_count: usize) -> f64 {
    let n = observation_count as f64;
    // The epsilon preserves finite report values if a deterministic fixture is an exact fit.
    n * (fit.sse / n).max(1e-12).ln() + fit.kind.parameter_count() as f64 * n.ln()
}

fn predict(fit: ModelFit, substrate: f64) -> f64 {
    match fit.kind {
        ModelKind::MichaelisMenten => michaelis_menten(substrate, fit.vmax, fit.km),
        ModelKind::SubstrateInhibition => {
            substrate_inhibition(substrate, fit.vmax, fit.km, fit.ki.unwrap_or(20.0))
        }
    }
}

fn decide(observations: &[Observation], mm: ModelFit, inhibition: ModelFit, sigma: f64) -> Decision {
    let delta_bic = bic(mm, observations.len()) - bic(inhibition, observations.len());
    let prediction_gap = (observations
        .iter()
        .map(|obs| {
            let gap = predict(mm, obs.substrate) - predict(inhibition, obs.substrate);
            gap * gap
        })
        .sum::<f64>()
        / observations.len() as f64)
        .sqrt()
        / sigma;
    let best_normalized_rmse = mm.rmse.min(inhibition.rmse) / sigma;

    // The ordering matters: first reject grossly inadequate candidate families, then preserve
    // observational equivalence, then allow the BIC threshold to select a family.
    let (selected, reason) = if best_normalized_rmse > MAX_NORMALIZED_RMSE_FOR_ADEQUACY {
        ("abstain", "both_candidate_models_fail_absolute_fit_gate")
    } else if prediction_gap < MIN_PREDICTION_GAP_IN_NOISE_SDS {
        ("abstain", "candidate_predictions_not_resolved_at_noise_scale")
    } else if delta_bic.abs() < DECISIVE_DELTA_BIC {
        ("abstain", "bic_difference_below_decisive_threshold")
    } else if delta_bic < 0.0 {
        ("michaelis_menten", "decisive_bic_favors_michaelis_menten")
    } else {
        ("substrate_inhibition", "decisive_bic_favors_substrate_inhibition")
    };

    Decision {
        selected,
        reason,
        delta_bic_mm_minus_inhibition: delta_bic,
        fitted_prediction_gap_in_noise_sd: prediction_gap,
        best_candidate_rmse_in_noise_sd: best_normalized_rmse,
    }
}

fn make_case(scenario: Scenario, seed: u64, sigma: f64) -> Value {
    let observations = observations_for(scenario, seed, sigma);
    let mm = fit_grid(&observations, ModelKind::MichaelisMenten);
    let inhibition = fit_grid(&observations, ModelKind::SubstrateInhibition);
    let decision = decide(&observations, mm, inhibition, sigma);
    let target_hit = decision.selected == scenario.expected_decision;

    json!({
        "stratum": scenario.stratum,
        "seed": seed,
        "measurement_sd": sigma,
        "observation_count": observations.len(),
        "substrates": scenario.substrate_set,
        "truth_family": scenario.truth_family,
        "expected_decision": scenario.expected_decision,
        "observed_decision": decision.selected,
        "target_hit": target_hit,
        "decision_reason": decision.reason,
        "delta_bic_mm_minus_inhibition": decision.delta_bic_mm_minus_inhibition,
        "fitted_prediction_gap_in_noise_sd": decision.fitted_prediction_gap_in_noise_sd,
        "best_candidate_rmse_in_noise_sd": decision.best_candidate_rmse_in_noise_sd,
        "fits": {
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
        }
    })
}

fn build_report() -> Result<Value> {
    let mut cases = Vec::new();
    let mut grouped: BTreeMap<&'static str, BTreeMap<String, usize>> = BTreeMap::new();
    let mut target_hits: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut target_totals: BTreeMap<&'static str, usize> = BTreeMap::new();

    for scenario in scenarios() {
        for &sigma in NOISE_SDS {
            for &seed in SEEDS {
                let case = make_case(scenario, seed, sigma);
                let selected = case["observed_decision"]
                    .as_str()
                    .context("decision label must be a string")?;
                let target_hit = case["target_hit"]
                    .as_bool()
                    .context("target_hit must be a boolean")?;

                *grouped.entry(scenario.stratum).or_default().entry(selected.to_string()).or_default() += 1;
                *target_totals.entry(scenario.stratum).or_default() += 1;
                if target_hit {
                    *target_hits.entry(scenario.stratum).or_default() += 1;
                }
                cases.push(case);
            }
        }
    }

    let total_cases = cases.len();
    let summary: Vec<Value> = scenarios()
        .iter()
        .map(|scenario| {
            let outcome_counts = grouped.get(scenario.stratum).cloned().unwrap_or_default();
            let total = *target_totals.get(scenario.stratum).unwrap_or(&0);
            let hits = *target_hits.get(scenario.stratum).unwrap_or(&0);
            let selected_models = outcome_counts.get("michaelis_menten").copied().unwrap_or(0)
                + outcome_counts.get("substrate_inhibition").copied().unwrap_or(0);
            let is_abstention_control = scenario.expected_decision == "abstain";
            json!({
                "stratum": scenario.stratum,
                "truth_family": scenario.truth_family,
                "expected_decision": scenario.expected_decision,
                "case_count": total,
                "target_hits": hits,
                "target_hit_rate": if total > 0 { hits as f64 / total as f64 } else { 0.0 },
                "false_decisive_count": if is_abstention_control { json!(selected_models) } else { Value::Null },
                "false_decisive_rate": if is_abstention_control && total > 0 { json!(selected_models as f64 / total as f64) } else { Value::Null },
                "outcome_counts": outcome_counts
            })
        })
        .collect();

    ensure!(total_cases == scenarios().len() * NOISE_SDS.len() * SEEDS.len(),
        "synthetic scenario matrix has unexpected case count");

    let control_case_count = cases
        .iter()
        .filter(|case| case["expected_decision"].as_str() == Some("abstain"))
        .count();
    let false_decisive_count = cases
        .iter()
        .filter(|case| {
            case["expected_decision"].as_str() == Some("abstain")
                && case["observed_decision"].as_str() != Some("abstain")
        })
        .count();
    let correct_abstention_count = control_case_count - false_decisive_count;

    Ok(json!({
        "schema_version": 1,
        "benchmark_id": "enzyme-abstention-matrix-v0",
        "status": "exploratory_synthetic_matrix_generated",
        "scope": "Deterministic synthetic stress matrix only. Results are not calibrated biochemical evidence, not validation of a general-purpose decision rule, and not a predictive benchmark.",
        "matrix": {
            "strata_count": scenarios().len(),
            "noise_levels": NOISE_SDS,
            "seeds_per_noise_level_and_stratum": SEEDS.len(),
            "case_count": total_cases,
            "substrate_design": "stratum-specific frozen substrate conditions",
            "noise_model": "deterministic bounded uniform residual with requested standard deviation",
            "fit_method": "coarse exhaustive parameter grid; intentionally a reproducible baseline, not a production estimator"
        },
        "decision_policy": {
            "order": [
                "abstain if both candidate models fail the absolute-fit adequacy gate",
                "abstain if fitted predictions are unresolved at the assumed measurement-noise scale",
                "abstain if absolute delta BIC is below the decisive threshold",
                "otherwise select the model favored by BIC"
            ],
            "delta_bic_definition": "BIC(Michaelis-Menten) - BIC(substrate inhibition); negative favors Michaelis-Menten",
            "decisive_absolute_delta_bic": DECISIVE_DELTA_BIC,
            "maximum_best_candidate_rmse_in_noise_sds": MAX_NORMALIZED_RMSE_FOR_ADEQUACY,
            "minimum_fitted_prediction_gap_in_noise_sds": MIN_PREDICTION_GAP_IN_NOISE_SDS,
            "calibration_status": "heuristic thresholds; not calibrated against real assay residuals"
        },
        "summary_by_stratum": summary,
        "abstention_control_summary": {
            "control_case_count": control_case_count,
            "false_decisive_count": false_decisive_count,
            "false_decisive_rate": if control_case_count > 0 { false_decisive_count as f64 / control_case_count as f64 } else { 0.0 },
            "correct_abstention_count": correct_abstention_count,
            "note": "Known-generator synthetic control summary; not an estimate of real-world false-certainty probability."
        },
        "cases": cases,
        "next_gates": [
            "increase to at least 1000 preregistered scenario/seed combinations and freeze scenario-generation code before scoring",
            "calibrate abstention thresholds using held-out synthetic scenarios, not the reported test matrix",
            "measure false-decisive rate separately for ambiguous and misspecified cases",
            "assess BIC selection calibration under heteroscedastic and correlated residuals",
            "compare with AICc, likelihood-ratio tests where valid, bootstrap model weights, and profile-likelihood diagnostics",
            "validate against rights-cleared raw assay time courses and independent laboratory/study holdouts"
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
    let bytes = serde_json::to_vec_pretty(&report).context("serialize abstention matrix report")?;
    fs::write(output, &bytes)
        .with_context(|| format!("write abstention matrix report {}", output.display()))?;
    println!(
        "Synthetic enzyme abstention matrix generated: output={}, bytes={}, cases={}, scope=exploratory-synthetic-only",
        output.display(),
        bytes.len(),
        report["matrix"]["case_count"].as_u64().unwrap_or(0)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fit(kind: ModelKind, vmax: f64, km: f64, ki: Option<f64>, rmse: f64) -> ModelFit {
        ModelFit { kind, vmax, km, ki, sse: rmse * rmse * 4.0, rmse }
    }

    #[test]
    fn observationally_equivalent_candidates_force_abstention() {
        let observations = vec![
            Observation { substrate: 0.5, rate: 20.0 },
            Observation { substrate: 1.0, rate: 33.0 },
            Observation { substrate: 2.0, rate: 50.0 },
        ];
        let mm = fit(ModelKind::MichaelisMenten, 100.0, 2.0, None, 0.5);
        let inhibition = fit(ModelKind::SubstrateInhibition, 100.0, 2.0, Some(10_000.0), 0.5);
        let decision = decide(&observations, mm, inhibition, 1.0);
        assert_eq!(decision.selected, "abstain");
        assert_eq!(decision.reason, "candidate_predictions_not_resolved_at_noise_scale");
    }

    #[test]
    fn gross_absolute_misfit_forces_abstention_before_model_selection() {
        let observations = vec![
            Observation { substrate: 1.0, rate: 20.0 },
            Observation { substrate: 16.0, rate: 80.0 },
            Observation { substrate: 64.0, rate: 10.0 },
        ];
        let mm = fit(ModelKind::MichaelisMenten, 80.0, 4.0, None, 5.0);
        let inhibition = fit(ModelKind::SubstrateInhibition, 80.0, 4.0, Some(20.0), 4.5);
        let decision = decide(&observations, mm, inhibition, 1.0);
        assert_eq!(decision.selected, "abstain");
        assert_eq!(decision.reason, "both_candidate_models_fail_absolute_fit_gate");
    }

    #[test]
    fn generated_matrix_has_fixed_denominators_and_explicit_targets() {
        let report = build_report().expect("matrix generation should succeed");
        assert_eq!(report["status"], "exploratory_synthetic_matrix_generated");
        assert_eq!(report["matrix"]["case_count"].as_u64().unwrap(), (scenarios().len() * NOISE_SDS.len() * SEEDS.len()) as u64);
        assert_eq!(report["cases"].as_array().unwrap().len(), 96);
        assert_eq!(report["summary_by_stratum"].as_array().unwrap().len(), 4);
        assert_eq!(report["abstention_control_summary"]["control_case_count"].as_u64().unwrap(), 48);
        assert!(report["abstention_control_summary"]["false_decisive_rate"].as_f64().is_some());
        assert!(report["cases"].as_array().unwrap().iter().all(|case| {
            case["observed_decision"].as_str().is_some()
                && case["decision_reason"].as_str().is_some()
                && case["target_hit"].as_bool().is_some()
        }));
        assert!(report["scope"].as_str().unwrap().contains("not calibrated biochemical evidence"));
    }
}
