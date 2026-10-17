// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Known-truth synthetic benchmark for quantity-of-interest-aware reference acquisition.
//!
//! This deliberately small linear-Gaussian problem tests the acquisition decision logic,
//! not an interatomic potential. It asks whether selecting a reference by expected reduction
//! in target-property uncertainty outperforms generic parameter-trace reduction when many
//! candidates are informative about irrelevant nuisance parameters.

use serde::Serialize;

const DIM: usize = 3;
const DEFAULT_NOISE_VARIANCE: f64 = 0.1;
const TARGET_QOI_VARIANCE_TOLERANCE: f64 = 0.03;

type Vector = [f64; DIM];
type Matrix = [[f64; DIM]; DIM];

#[derive(Debug, Clone, Serialize)]
pub struct SyntheticCandidate {
    pub candidate_id: String,
    pub family: String,
    pub features: Vector,
}

#[derive(Debug, Clone, Serialize)]
pub struct AcquisitionStep {
    pub budget_used: usize,
    pub selected_candidate_id: String,
    pub target_qoi_posterior_variance: f64,
    pub parameter_posterior_trace: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct AcquisitionArm {
    pub arm: String,
    pub mean_calls_to_target_variance_tolerance_among_successes: Option<f64>,
    pub fraction_reaching_target_variance_tolerance_within_budget: f64,
    pub steps: Vec<AcquisitionStep>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QoiAcquisitionReport {
    pub schema_version: &'static str,
    pub scenario_id: String,
    pub scenario_description: String,
    pub target_qoi_parameter_direction: Vector,
    pub prior_covariance: Matrix,
    pub observation_noise_variance: f64,
    pub target_qoi_variance_tolerance: f64,
    pub candidate_count: usize,
    pub reference_call_budget: usize,
    pub random_baseline_trials: usize,
    pub random_seed: u64,
    pub arms: Vec<AcquisitionArm>,
    pub claim_ceiling: String,
}

#[derive(Debug, Clone, Copy)]
enum Objective {
    TargetQoiVariance,
    ParameterTrace,
}

#[derive(Debug, Clone, Copy)]
struct DeterministicRng(u64);

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.0 = value;
        value
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next_u64() % upper as u64) as usize
    }

    fn shuffle(&mut self, values: &mut [usize]) {
        for index in (1..values.len()).rev() {
            let other = self.index(index + 1);
            values.swap(index, other);
        }
    }
}

fn candidates() -> Vec<SyntheticCandidate> {
    let mut rows = Vec::with_capacity(16);

    // These candidates directly identify the target parameter theta_0. Small nuisance
    // components keep the problem non-degenerate while preserving their QoI relevance.
    for index in 0..8 {
        let y = ((index % 4) as f64 - 1.5) * 0.025;
        let z = ((index / 4) as f64 - 0.5) * 0.03;
        rows.push(SyntheticCandidate {
            candidate_id: format!("qoi-focused-{index}"),
            family: "target-property-informative".to_owned(),
            features: [1.0, y, z],
        });
    }

    // These candidates are much more informative about nuisance parameters theta_1/theta_2
    // than the target theta_0. Their larger norms attract trace-based generic acquisition.
    for index in 0..8 {
        let x = 0.05 + 0.01 * (index % 3) as f64;
        let y = 1.0 + 0.03 * ((index % 4) as f64 - 1.5);
        let z = 1.0 + 0.025 * ((index / 4) as f64 - 0.5);
        rows.push(SyntheticCandidate {
            candidate_id: format!("nuisance-focused-{index}"),
            family: "nuisance-property-informative".to_owned(),
            features: [x, y, z],
        });
    }
    rows
}

fn identity() -> Matrix {
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
}

fn dot(left: Vector, right: Vector) -> f64 {
    (0..DIM).map(|index| left[index] * right[index]).sum()
}

fn mat_vec(matrix: &Matrix, vector: Vector) -> Vector {
    let mut result = [0.0; DIM];
    for row in 0..DIM {
        for column in 0..DIM {
            result[row] += matrix[row][column] * vector[column];
        }
    }
    result
}

fn trace(matrix: &Matrix) -> f64 {
    (0..DIM).map(|index| matrix[index][index]).sum()
}

fn target_variance(covariance: &Matrix, qoi_direction: Vector) -> f64 {
    dot(qoi_direction, mat_vec(covariance, qoi_direction))
}

/// Expected Gaussian posterior-covariance update after observing one linear reference.
/// P' = P - P x x' P / (noise_variance + x' P x).
fn update_covariance(covariance: &Matrix, features: Vector, noise_variance: f64) -> Matrix {
    let projected = mat_vec(covariance, features);
    let denominator = noise_variance + dot(features, projected);
    let mut updated = *covariance;
    for row in 0..DIM {
        for column in 0..DIM {
            updated[row][column] -= projected[row] * projected[column] / denominator;
        }
    }
    updated
}

fn expected_reduction(
    covariance: &Matrix,
    candidate: &SyntheticCandidate,
    objective: Objective,
    qoi_direction: Vector,
    noise_variance: f64,
) -> f64 {
    let projected = mat_vec(covariance, candidate.features);
    let denominator = noise_variance + dot(candidate.features, projected);
    match objective {
        Objective::TargetQoiVariance => dot(qoi_direction, projected).powi(2) / denominator,
        Objective::ParameterTrace => dot(projected, projected) / denominator,
    }
}

fn greedy_arm(
    all_candidates: &[SyntheticCandidate],
    objective: Objective,
    qoi_direction: Vector,
    noise_variance: f64,
    budget: usize,
    arm_name: &str,
) -> AcquisitionArm {
    let mut covariance = identity();
    let mut available = (0..all_candidates.len()).collect::<Vec<_>>();
    let mut steps = Vec::with_capacity(budget);

    for used in 1..=budget.min(all_candidates.len()) {
        let (position, _) = available
            .iter()
            .enumerate()
            .map(|(position, candidate_index)| {
                (
                    position,
                    expected_reduction(
                        &covariance,
                        &all_candidates[*candidate_index],
                        objective,
                        qoi_direction,
                        noise_variance,
                    ),
                )
            })
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .expect("candidate pool is non-empty until budget is exhausted");

        let selected_index = available.remove(position);
        let selected = &all_candidates[selected_index];
        covariance = update_covariance(&covariance, selected.features, noise_variance);
        steps.push(AcquisitionStep {
            budget_used: used,
            selected_candidate_id: selected.candidate_id.clone(),
            target_qoi_posterior_variance: target_variance(&covariance, qoi_direction),
            parameter_posterior_trace: trace(&covariance),
        });
    }

    let calls_to_target_variance_tolerance = steps
        .iter()
        .find(|step| step.target_qoi_posterior_variance <= TARGET_QOI_VARIANCE_TOLERANCE)
        .map(|step| step.budget_used as f64);
    AcquisitionArm {
        arm: arm_name.to_owned(),
        mean_calls_to_target_variance_tolerance_among_successes: calls_to_target_variance_tolerance,
        fraction_reaching_target_variance_tolerance_within_budget: if calls_to_target_variance_tolerance.is_some() { 1.0 } else { 0.0 },
        steps,
    }
}

fn random_arm_mean(
    all_candidates: &[SyntheticCandidate],
    qoi_direction: Vector,
    noise_variance: f64,
    budget: usize,
    trials: usize,
    seed: u64,
) -> AcquisitionArm {
    let max_budget = budget.min(all_candidates.len());
    let mut qoi_sums = vec![0.0; max_budget];
    let mut trace_sums = vec![0.0; max_budget];
    let mut target_tolerance_calls = Vec::new();
    let mut rng = DeterministicRng::new(seed);

    for _ in 0..trials {
        let mut order = (0..all_candidates.len()).collect::<Vec<_>>();
        rng.shuffle(&mut order);
        let mut covariance = identity();
        for (step_index, candidate_index) in order.into_iter().take(max_budget).enumerate() {
            covariance = update_covariance(
                &covariance,
                all_candidates[candidate_index].features,
                noise_variance,
            );
            let target_variance = target_variance(&covariance, qoi_direction);
            qoi_sums[step_index] += target_variance;
            trace_sums[step_index] += trace(&covariance);
            if target_tolerance_calls.is_empty() && target_variance <= TARGET_QOI_VARIANCE_TOLERANCE {
                target_tolerance_calls.push(step_index + 1);
            }
        }
    }

    let steps = (0..max_budget)
        .map(|index| AcquisitionStep {
            budget_used: index + 1,
            selected_candidate_id: format!("mean-of-{trials}-uniform-random-orders"),
            target_qoi_posterior_variance: qoi_sums[index] / trials as f64,
            parameter_posterior_trace: trace_sums[index] / trials as f64,
        })
        .collect();

    let calls_to_target_variance_tolerance = steps
        .iter()
        .find(|step| step.target_qoi_posterior_variance <= TARGET_QOI_VARIANCE_TOLERANCE)
        .map(|step| step.budget_used);
    let calls_to_target_variance_tolerance = target_tolerance_calls
        .iter()
        .map(|calls| *calls as f64)
        .sum::<f64>()
        / trials as f64;
    let fraction_reaching_target_variance_tolerance_within_budget =
        target_tolerance_calls.len() as f64 / trials as f64;
    AcquisitionArm {
        arm: "uniform_random_average".to_owned(),
        mean_calls_to_target_variance_tolerance_among_successes: if target_tolerance_calls.is_empty() {
            None
        } else {
            Some(
                target_tolerance_calls.iter().map(|calls| *calls as f64).sum::<f64>()
                    / target_tolerance_calls.len() as f64,
            )
        },
        fraction_reaching_target_variance_tolerance_within_budget,
        steps,
    }
}

/// Run a deterministic known-truth linear-Gaussian acquisition benchmark.
///
/// A reference call observes a noisy linear indicator of three latent parameters. The target
/// quantity of interest is theta_0. The benchmark contrasts greedy acquisition that maximizes
/// expected reduction in target-QoI variance, greedy reduction in total parameter covariance
/// trace, and a repeatable uniform-random baseline averaged over many candidate orders.
pub fn run_qoi_acquisition_benchmark(
    reference_budget: usize,
    random_trials: usize,
    random_seed: u64,
) -> Result<QoiAcquisitionReport, String> {
    if reference_budget == 0 {
        return Err("reference_budget must be greater than zero".to_owned());
    }
    if random_trials == 0 {
        return Err("random_trials must be greater than zero".to_owned());
    }
    if random_trials > MAX_RANDOM_TRIALS {
        return Err(format!(
            "random_trials exceeds the supported limit of {MAX_RANDOM_TRIALS}"
        ));
    }

    let candidates = candidates();
    let budget = reference_budget.min(candidates.len());
    let qoi_direction = [1.0, 0.0, 0.0];
    let noise_variance = DEFAULT_NOISE_VARIANCE;

    let qoi_greedy = greedy_arm(
        &candidates,
        Objective::TargetQoiVariance,
        qoi_direction,
        noise_variance,
        budget,
        "qoi_variance_greedy",
    );
    let generic_greedy = greedy_arm(
        &candidates,
        Objective::ParameterTrace,
        qoi_direction,
        noise_variance,
        budget,
        "parameter_trace_greedy",
    );
    let random = random_arm_mean(
        &candidates,
        qoi_direction,
        noise_variance,
        budget,
        random_trials,
        random_seed,
    );

    Ok(QoiAcquisitionReport {
        schema_version: "nano-qoi-acquisition-synthetic-v1",
        scenario_id: "linear-gaussian-target-vs-nuisance-v1".to_owned(),
        scenario_description: "Three latent parameters with identity prior covariance; target QoI is theta_0. Eight candidates are target-informative, while eight higher-norm candidates primarily inform nuisance parameters. Each reference adds one noisy linear observation with fixed known variance.".to_owned(),
        target_qoi_parameter_direction: qoi_direction,
        prior_covariance: identity(),
        observation_noise_variance: noise_variance,
        target_qoi_variance_tolerance: TARGET_QOI_VARIANCE_TOLERANCE,
        candidate_count: candidates.len(),
        reference_call_budget: budget,
        random_baseline_trials: random_trials,
        random_seed,
        arms: vec![qoi_greedy, generic_greedy, random],
        claim_ceiling: "This is a mathematical known-truth benchmark for acquisition logic only. Its features and noise model are synthetic; it is not an atomistic potential, not a materials result, and not evidence of Symthaea performance on real QoIs. The random baseline is averaged over the declared deterministic trial count. Real promotion requires independent material-family data, reference-cost accounting, calibrated uncertainty and preregistered downstream QoI gates.".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qoi_targeted_acquisition_beats_generic_trace_in_the_designed_nuisance_shift() {
        let report = run_qoi_acquisition_benchmark(4, 256, 42).unwrap();
        let qoi = &report.arms[0].steps;
        let generic = &report.arms[1].steps;

        assert!(
            qoi[0].target_qoi_posterior_variance
                < generic[0].target_qoi_posterior_variance
        );
        assert!(
            qoi[3].target_qoi_posterior_variance
                < generic[3].target_qoi_posterior_variance
        );
        assert!(qoi[0].selected_candidate_id.starts_with("qoi-focused-"));
        assert!(generic[0].selected_candidate_id.starts_with("nuisance-focused-"));
    }



    #[test]
    fn target_qoi_tolerance_reports_reference_call_efficiency() {
        let report = run_qoi_acquisition_benchmark(8, 256, 0x51a7).unwrap();
        assert_eq!(report.target_qoi_variance_tolerance, 0.03);
        assert_eq!(report.arms[0].mean_calls_to_target_variance_tolerance_among_successes, Some(4.0));
        assert_eq!(report.arms[1].mean_calls_to_target_variance_tolerance_among_successes, Some(7.0));
        assert_eq!(report.arms[2].fraction_reaching_target_variance_tolerance_within_budget, 177.0 / 256.0);
        assert!(
            (report.arms[2].mean_calls_to_target_variance_tolerance_among_successes.unwrap()
                - 6.474_576_271_186_44)
                .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn deterministic_curve_matches_independent_numeric_reference_values() {
        let report = run_qoi_acquisition_benchmark(8, 256, 0x51a7).unwrap();
        let qoi = &report.arms[0].steps;
        let generic = &report.arms[1].steps;
        let random = &report.arms[2].steps;

        assert!((qoi[0].target_qoi_posterior_variance - 0.091_224_064_386_775_1).abs() < 1.0e-12);
        assert!((qoi[3].target_qoi_posterior_variance - 0.024_390_243_902_439_032).abs() < 1.0e-12);
        assert!((qoi[7].target_qoi_posterior_variance - 0.012_345_679_012_345_682).abs() < 1.0e-12);
        assert!((generic[0].target_qoi_posterior_variance - 0.998_378_948_849_644_7).abs() < 1.0e-12);
        assert!((generic[3].target_qoi_posterior_variance - 0.047_673_853_275_877_52).abs() < 1.0e-12);
        assert!((random[0].target_qoi_posterior_variance - 0.530_845_796_210_270_1).abs() < 1.0e-12);
        assert!((random[3].target_qoi_posterior_variance - 0.079_575_644_253_677_45).abs() < 1.0e-12);
    }

    #[test]
    fn qoi_benchmark_is_deterministic_for_fixed_seed() {
        let first = run_qoi_acquisition_benchmark(8, 128, 0x51a7).unwrap();
        let second = run_qoi_acquisition_benchmark(8, 128, 0x51a7).unwrap();

        let a = serde_json::to_string(&first).unwrap();
        let b = serde_json::to_string(&second).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn invalid_budget_and_random_trial_counts_fail_closed() {
        assert!(run_qoi_acquisition_benchmark(0, 128, 1).is_err());
        assert!(run_qoi_acquisition_benchmark(4, 0, 1).is_err());
        assert!(run_qoi_acquisition_benchmark(4, 1_000_001, 1).is_err());
    }

    #[test]
    fn posterior_covariance_update_preserves_symmetry_and_positive_diagonal() {
        let covariance = update_covariance(&identity(), [0.05, 1.0, 1.0], DEFAULT_NOISE_VARIANCE);
        for row in 0..DIM {
            assert!(covariance[row][row] > 0.0);
            for column in 0..DIM {
                assert!((covariance[row][column] - covariance[column][row]).abs() < 1.0e-12);
            }
        }
    }
}
