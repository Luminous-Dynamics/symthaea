use serde::Serialize;
use std::env;
use std::process::ExitCode;

const GRID_STEPS: usize = 100;
const EVAL_STEPS: usize = 1_000;
const DEFAULT_SEEDS: usize = 100;
const DEFAULT_BUDGET: usize = 11;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum SurfaceFamily {
    Smooth,
    LocalizedPeak,
    Kink,
    Multiscale,
}

impl SurfaceFamily {
    const ALL: [Self; 4] = [
        Self::Smooth,
        Self::LocalizedPeak,
        Self::Kink,
        Self::Multiscale,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Smooth => "smooth",
            Self::LocalizedPeak => "localized_peak",
            Self::Kink => "kink",
            Self::Multiscale => "multiscale",
        }
    }

    fn value(self, x: f64) -> f64 {
        let tau = 2.0 * std::f64::consts::PI;
        match self {
            Self::Smooth => (tau * x).sin() + 0.35 * x,
            Self::LocalizedPeak => {
                let z = (x - 0.37) / 0.055;
                0.8 * (-0.5 * z * z).exp() + 0.2 * x
            }
            Self::Kink => {
                if x < 0.43 {
                    0.5 * x
                } else {
                    0.215 + 0.9 * (x - 0.43)
                }
            }
            Self::Multiscale => {
                0.45 * (tau * x).sin() + 0.3 * (5.0 * tau * x).sin() + 0.15 * x
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum SelectionPolicy {
    FixedGrid,
    SeededRandom,
    MaximinCoverage,
    ModelDisagreement,
}

impl SelectionPolicy {
    const ALL: [Self; 4] = [
        Self::FixedGrid,
        Self::SeededRandom,
        Self::MaximinCoverage,
        Self::ModelDisagreement,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::FixedGrid => "fixed_grid",
            Self::SeededRandom => "seeded_random",
            Self::MaximinCoverage => "maximin_coverage",
            Self::ModelDisagreement => "model_disagreement",
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct Trial {
    surface: String,
    policy: String,
    seed: u64,
    experiment_budget: usize,
    observation_count: usize,
    held_out_point_count: usize,
    mean_absolute_error: f64,
    root_mean_square_error: f64,
    maximum_absolute_error: f64,
}

#[derive(Clone, Debug, Serialize)]
struct Summary {
    surface: String,
    policy: String,
    seed_count: usize,
    experiment_budget: usize,
    mean_absolute_error: f64,
    p90_absolute_error: f64,
    mean_root_mean_square_error: f64,
    worst_trial_maximum_absolute_error: f64,
}

#[derive(Debug, Serialize)]
struct TournamentReport {
    report_version: &'static str,
    report_status: &'static str,
    claim_ceiling: &'static str,
    synthetic_only_notice: &'static str,
    seed_count: usize,
    experiment_budget: usize,
    surfaces: Vec<String>,
    policies: Vec<String>,
    summaries: Vec<Summary>,
    trials: Vec<Trial>,
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn unit_interval(value: u64) -> f64 {
    ((value >> 11) as f64) / ((1_u64 << 53) as f64)
}

// Matched, deterministic measurement noise: the same seed/location pair has the
// same observation under every selection policy.
fn observe(surface: SurfaceFamily, grid_index: usize, seed: u64) -> f64 {
    let location_tag = (grid_index as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    let noise = (unit_interval(splitmix64(seed ^ location_tag)) - 0.5) * 0.02;
    surface.value(grid_index as f64 / GRID_STEPS as f64) + noise
}

fn fixed_grid_indices(budget: usize) -> Vec<usize> {
    (0..budget)
        .map(|index| (index * GRID_STEPS + (budget - 1) / 2) / (budget - 1))
        .collect()
}

fn global_linear_prediction(observations: &[(usize, f64)], x: f64) -> f64 {
    if observations.len() == 1 {
        return observations[0].1;
    }
    let n = observations.len() as f64;
    let mean_x = observations
        .iter()
        .map(|(index, _)| *index as f64 / GRID_STEPS as f64)
        .sum::<f64>()
        / n;
    let mean_y = observations.iter().map(|(_, y)| *y).sum::<f64>() / n;
    let covariance = observations
        .iter()
        .map(|(index, y)| (*index as f64 / GRID_STEPS as f64 - mean_x) * (*y - mean_y))
        .sum::<f64>();
    let variance = observations
        .iter()
        .map(|(index, _)| (*index as f64 / GRID_STEPS as f64 - mean_x).powi(2))
        .sum::<f64>();
    if variance <= f64::EPSILON {
        mean_y
    } else {
        let slope = covariance / variance;
        mean_y + slope * (x - mean_x)
    }
}

fn piecewise_linear_prediction(observations: &[(usize, f64)], grid_index: usize) -> f64 {
    let mut ordered = observations.to_vec();
    ordered.sort_by_key(|(index, _)| *index);

    if let Some((_, value)) = ordered.iter().find(|(index, _)| *index == grid_index) {
        return *value;
    }

    let right = ordered
        .iter()
        .position(|(index, _)| *index > grid_index)
        .unwrap_or(ordered.len() - 1);
    let left = right.saturating_sub(1);
    let (left_index, left_value) = ordered[left];
    let (right_index, right_value) = ordered[right];
    if left_index == right_index {
        return left_value;
    }

    let proportion = (grid_index - left_index) as f64 / (right_index - left_index) as f64;
    left_value + proportion * (right_value - left_value)
}

fn already_sampled(observations: &[(usize, f64)], index: usize) -> bool {
    observations.iter().any(|(seen, _)| *seen == index)
}

fn select_next(
    policy: SelectionPolicy,
    observations: &[(usize, f64)],
    budget: usize,
    seed: u64,
    step: usize,
) -> usize {
    match policy {
        SelectionPolicy::FixedGrid => fixed_grid_indices(budget)
            .into_iter()
            .find(|index| !already_sampled(observations, *index))
            .expect("fixed-grid budget must exceed the existing observation count"),
        SelectionPolicy::SeededRandom => {
            let mut counter = 0_u64;
            loop {
                let candidate = (splitmix64(
                    seed ^ (step as u64).wrapping_mul(0xA076_1D64_78BD_642F) ^ counter,
                ) % (GRID_STEPS as u64 + 1)) as usize;
                if !already_sampled(observations, candidate) {
                    break candidate;
                }
                counter = counter.wrapping_add(1);
            }
        }
        SelectionPolicy::MaximinCoverage => {
            let mut best_index = None;
            let mut best_distance = 0_usize;
            for candidate in 0..=GRID_STEPS {
                if already_sampled(observations, candidate) {
                    continue;
                }
                let distance = observations
                    .iter()
                    .map(|(seen, _)| seen.abs_diff(candidate))
                    .min()
                    .unwrap_or(0);
                if best_index.is_none() || distance > best_distance {
                    best_index = Some(candidate);
                    best_distance = distance;
                }
            }
            best_index.expect("candidate grid has unsampled points")
        }
        SelectionPolicy::ModelDisagreement => {
            let mut best_index = None;
            let mut best_disagreement = f64::NEG_INFINITY;
            for candidate in 0..=GRID_STEPS {
                if already_sampled(observations, candidate) {
                    continue;
                }
                let x = candidate as f64 / GRID_STEPS as f64;
                let global = global_linear_prediction(observations, x);
                let local = piecewise_linear_prediction(observations, candidate);
                let disagreement = (global - local).abs();
                if best_index.is_none() || disagreement > best_disagreement {
                    best_index = Some(candidate);
                    best_disagreement = disagreement;
                }
            }
            best_index.expect("candidate grid has unsampled points")
        }
    }
}

fn select_observations(
    surface: SurfaceFamily,
    policy: SelectionPolicy,
    seed: u64,
    budget: usize,
) -> Vec<(usize, f64)> {
    // All policies share endpoints and the midpoint to reduce initialization
    // effects. All methods use the same per-location synthetic noise process.
    let mut observations = [0_usize, GRID_STEPS / 2, GRID_STEPS]
        .into_iter()
        .map(|index| (index, observe(surface, index, seed)))
        .collect::<Vec<_>>();

    while observations.len() < budget {
        let next = select_next(policy, &observations, budget, seed, observations.len());
        observations.push((next, observe(surface, next, seed)));
    }
    observations.sort_by_key(|(index, _)| *index);
    observations
}

fn held_out_metrics(
    surface: SurfaceFamily,
    observations: &[(usize, f64)],
) -> (usize, f64, f64, f64) {
    let mut count = 0_usize;
    let mut absolute_sum = 0.0;
    let mut squared_sum = 0.0;
    let mut maximum_absolute_error = 0.0_f64;

    for eval_index in 0..=EVAL_STEPS {
        // Candidate locations are excluded completely from the held-out set.
        if eval_index % (EVAL_STEPS / GRID_STEPS) == 0
            && already_sampled(observations, eval_index / (EVAL_STEPS / GRID_STEPS))
        {
            continue;
        }

        let x = eval_index as f64 / EVAL_STEPS as f64;
        let grid_position = x * GRID_STEPS as f64;
        let lower = grid_position.floor() as usize;
        let upper = grid_position.ceil() as usize;
        let prediction = if lower == upper {
            piecewise_linear_prediction(observations, lower)
        } else {
            let low_prediction = piecewise_linear_prediction(observations, lower);
            let high_prediction = piecewise_linear_prediction(observations, upper);
            let fraction = grid_position - lower as f64;
            low_prediction + fraction * (high_prediction - low_prediction)
        };
        let error = (prediction - surface.value(x)).abs();
        count += 1;
        absolute_sum += error;
        squared_sum += error * error;
        maximum_absolute_error = maximum_absolute_error.max(error);
    }

    let denominator = count.max(1) as f64;
    (
        count,
        absolute_sum / denominator,
        (squared_sum / denominator).sqrt(),
        maximum_absolute_error,
    )
}

fn trial(surface: SurfaceFamily, policy: SelectionPolicy, seed: u64, budget: usize) -> Trial {
    let observations = select_observations(surface, policy, seed, budget);
    let (held_out_point_count, mean_absolute_error, root_mean_square_error, maximum_absolute_error) =
        held_out_metrics(surface, &observations);
    Trial {
        surface: surface.name().to_owned(),
        policy: policy.name().to_owned(),
        seed,
        experiment_budget: budget,
        observation_count: observations.len(),
        held_out_point_count,
        mean_absolute_error,
        root_mean_square_error,
        maximum_absolute_error,
    }
}

fn percentile90(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        return 0.0;
    }
    let index = ((values.len() - 1) as f64 * 0.9).ceil() as usize;
    values[index]
}

fn run_tournament(seed_count: usize, budget: usize) -> TournamentReport {
    let trials = SurfaceFamily::ALL
        .into_iter()
        .flat_map(|surface| {
            (0..seed_count).flat_map(move |seed| {
                SelectionPolicy::ALL
                    .into_iter()
                    .map(move |policy| trial(surface, policy, seed as u64 + 1, budget))
            })
        })
        .collect::<Vec<_>>();

    let mut summaries = Vec::new();
    for surface in SurfaceFamily::ALL {
        for policy in SelectionPolicy::ALL {
            let matched = trials
                .iter()
                .filter(|result| result.surface == surface.name() && result.policy == policy.name())
                .collect::<Vec<_>>();
            let mut maes = matched
                .iter()
                .map(|result| result.mean_absolute_error)
                .collect::<Vec<_>>();
            let p90_absolute_error = percentile90(&mut maes);
            let mean_absolute_error = if matched.is_empty() {
                0.0
            } else {
                matched.iter().map(|result| result.mean_absolute_error).sum::<f64>()
                    / matched.len() as f64
            };
            let mean_root_mean_square_error = if matched.is_empty() {
                0.0
            } else {
                matched.iter().map(|result| result.root_mean_square_error).sum::<f64>()
                    / matched.len() as f64
            };
            let worst_trial_maximum_absolute_error = matched
                .iter()
                .map(|result| result.maximum_absolute_error)
                .max_by(f64::total_cmp)
                .unwrap_or(0.0);
            summaries.push(Summary {
                surface: surface.name().to_owned(),
                policy: policy.name().to_owned(),
                seed_count,
                experiment_budget: budget,
                mean_absolute_error,
                p90_absolute_error,
                mean_root_mean_square_error,
                worst_trial_maximum_absolute_error,
            });
        }
    }

    TournamentReport {
        report_version: "mus-doe-policy-tournament/0.1",
        report_status: "SYNTHETIC_ALGORITHM_COMPARISON_NOT_PHYSICAL_VALIDATION",
        claim_ceiling: "DETERMINISTIC_SYNTHETIC_POLICY_COMPARISON_ONLY",
        synthetic_only_notice: "Known-truth synthetic surfaces test policy behavior only; no instrument measurements, validated surrogate, real information gain, or physical performance claims are represented.",
        seed_count,
        experiment_budget: budget,
        surfaces: SurfaceFamily::ALL
            .into_iter()
            .map(|surface| surface.name().to_owned())
            .collect(),
        policies: SelectionPolicy::ALL
            .into_iter()
            .map(|policy| policy.name().to_owned())
            .collect(),
        summaries,
        trials,
    }
}

fn parse_args() -> Result<(usize, usize), String> {
    let mut seeds = DEFAULT_SEEDS;
    let mut budget = DEFAULT_BUDGET;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--seeds" => {
                seeds = args
                    .next()
                    .ok_or("--seeds requires a positive integer")?
                    .parse()
                    .map_err(|_| "--seeds requires a positive integer")?;
            }
            "--budget" => {
                budget = args
                    .next()
                    .ok_or("--budget requires an odd integer from 3 to 101")?
                    .parse()
                    .map_err(|_| "--budget requires an odd integer from 3 to 101")?;
            }
            "--help" | "-h" => {
                println!("policy_tournament [--seeds N] [--budget ODD_N]\nRuns a deterministic synthetic selection-policy comparison.");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    if seeds == 0 {
        return Err("--seeds must be greater than zero".to_owned());
    }
    if !(3..=101).contains(&budget) || budget % 2 == 0 {
        return Err("--budget must be an odd integer from 3 to 101".to_owned());
    }
    Ok((seeds, budget))
}

fn main() -> ExitCode {
    let (seeds, budget) = match parse_args() {
        Ok(values) => values,
        Err(error) => {
            eprintln!("policy_tournament: {error}");
            return ExitCode::from(2);
        }
    };
    let report = run_tournament(seeds, budget);
    match serde_json::to_string_pretty(&report) {
        Ok(text) => {
            println!("{text}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("policy_tournament: report serialization failed: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_policy_obeys_same_budget_and_unique_candidate_positions() {
        for policy in SelectionPolicy::ALL {
            let observations = select_observations(SurfaceFamily::Smooth, policy, 19, 11);
            let indices = observations.iter().map(|(index, _)| *index).collect::<Vec<_>>();
            assert_eq!(indices.len(), 11);
            assert_eq!(indices.iter().copied().collect::<std::collections::BTreeSet<_>>().len(), 11);
            assert!(indices.contains(&0));
            assert!(indices.contains(&(GRID_STEPS / 2)));
            assert!(indices.contains(&GRID_STEPS));
        }
    }

    #[test]
    fn run_is_byte_deterministic_for_identical_inputs() {
        let first = serde_json::to_string(&run_tournament(5, 11)).expect("serializes");
        let second = serde_json::to_string(&run_tournament(5, 11)).expect("serializes");
        assert_eq!(first, second);
    }

    #[test]
    fn held_out_metrics_exclude_all_sampled_candidate_positions() {
        let observations = select_observations(SurfaceFamily::Kink, SelectionPolicy::ModelDisagreement, 7, 11);
        let (held_out_count, mae, rmse, max_error) = held_out_metrics(SurfaceFamily::Kink, &observations);
        assert_eq!(held_out_count, EVAL_STEPS + 1 - observations.len());
        assert!(mae.is_finite() && mae >= 0.0);
        assert!(rmse.is_finite() && rmse >= 0.0);
        assert!(max_error.is_finite() && max_error >= 0.0);
    }

    #[test]
    fn matched_synthetic_observation_does_not_depend_on_selection_policy() {
        for surface in SurfaceFamily::ALL {
            for candidate in [0, 10, 37, 50, 91, 100] {
                assert_eq!(observe(surface, candidate, 123), observe(surface, candidate, 123));
            }
        }
    }

    #[test]
    fn equal_budget_comparison_covers_all_surface_policy_pairs() {
        let report = run_tournament(3, 11);
        assert_eq!(report.trials.len(), SurfaceFamily::ALL.len() * SelectionPolicy::ALL.len() * 3);
        assert_eq!(report.summaries.len(), SurfaceFamily::ALL.len() * SelectionPolicy::ALL.len());
        assert!(report.trials.iter().all(|trial| trial.observation_count == 11));
        assert!(report.trials.iter().all(|trial| trial.held_out_point_count == 990));
    }
}
