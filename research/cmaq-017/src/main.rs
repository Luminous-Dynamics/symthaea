use std::fmt::Write as _;

const SCHEMA: &str = "cmaq-017-multistep-exploitability-v0.1";
const EPISODES: usize = 16;
const ENVIRONMENT_STEPS: usize = 12;
const BASE_SEED: u64 = 20_261_031;
const HORIZONS: [usize; 3] = [1, 4, 8];
const BUDGETS: [usize; 3] = [8, 32, 64];
const EXECUTION_BLOCKS: [usize; 2] = [1, 4];
const DISCOUNT: f64 = 0.99;
const TRUE_TERMINATION_X: f64 = 0.84;
const TRAINING_X_SUPPORT: f64 = 0.35;
const TRAINING_V_SUPPORT: f64 = 0.45;
const ENSEMBLE_CENTERS: [f64; 3] = [0.12, 0.20, 0.28];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario {
    Benign,
    StateActionPocket,
    CompoundingDrift,
    TerminationSeam,
    RegimeMigration,
}

impl Scenario {
    const ALL: [Self; 5] = [
        Self::Benign,
        Self::StateActionPocket,
        Self::CompoundingDrift,
        Self::TerminationSeam,
        Self::RegimeMigration,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Benign => "benign",
            Self::StateActionPocket => "state_action_pocket",
            Self::CompoundingDrift => "compounding_drift",
            Self::TerminationSeam => "termination_seam",
            Self::RegimeMigration => "regime_migration",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Policy {
    DirectPolicy,
    ComputeMatchedDirectSearch,
    RandomPointSearch,
    AdaptivePointCem,
    PessimisticEnsembleCem,
    UncertaintyLcbCem,
    ShuffledFutureNegativeControl,
    OracleCem,
}

impl Policy {
    const ALL: [Self; 8] = [
        Self::DirectPolicy,
        Self::ComputeMatchedDirectSearch,
        Self::RandomPointSearch,
        Self::AdaptivePointCem,
        Self::PessimisticEnsembleCem,
        Self::UncertaintyLcbCem,
        Self::ShuffledFutureNegativeControl,
        Self::OracleCem,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::DirectPolicy => "direct_policy",
            Self::ComputeMatchedDirectSearch => "compute_matched_direct_search",
            Self::RandomPointSearch => "random_point_search",
            Self::AdaptivePointCem => "adaptive_point_cem",
            Self::PessimisticEnsembleCem => "pessimistic_ensemble_cem",
            Self::UncertaintyLcbCem => "uncertainty_lcb_cem",
            Self::ShuffledFutureNegativeControl => "shuffled_future_negative_control",
            Self::OracleCem => "oracle_cem",
        }
    }

    fn uses_world_model(self) -> bool {
        matches!(
            self,
            Self::RandomPointSearch
                | Self::AdaptivePointCem
                | Self::PessimisticEnsembleCem
                | Self::UncertaintyLcbCem
                | Self::ShuffledFutureNegativeControl
        )
    }

    fn is_oracle(self) -> bool {
        self == Self::OracleCem
    }

    fn heads(self) -> usize {
        match self {
            Self::PessimisticEnsembleCem | Self::UncertaintyLcbCem => ENSEMBLE_CENTERS.len(),
            Self::RandomPointSearch
            | Self::AdaptivePointCem
            | Self::ShuffledFutureNegativeControl => 1,
            Self::DirectPolicy | Self::ComputeMatchedDirectSearch | Self::OracleCem => 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct State {
    x: f64,
    v: f64,
}

#[derive(Clone, Copy, Debug)]
struct Transition {
    state: State,
    reward: f64,
    terminal: bool,
}

#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(f64::MIN_POSITIVE);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[derive(Clone, Copy, Debug)]
struct CandidateScore {
    ranking_score: f64,
    predicted_prefix_return: Option<f64>,
    model_transition_evaluations: usize,
    oracle_transition_evaluations: usize,
}

#[derive(Clone, Debug)]
struct Candidate {
    actions: Vec<f64>,
    score: CandidateScore,
}

#[derive(Clone, Debug)]
struct Plan {
    actions: Vec<f64>,
    predicted_prefix_return: Option<f64>,
    candidate_queries: usize,
    model_head_evaluations: usize,
    model_transition_evaluations: usize,
    oracle_transition_evaluations: usize,
}

#[derive(Clone, Copy, Debug)]
struct EpisodeMetrics {
    total_return: f64,
    predicted_minus_realized_block_return: Option<f64>,
    ood_visit_fraction: f64,
    selected_one_step_state_rmse: f64,
    reference_model_false_safe_rate: f64,
    candidate_queries: usize,
    model_head_evaluations: usize,
    model_transition_evaluations: usize,
    oracle_transition_evaluations: usize,
    executed_steps: usize,
}

fn clamp_action(action: f64) -> f64 {
    action.clamp(-1.0, 1.0)
}

fn direct_action(state: State) -> f64 {
    clamp_action(1.65 * (0.42 - state.x) - 0.82 * state.v)
}

fn initial_state(seed: u64) -> State {
    let mut rng = SplitMix64::new(seed ^ 0xA11C_E001);
    State {
        x: -0.27 + (rng.unit() - 0.5) * 0.08,
        v: (rng.unit() - 0.5) * 0.05,
    }
}

fn disturbance(seed: u64, step: usize) -> f64 {
    let mut rng = SplitMix64::new(
        seed ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD157_0B5A,
    );
    (rng.unit() * 2.0 - 1.0) * 0.005
}

fn gaussian_bump(value: f64, center: f64, width: f64) -> f64 {
    let z = (value - center) / width;
    (-0.5 * z * z).exp()
}

#[allow(clippy::too_many_arguments)]
fn transition_core(
    state: State,
    action: f64,
    step: usize,
    scenario: Scenario,
    seed: u64,
    head: usize,
    truth: bool,
    oracle: bool,
) -> Transition {
    let action = clamp_action(action);

    let gain = if (truth || oracle)
        && scenario == Scenario::RegimeMigration
        && step >= 6
    {
        -0.12
    } else if !truth && !oracle && scenario == Scenario::CompoundingDrift {
        // Small action-correlated model bias; repeated planning compounds it.
        0.205
    } else {
        // The learned model has not observed the hidden regime transition.
        0.16
    };

    let process_noise = if truth && scenario != Scenario::Benign {
        disturbance(seed, step)
    } else {
        0.0
    };
    let next_v = (0.86 * state.v + gain * action + process_noise).clamp(-1.2, 1.2);
    let next_x = state.x + 0.10 * next_v;

    let mut reward =
        1.0 - (next_x - 0.42).powi(2) - 0.10 * next_v.powi(2) - 0.02 * action.powi(2);

    if !truth && !oracle && scenario == Scenario::StateActionPocket && action > 0.55 {
        let center = ENSEMBLE_CENTERS[head.min(ENSEMBLE_CENTERS.len() - 1)];
        reward += 0.55 * gaussian_bump(state.x, center, 0.025) * ((action - 0.55) / 0.45);
    }

    let termination_x = if !truth && !oracle && scenario == Scenario::TerminationSeam {
        1.15
    } else {
        TRUE_TERMINATION_X
    };
    let terminal = next_x >= termination_x || next_x <= -0.62;
    if terminal {
        reward -= 2.0;
    }

    Transition {
        state: State {
            x: next_x,
            v: next_v,
        },
        reward,
        terminal,
    }
}

fn true_step(state: State, action: f64, step: usize, scenario: Scenario, seed: u64) -> Transition {
    transition_core(state, action, step, scenario, seed, 0, true, false)
}

fn model_step(
    state: State,
    action: f64,
    step: usize,
    scenario: Scenario,
    head: usize,
    oracle: bool,
) -> Transition {
    transition_core(state, action, step, scenario, 0, head, false, oracle)
}

#[allow(clippy::too_many_arguments)]
fn rollout_model(
    initial: State,
    actions: &[f64],
    start_step: usize,
    block_len: usize,
    scenario: Scenario,
    head: usize,
    oracle: bool,
) -> (f64, f64, usize) {
    let mut state = initial;
    let mut total = 0.0;
    let mut prefix = 0.0;
    let mut transition_evaluations = 0usize;

    for (index, action) in actions.iter().enumerate() {
        let transition = model_step(state, *action, start_step + index, scenario, head, oracle);
        transition_evaluations += 1;
        let weighted = DISCOUNT.powi(index as i32) * transition.reward;
        total += weighted;
        if index < block_len {
            prefix += weighted;
        }
        state = transition.state;
        if transition.terminal {
            break;
        }
    }

    (total, prefix, transition_evaluations)
}

fn direct_sequence_score(state: State, actions: &[f64]) -> f64 {
    let first_target = direct_action(state);
    let mut previous = first_target;
    let mut score = 0.0;
    for (index, action) in actions.iter().enumerate() {
        let target = first_target * 0.86_f64.powi(index as i32);
        score -= (*action - target).powi(2);
        score -= 0.025 * (*action - previous).powi(2);
        previous = *action;
    }
    score
}

fn aggregate_scores(values: &[(f64, f64)], policy: Policy) -> CandidateScore {
    let full: Vec<f64> = values.iter().map(|value| value.0).collect();
    let prefix: Vec<f64> = values.iter().map(|value| value.1).collect();
    let model_transition_evaluations = if policy.is_oracle() {
        0
    } else {
        values.iter().map(|value| value.2).sum()
    };
    let oracle_transition_evaluations = if policy.is_oracle() {
        values.iter().map(|value| value.2).sum()
    } else {
        0
    };

    match policy {
        Policy::PessimisticEnsembleCem => CandidateScore {
            ranking_score: full.iter().copied().fold(f64::INFINITY, f64::min),
            predicted_prefix_return: Some(
                prefix.iter().copied().fold(f64::INFINITY, f64::min),
            ),
            model_transition_evaluations,
            oracle_transition_evaluations,
        },
        Policy::UncertaintyLcbCem => {
            let mean_full = full.iter().sum::<f64>() / full.len() as f64;
            let mean_prefix = prefix.iter().sum::<f64>() / prefix.len() as f64;
            let std_full = (full
                .iter()
                .map(|value| (value - mean_full).powi(2))
                .sum::<f64>()
                / full.len() as f64)
                .sqrt();
            let std_prefix = (prefix
                .iter()
                .map(|value| (value - mean_prefix).powi(2))
                .sum::<f64>()
                / prefix.len() as f64)
                .sqrt();
            CandidateScore {
                ranking_score: mean_full - 1.25 * std_full,
                predicted_prefix_return: Some(mean_prefix - 1.25 * std_prefix),
                model_transition_evaluations,
                oracle_transition_evaluations,
            }
        }
        _ => CandidateScore {
            ranking_score: full[0],
            predicted_prefix_return: Some(prefix[0]),
            model_transition_evaluations,
            oracle_transition_evaluations,
        },
    }
}

fn evaluate_candidate(
    scenario: Scenario,
    policy: Policy,
    state: State,
    actions: &[f64],
    start_step: usize,
    block_len: usize,
) -> CandidateScore {
    if policy == Policy::ComputeMatchedDirectSearch {
        return CandidateScore {
            ranking_score: direct_sequence_score(state, actions),
            predicted_prefix_return: None,
            model_transition_evaluations: 0,
            oracle_transition_evaluations: 0,
        };
    }

    let prediction_actions: Vec<f64> = if policy == Policy::ShuffledFutureNegativeControl {
        actions.iter().rev().copied().collect()
    } else {
        actions.to_vec()
    };
    let oracle = policy.is_oracle();
    let head_count = policy.heads().max(1);
    let mut values = Vec::with_capacity(head_count);

    for head in 0..head_count {
        values.push(rollout_model(
            state,
            &prediction_actions,
            start_step,
            block_len,
            scenario,
            head,
            oracle,
        ));
    }
    aggregate_scores(&values, policy)
}

#[allow(clippy::needless_range_loop)]
fn update_cem_distribution(candidates: &mut [Candidate], mean: &mut [f64], std_dev: &mut [f64]) {
    candidates.sort_by(|left, right| right.score.ranking_score.total_cmp(&left.score.ranking_score));
    let elite_count = (candidates.len() / 5).max(1).min(candidates.len());

    for (index, (mean_value, std_value)) in mean.iter_mut().zip(std_dev.iter_mut()).enumerate() {
        let elite_mean = candidates[..elite_count]
            .iter()
            .map(|candidate| candidate.actions[index])
            .sum::<f64>()
            / elite_count as f64;
        let variance = candidates[..elite_count]
            .iter()
            .map(|candidate| (candidate.actions[index] - elite_mean).powi(2))
            .sum::<f64>()
            / elite_count as f64;
        *mean_value = elite_mean;
        *std_value = (variance.sqrt() * 1.25).clamp(0.025, 0.70);
    }
}

#[allow(clippy::too_many_arguments)]
fn plan_actions(
    scenario: Scenario,
    policy: Policy,
    state: State,
    start_step: usize,
    horizon: usize,
    block_len: usize,
    candidate_budget: usize,
    decision_seed: u64,
) -> Plan {
    if policy == Policy::DirectPolicy {
        return Plan {
            actions: vec![direct_action(state); horizon],
            predicted_prefix_return: None,
            candidate_queries: 0,
            model_head_evaluations: 0,
            model_transition_evaluations: 0,
            oracle_transition_evaluations: 0,
        };
    }

    let rounds = if policy == Policy::RandomPointSearch { 1 } else { 3 };
    let mut rng = SplitMix64::new(decision_seed);
    let first_target = direct_action(state);
    let mut mean: Vec<f64> = (0..horizon)
        .map(|index| first_target * 0.86_f64.powi(index as i32))
        .collect();
    let mut std_dev = vec![0.62; horizon];
    let mut best: Option<Candidate> = None;
    let mut model_transition_evaluations = 0usize;
    let mut oracle_transition_evaluations = 0usize;
    let base_per_round = candidate_budget / rounds;
    let remainder = candidate_budget % rounds;

    for round in 0..rounds {
        let count = base_per_round + if round < remainder { 1 } else { 0 };
        if count == 0 {
            continue;
        }
        let mut candidates = Vec::with_capacity(count);

        for _ in 0..count {
            let actions: Vec<f64> = if policy == Policy::RandomPointSearch {
                (0..horizon)
                    .map(|_| rng.unit() * 2.0 - 1.0)
                    .collect()
            } else {
                mean.iter()
                    .zip(std_dev.iter())
                    .map(|(mean_value, spread)| {
                        clamp_action(*mean_value + *spread * rng.normal())
                    })
                    .collect()
            };

            let score = evaluate_candidate(
                scenario,
                policy,
                state,
                &actions,
                start_step,
                block_len,
            );
            model_transition_evaluations += score.model_transition_evaluations;
            oracle_transition_evaluations += score.oracle_transition_evaluations;
            let candidate = Candidate { actions, score };

            let replace_best = best
                .as_ref()
                .map(|current| candidate.score.ranking_score > current.score.ranking_score)
                .unwrap_or(true);
            if replace_best {
                best = Some(candidate.clone());
            }
            candidates.push(candidate);
        }

        if policy != Policy::RandomPointSearch {
            update_cem_distribution(&mut candidates, &mut mean, &mut std_dev);
        }
    }

    let selected = best.expect("positive candidate budget always produces one candidate");
    Plan {
        actions: selected.actions,
        predicted_prefix_return: selected.score.predicted_prefix_return,
        candidate_queries: candidate_budget,
        model_head_evaluations: if policy.uses_world_model() {
            candidate_budget * policy.heads()
        } else {
            0
        },
        model_transition_evaluations,
        oracle_transition_evaluations,
    }
}

fn in_training_support(state: State) -> bool {
    state.x.abs() <= TRAINING_X_SUPPORT && state.v.abs() <= TRAINING_V_SUPPORT
}

fn uniform_grid_one_step_state_rmse(scenario: Scenario) -> f64 {
    const GRID_POINTS: usize = 201;
    let mut squared_error = 0.0;
    for index in 0..GRID_POINTS {
        let x = -TRAINING_X_SUPPORT
            + 2.0 * TRAINING_X_SUPPORT * index as f64 / (GRID_POINTS - 1) as f64;
        let v_index = (index * 37) % GRID_POINTS;
        let action_index = (index * 73) % GRID_POINTS;
        let v = -TRAINING_V_SUPPORT
            + 2.0 * TRAINING_V_SUPPORT * v_index as f64 / (GRID_POINTS - 1) as f64;
        let action = -1.0 + 2.0 * action_index as f64 / (GRID_POINTS - 1) as f64;
        let step = [0usize, 6, 8][index % 3];
        let state = State { x, v };
        let actual = true_step(state, action, step, scenario, BASE_SEED);
        let predicted = model_step(state, action, step, scenario, 1, false);
        squared_error += (actual.state.x - predicted.state.x).powi(2)
            + (actual.state.v - predicted.state.v).powi(2);
    }
    (squared_error / GRID_POINTS as f64).sqrt()
}


fn run_episode(
    scenario: Scenario,
    policy: Policy,
    horizon: usize,
    execution_block: usize,
    candidate_budget: usize,
    episode_seed: u64,
) -> EpisodeMetrics {
    let mut state = initial_state(episode_seed);
    let mut time = 0usize;
    let mut total_return = 0.0;
    let mut predicted_gap_sum = 0.0;
    let mut has_predicted_gap = false;
    let mut ood_visits = 0usize;
    let mut executed_steps = 0usize;
    let mut state_error_squared_sum = 0.0;
    let mut false_safe_count = 0usize;
    let mut candidate_queries = 0usize;
    let mut model_head_evaluations = 0usize;
    let mut model_transition_evaluations = 0usize;
    let mut oracle_transition_evaluations = 0usize;

    while time < ENVIRONMENT_STEPS {
        let block_len = execution_block.min(horizon).min(ENVIRONMENT_STEPS - time);
        let decision_seed = episode_seed
            ^ (time as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (horizon as u64).rotate_left(11)
            ^ (candidate_budget as u64).rotate_left(23);
        let plan = plan_actions(
            scenario,
            policy,
            state,
            time,
            horizon,
            block_len,
            candidate_budget,
            decision_seed,
        );

        candidate_queries += plan.candidate_queries;
        model_head_evaluations += plan.model_head_evaluations;
        model_transition_evaluations += plan.model_transition_evaluations;
        oracle_transition_evaluations += plan.oracle_transition_evaluations;

        let mut actual_block_return = 0.0;
        for index in 0..block_len {
            let action = if policy == Policy::DirectPolicy {
                direct_action(state)
            } else {
                plan.actions[index]
            };

            let truth = true_step(state, action, time, scenario, episode_seed);
            let diagnostic = model_step(state, action, time, scenario, 1, false);
            let dx = diagnostic.state.x - truth.state.x;
            let dv = diagnostic.state.v - truth.state.v;
            state_error_squared_sum += dx * dx + dv * dv;
            if !diagnostic.terminal && truth.terminal {
                false_safe_count += 1;
            }

            total_return += truth.reward;
            actual_block_return += DISCOUNT.powi(index as i32) * truth.reward;
            state = truth.state;
            if !in_training_support(state) {
                ood_visits += 1;
            }
            executed_steps += 1;
            time += 1;
            if truth.terminal || time >= ENVIRONMENT_STEPS {
                break;
            }
        }

        if let Some(predicted_prefix) = plan.predicted_prefix_return {
            predicted_gap_sum += predicted_prefix - actual_block_return;
            has_predicted_gap = true;
        }
    }

    let denominator = executed_steps.max(1) as f64;
    EpisodeMetrics {
        total_return,
        predicted_minus_realized_block_return: has_predicted_gap.then_some(predicted_gap_sum),
        ood_visit_fraction: ood_visits as f64 / denominator,
        selected_one_step_state_rmse: (state_error_squared_sum / denominator).sqrt(),
        reference_model_false_safe_rate: false_safe_count as f64 / denominator,
        candidate_queries,
        model_head_evaluations,
        model_transition_evaluations,
        oracle_transition_evaluations,
        executed_steps,
    }
}

fn mean_and_ci(values: &[f64]) -> (f64, f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if values.len() < 2 {
        return (mean, mean, mean);
    }
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    let margin = 1.96 * (variance / values.len() as f64).sqrt();
    (mean, mean - margin, mean + margin)
}

fn mean_optional(values: &[Option<f64>]) -> Option<f64> {
    let present: Vec<f64> = values.iter().filter_map(|value| *value).collect();
    (!present.is_empty()).then(|| present.iter().sum::<f64>() / present.len() as f64)
}

fn write_option(output: &mut String, value: Option<f64>) {
    match value {
        Some(number) if number.is_finite() => {
            write!(output, "{number:.9}").expect("writing to String cannot fail");
        }
        _ => output.push_str("null"),
    }
}

#[allow(clippy::too_many_arguments)]
fn append_row(
    output: &mut String,
    scenario: Scenario,
    horizon: usize,
    execution_block: usize,
    candidate_budget: usize,
    policy: Policy,
    episode_results: &[EpisodeMetrics],
    direct_returns: &[f64],
    matched_returns: &[f64],
    shuffled_returns: &[f64],
    oracle_returns: &[f64],
) {
    let returns: Vec<f64> = episode_results.iter().map(|row| row.total_return).collect();
    let paired_direct: Vec<f64> = returns
        .iter()
        .zip(direct_returns)
        .map(|(value, direct)| value - direct)
        .collect();
    let paired_matched: Vec<f64> = returns
        .iter()
        .zip(matched_returns)
        .map(|(value, matched)| value - matched)
        .collect();
    let paired_shuffled: Vec<f64> = returns
        .iter()
        .zip(shuffled_returns)
        .map(|(value, shuffled)| value - shuffled)
        .collect();
    let paired_oracle_gap: Vec<f64> = oracle_returns
        .iter()
        .zip(&returns)
        .map(|(oracle, value)| oracle - value)
        .collect();

    let (mean_return, return_low, return_high) = mean_and_ci(&returns);
    let mut sorted_returns = returns.clone();
    sorted_returns.sort_by(f64::total_cmp);
    let lower_tail_count = sorted_returns.len().div_ceil(10).max(1);
    let lower_decile_mean_return =
        sorted_returns[..lower_tail_count].iter().sum::<f64>() / lower_tail_count as f64;
    let worst_episode_return = sorted_returns[0];
    let (direct_delta, direct_low, direct_high) = mean_and_ci(&paired_direct);
    let (matched_delta, matched_low, matched_high) = mean_and_ci(&paired_matched);
    let (shuffled_delta, shuffled_low, shuffled_high) = mean_and_ci(&paired_shuffled);
    let (oracle_gap, oracle_low, oracle_high) = mean_and_ci(&paired_oracle_gap);

    let uniform_grid_rmse = uniform_grid_one_step_state_rmse(scenario);
    let mean_gap = mean_optional(
        &episode_results
            .iter()
            .map(|row| row.predicted_minus_realized_block_return)
            .collect::<Vec<_>>(),
    );
    let denominator = episode_results.len() as f64;
    let mean_ood = episode_results
        .iter()
        .map(|row| row.ood_visit_fraction)
        .sum::<f64>()
        / denominator;
    let mean_state_rmse = episode_results
        .iter()
        .map(|row| row.selected_one_step_state_rmse)
        .sum::<f64>()
        / denominator;
    let mean_false_safe = episode_results
        .iter()
        .map(|row| row.reference_model_false_safe_rate)
        .sum::<f64>()
        / denominator;
    let mean_queries = episode_results
        .iter()
        .map(|row| row.candidate_queries as f64)
        .sum::<f64>()
        / denominator;
    let mean_heads = episode_results
        .iter()
        .map(|row| row.model_head_evaluations as f64)
        .sum::<f64>()
        / denominator;
    let mean_model_transitions = episode_results
        .iter()
        .map(|row| row.model_transition_evaluations as f64)
        .sum::<f64>()
        / denominator;
    let mean_oracle_transitions = episode_results
        .iter()
        .map(|row| row.oracle_transition_evaluations as f64)
        .sum::<f64>()
        / denominator;
    let mean_steps = episode_results
        .iter()
        .map(|row| row.executed_steps as f64)
        .sum::<f64>()
        / denominator;

    write!(
        output,
        "{{\"scenario\":\"{}\",\"horizon\":{},\"execution_block\":{},\"candidate_budget\":{},\"policy\":\"{}\",\"episode_count\":{},",
        scenario.name(),
        horizon,
        execution_block,
        candidate_budget,
        policy.name(),
        episode_results.len()
    )
    .expect("writing to String cannot fail");
    write!(
        output,
        "\"mean_return\":{mean_return:.9},\"return_ci95_lower\":{return_low:.9},\"return_ci95_upper\":{return_high:.9},\"lower_decile_mean_return\":{lower_decile_mean_return:.9},\"worst_episode_return\":{worst_episode_return:.9},"
    )
    .expect("writing to String cannot fail");
    write!(
        output,
        "\"paired_return_delta_vs_direct\":{direct_delta:.9},\"paired_delta_vs_direct_ci95_lower\":{direct_low:.9},\"paired_delta_vs_direct_ci95_upper\":{direct_high:.9},"
    )
    .expect("writing to String cannot fail");
    write!(
        output,
        "\"paired_return_delta_vs_compute_matched_direct\":{matched_delta:.9},\"paired_delta_vs_compute_matched_ci95_lower\":{matched_low:.9},\"paired_delta_vs_compute_matched_ci95_upper\":{matched_high:.9},"
    )
    .expect("writing to String cannot fail");
    write!(
        output,
        "\"paired_return_delta_vs_shuffled_future\":{shuffled_delta:.9},\"paired_delta_vs_shuffled_ci95_lower\":{shuffled_low:.9},\"paired_delta_vs_shuffled_ci95_upper\":{shuffled_high:.9},"
    )
    .expect("writing to String cannot fail");
    write!(
        output,
        "\"paired_return_gap_to_oracle\":{oracle_gap:.9},\"paired_gap_to_oracle_ci95_lower\":{oracle_low:.9},\"paired_gap_to_oracle_ci95_upper\":{oracle_high:.9},"
    )
    .expect("writing to String cannot fail");
    write!(
        output,
        "\"uniform_grid_one_step_state_rmse\":{uniform_grid_rmse:.9},\"mean_predicted_minus_realized_block_return\":"
    )
    .expect("writing to String cannot fail");
    write_option(output, mean_gap);
    write!(
        output,
        ",\"mean_ood_visit_fraction\":{mean_ood:.9},\"mean_selected_one_step_state_rmse\":{mean_state_rmse:.9},\"reference_model_false_safe_rate\":{mean_false_safe:.9},\"mean_candidate_sequence_queries\":{mean_queries:.3},\"mean_model_head_evaluations\":{mean_heads:.3},\"mean_model_transition_evaluations\":{mean_model_transitions:.3},\"mean_oracle_transition_evaluations\":{mean_oracle_transitions:.3},\"mean_evaluator_diagnostic_transition_evaluations\":{mean_steps:.3},\"mean_executed_steps\":{mean_steps:.3}}}"
    )
    .expect("writing to String cannot fail");
}

fn render_report() -> String {
    let mut output = String::new();
    write!(
        &mut output,
        "{{\"schema_version\":\"{}\",\"episode_count\":{},\"environment_steps\":{},\"seed\":{},",
        SCHEMA, EPISODES, ENVIRONMENT_STEPS, BASE_SEED
    )
    .expect("writing to String cannot fail");
    output.push_str(
        "\"evaluator_uniform_grid_probe_count\":201,\"scenarios\":[\"benign\",\"state_action_pocket\",\"compounding_drift\",\"termination_seam\",\"regime_migration\"],\"policies\":[\"direct_policy\",\"compute_matched_direct_search\",\"random_point_search\",\"adaptive_point_cem\",\"pessimistic_ensemble_cem\",\"uncertainty_lcb_cem\",\"shuffled_future_negative_control\",\"oracle_cem\"],\"horizons\":[1,4,8],\"candidate_budgets\":[8,32,64],\"execution_blocks\":[1,4],",
    );
    write!(
        &mut output,
        "\"training_support\":{{\"abs_x_max\":{TRAINING_X_SUPPORT:.3},\"abs_v_max\":{TRAINING_V_SUPPORT:.3}}},\"summary\":["
    )
    .expect("writing to String cannot fail");

    let mut first = true;
    for scenario in Scenario::ALL {
        for horizon in HORIZONS {
            for execution_block in EXECUTION_BLOCKS {
                if execution_block > horizon {
                    continue;
                }
                for candidate_budget in BUDGETS {
                    let mut all_results: Vec<Vec<EpisodeMetrics>> =
                        Vec::with_capacity(Policy::ALL.len());
                    for policy in Policy::ALL {
                        let mut results = Vec::with_capacity(EPISODES);
                        for episode in 0..EPISODES {
                            let episode_seed = BASE_SEED
                                ^ (episode as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                            results.push(run_episode(
                                scenario,
                                policy,
                                horizon,
                                execution_block,
                                candidate_budget,
                                episode_seed,
                            ));
                        }
                        all_results.push(results);
                    }

                    let direct_returns: Vec<f64> = all_results[0]
                        .iter()
                        .map(|row| row.total_return)
                        .collect();
                    let matched_returns: Vec<f64> = all_results[1]
                        .iter()
                        .map(|row| row.total_return)
                        .collect();
                    let shuffled_returns: Vec<f64> = all_results[6]
                        .iter()
                        .map(|row| row.total_return)
                        .collect();
                    let oracle_returns: Vec<f64> = all_results[7]
                        .iter()
                        .map(|row| row.total_return)
                        .collect();

                    for (index, policy) in Policy::ALL.iter().enumerate() {
                        if !first {
                            output.push(',');
                        }
                        first = false;
                        append_row(
                            &mut output,
                            scenario,
                            horizon,
                            execution_block,
                            candidate_budget,
                            *policy,
                            &all_results[index],
                            &direct_returns,
                            &matched_returns,
                            &shuffled_returns,
                            &oracle_returns,
                        );
                    }
                }
            }
        }
    }

    output.push_str("]}");
    output
}

fn main() {
    println!("{}", render_report());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benign_model_matches_the_deterministic_environment() {
        let state = State { x: -0.18, v: 0.11 };
        let actual = true_step(state, 0.63, 3, Scenario::Benign, BASE_SEED);
        let predicted = model_step(state, 0.63, 3, Scenario::Benign, 1, false);
        assert!((actual.state.x - predicted.state.x).abs() < 1e-12);
        assert!((actual.state.v - predicted.state.v).abs() < 1e-12);
        assert!((actual.reward - predicted.reward).abs() < 1e-12);
    }

    #[test]
    fn state_action_pocket_is_localized_to_a_state_action_conjunction() {
        let state = State { x: 0.20, v: 0.0 };
        let optimistic = model_step(
            state,
            0.85,
            2,
            Scenario::StateActionPocket,
            1,
            false,
        );
        let outside_state = model_step(
            State { x: -0.45, v: 0.0 },
            0.85,
            2,
            Scenario::StateActionPocket,
            1,
            false,
        );
        let low_action = model_step(
            state,
            0.10,
            2,
            Scenario::StateActionPocket,
            1,
            false,
        );
        let actual = true_step(state, 0.85, 2, Scenario::StateActionPocket, BASE_SEED);
        assert!(optimistic.reward - actual.reward > 0.30);
        assert!((outside_state.reward
            - true_step(
                State { x: -0.45, v: 0.0 },
                0.85,
                2,
                Scenario::StateActionPocket,
                BASE_SEED,
            )
            .reward)
            .abs()
            < 0.01);
        assert!((low_action.reward
            - true_step(state, 0.10, 2, Scenario::StateActionPocket, BASE_SEED).reward)
            .abs()
            < 0.01);
    }

    #[test]
    fn small_transition_bias_is_action_correlated() {
        let state = State { x: 0.0, v: 0.0 };
        let actual = true_step(state, 0.8, 2, Scenario::CompoundingDrift, BASE_SEED);
        let predicted = model_step(state, 0.8, 2, Scenario::CompoundingDrift, 1, false);
        assert!(predicted.state.v > actual.state.v);
        let difference = predicted.state.v - actual.state.v;
        assert!((0.029..=0.043).contains(&difference));
    }

    #[test]
    fn termination_seam_creates_a_false_safe_prediction() {
        let state = State { x: 0.833, v: 0.0 };
        let actual = true_step(state, 0.8, 2, Scenario::TerminationSeam, BASE_SEED);
        let predicted = model_step(state, 0.8, 2, Scenario::TerminationSeam, 1, false);
        assert!(actual.terminal);
        assert!(!predicted.terminal);
    }

    #[test]
    fn model_does_not_track_the_hidden_regime_migration() {
        let state = State { x: 0.0, v: 0.0 };
        let actual = true_step(state, 0.8, 8, Scenario::RegimeMigration, BASE_SEED);
        let predicted = model_step(state, 0.8, 8, Scenario::RegimeMigration, 1, false);
        assert!(actual.state.v < 0.0);
        assert!(predicted.state.v > 0.0);
    }


    #[test]
    fn rollout_transition_count_stops_at_the_predicted_terminal() {
        let (_, _, counted) = rollout_model(
            State { x: 1.12, v: 0.5 },
            &[1.0, 1.0, 1.0, 1.0],
            0,
            4,
            Scenario::TerminationSeam,
            1,
            false,
        );
        assert_eq!(counted, 1);

        let (_, _, full_horizon) = rollout_model(
            State { x: -0.2, v: 0.0 },
            &[0.0, 0.0, 0.0, 0.0],
            0,
            4,
            Scenario::Benign,
            1,
            false,
        );
        assert_eq!(full_horizon, 4);
    }

    #[test]
    fn direct_policy_has_no_candidate_or_model_query_cost() {
        let plan = plan_actions(
            Scenario::Benign,
            Policy::DirectPolicy,
            State { x: -0.2, v: 0.0 },
            0,
            4,
            1,
            32,
            BASE_SEED,
        );
        assert_eq!(plan.candidate_queries, 0);
        assert_eq!(plan.model_head_evaluations, 0);
        assert_eq!(plan.oracle_transition_evaluations, 0);
    }

    #[test]
    fn matched_direct_search_uses_candidate_budget_without_world_model_calls() {
        let plan = plan_actions(
            Scenario::Benign,
            Policy::ComputeMatchedDirectSearch,
            State { x: -0.2, v: 0.0 },
            0,
            4,
            1,
            32,
            BASE_SEED,
        );
        assert_eq!(plan.candidate_queries, 32);
        assert_eq!(plan.model_head_evaluations, 0);
        assert_eq!(plan.oracle_transition_evaluations, 0);
        assert!(plan.predicted_prefix_return.is_none());
    }

    #[test]
    fn ensemble_cost_is_explicit_and_oracle_cost_is_separate() {
        let state = State { x: -0.2, v: 0.0 };
        let ensemble = plan_actions(
            Scenario::StateActionPocket,
            Policy::PessimisticEnsembleCem,
            state,
            0,
            4,
            1,
            32,
            BASE_SEED,
        );
        let oracle = plan_actions(
            Scenario::StateActionPocket,
            Policy::OracleCem,
            state,
            0,
            4,
            1,
            32,
            BASE_SEED,
        );
        assert_eq!(ensemble.model_head_evaluations, 32 * 3);
        assert_eq!(ensemble.oracle_transition_evaluations, 0);
        assert_eq!(oracle.model_head_evaluations, 0);
        assert_eq!(oracle.oracle_transition_evaluations, 32 * 4);
    }

    #[test]
    fn repeated_episode_is_deterministic() {
        let first = run_episode(
            Scenario::CompoundingDrift,
            Policy::AdaptivePointCem,
            4,
            1,
            32,
            BASE_SEED,
        );
        let second = run_episode(
            Scenario::CompoundingDrift,
            Policy::AdaptivePointCem,
            4,
            1,
            32,
            BASE_SEED,
        );
        assert_eq!(first.total_return.to_bits(), second.total_return.to_bits());
        assert_eq!(first.candidate_queries, second.candidate_queries);
        assert_eq!(first.model_head_evaluations, second.model_head_evaluations);
    }

    #[test]
    fn paired_interval_helper_is_centered_on_the_sample_mean() {
        let (mean, lower, upper) = mean_and_ci(&[1.0, 2.0, 3.0, 4.0]);
        assert!((mean - 2.5).abs() < 1e-12);
        assert!(lower < mean && mean < upper);
    }
}
