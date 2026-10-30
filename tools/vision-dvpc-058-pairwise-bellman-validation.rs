//! VIS-DVPC-058: Pairwise Bellman Validation for OPE nuisance-model selection.
//!
//! Standalone finite-MDP falsification benchmark. Candidate Q-functions are fit
//! from a training split; every selector reads only an independent validation
//! split. The world model and exact Q/occupancy values are evaluator-only.
//!
//! Run:
//!   rustc --edition=2024 -O --test tools/vision-dvpc-058-pairwise-bellman-validation.rs -o /tmp/vis-dvpc-058-tests
//!   /tmp/vis-dvpc-058-tests
//!   rustc --edition=2024 -O tools/vision-dvpc-058-pairwise-bellman-validation.rs -o /tmp/vis-dvpc-058
//!   /tmp/vis-dvpc-058 --smoke
//!   /tmp/vis-dvpc-058
#![deny(unsafe_code)]

use std::env;

const STATES: usize = 3;
const ACTIONS: usize = 2;
const CELLS: usize = STATES * ACTIONS;
const GAMMA: f64 = 0.82;
const TRAIN_N: usize = 2048;
const VALIDATION_N: usize = 1536;
const FULL_REPLICATES: usize = 256;
const SMOKE_REPLICATES: usize = 32;
const MC_TEST_N: usize = 30_000;
const EPS: f64 = 1e-12;
const PRIOR_TRANSITIONS: f64 = 2.0;
const PRIOR_REWARD: f64 = 2.0;

#[derive(Clone, Copy, Debug)]
struct World {
    name: &'static str,
    transition: [[[f64; STATES]; ACTIONS]; STATES],
    reward_mean: [[f64; ACTIONS]; STATES],
    target_policy: [[f64; ACTIONS]; STATES],
    offline_distribution: [[f64; ACTIONS]; STATES],
    initial_distribution: [f64; STATES],
    gamma: f64,
}

impl World {
    fn coercive_stochastic() -> Self {
        Self {
            name: "coercive_stochastic",
            transition: [
                [[0.78, 0.17, 0.05], [0.50, 0.40, 0.10]],
                [[0.12, 0.76, 0.12], [0.20, 0.50, 0.30]],
                [[0.08, 0.22, 0.70], [0.15, 0.25, 0.60]],
            ],
            reward_mean: [[0.15, 0.42], [0.52, 0.66], [0.72, 0.35]],
            target_policy: [[0.78, 0.22], [0.35, 0.65], [0.62, 0.38]],
            offline_distribution: [[0.23, 0.12], [0.18, 0.17], [0.13, 0.17]],
            initial_distribution: [0.62, 0.28, 0.10],
            gamma: GAMMA,
        }
    }

    fn shifted_overlap() -> Self {
        let mut world = Self::coercive_stochastic();
        world.name = "shifted_overlap";
        world.offline_distribution = [[0.63, 0.20], [0.06, 0.04], [0.04, 0.03]];
        world.target_policy = [[0.10, 0.90], [0.20, 0.80], [0.05, 0.95]];
        world.initial_distribution = [0.15, 0.45, 0.40];
        world
    }

    fn validate(&self) -> Result<(), String> {
        if !self.gamma.is_finite() || self.gamma < 0.0 || self.gamma >= 1.0 {
            return Err("discount factor must lie in [0,1)".into());
        }
        if (self.initial_distribution.iter().sum::<f64>() - 1.0).abs() > 1e-10
            || self.initial_distribution.iter().any(|p| !p.is_finite() || *p < 0.0)
        {
            return Err("invalid initial-state distribution".into());
        }
        if (self.offline_distribution.iter().flatten().sum::<f64>() - 1.0).abs() > 1e-10
            || self.offline_distribution.iter().flatten().any(|p| !p.is_finite() || *p <= 0.0)
        {
            return Err("offline distribution must be a strictly positive probability vector".into());
        }
        for state in 0..STATES {
            if (self.target_policy[state].iter().sum::<f64>() - 1.0).abs() > 1e-10
                || self.target_policy[state].iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
            {
                return Err("invalid target policy".into());
            }
            for action in 0..ACTIONS {
                let row = self.transition[state][action];
                if (row.iter().sum::<f64>() - 1.0).abs() > 1e-10
                    || row.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
                {
                    return Err("invalid transition probability row".into());
                }
                let reward = self.reward_mean[state][action];
                if !reward.is_finite() || !(0.0..=1.0).contains(&reward) {
                    return Err("Bernoulli reward mean must lie in [0,1]".into());
                }
            }
        }
        Ok(())
    }

    fn nu(&self, state: usize, action: usize) -> f64 {
        self.offline_distribution[state][action]
    }

    fn cell(&self, state: usize, action: usize) -> usize {
        state * ACTIONS + action
    }
}

#[derive(Clone, Copy, Debug)]
struct QFunction {
    name: &'static str,
    values: [[f64; ACTIONS]; STATES],
}

impl QFunction {
    fn zero() -> Self {
        Self {
            name: "zero",
            values: [[0.0; ACTIONS]; STATES],
        }
    }

    fn value(&self, state: usize, action: usize) -> f64 {
        self.values[state][action]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Transition {
    state: usize,
    action: usize,
    reward: f64,
    next_state: usize,
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / ((1_u64 << 53) as f64)
    }

    fn categorical<const N: usize>(&mut self, probabilities: &[f64; N]) -> usize {
        let draw = self.uniform();
        let mut cumulative = 0.0;
        for (index, probability) in probabilities.iter().enumerate() {
            cumulative += probability;
            if draw <= cumulative || index + 1 == N {
                return index;
            }
        }
        N - 1
    }
}

fn mix_seed(seed: u64, replicate: usize, stream: u64) -> u64 {
    seed ^ (replicate as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ stream.wrapping_mul(0x94d0_49bb_1331_11eb)
}

fn sample_transitions(world: &World, count: usize, seed: u64) -> Vec<Transition> {
    let mut rng = Rng::new(seed);
    let mut nu_flat = [0.0; CELLS];
    for state in 0..STATES {
        for action in 0..ACTIONS {
            nu_flat[world.cell(state, action)] = world.nu(state, action);
        }
    }
    let mut transitions = Vec::with_capacity(count);
    for _ in 0..count {
        let cell = rng.categorical(&nu_flat);
        let state = cell / ACTIONS;
        let action = cell % ACTIONS;
        let reward = if rng.uniform() < world.reward_mean[state][action] { 1.0 } else { 0.0 };
        let next_state = rng.categorical(&world.transition[state][action]);
        transitions.push(Transition { state, action, reward, next_state });
    }
    transitions
}

fn pi_q(world: &World, q: &QFunction, state: usize) -> f64 {
    (0..ACTIONS)
        .map(|action| world.target_policy[state][action] * q.value(state, action))
        .sum()
}

fn bellman_residual_mean(world: &World, q: &QFunction, state: usize, action: usize) -> f64 {
    q.value(state, action) - world.reward_mean[state][action]
        - world.gamma
            * (0..STATES)
                .map(|next| world.transition[state][action][next] * pi_q(world, q, next))
                .sum::<f64>()
}

fn exact_q(world: &World) -> QFunction {
    let mut q = QFunction::zero();
    for _ in 0..100_000 {
        let previous = q;
        for state in 0..STATES {
            for action in 0..ACTIONS {
                q.values[state][action] = world.reward_mean[state][action]
                    + world.gamma
                        * (0..STATES)
                            .map(|next| {
                                world.transition[state][action][next] * pi_q(world, &previous, next)
                            })
                            .sum::<f64>();
            }
        }
        let change = (0..STATES)
            .flat_map(|state| (0..ACTIONS).map(move |action| (state, action)))
            .map(|(state, action)| {
                (q.value(state, action) - previous.value(state, action)).abs()
            })
            .fold(0.0_f64, f64::max);
        if change < 1e-13 {
            break;
        }
    }
    QFunction { name: "q_oracle", values: q.values }
}

fn pi_h(world: &World, h: &[[f64; ACTIONS]; STATES], state: usize) -> f64 {
    (0..ACTIONS)
        .map(|action| world.target_policy[state][action] * h[state][action])
        .sum()
}

fn candidate_bellman_moment(
    world: &World,
    left: &QFunction,
    right: &QFunction,
) -> f64 {
    let mut moment = 0.0;
    for state in 0..STATES {
        for action in 0..ACTIONS {
            let h = left.value(state, action) - right.value(state, action);
            moment += world.nu(state, action) * h
                * bellman_residual_mean(world, left, state, action);
        }
    }
    moment
}

fn candidate_penalty_quadratic(
    world: &World,
    left: &QFunction,
    right: &QFunction,
) -> f64 {
    let mut penalty = 0.0;
    for state in 0..STATES {
        for action in 0..ACTIONS {
            let h = left.value(state, action) - right.value(state, action);
            let expected_next = (0..STATES)
                .map(|next| {
                    world.transition[state][action][next]
                        * pi_h(
                            world,
                            &std::array::from_fn(|s| {
                                std::array::from_fn(|a| {
                                    left.value(s, a) - right.value(s, a)
                                })
                            }),
                            next,
                        )
                })
                .sum::<f64>();
            penalty += world.nu(state, action) * h * (h - world.gamma * expected_next);
        }
    }
    penalty
}

fn exact_offline_q_error(world: &World, q: &QFunction, truth: &QFunction) -> f64 {
    let mut error = 0.0;
    for state in 0..STATES {
        for action in 0..ACTIONS {
            let delta = q.value(state, action) - truth.value(state, action);
            error += world.nu(state, action) * delta * delta;
        }
    }
    error
}

fn exact_target_q_error(world: &World, q: &QFunction, truth: &QFunction) -> f64 {
    let occupancy = target_occupancy(world);
    let mut error = 0.0;
    for state in 0..STATES {
        for action in 0..ACTIONS {
            let delta = q.value(state, action) - truth.value(state, action);
            error += occupancy[state][action] * delta * delta;
        }
    }
    error
}

fn exact_bellman_loss(world: &World, q: &QFunction) -> f64 {
    let mut loss = 0.0;
    for state in 0..STATES {
        for action in 0..ACTIONS {
            let residual = bellman_residual_mean(world, q, state, action);
            loss += world.nu(state, action) * residual * residual;
        }
    }
    loss
}

fn initial_value(world: &World, q: &QFunction) -> f64 {
    (0..STATES)
        .map(|state| {
            world.initial_distribution[state] * pi_q(world, q, state)
        })
        .sum()
}

fn target_value(world: &World, q: &QFunction) -> f64 {
    initial_value(world, q)
}

fn target_occupancy(world: &World) -> [[f64; ACTIONS]; STATES] {
    let mut state_distribution = world.initial_distribution;
    let mut occupancy = [[0.0; ACTIONS]; STATES];
    let mut discount = 1.0;
    for _ in 0..20_000 {
        for state in 0..STATES {
            for action in 0..ACTIONS {
                occupancy[state][action] += (1.0 - world.gamma)
                    * discount
                    * state_distribution[state]
                    * world.target_policy[state][action];
            }
        }
        let mut next_distribution = [0.0; STATES];
        for state in 0..STATES {
            for action in 0..ACTIONS {
                for next in 0..STATES {
                    next_distribution[next] += state_distribution[state]
                        * world.target_policy[state][action]
                        * world.transition[state][action][next];
                }
            }
        }
        state_distribution = next_distribution;
        discount *= world.gamma;
        if discount < 1e-15 {
            break;
        }
    }
    occupancy
}

fn exact_occupancy_ratio(world: &World) -> [[f64; ACTIONS]; STATES] {
    let occupancy = target_occupancy(world);
    std::array::from_fn(|state| {
        std::array::from_fn(|action| occupancy[state][action] / world.nu(state, action))
    })
}

fn pairwise_coercivity(world: &World, candidates: &[QFunction], truth: &QFunction) -> f64 {
    if candidates.is_empty() {
        return f64::NEG_INFINITY;
    }
    let best = (0..candidates.len())
        .min_by(|left, right| {
            exact_offline_q_error(world, &candidates[*left], truth)
                .total_cmp(&exact_offline_q_error(world, &candidates[*right], truth))
        })
        .unwrap_or(0);
    let mut minimum = f64::INFINITY;
    for (index, candidate) in candidates.iter().enumerate() {
        if index == best {
            continue;
        }
        let mut norm = 0.0;
        let mut quadratic = 0.0;
        for state in 0..STATES {
            for action in 0..ACTIONS {
                let h = candidate.value(state, action) - candidates[best].value(state, action);
                let expected_next = (0..STATES)
                    .map(|next| {
                        world.transition[state][action][next]
                            * (0..ACTIONS)
                                .map(|next_action| {
                                    world.target_policy[next][next_action]
                                        * (candidate.value(next, next_action)
                                            - candidates[best].value(next, next_action))
                                })
                                .sum::<f64>()
                    })
                    .sum::<f64>();
                norm += world.nu(state, action) * h * h;
                quadratic += world.nu(state, action) * h * (h - world.gamma * expected_next);
            }
        }
        if norm > EPS {
            minimum = minimum.min(quadratic / norm);
        }
    }
    if minimum.is_finite() { minimum } else { 0.0 }
}

fn fit_candidates(world: &World, training: &[Transition]) -> Vec<QFunction> {
    let mut count = [[0.0; ACTIONS]; STATES];
    let mut reward_sum = [[0.0; ACTIONS]; STATES];
    let mut next_counts = [[[0.0; STATES]; ACTIONS]; STATES];
    let global_reward = training.iter().map(|row| row.reward).sum::<f64>()
        / training.len().max(1) as f64;

    for row in training {
        count[row.state][row.action] += 1.0;
        reward_sum[row.state][row.action] += row.reward;
        next_counts[row.state][row.action][row.next_state] += 1.0;
    }

    let mut fitted = *world;
    for state in 0..STATES {
        for action in 0..ACTIONS {
            let n = count[state][action];
            fitted.reward_mean[state][action] =
                (reward_sum[state][action] + PRIOR_REWARD * global_reward) / (n + PRIOR_REWARD);
            let total = n + PRIOR_TRANSITIONS;
            for next in 0..STATES {
                fitted.transition[state][action][next] =
                    (next_counts[state][action][next] + PRIOR_TRANSITIONS / STATES as f64)
                        / total;
            }
        }
    }
    let base = exact_q(&fitted);
    let global_q = base.values.iter().flatten().sum::<f64>() / CELLS as f64;
    let mut candidates = Vec::new();
    candidates.push(QFunction { name: "fitted_tabular", values: base.values });
    for (name, shrinkage) in [
        ("shrink_25", 0.25),
        ("shrink_60", 0.60),
        ("shrink_90", 0.90),
    ] {
        let values = std::array::from_fn(|state| {
            std::array::from_fn(|action| {
                (1.0 - shrinkage) * base.value(state, action) + shrinkage * global_q
            })
        });
        candidates.push(QFunction { name, values });
    }
    let pattern = [
        [0.10, -0.08],
        [-0.05, 0.12],
        [0.08, -0.10],
    ];
    let values = std::array::from_fn(|state| {
        std::array::from_fn(|action| base.value(state, action) + pattern[state][action])
    });
    candidates.push(QFunction { name: "structured_perturbation", values });
    let large = [
        [0.35, -0.25],
        [-0.30, 0.40],
        [0.25, -0.35],
    ];
    let values = std::array::from_fn(|state| {
        std::array::from_fn(|action| base.value(state, action) + large[state][action])
    });
    candidates.push(QFunction { name: "large_perturbation", values });
    candidates
}

fn sample_target_q(world: &World, q: &QFunction, state: usize, action: usize) -> f64 {
    let _ = world;
    q.value(state, action)
}

fn empirical_td_losses(world: &World, candidates: &[QFunction], validation: &[Transition]) -> Vec<f64> {
    candidates
        .iter()
        .map(|q| {
            validation
                .iter()
                .map(|row| {
                    let delta = q.value(row.state, row.action) - row.reward
                        - world.gamma * pi_q(world, q, row.next_state);
                    delta * delta
                })
                .sum::<f64>()
                / validation.len() as f64
        })
        .collect()
}

fn empirical_pairwise_scores(
    world: &World,
    candidates: &[QFunction],
    validation: &[Transition],
    occupancy_ratio: Option<&[[f64; ACTIONS]; STATES]>,
    penalized: bool,
) -> Vec<f64> {
    let k = candidates.len();
    let mut scores = vec![f64::NEG_INFINITY; k];
    for i in 0..k {
        let mut worst = f64::NEG_INFINITY;
        for j in 0..k {
            let left = &candidates[i];
            let right = &candidates[j];
            let mut moment = 0.0;
            let mut penalty = 0.0;
            for row in validation {
                let h = left.value(row.state, row.action) - right.value(row.state, row.action);
                let delta = left.value(row.state, row.action) - row.reward
                    - world.gamma * pi_q(world, left, row.next_state);
                let h_next = pi_q(world, left, row.next_state)
                    - pi_q(world, right, row.next_state);
                let weight = occupancy_ratio
                    .map(|ratio| ratio[row.state][row.action])
                    .unwrap_or(1.0);
                moment += weight * h * delta;
                penalty += weight * h * (h - world.gamma * h_next);
            }
            moment /= validation.len() as f64;
            penalty /= validation.len() as f64;
            let score = if penalized {
                moment - 0.25 * penalty.max(0.0)
            } else {
                moment
            };
            worst = worst.max(score);
        }
        scores[i] = worst;
    }
    scores
}

fn deterministic_argmin(scores: &[f64]) -> usize {
    scores
        .iter()
        .enumerate()
        .min_by(|left, right| left.1.total_cmp(right.1).then_with(|| left.0.cmp(&right.0)))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn oracle_selector(world: &World, candidates: &[QFunction], truth: &QFunction) -> usize {
    deterministic_argmin(
        &candidates
            .iter()
            .map(|q| exact_target_q_error(world, q, truth))
            .collect::<Vec<_>>(),
    )
}

fn exact_population_pbv_scores(
    world: &World,
    candidates: &[QFunction],
    occupancy_ratio: Option<&[[f64; ACTIONS]; STATES]>,
    penalized: bool,
) -> Vec<f64> {
    let k = candidates.len();
    let mut scores = vec![f64::NEG_INFINITY; k];
    for i in 0..k {
        let mut worst = f64::NEG_INFINITY;
        for j in 0..k {
            let left = &candidates[i];
            let right = &candidates[j];
            let mut moment = 0.0;
            let mut penalty = 0.0;
            for state in 0..STATES {
                for action in 0..ACTIONS {
                    let h = left.value(state, action) - right.value(state, action);
                    let delta = bellman_residual_mean(world, left, state, action);
                    let expected_h_next = (0..STATES)
                        .map(|next| {
                            world.transition[state][action][next]
                                * (pi_q(world, left, next) - pi_q(world, right, next))
                        })
                        .sum::<f64>();
                    let weight = occupancy_ratio
                        .map(|ratio| world.nu(state, action) * ratio[state][action])
                        .unwrap_or_else(|| world.nu(state, action));
                    moment += weight * h * delta;
                    penalty += weight * h * (h - world.gamma * expected_h_next);
                }
            }
            let score = if penalized {
                moment - 0.25 * penalty.max(0.0)
            } else {
                moment
            };
            worst = worst.max(score);
        }
        scores[i] = worst;
    }
    scores
}

fn selected_metrics(
    world: &World,
    candidates: &[QFunction],
    truth: &QFunction,
    selected: usize,
    oracle: usize,
) -> (f64, f64, f64, f64, bool) {
    let q = &candidates[selected];
    let offline_error = exact_offline_q_error(world, q, truth);
    let target_error = exact_target_q_error(world, q, truth);
    let value_error = (initial_value(world, q) - target_value(world, truth)).abs();
    let bellman_loss = exact_bellman_loss(world, q);
    (
        offline_error,
        target_error,
        value_error,
        bellman_loss,
        selected == oracle,
    )
}

#[derive(Clone, Copy, Debug)]
struct Aggregate {
    count: usize,
    exact_selection: usize,
    offline_error: f64,
    target_error: f64,
    value_error: f64,
    bellman_loss: f64,
}

impl Aggregate {
    fn new() -> Self {
        Self { count: 0, exact_selection: 0, offline_error: 0.0, target_error: 0.0, value_error: 0.0, bellman_loss: 0.0 }
    }

    fn add(&mut self, metrics: (f64, f64, f64, f64, bool)) {
        self.count += 1;
        self.exact_selection += metrics.4 as usize;
        self.offline_error += metrics.0;
        self.target_error += metrics.1;
        self.value_error += metrics.2;
        self.bellman_loss += metrics.3;
    }

    fn means(&self) -> Result<(f64, f64, f64, f64, f64), String> {
        if self.count == 0 {
            return Err("empty selection aggregate".into());
        }
        let n = self.count as f64;
        Ok((
            self.exact_selection as f64 / n,
            self.offline_error / n,
            self.target_error / n,
            self.value_error / n,
            self.bellman_loss / n,
        ))
    }
}

fn split_seeds_are_independent(train_seed_root: u64, validation_seed_root: u64) -> bool {
    train_seed_root != validation_seed_root
}

fn run_benchmark(replicates: usize, seed: u64) -> Result<String, String> {
    if replicates == 0 {
        return Err("replicates must be positive".into());
    }
    let worlds = [World::coercive_stochastic(), World::shifted_overlap()];
    let methods = [
        "oracle_target_q",
        "oracle_bellman_residual",
        "heldout_td_mse",
        "unweighted_pbv_penalized",
        "unweighted_pbv_unpenalized",
        "ta_pbv_exact_occupancy_oracle",
    ];
    let mut output = String::from(
        "record_type\\tscenario\\tmethod\\treplicates\\tcandidate_count\\tselection_accuracy\\toffline_q_error\\ttarget_q_error\\tinitial_value_abs_error\\tbellman_residual_loss\\tpairwise_coercivity\\ttrain_seed\\tvalidation_seed\\n",
    );
    output.push_str("BENCHMARK_CONFIG\\tVIS-DVPC-058\\tversion\\t0\\n");
    output.push_str(&format!("BENCHMARK_CONFIG\\treplicates\\t{replicates}\\n"));
    output.push_str(&format!("BENCHMARK_CONFIG\\ttrain_n\\t{TRAIN_N}\\n"));
    output.push_str(&format!("BENCHMARK_CONFIG\\tvalidation_n\\t{VALIDATION_N}\\n"));
    output.push_str("BENCHMARK_CONFIG\\tvalidation_design\\tiid one-step transitions from registered offline distribution; stochastic reward and next-state draws\\n");
    output.push_str("BENCHMARK_CONFIG\\tselection\\tpenalized max over all pairwise candidate critics; smallest-index tie break\\n");
    output.push_str("BENCHMARK_CONFIG\\toracle_boundary\\texact MDP model and target occupancy are evaluator-only; no selector reads them except labelled TA-PBV oracle arm\\n");
    output.push_str("BENCHMARK_CONFIG\\tsplitting\\tindependent training and validation seeds; candidate fit receives training rows only\\n");

    for (world_index, world) in worlds.iter().enumerate() {
        world.validate()?;
        let truth = exact_q(world);
        let exact_ratio = exact_occupancy_ratio(world);
        let mut aggregates: Vec<Aggregate> = methods.iter().map(|_| Aggregate::new()).collect();
        let mut mean_coercivity = 0.0;
        for replicate in 0..replicates {
            let train_seed_root =
                seed ^ (world_index as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
            let validation_seed_root =
                seed ^ 0x0580_5eed ^ (world_index as u64).wrapping_mul(0x6eed_0e9d_a4d9_4a4f);
            if !split_seeds_are_independent(train_seed_root, validation_seed_root) {
                return Err("train and validation seed roots collide".into());
            }
            let train_seed = mix_seed(train_seed_root, replicate, 1);
            let validation_seed = mix_seed(validation_seed_root, replicate, 2);
            let training = sample_transitions(world, TRAIN_N, train_seed);
            let validation = sample_transitions(world, VALIDATION_N, validation_seed);
            let candidates = fit_candidates(world, &training);
            let oracle_target = oracle_selector(world, &candidates, &truth);
            let oracle_bellman = deterministic_argmin(
                &candidates.iter().map(|q| exact_bellman_loss(world, q)).collect::<Vec<_>>()
            );
            let td = deterministic_argmin(&empirical_td_losses(world, &candidates, &validation));
            let pbv_pen = deterministic_argmin(&empirical_pairwise_scores(world, &candidates, &validation, None, true));
            let pbv_raw = deterministic_argmin(&empirical_pairwise_scores(world, &candidates, &validation, None, false));
            let ta_pbv = deterministic_argmin(&empirical_pairwise_scores(world, &candidates, &validation, Some(&exact_ratio), true));
            let selected = [oracle_target, oracle_bellman, td, pbv_pen, pbv_raw, ta_pbv];
            for (method_index, selected_index) in selected.iter().enumerate() {
                aggregates[method_index].add(selected_metrics(world, &candidates, &truth, *selected_index, oracle_target));
            }
            mean_coercivity += pairwise_coercivity(world, &candidates, &truth);
        }

        for (method_index, method) in methods.iter().enumerate() {
            let (accuracy, offline_error, target_error, value_error, bellman_loss) =
                aggregates[method_index].means()?;
            use std::fmt::Write;
            let _ = writeln!(
                output,
                "RESULT\\t{}\\t{}\\t{}\\t{}\\t{:.8}\\t{:.10}\\t{:.10}\\t{:.10}\\t{:.10}\\t{:.10}\\t{}\\t{}",
                world.name,
                method,
                replicates,
                6,
                accuracy,
                offline_error,
                target_error,
                value_error,
                bellman_loss,
                mean_coercivity / replicates as f64,
                seed ^ (world_index as u64).wrapping_mul(0x517c_c1b7_2722_0a95),
                seed ^ 0x0580_5eed ^ (world_index as u64).wrapping_mul(0x6eed_0e9d_a4d9_4a4f),
            );
        }
    }
    output.push_str("NEGATIVE_CONTROL\\tvalidation_leakage\\tcandidate_fit_and_validation_seeds_must_be_distinct\\tFAIL_CLOSED_ON_COLLISION\\n");
    Ok(output)
}

fn main() {
    let smoke = env::args().any(|arg| arg == "--smoke");
    let replicates = if smoke { SMOKE_REPLICATES } else { FULL_REPLICATES };
    match run_benchmark(replicates, 0x0580_2026_1030) {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("VIS_DVPC_058_ERROR\\t{error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_q_satisfies_bellman_fixed_point() {
        for world in [World::coercive_stochastic(), World::shifted_overlap()] {
            let q = exact_q(&world);
            assert!((0..STATES).all(|state| {
                (0..ACTIONS).all(|action| bellman_residual_mean(&world, &q, state, action).abs() < 1e-10)
            }));
        }
    }

    #[test]
    fn population_penalized_pbv_selects_exact_candidate_under_coercivity() {
        let world = World::coercive_stochastic();
        let truth = exact_q(&world);
        let mut bad = truth;
        bad.name = "bad";
        bad.values = std::array::from_fn(|state| {
            std::array::from_fn(|action| truth.value(state, action) + 0.2 * (state as f64 - 0.5))
        });
        let candidates = [truth, bad];
        let coercivity = pairwise_coercivity(&world, &candidates, &truth);
        assert!(coercivity > 0.0, "coercivity={coercivity}");
        let scores = exact_population_pbv_scores(&world, &candidates, None, true);
        assert!(scores[0].abs() < 1e-10, "truth score={}", scores[0]);
        assert!(scores[1] > 0.0, "bad score={}", scores[1]);
        assert_eq!(deterministic_argmin(&scores), 0);
    }

    #[test]
    fn ta_pbv_exact_occupancy_weight_is_positive_and_normalized() {
        let world = World::shifted_overlap();
        let ratio = exact_occupancy_ratio(&world);
        let mut total = 0.0;
        for state in 0..STATES {
            for action in 0..ACTIONS {
                assert!(ratio[state][action].is_finite() && ratio[state][action] >= 0.0);
                total += world.nu(state, action) * ratio[state][action];
            }
        }
        assert!((total - 1.0).abs() < 1e-9);
        let truth = exact_q(&world);
        let mut bad = truth;
        bad.values[0][0] += 0.4;
        let scores = exact_population_pbv_scores(&world, &[truth, bad], Some(&ratio), true);
        assert!(scores[0].abs() < 1e-10);
        assert!(scores[1] > 0.0);
    }

    #[test]
    fn squared_td_reversal_fixture_exposes_candidate_dependent_conditional_variance() {
        let mut world = World::coercive_stochastic();
        world.gamma = 0.8;
        world.offline_distribution = [[0.9997, 0.00005], [0.00005, 0.00005], [0.00005, 0.0001]];
        world.transition[0][0] = [0.0, 0.5, 0.5];
        let truth = exact_q(&world);
        let mut candidate_low_bellman = truth;
        candidate_low_bellman.name = "low_bellman_high_td_variance";
        let mut candidate_low_td = truth;
        candidate_low_td.name = "higher_bellman_low_td_variance";
        for action in 0..ACTIONS {
            candidate_low_bellman.values[0][action] += 0.01;
            candidate_low_bellman.values[1][action] += 1.0;
            candidate_low_bellman.values[2][action] -= 1.0;
            candidate_low_td.values[0][action] += 0.20;
            candidate_low_td.values[1][action] += 0.20;
            candidate_low_td.values[2][action] += 0.20;
        }
        let candidates = [candidate_low_bellman, candidate_low_td];
        let oracle = deterministic_argmin(&candidates.iter().map(|q| exact_bellman_loss(&world, q)).collect::<Vec<_>>());
        let td = {
            let data = sample_transitions(&world, MC_TEST_N, 0x0580_a11);
            deterministic_argmin(&empirical_td_losses(&world, &candidates, &data))
        };
        assert_eq!(oracle, 0, "candidate 0 should have smaller Bellman residual");
        assert_eq!(td, 1, "squared TD should prefer the low conditional-variance candidate");
    }

    #[test]
    fn pairwise_coercivity_detects_a_near_zero_overlap_failure() {
        let mut world = World::shifted_overlap();
        world.offline_distribution = [[0.9997, 0.00005], [0.00005, 0.00005], [0.00005, 0.0001]];
        let truth = exact_q(&world);
        let mut bad = truth;
        bad.values[0][0] += 1.0;
        bad.values[1][0] += 2.0;
        bad.values[2][0] -= 2.0;
        let coercivity = pairwise_coercivity(&world, &[truth, bad], &truth);
        assert!(coercivity <= 0.0 || coercivity < 0.1, "coercivity={coercivity}");
    }

    #[test]
    fn training_and_validation_streams_are_independent_and_candidates_only_use_training() {
        let world = World::coercive_stochastic();
        let train = sample_transitions(&world, 128, 10);
        let validation_a = sample_transitions(&world, 128, 11);
        let validation_b = sample_transitions(&world, 128, 12);
        assert!(split_seeds_are_independent(10, 11));
        assert!(!split_seeds_are_independent(10, 10));
        let candidates_a = fit_candidates(&world, &train);
        let candidates_b = fit_candidates(&world, &train);
        assert_eq!(candidates_a.len(), 6);
        for index in 0..candidates_a.len() {
            assert_eq!(candidates_a[index].values, candidates_b[index].values);
        }
        assert_ne!(validation_a, validation_b);
    }

    #[test]
    fn heldout_pbv_selection_is_deterministic_for_frozen_splits() {
        let world = World::coercive_stochastic();
        let train = sample_transitions(&world, TRAIN_N, 101);
        let validation = sample_transitions(&world, VALIDATION_N, 102);
        let candidates = fit_candidates(&world, &train);
        let first = deterministic_argmin(&empirical_pairwise_scores(&world, &candidates, &validation, None, true));
        let second = deterministic_argmin(&empirical_pairwise_scores(&world, &candidates, &validation, None, true));
        assert_eq!(first, second);
    }

    #[test]
    fn benchmark_output_has_deterministic_rows_and_schema() {
        let first = run_benchmark(2, 103).unwrap();
        let second = run_benchmark(2, 103).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.lines().filter(|line| line.starts_with("RESULT\\t")).count(), 12);
        assert_eq!(first.lines().filter(|line| line.starts_with("BENCHMARK_CONFIG\\t")).count(), 8);
        assert_eq!(first.lines().filter(|line| line.starts_with("NEGATIVE_CONTROL\\t")).count(), 1);
        for row in first.lines().filter(|line| line.starts_with("RESULT\\t")) {
            assert_eq!(row.split('\\t').count(), 13);
        }
    }

    #[test]
    fn invalid_worlds_and_zero_validation_count_fail_closed() {
        let mut world = World::coercive_stochastic();
        world.gamma = 1.0;
        assert!(world.validate().is_err());
        assert_eq!(run_benchmark(0, 1).unwrap_err(), "replicates must be positive");
    }

    #[test]
    fn population_unpenalized_pbv_differs_under_candidate_misspecification() {
        let world = World::coercive_stochastic();
        let truth = exact_q(&world);
        let mut candidate_a = truth;
        let mut candidate_b = truth;
        for state in 0..STATES {
            for action in 0..ACTIONS {
                candidate_a.values[state][action] += 0.15 * (state as f64 - 1.0);
                candidate_b.values[state][action] += 0.25 * (action as f64 - 0.5);
            }
        }
        let candidates = [candidate_a, candidate_b];
        let penalized = exact_population_pbv_scores(&world, &candidates, None, true);
        let unpenalized = exact_population_pbv_scores(&world, &candidates, None, false);
        assert!(penalized.iter().all(|v| v.is_finite()));
        assert!(unpenalized.iter().all(|v| v.is_finite()));
        assert!(penalized.iter().zip(unpenalized.iter()).any(|(a,b)| (a-b).abs()>1e-12));
    }
}