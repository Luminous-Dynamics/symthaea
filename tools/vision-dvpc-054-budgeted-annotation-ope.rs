//! VIS-DVPC-054: budgeted multi-source counterfactual annotation for OPE.
//!
//! Run:
//!   rustc --edition=2024 -O --test tools/vision-dvpc-054-budgeted-annotation-ope.rs -o /tmp/vis-dvpc-054-tests
//!   /tmp/vis-dvpc-054-tests
//!   rustc --edition=2024 -O tools/vision-dvpc-054-budgeted-annotation-ope.rs -o /tmp/vis-dvpc-054
//!   /tmp/vis-dvpc-054 --smoke
//!   /tmp/vis-dvpc-054
//!
//! Counterfactual annotations augment only the reward-model / DM component.
//! The importance-sampling residual always uses the factual logged reward and
//! its exact behavior propensity. Annotation acquisition never repairs zero
//! factual support. The small allocation problem is solved exactly by
//! enumerating per-context choices and dynamic-programming over the shared
//! integer budget; greedy baselines are measured against that certificate.
#![deny(unsafe_code)]

use std::env;
use std::f64::consts::TAU;

const CONTEXTS: usize = 2;
const ACTIONS: usize = 3;
const SOURCES: usize = 3;
const EVAL_N: usize = 128;
const FULL_REPLICATES: usize = 4096;
const SMOKE_REPLICATES: usize = 48;
const MC_TEST_REPLICATES: usize = 4096;
const ACTION_CAP_PER_CONTEXT: u8 = 3;
const BUDGET_TOKENS: u32 = 8;
const EPS: f64 = 1e-12;

#[derive(Clone, Copy, Debug)]
struct World {
    name: &'static str,
    context_probability: [f64; CONTEXTS],
    target: [[f64; ACTIONS]; CONTEXTS],
    behavior: [[f64; ACTIONS]; CONTEXTS],
    reward_mean: [[f64; ACTIONS]; CONTEXTS],
    factual_count: [[u8; ACTIONS]; CONTEXTS],
    source_bias: [[[f64; SOURCES]; ACTIONS]; CONTEXTS],
    source_excess_variance: [[[f64; SOURCES]; ACTIONS]; CONTEXTS],
    /// Variances of low-rank shared annotation error factors.
    shared_factor_variance: [f64; 2],
    /// Source-specific loading on each shared factor. The induced covariance
    /// matrix is a sum of rank-one PSD matrices.
    source_shared_loadings: [[f64; 2]; SOURCES],
    source_cost: [u32; SOURCES],
    context_annotation_cap: [u8; CONTEXTS],
    budget: u32,
    default_reward: [[f64; ACTIONS]; CONTEXTS],
}

impl World {
    fn mixed_bias_cancellation() -> Self {
        Self {
            name: "mixed_bias_cancellation",
            context_probability: [0.60, 0.40],
            target: [[0.65, 0.25, 0.10], [0.10, 0.30, 0.60]],
            behavior: [[0.70, 0.20, 0.10], [0.20, 0.30, 0.50]],
            reward_mean: [[0.20, 0.55, 0.85], [0.75, 0.35, 0.15]],
            factual_count: [[2, 2, 2], [2, 2, 2]],
            source_bias: [
                [
                    [0.01, 0.20, -0.20],
                    [-0.01, -0.15, 0.15],
                    [0.00, 0.12, -0.12],
                ],
                [
                    [0.00, -0.13, 0.13],
                    [0.01, 0.16, -0.16],
                    [-0.01, -0.14, 0.14],
                ],
            ],
            source_excess_variance: [[[0.01, 0.08, 0.08]; ACTIONS], [[0.01, 0.08, 0.08]; ACTIONS]],
            shared_factor_variance: [0.0; 2],
            source_shared_loadings: [[1.0, 0.0]; SOURCES],
            source_cost: [3, 1, 2],
            context_annotation_cap: [ACTION_CAP_PER_CONTEXT; CONTEXTS],
            budget: BUDGET_TOKENS,
            default_reward: [[0.50; ACTIONS]; CONTEXTS],
        }
    }

    fn heterogeneous_source_quality() -> Self {
        let mut world = Self::mixed_bias_cancellation();
        world.name = "heterogeneous_source_quality";
        world.source_bias = [
            [
                [0.01, 0.02, -0.01],
                [0.00, 0.03, -0.02],
                [-0.02, 0.02, 0.00],
            ],
            [
                [0.16, -0.12, 0.08],
                [-0.12, 0.15, -0.10],
                [0.10, -0.14, 0.16],
            ],
        ];
        world.source_excess_variance =
            [[[0.01, 0.10, 0.06]; ACTIONS], [[0.10, 0.01, 0.06]; ACTIONS]];
        world
    }

    fn estimated_profiles(&self) -> Self {
        let mut estimated = *self;
        let bias_shift = [0.02, -0.04, 0.03];
        let variance_scale = [1.25, 0.80, 1.35];
        for x in 0..CONTEXTS {
            for a in 0..ACTIONS {
                for k in 0..SOURCES {
                    estimated.source_bias[x][a][k] += bias_shift[k];
                    estimated.source_excess_variance[x][a][k] *= variance_scale[k];
                }
            }
        }
        // The simple plugin pilot deliberately assumes independent source
        // errors. The hidden true shared-error factors remain in the evaluation
        // world, making under-estimated dependence visible as allocation regret.
        estimated.shared_factor_variance = [0.0; 2];
        estimated
    }

    fn validate(&self) -> Result<(), String> {
        if (self.context_probability.iter().sum::<f64>() - 1.0).abs() > 1e-10
            || self
                .context_probability
                .iter()
                .any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
        {
            return Err("invalid context distribution".into());
        }
        if self.source_cost.iter().any(|cost| *cost == 0) {
            return Err("annotation source costs must be positive".into());
        }
        if self
            .shared_factor_variance
            .iter()
            .any(|variance| !variance.is_finite() || *variance < 0.0)
            || self
                .source_shared_loadings
                .iter()
                .flatten()
                .any(|loading| !loading.is_finite())
        {
            return Err("invalid shared annotation covariance factor".into());
        }
        for x in 0..CONTEXTS {
            if (self.target[x].iter().sum::<f64>() - 1.0).abs() > 1e-10
                || (self.behavior[x].iter().sum::<f64>() - 1.0).abs() > 1e-10
            {
                return Err("target and behavior policies must normalize".into());
            }
            for a in 0..ACTIONS {
                let mu = self.behavior[x][a];
                if !mu.is_finite() || mu <= 0.0 || mu > 1.0 {
                    return Err("behavior policy must have full action support".into());
                }
                if !self.target[x][a].is_finite()
                    || self.target[x][a] < 0.0
                    || self.target[x][a] > 1.0
                {
                    return Err("invalid target policy probability".into());
                }
                let r = self.reward_mean[x][a];
                if !r.is_finite() || !(0.0..=1.0).contains(&r) {
                    return Err("reward means must lie in [0,1]".into());
                }
                for k in 0..SOURCES {
                    if !self.source_bias[x][a][k].is_finite()
                        || !self.source_excess_variance[x][a][k].is_finite()
                        || self.source_excess_variance[x][a][k] < 0.0
                    {
                        return Err("invalid annotation source profile".into());
                    }
                    let shared = (0..2)
                        .map(|factor| {
                            self.shared_factor_variance[factor]
                                * self.source_shared_loadings[k][factor].powi(2)
                        })
                        .sum::<f64>();
                    let marginal = reward_variance(r) + self.source_excess_variance[x][a][k];
                    if shared > marginal + EPS {
                        return Err(
                            "shared covariance exceeds source marginal error variance".into()
                        );
                    }
                }
            }
            let factual = self.factual_count[x].iter().map(|n| *n as u32).sum::<u32>();
            if self.context_annotation_cap[x] as u32 > factual {
                return Err("annotation capacity exceeds factual fit count".into());
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Allocation {
    /// Number of annotations from each source for each context-action cell.
    counts: [[[u8; SOURCES]; ACTIONS]; CONTEXTS],
}

impl Allocation {
    fn count(&self, x: usize, a: usize, k: usize) -> u8 {
        self.counts[x][a][k]
    }

    fn total_count(&self) -> u32 {
        self.counts
            .iter()
            .flat_map(|by_action| by_action.iter())
            .flat_map(|by_source| by_source.iter())
            .map(|n| *n as u32)
            .sum()
    }

    fn total_by_source(&self, k: usize) -> u32 {
        self.counts
            .iter()
            .flat_map(|by_action| by_action.iter())
            .map(|by_source| by_source[k] as u32)
            .sum()
    }

    fn cost(&self, world: &World) -> u32 {
        let mut cost = 0;
        for x in 0..CONTEXTS {
            for a in 0..ACTIONS {
                for k in 0..SOURCES {
                    cost += self.count(x, a, k) as u32 * world.source_cost[k];
                }
            }
        }
        cost
    }

    fn context_count(&self, x: usize) -> u32 {
        self.counts[x]
            .iter()
            .flat_map(|by_source| by_source.iter())
            .map(|n| *n as u32)
            .sum()
    }

    fn feasible(&self, world: &World) -> bool {
        self.cost(world) <= world.budget
            && (0..CONTEXTS)
                .all(|x| self.context_count(x) <= world.context_annotation_cap[x] as u32)
    }
}

#[derive(Clone, Copy)]
struct ContextOption {
    counts: [[u8; SOURCES]; ACTIONS],
    cost: u32,
    risk_component: f64,
}

#[derive(Clone, Copy, Debug)]
struct ExactResult {
    allocation: Allocation,
    context_options: usize,
    dp_transitions: usize,
}

#[derive(Clone, Copy, Debug)]
enum GreedyRule {
    MaxGain,
    GainPerCost,
}

#[derive(Clone, Copy, Debug)]
struct LocalMixOption {
    counts: [u8; SOURCES],
    score: f64,
    remaining_swap_gain: f64,
}

#[derive(Clone, Debug)]
struct MmRun {
    allocation: Allocation,
    iterations: usize,
    objective_trace: Vec<f64>,
    remaining_swap_gain: f64,
}

#[derive(Clone, Copy)]
struct MmCandidate {
    allocation: Allocation,
    remaining_swap_gain: f64,
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        })
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn uniform(&mut self) -> f64 {
        (((self.next_u64() >> 11) as f64) + 0.5) / ((1_u64 << 53) as f64)
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }
}

fn mix_seed(base: u64, replicate: usize, stream: u64) -> u64 {
    base ^ (replicate as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ stream.wrapping_mul(0x94d0_49bb_1331_11eb)
}

fn target_value(world: &World) -> f64 {
    let mut value = 0.0;
    for x in 0..CONTEXTS {
        for a in 0..ACTIONS {
            value += world.context_probability[x] * world.target[x][a] * world.reward_mean[x][a];
        }
    }
    value
}

fn context_value(world: &World, x: usize) -> f64 {
    (0..ACTIONS)
        .map(|a| world.target[x][a] * world.reward_mean[x][a])
        .sum()
}

fn reward_variance(mean: f64) -> f64 {
    mean * (1.0 - mean)
}

/// Sum of off-diagonal annotation covariances for one context-action
/// cell. For each factor f, a shared latent shock creates pair covariance
/// tau_f^2 * loading[k,f] * loading[l,f] between every distinct annotation
/// draw. This is PSD by construction (diagonal residual covariance plus
/// low-rank positive-semidefinite shared factors).
fn shared_annotation_covariance_numerator(
    world: &World,
    allocation: &Allocation,
    x: usize,
    a: usize,
) -> f64 {
    let mut extra = 0.0;
    for factor in 0..2 {
        let weighted_sum = (0..SOURCES)
            .map(|k| {
                allocation.count(x, a, k) as f64 * world.source_shared_loadings[k][factor]
            })
            .sum::<f64>();
        let diagonal = (0..SOURCES)
            .map(|k| {
                allocation.count(x, a, k) as f64
                    * world.source_shared_loadings[k][factor].powi(2)
            })
            .sum::<f64>();
        extra += world.shared_factor_variance[factor]
            * (weighted_sum * weighted_sum - diagonal);
    }
    extra
}

fn cell_bias_and_variance(
    world: &World,
    allocation: &Allocation,
    x: usize,
    a: usize,
) -> (f64, f64) {
    let nf = world.factual_count[x][a] as f64;
    let annotation_count = (0..SOURCES)
        .map(|k| allocation.count(x, a, k) as f64)
        .sum::<f64>();
    let total = nf + annotation_count;
    let reward_mean = world.reward_mean[x][a];
    if total <= EPS {
        return (world.default_reward[x][a] - reward_mean, 0.0);
    }

    let weighted_bias = (0..SOURCES)
        .map(|k| allocation.count(x, a, k) as f64 * world.source_bias[x][a][k])
        .sum::<f64>();
    let variance_numerator = nf * reward_variance(reward_mean)
        + (0..SOURCES)
            .map(|k| {
                allocation.count(x, a, k) as f64
                    * (reward_variance(reward_mean) + world.source_excess_variance[x][a][k])
            })
            .sum::<f64>()
        + shared_annotation_covariance_numerator(world, allocation, x, a);
    (weighted_bias / total, variance_numerator / (total * total))
}

/// Exact per-event variance of the annotation-augmented DM+IS/DR score.
///
/// The reward model is fitted on a sample independent of the final OPE sample.
/// For each cell, B is the mean error and W the variance of q_hat. The factual
/// residual always uses the observed Bernoulli reward and exact behavior
/// propensity. This expression integrates over both q_hat fitting noise and
/// the logged evaluation observation.
fn context_conditional_variance(world: &World, allocation: &Allocation, x: usize) -> f64 {
    let mut terms = 0.0;
    let mut target_weighted_bias = 0.0;
    for a in 0..ACTIONS {
        let pi = world.target[x][a];
        let mu = world.behavior[x][a];
        let sigma2 = reward_variance(world.reward_mean[x][a]);
        let (bias, q_variance) = cell_bias_and_variance(world, allocation, x, a);
        terms += pi * pi / mu * sigma2;
        terms += pi * pi * (1.0 / mu - 1.0) * q_variance;
        terms += pi * pi / mu * bias * bias;
        target_weighted_bias += pi * bias;
    }
    (terms - target_weighted_bias * target_weighted_bias).max(0.0)
}

fn variance_per_event(world: &World, allocation: &Allocation) -> Result<f64, String> {
    if !allocation.feasible(world) {
        return Err("annotation allocation is infeasible".into());
    }
    let value = target_value(world);
    let mut context_variance = 0.0;
    let mut conditional_variance = 0.0;
    for x in 0..CONTEXTS {
        let d = world.context_probability[x];
        let v_x = context_value(world, x);
        context_variance += d * (v_x - value).powi(2);
        conditional_variance += d * context_conditional_variance(world, allocation, x);
    }
    let total = context_variance + conditional_variance;
    if total.is_finite() && total >= 0.0 {
        Ok(total)
    } else {
        Err("non-finite or negative exact DR variance".into())
    }
}

fn analytic_mse(
    world: &World,
    allocation: &Allocation,
    evaluation_n: usize,
) -> Result<f64, String> {
    if evaluation_n == 0 {
        return Err("evaluation sample size must be positive".into());
    }
    Ok(variance_per_event(world, allocation)? / evaluation_n as f64)
}

fn weighted_abs_model_bias(world: &World, allocation: &Allocation) -> f64 {
    let mut result = 0.0;
    for x in 0..CONTEXTS {
        for a in 0..ACTIONS {
            let (bias, _) = cell_bias_and_variance(world, allocation, x, a);
            result += world.context_probability[x] * world.target[x][a] * bias.abs();
        }
    }
    result
}

fn enumerate_context_options(
    world: &World,
    context: usize,
    allowed_sources: [bool; SOURCES],
) -> Vec<ContextOption> {
    fn recurse(
        world: &World,
        context: usize,
        allowed_sources: [bool; SOURCES],
        position: usize,
        remaining: u8,
        used_cost: u32,
        counts: &mut [[u8; SOURCES]; ACTIONS],
        output: &mut Vec<ContextOption>,
    ) {
        if position == ACTIONS * SOURCES {
            let mut partial = Allocation::default();
            partial.counts[context] = *counts;
            let risk_component = world.context_probability[context]
                * context_conditional_variance(world, &partial, context);
            output.push(ContextOption {
                counts: *counts,
                cost: used_cost,
                risk_component,
            });
            return;
        }

        let action = position / SOURCES;
        let source = position % SOURCES;
        if !allowed_sources[source] {
            counts[action][source] = 0;
            recurse(
                world,
                context,
                allowed_sources,
                position + 1,
                remaining,
                used_cost,
                counts,
                output,
            );
            return;
        }

        let max_by_budget =
            ((world.budget - used_cost) / world.source_cost[source]).min(u8::MAX as u32) as u8;
        let max_count = remaining.min(max_by_budget);
        for count in 0..=max_count {
            counts[action][source] = count;
            recurse(
                world,
                context,
                allowed_sources,
                position + 1,
                remaining - count,
                used_cost + count as u32 * world.source_cost[source],
                counts,
                output,
            );
        }
        counts[action][source] = 0;
    }

    let mut output = Vec::new();
    recurse(
        world,
        context,
        allowed_sources,
        0,
        world.context_annotation_cap[context],
        0,
        &mut [[0; SOURCES]; ACTIONS],
        &mut output,
    );
    output.sort_by(|left, right| {
        left.cost
            .cmp(&right.cost)
            .then_with(|| left.risk_component.total_cmp(&right.risk_component))
    });
    output
}

/// Certified small-instance optimum: enumerate all feasible allocation choices
/// inside each context, then solve the shared budget exactly by dynamic
/// programming. Cross-action coupling is retained inside each context option.
fn exact_dynamic_program(
    world: &World,
    allowed_sources: [bool; SOURCES],
) -> Result<ExactResult, String> {
    let options: [Vec<ContextOption>; CONTEXTS] =
        std::array::from_fn(|x| enumerate_context_options(world, x, allowed_sources));
    let option_count = options.iter().map(Vec::len).sum::<usize>();
    let budget = world.budget as usize;
    let mut dp = vec![vec![f64::INFINITY; budget + 1]; CONTEXTS + 1];
    let mut parent: Vec<Vec<Option<(usize, usize)>>> = vec![vec![None; budget + 1]; CONTEXTS + 1];
    let mut transitions = 0usize;
    dp[0][0] = 0.0;

    for x in 0..CONTEXTS {
        for spent in 0..=budget {
            if !dp[x][spent].is_finite() {
                continue;
            }
            for (option_index, option) in options[x].iter().enumerate() {
                let next_spent = spent + option.cost as usize;
                if next_spent > budget {
                    continue;
                }
                transitions += 1;
                let candidate = dp[x][spent] + option.risk_component;
                if candidate + EPS < dp[x + 1][next_spent] {
                    dp[x + 1][next_spent] = candidate;
                    parent[x + 1][next_spent] = Some((spent, option_index));
                }
            }
        }
    }

    let mut best_spent = 0usize;
    for spent in 1..=budget {
        if dp[CONTEXTS][spent] + EPS < dp[CONTEXTS][best_spent] {
            best_spent = spent;
        }
    }
    if !dp[CONTEXTS][best_spent].is_finite() {
        return Err("exact dynamic program found no feasible allocation".into());
    }

    let mut allocation = Allocation::default();
    let mut cursor = best_spent;
    for x in (0..CONTEXTS).rev() {
        let (previous_spent, option_index) =
            parent[x + 1][cursor].ok_or_else(|| "missing exact-DP backpointer".to_string())?;
        allocation.counts[x] = options[x][option_index].counts;
        cursor = previous_spent;
    }

    if !allocation.feasible(world) {
        return Err("exact-DP backtracking produced an infeasible plan".into());
    }
    Ok(ExactResult {
        allocation,
        context_options: option_count,
        dp_transitions: transitions,
    })
}

fn greedy_allocate(world: &World, rule: GreedyRule) -> Result<Allocation, String> {
    let mut allocation = Allocation::default();
    loop {
        let current = analytic_mse(world, &allocation, EVAL_N)?;
        let mut best: Option<(f64, Allocation)> = None;
        for x in 0..CONTEXTS {
            for a in 0..ACTIONS {
                for k in 0..SOURCES {
                    let mut candidate = allocation;
                    candidate.counts[x][a][k] += 1;
                    if !candidate.feasible(world) {
                        continue;
                    }
                    let next = analytic_mse(world, &candidate, EVAL_N)?;
                    let gain = current - next;
                    if gain <= EPS {
                        continue;
                    }
                    let score = match rule {
                        GreedyRule::MaxGain => gain,
                        GreedyRule::GainPerCost => gain / world.source_cost[k] as f64,
                    };
                    if best.is_none_or(|(best_score, _)| score > best_score + EPS) {
                        best = Some((score, candidate));
                    }
                }
            }
        }
        match best {
            Some((_, candidate)) => allocation = candidate,
            None => break,
        }
    }
    Ok(allocation)
}

fn best_single_source(world: &World) -> Result<Allocation, String> {
    let mut best = Allocation::default();
    let mut best_score = analytic_mse(world, &best, EVAL_N)?;
    for k in 0..SOURCES {
        let mut mask = [false; SOURCES];
        mask[k] = true;
        let candidate = exact_dynamic_program(world, mask)?.allocation;
        let score = analytic_mse(world, &candidate, EVAL_N)?;
        if score + EPS < best_score {
            best = candidate;
            best_score = score;
        }
    }
    Ok(best)
}

fn pairwise_exchange(world: &World, mut allocation: Allocation) -> Result<Allocation, String> {
    for _ in 0..128 {
        let current = analytic_mse(world, &allocation, EVAL_N)?;
        let mut best: Option<(f64, Allocation)> = None;
        for x_from in 0..CONTEXTS {
            for a_from in 0..ACTIONS {
                for k_from in 0..SOURCES {
                    if allocation.count(x_from, a_from, k_from) == 0 {
                        continue;
                    }
                    for x_to in 0..CONTEXTS {
                        for a_to in 0..ACTIONS {
                            for k_to in 0..SOURCES {
                                if (x_from, a_from, k_from) == (x_to, a_to, k_to) {
                                    continue;
                                }
                                let mut candidate = allocation;
                                candidate.counts[x_from][a_from][k_from] -= 1;
                                candidate.counts[x_to][a_to][k_to] += 1;
                                if !candidate.feasible(world) {
                                    continue;
                                }
                                let score = analytic_mse(world, &candidate, EVAL_N)?;
                                let gain = current - score;
                                if gain > EPS
                                    && best.is_none_or(|(best_gain, _)| gain > best_gain + EPS)
                                {
                                    best = Some((gain, candidate));
                                }
                            }
                        }
                    }
                }
            }
        }
        match best {
            Some((_, candidate)) => allocation = candidate,
            None => break,
        }
    }
    Ok(allocation)
}

fn annotation_reward_model(
    world: &World,
    allocation: &Allocation,
    seed: u64,
) -> [[f64; ACTIONS]; CONTEXTS] {
    let mut model = [[0.0; ACTIONS]; CONTEXTS];
    for x in 0..CONTEXTS {
        for a in 0..ACTIONS {
            let mut rng = Rng::new(seed ^ ((x * ACTIONS + a + 1) as u64).wrapping_mul(0x9e37_79b9));
            let factual_n = world.factual_count[x][a] as usize;
            let mut sum = 0.0;
            for _ in 0..factual_n {
                sum += if rng.uniform() < world.reward_mean[x][a] {
                    1.0
                } else {
                    0.0
                };
            }
            let mut count = factual_n;
            let shared_draws: [f64; 2] = std::array::from_fn(|factor| {
                if world.shared_factor_variance[factor] > 0.0 {
                    rng.normal()
                } else {
                    0.0
                }
            });
            for k in 0..SOURCES {
                let mean = world.reward_mean[x][a] + world.source_bias[x][a][k];
                let marginal_variance = reward_variance(world.reward_mean[x][a])
                    + world.source_excess_variance[x][a][k];
                let shared_variance = (0..2)
                    .map(|factor| {
                        world.shared_factor_variance[factor]
                            * world.source_shared_loadings[k][factor].powi(2)
                    })
                    .sum::<f64>();
                let independent_variance = (marginal_variance - shared_variance).max(0.0);
                let shared_error = (0..2)
                    .map(|factor| {
                        world.shared_factor_variance[factor].sqrt()
                            * world.source_shared_loadings[k][factor]
                            * shared_draws[factor]
                    })
                    .sum::<f64>();
                for _ in 0..allocation.count(x, a, k) {
                    sum += mean + shared_error + independent_variance.sqrt() * rng.normal();
                    count += 1;
                }
            }
            model[x][a] = if count == 0 {
                world.default_reward[x][a]
            } else {
                sum / count as f64
            };
        }
    }
    model
}

fn sample_action(probabilities: &[f64; ACTIONS], rng: &mut Rng) -> usize {
    let draw = rng.uniform();
    let mut cumulative = 0.0;
    for (action, probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if draw <= cumulative || action + 1 == ACTIONS {
            return action;
        }
    }
    ACTIONS - 1
}

fn one_ope_estimate(
    world: &World,
    model: &[[f64; ACTIONS]; CONTEXTS],
    evaluation_n: usize,
    seed: u64,
    naive_annotation_residual_source: Option<usize>,
) -> f64 {
    let mut rng = Rng::new(seed);
    let mut estimate = 0.0;
    for _ in 0..evaluation_n {
        let context = if rng.uniform() < world.context_probability[0] {
            0
        } else {
            1
        };
        let action = sample_action(&world.behavior[context], &mut rng);
        let pi = world.target[context][action];
        let mu = world.behavior[context][action];
        let reward = match naive_annotation_residual_source {
            None => {
                if rng.uniform() < world.reward_mean[context][action] {
                    1.0
                } else {
                    0.0
                }
            }
            Some(k) => {
                let mean =
                    world.reward_mean[context][action] + world.source_bias[context][action][k];
                let variance = reward_variance(world.reward_mean[context][action])
                    + world.source_excess_variance[context][action][k];
                mean + variance.sqrt() * rng.normal()
            }
        };
        let dm = (0..ACTIONS)
            .map(|a| world.target[context][a] * model[context][a])
            .sum::<f64>();
        estimate += dm + pi / mu * (reward - model[context][action]);
    }
    estimate / evaluation_n as f64
}

#[derive(Clone, Copy, Debug)]
struct MonteCarloMetrics {
    bias: f64,
    variance: f64,
    mse: f64,
}

fn monte_carlo_evaluate(
    world: &World,
    allocation: &Allocation,
    evaluation_n: usize,
    replicates: usize,
    seed: u64,
    naive_annotation_residual_source: Option<usize>,
) -> MonteCarloMetrics {
    let truth = target_value(world);
    let mut errors = Vec::with_capacity(replicates);
    for replicate in 0..replicates {
        let q = annotation_reward_model(world, allocation, mix_seed(seed, replicate, 11));
        let estimate = one_ope_estimate(
            world,
            &q,
            evaluation_n,
            mix_seed(seed, replicate, 23),
            naive_annotation_residual_source,
        );
        errors.push(estimate - truth);
    }
    let n = errors.len() as f64;
    let bias = errors.iter().sum::<f64>() / n;
    let mse = errors.iter().map(|error| error * error).sum::<f64>() / n;
    let variance = (mse - bias * bias).max(0.0);
    MonteCarloMetrics {
        bias,
        variance,
        mse,
    }
}

fn allocation_sources(allocation: &Allocation) -> [u32; SOURCES] {
    std::array::from_fn(|k| allocation.total_by_source(k))
}

fn allocation_profile_bias(world: &World, allocation: &Allocation) -> f64 {
    weighted_abs_model_bias(world, allocation)
}

fn context_bias_sums(world: &World, allocation: &Allocation) -> [f64; CONTEXTS] {
    std::array::from_fn(|x| {
        (0..ACTIONS)
            .map(|a| world.target[x][a] * cell_bias_and_variance(world, allocation, x, a).0)
            .sum()
    })
}

/// Allocation-dependent part of the exact OPE variance objective:
/// J = J1 - J2 + J3. The factual reward variance and between-context variance
/// are invariant to the annotation allocation and are dropped from the optimizer.
fn allocation_dependent_objective(world: &World, allocation: &Allocation) -> Result<f64, String> {
    if !allocation.feasible(world) {
        return Err("MM objective requires a feasible allocation".into());
    }
    let beta = context_bias_sums(world, allocation);
    let mut objective = 0.0;
    for x in 0..CONTEXTS {
        let d = world.context_probability[x];
        for a in 0..ACTIONS {
            let pi = world.target[x][a];
            let mu = world.behavior[x][a];
            let (bias, variance) = cell_bias_and_variance(world, allocation, x, a);
            let alpha1 = d * pi * pi / mu;
            let alpha3 = d * pi * pi * (1.0 / mu - 1.0);
            objective += alpha1 * bias * bias + alpha3 * variance;
        }
        objective -= d * beta[x] * beta[x];
    }
    if objective.is_finite() && objective >= -EPS {
        Ok(objective.max(0.0))
    } else {
        Err("non-finite or materially negative allocation-dependent objective".into())
    }
}

/// Dropped-constant MM surrogate at the current iterate:
/// J1(n) + J3(n) - 2 sum_x d_x beta_x(current) beta_x(n).
fn mm_surrogate_value(
    world: &World,
    allocation: &Allocation,
    current_beta: &[f64; CONTEXTS],
) -> Result<f64, String> {
    if !allocation.feasible(world) {
        return Err("MM surrogate requires a feasible allocation".into());
    }
    let mut value = 0.0;
    for x in 0..CONTEXTS {
        let d = world.context_probability[x];
        for a in 0..ACTIONS {
            let pi = world.target[x][a];
            let mu = world.behavior[x][a];
            let (bias, variance) = cell_bias_and_variance(world, allocation, x, a);
            let alpha1 = d * pi * pi / mu;
            let alpha3 = d * pi * pi * (1.0 / mu - 1.0);
            let omega2 = -2.0 * d * current_beta[x] * pi;
            value += alpha1 * bias * bias + alpha3 * variance + omega2 * bias;
        }
    }
    if value.is_finite() {
        Ok(value)
    } else {
        Err("non-finite MM surrogate".into())
    }
}

/// Full tangent majorizer. Its gap to J is exactly
/// sum_x d_x (beta_x(n) - beta_x(current))^2.
fn full_mm_majorizer(
    world: &World,
    allocation: &Allocation,
    current_beta: &[f64; CONTEXTS],
) -> Result<f64, String> {
    let constant = (0..CONTEXTS)
        .map(|x| world.context_probability[x] * current_beta[x] * current_beta[x])
        .sum::<f64>();
    Ok(mm_surrogate_value(world, allocation, current_beta)? + constant)
}

fn mm_cell_surrogate(
    world: &World,
    context: usize,
    action: usize,
    source_counts: &[u8; SOURCES],
    current_beta: f64,
    lambda: f64,
) -> f64 {
    let mut partial = Allocation::default();
    partial.counts[context][action] = *source_counts;
    let (bias, variance) = cell_bias_and_variance(world, &partial, context, action);
    let d = world.context_probability[context];
    let pi = world.target[context][action];
    let mu = world.behavior[context][action];
    let alpha1 = d * pi * pi / mu;
    let alpha3 = d * pi * pi * (1.0 / mu - 1.0);
    let omega2 = -2.0 * d * current_beta * pi;
    let cost = (0..SOURCES)
        .map(|k| source_counts[k] as f64 * world.source_cost[k] as f64)
        .sum::<f64>();
    alpha1 * bias * bias + alpha3 * variance + omega2 * bias + lambda * cost
}

/// Algorithm 2's fixed-count source mixture search: initialize all annotations
/// at the cheapest source, then apply the best improving pairwise source swap.
fn local_mix_option(
    world: &World,
    context: usize,
    action: usize,
    total_annotations: usize,
    current_beta: f64,
    lambda: f64,
    swap_limit: usize,
) -> LocalMixOption {
    let cheapest = (0..SOURCES)
        .min_by_key(|k| (world.source_cost[*k], *k))
        .expect("at least one source is registered");
    let mut counts = [0_u8; SOURCES];
    counts[cheapest] = total_annotations as u8;

    for _ in 0..swap_limit {
        let current_score =
            mm_cell_surrogate(world, context, action, &counts, current_beta, lambda);
        let mut best_gain = EPS;
        let mut best_counts = None;
        for source_from in 0..SOURCES {
            if counts[source_from] == 0 {
                continue;
            }
            for source_to in 0..SOURCES {
                if source_from == source_to {
                    continue;
                }
                let mut candidate = counts;
                candidate[source_from] -= 1;
                candidate[source_to] += 1;
                let candidate_score =
                    mm_cell_surrogate(world, context, action, &candidate, current_beta, lambda);
                let gain = current_score - candidate_score;
                if gain > best_gain + EPS {
                    best_gain = gain;
                    best_counts = Some(candidate);
                }
            }
        }
        match best_counts {
            Some(candidate) => counts = candidate,
            None => break,
        }
    }

    let score = mm_cell_surrogate(world, context, action, &counts, current_beta, lambda);
    let mut remaining_swap_gain = 0.0_f64;
    for source_from in 0..SOURCES {
        if counts[source_from] == 0 {
            continue;
        }
        for source_to in 0..SOURCES {
            if source_from == source_to {
                continue;
            }
            let mut candidate = counts;
            candidate[source_from] -= 1;
            candidate[source_to] += 1;
            let candidate_score =
                mm_cell_surrogate(world, context, action, &candidate, current_beta, lambda);
            remaining_swap_gain = remaining_swap_gain.max((score - candidate_score).max(0.0));
        }
    }
    LocalMixOption {
        counts,
        score,
        remaining_swap_gain,
    }
}

/// For fixed lambda, perform per-cell pairwise source search and then an exact
/// DP over action annotation counts inside each context.
fn mm_candidate_for_lambda(
    world: &World,
    current_beta: &[f64; CONTEXTS],
    lambda: f64,
    swap_limit: usize,
) -> Result<MmCandidate, String> {
    if !lambda.is_finite() || lambda < 0.0 {
        return Err("MM multiplier must be finite and nonnegative".into());
    }
    let mut allocation = Allocation::default();
    let mut remaining_swap_gain = 0.0;

    for x in 0..CONTEXTS {
        let cap = world.context_annotation_cap[x] as usize;
        let factual_context_count = world.factual_count[x]
            .iter()
            .map(|count| *count as usize)
            .sum::<usize>();
        let mut tables: [Vec<LocalMixOption>; ACTIONS] = std::array::from_fn(|_| Vec::new());
        for a in 0..ACTIONS {
            let factual_cell_count = world.factual_count[x][a] as usize;
            let max_for_cell = factual_context_count
                .saturating_sub(factual_cell_count)
                .min(cap);
            for m in 0..=max_for_cell {
                tables[a].push(local_mix_option(
                    world,
                    x,
                    a,
                    m,
                    current_beta[x],
                    lambda,
                    swap_limit,
                ));
            }
        }

        let mut dp = vec![vec![f64::INFINITY; cap + 1]; ACTIONS + 1];
        let mut parent: Vec<Vec<Option<(usize, usize)>>> = vec![vec![None; cap + 1]; ACTIONS + 1];
        dp[0][0] = 0.0;
        for a in 0..ACTIONS {
            for used in 0..=cap {
                if !dp[a][used].is_finite() {
                    continue;
                }
                let max_m = (cap - used).min(tables[a].len() - 1);
                for m in 0..=max_m {
                    let next_used = used + m;
                    let candidate = dp[a][used] + tables[a][m].score;
                    if candidate + EPS < dp[a + 1][next_used] {
                        dp[a + 1][next_used] = candidate;
                        parent[a + 1][next_used] = Some((used, m));
                    }
                }
            }
        }
        let mut best_used = 0;
        for used in 1..=cap {
            if dp[ACTIONS][used] + EPS < dp[ACTIONS][best_used] {
                best_used = used;
            }
        }
        if !dp[ACTIONS][best_used].is_finite() {
            return Err("MM context DP has no feasible state".into());
        }
        let mut cursor = best_used;
        for a in (0..ACTIONS).rev() {
            let (previous, m) =
                parent[a + 1][cursor].ok_or_else(|| "missing MM DP backpointer".to_string())?;
            let option = tables[a][m];
            allocation.counts[x][a] = option.counts;
            remaining_swap_gain += option.remaining_swap_gain;
            cursor = previous;
        }
    }
    // The fixed-lambda subproblem may intentionally return a globally
    // over-budget candidate. The outer lambda bracket/bisection observes its
    // cost and raises the multiplier; only the outer loop may admit candidates.
    if (0..CONTEXTS).any(|x| allocation.context_count(x) > world.context_annotation_cap[x] as u32) {
        return Err("MM context DP violated a context annotation cap".into());
    }
    Ok(MmCandidate {
        allocation,
        remaining_swap_gain,
    })
}

/// Algorithm 1: linearize the squared context bias, search lambda with doubling
/// and bisection, and retain the current allocation as a descent-safe fallback.
/// Only budget-feasible candidates can be accepted.
fn mm_allocate(
    world: &World,
    initial: Allocation,
    outer_limit: usize,
    swap_limit: usize,
) -> Result<MmRun, String> {
    if !initial.feasible(world) {
        return Err("MM initial allocation is infeasible".into());
    }
    const INITIAL_LAMBDA: f64 = 1e-6;
    const LAMBDA_TOLERANCE: f64 = 1e-8;
    const MAX_DOUBLINGS: usize = 64;
    const MAX_BISECTION: usize = 40;

    let mut current = initial;
    let mut objective_trace = vec![allocation_dependent_objective(world, &current)?];
    let mut iterations = 0;
    let mut last_residual = 0.0;

    for _ in 0..outer_limit {
        let current_beta = context_bias_sums(world, &current);
        let mut best_allocation = current;
        let mut best_surrogate = mm_surrogate_value(world, &current, &current_beta)?;
        let mut best_residual = 0.0;
        let mut candidates = Vec::new();

        let zero_price = mm_candidate_for_lambda(world, &current_beta, 0.0, swap_limit)?;
        if zero_price.allocation.feasible(world) {
            candidates.push(zero_price);
        }

        let mut lambda_low = 0.0;
        let mut lambda_high = INITIAL_LAMBDA;
        let mut high_candidate =
            mm_candidate_for_lambda(world, &current_beta, lambda_high, swap_limit)?;
        let mut doublings = 0;
        while high_candidate.allocation.cost(world) > world.budget {
            lambda_low = lambda_high;
            lambda_high *= 2.0;
            doublings += 1;
            if doublings > MAX_DOUBLINGS || !lambda_high.is_finite() {
                return Err("MM failed to bracket a feasible Lagrange price".into());
            }
            high_candidate =
                mm_candidate_for_lambda(world, &current_beta, lambda_high, swap_limit)?;
        }
        candidates.push(high_candidate);

        for _ in 0..MAX_BISECTION {
            if lambda_high - lambda_low <= LAMBDA_TOLERANCE {
                break;
            }
            let lambda_mid = (lambda_low + lambda_high) / 2.0;
            let candidate = mm_candidate_for_lambda(world, &current_beta, lambda_mid, swap_limit)?;
            if candidate.allocation.cost(world) > world.budget {
                lambda_low = lambda_mid;
            } else {
                lambda_high = lambda_mid;
                candidates.push(candidate);
            }
        }

        for candidate in candidates {
            let surrogate = mm_surrogate_value(world, &candidate.allocation, &current_beta)?;
            if surrogate + EPS < best_surrogate {
                best_allocation = candidate.allocation;
                best_surrogate = surrogate;
                best_residual = candidate.remaining_swap_gain;
            }
        }
        if best_allocation == current {
            break;
        }

        let old_objective = *objective_trace.last().expect("initial objective");
        let new_objective = allocation_dependent_objective(world, &best_allocation)?;
        if new_objective > old_objective + 1e-10 {
            return Err(format!(
                "MM monotonicity violated: old={old_objective}, new={new_objective}"
            ));
        }
        current = best_allocation;
        objective_trace.push(new_objective);
        iterations += 1;
        last_residual = best_residual;
        if old_objective - new_objective <= EPS {
            break;
        }
    }

    Ok(MmRun {
        allocation: current,
        iterations,
        objective_trace,
        remaining_swap_gain: last_residual,
    })
}

type Plan = (&'static str, Allocation, usize, f64);

fn build_plans(world: &World) -> Result<Vec<Plan>, String> {
    let none = Allocation::default();
    let greedy_max = greedy_allocate(world, GreedyRule::MaxGain)?;
    let greedy_cost = greedy_allocate(world, GreedyRule::GainPerCost)?;
    let single_source = best_single_source(world)?;
    let exchange = pairwise_exchange(world, greedy_max)?;
    let exact = exact_dynamic_program(world, [true; SOURCES])?.allocation;
    let estimated_world = world.estimated_profiles();
    let plugin_profile = exact_dynamic_program(&estimated_world, [true; SOURCES])?.allocation;
    let mm_zero = mm_allocate(world, none, 24, 128)?;
    let mm_greedy = mm_allocate(world, greedy_max, 24, 128)?;

    Ok(vec![
        ("no_annotations", none, 0, 0.0),
        ("greedy_max_gain", greedy_max, 0, 0.0),
        ("greedy_cost_aware", greedy_cost, 0, 0.0),
        ("best_single_source_exact", single_source, 0, 0.0),
        ("pairwise_exchange", exchange, 0, 0.0),
        ("exact_dp_oracle", exact, 0, 0.0),
        ("plugin_profile_exact", plugin_profile, 0, 0.0),
        (
            "mm_from_zero",
            mm_zero.allocation,
            mm_zero.iterations,
            mm_zero.remaining_swap_gain,
        ),
        (
            "mm_from_greedy",
            mm_greedy.allocation,
            mm_greedy.iterations,
            mm_greedy.remaining_swap_gain,
        ),
    ])
}

fn local_phi(
    n: usize,
    factual_count: f64,
    reward_variance: f64,
    annotation_bias_squared: f64,
    annotation_excess_variance: f64,
    alpha1: f64,
    eta: f64,
) -> f64 {
    let annotation_n = n as f64;
    let alpha3 = eta * alpha1;
    let annotation_variance = reward_variance + annotation_excess_variance;
    let numerator = alpha1 * annotation_n * annotation_n * annotation_bias_squared
        + alpha3 * (factual_count * reward_variance + annotation_n * annotation_variance);
    numerator / (factual_count + annotation_n).powi(2)
}

fn first_annotation_threshold(
    factual_count: f64,
    reward_variance: f64,
    annotation_excess_variance: f64,
    annotation_bias_squared: f64,
    eta: f64,
) -> bool {
    annotation_bias_squared
        <= eta
            * ((factual_count + 1.0) * reward_variance / factual_count - annotation_excess_variance)
            + EPS
}

#[derive(Clone, Copy)]
struct LocalRegime {
    name: &'static str,
    excess_variance: f64,
    bias_squared: f64,
    capacity: usize,
}

fn local_regimes() -> [LocalRegime; 5] {
    [
        LocalRegime {
            name: "I_no_annotation",
            excess_variance: 0.20,
            bias_squared: 0.04,
            capacity: 100,
        },
        LocalRegime {
            name: "II_finite_batch",
            excess_variance: 0.00,
            bias_squared: 0.03,
            capacity: 100,
        },
        LocalRegime {
            name: "III_more_is_better",
            excess_variance: 0.00,
            bias_squared: 0.0025,
            capacity: 100,
        },
        LocalRegime {
            name: "IVa_lumpy_batch",
            excess_variance: 0.40,
            bias_squared: 0.01,
            capacity: 100,
        },
        LocalRegime {
            name: "IVb_no_annotation",
            excess_variance: 0.40,
            bias_squared: 0.03,
            capacity: 100,
        },
    ]
}

fn local_regime_rows() -> String {
    let factual_count = 4.0;
    let reward_variance_value = 0.20;
    let alpha1 = 1.0;
    let eta = 0.5;
    let mut output = String::new();
    for regime in local_regimes() {
        let phi0 = local_phi(
            0,
            factual_count,
            reward_variance_value,
            regime.bias_squared,
            regime.excess_variance,
            alpha1,
            eta,
        );
        let phi1 = local_phi(
            1,
            factual_count,
            reward_variance_value,
            regime.bias_squared,
            regime.excess_variance,
            alpha1,
            eta,
        );
        let phi_cap = local_phi(
            regime.capacity,
            factual_count,
            reward_variance_value,
            regime.bias_squared,
            regime.excess_variance,
            alpha1,
            eta,
        );
        let best_n = (0..=regime.capacity)
            .min_by(|left, right| {
                local_phi(
                    *left,
                    factual_count,
                    reward_variance_value,
                    regime.bias_squared,
                    regime.excess_variance,
                    alpha1,
                    eta,
                )
                .total_cmp(&local_phi(
                    *right,
                    factual_count,
                    reward_variance_value,
                    regime.bias_squared,
                    regime.excess_variance,
                    alpha1,
                    eta,
                ))
            })
            .unwrap_or(0);
        let first_helpful = first_annotation_threshold(
            factual_count,
            reward_variance_value,
            regime.excess_variance,
            regime.bias_squared,
            eta,
        );
        use std::fmt::Write;
        let _ = writeln!(
            output,
            "LOCAL_REGIME\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}",
            regime.name, phi0, phi1, phi_cap, regime.bias_squared, best_n, first_helpful,
        );
    }
    output
}

fn benchmark(replicates: usize, seed: u64) -> Result<String, String> {
    if replicates == 0 {
        return Err("replicate count must be positive".into());
    }
    let worlds = [
        World::mixed_bias_cancellation(),
        World::heterogeneous_source_quality(),
    ];
    let mut output = String::from(
        "record_type\tscenario\tmethod\tevaluation_n\treplicates\tbudget\tspent\tannotation_count\tanalytic_mse\tempirical_bias\tempirical_variance\tempirical_mse\trelative_reduction_vs_none\trelative_gap_to_exact\ttarget_weighted_abs_model_bias\tsource0_count\tsource1_count\tsource2_count\tmm_iterations\tremaining_swap_gain\n",
    );
    output.push_str("BENCHMARK_CONFIG\tVIS-DVPC-054\tversion\t0\n");
    output.push_str(&format!("BENCHMARK_CONFIG\treplicates\t{replicates}\n"));
    output.push_str(&format!("BENCHMARK_CONFIG\tevaluation_n\t{EVAL_N}\n"));
    output.push_str(&format!(
        "BENCHMARK_CONFIG\taction_cap_per_context\t{}\n",
        ACTION_CAP_PER_CONTEXT
    ));
    output.push_str(&format!(
        "BENCHMARK_CONFIG\tbudget_tokens\t{BUDGET_TOKENS}\n"
    ));
    output.push_str("BENCHMARK_CONFIG\testimator\tDM+IS / annotated DR; annotations augment q only; factual residual preserved\n");
    output.push_str("BENCHMARK_CONFIG\toptimizer\texact per-context allocation enumeration + global budget DP; greedy comparators\n");
    output.push_str("BENCHMARK_CONFIG\tprofile_pilot\tindependent estimated source profiles; plugin allocation evaluated under hidden true profiles\n");

    for (world_index, world) in worlds.iter().enumerate() {
        world.validate()?;
        let plans = build_plans(world)?;
        let no_annotation = analytic_mse(world, &Allocation::default(), EVAL_N)?;
        let exact_mse = plans
            .iter()
            .find(|(method, _, _, _)| *method == "exact_dp_oracle")
            .map(|(_, allocation, _, _)| analytic_mse(world, allocation, EVAL_N))
            .transpose()?
            .ok_or_else(|| "missing exact-DP oracle plan".to_string())?;
        for (method_index, (method, allocation, mm_iterations, mm_residual)) in
            plans.iter().enumerate()
        {
            if !allocation.feasible(world) {
                return Err(format!("infeasible plan: {method}"));
            }
            let analytic = analytic_mse(world, allocation, EVAL_N)?;
            let mc = monte_carlo_evaluate(
                world,
                allocation,
                EVAL_N,
                replicates,
                seed ^ (world_index as u64).wrapping_mul(0x517c_c1b7_2722_0a95)
                    ^ (method_index as u64).wrapping_mul(0x6eed_0e9d_a4d9_4a4f),
                None,
            );
            let sources = allocation_sources(allocation);
            let relative_reduction = if no_annotation > EPS {
                (no_annotation - analytic) / no_annotation
            } else {
                0.0
            };
            let relative_gap_to_exact = ((analytic - exact_mse) / exact_mse.max(EPS)).max(0.0);
            use std::fmt::Write;
            let _ = writeln!(
                output,
                "RESULT\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.10}\t{:.10}\t{:.10}\t{:.10}\t{:.8}\t{:.8}\t{:.8}\t{}\t{}\t{}\t{}\t{:.10}",
                world.name,
                method,
                EVAL_N,
                replicates,
                world.budget,
                allocation.cost(world),
                allocation.total_count(),
                analytic,
                mc.bias,
                mc.variance,
                mc.mse,
                relative_reduction,
                relative_gap_to_exact,
                allocation_profile_bias(world, allocation),
                sources[0],
                sources[1],
                sources[2],
                mm_iterations,
                mm_residual,
            );
        }

        // Deliberately wrong control: counterfactual annotations are substituted
        // for factual logged rewards inside the IS residual. This is not DM+IS.
        let source_one_only = {
            let mut mask = [false; SOURCES];
            mask[1] = true;
            exact_dynamic_program(world, mask)?.allocation
        };
        let wrong = monte_carlo_evaluate(
            world,
            &source_one_only,
            EVAL_N,
            replicates,
            seed ^ 0xdead_beef_cafe_babe ^ (world_index as u64),
            Some(1),
        );
        let proper_mse = analytic_mse(world, &source_one_only, EVAL_N)?;
        use std::fmt::Write;
        let _ = writeln!(
            output,
            "NEGATIVE_CONTROL\t{}\tannotation_as_factual_residual\t{}\t{}\t{:.10}\t{:.10}\t{:.10}\tIDENTIFIED_ONLY_BY_FACTUAL_LOGGER_SUPPORT",
            world.name, EVAL_N, replicates, wrong.bias, wrong.mse, proper_mse,
        );
    }

    output.push_str(&local_regime_rows());
    Ok(output)
}

fn main() {
    let smoke = env::args().any(|arg| arg == "--smoke");
    let replicates = if smoke {
        SMOKE_REPLICATES
    } else {
        FULL_REPLICATES
    };
    match benchmark(replicates, 0x0540_2026_1030) {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("VIS_DVPC_054_ERROR\t{error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_dp_matches_independent_product_enumeration_on_small_fixture() {
        let world = World::mixed_bias_cancellation();
        let exact = exact_dynamic_program(&world, [true; SOURCES]).unwrap();
        let left = enumerate_context_options(&world, 0, [true; SOURCES]);
        let right = enumerate_context_options(&world, 1, [true; SOURCES]);
        let mut brute_best = f64::INFINITY;
        for l in &left {
            for r in &right {
                if l.cost + r.cost <= world.budget {
                    brute_best = brute_best.min(l.risk_component + r.risk_component);
                }
            }
        }
        let exact_component = (0..CONTEXTS)
            .map(|x| {
                world.context_probability[x]
                    * context_conditional_variance(&world, &exact.allocation, x)
            })
            .sum::<f64>();
        assert!((exact_component - brute_best).abs() < 1e-10);
        assert!(exact.allocation.feasible(&world));
        assert!(exact.context_options >= CONTEXTS);
        assert!(exact.dp_transitions > 0);
    }

    #[test]
    fn exact_optimum_is_never_worse_than_no_annotation_plan() {
        for world in [
            World::mixed_bias_cancellation(),
            World::heterogeneous_source_quality(),
        ] {
            let none = Allocation::default();
            let exact = exact_dynamic_program(&world, [true; SOURCES])
                .unwrap()
                .allocation;
            assert!(
                analytic_mse(&world, &exact, EVAL_N).unwrap()
                    <= analytic_mse(&world, &none, EVAL_N).unwrap() + EPS
            );
        }
    }

    #[test]
    fn opposite_signed_sources_cancel_reward_model_bias() {
        let mut world = World::mixed_bias_cancellation();
        world.source_bias[0][0][1] = 0.20;
        world.source_bias[0][0][2] = -0.20;
        world.source_excess_variance[0][0][1] = 0.05;
        world.source_excess_variance[0][0][2] = 0.05;

        let mut single = Allocation::default();
        single.counts[0][0][1] = 2;
        let mut mixed = Allocation::default();
        mixed.counts[0][0][1] = 1;
        mixed.counts[0][0][2] = 1;

        let (single_bias, _) = cell_bias_and_variance(&world, &single, 0, 0);
        let (mixed_bias, _) = cell_bias_and_variance(&world, &mixed, 0, 0);
        assert!((mixed_bias).abs() < EPS);
        assert!(single_bias.abs() > 0.09);
        assert!(
            analytic_mse(&world, &mixed, EVAL_N).unwrap()
                < analytic_mse(&world, &single, EVAL_N).unwrap()
        );
    }

    #[test]
    fn first_annotation_threshold_matches_exact_phi_difference() {
        let nf = 4.0;
        let sigma2 = 0.20;
        let delta = 0.15;
        let eta = 0.5;
        for eps2 in [0.001, 0.02, 0.08] {
            let predicted = first_annotation_threshold(nf, sigma2, delta, eps2, eta);
            let phi0 = local_phi(0, nf, sigma2, eps2, delta, 1.0, eta);
            let phi1 = local_phi(1, nf, sigma2, eps2, delta, 1.0, eta);
            assert_eq!(predicted, phi1 <= phi0 + EPS);
        }
    }

    #[test]
    fn all_four_local_annotation_value_regimes_and_lumpy_batch_control_are_present() {
        let regimes = local_regimes();
        assert_eq!(regimes.len(), 5);
        let nf = 4.0;
        let sigma2 = 0.20;
        let eta = 0.5;
        let values: Vec<(f64, f64, f64, usize)> = regimes
            .iter()
            .map(|regime| {
                let p0 = local_phi(
                    0,
                    nf,
                    sigma2,
                    regime.bias_squared,
                    regime.excess_variance,
                    1.0,
                    eta,
                );
                let p1 = local_phi(
                    1,
                    nf,
                    sigma2,
                    regime.bias_squared,
                    regime.excess_variance,
                    1.0,
                    eta,
                );
                let pc = local_phi(
                    regime.capacity,
                    nf,
                    sigma2,
                    regime.bias_squared,
                    regime.excess_variance,
                    1.0,
                    eta,
                );
                let best = (0..=regime.capacity)
                    .min_by(|a, b| {
                        local_phi(
                            *a,
                            nf,
                            sigma2,
                            regime.bias_squared,
                            regime.excess_variance,
                            1.0,
                            eta,
                        )
                        .total_cmp(&local_phi(
                            *b,
                            nf,
                            sigma2,
                            regime.bias_squared,
                            regime.excess_variance,
                            1.0,
                            eta,
                        ))
                    })
                    .unwrap();
                (p0, p1, pc, best)
            })
            .collect();

        // Regime I: annotations are locally harmful and monotonically worse.
        assert!(values[0].1 > values[0].0);
        assert_eq!(values[0].3, 0);
        // Regime II: a finite interior batch is best.
        assert!(values[1].1 < values[1].0);
        assert!(values[1].3 > 0 && values[1].3 < 100);
        // Regime III: every extra annotation keeps helping.
        assert!(values[2].1 < values[2].0);
        assert_eq!(values[2].3, 100);
        // Regime IVa: first annotation hurts, but a sufficiently large batch wins.
        assert!(values[3].1 > values[3].0);
        assert!(values[3].2 < values[3].0);
        assert_eq!(values[3].3, 100);
        // Regime IVb: first annotation hurts and the large batch still does not win.
        assert!(values[4].1 > values[4].0);
        assert!(values[4].2 > values[4].0);
        assert_eq!(values[4].3, 0);
    }

    #[test]
    fn analytic_mse_matches_independent_monte_carlo() {
        let world = World::mixed_bias_cancellation();
        let allocation = exact_dynamic_program(&world, [true; SOURCES])
            .unwrap()
            .allocation;
        let analytic = analytic_mse(&world, &allocation, 64).unwrap();
        let mc = monte_carlo_evaluate(
            &world,
            &allocation,
            64,
            MC_TEST_REPLICATES,
            0x0540_abc0,
            None,
        );
        let relative_error = (mc.mse - analytic).abs() / analytic.max(1e-12);
        assert!(
            relative_error < 0.15,
            "analytic={analytic}, MC={}, relative={relative_error}",
            mc.mse
        );
        assert!(mc.bias.abs() < 0.03, "proper DM+IS bias={}", mc.bias);
    }

    #[test]
    fn naive_annotation_residual_is_biased_and_remains_a_negative_control() {
        let world = World::mixed_bias_cancellation();
        let mut allocation = Allocation::default();
        allocation.counts[0][0][1] = 2;
        allocation.counts[1][1][1] = 2;
        assert!(allocation.feasible(&world));
        let proper = monte_carlo_evaluate(&world, &allocation, 128, 3000, 0x0540_bad0, None);
        let naive = monte_carlo_evaluate(&world, &allocation, 128, 3000, 0x0540_bad0, Some(1));
        assert!(proper.bias.abs() < 0.03);
        assert!(naive.bias.abs() > 0.02);
    }

    #[test]
    fn annotations_do_not_repair_missing_factual_support() {
        let mut world = World::mixed_bias_cancellation();
        world.behavior[0][0] = 0.0;
        world.behavior[0][1] = 0.9;
        world.behavior[0][2] = 0.1;
        assert!(world.validate().is_err());
    }

    #[test]
    fn mm_majorizer_is_tangent_and_dominates_original_objective() {
        let world = World::mixed_bias_cancellation();
        let current = greedy_allocate(&world, GreedyRule::MaxGain).unwrap();
        let candidate = exact_dynamic_program(&world, [true; SOURCES])
            .unwrap()
            .allocation;
        let beta_current = context_bias_sums(&world, &current);
        let beta_candidate = context_bias_sums(&world, &candidate);
        let majorizer = full_mm_majorizer(&world, &candidate, &beta_current).unwrap();
        let objective = allocation_dependent_objective(&world, &candidate).unwrap();
        let expected_gap = (0..CONTEXTS)
            .map(|x| world.context_probability[x] * (beta_candidate[x] - beta_current[x]).powi(2))
            .sum::<f64>();
        assert!((majorizer - objective - expected_gap).abs() < 1e-10);
        let at_current = full_mm_majorizer(&world, &current, &beta_current).unwrap();
        let current_objective = allocation_dependent_objective(&world, &current).unwrap();
        assert!((at_current - current_objective).abs() < 1e-10);
        assert!(majorizer + EPS >= objective);
    }

    #[test]
    fn mm_descent_is_monotone_and_never_beats_the_exact_dp_oracle() {
        for world in [
            World::mixed_bias_cancellation(),
            World::heterogeneous_source_quality(),
        ] {
            let exact = exact_dynamic_program(&world, [true; SOURCES])
                .unwrap()
                .allocation;
            let exact_objective = allocation_dependent_objective(&world, &exact).unwrap();
            for initial in [
                Allocation::default(),
                greedy_allocate(&world, GreedyRule::MaxGain).unwrap(),
            ] {
                let initial_objective = allocation_dependent_objective(&world, &initial).unwrap();
                let result = mm_allocate(&world, initial, 24, 128).unwrap();
                assert!(result.allocation.feasible(&world));
                assert!(result.objective_trace.len() >= 1);
                assert!(
                    result
                        .objective_trace
                        .windows(2)
                        .all(|pair| pair[1] <= pair[0] + 1e-10),
                    "MM objective trace is not monotone: {:?}",
                    result.objective_trace
                );
                let mm_objective =
                    allocation_dependent_objective(&world, &result.allocation).unwrap();
                assert!(mm_objective <= initial_objective + 1e-10);
                assert!(mm_objective + 1e-9 >= exact_objective);
                assert_eq!(result.objective_trace.len(), result.iterations + 1);
            }
        }
    }

    #[test]
    fn output_is_deterministic_and_has_exact_row_counts() {
        let first = benchmark(2, 123).unwrap();
        let second = benchmark(2, 123).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            first
                .lines()
                .filter(|line| line.starts_with("RESULT\t"))
                .count(),
            18
        );
        assert_eq!(
            first
                .lines()
                .filter(|line| line.starts_with("NEGATIVE_CONTROL\t"))
                .count(),
            2
        );
        assert_eq!(
            first
                .lines()
                .filter(|line| line.starts_with("LOCAL_REGIME\t"))
                .count(),
            5
        );
        assert_eq!(
            first
                .lines()
                .filter(|line| line.starts_with("BENCHMARK_CONFIG\t"))
                .count(),
            8
        );
        for line in first.lines().filter(|line| line.starts_with("RESULT\t")) {
            assert_eq!(line.split('\t').count(), 20);
        }
    }

    #[test]
    fn invalid_profiles_budgets_and_evaluation_sizes_fail_closed() {
        let mut invalid = World::mixed_bias_cancellation();
        invalid.source_cost[0] = 0;
        assert!(invalid.validate().is_err());
        let world = World::mixed_bias_cancellation();
        assert!(analytic_mse(&world, &Allocation::default(), 0).is_err());
        assert_eq!(world.context_annotation_cap[0], ACTION_CAP_PER_CONTEXT);
    }
}
