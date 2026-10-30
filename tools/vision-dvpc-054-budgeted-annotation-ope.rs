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

fn correlated_stress_worlds() -> [World; 3] {
    let mut within_family = World::heterogeneous_source_quality();
    within_family.name = "within_family_panel";
    within_family.shared_factor_variance = [0.025, 0.0];
    within_family.source_shared_loadings = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]];

    let mut shared_rubric = World::heterogeneous_source_quality();
    shared_rubric.name = "shared_rubric_multi_family";
    shared_rubric.shared_factor_variance = [0.015, 0.012];
    shared_rubric.source_shared_loadings =
        [[0.25, 1.0], [1.0, 0.5], [0.20, 0.0]];

    let mut common_mode = World::mixed_bias_cancellation();
    common_mode.name = "high_common_mode";
    common_mode.shared_factor_variance = [0.03, 0.0];
    common_mode.source_shared_loadings = [[0.70, 0.0], [1.0, 0.0], [0.15, 0.0]];

    [within_family, shared_rubric, common_mode]
}

fn independent_profile(world: &World) -> World {
    let mut profile = *world;
    profile.shared_factor_variance = [0.0; 2];
    profile
}

fn upper_correlation_profile(world: &World, radius: f64) -> Result<World, String> {
    if !radius.is_finite() || radius < 1.0 {
        return Err("correlation uncertainty radius must be finite and at least one".into());
    }
    let mut upper = *world;
    for factor in 0..2 {
        upper.shared_factor_variance[factor] *= radius * radius;
    }
    upper.validate()?;
    Ok(upper)
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
fn shared_source_covariance(world: &World, left: usize, right: usize) -> f64 {
    (0..2)
        .map(|factor| {
            world.shared_factor_variance[factor]
                * world.source_shared_loadings[left][factor]
                * world.source_shared_loadings[right][factor]
        })
        .sum()
}

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
    let mut errors: Vec<[f64; PORTFOLIO_POLICIES]> = Vec::with_capacity(replicates);
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

fn weighted_q_variance_component(world: &World, allocation: &Allocation) -> f64 {
    let mut total = 0.0;
    for x in 0..CONTEXTS {
        for a in 0..ACTIONS {
            let pi = world.target[x][a];
            let mu = world.behavior[x][a];
            let alpha3 = world.context_probability[x] * pi * pi * (1.0 / mu - 1.0);
            let (_, q_variance) = cell_bias_and_variance(world, allocation, x, a);
            total += alpha3 * q_variance;
        }
    }
    total
}

/// Ratio of the OPE-weighted reward-model variance under correlated versus
/// independent source errors for exactly the same allocation. This is a
/// design-effect diagnostic, not a confidence interval or literal vote count.
fn weighted_q_variance_inflation(world: &World, allocation: &Allocation) -> f64 {
    let independent = independent_profile(world);
    let base = weighted_q_variance_component(&independent, allocation);
    let correlated = weighted_q_variance_component(world, allocation);
    if base > EPS {
        correlated / base
    } else if correlated <= EPS {
        1.0
    } else {
        f64::INFINITY
    }
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
    output.push_str(&correlation_benchmark_rows(
        replicates,
        seed ^ 0x0560_c011_2026_1030,
    )?);
    output.push_str(&policy_portfolio_benchmark_rows(
        replicates,
        seed ^ 0x0570_ba7c_2026_1030,
    )?);
    Ok(output)
}

fn correlation_benchmark_rows(replicates: usize, seed: u64) -> Result<String, String> {
    let mut output = String::new();
    for (scenario_index, world) in correlated_stress_worlds().iter().enumerate() {
        world.validate()?;
        let independent = independent_profile(world);
        let robust_profile = upper_correlation_profile(world, 1.5)?;

        let greedy_true = greedy_allocate(world, GreedyRule::MaxGain)?;
        let true_exact = exact_dynamic_program(world, [true; SOURCES])?.allocation;
        let independent_exact =
            exact_dynamic_program(&independent, [true; SOURCES])?.allocation;
        let robust_exact =
            exact_dynamic_program(&robust_profile, [true; SOURCES])?.allocation;
        let mm_zero = mm_allocate(world, Allocation::default(), 24, 128)?;
        let mm_greedy = mm_allocate(world, greedy_true, 24, 128)?;

        let plans = [
            ("independent_profile_exact", independent_exact, 0usize),
            ("correlation_aware_greedy", greedy_true, 0usize),
            ("correlation_aware_exact", true_exact, 0usize),
            ("robust_upper_bound_exact", robust_exact, 0usize),
            ("mm_true_from_zero", mm_zero.allocation, mm_zero.iterations),
            ("mm_true_from_greedy", mm_greedy.allocation, mm_greedy.iterations),
        ];
        let exact_mse = analytic_mse(world, &true_exact, EVAL_N)?;
        for (method_index, (method, allocation, iterations)) in plans.iter().enumerate() {
            if !allocation.feasible(world) {
                return Err(format!("correlation plan is infeasible: {method}"));
            }
            let analytic_true = analytic_mse(world, allocation, EVAL_N)?;
            let empirical = monte_carlo_evaluate(
                world,
                allocation,
                EVAL_N,
                replicates,
                seed ^ (scenario_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    ^ (method_index as u64).wrapping_mul(0x94d0_49bb_1331_11eb),
                None,
            );
            let factor0 = world.shared_factor_variance[0];
            let factor1 = world.shared_factor_variance[1];
            let profile0 = if *method == "independent_profile_exact" {
                0.0
            } else if *method == "robust_upper_bound_exact" {
                robust_profile.shared_factor_variance[0]
            } else {
                factor0
            };
            let profile1 = if *method == "independent_profile_exact" {
                0.0
            } else if *method == "robust_upper_bound_exact" {
                robust_profile.shared_factor_variance[1]
            } else {
                factor1
            };
            let worst_case_mse = analytic_mse(&robust_profile, allocation, EVAL_N)?;
            let sources = allocation_sources(allocation);
            let gap = ((analytic_true - exact_mse) / exact_mse.max(EPS)).max(0.0);
            let inflation = weighted_q_variance_inflation(world, allocation);
            use std::fmt::Write;
            let _ = writeln!(
                output,
                "CORR_RESULT\t{}\t{}\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.10}\t{:.10}\t{:.10}\t{:.10}\t{:.8}\t{:.8}\t{}\t{}\t{}\t{:.10}\t{}",
                world.name,
                method,
                replicates,
                world.budget,
                allocation.cost(world),
                allocation.total_count(),
                factor0,
                factor1,
                profile0,
                profile1,
                analytic_true,
                empirical.bias,
                empirical.variance,
                empirical.mse,
                inflation,
                gap,
                sources[0],
                sources[1],
                sources[2],
                worst_case_mse,
                iterations,
            );
        }
    }
    Ok(output)
}

const PORTFOLIO_POLICIES: usize = 3;

#[derive(Clone, Copy)]
struct PolicyPortfolio {
    names: [&'static str; PORTFOLIO_POLICIES],
    targets: [[[f64; ACTIONS]; CONTEXTS]; PORTFOLIO_POLICIES],
    prior_weights: [f64; PORTFOLIO_POLICIES],
}

#[derive(Clone, Copy, Debug)]
enum PortfolioObjective {
    PriorWeighted,
    WorstPolicy,
}

#[derive(Clone, Copy, Debug)]
struct PortfolioExactResult {
    allocation: Allocation,
    objective: f64,
    combinations_checked: usize,
}

#[derive(Clone, Copy, Debug)]
struct MultiPolicyMetrics {
    bias: [f64; PORTFOLIO_POLICIES],
    variance: [f64; PORTFOLIO_POLICIES],
    mse: [f64; PORTFOLIO_POLICIES],
    covariance: [[f64; PORTFOLIO_POLICIES]; PORTFOLIO_POLICIES],
    selection_accuracy: f64,
    expected_simple_regret: f64,
}

fn policy_portfolio(world: &World) -> PolicyPortfolio {
    let balanced = world.target;
    let conservative = [[0.78, 0.17, 0.05], [0.55, 0.30, 0.15]];
    let aggressive = [[0.10, 0.20, 0.70], [0.05, 0.15, 0.80]];
    PolicyPortfolio {
        names: ["conservative", "balanced", "aggressive"],
        targets: [conservative, balanced, aggressive],
        prior_weights: [0.25, 0.50, 0.25],
    }
}

fn world_with_target(
    world: &World,
    target: [[f64; ACTIONS]; CONTEXTS],
) -> World {
    let mut updated = *world;
    updated.target = target;
    updated
}

fn validate_portfolio(portfolio: &PolicyPortfolio) -> Result<(), String> {
    if (portfolio.prior_weights.iter().sum::<f64>() - 1.0).abs() > 1e-10
        || portfolio
            .prior_weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0 || *weight > 1.0)
    {
        return Err("policy portfolio prior weights must be a probability vector".into());
    }
    for policy in 0..PORTFOLIO_POLICIES {
        for context in 0..CONTEXTS {
            let probabilities = portfolio.targets[policy][context];
            if (probabilities.iter().sum::<f64>() - 1.0).abs() > 1e-10
                || probabilities
                    .iter()
                    .any(|probability| !probability.is_finite() || *probability < 0.0 || *probability > 1.0)
            {
                return Err(format!("invalid target policy {} in context {}", portfolio.names[policy], context));
            }
        }
    }
    Ok(())
}

fn portfolio_policy_mses(
    world: &World,
    allocation: &Allocation,
    portfolio: &PolicyPortfolio,
    evaluation_n: usize,
) -> Result<[f64; PORTFOLIO_POLICIES], String> {
    let mut output = [0.0; PORTFOLIO_POLICIES];
    for policy in 0..PORTFOLIO_POLICIES {
        let target_world = world_with_target(world, portfolio.targets[policy]);
        output[policy] = analytic_mse(&target_world, allocation, evaluation_n)?;
    }
    Ok(output)
}

fn portfolio_objective(
    mse: &[f64; PORTFOLIO_POLICIES],
    portfolio: &PolicyPortfolio,
    objective: PortfolioObjective,
) -> f64 {
    match objective {
        PortfolioObjective::PriorWeighted => (0..PORTFOLIO_POLICIES)
            .map(|policy| portfolio.prior_weights[policy] * mse[policy])
            .sum(),
        PortfolioObjective::WorstPolicy => mse.iter().copied().fold(0.0, f64::max),
    }
}

/// Exact finite-fixture portfolio oracle. It enumerates each context's feasible
/// source-count options and exhaustively combines them under the shared budget.
/// This is intentionally a small-instance certificate, not a scalable solver.
fn exact_portfolio_allocation(
    world: &World,
    portfolio: &PolicyPortfolio,
    objective: PortfolioObjective,
) -> Result<PortfolioExactResult, String> {
    world.validate()?;
    validate_portfolio(portfolio)?;
    let left = enumerate_context_options(world, 0, [true; SOURCES]);
    let right = enumerate_context_options(world, 1, [true; SOURCES]);
    let mut best_score = f64::INFINITY;
    let mut best_allocation = Allocation::default();
    let mut combinations_checked = 0usize;

    for left_option in &left {
        for right_option in &right {
            if left_option.cost + right_option.cost > world.budget {
                continue;
            }
            combinations_checked += 1;
            let mut allocation = Allocation::default();
            allocation.counts[0] = left_option.counts;
            allocation.counts[1] = right_option.counts;
            if !allocation.feasible(world) {
                return Err("portfolio oracle constructed an infeasible allocation".into());
            }
            let mses = portfolio_policy_mses(world, &allocation, portfolio, EVAL_N)?;
            let score = portfolio_objective(&mses, portfolio, objective);
            if score + EPS < best_score {
                best_score = score;
                best_allocation = allocation;
            }
        }
    }
    if !best_score.is_finite() {
        return Err("portfolio exact oracle found no feasible allocation".into());
    }
    Ok(PortfolioExactResult {
        allocation: best_allocation,
        objective: best_score,
        combinations_checked,
    })
}

fn greedy_portfolio_allocate(
    world: &World,
    portfolio: &PolicyPortfolio,
    objective: PortfolioObjective,
) -> Result<Allocation, String> {
    let mut allocation = Allocation::default();
    loop {
        let current_mses =
            portfolio_policy_mses(world, &allocation, portfolio, EVAL_N)?;
        let current_score = portfolio_objective(&current_mses, portfolio, objective);
        let mut best: Option<(f64, Allocation)> = None;
        for x in 0..CONTEXTS {
            for a in 0..ACTIONS {
                for k in 0..SOURCES {
                    let mut candidate = allocation;
                    candidate.counts[x][a][k] += 1;
                    if !candidate.feasible(world) {
                        continue;
                    }
                    let candidate_mses =
                        portfolio_policy_mses(world, &candidate, portfolio, EVAL_N)?;
                    let candidate_score =
                        portfolio_objective(&candidate_mses, portfolio, objective);
                    let gain = current_score - candidate_score;
                    if gain <= EPS {
                        continue;
                    }
                    let score = gain / world.source_cost[k] as f64;
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

/// Evaluates all portfolio policies on exactly the same contexts, logged actions,
/// factual rewards, and shared q-hat realization. The returned covariance matrix
/// therefore includes the dependence that matters for simultaneous screening and
/// best-policy selection, rather than treating K OPE estimates as independent.
fn multi_policy_monte_carlo(
    world: &World,
    allocation: &Allocation,
    portfolio: &PolicyPortfolio,
    evaluation_n: usize,
    replicates: usize,
    seed: u64,
) -> MultiPolicyMetrics {
    let true_values: [f64; PORTFOLIO_POLICIES] = std::array::from_fn(|policy| {
        target_value(&world_with_target(world, portfolio.targets[policy]))
    });
    let best_true = (0..PORTFOLIO_POLICIES)
        .max_by(|left, right| true_values[*left].total_cmp(&true_values[*right]))
        .unwrap_or(0);

    let mut errors = Vec::with_capacity(replicates);
    let mut selected_correct = 0usize;
    let mut total_regret = 0.0;
    for replicate in 0..replicates {
        let q = annotation_reward_model(world, allocation, mix_seed(seed, replicate, 71));
        let mut rng = Rng::new(mix_seed(seed, replicate, 89));
        let mut estimates = [0.0; PORTFOLIO_POLICIES];
        for _ in 0..evaluation_n {
            let context = if rng.uniform() < world.context_probability[0] {
                0
            } else {
                1
            };
            let action = sample_action(&world.behavior[context], &mut rng);
            let reward = if rng.uniform() < world.reward_mean[context][action] {
                1.0
            } else {
                0.0
            };
            let mu = world.behavior[context][action];
            for policy in 0..PORTFOLIO_POLICIES {
                let target = portfolio.targets[policy];
                let dm = (0..ACTIONS)
                    .map(|a| target[context][a] * q[context][a])
                    .sum::<f64>();
                let pi = target[context][action];
                estimates[policy] += dm + pi / mu * (reward - q[context][action]);
            }
        }
        for estimate in &mut estimates {
            *estimate /= evaluation_n as f64;
        }
        let errors_for_replicate: [f64; PORTFOLIO_POLICIES] =
            std::array::from_fn(|policy| estimates[policy] - true_values[policy]);
        let selected = (0..PORTFOLIO_POLICIES)
            .max_by(|left, right| estimates[*left].total_cmp(&estimates[*right]))
            .unwrap_or(0);
        if selected == best_true {
            selected_correct += 1;
        }
        total_regret += (true_values[best_true] - true_values[selected]).max(0.0);
        errors.push(errors_for_replicate);
    }

    let n = replicates as f64;
    let bias: [f64; PORTFOLIO_POLICIES] = std::array::from_fn(|policy| {
        errors.iter().map(|row| row[policy]).sum::<f64>() / n
    });
    let mse: [f64; PORTFOLIO_POLICIES] = std::array::from_fn(|policy| {
        errors.iter().map(|row| row[policy].powi(2)).sum::<f64>() / n
    });
    let variance: [f64; PORTFOLIO_POLICIES] =
        std::array::from_fn(|policy| (mse[policy] - bias[policy] * bias[policy]).max(0.0));
    let covariance: [[f64; PORTFOLIO_POLICIES]; PORTFOLIO_POLICIES] = std::array::from_fn(|left| {
        std::array::from_fn(|right| {
            let product = errors
                .iter()
                .map(|row| row[left] * row[right])
                .sum::<f64>()
                / n;
            product - bias[left] * bias[right]
        })
    });

    MultiPolicyMetrics {
        bias,
        variance,
        mse,
        covariance,
        selection_accuracy: selected_correct as f64 / n,
        expected_simple_regret: total_regret / n,
    }
}

fn policy_portfolio_benchmark_rows(replicates: usize, seed: u64) -> Result<String, String> {
    let mut output = String::new();
    for (scenario_index, world) in correlated_stress_worlds().iter().enumerate() {
        let portfolio = policy_portfolio(world);
        validate_portfolio(&portfolio)?;
        for policy in 0..PORTFOLIO_POLICIES {
            use std::fmt::Write;
            let _ = writeln!(
                output,
                "PORTFOLIO_CONFIG\t{}\t{}\t{}\t{:.8}",
                world.name,
                policy,
                portfolio.names[policy],
                portfolio.prior_weights[policy],
            );
        }
        let balanced_world = world_with_target(world, portfolio.targets[1]);
        let balanced_plan = exact_dynamic_program(&balanced_world, [true; SOURCES])?.allocation;
        let prior_exact =
            exact_portfolio_allocation(world, &portfolio, PortfolioObjective::PriorWeighted)?;
        let worst_exact =
            exact_portfolio_allocation(world, &portfolio, PortfolioObjective::WorstPolicy)?;
        let worst_greedy =
            greedy_portfolio_allocate(world, &portfolio, PortfolioObjective::WorstPolicy)?;

        let plans = [
            ("balanced_single_policy_exact", balanced_plan),
            ("portfolio_prior_weighted_exact", prior_exact.allocation),
            ("portfolio_worst_case_exact", worst_exact.allocation),
            ("portfolio_worst_case_greedy", worst_greedy),
        ];
        for (method_index, (method, allocation)) in plans.iter().enumerate() {
            if !allocation.feasible(world) {
                return Err(format!("portfolio benchmark plan is infeasible: {method}"));
            }
            let policy_mses =
                portfolio_policy_mses(world, allocation, &portfolio, EVAL_N)?;
            let prior_score =
                portfolio_objective(&policy_mses, &portfolio, PortfolioObjective::PriorWeighted);
            let worst_score =
                portfolio_objective(&policy_mses, &portfolio, PortfolioObjective::WorstPolicy);
            let mc = multi_policy_monte_carlo(
                world,
                allocation,
                &portfolio,
                EVAL_N,
                replicates,
                seed ^ (scenario_index as u64).wrapping_mul(0x517c_c1b7_2722_0a95)
                    ^ (method_index as u64).wrapping_mul(0x6eed_0e9d_a4d9_4a4f),
            );
            let prior_gap =
                ((prior_score - prior_exact.objective) / prior_exact.objective.max(EPS)).max(0.0);
            let worst_gap =
                ((worst_score - worst_exact.objective) / worst_exact.objective.max(EPS)).max(0.0);
            let sources = allocation_sources(allocation);
            use std::fmt::Write;
            let _ = writeln!(
                output,
                "PORTFOLIO_RESULT\t{}\t{}\t{}\t{}\t{}\t{}\t{:.10}\t{:.10}\t{:.10}\t{:.10}\t{:.10}\t{:.10}\t{:.10}\t{:.10}\t{:.10}\t{:.10}\t{:.10}\t{:.10}\t{:.10}\t{:.10}\t{:.10}\t{:.10}\t{:.10}\t{:.8}\t{:.10}\t{}\t{}\t{}\t{:.8}\t{:.8}",
                world.name,
                method,
                replicates,
                world.budget,
                allocation.cost(world),
                allocation.total_count(),
                prior_score,
                worst_score,
                policy_mses[0],
                policy_mses[1],
                policy_mses[2],
                mc.bias[0],
                mc.bias[1],
                mc.bias[2],
                mc.variance[0],
                mc.variance[1],
                mc.variance[2],
                mc.mse[0],
                mc.mse[1],
                mc.mse[2],
                mc.covariance[0][1],
                mc.covariance[0][2],
                mc.covariance[1][2],
                mc.selection_accuracy,
                mc.expected_simple_regret,
                sources[0],
                sources[1],
                sources[2],
                prior_gap,
                worst_gap,
            );
        }
    }
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
    fn zero_shared_variance_reduces_exactly_to_independent_annotation_formula() {
        let world = World::mixed_bias_cancellation();
        assert_eq!(world.shared_factor_variance, [0.0; 2]);
        let mut allocation = Allocation::default();
        allocation.counts[0][0] = [0, 2, 1];
        let x = 0;
        let a = 0;
        let nf = world.factual_count[x][a] as f64;
        let n = nf + allocation.counts[x][a].iter().map(|count| *count as f64).sum::<f64>();
        let sigma2 = reward_variance(world.reward_mean[x][a]);
        let independent_numerator = nf * sigma2
            + (0..SOURCES)
                .map(|k| {
                    allocation.count(x, a, k) as f64
                        * (sigma2 + world.source_excess_variance[x][a][k])
                })
                .sum::<f64>();
        let (_, q_variance) = cell_bias_and_variance(&world, &allocation, x, a);
        assert!((q_variance - independent_numerator / n.powi(2)).abs() < 1e-14);
    }

    #[test]
    fn low_rank_shared_source_covariance_is_positive_semidefinite() {
        for world in correlated_stress_worlds() {
            world.validate().unwrap();
            let covariance: [[f64; SOURCES]; SOURCES] =
                std::array::from_fn(|k| {
                    std::array::from_fn(|l| shared_source_covariance(&world, k, l))
                });
            for vector in [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, -1.0, 0.5],
                [-0.25, 0.75, 2.0],
                [1.0, 1.0, 1.0],
            ] {
                let quadratic = (0..SOURCES)
                    .flat_map(|k| (0..SOURCES).map(move |l| (k, l)))
                    .map(|(k, l)| vector[k] * covariance[k][l] * vector[l])
                    .sum::<f64>();
                assert!(quadratic >= -1e-12, "quadratic={quadratic}");
            }
        }
    }

    #[test]
    fn shared_panel_errors_inflate_repeated_source_samples_but_not_independent_source_mix() {
        let world = correlated_stress_worlds()[0];
        world.validate().unwrap();
        let independent = independent_profile(&world);
        let mut same_source = Allocation::default();
        same_source.counts[0][0][1] = 2;
        let mut diverse_mix = Allocation::default();
        diverse_mix.counts[0][0][1] = 1;
        diverse_mix.counts[0][0][2] = 1;
        let (_, same_corr) = cell_bias_and_variance(&world, &same_source, 0, 0);
        let (_, same_ind) = cell_bias_and_variance(&independent, &same_source, 0, 0);
        let (_, mix_corr) = cell_bias_and_variance(&world, &diverse_mix, 0, 0);
        let (_, mix_ind) = cell_bias_and_variance(&independent, &diverse_mix, 0, 0);
        assert!(same_corr > same_ind);
        assert!((mix_corr - mix_ind).abs() < 1e-14);
        assert!(weighted_q_variance_inflation(&world, &same_source) > 1.0);
    }

    #[test]
    fn analytic_correlated_ope_mse_matches_independent_monte_carlo() {
        for world in correlated_stress_worlds() {
            world.validate().unwrap();
            let allocation = exact_dynamic_program(&world, [true; SOURCES])
                .unwrap()
                .allocation;
            let analytic = analytic_mse(&world, &allocation, 64).unwrap();
            let mc = monte_carlo_evaluate(
                &world,
                &allocation,
                64,
                MC_TEST_REPLICATES,
                0x0560_c0a1 ^ world.name.len() as u64,
                None,
            );
            let relative_error = (mc.mse - analytic).abs() / analytic.max(1e-12);
            assert!(
                relative_error < 0.18,
                "scenario={}, analytic={analytic}, MC={}, relative={relative_error}",
                world.name,
                mc.mse
            );
            assert!(mc.bias.abs() < 0.04, "scenario={}, bias={}", world.name, mc.bias);
        }
    }

    #[test]
    fn robust_upper_correlation_plan_minimizes_its_declared_worst_case_profile() {
        for world in correlated_stress_worlds() {
            let independent = independent_profile(&world);
            let robust_profile = upper_correlation_profile(&world, 1.5).unwrap();
            let plugin = exact_dynamic_program(&independent, [true; SOURCES])
                .unwrap()
                .allocation;
            let robust = exact_dynamic_program(&robust_profile, [true; SOURCES])
                .unwrap()
                .allocation;
            let plugin_worst = analytic_mse(&robust_profile, &plugin, EVAL_N).unwrap();
            let robust_worst = analytic_mse(&robust_profile, &robust, EVAL_N).unwrap();
            assert!(robust_worst <= plugin_worst + 1e-12);
            assert!(robust.feasible(&world));
        }
    }

    #[test]
    fn invalid_shared_covariance_factor_and_marginal_variance_fail_closed() {
        let mut invalid = correlated_stress_worlds()[0];
        invalid.shared_factor_variance[0] = -0.01;
        assert!(invalid.validate().is_err());

        let mut excessive = correlated_stress_worlds()[0];
        excessive.shared_factor_variance[0] = 10.0;
        assert!(excessive.validate().is_err());

        let mut non_finite = correlated_stress_worlds()[0];
        non_finite.source_shared_loadings[1][0] = f64::NAN;
        assert!(non_finite.validate().is_err());

        assert!(upper_correlation_profile(&correlated_stress_worlds()[0], 0.9).is_err());
        let mut invalid_variance = correlated_stress_worlds()[0];
        invalid_variance.shared_factor_variance[1] = f64::INFINITY;
        assert!(invalid_variance.validate().is_err());
    }

    #[test]
    fn exact_portfolio_prior_and_worst_oracles_match_independent_exhaustive_enumeration() {
        let world = correlated_stress_worlds()[1];
        let portfolio = policy_portfolio(&world);
        let left = enumerate_context_options(&world, 0, [true; SOURCES]);
        let right = enumerate_context_options(&world, 1, [true; SOURCES]);

        let prior_exact = exact_portfolio_allocation(
            &world,
            &portfolio,
            PortfolioObjective::PriorWeighted,
        )
        .unwrap();
        let worst_exact = exact_portfolio_allocation(
            &world,
            &portfolio,
            PortfolioObjective::WorstPolicy,
        )
        .unwrap();

        let mut brute_prior = f64::INFINITY;
        let mut brute_worst = f64::INFINITY;
        let mut combinations = 0usize;
        for left_option in &left {
            for right_option in &right {
                if left_option.cost + right_option.cost > world.budget {
                    continue;
                }
                let mut allocation = Allocation::default();
                allocation.counts[0] = left_option.counts;
                allocation.counts[1] = right_option.counts;
                let mses = portfolio_policy_mses(&world, &allocation, &portfolio, EVAL_N).unwrap();
                brute_prior = brute_prior.min(portfolio_objective(
                    &mses,
                    &portfolio,
                    PortfolioObjective::PriorWeighted,
                ));
                brute_worst = brute_worst.min(portfolio_objective(
                    &mses,
                    &portfolio,
                    PortfolioObjective::WorstPolicy,
                ));
                combinations += 1;
            }
        }
        assert_eq!(prior_exact.combinations_checked, combinations);
        assert_eq!(worst_exact.combinations_checked, combinations);
        assert!((prior_exact.objective - brute_prior).abs() < 1e-12);
        assert!((worst_exact.objective - brute_worst).abs() < 1e-12);
    }

    #[test]
    fn portfolio_minimax_and_prior_plans_are_no_worse_than_single_policy_baseline() {
        for world in correlated_stress_worlds() {
            let portfolio = policy_portfolio(&world);
            let balanced_world = world_with_target(&world, portfolio.targets[1]);
            let single = exact_dynamic_program(&balanced_world, [true; SOURCES])
                .unwrap()
                .allocation;
            let prior = exact_portfolio_allocation(
                &world,
                &portfolio,
                PortfolioObjective::PriorWeighted,
            )
            .unwrap()
            .allocation;
            let worst = exact_portfolio_allocation(
                &world,
                &portfolio,
                PortfolioObjective::WorstPolicy,
            )
            .unwrap()
            .allocation;
            let single_mses = portfolio_policy_mses(&world, &single, &portfolio, EVAL_N).unwrap();
            let prior_mses = portfolio_policy_mses(&world, &prior, &portfolio, EVAL_N).unwrap();
            let worst_mses = portfolio_policy_mses(&world, &worst, &portfolio, EVAL_N).unwrap();
            assert!(
                portfolio_objective(&prior_mses, &portfolio, PortfolioObjective::PriorWeighted)
                    <= portfolio_objective(&single_mses, &portfolio, PortfolioObjective::PriorWeighted) + 1e-12
            );
            assert!(
                portfolio_objective(&worst_mses, &portfolio, PortfolioObjective::WorstPolicy)
                    <= portfolio_objective(&single_mses, &portfolio, PortfolioObjective::WorstPolicy) + 1e-12
            );
            assert!(prior.feasible(&world) && worst.feasible(&world));
        }
    }

    #[test]
    fn common_sample_multi_policy_covariance_is_psd_and_selection_metrics_are_bounded() {
        let world = correlated_stress_worlds()[2];
        let portfolio = policy_portfolio(&world);
        let allocation = exact_portfolio_allocation(
            &world,
            &portfolio,
            PortfolioObjective::WorstPolicy,
        )
        .unwrap()
        .allocation;
        let metrics = multi_policy_monte_carlo(&world, &allocation, &portfolio, 64, 1024, 0x0570_001);
        assert!((0.0..=1.0).contains(&metrics.selection_accuracy));
        assert!(metrics.expected_simple_regret >= 0.0);
        for policy in 0..PORTFOLIO_POLICIES {
            assert!(metrics.mse[policy] >= 0.0);
            assert!(metrics.variance[policy] >= 0.0);
            assert!((metrics.mse[policy] - metrics.bias[policy].powi(2) - metrics.variance[policy]).abs() < 1e-10);
        }
        for vector in [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, -1.0, 0.5],
            [-0.25, 0.75, 2.0],
        ] {
            let quadratic = (0..PORTFOLIO_POLICIES)
                .flat_map(|i| (0..PORTFOLIO_POLICIES).map(move |j| (i, j)))
                .map(|(i, j)| vector[i] * metrics.covariance[i][j] * vector[j])
                .sum::<f64>();
            assert!(quadratic >= -1e-10, "portfolio covariance quadratic={quadratic}");
        }
    }

    #[test]
    fn output_includes_deterministic_policy_portfolio_rows() {
        let first = policy_portfolio_benchmark_rows(2, 789).unwrap();
        let second = policy_portfolio_benchmark_rows(2, 789).unwrap();
        assert_eq!(first, second);
        let results: Vec<_> = first
            .lines()
            .filter(|line| line.starts_with("PORTFOLIO_RESULT\\\t"))
            .collect();
        let configs: Vec<_> = first
            .lines()
            .filter(|line| line.starts_with("PORTFOLIO_CONFIG\\\t"))
            .collect();
        assert_eq!(results.len(), 12);
        assert_eq!(configs.len(), 9);
        for row in results {
            assert_eq!(row.split('\\\t').count(), 31);
        }
    }

    #[test]
    fn output_includes_deterministic_correlated_stress_rows() {
        let first = correlation_benchmark_rows(2, 456).unwrap();
        let second = correlation_benchmark_rows(2, 456).unwrap();
        assert_eq!(first, second);
        let rows: Vec<_> = first
            .lines()
            .filter(|line| line.starts_with("CORR_RESULT\t"))
            .collect();
        assert_eq!(rows.len(), 18);
        for row in rows {
            assert_eq!(row.split('\t').count(), 22);
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
        assert_eq!(
            first
                .lines()
                .filter(|line| line.starts_with("CORR_RESULT\t"))
                .count(),
            18
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
