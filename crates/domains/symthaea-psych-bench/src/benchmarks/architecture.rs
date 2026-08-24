// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! SYM-ARCH-001: continual compositional adaptation architecture discrimination.
//!
//! This benchmark is deliberately narrower than a "general intelligence" claim.
//! It asks whether mechanisms already present in Symthaea buy measurable value on
//! three pre-registered phenomena under one deterministic task family:
//!
//! 1. retention across sequentially learned relational worlds,
//! 2. generalization to held-out factor combinations, and
//! 3. adaptation after a contingency reversal.
//!
//! The candidate is an HDC-LTC representation with Hebbian plasticity and a
//! common associative prototype readout. Ablations/controls are:
//!
//! - online linear SGD (simple conventional control),
//! - vanilla HDC + the same prototype readout,
//! - a fixed diagonal SSM + the same prototype readout, and
//! - HDC-LTC with liquid dynamics but frozen HDC-LTC weights.
//!
//! This is a *mechanism-level* benchmark. It does NOT exercise the full live
//! `CognitiveLoopService`, and it must not be reported as a frontier-model or
//! full-Symthaea comparison. In particular, it intentionally bypasses the old
//! reward-driven `cycle_with_hv()` live harness path whose reward-consumption gap
//! was frozen in `SYMTHAEA_UAL_LIVE_DIAGNOSTIC_P1_COLLAPSE_TRACE_2026-07-30.md`.

use serde::{Deserialize, Serialize};
use std::time::Instant;
use symthaea_core::hdc::ContinuousHV;
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_ssm::{SelectiveParams, SsmState};

const WORLDS: usize = 4;
const VALUES: usize = 4;
const HELD_OUT_PER_WORLD: usize = 4;
const REVERSAL_WINDOW: usize = 32;

/// Pre-registered experiment configuration. Defaults are the PR campaign.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymArch001Config {
    pub dimension: usize,
    pub seeds: usize,
    pub train_epochs_per_world: usize,
    pub reversal_epochs: usize,
    pub prototype_alpha: f32,
    pub hebbian_learning_rate: f32,
    pub win_margin: f64,
    pub regression_tolerance: f64,
}

impl Default for SymArch001Config {
    fn default() -> Self {
        Self {
            dimension: 512,
            seeds: 16,
            train_epochs_per_world: 16,
            reversal_epochs: 12,
            prototype_alpha: 0.15,
            hebbian_learning_rate: 0.002,
            win_margin: 0.05,
            regression_tolerance: 0.05,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedMetrics {
    pub seed: u64,
    /// Mean accuracy over each world's *trained* combinations after all worlds.
    pub final_retention_accuracy: f64,
    /// Mean accuracy over balanced held-out combinations after all worlds.
    pub heldout_compositional_accuracy: f64,
    /// Mean peak-to-final drop on the trained combinations. Lower is better.
    pub mean_forgetting: f64,
    /// Accuracy on the final 32 pre-update predictions after the rule is inverted.
    pub reversal_final_accuracy: f64,
    /// First trial index whose trailing 32-trial window reaches >= 75%.
    /// `reversal_trials + 1` means the criterion was never reached.
    pub reversal_adaptation_latency: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeanCi95 {
    pub mean: f64,
    pub ci95_low: f64,
    pub ci95_high: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSummary {
    pub agent: String,
    pub retention: MeanCi95,
    pub compositional: MeanCi95,
    pub forgetting: MeanCi95,
    pub reversal_final: MeanCi95,
    pub reversal_latency: MeanCi95,
    pub per_seed: Vec<SeedMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionEvidence {
    pub verdict: String,
    pub candidate: String,
    pub retention_delta_vs_best_control: f64,
    pub compositional_delta_vs_best_control: f64,
    pub reversal_delta_vs_best_control: f64,
    pub forgetting_delta_vs_best_control: f64,
    pub target_wins_at_margin: usize,
    pub target_regressions_beyond_tolerance: usize,
    pub preregistered_rule: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceEvidence {
    pub wall_time_ms: u128,
    pub observations_per_agent: usize,
    pub representation_dimension: usize,
    pub ssm_state_per_dimension: usize,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymArch001Report {
    pub schema: String,
    pub source_revision: Option<String>,
    pub scope: String,
    pub config: SymArch001Config,
    pub agents: Vec<AgentSummary>,
    pub decision: DecisionEvidence,
    pub resources: ResourceEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Pair {
    i: usize,
    j: usize,
}

fn rule(world: usize, p: Pair) -> bool {
    match world {
        // Balanced relational rules: 8 positive / 8 negative each.
        0 => (p.i & 1) == (p.j & 1),
        1 => ((p.i + p.j) & 3) < 2,
        2 => ((p.i + VALUES - p.j) & 3) < 2,
        3 => ((p.i ^ p.j) & 2) == 0,
        _ => unreachable!("world index is bounded by WORLDS"),
    }
}

fn held_out(world: usize) -> Vec<Pair> {
    // Deterministically take the first two positives and first two negatives.
    // This makes both the training set (6/6) and held-out set (2/2) balanced.
    let mut positives = Vec::with_capacity(2);
    let mut negatives = Vec::with_capacity(2);
    for i in 0..VALUES {
        for j in 0..VALUES {
            let p = Pair { i, j };
            if rule(world, p) {
                if positives.len() < 2 {
                    positives.push(p);
                }
            } else if negatives.len() < 2 {
                negatives.push(p);
            }
        }
    }
    positives.extend(negatives);
    debug_assert_eq!(positives.len(), HELD_OUT_PER_WORLD);
    positives
}

fn training_pairs(world: usize) -> Vec<Pair> {
    let held = held_out(world);
    let mut out = Vec::with_capacity(VALUES * VALUES - HELD_OUT_PER_WORLD);
    for i in 0..VALUES {
        for j in 0..VALUES {
            let p = Pair { i, j };
            if !held.contains(&p) {
                out.push(p);
            }
        }
    }
    out
}

#[derive(Clone)]
struct TaskSpace {
    values: [ContinuousHV; VALUES],
    worlds: [ContinuousHV; WORLDS],
    shape_role: ContinuousHV,
    texture_role: ContinuousHV,
}

impl TaskSpace {
    fn new(dim: usize, seed: u64) -> Self {
        let values = std::array::from_fn(|i| {
            ContinuousHV::random(dim, mix_seed(seed, 0x1000 + i as u64))
        });
        let worlds = std::array::from_fn(|i| {
            ContinuousHV::random(dim, mix_seed(seed, 0x2000 + i as u64))
        });
        Self {
            values,
            worlds,
            shape_role: ContinuousHV::random(dim, mix_seed(seed, 0x3001)),
            texture_role: ContinuousHV::random(dim, mix_seed(seed, 0x3002)),
        }
    }

    fn encode(&self, world: usize, p: Pair) -> ContinuousHV {
        let shape = self.values[p.i].bind(&self.shape_role);
        let texture = self.values[p.j].bind(&self.texture_role);
        shape
            .bind(&texture)
            .bind(&self.worlds[world])
            .normalize()
    }
}

#[derive(Debug, Clone)]
struct PrototypeMemory {
    negative: ContinuousHV,
    positive: ContinuousHV,
    negative_seen: usize,
    positive_seen: usize,
    alpha: f32,
}

impl PrototypeMemory {
    fn new(dim: usize, alpha: f32) -> Self {
        Self {
            negative: ContinuousHV::zero(dim),
            positive: ContinuousHV::zero(dim),
            negative_seen: 0,
            positive_seen: 0,
            alpha,
        }
    }

    fn update(&mut self, representation: &ContinuousHV, label: bool) {
        let (slot, seen) = if label {
            (&mut self.positive, &mut self.positive_seen)
        } else {
            (&mut self.negative, &mut self.negative_seen)
        };
        if *seen == 0 {
            *slot = representation.normalize();
        } else {
            let old = slot.clone();
            *slot = ContinuousHV::weighted_bundle(
                &[&old, representation],
                &[1.0 - self.alpha, self.alpha],
            )
            .normalize();
        }
        *seen += 1;
    }

    fn predict(&self, representation: &ContinuousHV) -> bool {
        match (self.negative_seen, self.positive_seen) {
            (0, 0) => false,
            (0, _) => true,
            (_, 0) => false,
            _ => {
                representation.similarity(&self.positive)
                    > representation.similarity(&self.negative)
            }
        }
    }
}

#[derive(Debug, Clone)]
struct LinearSgd {
    weights: Vec<f32>,
    bias: f32,
    learning_rate: f32,
}

impl LinearSgd {
    fn new(dim: usize) -> Self {
        Self {
            weights: vec![0.0; dim],
            bias: 0.0,
            learning_rate: 0.08,
        }
    }

    fn logit(&self, x: &ContinuousHV) -> f32 {
        self.bias
            + self
                .weights
                .iter()
                .zip(x.values.iter())
                .map(|(w, v)| w * v)
                .sum::<f32>()
    }

    fn predict(&self, x: &ContinuousHV) -> bool {
        self.logit(x) >= 0.0
    }

    fn observe(&mut self, x: &ContinuousHV, label: bool) {
        let z = self.logit(x).clamp(-20.0, 20.0);
        let p = 1.0 / (1.0 + (-z).exp());
        let target = if label { 1.0 } else { 0.0 };
        let error = target - p;
        for (w, v) in self.weights.iter_mut().zip(x.values.iter()) {
            *w += self.learning_rate * error * *v;
        }
        self.bias += self.learning_rate * error;
    }
}

#[derive(Clone)]
enum Agent {
    Linear(LinearSgd),
    Vanilla {
        memory: PrototypeMemory,
    },
    Ssm {
        params: SelectiveParams,
        state: SsmState,
        memory: PrototypeMemory,
        dim: usize,
    },
    LiquidFrozen {
        liquid: HdcLtcUnifiedNeuron,
        memory: PrototypeMemory,
    },
    LiquidHebbian {
        liquid: HdcLtcUnifiedNeuron,
        memory: PrototypeMemory,
        hebbian_lr: f32,
    },
}

impl Agent {
    fn all(config: &SymArch001Config, seed: u64) -> Vec<Self> {
        let dim = config.dimension;
        let liquid_config = UnifiedConfig {
            dimension: dim,
            learning_rate: config.hebbian_learning_rate,
            ..Default::default()
        };
        vec![
            Self::Linear(LinearSgd::new(dim)),
            Self::Vanilla {
                memory: PrototypeMemory::new(dim, config.prototype_alpha),
            },
            Self::Ssm {
                params: SelectiveParams::new(dim, 2),
                state: SsmState::new(dim, 2),
                memory: PrototypeMemory::new(dim, config.prototype_alpha),
                dim,
            },
            Self::LiquidFrozen {
                liquid: HdcLtcUnifiedNeuron::new(liquid_config.clone(), mix_seed(seed, 0xA001)),
                memory: PrototypeMemory::new(dim, config.prototype_alpha),
            },
            Self::LiquidHebbian {
                liquid: HdcLtcUnifiedNeuron::new(liquid_config, mix_seed(seed, 0xA002)),
                memory: PrototypeMemory::new(dim, config.prototype_alpha),
                hebbian_lr: config.hebbian_learning_rate,
            },
        ]
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Linear(_) => "linear_sgd",
            Self::Vanilla { .. } => "vanilla_hdc",
            Self::Ssm { .. } => "fixed_diagonal_ssm",
            Self::LiquidFrozen { .. } => "hdc_ltc_frozen",
            Self::LiquidHebbian { .. } => "hdc_ltc_hebbian",
        }
    }

    fn predict(&self, input: &ContinuousHV) -> bool {
        match self {
            Self::Linear(linear) => linear.predict(input),
            Self::Vanilla { memory } => memory.predict(input),
            Self::Ssm {
                params,
                state,
                memory,
                dim,
            } => {
                let mut shadow = state.clone();
                let mut output = vec![0.0_f32; *dim];
                shadow.step(&input.values, params, &mut output);
                let representation = ContinuousHV::from_values(output).normalize();
                memory.predict(&representation)
            }
            Self::LiquidFrozen { liquid, memory }
            | Self::LiquidHebbian { liquid, memory, .. } => {
                let mut shadow = liquid.clone();
                shadow.evolve_closed_form(0.05, input);
                memory.predict(shadow.state())
            }
        }
    }

    fn observe(&mut self, input: &ContinuousHV, label: bool) {
        match self {
            Self::Linear(linear) => linear.observe(input, label),
            Self::Vanilla { memory } => memory.update(input, label),
            Self::Ssm {
                params,
                state,
                memory,
                dim,
            } => {
                let mut output = vec![0.0_f32; *dim];
                state.step(&input.values, params, &mut output);
                let representation = ContinuousHV::from_values(output).normalize();
                memory.update(&representation, label);
            }
            Self::LiquidFrozen { liquid, memory } => {
                liquid.evolve_closed_form(0.05, input);
                let representation = liquid.state().clone();
                memory.update(&representation, label);
            }
            Self::LiquidHebbian {
                liquid,
                memory,
                hebbian_lr,
            } => {
                liquid.evolve_closed_form(0.05, input);
                let representation = liquid.state().clone();
                memory.update(&representation, label);
                liquid.hebbian_update(input, Some(*hebbian_lr));
            }
        }
    }
}

fn accuracy(agent: &Agent, task: &TaskSpace, world: usize, pairs: &[Pair], invert: bool) -> f64 {
    let correct = pairs
        .iter()
        .filter(|&&p| {
            let target = rule(world, p) ^ invert;
            agent.predict(&task.encode(world, p)) == target
        })
        .count();
    correct as f64 / pairs.len() as f64
}

fn shuffled_pairs(world: usize, seed: u64, epoch: usize) -> Vec<Pair> {
    let mut pairs = training_pairs(world);
    let mut rng = mix_seed(seed, 0x5000 + world as u64 * 257 + epoch as u64);
    for i in (1..pairs.len()).rev() {
        let j = (next_u64(&mut rng) as usize) % (i + 1);
        pairs.swap(i, j);
    }
    pairs
}

fn run_seed(config: &SymArch001Config, seed: u64) -> Vec<(String, SeedMetrics)> {
    let task = TaskSpace::new(config.dimension, seed);
    let mut agents = Agent::all(config, seed);
    let mut peaks = vec![[0.0_f64; WORLDS]; agents.len()];

    // Sequential worlds A -> B -> C -> D, with no replay of old worlds.
    for world in 0..WORLDS {
        for epoch in 0..config.train_epochs_per_world {
            for p in shuffled_pairs(world, seed, epoch) {
                let x = task.encode(world, p);
                let y = rule(world, p);
                for agent in &mut agents {
                    agent.observe(&x, y);
                }
            }
        }

        // Evaluate every world seen so far, but do not mutate agent state.
        for (agent_idx, agent) in agents.iter().enumerate() {
            for seen_world in 0..=world {
                let score = accuracy(agent, &task, seen_world, &training_pairs(seen_world), false);
                peaks[agent_idx][seen_world] = peaks[agent_idx][seen_world].max(score);
            }
        }
    }

    let all_pairs: Vec<Pair> = (0..VALUES)
        .flat_map(|i| (0..VALUES).map(move |j| Pair { i, j }))
        .collect();
    let reversal_trials = config.reversal_epochs * all_pairs.len();

    agents
        .iter()
        .enumerate()
        .map(|(agent_idx, agent)| {
            let final_by_world: Vec<f64> = (0..WORLDS)
                .map(|world| accuracy(agent, &task, world, &training_pairs(world), false))
                .collect();
            let heldout_by_world: Vec<f64> = (0..WORLDS)
                .map(|world| accuracy(agent, &task, world, &held_out(world), false))
                .collect();
            let forgetting: Vec<f64> = (0..WORLDS)
                .map(|world| (peaks[agent_idx][world] - final_by_world[world]).max(0.0))
                .collect();

            // Isolate the reversal from the retention/composition measurement.
            // Prediction is scored *before* each supervised update.
            let mut reversed = agent.clone();
            let mut outcomes = Vec::with_capacity(reversal_trials);
            let mut adaptation_latency = reversal_trials + 1;
            for epoch in 0..config.reversal_epochs {
                let mut order = all_pairs.clone();
                let mut rng = mix_seed(seed, 0x9000 + epoch as u64);
                for i in (1..order.len()).rev() {
                    let j = (next_u64(&mut rng) as usize) % (i + 1);
                    order.swap(i, j);
                }
                for p in order {
                    let x = task.encode(WORLDS - 1, p);
                    let target = !rule(WORLDS - 1, p);
                    let correct = reversed.predict(&x) == target;
                    outcomes.push(correct);
                    reversed.observe(&x, target);
                    if outcomes.len() >= REVERSAL_WINDOW && adaptation_latency > reversal_trials {
                        let tail = &outcomes[outcomes.len() - REVERSAL_WINDOW..];
                        let rolling = tail.iter().filter(|&&v| v).count() as f64
                            / REVERSAL_WINDOW as f64;
                        if rolling >= 0.75 {
                            adaptation_latency = outcomes.len();
                        }
                    }
                }
            }
            let tail_len = REVERSAL_WINDOW.min(outcomes.len());
            let reversal_final = outcomes[outcomes.len() - tail_len..]
                .iter()
                .filter(|&&v| v)
                .count() as f64
                / tail_len as f64;

            (
                agent.name().to_string(),
                SeedMetrics {
                    seed,
                    final_retention_accuracy: mean(&final_by_world),
                    heldout_compositional_accuracy: mean(&heldout_by_world),
                    mean_forgetting: mean(&forgetting),
                    reversal_final_accuracy: reversal_final,
                    reversal_adaptation_latency: adaptation_latency,
                },
            )
        })
        .collect()
}

/// Run the pre-registered SYM-ARCH-001 campaign.
pub fn run_sym_arch_001(
    config: SymArch001Config,
    source_revision: Option<String>,
) -> SymArch001Report {
    assert!(config.dimension >= 32, "dimension must be >= 32");
    assert!(config.seeds > 0, "at least one seed is required");
    assert!(config.train_epochs_per_world > 0);
    assert!(config.reversal_epochs > 0);

    let started = Instant::now();
    let mut by_agent: Vec<(String, Vec<SeedMetrics>)> = Vec::new();
    for seed_idx in 0..config.seeds {
        let seed = mix_seed(0x5A17_2026_0000_0001, seed_idx as u64);
        for (name, metrics) in run_seed(&config, seed) {
            if let Some((_, rows)) = by_agent.iter_mut().find(|(n, _)| n == &name) {
                rows.push(metrics);
            } else {
                by_agent.push((name, vec![metrics]));
            }
        }
    }

    let agents: Vec<AgentSummary> = by_agent
        .into_iter()
        .map(|(agent, rows)| AgentSummary {
            retention: mean_ci95(rows.iter().map(|r| r.final_retention_accuracy)),
            compositional: mean_ci95(rows.iter().map(|r| r.heldout_compositional_accuracy)),
            forgetting: mean_ci95(rows.iter().map(|r| r.mean_forgetting)),
            reversal_final: mean_ci95(rows.iter().map(|r| r.reversal_final_accuracy)),
            reversal_latency: mean_ci95(
                rows.iter()
                    .map(|r| r.reversal_adaptation_latency as f64),
            ),
            agent,
            per_seed: rows,
        })
        .collect();

    let decision = classify(&agents, &config);
    let train_pairs_per_world = VALUES * VALUES - HELD_OUT_PER_WORLD;
    let observations_per_agent = WORLDS * config.train_epochs_per_world * train_pairs_per_world
        + config.reversal_epochs * VALUES * VALUES;

    SymArch001Report {
        schema: "symthaea.sym-arch-001.v1".to_string(),
        source_revision,
        scope: "mechanism-level HDC/HDC-LTC/SSM/linear discrimination; not full live Symthaea"
            .to_string(),
        config: config.clone(),
        agents,
        decision,
        resources: ResourceEvidence {
            wall_time_ms: started.elapsed().as_millis(),
            observations_per_agent,
            representation_dimension: config.dimension,
            ssm_state_per_dimension: 2,
            note: "Wall time is observational and runner-dependent; sample counts and dimensions are the reproducible resource-normalization anchors.".to_string(),
        },
    }
}

fn classify(agents: &[AgentSummary], config: &SymArch001Config) -> DecisionEvidence {
    let candidate_name = "hdc_ltc_hebbian";
    let candidate = agents
        .iter()
        .find(|a| a.agent == candidate_name)
        .expect("candidate summary must exist");
    let controls: Vec<&AgentSummary> = agents
        .iter()
        .filter(|a| a.agent != candidate_name)
        .collect();

    let best_retention = controls
        .iter()
        .map(|a| a.retention.mean)
        .fold(f64::NEG_INFINITY, f64::max);
    let best_composition = controls
        .iter()
        .map(|a| a.compositional.mean)
        .fold(f64::NEG_INFINITY, f64::max);
    let best_reversal = controls
        .iter()
        .map(|a| a.reversal_final.mean)
        .fold(f64::NEG_INFINITY, f64::max);
    let best_forgetting = controls
        .iter()
        .map(|a| a.forgetting.mean)
        .fold(f64::INFINITY, f64::min);

    let retention_delta = candidate.retention.mean - best_retention;
    let composition_delta = candidate.compositional.mean - best_composition;
    let reversal_delta = candidate.reversal_final.mean - best_reversal;
    // Positive is better: control forgetting minus candidate forgetting.
    let forgetting_delta = best_forgetting - candidate.forgetting.mean;

    let targets = [retention_delta, composition_delta, reversal_delta];
    let wins = targets
        .iter()
        .filter(|&&d| d >= config.win_margin)
        .count();
    let regressions = targets
        .iter()
        .filter(|&&d| d < -config.regression_tolerance)
        .count();
    let forgetting_regression = forgetting_delta < -config.regression_tolerance;

    let verdict = if wins >= 2 && regressions == 0 && !forgetting_regression {
        "PASS"
    } else if regressions >= 2 || forgetting_regression {
        "NEGATIVE"
    } else if wins >= 1 && regressions == 0 {
        "MIXED"
    } else {
        "NULL"
    };

    DecisionEvidence {
        verdict: verdict.to_string(),
        candidate: candidate_name.to_string(),
        retention_delta_vs_best_control: retention_delta,
        compositional_delta_vs_best_control: composition_delta,
        reversal_delta_vs_best_control: reversal_delta,
        forgetting_delta_vs_best_control: forgetting_delta,
        target_wins_at_margin: wins,
        target_regressions_beyond_tolerance: regressions + usize::from(forgetting_regression),
        preregistered_rule: format!(
            "PASS iff candidate beats the strongest control by >= {:.2} on at least two of retention/composition/reversal, loses by no more than {:.2} on the other target(s), and forgetting is not worse than the best control by > {:.2}; MIXED requires >=1 target win with no target regressions; NEGATIVE requires >=2 target regressions or a forgetting regression; otherwise NULL.",
            config.win_margin, config.regression_tolerance, config.regression_tolerance
        ),
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn mean_ci95<I>(values: I) -> MeanCi95
where
    I: IntoIterator<Item = f64>,
{
    let values: Vec<f64> = values.into_iter().collect();
    let m = mean(&values);
    if values.len() < 2 {
        return MeanCi95 {
            mean: m,
            ci95_low: m,
            ci95_high: m,
        };
    }
    let variance = values
        .iter()
        .map(|v| (v - m).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    let half = 1.96 * variance.sqrt() / (values.len() as f64).sqrt();
    MeanCi95 {
        mean: m,
        ci95_low: (m - half).max(0.0),
        ci95_high: m + half,
    }
}

fn mix_seed(seed: u64, salt: u64) -> u64 {
    let mut x = seed ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    // SplitMix64 finalizer.
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn next_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_splits_are_balanced_and_disjoint() {
        for world in 0..WORLDS {
            let held = held_out(world);
            let train = training_pairs(world);
            assert_eq!(held.len(), 4);
            assert_eq!(train.len(), 12);
            assert!(held.iter().all(|p| !train.contains(p)));
            assert_eq!(held.iter().filter(|&&p| rule(world, p)).count(), 2);
            assert_eq!(train.iter().filter(|&&p| rule(world, p)).count(), 6);
        }
    }

    #[test]
    fn smoke_campaign_produces_finite_metrics() {
        let report = run_sym_arch_001(
            SymArch001Config {
                dimension: 64,
                seeds: 1,
                train_epochs_per_world: 1,
                reversal_epochs: 2,
                ..Default::default()
            },
            None,
        );
        assert_eq!(report.agents.len(), 5);
        for agent in &report.agents {
            for value in [
                agent.retention.mean,
                agent.compositional.mean,
                agent.forgetting.mean,
                agent.reversal_final.mean,
                agent.reversal_latency.mean,
            ] {
                assert!(value.is_finite(), "{} emitted non-finite metric", agent.agent);
            }
        }
        assert!(matches!(
            report.decision.verdict.as_str(),
            "PASS" | "MIXED" | "NULL" | "NEGATIVE"
        ));
    }

    #[test]
    fn fixed_seed_is_metric_deterministic() {
        let cfg = SymArch001Config {
            dimension: 64,
            seeds: 1,
            train_epochs_per_world: 1,
            reversal_epochs: 2,
            ..Default::default()
        };
        let a = run_sym_arch_001(cfg.clone(), None);
        let b = run_sym_arch_001(cfg, None);
        for (left, right) in a.agents.iter().zip(b.agents.iter()) {
            assert_eq!(left.agent, right.agent);
            assert_eq!(left.retention.mean, right.retention.mean);
            assert_eq!(left.compositional.mean, right.compositional.mean);
            assert_eq!(left.forgetting.mean, right.forgetting.mean);
            assert_eq!(left.reversal_final.mean, right.reversal_final.mean);
            assert_eq!(left.reversal_latency.mean, right.reversal_latency.mean);
        }
    }
}
