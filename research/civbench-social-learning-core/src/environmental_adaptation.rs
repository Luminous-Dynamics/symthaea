// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! Environmental-shift cultural-learning benchmark.
//!
//! Inspired by Roberts-Gaal, Bolic & Cushman (PNAS, 2025), which found that
//! stable environments elicited more procedural transmission while variable
//! environments elicited more outcome/causal information. This module is a
//! deterministic, synthetic mechanism benchmark: it is not a reproduction of
//! the human fishing task or its statistical analysis.

use core::fmt;

pub const ACTION_COUNT: usize = 4;
pub const OUTCOME_COUNT: usize = 4;
pub const TRIALS_PER_GENERATION: usize = 12;
pub const GENERATIONS: usize = 6;
pub const DEFAULT_TARGET_OUTCOME: u8 = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvironmentRegime {
    /// The action-to-outcome mapping remains fixed across generations.
    Stable,
    /// The outcome labels are reversed on every generation transition.
    ReverseEveryGeneration,
    /// A new action-to-outcome permutation is sampled each generation.
    RandomEveryGeneration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageKind {
    Procedure,
    OutcomeGoal,
    CausalModel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingPolicy {
    AlwaysProcedure,
    AlwaysOutcomeGoal,
    AlwaysCausalModel,
    /// Use a short procedure when it worked well; otherwise send portable
    /// outcome information or a partial causal map after observed mismatch.
    Adaptive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnvironmentMap {
    pub outcome_by_action: [u8; ACTION_COUNT],
}

impl EnvironmentMap {
    pub fn validate(self) -> Result<Self, EnvironmentError> {
        let mut seen = [false; OUTCOME_COUNT];
        for outcome in self.outcome_by_action {
            if usize::from(outcome) >= OUTCOME_COUNT {
                return Err(EnvironmentError::InvalidOutcome { outcome });
            }
            if seen[usize::from(outcome)] {
                return Err(EnvironmentError::NonPermutation);
            }
            seen[usize::from(outcome)] = true;
        }
        Ok(self)
    }

    pub fn target_action(self, target_outcome: u8) -> Option<u8> {
        self.outcome_by_action
            .iter()
            .position(|&outcome| outcome == target_outcome)
            .map(|index| index as u8)
    }

    pub fn reversed(self) -> Self {
        Self {
            outcome_by_action: self.outcome_by_action.map(|outcome| {
                (OUTCOME_COUNT as u8 - 1) - outcome
            }),
        }
    }
}

/// A transmitted representation. Causal models may be partial: an absent
/// mapping means the teacher did not have enough observations for that action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TeachingMessage {
    Procedure {
        recommended_action: u8,
    },
    OutcomeGoal {
        target_outcome: u8,
    },
    CausalModel {
        outcome_by_action: [Option<u8>; ACTION_COUNT],
        target_outcome: u8,
    },
}

impl TeachingMessage {
    pub fn kind(self) -> MessageKind {
        match self {
            Self::Procedure { .. } => MessageKind::Procedure,
            Self::OutcomeGoal { .. } => MessageKind::OutcomeGoal,
            Self::CausalModel { .. } => MessageKind::CausalModel,
        }
    }

    /// A small, explicit payload-size proxy. This is not a language-token or
    /// wire-protocol measurement.
    pub fn payload_bits(self) -> usize {
        match self {
            Self::Procedure { .. } => 2,
            Self::OutcomeGoal { .. } => 2,
            Self::CausalModel { outcome_by_action, .. } => {
                // Two target bits, one known/unknown bit and two outcome bits
                // per action. The bitmap makes partial maps unambiguous.
                2 + ACTION_COUNT * 3
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvironmentError {
    InvalidOutcome { outcome: u8 },
    NonPermutation,
    InvalidAction { action: u8 },
    InvalidTarget { target: u8 },
    InvalidProbability,
    InvalidGenerationCount,
    InvalidTrialCount,
}

impl fmt::Display for EnvironmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOutcome { outcome } => write!(f, "outcome {outcome} is outside 0..{OUTCOME_COUNT}"),
            Self::NonPermutation => write!(f, "environment mapping must be a permutation"),
            Self::InvalidAction { action } => write!(f, "action {action} is outside 0..{ACTION_COUNT}"),
            Self::InvalidTarget { target } => write!(f, "target outcome {target} is outside 0..{OUTCOME_COUNT}"),
            Self::InvalidProbability => write!(f, "correct-outcome probability must be finite and in (0.5, 1]"),
            Self::InvalidGenerationCount => write!(f, "generation count must be positive"),
            Self::InvalidTrialCount => write!(f, "trial count must be positive"),
        }
    }
}
impl std::error::Error for EnvironmentError {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub generations: usize,
    pub trials_per_generation: usize,
    pub target_outcome: u8,
    /// Probability that an action yields its mapped outcome. The remainder is
    /// distributed uniformly over the other outcomes.
    pub mapping_fidelity: f64,
    /// Threshold for sending a compact procedure in the adaptive treatment.
    pub procedure_success_threshold: f64,
    /// Minimum observed action count before sending a causal map rather than
    /// a goal-only message after a poor procedure outcome.
    pub causal_min_observed_actions: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            generations: GENERATIONS,
            trials_per_generation: TRIALS_PER_GENERATION,
            target_outcome: DEFAULT_TARGET_OUTCOME,
            mapping_fidelity: 0.9,
            procedure_success_threshold: 0.72,
            causal_min_observed_actions: 3,
        }
    }
}

impl Config {
    pub fn validate(self) -> Result<Self, EnvironmentError> {
        if self.generations == 0 {
            return Err(EnvironmentError::InvalidGenerationCount);
        }
        if self.trials_per_generation == 0 {
            return Err(EnvironmentError::InvalidTrialCount);
        }
        if usize::from(self.target_outcome) >= OUTCOME_COUNT {
            return Err(EnvironmentError::InvalidTarget { target: self.target_outcome });
        }
        if !self.mapping_fidelity.is_finite()
            || self.mapping_fidelity <= 0.5
            || self.mapping_fidelity > 1.0
        {
            return Err(EnvironmentError::InvalidProbability);
        }
        if !self.procedure_success_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.procedure_success_threshold)
        {
            return Err(EnvironmentError::InvalidProbability);
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrialRecord {
    pub generation: usize,
    pub trial: usize,
    pub offered_actions: [u8; 2],
    pub chosen_action: u8,
    pub observed_outcome: u8,
    pub hit_target: bool,
    /// Evaluator-only diagnostic; never supplied to the learner.
    pub action_was_optimal: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationMetrics {
    pub generation: usize,
    pub incoming_message: Option<MessageKind>,
    pub outgoing_message: MessageKind,
    pub outgoing_payload_bits: usize,
    pub trials: usize,
    pub target_hits: usize,
    pub target_hit_rate: f64,
    pub optimal_action_choices: usize,
    pub optimal_action_rate: f64,
    pub distinct_actions_observed: usize,
    pub causal_conflicts: usize,
    pub first_optimal_choice_trial: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransmissionRun {
    pub regime: EnvironmentRegime,
    pub encoding_policy: EncodingPolicy,
    pub metrics: Vec<GenerationMetrics>,
    pub trials: Vec<TrialRecord>,
    pub messages: Vec<TeachingMessage>,
}

impl TransmissionRun {
    pub fn mean_target_hit_rate(&self) -> f64 {
        if self.metrics.is_empty() {
            0.0
        } else {
            self.metrics.iter().map(|m| m.target_hit_rate).sum::<f64>()
                / self.metrics.len() as f64
        }
    }

    pub fn mean_payload_bits(&self) -> f64 {
        if self.metrics.is_empty() {
            0.0
        } else {
            self.metrics.iter().map(|m| m.outgoing_payload_bits as f64).sum::<f64>()
                / self.metrics.len() as f64
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

    fn index(&mut self, upper: usize) -> usize {
        (self.next_u64() as usize) % upper
    }

    fn bernoulli(&mut self, probability: f64) -> bool {
        let draw = ((self.next_u64() >> 11) as f64 + 0.5)
            / 9_007_199_254_740_992.0;
        draw < probability
    }
}

fn shuffled_base_mapping(rng: &mut SplitMix64) -> EnvironmentMap {
    let mut values = [0_u8, 1, 2, 3];
    for i in 0..ACTION_COUNT {
        let j = i + rng.index(ACTION_COUNT - i);
        values.swap(i, j);
    }
    EnvironmentMap { outcome_by_action: values }
}

fn mapping_for_generation(
    base: EnvironmentMap,
    regime: EnvironmentRegime,
    generation: usize,
    seed: u64,
) -> EnvironmentMap {
    match regime {
        EnvironmentRegime::Stable => base,
        EnvironmentRegime::ReverseEveryGeneration => {
            if generation % 2 == 0 { base } else { base.reversed() }
        }
        EnvironmentRegime::RandomEveryGeneration => {
            let mut rng = SplitMix64::new(
                seed ^ (generation as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93),
            );
            shuffled_base_mapping(&mut rng)
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LearnerMemory {
    outcome_by_action: [Option<u8>; ACTION_COUNT],
    observations_by_action: [u16; ACTION_COUNT],
    target_hits_by_action: [u16; ACTION_COUNT],
    causal_prior: [Option<u8>; ACTION_COUNT],
    procedure_action: Option<u8>,
    target_outcome: u8,
    causal_conflicts: usize,
}

impl LearnerMemory {
    fn new(message: Option<TeachingMessage>, target_outcome: u8) -> Self {
        let mut memory = Self {
            outcome_by_action: [None; ACTION_COUNT],
            observations_by_action: [0; ACTION_COUNT],
            target_hits_by_action: [0; ACTION_COUNT],
            causal_prior: [None; ACTION_COUNT],
            procedure_action: None,
            target_outcome,
            causal_conflicts: 0,
        };
        match message {
            Some(TeachingMessage::Procedure { recommended_action }) => {
                memory.procedure_action = Some(recommended_action);
            }
            Some(TeachingMessage::OutcomeGoal { target_outcome }) => {
                memory.target_outcome = target_outcome;
            }
            Some(TeachingMessage::CausalModel {
                outcome_by_action,
                target_outcome,
            }) => {
                memory.causal_prior = outcome_by_action;
                memory.target_outcome = target_outcome;
            }
            None => {}
        }
        memory
    }

    fn record(&mut self, action: u8, outcome: u8) {
        let index = usize::from(action);
        self.observations_by_action[index] =
            self.observations_by_action[index].saturating_add(1);
        if outcome == self.target_outcome {
            self.target_hits_by_action[index] =
                self.target_hits_by_action[index].saturating_add(1);
        }

        let prior = self.causal_prior[index];
        if prior.is_some_and(|expected| expected != outcome) {
            self.causal_conflicts = self.causal_conflicts.saturating_add(1);
            self.causal_prior[index] = Some(outcome);
        }

        // Use the most recently observed outcome as a compact local map.
        self.outcome_by_action[index] = Some(outcome);
    }

    fn observed_action_count(self) -> usize {
        self.observations_by_action.iter().filter(|&&count| count > 0).count()
    }

    fn best_empirical_action(self) -> u8 {
        (0..ACTION_COUNT)
            .max_by(|&left, &right| {
                let left_n = f64::from(self.observations_by_action[left]);
                let right_n = f64::from(self.observations_by_action[right]);
                let left_rate = if left_n == 0.0 {
                    -1.0
                } else {
                    f64::from(self.target_hits_by_action[left]) / left_n
                };
                let right_rate = if right_n == 0.0 {
                    -1.0
                } else {
                    f64::from(self.target_hits_by_action[right]) / right_n
                };
                left_rate.total_cmp(&right_rate).then_with(|| right.cmp(&left))
            })
            .unwrap_or(0) as u8
    }

    fn causal_map(self) -> [Option<u8>; ACTION_COUNT] {
        let mut map = self.outcome_by_action;
        for (index, prior) in self.causal_prior.iter().copied().enumerate() {
            if map[index].is_none() {
                map[index] = prior;
            }
        }
        map
    }
}

fn choose_unexplored_action(
    offered: [u8; 2],
    memory: LearnerMemory,
) -> u8 {
    let left = usize::from(offered[0]);
    let right = usize::from(offered[1]);
    let left_n = memory.observations_by_action[left];
    let right_n = memory.observations_by_action[right];
    match (left_n == 0, right_n == 0) {
        (true, false) => offered[0],
        (false, true) => offered[1],
        (true, true) => offered[0].min(offered[1]),
        (false, false) => {
            let left_rate = f64::from(memory.target_hits_by_action[left])
                / f64::from(left_n);
            let right_rate = f64::from(memory.target_hits_by_action[right])
                / f64::from(right_n);
            if left_rate > right_rate {
                offered[0]
            } else if right_rate > left_rate {
                offered[1]
            } else {
                offered[0].min(offered[1])
            }
        }
    }
}

fn choose_action(
    offered: [u8; 2],
    message: Option<TeachingMessage>,
    memory: LearnerMemory,
) -> u8 {
    match message {
        Some(TeachingMessage::Procedure { recommended_action }) => {
            if offered.contains(&recommended_action) {
                recommended_action
            } else {
                // A rote procedure gives no portable rule for an unseen action.
                offered[0].min(offered[1])
            }
        }
        Some(TeachingMessage::OutcomeGoal { .. }) | None => {
            choose_unexplored_action(offered, memory)
        }
        Some(TeachingMessage::CausalModel { .. }) => {
            let target_actions: Vec<u8> = offered.iter().copied().filter(|&action| {
                memory.causal_prior[usize::from(action)] == Some(memory.target_outcome)
            }).collect();
            if let Some(action) = target_actions.first() {
                *action
            } else {
                choose_unexplored_action(offered, memory)
            }
        }
    }
}

fn encode_message(
    policy: EncodingPolicy,
    memory: LearnerMemory,
    hit_rate: f64,
    config: Config,
) -> TeachingMessage {
    match policy {
        EncodingPolicy::AlwaysProcedure => TeachingMessage::Procedure {
            recommended_action: memory.best_empirical_action(),
        },
        EncodingPolicy::AlwaysOutcomeGoal => TeachingMessage::OutcomeGoal {
            target_outcome: memory.target_outcome,
        },
        EncodingPolicy::AlwaysCausalModel => TeachingMessage::CausalModel {
            outcome_by_action: memory.causal_map(),
            target_outcome: memory.target_outcome,
        },
        EncodingPolicy::Adaptive => {
            if hit_rate >= config.procedure_success_threshold
                && memory.causal_conflicts == 0
            {
                TeachingMessage::Procedure {
                    recommended_action: memory.best_empirical_action(),
                }
            } else if memory.observed_action_count() >= config.causal_min_observed_actions
                && memory.causal_conflicts > 0
            {
                TeachingMessage::CausalModel {
                    outcome_by_action: memory.causal_map(),
                    target_outcome: memory.target_outcome,
                }
            } else {
                TeachingMessage::OutcomeGoal {
                    target_outcome: memory.target_outcome,
                }
            }
        }
    }
}

fn offered_actions(rng: &mut SplitMix64) -> [u8; 2] {
    let first = rng.index(ACTION_COUNT) as u8;
    let mut second = rng.index(ACTION_COUNT - 1) as u8;
    if second >= first {
        second += 1;
    }
    [first, second]
}

fn observe_outcome(
    mapping: EnvironmentMap,
    action: u8,
    fidelity: f64,
    rng: &mut SplitMix64,
) -> u8 {
    if rng.bernoulli(fidelity) {
        mapping.outcome_by_action[usize::from(action)]
    } else {
        let mapped = mapping.outcome_by_action[usize::from(action)];
        let mut alternative = rng.index(OUTCOME_COUNT - 1) as u8;
        if alternative >= mapped {
            alternative += 1;
        }
        alternative
    }
}

/// Run one six-generation iterated-learning chain using a common RNG seed.
///
/// The environment truth is available only to the evaluator. Learners see the
/// previous generation's message and the feedback from actions they chose.
pub fn simulate(
    regime: EnvironmentRegime,
    policy: EncodingPolicy,
    config: Config,
    seed: u64,
) -> Result<TransmissionRun, EnvironmentError> {
    let config = config.validate()?;
    let mut rng = SplitMix64::new(seed);
    let base = shuffled_base_mapping(&mut rng);
    base.validate()?;

    let mut message: Option<TeachingMessage> = None;
    let mut metrics = Vec::with_capacity(config.generations);
    let mut trials = Vec::with_capacity(config.generations * config.trials_per_generation);
    let mut messages = Vec::with_capacity(config.generations);

    for generation in 0..config.generations {
        let environment = mapping_for_generation(base, regime, generation, seed);
        let optimal_action = environment.target_action(config.target_outcome)
            .ok_or(EnvironmentError::NonPermutation)?;
        let incoming_message = message;
        let mut memory = LearnerMemory::new(incoming_message, config.target_outcome);
        let mut target_hits = 0usize;
        let mut optimal_choices = 0usize;
        let mut first_optimal_choice_trial = None;

        for trial in 0..config.trials_per_generation {
            let offered = offered_actions(&mut rng);
            let chosen = choose_action(offered, incoming_message, memory);
            if chosen >= ACTION_COUNT as u8 {
                return Err(EnvironmentError::InvalidAction { action: chosen });
            }
            let action_was_optimal = chosen == optimal_action;
            if action_was_optimal {
                optimal_choices += 1;
                if first_optimal_choice_trial.is_none() {
                    first_optimal_choice_trial = Some(trial);
                }
            }
            let observed = observe_outcome(
                environment,
                chosen,
                config.mapping_fidelity,
                &mut rng,
            );
            let hit_target = observed == config.target_outcome;
            if hit_target {
                target_hits += 1;
            }
            memory.record(chosen, observed);

            trials.push(TrialRecord {
                generation,
                trial,
                offered_actions: offered,
                chosen_action: chosen,
                observed_outcome: observed,
                hit_target,
                action_was_optimal,
            });
        }

        let hit_rate = target_hits as f64 / config.trials_per_generation as f64;
        let outgoing = encode_message(policy, memory, hit_rate, config);
        let distinct_actions = memory.observed_action_count();
        metrics.push(GenerationMetrics {
            generation,
            incoming_message: incoming_message.map(TeachingMessage::kind),
            outgoing_message: outgoing.kind(),
            outgoing_payload_bits: outgoing.payload_bits(),
            trials: config.trials_per_generation,
            target_hits,
            target_hit_rate: hit_rate,
            optimal_action_choices: optimal_choices,
            optimal_action_rate: optimal_choices as f64
                / config.trials_per_generation as f64,
            distinct_actions_observed: distinct_actions,
            causal_conflicts: memory.causal_conflicts,
            first_optimal_choice_trial,
        });
        messages.push(outgoing);
        message = Some(outgoing);
    }

    Ok(TransmissionRun {
        regime,
        encoding_policy: policy,
        metrics,
        trials,
        messages,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_validation_rejects_duplicate_outcomes() {
        assert_eq!(
            EnvironmentMap {
                outcome_by_action: [0, 1, 1, 3],
            }.validate(),
            Err(EnvironmentError::NonPermutation)
        );
    }

    #[test]
    fn reversal_changes_the_optimal_action_for_a_fixed_target() {
        let base = EnvironmentMap {
            outcome_by_action: [0, 1, 2, 3],
        };
        let stable = mapping_for_generation(base, EnvironmentRegime::Stable, 0, 7);
        let reversed = mapping_for_generation(base, EnvironmentRegime::ReverseEveryGeneration, 1, 7);
        assert_ne!(
            stable.target_action(0),
            reversed.target_action(0)
        );
        assert_eq!(
            mapping_for_generation(base, EnvironmentRegime::ReverseEveryGeneration, 2, 7),
            base
        );
    }

    #[test]
    fn fixed_seed_reproduces_all_generation_metrics_and_trials() {
        let left = simulate(
            EnvironmentRegime::ReverseEveryGeneration,
            EncodingPolicy::Adaptive,
            Config::default(),
            1234,
        ).unwrap();
        let right = simulate(
            EnvironmentRegime::ReverseEveryGeneration,
            EncodingPolicy::Adaptive,
            Config::default(),
            1234,
        ).unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn reference_design_uses_six_generations_and_twelve_trials() {
        let run = simulate(
            EnvironmentRegime::Stable,
            EncodingPolicy::Adaptive,
            Config::default(),
            1,
        ).unwrap();
        assert_eq!(run.metrics.len(), GENERATIONS);
        assert!(run.metrics.iter().all(|m| m.trials == TRIALS_PER_GENERATION));
        assert_eq!(run.trials.len(), GENERATIONS * TRIALS_PER_GENERATION);
    }

    #[test]
    fn message_payload_cost_distinguishes_compact_and_structured_knowledge() {
        let procedure = TeachingMessage::Procedure { recommended_action: 2 };
        let goal = TeachingMessage::OutcomeGoal { target_outcome: 0 };
        let causal = TeachingMessage::CausalModel {
            outcome_by_action: [Some(0), Some(1), None, Some(3)],
            target_outcome: 0,
        };
        assert_eq!(procedure.payload_bits(), 2);
        assert_eq!(goal.payload_bits(), 2);
        assert!(causal.payload_bits() > procedure.payload_bits());
    }

    #[test]
    fn adaptive_encoder_uses_compact_procedure_when_it_is_reliable() {
        let memory = LearnerMemory {
            outcome_by_action: [Some(0), Some(1), Some(2), Some(3)],
            observations_by_action: [3, 1, 1, 1],
            target_hits_by_action: [3, 0, 0, 0],
            causal_prior: [None; ACTION_COUNT],
            procedure_action: None,
            target_outcome: 0,
            causal_conflicts: 0,
        };
        let message = encode_message(
            EncodingPolicy::Adaptive,
            memory,
            0.9,
            Config::default(),
        );
        assert_eq!(message.kind(), MessageKind::Procedure);
    }

    #[test]
    fn adaptive_encoder_sends_portable_goal_after_low_performance() {
        let memory = LearnerMemory {
            outcome_by_action: [Some(1), Some(2), None, None],
            observations_by_action: [1, 1, 0, 0],
            target_hits_by_action: [0, 0, 0, 0],
            causal_prior: [None; ACTION_COUNT],
            procedure_action: None,
            target_outcome: 0,
            causal_conflicts: 0,
        };
        let message = encode_message(
            EncodingPolicy::Adaptive,
            memory,
            0.1,
            Config::default(),
        );
        assert_eq!(message.kind(), MessageKind::OutcomeGoal);
    }

    #[test]
    fn causal_message_is_corrected_when_the_environment_changes() {
        let config = Config {
            generations: 2,
            trials_per_generation: 12,
            mapping_fidelity: 1.0,
            ..Config::default()
        };
        let run = simulate(
            EnvironmentRegime::ReverseEveryGeneration,
            EncodingPolicy::AlwaysCausalModel,
            config,
            77,
        ).unwrap();
        assert!(run.metrics[1].causal_conflicts > 0);
    }

    #[test]
    fn stable_procedure_is_retained_as_the_compact_adaptive_message() {
        let config = Config {
            mapping_fidelity: 1.0,
            procedure_success_threshold: 0.0,
            ..Config::default()
        };
        let run = simulate(
            EnvironmentRegime::Stable,
            EncodingPolicy::Adaptive,
            config,
            19,
        ).unwrap();
        assert_eq!(run.messages[0].kind(), MessageKind::Procedure);
    }
}
