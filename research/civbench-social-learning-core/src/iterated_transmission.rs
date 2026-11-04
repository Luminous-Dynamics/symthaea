// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! Synthetic multi-generation content-drift and cultural-attractor reference.
//!
//! The four normalized attributes are illustrative proxies for dimensions
//! studied in Perez et al. (ICLR 2025): toxicity, positivity, difficulty and
//! length. This is a controlled metric fixture, not an LLM or text-semantics
//! implementation and not a reproduction of that paper's results.

use std::fmt;

pub const ATTRIBUTE_NAMES: [&str; 4] = [
    "toxicity",
    "positivity",
    "difficulty",
    "length",
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CultureSignal {
    /// Normalized, bounded attributes in the order defined by ATTRIBUTE_NAMES.
    pub attributes: [f64; 4],
}

impl CultureSignal {
    pub fn new(attributes: [f64; 4]) -> Result<Self, TransmissionError> {
        for (index, value) in attributes.iter().copied().enumerate() {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(TransmissionError::InvalidAttribute { index, value_bits: value.to_bits() });
            }
        }
        Ok(Self { attributes })
    }

    pub fn euclidean_distance(self, other: Self) -> f64 {
        self.attributes.iter().zip(other.attributes)
            .map(|(left, right)| (left - right).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransmissionChannelConfig {
    /// Hypothetical attractor target for each attribute; not an empirical LLM mean.
    pub attractor: [f64; 4],
    /// Per-attribute pull strength before the open-endedness multiplier.
    pub attribute_bias: [f64; 4],
    /// Per-attribute standard deviation of each transmission step's noise.
    pub noise_std: [f64; 4],
    /// 0 means fully constrained/identity-like; 1 applies full configured bias.
    pub open_endedness: f64,
}

impl Default for TransmissionChannelConfig {
    fn default() -> Self {
        Self {
            attractor: [0.2, 0.8, 0.3, 0.5],
            attribute_bias: [0.4, 0.35, 0.25, 0.15],
            noise_std: [0.01, 0.01, 0.01, 0.01],
            open_endedness: 0.5,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransmissionError {
    InvalidAttribute { index: usize, value_bits: u64 },
    InvalidAttractor { index: usize },
    InvalidBias { index: usize },
    InvalidNoise { index: usize },
    InvalidOpenEndedness,
    ZeroSteps,
    EmptyChainSet,
    InconsistentChainLengths,
    ZeroPopulationSize,
}

impl fmt::Display for TransmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAttribute { index, value_bits } => {
                write!(f, "attribute {index} is invalid (f64 bits {value_bits:#x}); expected finite [0,1]")
            }
            Self::InvalidAttractor { index } => write!(f, "attractor[{index}] must be finite and in [0,1]"),
            Self::InvalidBias { index } => write!(f, "attribute_bias[{index}] must be finite and in [0,1]"),
            Self::InvalidNoise { index } => write!(f, "noise_std[{index}] must be finite and non-negative"),
            Self::InvalidOpenEndedness => write!(f, "open_endedness must be finite and in [0,1]"),
            Self::ZeroSteps => write!(f, "transmission chain must have at least one step"),
            Self::EmptyChainSet => write!(f, "population metrics require at least one chain"),
            Self::InconsistentChainLengths => write!(f, "all transmission chains must have equal step counts"),
            Self::ZeroPopulationSize => write!(f, "branching population requires at least one initial signal"),
        }
    }
}
impl std::error::Error for TransmissionError {}

impl TransmissionChannelConfig {
    pub fn validate(self) -> Result<(), TransmissionError> {
        for index in 0..4 {
            if !self.attractor[index].is_finite() || !(0.0..=1.0).contains(&self.attractor[index]) {
                return Err(TransmissionError::InvalidAttractor { index });
            }
            if !self.attribute_bias[index].is_finite() || !(0.0..=1.0).contains(&self.attribute_bias[index]) {
                return Err(TransmissionError::InvalidBias { index });
            }
            if !self.noise_std[index].is_finite() || self.noise_std[index] < 0.0 {
                return Err(TransmissionError::InvalidNoise { index });
            }
        }
        if !self.open_endedness.is_finite() || !(0.0..=1.0).contains(&self.open_endedness) {
            return Err(TransmissionError::InvalidOpenEndedness);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransmissionChain {
    pub seed: u64,
    /// Includes the initial artifact at index zero.
    pub generations: Vec<CultureSignal>,
}

impl TransmissionChain {
    pub fn initial(&self) -> CultureSignal {
        self.generations[0]
    }

    pub fn final_signal(&self) -> CultureSignal {
        *self.generations.last().expect("validated chain contains initial state")
    }

    pub fn steps(&self) -> usize {
        self.generations.len().saturating_sub(1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChainPopulationMetrics {
    pub chain_count: usize,
    pub steps: usize,
    pub mean_signed_drift: [f64; 4],
    pub mean_absolute_drift: [f64; 4],
    pub mean_final_attractor_distance: [f64; 4],
    pub mean_initial_pairwise_distance: f64,
    pub mean_final_pairwise_distance: f64,
    /// 1 - final pairwise distance / initial pairwise distance.
    /// Positive values indicate convergence; negative values indicate divergence.
    pub attractor_convergence_index: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
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

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(f64::MIN_POSITIVE);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next_u64() as usize) % upper
    }
}

/// Apply one transformation step. Each attribute has its own bias/noise so
/// that aggregate similarity cannot hide attribute-specific drift.
fn transmit_once(
    input: CultureSignal,
    config: TransmissionChannelConfig,
    rng: &mut impl TransmissionRandom,
) -> CultureSignal {
    let mut output = [0.0; 4];
    for index in 0..4 {
        let pull = config.attribute_bias[index] * config.open_endedness;
        let drift = pull * (config.attractor[index] - input.attributes[index]);
        let noise = config.noise_std[index] * rng.normal();
        output[index] = (input.attributes[index] + drift + noise).clamp(0.0, 1.0);
    }
    CultureSignal { attributes: output }
}

/// Random interface kept local and deterministic for this benchmark.
trait TransmissionRandom {
    fn normal(&mut self) -> f64;
}

impl TransmissionRandom for DeterministicRng {
    fn normal(&mut self) -> f64 {
        DeterministicRng::normal(self)
    }
}

/// Execute a linear chain of repeated transformations.
pub fn simulate_chain(
    initial: CultureSignal,
    config: TransmissionChannelConfig,
    steps: usize,
    seed: u64,
) -> Result<TransmissionChain, TransmissionError> {
    config.validate()?;
    if steps == 0 {
        return Err(TransmissionError::ZeroSteps);
    }
    for (index, value) in initial.attributes.iter().copied().enumerate() {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(TransmissionError::InvalidAttribute {
                index,
                value_bits: value.to_bits(),
            });
        }
    }
    let mut rng = DeterministicRng::new(seed);
    let mut generations = Vec::with_capacity(steps + 1);
    generations.push(initial);
    let mut current = initial;
    for _ in 0..steps {
        current = transmit_once(current, config, &mut rng);
        generations.push(current);
    }
    Ok(TransmissionChain { seed, generations })
}

/// Execute one chain per supplied initial signal using a deterministic seed
/// schedule. These are independent chains, not a branching transmission graph.
pub fn simulate_parallel_chains(
    initial_signals: &[CultureSignal],
    config: TransmissionChannelConfig,
    steps: usize,
    base_seed: u64,
) -> Result<Vec<TransmissionChain>, TransmissionError> {
    if initial_signals.is_empty() {
        return Err(TransmissionError::EmptyChainSet);
    }
    initial_signals.iter().enumerate()
        .map(|(index, initial)| {
            simulate_chain(*initial, config, steps, base_seed.wrapping_add(index as u64))
        })
        .collect()
}

fn mean_pairwise_distance(signals: &[CultureSignal]) -> f64 {
    if signals.len() < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for left in 0..signals.len() {
        for right in (left + 1)..signals.len() {
            sum += signals[left].euclidean_distance(signals[right]);
            count += 1;
        }
    }
    sum / count as f64
}

pub fn population_metrics(
    chains: &[TransmissionChain],
    config: TransmissionChannelConfig,
) -> Result<ChainPopulationMetrics, TransmissionError> {
    if chains.is_empty() {
        return Err(TransmissionError::EmptyChainSet);
    }
    let steps = chains[0].steps();
    if chains.iter().any(|chain| chain.steps() != steps) {
        return Err(TransmissionError::InconsistentChainLengths);
    }

    let mut mean_signed_drift = [0.0; 4];
    let mut mean_absolute_drift = [0.0; 4];
    let mut mean_final_attractor_distance = [0.0; 4];
    let mut initial = Vec::with_capacity(chains.len());
    let mut final_signals = Vec::with_capacity(chains.len());

    for chain in chains {
        let start = chain.initial();
        let end = chain.final_signal();
        initial.push(start);
        final_signals.push(end);
        for index in 0..4 {
            let delta = end.attributes[index] - start.attributes[index];
            mean_signed_drift[index] += delta;
            mean_absolute_drift[index] += delta.abs();
            mean_final_attractor_distance[index] += (end.attributes[index] - config.attractor[index]).abs();
        }
    }

    let n = chains.len() as f64;
    for index in 0..4 {
        mean_signed_drift[index] /= n;
        mean_absolute_drift[index] /= n;
        mean_final_attractor_distance[index] /= n;
    }

    let mean_initial_pairwise_distance = mean_pairwise_distance(&initial);
    let mean_final_pairwise_distance = mean_pairwise_distance(&final_signals);
    let attractor_convergence_index = if mean_initial_pairwise_distance > f64::EPSILON {
        Some(1.0 - mean_final_pairwise_distance / mean_initial_pairwise_distance)
    } else {
        None
    };

    Ok(ChainPopulationMetrics {
        chain_count: chains.len(),
        steps,
        mean_signed_drift,
        mean_absolute_drift,
        mean_final_attractor_distance,
        mean_initial_pairwise_distance,
        mean_final_pairwise_distance,
        attractor_convergence_index,
    })
}

/// One descendant artifact in a branching cultural population.
#[derive(Debug, Clone, PartialEq)]
pub struct BranchingAgent {
    pub signal: CultureSignal,
    /// Root-source identity; preserved across all descendants.
    pub lineage_id: usize,
    /// Initial signal for this root lineage, used for cumulative drift.
    pub root_signal: CultureSignal,
    /// Parent index in the previous generation; None only for generation zero.
    pub parent_index: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BranchingTransmissionRun {
    pub generations: Vec<Vec<BranchingAgent>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BranchingGenerationMetrics {
    pub generation: usize,
    pub population_size: usize,
    pub unique_lineages: usize,
    pub lineage_hhi: f64,
    pub effective_lineage_diversity: f64,
    pub mean_pairwise_content_distance: f64,
    pub mean_distance_from_root_signal: f64,
    pub extinct_root_lineages: usize,
}

/// Evolve a fixed-size population. Every descendant samples one parent
/// uniformly from the preceding generation, then passes through the specified
/// content-transmission channel. This is a branching population, not a single
/// linear chain; lineage identity is retained for extinction/diversity metrics.
pub fn simulate_branching_population(
    initial_signals: &[CultureSignal],
    config: TransmissionChannelConfig,
    generations: usize,
    seed: u64,
) -> Result<BranchingTransmissionRun, TransmissionError> {
    config.validate()?;
    if initial_signals.is_empty() {
        return Err(TransmissionError::ZeroPopulationSize);
    }
    if generations == 0 {
        return Err(TransmissionError::ZeroSteps);
    }
    for signal in initial_signals {
        for (index, value) in signal.attributes.iter().copied().enumerate() {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(TransmissionError::InvalidAttribute {
                    index,
                    value_bits: value.to_bits(),
                });
            }
        }
    }

    let mut rng = DeterministicRng::new(seed);
    let mut population: Vec<BranchingAgent> = initial_signals.iter().copied().enumerate()
        .map(|(lineage_id, signal)| BranchingAgent {
            signal,
            lineage_id,
            root_signal: signal,
            parent_index: None,
        })
        .collect();
    let mut history = Vec::with_capacity(generations + 1);
    history.push(population.clone());

    for _ in 0..generations {
        let mut next = Vec::with_capacity(initial_signals.len());
        for _ in 0..initial_signals.len() {
            let parent_index = rng.index(population.len());
            let parent = &population[parent_index];
            let signal = transmit_once(parent.signal, config, &mut rng);
            next.push(BranchingAgent {
                signal,
                lineage_id: parent.lineage_id,
                root_signal: parent.root_signal,
                parent_index: Some(parent_index),
            });
        }
        population = next;
        history.push(population.clone());
    }

    Ok(BranchingTransmissionRun { generations: history })
}

pub fn branching_generation_metrics(
    run: &BranchingTransmissionRun,
) -> Result<Vec<BranchingGenerationMetrics>, TransmissionError> {
    if run.generations.is_empty() || run.generations.iter().any(Vec::is_empty) {
        return Err(TransmissionError::ZeroPopulationSize);
    }
    let total_root_lineages = run.generations[0].iter()
        .map(|agent| agent.lineage_id)
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let mut metrics = Vec::with_capacity(run.generations.len());

    for (generation, agents) in run.generations.iter().enumerate() {
        let mut lineage_counts = std::collections::BTreeMap::<usize, usize>::new();
        let mut root_distance = 0.0;
        for agent in agents {
            *lineage_counts.entry(agent.lineage_id).or_default() += 1;
            root_distance += agent.signal.euclidean_distance(agent.root_signal);
        }
        let population_size = agents.len();
        let lineage_hhi = lineage_counts.values()
            .map(|count| {
                let share = *count as f64 / population_size as f64;
                share * share
            })
            .sum::<f64>();
        let mut pairwise_distance = 0.0;
        let mut pair_count = 0usize;
        for left in 0..agents.len() {
            for right in (left + 1)..agents.len() {
                pairwise_distance += agents[left].signal.euclidean_distance(agents[right].signal);
                pair_count += 1;
            }
        }

        metrics.push(BranchingGenerationMetrics {
            generation,
            population_size,
            unique_lineages: lineage_counts.len(),
            lineage_hhi,
            effective_lineage_diversity: if lineage_hhi > 0.0 { 1.0 / lineage_hhi } else { 0.0 },
            mean_pairwise_content_distance: if pair_count == 0 {
                0.0
            } else {
                pairwise_distance / pair_count as f64
            },
            mean_distance_from_root_signal: root_distance / population_size as f64,
            extinct_root_lineages: total_root_lineages.saturating_sub(lineage_counts.len()),
        });
    }
    Ok(metrics)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal(values: [f64; 4]) -> CultureSignal {
        CultureSignal::new(values).unwrap()
    }

    fn zero_noise_config(open_endedness: f64, bias: [f64; 4]) -> TransmissionChannelConfig {
        TransmissionChannelConfig {
            attractor: [0.5, 0.5, 0.5, 0.5],
            attribute_bias: bias,
            noise_std: [0.0; 4],
            open_endedness,
        }
    }

    #[test]
    fn zero_bias_zero_noise_is_an_identity_channel() {
        let initial = signal([0.1, 0.2, 0.8, 0.9]);
        let config = zero_noise_config(1.0, [0.0; 4]);
        let chain = simulate_chain(initial, config, 12, 1).unwrap();
        assert!(chain.generations.iter().all(|value| *value == initial));
    }

    #[test]
    fn strong_bias_moves_each_attribute_to_its_declared_attractor() {
        let initial = signal([0.0, 0.0, 0.0, 0.0]);
        let config = zero_noise_config(1.0, [1.0; 4]);
        let chain = simulate_chain(initial, config, 1, 2).unwrap();
        assert_eq!(chain.final_signal(), signal([0.5, 0.5, 0.5, 0.5]));
    }

    #[test]
    fn open_ended_transmission_increases_configured_drift() {
        let initial = signal([0.0, 0.0, 0.0, 0.0]);
        let constrained = simulate_chain(
            initial,
            zero_noise_config(0.1, [0.8; 4]),
            1,
            3,
        ).unwrap();
        let open = simulate_chain(
            initial,
            zero_noise_config(1.0, [0.8; 4]),
            1,
            3,
        ).unwrap();
        assert!(open.final_signal().attributes[0] > constrained.final_signal().attributes[0]);
    }

    #[test]
    fn attributes_have_independent_drift_sensitivity() {
        let initial = signal([0.0, 0.0, 0.0, 0.0]);
        let config = zero_noise_config(1.0, [1.0, 0.0, 0.5, 0.0]);
        let chain = simulate_chain(initial, config, 1, 4).unwrap();
        let final_values = chain.final_signal().attributes;
        assert_eq!(final_values[0], 0.5);
        assert_eq!(final_values[1], 0.0);
        assert_eq!(final_values[2], 0.25);
        assert_eq!(final_values[3], 0.0);
    }

    #[test]
    fn distinct_initial_signals_can_converge_toward_a_shared_attractor() {
        let initials = [
            signal([0.0, 0.0, 0.0, 0.0]),
            signal([1.0, 1.0, 1.0, 1.0]),
        ];
        let config = zero_noise_config(1.0, [0.5; 4]);
        let chains = simulate_parallel_chains(&initials, config, 1, 5).unwrap();
        let metrics = population_metrics(&chains, config).unwrap();
        assert_eq!(metrics.attractor_convergence_index, Some(0.5));
    }

    #[test]
    fn identical_seed_and_configuration_reproduce_chain() {
        let initial = signal([0.3, 0.4, 0.5, 0.6]);
        let config = TransmissionChannelConfig::default();
        assert_eq!(
            simulate_chain(initial, config, 16, 77).unwrap(),
            simulate_chain(initial, config, 16, 77).unwrap()
        );
    }

    #[test]
    fn branching_population_tracks_parent_and_root_lineage() {
        let initial = [
            signal([0.1, 0.2, 0.3, 0.4]),
            signal([0.8, 0.7, 0.6, 0.5]),
            signal([0.2, 0.8, 0.4, 0.6]),
            signal([0.9, 0.1, 0.7, 0.3]),
        ];
        let config = zero_noise_config(0.0, [0.0; 4]);
        let run = simulate_branching_population(&initial, config, 3, 123).unwrap();
        assert_eq!(run.generations.len(), 4);
        for agent in &run.generations[0] {
            assert_eq!(agent.parent_index, None);
            assert_eq!(agent.signal, agent.root_signal);
            assert_eq!(agent.signal, initial[agent.lineage_id]);
        }
        for generation in run.generations.iter().skip(1) {
            for agent in generation {
                assert!(agent.parent_index.is_some());
                assert_eq!(agent.signal, agent.root_signal);
            }
        }
    }

    #[test]
    fn branching_population_reports_lineage_extinction_and_diversity() {
        let initial = [
            signal([0.1, 0.2, 0.3, 0.4]),
            signal([0.8, 0.7, 0.6, 0.5]),
            signal([0.2, 0.8, 0.4, 0.6]),
            signal([0.9, 0.1, 0.7, 0.3]),
        ];
        let config = zero_noise_config(0.0, [0.0; 4]);
        let run = simulate_branching_population(&initial, config, 4, 77).unwrap();
        let metrics = branching_generation_metrics(&run).unwrap();
        assert_eq!(metrics[0].unique_lineages, 4);
        assert_eq!(metrics[0].extinct_root_lineages, 0);
        assert_eq!(metrics[0].effective_lineage_diversity, 4.0);
        assert!(metrics[4].unique_lineages <= 4);
        assert_eq!(
            metrics[4].unique_lineages + metrics[4].extinct_root_lineages,
            4
        );
    }

    #[test]
    fn branching_channel_and_metrics_are_reproducible_for_fixed_seed() {
        let initial = [
            signal([0.1, 0.2, 0.3, 0.4]),
            signal([0.8, 0.7, 0.6, 0.5]),
            signal([0.2, 0.8, 0.4, 0.6]),
            signal([0.9, 0.1, 0.7, 0.3]),
        ];
        let config = TransmissionChannelConfig::default();
        let a = simulate_branching_population(&initial, config, 6, 88).unwrap();
        let b = simulate_branching_population(&initial, config, 6, 88).unwrap();
        assert_eq!(a, b);
        assert_eq!(branching_generation_metrics(&a), branching_generation_metrics(&b));
    }

    #[test]
    fn invalid_values_and_zero_steps_are_rejected() {
        assert!(matches!(
            CultureSignal::new([0.0, 1.1, 0.5, 0.5]),
            Err(TransmissionError::InvalidAttribute { index: 1, .. })
        ));
        assert_eq!(
            simulate_chain(
                signal([0.0; 4]),
                TransmissionChannelConfig::default(),
                0,
                1
            ),
            Err(TransmissionError::ZeroSteps)
        );
    }
}
