// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Query-budgeted adversarial search against discrete model-confidence surfaces.
//!
//! This is a red-team benchmark primitive, not a life detector. It searches for candidates
//! that a supplied model scores highly; it does not decide whether those candidates are
//! replicators, alive, adaptive, intelligent, or conscious. Ground truth must be evaluated
//! separately after the search has ended.
//!
//! The initial strategy reproduces the *shape* of the greedy single-site mutation attack in
//! Gupta & Adami (2026), arXiv:2604.11915. The module now also exposes random-walk,
//! Metropolis-style, and population-search controls. A caller supplies the model and a
//! separate, independently held-out ground-truth evaluator; no model score is treated as truth.
//!
//! The query budget in this API counts every model evaluation, including the initial
//! candidate evaluation. This convention is explicit so benchmark adapters can translate
//! it to a paper's query-count convention without silently shifting the x-axis.

use std::error::Error;
use std::fmt;

/// Search policy used to explore the model's confidence surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchStrategy {
    /// Accept a single-site substitution only when model confidence strictly increases.
    GreedyAscent,
    /// Accept every proposed substitution while separately retaining the best-scoring
    /// candidate encountered. This is the no-selection control used to measure search
    /// pressure, not a defense.
    RandomWalk,
}

/// Configuration for one independently seeded attack run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttackConfig {
    /// Total model calls, including the initial candidate score.
    pub max_model_queries: usize,
    /// Deterministic PRNG seed. Zero is mapped to a documented non-zero state.
    pub seed: u64,
    pub strategy: SearchStrategy,
}

impl Default for AttackConfig {
    fn default() -> Self {
        Self {
            max_model_queries: 300,
            seed: 1,
            strategy: SearchStrategy::GreedyAscent,
        }
    }
}

/// Stochastic single-candidate search using a fixed-temperature Metropolis acceptance rule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StochasticAttackConfig {
    /// Total model calls, including the initial candidate score.
    pub max_model_queries: usize,
    pub seed: u64,
    /// Positive temperature. Larger values accept more downhill proposals.
    pub temperature: f64,
}

/// Population-selection family used by the paper-inspired evolutionary controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PopulationSelection {
    /// Select parents by k-way confidence tournaments.
    Tournament,
    /// Select parents uniformly from the top fraction of the ranked population.
    Truncation,
}

/// Fixed population and evaluation budget for one evolutionary attack run.
///
/// Supply the complete, ordered initial population as input. The budget counts every
/// model evaluation, including evaluation of the initial population. A 30-member
/// population and 300 total model evaluations therefore yields ten evaluated
/// populations (the initial population plus nine offspring generations).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PopulationAttackConfig {
    pub max_model_queries: usize,
    pub seed: u64,
    pub population_size: usize,
    /// Number of competitors sampled with replacement for each tournament parent.
    pub tournament_size: usize,
    /// Parent pool fraction for truncation selection, in (0, 1].
    pub truncation_fraction: f64,
}

/// Inference-only interface. Implementations must return a probability-like score in [0, 1].
///
/// This interface intentionally exposes no ground-truth method. The attack harness cannot
/// ask the model whether its own candidate is truly viable.
pub trait ConfidenceModel {
    fn confidence(&self, candidate: &[u8]) -> Result<f64, String>;
}

/// One completed model query in an attack trajectory.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryTrace {
    /// One-based model-call count, including the initial candidate evaluation.
    pub query_index: usize,
    pub candidate: Vec<u8>,
    pub confidence: f64,
    /// Whether the proposed candidate became the current search state.
    pub accepted_as_current: bool,
    /// Whether this candidate improved the best confidence seen so far.
    pub new_best: bool,
}

/// Result of one red-team search.
///
/// There is deliberately no `is_alive`, `is_replicator`, or `is_intelligent` field.
/// The caller must pass `best_candidate` to a separate, independently controlled oracle.
#[derive(Debug, Clone, PartialEq)]
pub struct AttackResult {
    pub strategy: SearchStrategy,
    pub seed: u64,
    pub model_queries: usize,
    pub initial_candidate: Vec<u8>,
    pub initial_confidence: f64,
    pub final_candidate: Vec<u8>,
    pub final_confidence: f64,
    pub best_candidate: Vec<u8>,
    pub best_confidence: f64,
    pub trace: Vec<QueryTrace>,
}

/// Result of a stochastic attack. Functional ground truth remains an external oracle.
#[derive(Debug, Clone, PartialEq)]
pub struct StochasticAttackResult {
    pub seed: u64,
    pub temperature: f64,
    pub model_queries: usize,
    pub initial_candidate: Vec<u8>,
    pub initial_confidence: f64,
    pub final_candidate: Vec<u8>,
    pub final_confidence: f64,
    pub best_candidate: Vec<u8>,
    pub best_confidence: f64,
    pub trace: Vec<QueryTrace>,
}

/// Trace row for one evaluated population member.
#[derive(Debug, Clone, PartialEq)]
pub struct PopulationQueryTrace {
    /// One-based model-call count over the entire population run.
    pub query_index: usize,
    /// Zero is the supplied initial population; later values are offspring generations.
    pub generation: usize,
    pub candidate: Vec<u8>,
    pub confidence: f64,
}

/// Result of a paper-inspired population search. This contains model scores only.
#[derive(Debug, Clone, PartialEq)]
pub struct PopulationAttackResult {
    pub selection: PopulationSelection,
    pub seed: u64,
    pub model_queries: usize,
    pub generations_evaluated: usize,
    pub best_candidate: Vec<u8>,
    pub best_confidence: f64,
    pub final_population: Vec<Vec<u8>>,
    pub trace: Vec<PopulationQueryTrace>,
}

/// Invalid search configuration, model response, or candidate.
#[derive(Debug, Clone, PartialEq)]
pub enum AttackError {
    EmptyCandidate,
    AlphabetSizeOutOfRange { alphabet_size: u16 },
    InitialSymbolOutOfRange { position: usize, symbol: u8, alphabet_size: u16 },
    ZeroQueryBudget,
    InvalidTemperature(f64),
    InvalidPopulationSize(usize),
    QueryBudgetBelowPopulation { budget: usize, population_size: usize },
    PopulationSizeMismatch { expected: usize, actual: usize },
    CandidateLengthMismatch { expected: usize, actual: usize },
    InvalidTournamentSize(usize),
    InvalidTruncationFraction(f64),
    ModelFailure(String),
    InvalidConfidence(f64),
}

impl fmt::Display for AttackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCandidate => write!(f, "candidate must contain at least one symbol"),
            Self::AlphabetSizeOutOfRange { alphabet_size } => write!(
                f,
                "alphabet size must be in 2..=256, got {alphabet_size}"
            ),
            Self::InitialSymbolOutOfRange {
                position,
                symbol,
                alphabet_size,
            } => write!(
                f,
                "initial symbol {symbol} at position {position} is outside alphabet size {alphabet_size}"
            ),
            Self::ZeroQueryBudget => write!(f, "model query budget must be at least one"),
            Self::InvalidTemperature(value) => {
                write!(f, "temperature must be finite and > 0, got {value}")
            }
            Self::InvalidPopulationSize(value) => {
                write!(f, "population size must be at least 2, got {value}")
            }
            Self::QueryBudgetBelowPopulation { budget, population_size } => write!(
                f,
                "model query budget {budget} is smaller than initial population size {population_size}"
            ),
            Self::PopulationSizeMismatch { expected, actual } => write!(
                f,
                "initial population length mismatch: expected {expected}, got {actual}"
            ),
            Self::CandidateLengthMismatch { expected, actual } => write!(
                f,
                "candidate length mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidTournamentSize(value) => {
                write!(f, "tournament size must be at least 2, got {value}")
            }
            Self::InvalidTruncationFraction(value) => write!(
                f,
                "truncation fraction must be finite and in (0, 1], got {value}"
            ),
            Self::ModelFailure(message) => write!(f, "confidence model failed: {message}"),
            Self::InvalidConfidence(value) => write!(
                f,
                "confidence must be finite and in [0, 1], got {value}"
            ),
        }
    }
}

impl Error for AttackError {}

/// Run a deterministic, query-budgeted single-site confidence-surface attack.
///
/// `initial_candidate` is a vector of symbol indices in `0..alphabet_size`. Each proposal
/// changes exactly one position to a different symbol. The model is queried once per proposal.
/// The initial score consumes one query from the configured budget.
///
/// Greedy ascent and random walk share this single-candidate interface. Stochastic
/// acceptance and population policies have separate config/result types so their temperature,
/// population size, parent-selection rule, and query accounting remain explicit.
pub fn run_confidence_attack<M: ConfidenceModel>(
    model: &M,
    initial_candidate: &[u8],
    alphabet_size: u16,
    config: AttackConfig,
) -> Result<AttackResult, AttackError> {
    if initial_candidate.is_empty() {
        return Err(AttackError::EmptyCandidate);
    }
    if !(2..=256).contains(&alphabet_size) {
        return Err(AttackError::AlphabetSizeOutOfRange { alphabet_size });
    }
    if config.max_model_queries == 0 {
        return Err(AttackError::ZeroQueryBudget);
    }
    for (position, &symbol) in initial_candidate.iter().enumerate() {
        if u16::from(symbol) >= alphabet_size {
            return Err(AttackError::InitialSymbolOutOfRange {
                position,
                symbol,
                alphabet_size,
            });
        }
    }

    let mut rng = XorShift64::new(config.seed);
    let mut current_candidate = initial_candidate.to_vec();
    let initial_confidence = evaluate(model, &current_candidate)?;
    let mut current_confidence = initial_confidence;
    let mut best_candidate = current_candidate.clone();
    let mut best_confidence = initial_confidence;
    let mut trace = Vec::with_capacity(config.max_model_queries);
    trace.push(QueryTrace {
        query_index: 1,
        candidate: current_candidate.clone(),
        confidence: initial_confidence,
        accepted_as_current: true,
        new_best: true,
    });

    for query_index in 2..=config.max_model_queries {
        let mut proposal = current_candidate.clone();
        mutate_one_site(&mut proposal, alphabet_size, &mut rng);
        let proposal_confidence = evaluate(model, &proposal)?;

        let accepted_as_current = match config.strategy {
            SearchStrategy::GreedyAscent => proposal_confidence > current_confidence,
            SearchStrategy::RandomWalk => true,
        };
        if accepted_as_current {
            current_candidate = proposal.clone();
            current_confidence = proposal_confidence;
        }

        let new_best = proposal_confidence > best_confidence;
        if new_best {
            best_candidate = proposal.clone();
            best_confidence = proposal_confidence;
        }

        trace.push(QueryTrace {
            query_index,
            candidate: proposal,
            confidence: proposal_confidence,
            accepted_as_current,
            new_best,
        });
    }

    Ok(AttackResult {
        strategy: config.strategy,
        seed: config.seed,
        model_queries: trace.len(),
        initial_candidate: initial_candidate.to_vec(),
        initial_confidence,
        final_candidate: current_candidate,
        final_confidence: current_confidence,
        best_candidate,
        best_confidence,
        trace,
    })
}

/// Run fixed-temperature Metropolis search over one discrete candidate.
///
/// A downhill proposal is accepted with probability `exp((proposal-current)/temperature`.
/// The attack records the best-scoring candidate separately from its terminal state.
/// This is a paper-inspired implementation; exact source-code parity with the reference
/// Avida experiment must be established before reporting a reproduction.
pub fn run_stochastic_attack<M: ConfidenceModel>(
    model: &M,
    initial_candidate: &[u8],
    alphabet_size: u16,
    config: StochasticAttackConfig,
) -> Result<StochasticAttackResult, AttackError> {
    validate_candidate(initial_candidate, alphabet_size)?;
    if config.max_model_queries == 0 {
        return Err(AttackError::ZeroQueryBudget);
    }
    if !config.temperature.is_finite() || config.temperature <= 0.0 {
        return Err(AttackError::InvalidTemperature(config.temperature));
    }

    let mut rng = XorShift64::new(config.seed);
    let mut current_candidate = initial_candidate.to_vec();
    let initial_confidence = evaluate(model, &current_candidate)?;
    let mut current_confidence = initial_confidence;
    let mut best_candidate = current_candidate.clone();
    let mut best_confidence = initial_confidence;
    let mut trace = vec![QueryTrace {
        query_index: 1,
        candidate: current_candidate.clone(),
        confidence: initial_confidence,
        accepted_as_current: true,
        new_best: true,
    }];

    for query_index in 2..=config.max_model_queries {
        let mut proposal = current_candidate.clone();
        mutate_one_site(&mut proposal, alphabet_size, &mut rng);
        let proposal_confidence = evaluate(model, &proposal)?;
        let delta = proposal_confidence - current_confidence;
        let acceptance_probability = if delta >= 0.0 {
            1.0
        } else {
            (delta / config.temperature).exp()
        };
        let accepted_as_current = rng.next_unit() < acceptance_probability;
        if accepted_as_current {
            current_candidate = proposal.clone();
            current_confidence = proposal_confidence;
        }

        let new_best = proposal_confidence > best_confidence;
        if new_best {
            best_candidate = proposal.clone();
            best_confidence = proposal_confidence;
        }
        trace.push(QueryTrace {
            query_index,
            candidate: proposal,
            confidence: proposal_confidence,
            accepted_as_current,
            new_best,
        });
    }

    Ok(StochasticAttackResult {
        seed: config.seed,
        temperature: config.temperature,
        model_queries: trace.len(),
        initial_candidate: initial_candidate.to_vec(),
        initial_confidence,
        final_candidate: current_candidate,
        final_confidence: current_confidence,
        best_candidate,
        best_confidence,
        trace,
    })
}

/// Run a paper-inspired tournament or truncation search under one global query budget.
///
/// The caller provides an ordered initial population. Every initial member is evaluated
/// once; each subsequent generation proposes up to `population_size` mutated offspring.
/// Parent selection is either k-way tournament or uniform sampling from the top
/// `truncation_fraction` of the current ranked population. Survivors are the top-scoring
/// `population_size` members of parents plus offspring (elitist truncation).
///
/// The query budget counts all initial and offspring evaluations. With population_size=30
/// and max_model_queries=300, the result contains 30 initial evaluations plus nine
/// generations of 30 offspring evaluations. This accounting is explicit and must be
/// reconciled with the exact reference implementation before claiming source reproduction.
pub fn run_population_attack<M: ConfidenceModel>(
    model: &M,
    initial_population: &[Vec<u8>],
    alphabet_size: u16,
    selection: PopulationSelection,
    config: PopulationAttackConfig,
) -> Result<PopulationAttackResult, AttackError> {
    if config.max_model_queries == 0 {
        return Err(AttackError::ZeroQueryBudget);
    }
    if config.population_size < 2 {
        return Err(AttackError::InvalidPopulationSize(config.population_size));
    }
    if initial_population.len() != config.population_size {
        return Err(AttackError::PopulationSizeMismatch {
            expected: config.population_size,
            actual: initial_population.len(),
        });
    }
    if config.tournament_size < 2 {
        return Err(AttackError::InvalidTournamentSize(config.tournament_size));
    }
    if !config.truncation_fraction.is_finite()
        || config.truncation_fraction <= 0.0
        || config.truncation_fraction > 1.0
    {
        return Err(AttackError::InvalidTruncationFraction(
            config.truncation_fraction,
        ));
    }
    if config.max_model_queries < config.population_size {
        return Err(AttackError::QueryBudgetBelowPopulation {
            budget: config.max_model_queries,
            population_size: config.population_size,
        });
    }

    let candidate_len = initial_population
        .first()
        .ok_or(AttackError::InvalidPopulationSize(config.population_size))?
        .len();
    if candidate_len == 0 {
        return Err(AttackError::EmptyCandidate);
    }
    for candidate in initial_population {
        validate_candidate(candidate, alphabet_size)?;
        if candidate.len() != candidate_len {
            return Err(AttackError::CandidateLengthMismatch {
                expected: candidate_len,
                actual: candidate.len(),
            });
        }
    }

    let mut rng = XorShift64::new(config.seed);
    let mut population = Vec::with_capacity(config.population_size);
    let mut trace = Vec::with_capacity(config.max_model_queries);
    let mut best_candidate = Vec::new();
    let mut best_confidence = f64::NEG_INFINITY;
    let mut query_index = 0usize;

    for candidate in initial_population {
        let confidence = evaluate(model, candidate)?;
        query_index += 1;
        trace.push(PopulationQueryTrace {
            query_index,
            generation: 0,
            candidate: candidate.clone(),
            confidence,
        });
        if confidence > best_confidence {
            best_candidate = candidate.clone();
            best_confidence = confidence;
        }
        population.push((candidate.clone(), confidence));
    }

    let mut generation = 0usize;
    while query_index < config.max_model_queries {
        generation += 1;
        let mut offspring = Vec::with_capacity(config.population_size);
        let remaining = config.max_model_queries - query_index;
        let offspring_count = remaining.min(config.population_size);

        for _ in 0..offspring_count {
            let parent_index = match selection {
                PopulationSelection::Tournament => {
                    tournament_parent_index(&population, config.tournament_size, &mut rng)
                }
                PopulationSelection::Truncation => {
                    truncation_parent_index(
                        &population,
                        config.truncation_fraction,
                        &mut rng,
                    )
                }
            };
            let mut candidate = population[parent_index].0.clone();
            mutate_one_site(&mut candidate, alphabet_size, &mut rng);
            let confidence = evaluate(model, &candidate)?;
            query_index += 1;
            trace.push(PopulationQueryTrace {
                query_index,
                generation,
                candidate: candidate.clone(),
                confidence,
            });
            if confidence > best_confidence {
                best_candidate = candidate.clone();
                best_confidence = confidence;
            }
            offspring.push((candidate, confidence));
        }

        population.extend(offspring);
        population.sort_by(|left, right| right.1.total_cmp(&left.1));
        population.truncate(config.population_size);
    }

    population.sort_by(|left, right| right.1.total_cmp(&left.1));
    let generations_evaluated = generation + 1;
    Ok(PopulationAttackResult {
        selection,
        seed: config.seed,
        model_queries: query_index,
        generations_evaluated,
        best_candidate,
        best_confidence,
        final_population: population.into_iter().map(|entry| entry.0).collect(),
        trace,
    })
}

fn tournament_parent_index(
    population: &[(Vec<u8>, f64)],
    tournament_size: usize,
    rng: &mut XorShift64,
) -> usize {
    let mut best_index = rng.next_usize(population.len());
    for _ in 1..tournament_size {
        let challenger = rng.next_usize(population.len());
        if population[challenger].1 > population[best_index].1 {
            best_index = challenger;
        }
    }
    best_index
}

fn truncation_parent_index(
    population: &[(Vec<u8>, f64)],
    truncation_fraction: f64,
    rng: &mut XorShift64,
) -> usize {
    let mut ranked: Vec<usize> = (0..population.len()).collect();
    ranked.sort_by(|&left, &right| population[right].1.total_cmp(&population[left].1));
    let selected = ((population.len() as f64 * truncation_fraction).ceil() as usize)
        .max(1)
        .min(population.len());
    ranked[rng.next_usize(selected)]
}

fn validate_candidate(candidate: &[u8], alphabet_size: u16) -> Result<(), AttackError> {
    if candidate.is_empty() {
        return Err(AttackError::EmptyCandidate);
    }
    if !(2..=256).contains(&alphabet_size) {
        return Err(AttackError::AlphabetSizeOutOfRange { alphabet_size });
    }
    for (position, &symbol) in candidate.iter().enumerate() {
        if u16::from(symbol) >= alphabet_size {
            return Err(AttackError::InitialSymbolOutOfRange {
                position,
                symbol,
                alphabet_size,
            });
        }
    }
    Ok(())
}

fn evaluate<M: ConfidenceModel>(model: &M, candidate: &[u8]) -> Result<f64, AttackError> {
    let confidence = model
        .confidence(candidate)
        .map_err(AttackError::ModelFailure)?;
    if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
        return Err(AttackError::InvalidConfidence(confidence));
    }
    Ok(confidence)
}

fn mutate_one_site(candidate: &mut [u8], alphabet_size: u16, rng: &mut XorShift64) {
    let position = rng.next_usize(candidate.len());
    let current = u16::from(candidate[position]);
    let offset = rng.next_usize(usize::from(alphabet_size - 1)) as u16;
    let replacement = if offset >= current { offset + 1 } else { offset };
    candidate[position] = replacement as u8;
}

/// Small, local, deterministic generator; no external random-number dependency is required.
/// The seed and algorithm are part of the experiment configuration and must be recorded.
#[derive(Debug, Clone, Copy)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        value
    }

    fn next_unit(&mut self) -> f64 {
        // Use the top 53 bits so the result is always in [0, 1), including for u64::MAX.
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn next_usize(&mut self, upper_exclusive: usize) -> usize {
        assert!(upper_exclusive > 0, "upper_exclusive must be non-zero");
        let upper = upper_exclusive as u64;
        // Rejection sampling removes modulo bias while preserving deterministic replay.
        let zone = u64::MAX - (u64::MAX % upper);
        loop {
            let value = self.next_u64();
            if value < zone {
                return (value % upper) as usize;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct OneSymbolAttractor;

    impl ConfidenceModel for OneSymbolAttractor {
        fn confidence(&self, candidate: &[u8]) -> Result<f64, String> {
            Ok(if candidate.len() == 1 && candidate[0] == 1 { 0.99 } else { 0.10 })
        }
    }

    struct ZeroAttractor;

    impl ConfidenceModel for ZeroAttractor {
        fn confidence(&self, candidate: &[u8]) -> Result<f64, String> {
            Ok(if candidate == [0] { 0.80 } else { 0.20 })
        }
    }

    struct InvalidModel(f64);

    impl ConfidenceModel for InvalidModel {
        fn confidence(&self, _candidate: &[u8]) -> Result<f64, String> {
            Ok(self.0)
        }
    }

    struct MeanSymbolConfidence;

    impl ConfidenceModel for MeanSymbolConfidence {
        fn confidence(&self, candidate: &[u8]) -> Result<f64, String> {
            let sum: u64 = candidate.iter().map(|&symbol| u64::from(symbol)).sum();
            let max = candidate.len() as u64 * 25;
            Ok(sum as f64 / max as f64)
        }
    }

    #[test]
    fn greedy_ascent_can_find_a_high_confidence_non_replicator_surface() {
        let result = run_confidence_attack(
            &OneSymbolAttractor,
            &[0],
            2,
            AttackConfig {
                max_model_queries: 4,
                seed: 7,
                strategy: SearchStrategy::GreedyAscent,
            },
        )
        .expect("attack should run");

        assert_eq!(result.model_queries, 4);
        assert_eq!(result.trace.len(), 4);
        assert_eq!(result.best_candidate, vec![1]);
        assert_eq!(result.best_confidence, 0.99);

        // Ground truth is deliberately checked by a separate test-side oracle, after the
        // search. High model confidence does not imply functional replication.
        let independently_verified_replicator = result.best_candidate.as_slice() == [0];
        assert!(!independently_verified_replicator);
    }

    #[test]
    fn random_walk_tracks_best_candidate_separately_from_terminal_state() {
        let result = run_confidence_attack(
            &ZeroAttractor,
            &[0],
            2,
            AttackConfig {
                max_model_queries: 2,
                seed: 3,
                strategy: SearchStrategy::RandomWalk,
            },
        )
        .expect("attack should run");

        assert_eq!(result.initial_confidence, 0.80);
        assert_eq!(result.best_candidate, vec![0]);
        assert_eq!(result.final_candidate, vec![1]);
        assert_eq!(result.best_confidence, 0.80);
        assert_eq!(result.final_confidence, 0.20);
    }

    #[test]
    fn same_seed_and_configuration_produce_identical_traces() {
        let config = AttackConfig {
            max_model_queries: 50,
            seed: 0x0A11_CE55,
            strategy: SearchStrategy::GreedyAscent,
        };
        let first =
            run_confidence_attack(&MeanSymbolConfidence, &[0; 9], 26, config).unwrap();
        let second =
            run_confidence_attack(&MeanSymbolConfidence, &[0; 9], 26, config).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn query_budget_includes_the_initial_model_evaluation() {
        let result = run_confidence_attack(
            &OneSymbolAttractor,
            &[0],
            2,
            AttackConfig {
                max_model_queries: 1,
                seed: 1,
                strategy: SearchStrategy::GreedyAscent,
            },
        )
        .unwrap();
        assert_eq!(result.model_queries, 1);
        assert_eq!(result.trace.len(), 1);
    }

    #[test]
    fn stochastic_attack_is_reproducible_and_keeps_best_separate() {
        let config = StochasticAttackConfig {
            max_model_queries: 20,
            seed: 0x5157_0C,
            temperature: 0.5,
        };
        let first = run_stochastic_attack(&MeanSymbolConfidence, &[0; 9], 26, config).unwrap();
        let second = run_stochastic_attack(&MeanSymbolConfidence, &[0; 9], 26, config).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.model_queries, 20);
        assert!(first.best_confidence >= first.final_confidence);
    }

    #[test]
    fn stochastic_attack_can_accept_a_downhill_move_at_high_temperature() {
        let result = run_stochastic_attack(
            &ZeroAttractor,
            &[0],
            2,
            StochasticAttackConfig {
                max_model_queries: 2,
                seed: 3,
                temperature: 1.0e12,
            },
        )
        .unwrap();
        assert_eq!(result.trace.len(), 2);
        assert!(result.trace[1].accepted_as_current);
        assert_eq!(result.final_confidence, 0.20);
        assert_eq!(result.best_confidence, 0.80);
    }

    #[test]
    fn stochastic_attack_rejects_invalid_temperature() {
        for temperature in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let error = run_stochastic_attack(
                &ZeroAttractor,
                &[0],
                2,
                StochasticAttackConfig {
                    max_model_queries: 2,
                    seed: 1,
                    temperature,
                },
            )
            .unwrap_err();
            assert!(matches!(error, AttackError::InvalidTemperature(_)));
        }
    }

    #[test]
    fn population_tournament_and_truncation_respect_global_query_budget() {
        let initial = vec![
            vec![0; 9],
            vec![1; 9],
            vec![2; 9],
            vec![3; 9],
            vec![4; 9],
            vec![5; 9],
        ];
        let base = PopulationAttackConfig {
            max_model_queries: 18,
            seed: 91,
            population_size: 6,
            tournament_size: 3,
            truncation_fraction: 0.2,
        };
        for selection in [
            PopulationSelection::Tournament,
            PopulationSelection::Truncation,
        ] {
            let first = run_population_attack(
                &MeanSymbolConfidence,
                &initial,
                26,
                selection,
                base,
            )
            .unwrap();
            let second = run_population_attack(
                &MeanSymbolConfidence,
                &initial,
                26,
                selection,
                base,
            )
            .unwrap();
            assert_eq!(first, second);
            assert_eq!(first.model_queries, 18);
            assert_eq!(first.trace.len(), 18);
            assert_eq!(first.final_population.len(), 6);
            assert!(first.best_confidence >= 0.0);
        }
    }

    #[test]
    fn population_attack_rejects_budget_below_initial_population_and_mixed_lengths() {
        let initial = vec![vec![0; 9]; 4];
        let base = PopulationAttackConfig {
            max_model_queries: 3,
            seed: 1,
            population_size: 4,
            tournament_size: 2,
            truncation_fraction: 0.25,
        };
        assert!(matches!(
            run_population_attack(
                &MeanSymbolConfidence,
                &initial,
                26,
                PopulationSelection::Tournament,
                base
            ),
            Err(AttackError::QueryBudgetBelowPopulation { .. })
        ));

        let mixed = vec![vec![0; 9], vec![0; 8], vec![0; 9], vec![0; 9]];
        assert!(matches!(
            run_population_attack(
                &MeanSymbolConfidence,
                &mixed,
                26,
                PopulationSelection::Tournament,
                PopulationAttackConfig {
                    max_model_queries: 4,
                    ..base
                }
            ),
            Err(AttackError::CandidateLengthMismatch { .. })
        ));
    }

    #[test]
    fn population_attack_rejects_mismatched_population_and_bad_selection() {
        let initial = vec![vec![0; 9]; 3];
        let config = PopulationAttackConfig {
            max_model_queries: 30,
            seed: 1,
            population_size: 4,
            tournament_size: 3,
            truncation_fraction: 0.2,
        };
        assert!(matches!(
            run_population_attack(
                &MeanSymbolConfidence,
                &initial,
                26,
                PopulationSelection::Tournament,
                config
            ),
            Err(AttackError::PopulationSizeMismatch { .. })
        ));

        let valid_initial = vec![vec![0; 9]; 4];
        for fraction in [0.0, -0.1, 1.1, f64::NAN] {
            let error = run_population_attack(
                &MeanSymbolConfidence,
                &valid_initial,
                26,
                PopulationSelection::Truncation,
                PopulationAttackConfig {
                    truncation_fraction: fraction,
                    ..config
                },
            )
            .unwrap_err();
            assert!(matches!(error, AttackError::InvalidTruncationFraction(_)));
        }
    }

    #[test]
    fn rejects_non_finite_and_out_of_range_confidence() {
        for value in [f64::NAN, f64::INFINITY, -0.01, 1.01] {
            let error = run_confidence_attack(
                &InvalidModel(value),
                &[0],
                2,
                AttackConfig {
                    max_model_queries: 1,
                    seed: 1,
                    strategy: SearchStrategy::GreedyAscent,
                },
            )
            .expect_err("invalid confidence must fail closed");
            assert!(matches!(error, AttackError::InvalidConfidence(_)));
        }
    }

    #[test]
    fn rejects_invalid_candidates_and_budgets() {
        let config = AttackConfig::default();
        assert_eq!(
            run_confidence_attack(&OneSymbolAttractor, &[], 2, config).unwrap_err(),
            AttackError::EmptyCandidate
        );
        assert_eq!(
            run_confidence_attack(&OneSymbolAttractor, &[2], 2, config).unwrap_err(),
            AttackError::InitialSymbolOutOfRange {
                position: 0,
                symbol: 2,
                alphabet_size: 2,
            }
        );
        assert_eq!(
            run_confidence_attack(&OneSymbolAttractor, &[0], 1, config).unwrap_err(),
            AttackError::AlphabetSizeOutOfRange { alphabet_size: 1 }
        );
        assert_eq!(
            run_confidence_attack(
                &OneSymbolAttractor,
                &[0],
                2,
                AttackConfig {
                    max_model_queries: 0,
                    ..config
                }
            )
            .unwrap_err(),
            AttackError::ZeroQueryBudget
        );
    }
}
