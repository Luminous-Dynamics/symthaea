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
//! The first strategy reproduces the *shape* of the greedy single-site mutation attack in
//! Gupta & Adami (2026), arXiv:2604.11915: propose one symbol substitution, query the model,
//! and accept only a confidence-increasing proposal. A caller must supply the model and the
//! independently held-out ground-truth evaluator. No model score is treated as truth.
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

/// Invalid search configuration, model response, or candidate.
#[derive(Debug, Clone, PartialEq)]
pub enum AttackError {
    EmptyCandidate,
    AlphabetSizeOutOfRange { alphabet_size: u16 },
    InitialSymbolOutOfRange { position: usize, symbol: u8, alphabet_size: u16 },
    ZeroQueryBudget,
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
/// Greedy ascent and random walk are included as the first paired red-team/control policies.
/// Stochastic acceptance and population methods belong in a later benchmark increment so
/// their temperature, population size, generation count, and query accounting can be frozen
/// independently instead of hidden behind a generic "evolutionary" option.
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
    let offset = (rng.next_u64() % u64::from(alphabet_size - 1)) as u16;
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

    fn next_usize(&mut self, upper_exclusive: usize) -> usize {
        (self.next_u64() % upper_exclusive as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct OneSymbolAttractor;

    impl ConfidenceModel for OneSymbolAttractor {
        fn confidence(&self, candidate: &[u8]) -> Result<f64, String> {
            Ok(if candidate == [1] { 0.99 } else { 0.10 })
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
        let independently_verified_replicator = result.best_candidate == [0];
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
