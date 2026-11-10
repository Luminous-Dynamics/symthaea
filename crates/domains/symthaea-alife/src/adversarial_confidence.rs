// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic, model-agnostic confidence-surface attacks for OLI research.
//!
//! The model proposes a confidence; an independent oracle supplies ground truth.
//! This module contains no classifier and makes no life-detection claim. It provides a
//! reproducible search harness for registered synthetic/digital-life challenge domains.
//!
//! The implemented baseline is greedy single-site hill climbing over a finite alphabet,
//! matching the attack family used by Gupta & Adami (2026). Query accounting is explicit:
//! the initial sequence evaluation consumes one model query; every proposed mutation
//! consumes one additional query. The configured budget bounds total model evaluations.

use std::collections::HashSet;
use std::convert::Infallible;

/// Model interface under attack.
///
/// Implementations should expose inference only. They must not be given the known-truth
/// oracle or hidden generator state.
pub trait ConfidenceModel {
    type Error;

    /// Return the model's probability/confidence for the target class.
    fn confidence(&mut self, candidate: &[u8]) -> Result<f64, Self::Error>;
}

/// Independent ground-truth oracle.
///
/// In a benchmark, this should be a trusted executable generator or separately qualified
/// reference implementation, never the model under attack.
pub trait GroundTruthOracle {
    fn is_positive(&self, candidate: &[u8]) -> bool;
}

/// Exact-membership oracle for finite known-truth sequence sets.
///
/// Useful for Avida's published 9-mer replicator landscape once the source-pinned set has
/// been admitted. This struct does not bundle or redistribute that external dataset.
#[derive(Debug, Clone, Default)]
pub struct ExactSequenceOracle {
    positives: HashSet<Vec<u8>>,
}

impl ExactSequenceOracle {
    pub fn new<I>(positives: I) -> Self
    where
        I: IntoIterator<Item = Vec<u8>>,
    {
        Self {
            positives: positives.into_iter().collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.positives.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positives.is_empty()
    }
}

impl GroundTruthOracle for ExactSequenceOracle {
    fn is_positive(&self, candidate: &[u8]) -> bool {
        self.positives.contains(candidate)
    }
}

/// Fixed parameters for a confidence hill-climb run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HillClimbConfig {
    /// Maximum total calls to the model, including the starting-sequence evaluation.
    pub max_model_queries: usize,
    /// Deterministic seed for the mutation proposal stream.
    pub seed: u64,
}

/// One model evaluation in the search trace.
#[derive(Debug, Clone, PartialEq)]
pub struct AttackStep {
    /// One-based model query index. Query 1 evaluates the initial sequence.
    pub query_index: usize,
    pub candidate: Vec<u8>,
    pub confidence: f64,
    /// Whether the candidate became the active search state.
    pub accepted: bool,
    pub active_after_query: Vec<u8>,
    pub active_confidence_after_query: f64,
}

/// Result of one confidence-surface attack.
#[derive(Debug, Clone, PartialEq)]
pub struct HillClimbResult {
    pub seed: u64,
    pub start: Vec<u8>,
    pub start_confidence: f64,
    pub endpoint: Vec<u8>,
    pub endpoint_confidence: f64,
    pub endpoint_is_positive: bool,
    pub model_queries: usize,
    pub accepted_mutations: usize,
    pub trace: Vec<AttackStep>,
}

/// Errors detected before or during model evaluation.
#[derive(Debug)]
pub enum HillClimbError<E> {
    EmptyStart,
    EmptyAlphabet,
    DuplicateAlphabetSymbol(u8),
    StartSymbolOutsideAlphabet(u8),
    ZeroQueryBudget,
    Model(E),
    InvalidConfidence { query_index: usize, confidence: f64 },
}

impl<E> From<E> for HillClimbError<E> {
    fn from(error: E) -> Self {
        Self::Model(error)
    }
}

/// Run greedy single-site confidence ascent.
///
/// A mutation selects one position uniformly from the sequence and one replacement symbol
/// uniformly from the alphabet excluding the current symbol at that position. The proposal
/// is accepted only if its confidence is strictly greater than the current active state's
/// confidence. Ties and lower-confidence proposals are rejected.
///
/// The initial evaluation counts against `max_model_queries`. An alphabet of one symbol
/// is valid but cannot propose a distinct mutation. Invalid/non-finite model confidence
/// values fail closed.
pub fn greedy_single_site_hill_climb<M, O>(
    model: &mut M,
    oracle: &O,
    start: &[u8],
    alphabet: &[u8],
    config: HillClimbConfig,
) -> Result<HillClimbResult, HillClimbError<M::Error>>
where
    M: ConfidenceModel,
    O: GroundTruthOracle,
{
    if start.is_empty() {
        return Err(HillClimbError::EmptyStart);
    }
    if alphabet.is_empty() {
        return Err(HillClimbError::EmptyAlphabet);
    }
    if config.max_model_queries == 0 {
        return Err(HillClimbError::ZeroQueryBudget);
    }

    let mut seen = HashSet::with_capacity(alphabet.len());
    for &symbol in alphabet {
        if !seen.insert(symbol) {
            return Err(HillClimbError::DuplicateAlphabetSymbol(symbol));
        }
    }
    for &symbol in start {
        if !seen.contains(&symbol) {
            return Err(HillClimbError::StartSymbolOutsideAlphabet(symbol));
        }
    }

    let mut rng = DeterministicRng::new(config.seed);
    let mut active = start.to_vec();
    let start_confidence = checked_confidence(model, &active, 1)?;
    let mut active_confidence = start_confidence;
    let mut trace = Vec::with_capacity(config.max_model_queries);
    trace.push(AttackStep {
        query_index: 1,
        candidate: active.clone(),
        confidence: start_confidence,
        accepted: true,
        active_after_query: active.clone(),
        active_confidence_after_query: active_confidence,
    });

    let mut model_queries = 1;
    let mut accepted_mutations = 0;

    while model_queries < config.max_model_queries && alphabet.len() > 1 {
        let position = rng.index(active.len());
        let replacement = rng.replacement(alphabet, active[position]);
        let mut candidate = active.clone();
        candidate[position] = replacement;

        let query_index = model_queries + 1;
        let candidate_confidence = checked_confidence(model, &candidate, query_index)?;
        let accepted = candidate_confidence > active_confidence;
        if accepted {
            active = candidate.clone();
            active_confidence = candidate_confidence;
            accepted_mutations += 1;
        }

        model_queries += 1;
        trace.push(AttackStep {
            query_index,
            candidate,
            confidence: candidate_confidence,
            accepted,
            active_after_query: active.clone(),
            active_confidence_after_query: active_confidence,
        });
    }

    Ok(HillClimbResult {
        seed: config.seed,
        start: start.to_vec(),
        start_confidence,
        endpoint_is_positive: oracle.is_positive(&active),
        endpoint: active,
        endpoint_confidence: active_confidence,
        model_queries,
        accepted_mutations,
        trace,
    })
}

fn checked_confidence<M: ConfidenceModel>(
    model: &mut M,
    candidate: &[u8],
    query_index: usize,
) -> Result<f64, HillClimbError<M::Error>> {
    let confidence = model.confidence(candidate).map_err(HillClimbError::Model)?;
    if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
        return Err(HillClimbError::InvalidConfidence {
            query_index,
            confidence,
        });
    }
    Ok(confidence)
}

/// Small deterministic PRNG used only for proposal generation.
///
/// This is not a cryptographic RNG. Its purpose is reproducible benchmark replay with
/// an explicit seed, not security or physical randomness.
#[derive(Debug, Clone, Copy)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
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

    fn index(&mut self, upper_exclusive: usize) -> usize {
        (self.next_u64() % upper_exclusive as u64) as usize
    }

    fn replacement(&mut self, alphabet: &[u8], current: u8) -> u8 {
        let choice = self.index(alphabet.len() - 1);
        let mut seen = 0;
        for &symbol in alphabet {
            if symbol != current {
                if seen == choice {
                    return symbol;
                }
                seen += 1;
            }
        }
        unreachable!("alphabet contains at least one symbol distinct from current");
    }
}

/// A model-free evaluator for tests and examples that need no error channel.
pub struct InfallibleConfidence<F>(pub F);

impl<F> ConfidenceModel for InfallibleConfidence<F>
where
    F: FnMut(&[u8]) -> f64,
{
    type Error = Infallible;

    fn confidence(&mut self, candidate: &[u8]) -> Result<f64, Self::Error> {
        Ok((self.0)(candidate))
    }
}
