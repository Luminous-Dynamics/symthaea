// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CogBench: Cognitive psychology experiments via FEP active inference.
//!
//! 8 experiments testing exploration, model-based reasoning, learning,
//! temporal discounting, risk-taking, and cognitive flexibility.
//!
//! With `--features symthaea-backend`, additional `[Mind]` variants of
//! the Restless Bandit and Instrumental Learning benchmarks are available.
//! These use Symthaea's `ContinuousMind` (working memory + consciousness
//! monitoring) instead of FEP, testing WM-based decision-making.
//!
//! All benchmarks use stochastic action sampling from the agent's softmax
//! distribution (the standard active inference formulation) rather than
//! greedy argmax, ensuring genuine exploration/exploitation behavior.

pub mod bart;
pub mod horizon;
pub mod instrumental;
#[cfg(feature = "symthaea-backend")]
pub mod mind_agent;
pub mod probabilistic;
pub mod restless_bandit;
pub mod reversal_learning;
pub mod temporal_discounting;
pub mod two_step;

/// Sample an action from the FEP agent's softmax probability distribution.
///
/// Uses xorshift64 for deterministic, reproducible sampling. This implements
/// the standard active inference decision procedure: actions are sampled from
/// the softmax over negative expected free energies.
pub(crate) fn sample_action(probabilities: &[f64], rng_state: &mut u64) -> usize {
    *rng_state ^= *rng_state << 13;
    *rng_state ^= *rng_state >> 7;
    *rng_state ^= *rng_state << 17;
    let r = (*rng_state % 10000) as f64 / 10000.0;
    let mut cumsum = 0.0;
    for (i, &p) in probabilities.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }
    probabilities.len() - 1
}

pub use bart::BartBenchmark;
pub use horizon::HorizonBenchmark;
pub use instrumental::InstrumentalLearningBenchmark;
pub use probabilistic::ProbabilisticReasoningBenchmark;
pub use restless_bandit::RestlessBanditBenchmark;
pub use reversal_learning::ReversalLearningBenchmark;
pub use temporal_discounting::TemporalDiscountingBenchmark;
pub use two_step::TwoStepBenchmark;

#[cfg(feature = "symthaea-backend")]
pub use instrumental::InstrumentalLearningMindBenchmark;
#[cfg(feature = "symthaea-backend")]
pub use restless_bandit::RestlessBanditMindBenchmark;
