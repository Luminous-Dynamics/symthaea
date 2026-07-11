// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-markov
//!
//! Discrete-time Markov chains and stochastic-process analysis — foundational
//! machinery the workspace lacked, though queueing (`operations-research`),
//! disease dynamics (`epidemiology`), decision processes, and ranking all rest
//! on it.
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link. Checked against
//! closed-form results (two-state stationary distribution, gambler's-ruin
//! absorption).
//!
//! ## Contents
//! - [`chain::MarkovChain`] — validated transition matrix, n-step and
//!   stationary distributions
//! - [`absorbing`] — absorbing-chain analysis: expected steps to absorption and
//!   absorption probabilities, via the fundamental matrix N = (I − Q)⁻¹
//!
//! ## Example
//!
//! ```
//! use symthaea_markov::chain::MarkovChain;
//! // Two-state chain: stationary distribution is (5/6, 1/6).
//! let c = MarkovChain::new(vec![vec![0.9, 0.1], vec![0.5, 0.5]]).unwrap();
//! let pi = c.stationary_distribution(1000);
//! assert!((pi[0] - 5.0 / 6.0).abs() < 1e-9);
//! ```

pub mod absorbing;
pub mod chain;

pub use absorbing::{Absorbing, absorption_probabilities, classify, expected_steps_to_absorption};
pub use chain::MarkovChain;
