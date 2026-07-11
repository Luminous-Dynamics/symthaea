// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-social-choice
//!
//! Social-choice & voting theory — the formal core of "how a group decides".
//! Complements `symthaea-economics` (game theory) and `symthaea-legal-reasoning`
//! (deontic logic), and has a concrete downstream consumer: **Mycelix
//! governance** (proposals, voting, councils).
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link. Every result is
//! checked against a worked textbook example.
//!
//! ## Layers
//! - [`voting`] — ranked-ballot rules: plurality, Borda, Condorcet,
//!   instant-runoff. Demonstrates that the rule, not just the votes, picks the
//!   winner (plurality vs. Condorcet can disagree).
//! - [`apportionment`] — turning vote counts into whole seats: D'Hondt,
//!   Sainte-Laguë, Hamilton.
//! - [`power`] — voting-power indices (Banzhaf, Shapley-Shubik) for weighted
//!   voting games: influence ≠ raw weight.
//!
//! ## Example
//!
//! ```
//! use symthaea_social_choice::voting::{plurality_winner, condorcet_winner};
//! // 4×(a>b>c), 3×(b>c>a), 2×(c>b>a): plurality picks a, but b beats everyone
//! // head-to-head — the rule decides the outcome.
//! let mk = |o: &[&str]| o.iter().map(|s| s.to_string()).collect::<Vec<_>>();
//! let mut ballots = vec![];
//! for _ in 0..4 { ballots.push(mk(&["a","b","c"])); }
//! for _ in 0..3 { ballots.push(mk(&["b","c","a"])); }
//! for _ in 0..2 { ballots.push(mk(&["c","b","a"])); }
//! assert_eq!(plurality_winner(&ballots), Some("a".to_string()));
//! assert_eq!(condorcet_winner(&ballots), Some("b".to_string()));
//! ```

pub mod apportionment;
pub mod power;
pub mod voting;

pub use apportionment::{dhondt, hamilton, sainte_lague};
pub use power::{banzhaf, shapley_shubik};
pub use voting::{borda_winner, condorcet_winner, instant_runoff_winner, plurality_winner};
