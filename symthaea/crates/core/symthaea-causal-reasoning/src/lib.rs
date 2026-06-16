// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Causal Reasoning for Symthaea
//!
//! Pearl's do-calculus, counterfactual reasoning, and causal emergence analysis.
//!
//! - [`causal_calculus`]: Structural Causal Models, do-calculus rules, interventional queries
//! - [`causal_emergence`]: Hoel's Effective Information and causal emergence measurement
//! - [`counterfactual`]: Backdoor/frontdoor identification, HDC graph surgery, semantic roles

#![deny(unsafe_code)]
#![allow(clippy::needless_range_loop)]

pub mod causal_calculus;
pub mod causal_emergence;

#[cfg(feature = "counterfactual")]
pub mod counterfactual;
