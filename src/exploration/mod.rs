// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Exploration Strategies for Symthaea
//!
//! Re-exports from the `symthaea-exploration` sub-crate plus experimental
//! measurement-only evaluation contracts that are not yet part of action selection.

/// Niche-preserving, confidence-qualified Pareto comparison without scalar fitness.
pub mod archive;
/// Evidence-aware Pareto admission that keeps support separate from reported confidence.
pub mod evidence_admission;
/// Content-addressed binding from evidence-plane runs into generativity evidence.
pub mod evidence_binding;
/// Evidence-bearing, non-authoritative evaluation of future capacity and optionality.
pub mod generativity;
/// Typed outcomes that keep surprise reduction distinct from broader discovery value.
pub mod outcome;
/// Re-verifiable persistence boundary for evidence-backed generativity assessments.
pub mod persisted_evidence;

pub use symthaea_exploration::*;

/// Re-export the sub-crate itself as `surprise_driven` for backwards compat.
pub use symthaea_exploration as surprise_driven;
