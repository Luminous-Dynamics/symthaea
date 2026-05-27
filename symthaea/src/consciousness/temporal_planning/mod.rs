// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Temporal Planning Subsystem
//!
//! O(1) temporal planning via budget-bounded MCTS with ForkedState.
//!
//! - **ForkedState**: Arc-shared weights + cloned states for zero-copy forks
//! - **Micro-MCTS**: Budget-bounded, EVS-gated, dream-prior-biased
//! - **Budget Tiers**: Tier0 (≤2ms), Tier1 (≤8ms), Tier2 (≤20ms)
//! - **Dream Integration**: Counterfactual insights → action priors

pub mod dream_integration;
pub mod mcts;
pub mod snapshot;
pub mod types;

// Re-export key types
pub use dream_integration::uniform_priors;
pub use mcts::{MctsPlanner, evs};
pub use snapshot::SnapshotManager;
pub use types::{BudgetTier, ForkedState, MctsConfig, MctsResult, PlannedAction, ReasoningBudget};
