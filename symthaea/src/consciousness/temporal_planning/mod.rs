//! Temporal Planning Subsystem
//!
//! O(1) temporal planning via budget-bounded MCTS with ForkedState.
//!
//! - **ForkedState**: Arc-shared weights + cloned states for zero-copy forks
//! - **Micro-MCTS**: Budget-bounded, EVS-gated, dream-prior-biased
//! - **Budget Tiers**: Tier0 (≤2ms), Tier1 (≤8ms), Tier2 (≤20ms)
//! - **Dream Integration**: Counterfactual insights → action priors

pub mod types;
pub mod snapshot;
pub mod mcts;
pub mod dream_integration;

// Re-export key types
pub use types::{
    BudgetTier, ReasoningBudget, MctsConfig, MctsResult,
    ForkedState, PlannedAction,
};
pub use snapshot::SnapshotManager;
pub use mcts::{MctsPlanner, evs};
pub use dream_integration::uniform_priors;
