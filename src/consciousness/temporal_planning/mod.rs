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
pub use mcts::{evs, MctsPlanner};
pub use snapshot::SnapshotManager;
pub use types::{BudgetTier, ForkedState, MctsConfig, MctsResult, PlannedAction, ReasoningBudget};
