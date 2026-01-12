//! Domain Adapters - Pluggable domains for the generalized agent architecture
//!
//! This module contains domain-specific implementations of the core traits,
//! enabling the same planning and reasoning infrastructure to work across:
//!
//! - **Task Domain** (`task.rs`): MMLU-style reasoning with semantic concepts
//! - **GridWorld** (`../core/gridworld.rs`): Simple 2D navigation (validation domain)
//! - **Consciousness** (in `consciousness/` module): The original domain
//!
//! ## Design Principle
//!
//! Each domain implements:
//! - `State`: What the world looks like
//! - `Action`: What the agent can do
//! - `Goal`: What the agent is trying to achieve
//! - `DomainAdapter`: How to configure the infrastructure for this domain
//! - `ActionProvider`: How to enumerate available actions
//!
//! ## Validation Strategy
//!
//! - **Phase 1** (Complete): GridWorld validates the generalization infrastructure works
//! - **Phase 2** (Current): TaskState validates it works for REASONING problems
//! - **Phase 3** (Future): Correlate Φ with MMLU accuracy to validate the science

pub mod task;

pub use task::{
    TaskState,
    TaskAction,
    SemanticGoal,
    TaskDynamics,
    TaskDomainAdapter,
};
