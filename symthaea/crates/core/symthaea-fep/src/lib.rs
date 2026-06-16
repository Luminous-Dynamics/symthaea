#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![allow(clippy::needless_range_loop, clippy::manual_clamp)]
//! # Full Active Inference Implementation (FEP Integration)
//!
//! Implements Karl Friston's Free Energy Principle (FEP) as a complete active inference loop
//! with motor command generation and temporal difference learning.
//!
//! ## Mathematical Foundation
//!
//! The Free Energy Principle posits that biological systems minimize variational free energy:
//!
//! ```text
//! F = E_q[ln q(s) - ln p(o,s)]
//!   = D_KL[q(s) || p(s|o)] - ln p(o)
//!   ≥ -ln p(o)  (Surprise)
//! ```
//!
//! where:
//! - `p(o,s)` is the generative model (joint distribution over observations and states)
//! - `q(s)` is the recognition model (approximate posterior over hidden states)
//! - `F` is variational free energy (upper bound on surprise)
//!
//! ## Active Inference Loop
//!
//! 1. **Perception**: Update beliefs q(s) to minimize free energy given observations
//! 2. **Action Selection**: Choose actions that minimize expected free energy
//! 3. **Model Learning**: Update generative model parameters based on prediction errors
//!
//! ## Architecture Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    FEP ACTIVE INFERENCE PIPELINE                        │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │  Observation (phi, integration, coherence, attention)                   │
//! │           │                                                             │
//! │           ▼                                                             │
//! │  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐     │
//! │  │  Generative     │───▶│  Free Energy     │───▶│  Precision     │     │
//! │  │  Model P(o,s)   │    │  Calculator      │    │  Estimator     │     │
//! │  └─────────────────┘    └──────────────────┘    └───────┬────────┘     │
//! │                                                          │              │
//! │           ┌──────────────────────────────────────────────┘              │
//! │           ▼                                                             │
//! │  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐     │
//! │  │  Expected FE    │───▶│  Action          │───▶│  Motor         │     │
//! │  │  Computer       │    │  Selection       │    │  Command       │     │
//! │  └─────────────────┘    └──────────────────┘    └───────┬────────┘     │
//! │                                                          │              │
//! │           ┌──────────────────────────────────────────────┘              │
//! │           ▼                                                             │
//! │  ┌─────────────────┐    ┌──────────────────┐                           │
//! │  │  Motor System   │───▶│  TD Learner      │                           │
//! │  │  (Execute)      │    │  (Update Model)  │                           │
//! │  └─────────────────┘    └──────────────────┘                           │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Motor Command System
//!
//! The FEP bridge outputs one of 8 [`MotorCommandType`] variants based on predicted
//! expected free energy. Each command represents a different cognitive action:
//!
//! | Command | Index | When It Fires | Effect |
//! |---------|-------|---------------|--------|
//! | `AttentionShift` | 0 | High precision error in specific modality | Redirect processing focus |
//! | `LearningRateAdjust` | 1 | Model confidence is changing | Increase/decrease learning |
//! | `ExplorationTrigger` | 2 | Low epistemic value, high uncertainty | Seek novel inputs |
//! | `ReflectionInitiate` | 3 | High free energy, stable state | Pause for metacognition |
//! | `MemoryConsolidate` | 4 | High confidence, low prediction error | Strengthen representations |
//! | `ExpectationReset` | 5 | Persistent high prediction error | Clear prediction cache |
//! | `MotorOutput` | 6 | Pragmatic action needed | Execute external action |
//! | `NoOp` | 7 | System at equilibrium | Maintain current state |
//!
//! ## Temporal Difference Learning
//!
//! The module implements TD(λ) learning with eligibility traces:
//!
//! ```text
//! δ = r + γV(s') - V(s)    // TD error
//! e(s,a) = γλe(s,a) + ∇    // Eligibility trace
//! θ ← θ + αδe             // Parameter update
//! ```
//!
//! Key configuration parameters in [`TemporalDifferenceLearningConfig`]:
//! - `gamma` (default 0.99): Discount factor for future rewards
//! - `lambda` (default 0.8): Eligibility trace decay (0=TD(0), 1=Monte Carlo)
//! - `initial_learning_rate` (default 0.1): Starting learning rate
//!
//! ## Components
//!
//! - [`GenerativeModel`]: Maps hidden states → predicted observations
//! - [`FreeEnergyCalculator`]: Computes variational free energy and its components
//! - [`PrecisionEstimator`]: Dynamic precision weighting for confidence-weighted errors
//! - [`ActiveInferenceAgent`]: Full perception-action loop
//! - [`MotorSystem`]: Executes commands and tracks proprioceptive feedback
//! - [`TemporalDifferenceLearner`]: Updates model based on prediction errors
//! - [`EnhancedFEPBridge`]: High-level integration with cognitive loop
//!
//! ## Integration with Cognitive Loop
//!
//! Use [`EnhancedFEPBridge`] to connect FEP to the cognitive loop:
//!
//! ```rust,ignore
//! use symthaea_fep::{EnhancedFEPBridge, ActiveInferenceAgentConfig};
//!
//! let config = ActiveInferenceAgentConfig::default();
//! let mut bridge = EnhancedFEPBridge::new(config, 4);
//!
//! // Each cognitive cycle:
//! let result = bridge.cycle(phi, integration, coherence, attention);
//!
//! // Use the motor command
//! match result.motor_command.command_type {
//!     MotorCommandType::AttentionShift => { /* redirect focus */ }
//!     MotorCommandType::ExplorationTrigger => { /* seek novelty */ }
//!     // ...
//! }
//!
//! // Check if learning should occur
//! if result.should_learn {
//!     // Apply learning signal to downstream systems
//!     let lr = result.learning_signal;
//! }
//! ```
//!
//! ## References
//!
//! - Friston, K. (2010). The free-energy principle: a unified brain theory?
//! - Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017).
//!   Active Inference: A Process Theory.
//! - Parr, T., Pezzulo, G., & Friston, K. J. (2022). Active Inference: The Free Energy
//!   Principle in Mind, Brain, and Behavior.
//! - Sutton, R.S. & Barto, A.G. (2018). Reinforcement Learning: An Introduction (2nd ed.)

mod agent;
mod bridge;
pub mod free_energy;
pub mod generative_model;
pub mod haptic_semantic_binder;
pub mod hierarchical;
pub mod markov_blanket;
mod motor;
mod td_learning;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export all public types to maintain the same external API
pub use types::{
    ActionOutcome, ActionSelectionResult, ActiveInferenceAgentStats, ActiveInferenceSummary,
    CognitiveLoopFEPResult, EnhancedFEPCycleResult, ExpectedFreeEnergyResult, FreeEnergyComponents,
    HiddenState, MotorCommand, MotorCommandStats, MotorCommandType, MotorOutcome, Observation,
    PerceptionResult, PrecisionSnapshot,
};

pub use td_learning::{
    EligibilityTraces, ModelConfidenceTracker, StateTransition, TemporalDifferenceLearner,
    TemporalDifferenceLearningConfig, TemporalDifferenceLearningStats,
};

pub use generative_model::GenerativeModel;

pub use haptic_semantic_binder::HapticSemanticBinder;

pub use free_energy::{ExpectedFreeEnergyComputer, FreeEnergyCalculator, PrecisionEstimator};

pub use agent::{ActiveInferenceAgent, ActiveInferenceAgentConfig};

pub use motor::MotorSystem;

pub use bridge::{CognitiveLoopFEPBridge, EnhancedFEPBridge};

pub use markov_blanket::{
    BlanketPermeability, BlanketTelemetry, MarkovBoundaryOperator, MarkovPartition,
    PermeabilityInputs, SwarmCoalition, TopologyBoundaryInputs, identify_coalitions,
};
