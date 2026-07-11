// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-alife
//!
//! Artificial life for Symthaea, per `ALIFE_PLAN_2026-07-08.md`.
//!
//! An [`Organism`] is a Markov blanket ([`symthaea_fep::markov_blanket`]) wrapping one
//! [`symthaea_fep::ActiveInferenceAgent`]: it perceives an exogenous environment signal it
//! does not control, selects actions to minimize expected free energy, and pays real
//! metabolic energy for those actions. Deliberately built directly on `symthaea-fep` —
//! no HDC hypervectors, no `EmbodimentBridge`, no consciousness-loop coupling.
//!
//! ## Phase 0 scope
//!
//! One organism, one exogenous resource signal, two actions (forage / rest), a real energy
//! budget. Two claims, both ground-truth tested in `tests/phase0_ground_truth.rs`:
//!
//! 1. An organism that actually calls `perceive()` tracks the resource signal better than
//!    one that never does (tests that perception is doing real work, not theater).
//! 2. An organism whose actions come from `select_action()` regulates its energy better than
//!    one whose actions are uniform-random (tests that action selection is doing real work).
//!
//! Neither claim is about consciousness or Φ — see the plan doc's Non-goals.

pub mod coalition;
pub mod earth_forcing;
pub mod environment;
pub mod genome;
pub mod hierarchy;
pub mod metabolism;
pub mod organism;
pub mod population;
pub mod predator_prey;
pub mod types;

pub use coalition::{Coalition, detect_coalitions, detect_paying_coalitions};
pub use earth_forcing::EarthForcedEnvironment;
pub use environment::Environment;
pub use genome::Genome;
pub use hierarchy::HierarchicalStack;
pub use metabolism::{
    K_ALIFE_BOLTZMANN, landauer_minimum, prigogine_dissipation_cost, shannon_entropy_bits,
};
pub use organism::{Action, Organism, OrganismConfig, OrganismTick};
pub use population::{InheritanceMode, Population, PopulationConfig, StepSummary};
pub use predator_prey::{PredatorPreyConfig, PredatorPreySim, PredatorPreyStep};
pub use types::BoundaryModulators;
