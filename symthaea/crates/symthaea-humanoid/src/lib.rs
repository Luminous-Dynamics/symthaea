// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

pub mod simulator;
pub mod types;
pub mod morphology; 
pub mod evolution;
pub mod control;

pub use evolution::{GaitGenome, Rng};
pub use control::{GenerativePrior, MachineState, CfcCpg, HdcWorkspace, LocomotionModule, execute_modular_gait};
pub use crate::control::GaitControlProfile;
