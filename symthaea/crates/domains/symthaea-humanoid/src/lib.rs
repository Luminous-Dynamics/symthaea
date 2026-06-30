// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

pub mod control;
pub mod controller;
pub mod embodiment;
pub mod encoder;
pub mod evolution;
pub mod fep_agent;
pub mod gait;
pub mod morphology;
pub mod reward;
pub mod simulator;
pub mod training;
pub mod types;
pub use controller::*;
pub use encoder::*;
pub use simulator::*;
pub use types::*;

pub use crate::control::GaitControlProfile;
pub use control::{
    CfcCpg, GenerativePrior, HdcWorkspace, LocomotionModule, MachineState, execute_modular_gait,
};
pub use evolution::{GaitGenome, Rng};
