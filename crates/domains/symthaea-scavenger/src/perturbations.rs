// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScavengerPerturbation {
    ExternalForce {
        magnitude: f64,
        channel: usize,
        at_step: usize,
    },
    ActuatorFailure {
        actuator: usize,
        at_step: usize,
    },
}

#[derive(Debug, Clone, Default)]
pub struct PerturbationSchedule {
    pub perturbations: Vec<ScavengerPerturbation>,
}
