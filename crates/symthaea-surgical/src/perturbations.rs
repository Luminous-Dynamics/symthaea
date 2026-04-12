// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SurgicalPerturbation {
    TissueAnomaly {
        stiffness_mult: f64,
        at_step: usize,
    },
    PatientMovement {
        displacement_mm: [f64; 3],
        at_step: usize,
    },
    CameraFog {
        noise: f64,
        at_step: usize,
    },
}
#[derive(Debug, Clone, Default)]
pub struct PerturbationSchedule {
    pub perturbations: Vec<SurgicalPerturbation>,
}
