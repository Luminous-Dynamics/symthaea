// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Report types for standardized simulation output.

use serde::{Deserialize, Serialize};

/// High-level summary of simulation results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutiveSummary {
    pub title: String,
    pub outcome: String,
    pub key_findings: Vec<String>,
    pub wall_time_seconds: f64,
    pub final_cvs: f64,
    pub final_population: u64,
    pub worlds_surviving: usize,
    pub critical_events: usize,
}

/// Outcome classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimulationOutcome {
    Thrived { peak_cvs: f64 },
    Survived { final_cvs: f64 },
    Collapsed { at_year: f64 },
}

impl std::fmt::Display for SimulationOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimulationOutcome::Thrived { peak_cvs } => {
                write!(f, "Thrived (peak CVS: {:.3})", peak_cvs)
            }
            SimulationOutcome::Survived { final_cvs } => {
                write!(f, "Survived (final CVS: {:.3})", final_cvs)
            }
            SimulationOutcome::Collapsed { at_year } => {
                write!(f, "Collapsed at year {:.0}", at_year)
            }
        }
    }
}

/// Reproducibility metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityInfo {
    pub seed: u64,
    pub total_ticks: u32,
    pub version: String,
    pub config_hash: String,
}
