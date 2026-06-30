// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Simulation module trait and supporting types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique module identifier (string for human readability).
pub type ModuleId = &'static str;

/// Tick phase ordering for module execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TickPhase {
    Viability = 0,
    Environment = 1,
    Disasters = 2,
    Demographics = 3,
    Economy = 4,
    Consciousness = 5,
    Governance = 6,
    PostProcessing = 7,
}

/// Outputs produced by a module on each tick.
#[derive(Debug, Clone, Default)]
pub struct ModuleOutputs {
    /// Key-value feedback signals available to later-phase modules.
    pub feedback_signals: HashMap<String, f64>,
    /// Named metrics for reporting.
    pub metrics: Vec<(String, f64)>,
    /// Warnings generated this tick.
    pub warnings: Vec<String>,
}

/// A section of the final simulation report contributed by a module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSection {
    pub module_id: String,
    pub module_name: String,
    pub summary: String,
    pub metrics: Vec<(String, f64)>,
    pub warnings: Vec<String>,
}

/// Honest self-assessment from a module.
#[derive(Debug, Clone)]
pub struct ModuleAssessment {
    pub module_id: String,
    pub confidence: String,
    pub limitations: Vec<String>,
}

/// A parameter that can be varied for sensitivity analysis.
#[derive(Debug, Clone)]
pub struct SensitivityParam {
    pub name: String,
    pub description: String,
    pub default_value: f64,
    pub min_value: f64,
    pub max_value: f64,
    pub citation: Option<String>,
}

/// Trait for pluggable simulation modules.
pub trait SimulationModule {
    /// Unique identifier.
    fn id(&self) -> ModuleId;

    /// Human-readable name.
    fn name(&self) -> &str;

    /// Execution phase (determines ordering).
    fn phase(&self) -> TickPhase;

    /// Execute one tick. Receives feedback from earlier-phase modules.
    fn tick(&mut self, tick: u32, feedback: &HashMap<String, f64>)
    -> Result<ModuleOutputs, String>;

    /// Optional report section for the final summary.
    fn report_section(&self) -> Option<ReportSection> {
        None
    }

    /// Optional honest self-assessment of limitations.
    fn honest_assessment(&self) -> Option<ModuleAssessment> {
        None
    }

    /// Optional sensitivity parameters for analysis.
    fn sensitivity_params(&self) -> Vec<SensitivityParam> {
        Vec::new()
    }
}
