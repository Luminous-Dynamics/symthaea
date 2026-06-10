// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Structured telemetry output for crucible runs.

use serde::{Deserialize, Serialize};

/// Telemetry for a single cycle's ethics evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleTelemetry {
    pub cycle: u64,
    pub moral_score: f64,
    pub moral_verdict: String,
    pub consent_violation: bool,
    pub harmony_coordinates: [f64; 8],
    pub moral_free_energy: f64,
    pub anomaly_score: f64,
    /// Which anomaly flags fired this cycle.
    pub anomaly_flags: AnomalyFlags,
}

/// Per-cycle anomaly flag snapshot.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnomalyFlags {
    pub value_inversion: bool,
    pub fe_spike: bool,
    pub fragmentation_increase: bool,
    pub drift_alert: bool,
    pub trajectory_convergence: bool,
}

/// Complete telemetry report for one crucible run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrucibleReport {
    pub scenario_name: String,
    pub total_cycles: u64,
    /// Per-cycle telemetry (only collected at scenario injection points + topology fires).
    pub cycle_telemetry: Vec<CycleTelemetry>,
    /// Harmony coordinate trajectory (sampled at each scenario injection).
    pub harmony_trajectory: Vec<[f64; 8]>,
    /// Free energy trace over the run.
    pub free_energy_trace: Vec<f64>,
    /// Betti number evolution (β₀, β₁, β₂) at each topology fire.
    pub betti_trace: Vec<[usize; 3]>,
    /// Any invariant violations detected during the run.
    pub invariant_violations: Vec<String>,
    /// Summary statistics.
    pub summary: RunSummary,
}

/// Aggregate statistics for a crucible run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub mean_moral_score: f64,
    pub min_moral_score: f64,
    pub max_moral_score: f64,
    pub mean_free_energy: f64,
    pub max_free_energy: f64,
    pub total_anomalies: usize,
    pub consent_violations: usize,
    pub max_anomaly_score: f64,
}

impl CrucibleReport {
    /// Create a new empty report for a scenario.
    pub fn new(scenario_name: impl Into<String>) -> Self {
        Self {
            scenario_name: scenario_name.into(),
            total_cycles: 0,
            cycle_telemetry: Vec::new(),
            harmony_trajectory: Vec::new(),
            free_energy_trace: Vec::new(),
            betti_trace: Vec::new(),
            invariant_violations: Vec::new(),
            summary: RunSummary::default(),
        }
    }

    /// Compute summary statistics from collected telemetry.
    pub fn finalize(&mut self) {
        if self.cycle_telemetry.is_empty() {
            return;
        }

        let scores: Vec<f64> = self.cycle_telemetry.iter().map(|t| t.moral_score).collect();
        let n = scores.len() as f64;

        self.summary.mean_moral_score = scores.iter().sum::<f64>() / n;
        self.summary.min_moral_score = scores.iter().copied().fold(f64::INFINITY, f64::min);
        self.summary.max_moral_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let fes: Vec<f64> = self
            .cycle_telemetry
            .iter()
            .map(|t| t.moral_free_energy)
            .collect();
        if !fes.is_empty() {
            let fn_count = fes.len() as f64;
            self.summary.mean_free_energy = fes.iter().sum::<f64>() / fn_count;
            self.summary.max_free_energy = fes.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        }

        self.summary.total_anomalies = self
            .cycle_telemetry
            .iter()
            .filter(|t| t.anomaly_score > 0.0)
            .count();
        self.summary.consent_violations = self
            .cycle_telemetry
            .iter()
            .filter(|t| t.consent_violation)
            .count();
        self.summary.max_anomaly_score = self
            .cycle_telemetry
            .iter()
            .map(|t| t.anomaly_score)
            .fold(0.0f64, f64::max);
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"))
    }

    /// True if no invariant violations were detected.
    pub fn passed(&self) -> bool {
        self.invariant_violations.is_empty()
    }
}

impl Default for RunSummary {
    fn default() -> Self {
        Self {
            mean_moral_score: 0.0,
            min_moral_score: 0.0,
            max_moral_score: 0.0,
            mean_free_energy: 0.0,
            max_free_energy: 0.0,
            total_anomalies: 0,
            consent_violations: 0,
            max_anomaly_score: 0.0,
        }
    }
}
