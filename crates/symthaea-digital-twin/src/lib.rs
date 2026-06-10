// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Digital twin primitives for engineered systems.
//!
//! This crate stores the small, auditable state Symthaea needs before deeper
//! physics adapters exist: identity, telemetry, health estimates, free-energy
//! trend, and intervention candidates.

#![deny(unsafe_code)]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Stable class of engineered asset being mirrored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AssetClass {
    /// Bridge, building, tunnel, road, or civil structure.
    CivilStructure,
    /// Machine, mechanism, robot, turbine, or thermal device.
    MechanicalSystem,
    /// Circuit, grid, battery, or power electronics system.
    ElectricalSystem,
    /// Aircraft, spacecraft, or launch/orbital support system.
    AerospaceSystem,
    /// Chemical plant, reactor, separator, or process skid.
    ProcessSystem,
    /// Robot, manipulator, mobile platform, or embodied autonomous system.
    RoboticSystem,
    /// Nuclear facility, isotope system, radiation-monitoring asset, or safeguards target.
    NuclearSystem,
    /// Material sample, component coupon, critical mineral asset, or degradation twin.
    MaterialSystem,
    /// Watershed, habitat, carbon asset, climate-exposed infrastructure, or ecological system.
    EnvironmentalSystem,
    /// Cross-domain system of systems.
    SystemOfSystems,
}

/// Single telemetry reading with unit and timestamp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetryPoint {
    /// Sensor or derived metric name.
    pub channel: String,
    /// Numeric value.
    pub value: f64,
    /// Unit string in the originating system.
    pub unit: String,
    /// UTC timestamp for the reading.
    pub timestamp: DateTime<Utc>,
}

/// Difference between an expected and actual telemetry reading.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredictionResidual {
    /// Channel evaluated.
    pub channel: String,
    /// Predicted nominal value.
    pub predicted: f64,
    /// Observed value.
    pub actual: f64,
    /// Absolute prediction error normalized by tolerance.
    pub normalized_error: f64,
    /// Epistemic uncertainty after this observation.
    pub epistemic_uncertainty: f64,
    /// Aleatoric uncertainty attached to this observation.
    pub aleatoric_uncertainty: f64,
    /// Timestamp copied from the observation.
    pub timestamp: DateTime<Utc>,
}

/// Decomposed free-energy proxy for digital-twin updates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FreeEnergyBreakdown {
    /// Prediction error / negative accuracy contribution.
    pub accuracy_loss: f64,
    /// Penalty for how much the twin state had to move.
    pub complexity_penalty: f64,
    /// Expected information gain from acting or collecting more data.
    pub epistemic_value: f64,
    /// Risk of violating health or operational preferences.
    pub pragmatic_risk: f64,
    /// Expected free energy for policy ranking.
    pub expected_free_energy: f64,
    /// Current variational free-energy proxy.
    pub total: f64,
}

impl Default for FreeEnergyBreakdown {
    fn default() -> Self {
        Self {
            accuracy_loss: 0.0,
            complexity_penalty: 0.0,
            epistemic_value: 0.0,
            pragmatic_risk: 0.0,
            expected_free_energy: 0.0,
            total: 0.0,
        }
    }
}

impl TelemetryPoint {
    /// Construct a telemetry point stamped with the current UTC time.
    pub fn now(channel: impl Into<String>, value: f64, unit: impl Into<String>) -> Self {
        Self {
            channel: channel.into(),
            value,
            unit: unit.into(),
            timestamp: Utc::now(),
        }
    }
}

/// Compact live state for an engineered asset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TwinState {
    /// Stable twin identifier.
    pub id: String,
    /// Asset class.
    pub class: AssetClass,
    /// Human-readable asset label.
    pub label: String,
    /// Estimated health from 0.0 to 1.0.
    pub health: f64,
    /// Current free-energy/surprise proxy. Higher means less expected state.
    pub free_energy: f64,
    /// Decomposed free-energy terms from the latest update.
    #[serde(default)]
    pub free_energy_breakdown: FreeEnergyBreakdown,
    /// Reducible/model uncertainty, 0.0 certain to 1.0 unknown.
    #[serde(default = "default_uncertainty")]
    pub epistemic_uncertainty: f64,
    /// Irreducible/noise uncertainty, 0.0 deterministic to 1.0 noisy.
    #[serde(default = "default_uncertainty")]
    pub aleatoric_uncertainty: f64,
    /// History of expected-vs-actual residuals.
    #[serde(default)]
    pub prediction_residuals: Vec<PredictionResidual>,
    /// Most recent telemetry slice.
    pub telemetry: Vec<TelemetryPoint>,
}

impl TwinState {
    /// Construct a healthy twin with no telemetry.
    pub fn new(id: impl Into<String>, class: AssetClass, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            class,
            label: label.into(),
            health: 1.0,
            free_energy: 0.0,
            free_energy_breakdown: FreeEnergyBreakdown::default(),
            epistemic_uncertainty: 0.5,
            aleatoric_uncertainty: 0.1,
            prediction_residuals: Vec::new(),
            telemetry: Vec::new(),
        }
    }

    /// Ingest telemetry and update a bounded free-energy proxy.
    pub fn ingest(&mut self, point: TelemetryPoint, expected_value: f64, tolerance: f64) {
        self.ingest_with_uncertainty(point, expected_value, tolerance, 0.1);
    }

    /// Ingest telemetry with an explicit aleatoric uncertainty estimate.
    pub fn ingest_with_uncertainty(
        &mut self,
        point: TelemetryPoint,
        expected_value: f64,
        tolerance: f64,
        aleatoric_uncertainty: f64,
    ) {
        let normalized_error = if tolerance.abs() <= f64::EPSILON {
            0.0
        } else {
            ((point.value - expected_value) / tolerance).abs()
        };
        let previous_free_energy = self.free_energy;
        let previous_epistemic = self.epistemic_uncertainty;
        let new_epistemic = (0.85 * self.epistemic_uncertainty
            + 0.15 * (normalized_error / 3.0).clamp(0.0, 1.0))
        .clamp(0.0, 1.0);
        let new_aleatoric = (0.85 * self.aleatoric_uncertainty
            + 0.15 * aleatoric_uncertainty.clamp(0.0, 1.0))
        .clamp(0.0, 1.0);

        self.free_energy = 0.8 * self.free_energy + 0.2 * normalized_error;
        self.health = (1.0 / (1.0 + self.free_energy)).clamp(0.0, 1.0);
        self.epistemic_uncertainty = new_epistemic;
        self.aleatoric_uncertainty = new_aleatoric;

        let complexity_penalty = (self.free_energy - previous_free_energy).abs();
        let epistemic_value = (previous_epistemic - new_epistemic).max(0.0);
        let pragmatic_risk = 1.0 - self.health;
        self.free_energy_breakdown = FreeEnergyBreakdown {
            accuracy_loss: normalized_error,
            complexity_penalty,
            epistemic_value,
            pragmatic_risk,
            expected_free_energy: (pragmatic_risk + new_aleatoric - epistemic_value).max(0.0),
            total: self.free_energy,
        };

        self.prediction_residuals.push(PredictionResidual {
            channel: point.channel.clone(),
            predicted: expected_value,
            actual: point.value,
            normalized_error,
            epistemic_uncertainty: new_epistemic,
            aleatoric_uncertainty: new_aleatoric,
            timestamp: point.timestamp,
        });
        self.telemetry.push(point);
    }

    /// Returns true when the twin should request simulation, inspection, or data.
    pub fn needs_intervention(&self, free_energy_threshold: f64) -> bool {
        self.free_energy >= free_energy_threshold
    }

    /// Expected free energy used to rank intervention or data-gathering policies.
    pub fn expected_free_energy(&self) -> f64 {
        self.free_energy_breakdown.expected_free_energy
    }
}

fn default_uncertainty() -> f64 {
    0.5
}

/// Intervention candidate produced by reasoning or monitoring.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Intervention {
    /// Target twin id.
    pub twin_id: String,
    /// Action class, such as `inspect`, `simulate`, `derate`, or `repair`.
    pub action: String,
    /// Why this action is being proposed.
    pub rationale: String,
    /// Urgency from 0.0 to 1.0.
    pub urgency: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ingest_updates_free_energy_and_health() {
        let mut twin = TwinState::new("asset-1", AssetClass::CivilStructure, "test span");
        twin.ingest(
            TelemetryPoint::now("strain", 125.0, "microstrain"),
            100.0,
            10.0,
        );

        assert!(twin.free_energy > 0.0);
        assert!(twin.health < 1.0);
        assert_eq!(twin.telemetry.len(), 1);
        assert_eq!(twin.prediction_residuals.len(), 1);
        assert!(twin.free_energy_breakdown.accuracy_loss > 0.0);
        assert!(twin.expected_free_energy().is_finite());
    }

    #[test]
    fn stable_observations_reduce_epistemic_uncertainty() {
        let mut twin = TwinState::new("asset-2", AssetClass::MechanicalSystem, "test actuator");
        let initial = twin.epistemic_uncertainty;
        for _ in 0..5 {
            twin.ingest_with_uncertainty(
                TelemetryPoint::now("position", 10.0, "mm"),
                10.0,
                1.0,
                0.05,
            );
        }

        assert!(twin.epistemic_uncertainty < initial);
        assert!(twin.free_energy_breakdown.epistemic_value >= 0.0);
    }
}
