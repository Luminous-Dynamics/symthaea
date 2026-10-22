// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Neutral physical-computation backend contracts for Symthaea research.
//!
//! This crate deliberately separates **computation backend identity** from
//! `symthaea-core::hdc::substrate_independence::SubstrateType`, which models
//! hypotheses about substrate-dependent cognition/consciousness feasibility.
//! A backend implementing this crate therefore does **not** acquire any
//! consciousness, superiority, efficiency, or hardware-validation claim.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Where an observation came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutionBoundary {
    /// Pure software/reference computation on conventional digital hardware.
    ClassicalReference,
    /// Numerical model of a physical device or material.
    Simulation,
    /// Emulator supplied by a backend/provider but not physical hardware.
    Emulator,
    /// Measurement from a physical device.
    PhysicalDevice,
}

/// The accounting boundary attached to an energy number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EnergyBoundary {
    /// No defensible energy measurement was made.
    Unmeasured,
    /// Energy attributed only to the active device/material element.
    Device,
    /// Energy including control, readout, cooling, lasers, vacuum, or other
    /// infrastructure required for the measurement.
    WholeSystem,
}

/// Strength of evidence attached to an observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvidenceLevel {
    /// A model or hypothesis with no local execution evidence.
    Theoretical,
    /// Deterministic or stochastic software simulation.
    Simulated,
    /// Observation from a provider/device emulator.
    Emulated,
    /// Observation measured on physical hardware.
    Measured,
    /// Physical observation independently replicated or externally validated.
    Replicated,
}

/// Stable identity for a physical-computation backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendIdentity {
    /// Canonical backend family, e.g. `classical:echo-state` or
    /// `physical:gold-nanojunction`.
    pub family: String,
    /// Backend/model version controlled by the implementation.
    pub version: String,
    /// Human-readable implementation name.
    pub implementation: String,
}

impl BackendIdentity {
    /// Construct an identity, rejecting empty or non-canonical fields.
    pub fn new(
        family: impl Into<String>,
        version: impl Into<String>,
        implementation: impl Into<String>,
    ) -> Result<Self, ContractError> {
        let identity = Self {
            family: family.into(),
            version: version.into(),
            implementation: implementation.into(),
        };
        identity.validate()?;
        Ok(identity)
    }

    /// Validate stable identity fields.
    pub fn validate(&self) -> Result<(), ContractError> {
        validate_token("family", &self.family, true)?;
        validate_token("version", &self.version, false)?;
        validate_token("implementation", &self.implementation, true)?;
        Ok(())
    }
}

/// Input sent to a physical-computation backend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControlFrame {
    /// Monotonic logical frame index.
    pub sequence: u64,
    /// Optional logical duration represented by this control in nanoseconds.
    pub duration_ns: Option<u64>,
    /// Ordered scalar controls. Meanings belong to the backend schema.
    pub scalars: BTreeMap<String, f64>,
}

impl ControlFrame {
    /// Construct an empty control frame.
    pub fn new(sequence: u64) -> Self {
        Self {
            sequence,
            duration_ns: None,
            scalars: BTreeMap::new(),
        }
    }

    /// Validate that every scalar is finite and every key is canonical.
    pub fn validate(&self) -> Result<(), ContractError> {
        for (key, value) in &self.scalars {
            validate_token("control key", key, true)?;
            if !value.is_finite() {
                return Err(ContractError::NonFiniteValue(key.clone()));
            }
        }
        Ok(())
    }
}

/// Observation emitted by a backend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationFrame {
    /// Sequence index of the input/control this observation corresponds to.
    pub sequence: u64,
    /// Ordered measured/simulated scalar outputs.
    pub scalars: BTreeMap<String, f64>,
    /// Number of repeated samples/shots contributing to this observation.
    pub samples: Option<u64>,
}

impl ObservationFrame {
    /// Validate output values and metadata.
    pub fn validate(&self) -> Result<(), ContractError> {
        for (key, value) in &self.scalars {
            validate_token("observation key", key, true)?;
            if !value.is_finite() {
                return Err(ContractError::NonFiniteValue(key.clone()));
            }
        }
        if self.samples == Some(0) {
            return Err(ContractError::ZeroSamples);
        }
        Ok(())
    }
}

/// Resource measurements associated with one observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceEnvelope {
    /// Measured wall-clock latency in nanoseconds, if available.
    pub latency_ns: Option<u64>,
    /// Energy in joules under the declared accounting boundary.
    pub energy_joules: Option<f64>,
    /// Boundary for the energy value.
    pub energy_boundary: EnergyBoundary,
    /// Optional number of backend reads/samples performed.
    pub readout_count: Option<u64>,
}

impl Default for ResourceEnvelope {
    fn default() -> Self {
        Self {
            latency_ns: None,
            energy_joules: None,
            energy_boundary: EnergyBoundary::Unmeasured,
            readout_count: None,
        }
    }
}

impl ResourceEnvelope {
    /// Validate resource metadata without inventing missing measurements.
    pub fn validate(&self) -> Result<(), ContractError> {
        match (self.energy_joules, self.energy_boundary) {
            (None, EnergyBoundary::Unmeasured) => {}
            (Some(value), EnergyBoundary::Device | EnergyBoundary::WholeSystem)
                if value.is_finite() && value >= 0.0 => {}
            (Some(_), EnergyBoundary::Unmeasured) | (None, EnergyBoundary::Device | EnergyBoundary::WholeSystem) => {
                return Err(ContractError::EnergyBoundaryMismatch);
            }
            (Some(_), _) => return Err(ContractError::InvalidEnergy),
        }
        if self.readout_count == Some(0) {
            return Err(ContractError::ZeroReadouts);
        }
        Ok(())
    }
}

/// Provenance attached to one backend observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationProvenance {
    /// Backend that produced the observation.
    pub backend: BackendIdentity,
    /// Execution boundary actually used.
    pub execution: ExecutionBoundary,
    /// Evidence strength justified by this execution.
    pub evidence: EvidenceLevel,
    /// Free-form caveats retained with the observation.
    pub caveats: Vec<String>,
}

impl ObservationProvenance {
    /// Enforce the minimum claim boundary implied by the execution source.
    pub fn validate(&self) -> Result<(), ContractError> {
        self.backend.validate()?;
        let allowed = match self.execution {
            ExecutionBoundary::ClassicalReference | ExecutionBoundary::Simulation => {
                matches!(self.evidence, EvidenceLevel::Theoretical | EvidenceLevel::Simulated)
            }
            ExecutionBoundary::Emulator => matches!(
                self.evidence,
                EvidenceLevel::Theoretical | EvidenceLevel::Simulated | EvidenceLevel::Emulated
            ),
            ExecutionBoundary::PhysicalDevice => true,
        };
        if !allowed {
            return Err(ContractError::EvidencePromotion {
                execution: self.execution,
                evidence: self.evidence,
            });
        }
        Ok(())
    }
}

/// Complete result of one backend evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysicalObservation {
    /// Backend output.
    pub observation: ObservationFrame,
    /// Resource measurements.
    pub resources: ResourceEnvelope,
    /// Execution/evidence provenance.
    pub provenance: ObservationProvenance,
}

impl PhysicalObservation {
    /// Validate all nested contract boundaries.
    pub fn validate(&self) -> Result<(), ContractError> {
        self.observation.validate()?;
        self.resources.validate()?;
        self.provenance.validate()?;
        Ok(())
    }
}

/// Minimal backend interface. Implementations may be deterministic or stochastic.
pub trait PhysicalBackend {
    /// Stable backend identity.
    fn identity(&self) -> BackendIdentity;

    /// Reset backend state before a frozen experiment/replay.
    fn reset(&mut self) -> Result<(), ContractError>;

    /// Evaluate one control frame and return a provenance-bearing observation.
    fn evaluate(&mut self, input: &ControlFrame) -> Result<PhysicalObservation, ContractError>;
}

/// Contract-level validation errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractError {
    /// A stable token was empty or contained unsupported characters.
    InvalidToken { field: &'static str, value: String },
    /// A control/observation contained NaN or infinity.
    NonFiniteValue(String),
    /// A sampled observation declared zero samples.
    ZeroSamples,
    /// A resource measurement declared zero readouts.
    ZeroReadouts,
    /// Energy and its accounting boundary disagree.
    EnergyBoundaryMismatch,
    /// Energy was negative or non-finite.
    InvalidEnergy,
    /// Evidence was promoted beyond what the execution boundary supports.
    EvidencePromotion {
        /// Actual execution boundary.
        execution: ExecutionBoundary,
        /// Claimed evidence level.
        evidence: EvidenceLevel,
    },
}

impl std::fmt::Display for ContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidToken { field, value } => write!(f, "invalid {field}: {value:?}"),
            Self::NonFiniteValue(key) => write!(f, "non-finite scalar at {key}"),
            Self::ZeroSamples => write!(f, "sample count must be non-zero when present"),
            Self::ZeroReadouts => write!(f, "readout count must be non-zero when present"),
            Self::EnergyBoundaryMismatch => write!(f, "energy value and accounting boundary disagree"),
            Self::InvalidEnergy => write!(f, "energy must be finite and non-negative"),
            Self::EvidencePromotion { execution, evidence } => write!(
                f,
                "evidence {evidence:?} is not justified by execution boundary {execution:?}"
            ),
        }
    }
}

impl std::error::Error for ContractError {}

fn validate_token(field: &'static str, value: &str, allow_colon: bool) -> Result<(), ContractError> {
    let valid = !value.is_empty()
        && value.len() <= 96
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(byte, b'-' | b'_' | b'.')
                || (allow_colon && byte == b':')
        });
    if valid {
        Ok(())
    } else {
        Err(ContractError::InvalidToken {
            field,
            value: value.to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulation_cannot_claim_measured_evidence() {
        let provenance = ObservationProvenance {
            backend: BackendIdentity::new("physical:gold-nanojunction", "v1", "sim").unwrap(),
            execution: ExecutionBoundary::Simulation,
            evidence: EvidenceLevel::Measured,
            caveats: vec![],
        };
        assert!(matches!(
            provenance.validate(),
            Err(ContractError::EvidencePromotion { .. })
        ));
    }

    #[test]
    fn energy_requires_an_explicit_boundary() {
        let resources = ResourceEnvelope {
            latency_ns: None,
            energy_joules: Some(1e-9),
            energy_boundary: EnergyBoundary::Unmeasured,
            readout_count: None,
        };
        assert_eq!(
            resources.validate(),
            Err(ContractError::EnergyBoundaryMismatch)
        );
    }

    #[test]
    fn canonical_backend_identity_accepts_research_names() {
        let identity = BackendIdentity::new(
            "physical:gold-nanojunction",
            "v1.0",
            "aurum-reservoir-sim",
        )
        .unwrap();
        assert_eq!(identity.family, "physical:gold-nanojunction");
    }

    #[test]
    fn control_frame_rejects_nan() {
        let mut frame = ControlFrame::new(7);
        frame.scalars.insert("drive_voltage".into(), f64::NAN);
        assert_eq!(
            frame.validate(),
            Err(ContractError::NonFiniteValue("drive_voltage".into()))
        );
    }
}
