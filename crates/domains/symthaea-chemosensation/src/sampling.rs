// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Typed acquisition context for chemical observations.
//!
//! Chemical measurements are only interpretable relative to how a sample was
//! acquired. A clean-air baseline, an odor exposure, a tongue rinse, and a
//! recovery measurement may contain similar numeric channels while carrying
//! very different evidentiary meaning. This module records that context without
//! prescribing one universal hardware protocol.

use serde::{Deserialize, Serialize};

/// Acquisition phase associated with a chemical observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SamplingPhase {
    /// Instrument/reference calibration measurement.
    Calibration,
    /// Pre-exposure clean-air, blank, or reference-liquid measurement.
    Baseline,
    /// Target sample is actively presented to the transducer.
    Exposure,
    /// Gas-path purge or clean-air flush.
    Purge,
    /// Liquid-path rinse between samples.
    Rinse,
    /// Post-exposure recovery / desorption / stabilization measurement.
    Recovery,
}

/// Stable metadata that binds one observation to an acquisition protocol.
///
/// Free-form physical parameters (flow rate, heater program, dilution, etc.)
/// intentionally do not live here as untyped floats. Hardware-specific protocol
/// structs can carry those values with explicit units and reference this shared
/// context by `protocol_id` and `run_id`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SamplingContext {
    /// Versioned protocol identifier, e.g. `olfact-od001-v1`.
    pub protocol_id: String,
    /// One concrete execution of the protocol.
    pub run_id: String,
    /// Optional sample/specimen identifier. This is an identifier, not a learned
    /// odor/taste label or asserted chemical identity.
    pub sample_id: Option<String>,
    /// Acquisition phase for this observation.
    pub phase: SamplingPhase,
    /// Monotonic protocol step index. Multiple observations may share one step.
    pub step_index: u32,
    /// Replicate number within the run, when applicable.
    pub replicate: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SamplingContextError {
    EmptyProtocolId,
    EmptyRunId,
    EmptySampleId,
}

impl SamplingContext {
    pub fn new(
        protocol_id: impl Into<String>,
        run_id: impl Into<String>,
        phase: SamplingPhase,
        step_index: u32,
    ) -> Result<Self, SamplingContextError> {
        let protocol_id = protocol_id.into();
        let run_id = run_id.into();
        if protocol_id.trim().is_empty() {
            return Err(SamplingContextError::EmptyProtocolId);
        }
        if run_id.trim().is_empty() {
            return Err(SamplingContextError::EmptyRunId);
        }
        Ok(Self {
            protocol_id,
            run_id,
            sample_id: None,
            phase,
            step_index,
            replicate: 0,
        })
    }

    pub fn with_sample_id(
        mut self,
        sample_id: impl Into<String>,
    ) -> Result<Self, SamplingContextError> {
        let sample_id = sample_id.into();
        if sample_id.trim().is_empty() {
            return Err(SamplingContextError::EmptySampleId);
        }
        self.sample_id = Some(sample_id);
        Ok(self)
    }

    pub fn with_replicate(mut self, replicate: u32) -> Self {
        self.replicate = replicate;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_requires_stable_protocol_and_run_ids() {
        assert_eq!(
            SamplingContext::new("", "run-1", SamplingPhase::Exposure, 1),
            Err(SamplingContextError::EmptyProtocolId)
        );
        assert_eq!(
            SamplingContext::new("od001-v1", "   ", SamplingPhase::Exposure, 1),
            Err(SamplingContextError::EmptyRunId)
        );
    }

    #[test]
    fn sample_identifier_is_optional_but_not_blank() {
        let base = SamplingContext::new(
            "od001-v1",
            "run-1",
            SamplingPhase::Baseline,
            0,
        )
        .unwrap();
        assert!(base.sample_id.is_none());
        assert_eq!(
            base.clone().with_sample_id(" "),
            Err(SamplingContextError::EmptySampleId)
        );
        let exposure = base.with_sample_id("sample-a").unwrap().with_replicate(2);
        assert_eq!(exposure.sample_id.as_deref(), Some("sample-a"));
        assert_eq!(exposure.replicate, 2);
    }
}