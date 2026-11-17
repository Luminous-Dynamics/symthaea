// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-typed plasma observations and decision-input abstention.
//!
//! Prevents unavailable, delayed, degraded, invalid, or synthetic diagnostic
//! values from being silently treated as direct measurements. This is an
//! input-contract gate, not a plasma estimator or hardware safety interlock.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const PLASMA_OBSERVATION_SCHEMA: &str = "luminous.plasma-observation-frame.v0.1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationIdentity {
    pub calibration_id: String,
    pub calibration_revision: String,
}

impl CalibrationIdentity {
    fn validate(&self, channel_id: &str) -> Result<(), ObservationContractError> {
        if self.calibration_id.trim().is_empty() || self.calibration_revision.trim().is_empty() {
            return Err(ObservationContractError::new(format!(
                "channel {channel_id} has incomplete calibration identity"
            )));
        }
        Ok(())
    }
}

fn validate_sha256(value: &str, field: &str, channel_id: &str) -> Result<(), ObservationContractError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ObservationContractError::new(format!(
            "channel {channel_id} {field} must be 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservationContractError {
    pub message: String,
}

impl ObservationContractError {
    fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

impl std::fmt::Display for ObservationContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ObservationContractError {}

/// A channel's evidence disposition at a declared frame time.
/// Synthetic reconstruction is a separate variant from direct measurement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum ObservationDisposition {
    Measured {
        value: f64,
        units: String,
        source_id: String,
        source_digest: String,
        calibration: CalibrationIdentity,
        observed_at_ns: u64,
    },
    Synthetic {
        value: f64,
        units: String,
        model_id: String,
        model_revision: String,
        input_digest: String,
    },
    Unavailable { reason: String },
    Invalid { reason: String },
    Delayed {
        value: f64,
        units: String,
        source_id: String,
        source_digest: String,
        calibration: CalibrationIdentity,
        observed_at_ns: u64,
        latency_ns: u64,
    },
    Degraded {
        value: f64,
        units: String,
        source_id: String,
        source_digest: String,
        calibration: CalibrationIdentity,
        observed_at_ns: u64,
        reason: String,
    },
}

impl ObservationDisposition {
    fn validate(
        &self,
        frame_time_ns: u64,
        channel_id: &str,
    ) -> Result<(), ObservationContractError> {
        let finite_value = |value: f64| {
            if value.is_finite() {
                Ok(())
            } else {
                Err(ObservationContractError::new(format!(
                    "channel {channel_id} has a non-finite value"
                )))
            }
        };
        let non_empty = |value: &str, field: &str| {
            if value.trim().is_empty() {
                Err(ObservationContractError::new(format!(
                    "channel {channel_id} has an empty {field}"
                )))
            } else {
                Ok(())
            }
        };
        let check_timestamp = |observed_at_ns: u64| {
            if observed_at_ns > frame_time_ns {
                Err(ObservationContractError::new(format!(
                    "channel {channel_id} observation timestamp is after the frame time"
                )))
            } else {
                Ok(())
            }
        };

        match self {
            Self::Measured {
                value,
                units,
                source_id,
                source_digest,
                calibration,
                observed_at_ns,
            } => {
                finite_value(*value)?;
                non_empty(units, "units")?;
                non_empty(source_id, "source_id")?;
                validate_sha256(source_digest, "source_digest", channel_id)?;
                calibration.validate(channel_id)?;
                check_timestamp(*observed_at_ns)?;
            }
            Self::Synthetic {
                value,
                units,
                model_id,
                model_revision,
                input_digest,
            } => {
                finite_value(*value)?;
                non_empty(units, "units")?;
                non_empty(model_id, "model_id")?;
                non_empty(model_revision, "model_revision")?;
                if input_digest.len() != 64
                    || !input_digest
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                {
                    return Err(ObservationContractError::new(format!(
                        "channel {channel_id} synthetic input_digest must be 64 lowercase hexadecimal characters"
                    )));
                }
            }
            Self::Unavailable { reason } | Self::Invalid { reason } => {
                non_empty(reason, "reason")?;
            }
            Self::Delayed {
                value,
                units,
                source_id,
                source_digest,
                calibration,
                observed_at_ns,
                latency_ns,
            } => {
                finite_value(*value)?;
                non_empty(units, "units")?;
                non_empty(source_id, "source_id")?;
                validate_sha256(source_digest, "source_digest", channel_id)?;
                calibration.validate(channel_id)?;
                check_timestamp(*observed_at_ns)?;
                let expected_latency = frame_time_ns - observed_at_ns;
                if *latency_ns == 0 || *latency_ns != expected_latency {
                    return Err(ObservationContractError::new(format!(
                        "channel {channel_id} delayed latency does not match frame_time_ns - observed_at_ns"
                    )));
                }
            }
            Self::Degraded {
                value,
                units,
                source_id,
                source_digest,
                calibration,
                observed_at_ns,
                reason,
            } => {
                finite_value(*value)?;
                non_empty(units, "units")?;
                non_empty(source_id, "source_id")?;
                non_empty(reason, "reason")?;
                validate_sha256(source_digest, "source_digest", channel_id)?;
                calibration.validate(channel_id)?;
                check_timestamp(*observed_at_ns)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticChannelObservation {
    pub channel_id: String,
    pub diagnostic_system_id: String,
    pub disposition: ObservationDisposition,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlasmaObservationFrame {
    pub schema_version: String,
    pub shot_id: String,
    pub frame_id: String,
    pub clock_domain: String,
    pub frame_time_ns: u64,
    /// Exact expected inventory. Unavailable channels must still be explicit.
    pub expected_channel_ids: Vec<String>,
    pub channels: Vec<DiagnosticChannelObservation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObservationCounts {
    pub measured: usize,
    pub synthetic: usize,
    pub unavailable: usize,
    pub invalid: usize,
    pub delayed: usize,
    pub degraded: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbstentionReason {
    SyntheticNotMeasurement,
    Unavailable,
    Invalid,
    Delayed,
    Degraded,
    WrongUnits,
    Stale,
    UnapprovedSource,
    CalibrationMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockedChannel {
    pub channel_id: String,
    pub reason: AbstentionReason,
}

/// Requirements for admitting one measured channel to a specific decision.
///
/// This gate checks measurement provenance and time validity only. Passing it
/// does not prove that the downstream physical quantity is observable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionChannelRequirement {
    pub channel_id: String,
    pub expected_units: String,
    pub max_age_ns: u64,
    pub allowed_source_ids: Vec<String>,
    pub calibration: CalibrationIdentity,
}

impl DecisionChannelRequirement {
    fn validate(&self) -> Result<(), ObservationContractError> {
        if self.channel_id.trim().is_empty()
            || self.expected_units.trim().is_empty()
            || self.max_age_ns == 0
            || self.allowed_source_ids.is_empty()
            || self.allowed_source_ids.iter().any(|source| source.trim().is_empty())
        {
            return Err(ObservationContractError::new(
                "decision channel requirement is incomplete",
            ));
        }
        self.calibration.validate(&self.channel_id)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasuredValue {
    channel_id: String,
    diagnostic_system_id: String,
    value: f64,
    units: String,
    source_id: String,
    source_digest: String,
    calibration_id: String,
    calibration_revision: String,
    observed_at_ns: u64,
}

impl MeasuredValue {
    pub fn channel_id(&self) -> &str {
        &self.channel_id
    }

    pub fn diagnostic_system_id(&self) -> &str {
        &self.diagnostic_system_id
    }

    pub fn value(&self) -> f64 {
        self.value
    }

    pub fn units(&self) -> &str {
        &self.units
    }

    pub fn source_id(&self) -> &str {
        &self.source_id
    }

    pub fn source_digest(&self) -> &str {
        &self.source_digest
    }

    pub fn calibration_id(&self) -> &str {
        &self.calibration_id
    }

    pub fn calibration_revision(&self) -> &str {
        &self.calibration_revision
    }

    pub fn observed_at_ns(&self) -> u64 {
        self.observed_at_ns
    }
}

/// Decision values admitted only after every required channel was verified as
/// directly measured. Private fields prevent construction around the gate.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasuredDecisionInputs {
    frame_id: String,
    shot_id: String,
    frame_time_ns: u64,
    values: Vec<MeasuredValue>,
}

impl MeasuredDecisionInputs {
    pub fn frame_id(&self) -> &str {
        &self.frame_id
    }

    pub fn shot_id(&self) -> &str {
        &self.shot_id
    }

    pub fn frame_time_ns(&self) -> u64 {
        self.frame_time_ns
    }

    pub fn values(&self) -> &[MeasuredValue] {
        &self.values
    }

    pub fn get(&self, channel_id: &str) -> Option<&MeasuredValue> {
        self.values
            .binary_search_by(|value| value.channel_id.as_str().cmp(channel_id))
            .ok()
            .map(|index| &self.values[index])
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Result of a measurement-availability check.
///
/// MeasuredInputsAvailable means only that every declared required channel
/// is represented by a validated direct measurement. It does NOT prove that
/// the physical quantity of interest is identifiable, observable, calibrated
/// for a machine/regime, or safe to use for actuation.
pub enum DecisionInputGate {
    MeasuredInputsAvailable(MeasuredDecisionInputs),
    Abstain {
        frame_id: String,
        blocked_channels: Vec<BlockedChannel>,
    },
}

impl PlasmaObservationFrame {
    /// Validate schema, exact channel inventory, evidence metadata, timestamps,
    /// and finite numeric values.
    pub fn validate(&self) -> Result<(), ObservationContractError> {
        if self.schema_version != PLASMA_OBSERVATION_SCHEMA {
            return Err(ObservationContractError::new(
                "unsupported plasma observation schema version",
            ));
        }
        for (value, field) in [
            (self.shot_id.as_str(), "shot_id"),
            (self.frame_id.as_str(), "frame_id"),
            (self.clock_domain.as_str(), "clock_domain"),
        ] {
            if value.trim().is_empty() {
                return Err(ObservationContractError::new(format!(
                    "{field} must be non-empty"
                )));
            }
        }
        if self.expected_channel_ids.is_empty() {
            return Err(ObservationContractError::new(
                "expected_channel_ids must not be empty",
            ));
        }

        let mut expected = BTreeSet::new();
        for channel_id in &self.expected_channel_ids {
            if channel_id.trim().is_empty() {
                return Err(ObservationContractError::new(
                    "expected channel ID must be non-empty",
                ));
            }
            if !expected.insert(channel_id.as_str()) {
                return Err(ObservationContractError::new(format!(
                    "duplicate expected channel ID: {channel_id}"
                )));
            }
        }

        let mut actual = BTreeSet::new();
        for channel in &self.channels {
            if channel.channel_id.trim().is_empty()
                || channel.diagnostic_system_id.trim().is_empty()
            {
                return Err(ObservationContractError::new(
                    "channel_id and diagnostic_system_id must be non-empty",
                ));
            }
            if !actual.insert(channel.channel_id.as_str()) {
                return Err(ObservationContractError::new(format!(
                    "duplicate observed channel ID: {}",
                    channel.channel_id
                )));
            }
            if !expected.contains(channel.channel_id.as_str()) {
                return Err(ObservationContractError::new(format!(
                    "unexpected channel in frame: {}",
                    channel.channel_id
                )));
            }
            channel
                .disposition
                .validate(self.frame_time_ns, &channel.channel_id)?;
        }

        let missing: Vec<&str> = expected.difference(&actual).copied().collect();
        if !missing.is_empty() {
            return Err(ObservationContractError::new(format!(
                "frame omitted expected channel(s): {}",
                missing.join(",")
            )));
        }
        Ok(())
    }

    pub fn counts(&self) -> Result<ObservationCounts, ObservationContractError> {
        self.validate()?;
        let mut counts = ObservationCounts {
            measured: 0,
            synthetic: 0,
            unavailable: 0,
            invalid: 0,
            delayed: 0,
            degraded: 0,
        };
        for channel in &self.channels {
            match &channel.disposition {
                ObservationDisposition::Measured { .. } => counts.measured += 1,
                ObservationDisposition::Synthetic { .. } => counts.synthetic += 1,
                ObservationDisposition::Unavailable { .. } => counts.unavailable += 1,
                ObservationDisposition::Invalid { .. } => counts.invalid += 1,
                ObservationDisposition::Delayed { .. } => counts.delayed += 1,
                ObservationDisposition::Degraded { .. } => counts.degraded += 1,
            }
        }
        Ok(counts)
    }

    /// Admit a decision only when every required channel is directly measured
    /// and matches its declared units, freshness, source and calibration.
    ///
    /// This is a measurement-quality gate. Even a returned
    /// MeasuredInputsAvailable result does not establish physical
    /// identifiability or observability; those require separate analysis.
    pub fn evaluate_decision_requirements(
        &self,
        requirements: &[DecisionChannelRequirement],
    ) -> Result<DecisionInputGate, ObservationContractError> {
        if requirements.is_empty() {
            return Err(ObservationContractError::new(
                "a decision must declare at least one channel requirement",
            ));
        }
        for requirement in requirements {
            requirement.validate()?;
        }

        let mut seen = BTreeSet::new();
        for requirement in requirements {
            if !seen.insert(requirement.channel_id.as_str()) {
                return Err(ObservationContractError::new(format!(
                    "duplicate decision requirement: {}",
                    requirement.channel_id
                )));
            }
        }

        let required_ids: Vec<&str> = requirements
            .iter()
            .map(|requirement| requirement.channel_id.as_str())
            .collect();
        let gate = self.evaluate_decision_gate(&required_ids)?;
        let DecisionInputGate::MeasuredInputsAvailable(inputs) = gate else {
            return Ok(gate);
        };

        let requirements_by_id: BTreeMap<&str, &DecisionChannelRequirement> = requirements
            .iter()
            .map(|requirement| (requirement.channel_id.as_str(), requirement))
            .collect();
        let mut blocked_channels = Vec::new();

        for value in inputs.values() {
            let requirement = requirements_by_id
                .get(value.channel_id())
                .copied()
                .ok_or_else(|| {
                    ObservationContractError::new(format!(
                        "missing requirement for measured channel {}",
                        value.channel_id()
                    ))
                })?;

            let reason = if value.units() != requirement.expected_units {
                Some(AbstentionReason::WrongUnits)
            } else if self.frame_time_ns.saturating_sub(value.observed_at_ns())
                > requirement.max_age_ns
            {
                Some(AbstentionReason::Stale)
            } else if !requirement
                .allowed_source_ids
                .iter()
                .any(|source_id| source_id == value.source_id())
            {
                Some(AbstentionReason::UnapprovedSource)
            } else if value.calibration_id() != requirement.calibration.calibration_id
                || value.calibration_revision() != requirement.calibration.calibration_revision
            {
                Some(AbstentionReason::CalibrationMismatch)
            } else {
                None
            };

            if let Some(reason) = reason {
                blocked_channels.push(BlockedChannel {
                    channel_id: value.channel_id().to_owned(),
                    reason,
                });
            }
        }

        if blocked_channels.is_empty() {
            Ok(DecisionInputGate::MeasuredInputsAvailable(inputs))
        } else {
            Ok(DecisionInputGate::Abstain {
                frame_id: self.frame_id.clone(),
                blocked_channels,
            })
        }
    }

    /// Admit a decision only when all required channels are directly measured.
    /// Optional channels are not included in the projection.
    pub fn evaluate_decision_gate(
        &self,
        required_channel_ids: &[&str],
    ) -> Result<DecisionInputGate, ObservationContractError> {
        self.validate()?;
        if required_channel_ids.is_empty() {
            return Err(ObservationContractError::new(
                "a decision must declare at least one required channel",
            ));
        }

        let channels: BTreeMap<&str, &DiagnosticChannelObservation> = self
            .channels
            .iter()
            .map(|channel| (channel.channel_id.as_str(), channel))
            .collect();
        let mut required = BTreeSet::new();
        for channel_id in required_channel_ids {
            if channel_id.trim().is_empty() {
                return Err(ObservationContractError::new(
                    "required channel ID must be non-empty",
                ));
            }
            if !required.insert(*channel_id) {
                return Err(ObservationContractError::new(format!(
                    "duplicate required channel ID: {channel_id}"
                )));
            }
            if !channels.contains_key(channel_id) {
                return Err(ObservationContractError::new(format!(
                    "decision requires undeclared channel: {channel_id}"
                )));
            }
        }

        let mut values = Vec::with_capacity(required.len());
        let mut blocked_channels = Vec::new();
        for channel_id in required {
            let channel = channels
                .get(channel_id)
                .copied()
                .ok_or_else(|| {
                    ObservationContractError::new(format!(
                        "required channel disappeared during projection: {channel_id}"
                    ))
                })?;
            match &channel.disposition {
                ObservationDisposition::Measured {
                    value,
                    units,
                    source_id,
                    source_digest,
                    calibration,
                    observed_at_ns,
                } => values.push(MeasuredValue {
                    channel_id: channel.channel_id.clone(),
                    diagnostic_system_id: channel.diagnostic_system_id.clone(),
                    value: *value,
                    units: units.clone(),
                    source_id: source_id.clone(),
                    source_digest: source_digest.clone(),
                    calibration_id: calibration.calibration_id.clone(),
                    calibration_revision: calibration.calibration_revision.clone(),
                    observed_at_ns: *observed_at_ns,
                }),
                disposition => blocked_channels.push(BlockedChannel {
                    channel_id: channel.channel_id.clone(),
                    reason: match disposition {
                        ObservationDisposition::Synthetic { .. } => {
                            AbstentionReason::SyntheticNotMeasurement
                        }
                        ObservationDisposition::Unavailable { .. } => {
                            AbstentionReason::Unavailable
                        }
                        ObservationDisposition::Invalid { .. } => AbstentionReason::Invalid,
                        ObservationDisposition::Delayed { .. } => AbstentionReason::Delayed,
                        ObservationDisposition::Degraded { .. } => AbstentionReason::Degraded,
                        ObservationDisposition::Measured { .. } => unreachable!(),
                    },
                }),
            }
        }

        if !blocked_channels.is_empty() {
            return Ok(DecisionInputGate::Abstain {
                frame_id: self.frame_id.clone(),
                blocked_channels,
            });
        }

        values.sort_by(|left, right| left.channel_id.cmp(&right.channel_id));
        Ok(DecisionInputGate::MeasuredInputsAvailable(MeasuredDecisionInputs {
            frame_id: self.frame_id.clone(),
            shot_id: self.shot_id.clone(),
            frame_time_ns: self.frame_time_ns,
            values,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measured(value: f64, observed_at_ns: u64) -> ObservationDisposition {
        ObservationDisposition::Measured {
            value,
            units: "10^20 m^-3".to_owned(),
            source_id: "interferometer-01".to_owned(),
            source_digest: "d".repeat(64),
            calibration: CalibrationIdentity {
                calibration_id: "calibration-density-01".to_owned(),
                calibration_revision: "2026-11-01".to_owned(),
            },
            observed_at_ns,
        }
    }

    fn channel(
        channel_id: &str,
        system_id: &str,
        disposition: ObservationDisposition,
    ) -> DiagnosticChannelObservation {
        DiagnosticChannelObservation {
            channel_id: channel_id.to_owned(),
            diagnostic_system_id: system_id.to_owned(),
            disposition,
        }
    }

    fn base_frame(channels: Vec<DiagnosticChannelObservation>) -> PlasmaObservationFrame {
        PlasmaObservationFrame {
            schema_version: PLASMA_OBSERVATION_SCHEMA.to_owned(),
            shot_id: "synthetic-shot-001".to_owned(),
            frame_id: "frame-001".to_owned(),
            clock_domain: "shot-relative-ns".to_owned(),
            frame_time_ns: 100,
            expected_channel_ids: vec!["density".to_owned(), "temperature".to_owned()],
            channels,
        }
    }

    #[test]
    fn measured_zero_is_a_real_value_not_missingness() {
        let frame = base_frame(vec![
            channel("density", "interferometer", measured(0.0, 100)),
            channel(
                "temperature",
                "thomson",
                ObservationDisposition::Measured {
                    value: 4.0,
                    units: "keV".to_owned(),
                    source_id: "thomson-01".to_owned(),
                    source_digest: "e".repeat(64),
                    calibration: CalibrationIdentity {
                        calibration_id: "calibration-temperature-01".to_owned(),
                        calibration_revision: "2026-11-01".to_owned(),
                    },
                    observed_at_ns: 100,
                },
            ),
        ]);
        let gate = frame
            .evaluate_decision_gate(&["density", "temperature"])
            .expect("valid frame should evaluate");
        let DecisionInputGate::MeasuredInputsAvailable(inputs) = gate else {
            panic!("measured channels should be available");
        };
        assert_eq!(inputs.get("density").expect("density").value(), 0.0);
        assert_eq!(inputs.values().len(), 2);
    }

    #[test]
    fn unavailable_channel_causes_abstention_not_zero_imputation() {
        let frame = base_frame(vec![
            channel(
                "density",
                "interferometer",
                ObservationDisposition::Unavailable {
                    reason: "acquisition dropout".to_owned(),
                },
            ),
            channel("temperature", "thomson", measured(4.0, 100)),
        ]);
        let gate = frame
            .evaluate_decision_gate(&["density", "temperature"])
            .expect("valid frame should evaluate");
        assert_eq!(
            gate,
            DecisionInputGate::Abstain {
                frame_id: "frame-001".to_owned(),
                blocked_channels: vec![BlockedChannel {
                    channel_id: "density".to_owned(),
                    reason: AbstentionReason::Unavailable,
                }],
            }
        );
    }

    #[test]
    fn synthetic_reconstruction_never_satisfies_measured_gate() {
        let frame = base_frame(vec![
            channel(
                "density",
                "interferometer",
                ObservationDisposition::Synthetic {
                    value: 0.9,
                    units: "10^20 m^-3".to_owned(),
                    model_id: "diag-recovery".to_owned(),
                    model_revision: "abc123".to_owned(),
                    input_digest: "a".repeat(64),
                },
            ),
            channel("temperature", "thomson", measured(4.0, 100)),
        ]);
        let gate = frame
            .evaluate_decision_gate(&["density", "temperature"])
            .expect("valid frame should evaluate");
        let DecisionInputGate::Abstain { blocked_channels, .. } = gate else {
            panic!("synthetic density must not be admitted as measured");
        };
        assert_eq!(
            blocked_channels[0].reason,
            AbstentionReason::SyntheticNotMeasurement
        );
    }

    #[test]
    fn delayed_and_degraded_channels_are_not_silently_admitted() {
        let frame = base_frame(vec![
            channel(
                "density",
                "interferometer",
                ObservationDisposition::Delayed {
                    value: 0.9,
                    units: "10^20 m^-3".to_owned(),
                    source_id: "interferometer-01".to_owned(),
                    source_digest: "f".repeat(64),
                    calibration: CalibrationIdentity {
                        calibration_id: "calibration-density-01".to_owned(),
                        calibration_revision: "2026-11-01".to_owned(),
                    },
                    observed_at_ns: 90,
                    latency_ns: 10,
                },
            ),
            channel(
                "temperature",
                "thomson",
                ObservationDisposition::Degraded {
                    value: 4.0,
                    units: "keV".to_owned(),
                    source_id: "thomson-01".to_owned(),
                    source_digest: "1".repeat(64),
                    calibration: CalibrationIdentity {
                        calibration_id: "calibration-temperature-01".to_owned(),
                        calibration_revision: "2026-11-01".to_owned(),
                    },
                    observed_at_ns: 100,
                    reason: "calibration drift suspected".to_owned(),
                },
            ),
        ]);
        let gate = frame
            .evaluate_decision_gate(&["density", "temperature"])
            .expect("valid frame should evaluate");
        let DecisionInputGate::Abstain { blocked_channels, .. } = gate else {
            panic!("delayed and degraded channels must block strict measured gate");
        };
        assert_eq!(blocked_channels.len(), 2);
        assert_eq!(blocked_channels[0].reason, AbstentionReason::Delayed);
        assert_eq!(blocked_channels[1].reason, AbstentionReason::Degraded);
    }

    #[test]
    fn synthetic_optional_channel_does_not_pollute_measured_projection() {
        let frame = PlasmaObservationFrame {
            expected_channel_ids: vec![
                "density".to_owned(),
                "temperature".to_owned(),
                "radiated_power".to_owned(),
            ],
            channels: vec![
                channel("density", "interferometer", measured(0.9, 100)),
                channel(
                    "temperature",
                    "thomson",
                    ObservationDisposition::Measured {
                        value: 4.0,
                        units: "keV".to_owned(),
                        source_id: "thomson-01".to_owned(),
                        source_digest: "2".repeat(64),
                        calibration: CalibrationIdentity {
                            calibration_id: "calibration-temperature-01".to_owned(),
                            calibration_revision: "2026-11-01".to_owned(),
                        },
                        observed_at_ns: 100,
                    },
                ),
                channel(
                    "radiated_power",
                    "bolometer",
                    ObservationDisposition::Synthetic {
                        value: 3.0,
                        units: "MW".to_owned(),
                        model_id: "diag-recovery".to_owned(),
                        model_revision: "abc123".to_owned(),
                        input_digest: "b".repeat(64),
                    },
                ),
            ],
            ..base_frame(Vec::new())
        };
        let gate = frame
            .evaluate_decision_gate(&["temperature", "density"])
            .expect("valid required channels should evaluate");
        let DecisionInputGate::MeasuredInputsAvailable(inputs) = gate else {
            panic!("optional synthetic channel must not block required measured-input availability");
        };
        assert_eq!(inputs.values().len(), 2);
        assert!(inputs.get("radiated_power").is_none());
    }

    #[test]
    fn frame_inventory_must_be_exact() {
        let missing = base_frame(vec![channel(
            "density",
            "interferometer",
            measured(0.9, 100),
        )]);
        assert!(missing.validate().is_err());

        let duplicate = base_frame(vec![
            channel("density", "interferometer", measured(0.9, 100)),
            channel("density", "interferometer-2", measured(0.8, 100)),
        ]);
        assert!(duplicate.validate().is_err());
    }

    #[test]
    fn non_finite_measured_values_are_rejected() {
        let frame = base_frame(vec![
            channel("density", "interferometer", measured(f64::NAN, 100)),
            channel("temperature", "thomson", measured(4.0, 100)),
        ]);
        assert!(frame.validate().is_err());
    }

    #[test]
    fn future_timestamps_and_inconsistent_delay_metadata_are_rejected() {
        let future = base_frame(vec![
            channel("density", "interferometer", measured(0.9, 101)),
            channel("temperature", "thomson", measured(4.0, 100)),
        ]);
        assert!(future.validate().is_err());

        let bad_delay = base_frame(vec![
            channel(
                "density",
                "interferometer",
                ObservationDisposition::Delayed {
                    value: 0.9,
                    units: "10^20 m^-3".to_owned(),
                    source_id: "interferometer-01".to_owned(),
                    source_digest: "f".repeat(64),
                    calibration: CalibrationIdentity {
                        calibration_id: "calibration-density-01".to_owned(),
                        calibration_revision: "2026-11-01".to_owned(),
                    },
                    observed_at_ns: 90,
                    latency_ns: 5,
                },
            ),
            channel("temperature", "thomson", measured(4.0, 100)),
        ]);
        assert!(bad_delay.validate().is_err());
    }

    #[test]
    fn undeclared_or_duplicate_decision_channels_are_rejected() {
        let frame = base_frame(vec![
            channel("density", "interferometer", measured(0.9, 100)),
            channel("temperature", "thomson", measured(4.0, 100)),
        ]);
        assert!(frame.evaluate_decision_gate(&["missing"]).is_err());
        assert!(frame.evaluate_decision_gate(&["density", "density"]).is_err());
    }

    #[test]
    fn measurement_source_digest_and_calibration_identity_are_mandatory() {
        let missing_digest = base_frame(vec![
            channel(
                "density",
                "interferometer",
                ObservationDisposition::Measured {
                    value: 0.9,
                    units: "10^20 m^-3".to_owned(),
                    source_id: "interferometer-01".to_owned(),
                    source_digest: "not-a-digest".to_owned(),
                    calibration: CalibrationIdentity {
                        calibration_id: "calibration-density-01".to_owned(),
                        calibration_revision: "2026-11-01".to_owned(),
                    },
                    observed_at_ns: 100,
                },
            ),
            channel("temperature", "thomson", measured(4.0, 100)),
        ]);
        assert!(missing_digest.validate().is_err());

        let missing_calibration = base_frame(vec![
            channel(
                "density",
                "interferometer",
                ObservationDisposition::Measured {
                    value: 0.9,
                    units: "10^20 m^-3".to_owned(),
                    source_id: "interferometer-01".to_owned(),
                    source_digest: "4".repeat(64),
                    calibration: CalibrationIdentity {
                        calibration_id: String::new(),
                        calibration_revision: "2026-11-01".to_owned(),
                    },
                    observed_at_ns: 100,
                },
            ),
            channel("temperature", "thomson", measured(4.0, 100)),
        ]);
        assert!(missing_calibration.validate().is_err());
    }

    fn density_requirement() -> DecisionChannelRequirement {
        DecisionChannelRequirement {
            channel_id: "density".to_owned(),
            expected_units: "10^20 m^-3".to_owned(),
            max_age_ns: 20,
            allowed_source_ids: vec!["interferometer-01".to_owned()],
            calibration: CalibrationIdentity {
                calibration_id: "calibration-density-01".to_owned(),
                calibration_revision: "2026-11-01".to_owned(),
            },
        }
    }

    #[test]
    fn decision_requirements_admit_matching_fresh_calibrated_measurement() {
        let frame = base_frame(vec![
            channel("density", "interferometer", measured(0.9, 95)),
            channel("temperature", "thomson", measured(4.0, 100)),
        ]);
        let gate = frame
            .evaluate_decision_requirements(&[density_requirement()])
            .expect("valid requirement should evaluate");
        let DecisionInputGate::MeasuredInputsAvailable(inputs) = gate else {
            panic!("matching measured channel should be available");
        };
        assert_eq!(inputs.values().len(), 1);
        assert_eq!(inputs.values()[0].channel_id(), "density");
    }

    #[test]
    fn stale_measurement_abstains() {
        let frame = base_frame(vec![
            channel("density", "interferometer", measured(0.9, 70)),
            channel("temperature", "thomson", measured(4.0, 100)),
        ]);
        let gate = frame
            .evaluate_decision_requirements(&[density_requirement()])
            .expect("valid frame should evaluate");
        let DecisionInputGate::Abstain { blocked_channels, .. } = gate else {
            panic!("stale measurement must not be admitted");
        };
        assert_eq!(blocked_channels[0].reason, AbstentionReason::Stale);
    }

    #[test]
    fn units_source_and_calibration_must_match_the_decision_contract() {
        let cases = [
            (
                "wrong-units",
                ObservationDisposition::Measured {
                    value: 0.9,
                    units: "m^-3".to_owned(),
                    source_id: "interferometer-01".to_owned(),
                    source_digest: "5".repeat(64),
                    calibration: CalibrationIdentity {
                        calibration_id: "calibration-density-01".to_owned(),
                        calibration_revision: "2026-11-01".to_owned(),
                    },
                    observed_at_ns: 100,
                },
                AbstentionReason::WrongUnits,
            ),
            (
                "wrong-source",
                ObservationDisposition::Measured {
                    value: 0.9,
                    units: "10^20 m^-3".to_owned(),
                    source_id: "unknown-interferometer".to_owned(),
                    source_digest: "6".repeat(64),
                    calibration: CalibrationIdentity {
                        calibration_id: "calibration-density-01".to_owned(),
                        calibration_revision: "2026-11-01".to_owned(),
                    },
                    observed_at_ns: 100,
                },
                AbstentionReason::UnapprovedSource,
            ),
            (
                "wrong-calibration",
                ObservationDisposition::Measured {
                    value: 0.9,
                    units: "10^20 m^-3".to_owned(),
                    source_id: "interferometer-01".to_owned(),
                    source_digest: "7".repeat(64),
                    calibration: CalibrationIdentity {
                        calibration_id: "calibration-density-01".to_owned(),
                        calibration_revision: "unregistered-revision".to_owned(),
                    },
                    observed_at_ns: 100,
                },
                AbstentionReason::CalibrationMismatch,
            ),
        ];

        for (case_id, disposition, expected_reason) in cases {
            let frame = base_frame(vec![
                channel("density", "interferometer", disposition),
                channel("temperature", "thomson", measured(4.0, 100)),
            ]);
            let gate = frame
                .evaluate_decision_requirements(&[density_requirement()])
                .unwrap_or_else(|error| panic!("{case_id}: {error}"));
            let DecisionInputGate::Abstain { blocked_channels, .. } = gate else {
                panic!("{case_id}: measurement mismatch must abstain");
            };
            assert_eq!(blocked_channels[0].reason, expected_reason, "{case_id}");
        }
    }

    #[test]
    fn requirements_must_be_explicit_and_unique() {
        let frame = base_frame(vec![
            channel("density", "interferometer", measured(0.9, 100)),
            channel("temperature", "thomson", measured(4.0, 100)),
        ]);
        assert!(frame.evaluate_decision_requirements(&[]).is_err());
        let requirement = density_requirement();
        assert!(
            frame
                .evaluate_decision_requirements(&[requirement.clone(), requirement])
                .is_err()
        );
        let mut invalid = density_requirement();
        invalid.allowed_source_ids.clear();
        assert!(frame.evaluate_decision_requirements(&[invalid]).is_err());
    }

    #[test]
    fn two_channel_status_matrix_never_promotes_non_measured_evidence() {
        let statuses = vec![
            measured(0.9, 100),
            ObservationDisposition::Synthetic {
                value: 0.9,
                units: "10^20 m^-3".to_owned(),
                model_id: "diag-recovery".to_owned(),
                model_revision: "abc123".to_owned(),
                input_digest: "8".repeat(64),
            },
            ObservationDisposition::Unavailable {
                reason: "dropout".to_owned(),
            },
            ObservationDisposition::Invalid {
                reason: "integrity check failed".to_owned(),
            },
            ObservationDisposition::Delayed {
                value: 0.9,
                units: "10^20 m^-3".to_owned(),
                source_id: "interferometer-01".to_owned(),
                source_digest: "9".repeat(64),
                calibration: CalibrationIdentity {
                    calibration_id: "calibration-density-01".to_owned(),
                    calibration_revision: "2026-11-01".to_owned(),
                },
                observed_at_ns: 90,
                latency_ns: 10,
            },
            ObservationDisposition::Degraded {
                value: 0.9,
                units: "10^20 m^-3".to_owned(),
                source_id: "interferometer-01".to_owned(),
                source_digest: "a".repeat(64),
                calibration: CalibrationIdentity {
                    calibration_id: "calibration-density-01".to_owned(),
                    calibration_revision: "2026-11-01".to_owned(),
                },
                observed_at_ns: 100,
                reason: "quality below registered threshold".to_owned(),
            },
        ];

        let mut available_cases = 0usize;
        let mut abstained_cases = 0usize;
        for (density_index, density_status) in statuses.iter().enumerate() {
            for (temperature_index, temperature_status) in statuses.iter().enumerate() {
                let frame = base_frame(vec![
                    channel("density", "interferometer", density_status.clone()),
                    channel("temperature", "thomson", temperature_status.clone()),
                ]);
                let gate = frame
                    .evaluate_decision_gate(&["density", "temperature"])
                    .expect("all matrix cases have valid frame syntax");

                if density_index == 0 && temperature_index == 0 {
                    assert!(
                        matches!(gate, DecisionInputGate::MeasuredInputsAvailable(_)),
                        "measured+measured must be available"
                    );
                    available_cases += 1;
                } else {
                    assert!(
                        matches!(gate, DecisionInputGate::Abstain { .. }),
                        "status pair ({density_index}, {temperature_index}) must abstain"
                    );
                    abstained_cases += 1;
                }
            }
        }

        assert_eq!(available_cases, 1);
        assert_eq!(abstained_cases, 35);
    }

    #[test]
    fn counts_preserve_measured_and_synthetic_as_separate_categories() {
        let frame = base_frame(vec![
            channel("density", "interferometer", measured(0.9, 100)),
            channel(
                "temperature",
                "thomson",
                ObservationDisposition::Synthetic {
                    value: 4.0,
                    units: "keV".to_owned(),
                    model_id: "diag-recovery".to_owned(),
                    model_revision: "abc123".to_owned(),
                    input_digest: "c".repeat(64),
                },
            ),
        ]);
        let counts = frame.counts().expect("valid frame should count");
        assert_eq!(counts.measured, 1);
        assert_eq!(counts.synthetic, 1);
        assert_eq!(counts.unavailable, 0);
    }
}
