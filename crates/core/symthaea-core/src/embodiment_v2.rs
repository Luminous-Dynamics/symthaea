// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Universal Embodiment Contract v2: typed physical observations and action lineage.
//!
//! V2 supplements the v1 EmbodimentBridge compatibility facade. It makes identity,
//! time, units, missingness, and causal action stages explicit. Validation here is
//! contract validation, not a safety case, calibration proof, or hardware qualification.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

pub const EMBODIMENT_CONTRACT_VERSION_V2: u16 = 2;

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn validate_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Immutable profile identity plus the bound instance and backend provenance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbodimentDescriptorV2 {
    pub contract_version: u16,
    pub profile_id: String,
    pub platform_family_id: String,
    pub instance_id: String,
    pub morphology_id: String,
    pub observation_schema_id: String,
    pub action_schema_id: String,
    pub frame_graph_id: Option<String>,
    pub clock_domain_id: String,
    pub actuator_capability_ids: Vec<String>,
    pub sensor_capability_ids: Vec<String>,
    pub safe_fallback_profile_id: String,
    pub calibration_profile_id: Option<String>,
    /// Provenance reference for the simulator, middleware, or physical backend.
    pub backend_provenance_id: String,
    /// Optional content digest; presence is not proof that the descriptor is trusted.
    pub descriptor_digest: Option<String>,
}

impl EmbodimentDescriptorV2 {
    pub fn validate(&self) -> Result<(), String> {
        if self.contract_version != EMBODIMENT_CONTRACT_VERSION_V2 {
            return Err(format!("descriptor contract version {} is not v{}", self.contract_version, EMBODIMENT_CONTRACT_VERSION_V2));
        }
        for (field, value) in [
            ("profile_id", self.profile_id.as_str()),
            ("platform_family_id", self.platform_family_id.as_str()),
            ("instance_id", self.instance_id.as_str()),
            ("morphology_id", self.morphology_id.as_str()),
            ("observation_schema_id", self.observation_schema_id.as_str()),
            ("action_schema_id", self.action_schema_id.as_str()),
            ("clock_domain_id", self.clock_domain_id.as_str()),
            ("safe_fallback_profile_id", self.safe_fallback_profile_id.as_str()),
            ("backend_provenance_id", self.backend_provenance_id.as_str()),
        ] {
            if !non_empty(value) {
                return Err(format!("{field} must not be empty"));
            }
        }
        if self.frame_graph_id.as_deref().is_some_and(|v| !non_empty(v)) {
            return Err("frame_graph_id must be absent or non-empty".into());
        }
        if self.calibration_profile_id.as_deref().is_some_and(|v| !non_empty(v)) {
            return Err("calibration_profile_id must be absent or non-empty".into());
        }
        if self.descriptor_digest.as_deref().is_some_and(|d| !validate_digest(d)) {
            return Err("descriptor_digest must be 64 hexadecimal characters".into());
        }
        validate_unique_ids("actuator_capability_ids", &self.actuator_capability_ids)?;
        validate_unique_ids("sensor_capability_ids", &self.sensor_capability_ids)?;
        Ok(())
    }
}

fn validate_unique_ids(field: &str, values: &[String]) -> Result<(), String> {
    let mut seen = HashSet::with_capacity(values.len());
    for value in values {
        if !non_empty(value) {
            return Err(format!("{field} contains an empty identifier"));
        }
        if !seen.insert(value) {
            return Err(format!("{field} contains duplicate identifier {value:?}"));
        }
    }
    Ok(())
}

/// Source-provided clock coordinates. Missing sequence/time remains missing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockStampV2 {
    pub clock_domain_id: String,
    pub sequence: Option<u64>,
    pub timestamp_ns: Option<u64>,
}

impl ClockStampV2 {
    pub fn validate(&self) -> Result<(), String> {
        if !non_empty(&self.clock_domain_id) {
            return Err("clock_domain_id must not be empty".into());
        }
        Ok(())
    }
}

/// Explicit reason a physical observation is unavailable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissingObservationReasonV2 {
    NotReported,
    SensorUnavailable,
    Stale,
    Invalid,
    OutOfRange,
    Occluded,
    TransportFailure,
}

/// One typed scalar observation. Units are mandatory and uncertainty is never inferred.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationValueV2 {
    pub address: String,
    pub quantity: String,
    pub unit: String,
    pub value: Option<f64>,
    pub missing_reason: Option<MissingObservationReasonV2>,
    /// Absolute uncertainty in the declared unit, if known.
    pub uncertainty: Option<f64>,
}

impl ObservationValueV2 {
    pub fn validate(&self) -> Result<(), String> {
        for (field, value) in [
            ("address", self.address.as_str()),
            ("quantity", self.quantity.as_str()),
            ("unit", self.unit.as_str()),
        ] {
            if !non_empty(value) {
                return Err(format!("observation {field} must not be empty"));
            }
        }
        match (self.value, self.missing_reason) {
            (Some(value), None) if value.is_finite() => {}
            (Some(_), None) => return Err("observation value must be finite".into()),
            (None, Some(_)) if self.uncertainty.is_none() => {}
            (None, Some(_)) => return Err("missing observation cannot assert numeric uncertainty".into()),
            (Some(_), Some(_)) => return Err("observation cannot be present and missing simultaneously".into()),
            (None, None) => return Err("observation needs a value or explicit missing reason".into()),
        }
        if self.uncertainty.is_some_and(|u| !u.is_finite() || u < 0.0) {
            return Err("observation uncertainty must be finite and non-negative".into());
        }
        Ok(())
    }
}

/// Typed source observations, not hidden plant truth.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationFrameV2 {
    pub contract_version: u16,
    pub profile_id: String,
    pub instance_id: String,
    pub clock: ClockStampV2,
    pub frame_graph_id: Option<String>,
    pub source_id: String,
    pub observations: Vec<ObservationValueV2>,
}

impl ObservationFrameV2 {
    pub fn validate(&self) -> Result<(), String> {
        if self.contract_version != EMBODIMENT_CONTRACT_VERSION_V2 {
            return Err(format!("observation contract version {} is not v{}", self.contract_version, EMBODIMENT_CONTRACT_VERSION_V2));
        }
        for (field, value) in [
            ("profile_id", self.profile_id.as_str()),
            ("instance_id", self.instance_id.as_str()),
            ("source_id", self.source_id.as_str()),
        ] {
            if !non_empty(value) {
                return Err(format!("{field} must not be empty"));
            }
        }
        self.clock.validate()?;
        if self.frame_graph_id.as_deref().is_some_and(|v| !non_empty(v)) {
            return Err("frame_graph_id must be absent or non-empty".into());
        }
        if self.observations.is_empty() {
            return Err("observation frame must contain at least one observation".into());
        }
        let mut addresses = HashSet::with_capacity(self.observations.len());
        for observation in &self.observations {
            observation.validate()?;
            if !addresses.insert(&observation.address) {
                return Err(format!("duplicate observation address {:?}", observation.address));
            }
        }
        Ok(())
    }
}

/// Causal stage; a transport receipt is not physical execution evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionStageV2 {
    Requested,
    Projected,
    TransportAccepted,
    Applied,
    Measured,
}

/// One physical or semantic actuator quantity with explicit units.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionValueV2 {
    pub address: String,
    pub quantity: String,
    pub unit: String,
    pub value: f64,
}

impl ActionValueV2 {
    pub fn validate(&self) -> Result<(), String> {
        for (field, value) in [
            ("address", self.address.as_str()),
            ("quantity", self.quantity.as_str()),
            ("unit", self.unit.as_str()),
        ] {
            if !non_empty(value) {
                return Err(format!("action {field} must not be empty"));
            }
        }
        if !self.value.is_finite() {
            return Err("action value must be finite".into());
        }
        Ok(())
    }
}

/// Requested/projected/published/applied/measured action lineage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionFrameV2 {
    pub contract_version: u16,
    pub profile_id: String,
    pub instance_id: String,
    pub clock: ClockStampV2,
    pub frame_graph_id: Option<String>,
    pub source_id: String,
    pub stage: ActionStageV2,
    pub actions: Vec<ActionValueV2>,
}

impl ActionFrameV2 {
    pub fn validate(&self) -> Result<(), String> {
        if self.contract_version != EMBODIMENT_CONTRACT_VERSION_V2 {
            return Err(format!("action contract version {} is not v{}", self.contract_version, EMBODIMENT_CONTRACT_VERSION_V2));
        }
        for (field, value) in [
            ("profile_id", self.profile_id.as_str()),
            ("instance_id", self.instance_id.as_str()),
            ("source_id", self.source_id.as_str()),
        ] {
            if !non_empty(value) {
                return Err(format!("{field} must not be empty"));
            }
        }
        self.clock.validate()?;
        if self.frame_graph_id.as_deref().is_some_and(|v| !non_empty(v)) {
            return Err("frame_graph_id must be absent or non-empty".into());
        }
        if self.actions.is_empty() {
            return Err("action frame must contain at least one action".into());
        }
        let mut addresses = HashSet::with_capacity(self.actions.len());
        for action in &self.actions {
            action.validate()?;
            if !addresses.insert(&action.address) {
                return Err(format!("duplicate action address {:?}", action.address));
            }
        }
        Ok(())
    }
}

/// Optional typed facade implemented by platforms migrating off the opaque v1 call.
pub trait EmbodimentV2Adapter: Send + Sync {
    fn descriptor_v2(&self, instance_id: &str) -> EmbodimentDescriptorV2;
    fn observation_frame_v2(&self, instance_id: &str) -> Result<ObservationFrameV2, String>;
    fn last_action_frame_v2(&self, instance_id: &str) -> Result<Option<ActionFrameV2>, String>;
}

/// Common, read-only conformance validator. It never actuates a platform or synthesizes receipts.
pub fn validate_adapter_v2_conformance(
    adapter: &dyn EmbodimentV2Adapter,
    instance_id: &str,
) -> Result<(), Vec<String>> {
    let mut failures = Vec::new();
    let descriptor = adapter.descriptor_v2(instance_id);
    if let Err(error) = descriptor.validate() {
        failures.push(format!("descriptor: {error}"));
    }
    if descriptor.instance_id != instance_id {
        failures.push("descriptor instance_id differs from requested instance".into());
    }

    match adapter.observation_frame_v2(instance_id) {
        Ok(frame) => {
            if let Err(error) = frame.validate() {
                failures.push(format!("observation frame: {error}"));
            }
            if frame.profile_id != descriptor.profile_id {
                failures.push("observation profile differs from descriptor".into());
            }
            if frame.instance_id != descriptor.instance_id {
                failures.push("observation instance differs from descriptor".into());
            }
            if frame.clock.clock_domain_id != descriptor.clock_domain_id {
                failures.push("observation clock domain differs from descriptor".into());
            }
        }
        Err(error) => failures.push(format!("observation frame unavailable: {error}")),
    }

    match adapter.last_action_frame_v2(instance_id) {
        Ok(Some(frame)) => {
            if let Err(error) = frame.validate() {
                failures.push(format!("action frame: {error}"));
            }
            if frame.profile_id != descriptor.profile_id {
                failures.push("action profile differs from descriptor".into());
            }
            if frame.instance_id != descriptor.instance_id {
                failures.push("action instance differs from descriptor".into());
            }
            if frame.clock.clock_domain_id != descriptor.clock_domain_id {
                failures.push("action clock domain differs from descriptor".into());
            }
            if frame.stage == ActionStageV2::Applied
                && frame.source_id != descriptor.backend_provenance_id
            {
                failures.push("Applied action source differs from descriptor backend provenance".into());
            }
        }
        Ok(None) => {}
        Err(error) => failures.push(format!("action frame unavailable: {error}")),
    }

    if failures.is_empty() { Ok(()) } else { Err(failures) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clock() -> ClockStampV2 {
        ClockStampV2 { clock_domain_id: "sim-time".into(), sequence: Some(1), timestamp_ns: Some(1_000_000) }
    }

    fn observation(value: Option<f64>, missing_reason: Option<MissingObservationReasonV2>) -> ObservationValueV2 {
        ObservationValueV2 {
            address: "body/position/z".into(),
            quantity: "position".into(),
            unit: "m".into(),
            value,
            missing_reason,
            uncertainty: None,
        }
    }

    fn frame() -> ObservationFrameV2 {
        ObservationFrameV2 {
            contract_version: EMBODIMENT_CONTRACT_VERSION_V2,
            profile_id: "test.profile.v1".into(),
            instance_id: "test-instance".into(),
            clock: clock(),
            frame_graph_id: Some("world/body".into()),
            source_id: "test-source".into(),
            observations: vec![observation(Some(1.25), None)],
        }
    }

    #[test]
    fn valid_observation_frame_passes() { assert!(frame().validate().is_ok()); }

    #[test]
    fn explicit_missingness_is_valid_but_ambiguous_missingness_fails() {
        assert!(observation(None, Some(MissingObservationReasonV2::NotReported)).validate().is_ok());
        assert!(observation(None, None).validate().is_err());
        assert!(observation(Some(1.0), Some(MissingObservationReasonV2::Stale)).validate().is_err());
    }

    #[test]
    fn non_finite_values_negative_uncertainty_and_duplicate_addresses_fail_closed() {
        assert!(observation(Some(f64::NAN), None).validate().is_err());
        let mut item = observation(Some(1.0), None);
        item.uncertainty = Some(-0.01);
        assert!(item.validate().is_err());
        let mut duplicate = frame();
        duplicate.observations.push(duplicate.observations[0].clone());
        assert!(duplicate.validate().is_err());
    }

    #[test]
    fn action_stage_is_explicit_and_non_finite_actions_fail() {
        let mut action = ActionFrameV2 {
            contract_version: EMBODIMENT_CONTRACT_VERSION_V2,
            profile_id: "test.profile.v1".into(),
            instance_id: "test-instance".into(),
            clock: clock(),
            frame_graph_id: Some("world/body".into()),
            source_id: "test-backend".into(),
            stage: ActionStageV2::TransportAccepted,
            actions: vec![ActionValueV2 { address: "body/thrust".into(), quantity: "force".into(), unit: "N".into(), value: 0.25 }],
        };
        assert!(action.validate().is_ok());
        assert_eq!(action.stage, ActionStageV2::TransportAccepted);
        action.actions[0].value = f64::INFINITY;
        assert!(action.validate().is_err());
    }

    #[test]
    fn descriptor_rejects_missing_fallback_and_bad_digest() {
        let descriptor = EmbodimentDescriptorV2 {
            contract_version: EMBODIMENT_CONTRACT_VERSION_V2,
            profile_id: "test.profile.v1".into(),
            platform_family_id: "test".into(),
            instance_id: "test-instance".into(),
            morphology_id: "test.morphology.v1".into(),
            observation_schema_id: "uec.observation.v2".into(),
            action_schema_id: "uec.action.v2".into(),
            frame_graph_id: Some("world/body".into()),
            clock_domain_id: "sim-time".into(),
            actuator_capability_ids: vec!["body/thrust".into()],
            sensor_capability_ids: vec!["body/position".into()],
            safe_fallback_profile_id: " ".into(),
            calibration_profile_id: None,
            backend_provenance_id: "test-backend".into(),
            descriptor_digest: Some("not-a-digest".into()),
        };
        assert!(descriptor.validate().is_err());
    }
}
