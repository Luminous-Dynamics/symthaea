// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Readiness-relevant projection of PX4 `VehicleStatus`.

use serde::{Deserialize, Serialize};
use symthaea_core::embodiment_evidence::{ClockDomainId, TimestampV1};

use crate::readiness::{Px4ReadinessFactV1, Px4ReadinessRequirementV1};

use super::{
    PX4_READINESS_STATUS_SCHEMA_V1, Px4StatusEvidenceError, digest_hex, feed_str,
    readiness_fact, status_timestamp, validate_identifier, validate_schema,
};

const COMMITMENT_DOMAIN_V1: &[u8] =
    b"symthaea.autopilot.px4.vehicle-status-readiness.v1\0";

/// Content-addressed readiness-specific projection of one PX4 `VehicleStatus`.
///
/// Raw mode/arming/HIL values are preserved as integers so future PX4 enum values
/// are not silently mapped into an invented meaning. `nav_state` and
/// `nav_state_user_intention` are retained separately because failsafe/executor
/// behavior may make them differ.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4VehicleStatusReadinessEvidenceV1 {
    /// Schema version.
    pub schema_version: u16,
    /// Adapter/producer profile that captured the projection.
    pub producer_profile_id: String,
    /// PX4-local status timestamp in microseconds since system start.
    pub timestamp_us: u64,
    /// Raw PX4 arming-state value.
    pub arming_state: u8,
    /// Raw user-intended navigation-state value.
    pub nav_state_user_intention: u8,
    /// Raw currently active navigation-state value.
    pub nav_state: u8,
    /// Whether the current PX4 mode reports that it accepts Offboard setpoints.
    pub accepts_offboard_setpoints: bool,
    /// Whether PX4 currently reports an active failsafe.
    pub failsafe: bool,
    /// Whether PX4 reports failsafe active with user takeover.
    pub failsafe_and_user_took_over: bool,
    /// Raw PX4 HIL-state value.
    pub hil_state: u8,
    /// Domain-separated content commitment.
    pub status_digest_hex: String,
}

impl Px4VehicleStatusReadinessEvidenceV1 {
    /// Construct, content-bind, and validate one readiness projection.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        producer_profile_id: impl Into<String>,
        timestamp_us: u64,
        arming_state: u8,
        nav_state_user_intention: u8,
        nav_state: u8,
        accepts_offboard_setpoints: bool,
        failsafe: bool,
        failsafe_and_user_took_over: bool,
        hil_state: u8,
    ) -> Result<Self, Px4StatusEvidenceError> {
        let mut value = Self {
            schema_version: PX4_READINESS_STATUS_SCHEMA_V1,
            producer_profile_id: producer_profile_id.into(),
            timestamp_us,
            arming_state,
            nav_state_user_intention,
            nav_state,
            accepts_offboard_setpoints,
            failsafe,
            failsafe_and_user_took_over,
            hil_state,
            status_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.status_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate schema, producer identity, and content commitment.
    pub fn validate(&self) -> Result<(), Px4StatusEvidenceError> {
        self.validate_without_digest()?;
        if self.status_digest_hex != self.compute_digest_hex() {
            return Err(Px4StatusEvidenceError::DigestMismatch("vehicle_status"));
        }
        Ok(())
    }

    /// Content-addressed identity of this exact projection.
    pub fn status_id(&self) -> Result<String, Px4StatusEvidenceError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.px4.vehicle-status-readiness.v1:{}",
            self.status_digest_hex
        ))
    }

    /// PX4-local observation time in an explicit caller-supplied PX4 clock domain.
    pub fn observed_at(
        &self,
        px4_clock_domain: ClockDomainId,
    ) -> Result<TimestampV1, Px4StatusEvidenceError> {
        self.validate()?;
        status_timestamp(self.timestamp_us, px4_clock_domain)
    }

    /// Evidence-backed `AcceptsOffboardSetpoints` fact from the exact PX4 field.
    pub fn accepts_offboard_setpoints_fact(
        &self,
    ) -> Result<Px4ReadinessFactV1, Px4StatusEvidenceError> {
        readiness_fact(
            Px4ReadinessRequirementV1::AcceptsOffboardSetpoints,
            self.accepts_offboard_setpoints,
            self.status_id()?,
        )
    }

    /// Evidence-backed `FailsafeInactive` fact from `!VehicleStatus.failsafe`.
    pub fn failsafe_inactive_fact(
        &self,
    ) -> Result<Px4ReadinessFactV1, Px4StatusEvidenceError> {
        readiness_fact(
            Px4ReadinessRequirementV1::FailsafeInactive,
            !self.failsafe,
            self.status_id()?,
        )
    }

    fn validate_without_digest(&self) -> Result<(), Px4StatusEvidenceError> {
        validate_schema(self.schema_version)?;
        validate_identifier(&self.producer_profile_id, "producer_profile_id")
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(COMMITMENT_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.producer_profile_id);
        hasher.update(&self.timestamp_us.to_le_bytes());
        hasher.update(&[
            self.arming_state,
            self.nav_state_user_intention,
            self.nav_state,
            u8::from(self.accepts_offboard_setpoints),
            u8::from(self.failsafe),
            u8::from(self.failsafe_and_user_took_over),
            self.hil_state,
        ]);
        digest_hex(hasher.finalize().as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::readiness::Px4ReadinessFactStateV1;

    fn evidence(accepts: bool, failsafe: bool) -> Px4VehicleStatusReadinessEvidenceV1 {
        Px4VehicleStatusReadinessEvidenceV1::new(
            "px4.vehicle-status.fixture.v1",
            42,
            1,
            14,
            5,
            accepts,
            failsafe,
            false,
            0,
        )
        .unwrap()
    }

    #[test]
    fn active_and_user_intended_nav_states_remain_distinct() {
        let value = evidence(true, false);
        assert_eq!(value.nav_state_user_intention, 14);
        assert_eq!(value.nav_state, 5);
        assert_ne!(value.nav_state_user_intention, value.nav_state);
    }

    #[test]
    fn readiness_facts_map_only_exact_vehicle_status_fields() {
        let value = evidence(false, true);
        let status_id = value.status_id().unwrap();
        let accepts = value.accepts_offboard_setpoints_fact().unwrap();
        let failsafe = value.failsafe_inactive_fact().unwrap();
        assert_eq!(accepts.requirement, Px4ReadinessRequirementV1::AcceptsOffboardSetpoints);
        assert_eq!(accepts.state, Px4ReadinessFactStateV1::Unsatisfied);
        assert_eq!(failsafe.requirement, Px4ReadinessRequirementV1::FailsafeInactive);
        assert_eq!(failsafe.state, Px4ReadinessFactStateV1::Unsatisfied);
        assert_eq!(accepts.evidence_id.as_deref(), Some(status_id.as_str()));
        assert_eq!(failsafe.evidence_id.as_deref(), Some(status_id.as_str()));
    }

    #[test]
    fn mutation_invalidates_fact_derivation() {
        let mut value = evidence(true, false);
        value.failsafe = true;
        assert_eq!(
            value.failsafe_inactive_fact(),
            Err(Px4StatusEvidenceError::DigestMismatch("vehicle_status"))
        );
    }
}
