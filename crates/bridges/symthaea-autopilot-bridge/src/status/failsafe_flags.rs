// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Readiness-relevant projection of PX4 `FailsafeFlags`.

use serde::{Deserialize, Serialize};
use symthaea_core::embodiment_evidence::{ClockDomainId, TimestampV1};

use crate::readiness::{Px4ReadinessFactV1, Px4ReadinessRequirementV1};

use super::{
    PX4_READINESS_STATUS_SCHEMA_V1, Px4StatusEvidenceError, digest_hex, feed_str,
    readiness_fact, status_timestamp, validate_identifier, validate_schema,
};

const COMMITMENT_DOMAIN_V1: &[u8] =
    b"symthaea.autopilot.px4.failsafe-flags-readiness.v1\0";

/// Content-addressed readiness projection of one PX4 `FailsafeFlags` message.
///
/// Strict and relaxed local/global position invalidity are retained separately.
/// The fact helpers below only invert the exact PX4 `*_invalid` / `*_lost`
/// proposition corresponding to the requested readiness fact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4FailsafeFlagsReadinessEvidenceV1 {
    /// Schema version.
    pub schema_version: u16,
    /// Adapter/producer profile that captured the projection.
    pub producer_profile_id: String,
    /// PX4-local status timestamp in microseconds since system start.
    pub timestamp_us: u64,
    /// PX4 reports angular-velocity estimate invalid.
    pub angular_velocity_invalid: bool,
    /// PX4 reports attitude estimate invalid.
    pub attitude_invalid: bool,
    /// PX4 reports local-altitude estimate invalid.
    pub local_altitude_invalid: bool,
    /// PX4 reports strict local-position estimate invalid.
    pub local_position_invalid: bool,
    /// PX4 reports relaxed local-position estimate invalid.
    pub local_position_invalid_relaxed: bool,
    /// PX4 reports local-velocity estimate invalid.
    pub local_velocity_invalid: bool,
    /// PX4 reports strict global-position estimate invalid.
    pub global_position_invalid: bool,
    /// PX4 reports relaxed global-position estimate invalid.
    pub global_position_invalid_relaxed: bool,
    /// PX4 reports the Offboard control signal lost.
    pub offboard_control_signal_lost: bool,
    /// PX4 reports manual-control signal lost.
    pub manual_control_signal_lost: bool,
    /// PX4 reports GCS connection lost.
    pub gcs_connection_lost: bool,
    /// Domain-separated content commitment.
    pub status_digest_hex: String,
}

impl Px4FailsafeFlagsReadinessEvidenceV1 {
    /// Construct, content-bind, and validate one readiness projection.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        producer_profile_id: impl Into<String>,
        timestamp_us: u64,
        angular_velocity_invalid: bool,
        attitude_invalid: bool,
        local_altitude_invalid: bool,
        local_position_invalid: bool,
        local_position_invalid_relaxed: bool,
        local_velocity_invalid: bool,
        global_position_invalid: bool,
        global_position_invalid_relaxed: bool,
        offboard_control_signal_lost: bool,
        manual_control_signal_lost: bool,
        gcs_connection_lost: bool,
    ) -> Result<Self, Px4StatusEvidenceError> {
        let mut value = Self {
            schema_version: PX4_READINESS_STATUS_SCHEMA_V1,
            producer_profile_id: producer_profile_id.into(),
            timestamp_us,
            angular_velocity_invalid,
            attitude_invalid,
            local_altitude_invalid,
            local_position_invalid,
            local_position_invalid_relaxed,
            local_velocity_invalid,
            global_position_invalid,
            global_position_invalid_relaxed,
            offboard_control_signal_lost,
            manual_control_signal_lost,
            gcs_connection_lost,
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
            return Err(Px4StatusEvidenceError::DigestMismatch("failsafe_flags"));
        }
        Ok(())
    }

    /// Content-addressed identity of this exact projection.
    pub fn status_id(&self) -> Result<String, Px4StatusEvidenceError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.px4.failsafe-flags-readiness.v1:{}",
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

    /// `AngularVelocityValid = !angular_velocity_invalid`.
    pub fn angular_velocity_valid_fact(
        &self,
    ) -> Result<Px4ReadinessFactV1, Px4StatusEvidenceError> {
        self.validity_fact(
            Px4ReadinessRequirementV1::AngularVelocityValid,
            !self.angular_velocity_invalid,
        )
    }

    /// `AttitudeValid = !attitude_invalid`.
    pub fn attitude_valid_fact(&self) -> Result<Px4ReadinessFactV1, Px4StatusEvidenceError> {
        self.validity_fact(
            Px4ReadinessRequirementV1::AttitudeValid,
            !self.attitude_invalid,
        )
    }

    /// `LocalAltitudeValid = !local_altitude_invalid`.
    pub fn local_altitude_valid_fact(
        &self,
    ) -> Result<Px4ReadinessFactV1, Px4StatusEvidenceError> {
        self.validity_fact(
            Px4ReadinessRequirementV1::LocalAltitudeValid,
            !self.local_altitude_invalid,
        )
    }

    /// `LocalPositionValid = !local_position_invalid` using the strict PX4 flag.
    pub fn local_position_valid_fact(
        &self,
    ) -> Result<Px4ReadinessFactV1, Px4StatusEvidenceError> {
        self.validity_fact(
            Px4ReadinessRequirementV1::LocalPositionValid,
            !self.local_position_invalid,
        )
    }

    /// `LocalVelocityValid = !local_velocity_invalid`.
    pub fn local_velocity_valid_fact(
        &self,
    ) -> Result<Px4ReadinessFactV1, Px4StatusEvidenceError> {
        self.validity_fact(
            Px4ReadinessRequirementV1::LocalVelocityValid,
            !self.local_velocity_invalid,
        )
    }

    /// `GlobalPositionValid = !global_position_invalid` using the strict PX4 flag.
    pub fn global_position_valid_fact(
        &self,
    ) -> Result<Px4ReadinessFactV1, Px4StatusEvidenceError> {
        self.validity_fact(
            Px4ReadinessRequirementV1::GlobalPositionValid,
            !self.global_position_invalid,
        )
    }

    /// `OffboardSignalPresent = !offboard_control_signal_lost`.
    pub fn offboard_signal_present_fact(
        &self,
    ) -> Result<Px4ReadinessFactV1, Px4StatusEvidenceError> {
        self.validity_fact(
            Px4ReadinessRequirementV1::OffboardSignalPresent,
            !self.offboard_control_signal_lost,
        )
    }

    fn validity_fact(
        &self,
        requirement: Px4ReadinessRequirementV1,
        satisfied: bool,
    ) -> Result<Px4ReadinessFactV1, Px4StatusEvidenceError> {
        readiness_fact(requirement, satisfied, self.status_id()?)
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
            u8::from(self.angular_velocity_invalid),
            u8::from(self.attitude_invalid),
            u8::from(self.local_altitude_invalid),
            u8::from(self.local_position_invalid),
            u8::from(self.local_position_invalid_relaxed),
            u8::from(self.local_velocity_invalid),
            u8::from(self.global_position_invalid),
            u8::from(self.global_position_invalid_relaxed),
            u8::from(self.offboard_control_signal_lost),
            u8::from(self.manual_control_signal_lost),
            u8::from(self.gcs_connection_lost),
        ]);
        digest_hex(hasher.finalize().as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::readiness::Px4ReadinessFactStateV1;

    fn evidence() -> Px4FailsafeFlagsReadinessEvidenceV1 {
        Px4FailsafeFlagsReadinessEvidenceV1::new(
            "px4.failsafe-flags.fixture.v1",
            50,
            false,
            true,
            false,
            true,
            false,
            false,
            false,
            true,
            true,
            false,
            false,
        )
        .unwrap()
    }

    #[test]
    fn strict_and_relaxed_position_validity_are_not_collapsed() {
        let value = evidence();
        assert!(value.local_position_invalid);
        assert!(!value.local_position_invalid_relaxed);
        assert!(!value.global_position_invalid);
        assert!(value.global_position_invalid_relaxed);
    }

    #[test]
    fn exact_invalid_and_lost_flags_map_to_matching_facts() {
        let value = evidence();
        assert_eq!(
            value.angular_velocity_valid_fact().unwrap().state,
            Px4ReadinessFactStateV1::Satisfied
        );
        assert_eq!(
            value.attitude_valid_fact().unwrap().state,
            Px4ReadinessFactStateV1::Unsatisfied
        );
        assert_eq!(
            value.local_position_valid_fact().unwrap().state,
            Px4ReadinessFactStateV1::Unsatisfied
        );
        assert_eq!(
            value.global_position_valid_fact().unwrap().state,
            Px4ReadinessFactStateV1::Satisfied
        );
        assert_eq!(
            value.offboard_signal_present_fact().unwrap().state,
            Px4ReadinessFactStateV1::Unsatisfied
        );
    }

    #[test]
    fn mutation_invalidates_fact_derivation() {
        let mut value = evidence();
        value.attitude_invalid = false;
        assert_eq!(
            value.attitude_valid_fact(),
            Err(Px4StatusEvidenceError::DigestMismatch("failsafe_flags"))
        );
    }
}
