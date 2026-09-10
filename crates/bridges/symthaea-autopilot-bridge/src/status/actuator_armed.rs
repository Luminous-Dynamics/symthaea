// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Readiness-relevant projection of PX4 `ActuatorArmed`.

use serde::{Deserialize, Serialize};
use symthaea_core::embodiment_evidence::{ClockDomainId, TimestampV1};

use crate::readiness::{Px4ReadinessFactV1, Px4ReadinessRequirementV1};

use super::{
    PX4_READINESS_STATUS_SCHEMA_V1, Px4StatusEvidenceError, digest_hex, feed_str,
    readiness_fact, status_timestamp, validate_identifier, validate_schema,
};

const COMMITMENT_DOMAIN_V1: &[u8] =
    b"symthaea.autopilot.px4.actuator-armed-readiness.v1\0";

/// Content-addressed readiness projection of one PX4 `ActuatorArmed` message.
///
/// `ready_to_arm` is preserved as a PX4 status proposition only. It does not
/// represent operator/delegation authority, a Symthaea safety decision, or an
/// instruction to arm.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Px4ActuatorArmedReadinessEvidenceV1 {
    /// Schema version.
    pub schema_version: u16,
    /// Adapter/producer profile that captured the projection.
    pub producer_profile_id: String,
    /// PX4-local status timestamp in microseconds since system start.
    pub timestamp_us: u64,
    /// PX4 reports the vehicle armed.
    pub armed: bool,
    /// PX4 reports actuator safety disabled while motors are not armed.
    pub prearmed: bool,
    /// PX4 reports the system ready to arm; this is not arm authority.
    pub ready_to_arm: bool,
    /// PX4 reports actuator lockdown active.
    pub lockdown: bool,
    /// PX4 reports manual kill active.
    pub kill: bool,
    /// PX4 reports actuator termination active.
    pub termination: bool,
    /// PX4 reports ESC calibration mode active.
    pub in_esc_calibration_mode: bool,
    /// Domain-separated content commitment.
    pub status_digest_hex: String,
}

impl Px4ActuatorArmedReadinessEvidenceV1 {
    /// Construct, content-bind, and validate one readiness projection.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        producer_profile_id: impl Into<String>,
        timestamp_us: u64,
        armed: bool,
        prearmed: bool,
        ready_to_arm: bool,
        lockdown: bool,
        kill: bool,
        termination: bool,
        in_esc_calibration_mode: bool,
    ) -> Result<Self, Px4StatusEvidenceError> {
        let mut value = Self {
            schema_version: PX4_READINESS_STATUS_SCHEMA_V1,
            producer_profile_id: producer_profile_id.into(),
            timestamp_us,
            armed,
            prearmed,
            ready_to_arm,
            lockdown,
            kill,
            termination,
            in_esc_calibration_mode,
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
            return Err(Px4StatusEvidenceError::DigestMismatch("actuator_armed"));
        }
        Ok(())
    }

    /// Content-addressed identity of this exact projection.
    pub fn status_id(&self) -> Result<String, Px4StatusEvidenceError> {
        self.validate()?;
        Ok(format!(
            "symthaea.autopilot.px4.actuator-armed-readiness.v1:{}",
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

    /// Evidence-backed `ReadyToArm` fact from the exact PX4 field.
    ///
    /// A satisfied result still does not authorize arming.
    pub fn ready_to_arm_fact(&self) -> Result<Px4ReadinessFactV1, Px4StatusEvidenceError> {
        self.fact(Px4ReadinessRequirementV1::ReadyToArm, self.ready_to_arm)
    }

    /// Evidence-backed `LockdownInactive = !lockdown` fact.
    pub fn lockdown_inactive_fact(
        &self,
    ) -> Result<Px4ReadinessFactV1, Px4StatusEvidenceError> {
        self.fact(Px4ReadinessRequirementV1::LockdownInactive, !self.lockdown)
    }

    /// Evidence-backed `KillInactive = !kill` fact.
    pub fn kill_inactive_fact(&self) -> Result<Px4ReadinessFactV1, Px4StatusEvidenceError> {
        self.fact(Px4ReadinessRequirementV1::KillInactive, !self.kill)
    }

    /// Evidence-backed `TerminationInactive = !termination` fact.
    pub fn termination_inactive_fact(
        &self,
    ) -> Result<Px4ReadinessFactV1, Px4StatusEvidenceError> {
        self.fact(
            Px4ReadinessRequirementV1::TerminationInactive,
            !self.termination,
        )
    }

    fn fact(
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
            u8::from(self.armed),
            u8::from(self.prearmed),
            u8::from(self.ready_to_arm),
            u8::from(self.lockdown),
            u8::from(self.kill),
            u8::from(self.termination),
            u8::from(self.in_esc_calibration_mode),
        ]);
        digest_hex(hasher.finalize().as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::readiness::Px4ReadinessFactStateV1;

    fn evidence() -> Px4ActuatorArmedReadinessEvidenceV1 {
        Px4ActuatorArmedReadinessEvidenceV1::new(
            "px4.actuator-armed.fixture.v1",
            77,
            false,
            false,
            true,
            false,
            true,
            false,
            false,
        )
        .unwrap()
    }

    #[test]
    fn ready_to_arm_is_status_fact_not_command_or_authority() {
        let value = evidence();
        let status_id = value.status_id().unwrap();
        let fact = value.ready_to_arm_fact().unwrap();
        assert_eq!(fact.requirement, Px4ReadinessRequirementV1::ReadyToArm);
        assert_eq!(fact.state, Px4ReadinessFactStateV1::Satisfied);
        assert_eq!(fact.evidence_id.as_deref(), Some(status_id.as_str()));
    }

    #[test]
    fn actuator_interlocks_remain_independent_readiness_facts() {
        let value = evidence();
        assert_eq!(
            value.lockdown_inactive_fact().unwrap().state,
            Px4ReadinessFactStateV1::Satisfied
        );
        assert_eq!(
            value.kill_inactive_fact().unwrap().state,
            Px4ReadinessFactStateV1::Unsatisfied
        );
        assert_eq!(
            value.termination_inactive_fact().unwrap().state,
            Px4ReadinessFactStateV1::Satisfied
        );
    }

    #[test]
    fn mutation_invalidates_fact_derivation() {
        let mut value = evidence();
        value.kill = false;
        assert_eq!(
            value.kill_inactive_fact(),
            Err(Px4StatusEvidenceError::DigestMismatch("actuator_armed"))
        );
    }
}
