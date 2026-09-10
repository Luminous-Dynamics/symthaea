// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Readiness-specific, content-addressed PX4 status projections.
//!
//! These are deliberately narrow observational projections of PX4 status
//! messages. They are not generated uORB bindings, command/control APIs, or
//! authority records. Each projection preserves the raw fields used to derive
//! specific readiness facts and refuses to manufacture unrelated facts.

use std::fmt::Write as _;

use symthaea_core::embodiment_evidence::{
    ClockDomainId, EvidenceValidationError, TimestampV1,
};
use thiserror::Error;

use crate::readiness::{
    Px4ReadinessFactV1, Px4ReadinessRequirementV1, Px4ReadinessValidationError,
};

mod actuator_armed;
mod failsafe_flags;
mod vehicle_status;

pub use actuator_armed::Px4ActuatorArmedReadinessEvidenceV1;
pub use failsafe_flags::Px4FailsafeFlagsReadinessEvidenceV1;
pub use vehicle_status::Px4VehicleStatusReadinessEvidenceV1;

/// Schema version shared by readiness-specific PX4 status projections.
pub const PX4_READINESS_STATUS_SCHEMA_V1: u16 = 1;

pub(super) fn validate_schema(schema_version: u16) -> Result<(), Px4StatusEvidenceError> {
    if schema_version != PX4_READINESS_STATUS_SCHEMA_V1 {
        return Err(Px4StatusEvidenceError::UnsupportedSchemaVersion {
            found: schema_version,
        });
    }
    Ok(())
}

pub(super) fn validate_identifier(
    value: &str,
    field: &'static str,
) -> Result<(), Px4StatusEvidenceError> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.len() != value.len()
        || value.len() > 512
        || value.chars().any(char::is_control)
    {
        return Err(Px4StatusEvidenceError::InvalidIdentifier(field));
    }
    Ok(())
}

pub(super) fn status_timestamp(
    timestamp_us: u64,
    clock_domain: ClockDomainId,
) -> Result<TimestampV1, Px4StatusEvidenceError> {
    clock_domain
        .validate()
        .map_err(Px4StatusEvidenceError::ClockDomain)?;
    let nanoseconds = timestamp_us
        .checked_mul(1_000)
        .ok_or(Px4StatusEvidenceError::TimestampOverflow("timestamp_us"))?;
    Ok(TimestampV1::new(clock_domain, nanoseconds))
}

pub(super) fn readiness_fact(
    requirement: Px4ReadinessRequirementV1,
    satisfied: bool,
    evidence_id: String,
) -> Result<Px4ReadinessFactV1, Px4StatusEvidenceError> {
    if satisfied {
        Px4ReadinessFactV1::satisfied(requirement, evidence_id)
    } else {
        Px4ReadinessFactV1::unsatisfied(requirement, evidence_id)
    }
    .map_err(Px4StatusEvidenceError::Readiness)
}

pub(super) fn feed_str(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

pub(super) fn digest_hex(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

/// Validation failure for readiness-specific PX4 status evidence.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum Px4StatusEvidenceError {
    /// Projection uses an unsupported schema version.
    #[error("unsupported PX4 readiness-status schema version {found}")]
    UnsupportedSchemaVersion {
        /// Unsupported version encountered.
        found: u16,
    },
    /// Required identifier is empty, padded, too long, or contains controls.
    #[error("invalid PX4 status evidence identifier field {0}")]
    InvalidIdentifier(&'static str),
    /// Caller-supplied PX4 clock domain is invalid.
    #[error("invalid PX4 status clock domain: {0}")]
    ClockDomain(EvidenceValidationError),
    /// PX4 microsecond timestamp cannot be represented in nanoseconds.
    #[error("PX4 status field {0} overflows nanosecond representation")]
    TimestampOverflow(&'static str),
    /// Stored projection commitment no longer matches its fields.
    #[error("PX4 {0} readiness projection commitment mismatch")]
    DigestMismatch(&'static str),
    /// Derived readiness fact failed readiness-layer validation.
    #[error("invalid derived PX4 readiness fact: {0}")]
    Readiness(Px4ReadinessValidationError),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clock() -> ClockDomainId {
        ClockDomainId::new("px4.hrt").unwrap()
    }

    #[test]
    fn microsecond_timestamp_overflow_fails_closed() {
        assert_eq!(
            status_timestamp(u64::MAX, clock()),
            Err(Px4StatusEvidenceError::TimestampOverflow("timestamp_us"))
        );
    }

    #[test]
    fn deserialized_invalid_clock_domain_fails_before_timestamp_promotion() {
        let invalid_clock: ClockDomainId = serde_json::from_str("\" padded\"").unwrap();
        assert_eq!(
            status_timestamp(1, invalid_clock),
            Err(Px4StatusEvidenceError::ClockDomain(
                EvidenceValidationError::InvalidClockDomain
            ))
        );
    }

    #[test]
    fn all_status_projection_serde_round_trips_preserve_identity() {
        let vehicle = Px4VehicleStatusReadinessEvidenceV1::new(
            "px4.vehicle-status.fixture.v1",
            10,
            1,
            14,
            14,
            true,
            false,
            false,
            0,
        )
        .unwrap();
        let vehicle_bytes = serde_json::to_vec(&vehicle).unwrap();
        let vehicle_restored: Px4VehicleStatusReadinessEvidenceV1 =
            serde_json::from_slice(&vehicle_bytes).unwrap();
        vehicle_restored.validate().unwrap();
        assert_eq!(vehicle_restored.status_id().unwrap(), vehicle.status_id().unwrap());

        let failsafe = Px4FailsafeFlagsReadinessEvidenceV1::new(
            "px4.failsafe-flags.fixture.v1",
            11,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
        )
        .unwrap();
        let failsafe_bytes = serde_json::to_vec(&failsafe).unwrap();
        let failsafe_restored: Px4FailsafeFlagsReadinessEvidenceV1 =
            serde_json::from_slice(&failsafe_bytes).unwrap();
        failsafe_restored.validate().unwrap();
        assert_eq!(failsafe_restored.status_id().unwrap(), failsafe.status_id().unwrap());

        let armed = Px4ActuatorArmedReadinessEvidenceV1::new(
            "px4.actuator-armed.fixture.v1",
            12,
            false,
            false,
            true,
            false,
            false,
            false,
            false,
        )
        .unwrap();
        let armed_bytes = serde_json::to_vec(&armed).unwrap();
        let armed_restored: Px4ActuatorArmedReadinessEvidenceV1 =
            serde_json::from_slice(&armed_bytes).unwrap();
        armed_restored.validate().unwrap();
        assert_eq!(armed_restored.status_id().unwrap(), armed.status_id().unwrap());
    }

    #[test]
    fn status_timestamp_requires_explicit_clock_identity() {
        let vehicle = Px4VehicleStatusReadinessEvidenceV1::new(
            "px4.vehicle-status.fixture.v1",
            42,
            1,
            14,
            14,
            true,
            false,
            false,
            0,
        )
        .unwrap();
        let observed = vehicle.observed_at(clock()).unwrap();
        assert_eq!(observed.clock_domain.as_str(), "px4.hrt");
        assert_eq!(observed.nanoseconds, 42_000);
    }
}
