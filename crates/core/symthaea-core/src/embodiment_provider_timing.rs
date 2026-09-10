// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Time-addressed lifecycle receipts for Embodiment v2 providers.
//!
//! [`crate::embodiment_provider::BoundEmbodimentLifecycleV1`] states a provider
//! lifecycle proposition for one bound instance. This module binds that exact
//! proposition to an observation timestamp in an explicit clock domain without
//! turning lifecycle state itself into a clock, transition policy, or authority
//! signal.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::embodiment_evidence::{EvidenceValidationError, TimestampV1};
use crate::embodiment_provider::{
    BoundEmbodimentLifecycleV1, ProviderBindingValidationError,
};

/// Schema version for [`TimedBoundEmbodimentLifecycleV1`].
pub const TIMED_BOUND_EMBODIMENT_LIFECYCLE_SCHEMA_V1: u16 = 1;
const TIMED_LIFECYCLE_DOMAIN_V1: &[u8] =
    b"symthaea.embodiment.bound-lifecycle-timed.v1\0";

/// One validated provider lifecycle proposition observed at an explicit time.
///
/// The timestamp makes lifecycle evidence orderable and ageable *within the same
/// clock domain*. Cross-clock comparison still requires separate clock-alignment
/// evidence; this type never assumes synchronization merely because two timestamps
/// are present.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimedBoundEmbodimentLifecycleV1 {
    /// Schema version. Must equal [`TIMED_BOUND_EMBODIMENT_LIFECYCLE_SCHEMA_V1`].
    pub schema_version: u16,
    /// Exact content-bound lifecycle proposition being timestamped.
    pub lifecycle: BoundEmbodimentLifecycleV1,
    /// Time at which the producer observed/established the lifecycle proposition.
    pub observed_at: TimestampV1,
    /// Domain-separated commitment over lifecycle identity and observation time.
    pub receipt_digest_hex: String,
}

impl TimedBoundEmbodimentLifecycleV1 {
    /// Construct, content-bind, and validate one timed lifecycle receipt.
    pub fn new(
        lifecycle: BoundEmbodimentLifecycleV1,
        observed_at: TimestampV1,
    ) -> Result<Self, TimedLifecycleValidationError> {
        let mut value = Self {
            schema_version: TIMED_BOUND_EMBODIMENT_LIFECYCLE_SCHEMA_V1,
            lifecycle,
            observed_at,
            receipt_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.receipt_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate nested lifecycle evidence, timestamp, schema, and commitment.
    pub fn validate(&self) -> Result<(), TimedLifecycleValidationError> {
        self.validate_without_digest()?;
        if self.receipt_digest_hex != self.compute_digest_hex() {
            return Err(TimedLifecycleValidationError::DigestMismatch);
        }
        Ok(())
    }

    /// Content-addressed identity of this exact timed lifecycle observation.
    pub fn receipt_id(&self) -> Result<String, TimedLifecycleValidationError> {
        self.validate()?;
        Ok(format!(
            "symthaea.embodiment.bound-lifecycle-timed.v1:{}",
            self.receipt_digest_hex
        ))
    }

    /// Age this observation at `now`, in nanoseconds, only when both timestamps
    /// share the same clock domain and time has not moved backward.
    pub fn age_at(&self, now: &TimestampV1) -> Result<u64, TimedLifecycleValidationError> {
        self.validate()?;
        now.elapsed_since(&self.observed_at)
            .map_err(TimedLifecycleValidationError::Timestamp)
    }

    /// Whether this lifecycle observation is no older than `max_age_ns` at
    /// `now`, under the same-clock rules enforced by [`Self::age_at`].
    pub fn is_fresh_at(
        &self,
        now: &TimestampV1,
        max_age_ns: u64,
    ) -> Result<bool, TimedLifecycleValidationError> {
        Ok(self.age_at(now)? <= max_age_ns)
    }

    fn validate_without_digest(&self) -> Result<(), TimedLifecycleValidationError> {
        if self.schema_version != TIMED_BOUND_EMBODIMENT_LIFECYCLE_SCHEMA_V1 {
            return Err(TimedLifecycleValidationError::UnsupportedSchemaVersion {
                found: self.schema_version,
            });
        }
        self.lifecycle
            .validate()
            .map_err(TimedLifecycleValidationError::Lifecycle)?;
        self.observed_at
            .validate()
            .map_err(TimedLifecycleValidationError::Timestamp)
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(TIMED_LIFECYCLE_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.lifecycle.lifecycle_digest_hex);
        feed_str(&mut hasher, self.observed_at.clock_domain.as_str());
        hasher.update(&self.observed_at.nanoseconds.to_le_bytes());
        digest_hex(hasher.finalize().as_bytes())
    }
}

fn feed_str(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn digest_hex(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

/// Validation failure for a timed provider lifecycle receipt.
#[derive(Debug, Clone, PartialEq)]
pub enum TimedLifecycleValidationError {
    /// Receipt uses an unsupported schema version.
    UnsupportedSchemaVersion {
        /// Unsupported version encountered.
        found: u16,
    },
    /// Nested provider lifecycle proposition is invalid.
    Lifecycle(ProviderBindingValidationError),
    /// Observation/current timestamp or same-clock age calculation is invalid.
    Timestamp(EvidenceValidationError),
    /// Stored content commitment does not match receipt fields.
    DigestMismatch,
}

impl std::fmt::Display for TimedLifecycleValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => {
                write!(f, "unsupported timed lifecycle schema version {found}")
            }
            Self::Lifecycle(error) => write!(f, "invalid lifecycle proposition: {error}"),
            Self::Timestamp(error) => write!(f, "invalid lifecycle timestamp: {error}"),
            Self::DigestMismatch => write!(f, "timed lifecycle content commitment mismatch"),
        }
    }
}

impl std::error::Error for TimedLifecycleValidationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embodiment_evidence::ClockDomainId;
    use crate::embodiment_provider::{
        BackendProviderId, BoundEmbodimentLifecycleStateV1, BoundEmbodimentLifecycleV1,
        EmbodimentBindingResultV1, EmbodimentBindingTargetV1, EmbodimentInstanceId,
        EmbodimentProfileId, PlatformFamilyId,
    };

    fn ts(clock: &str, nanoseconds: u64) -> TimestampV1 {
        TimestampV1::new(ClockDomainId::new(clock).unwrap(), nanoseconds)
    }

    fn ready_lifecycle() -> BoundEmbodimentLifecycleV1 {
        let target = EmbodimentBindingTargetV1::new(
            PlatformFamilyId::new("org.luminous.multirotor").unwrap(),
            EmbodimentProfileId::new("org.luminous.multirotor.quad-x.v1").unwrap(),
            BackendProviderId::new("org.luminous.px4.mavlink.v1").unwrap(),
        )
        .unwrap();
        let result = EmbodimentBindingResultV1::bound(
            target,
            EmbodimentInstanceId::new("px4:sysid-1:compid-1").unwrap(),
            "binding:px4:1",
        )
        .unwrap();
        BoundEmbodimentLifecycleV1::new(
            result.bound_identity().unwrap().unwrap(),
            BoundEmbodimentLifecycleStateV1::Ready,
            "readiness:px4:1",
        )
        .unwrap()
    }

    #[test]
    fn lifecycle_receipt_is_ageable_in_same_clock() {
        let receipt = TimedBoundEmbodimentLifecycleV1::new(
            ready_lifecycle(),
            ts("host.monotonic", 1_000),
        )
        .unwrap();

        assert_eq!(receipt.age_at(&ts("host.monotonic", 1_250)).unwrap(), 250);
        assert!(receipt.is_fresh_at(&ts("host.monotonic", 1_250), 250).unwrap());
        assert!(!receipt.is_fresh_at(&ts("host.monotonic", 1_251), 250).unwrap());
    }

    #[test]
    fn cross_clock_age_is_rejected() {
        let receipt = TimedBoundEmbodimentLifecycleV1::new(
            ready_lifecycle(),
            ts("host.monotonic", 1_000),
        )
        .unwrap();

        assert!(matches!(
            receipt.age_at(&ts("px4.hrt", 1_250)),
            Err(TimedLifecycleValidationError::Timestamp(
                EvidenceValidationError::ClockDomainMismatch
            ))
        ));
    }

    #[test]
    fn backward_time_is_rejected() {
        let receipt = TimedBoundEmbodimentLifecycleV1::new(
            ready_lifecycle(),
            ts("host.monotonic", 1_000),
        )
        .unwrap();

        assert!(matches!(
            receipt.age_at(&ts("host.monotonic", 999)),
            Err(TimedLifecycleValidationError::Timestamp(
                EvidenceValidationError::NonMonotonicTimestamp
            ))
        ));
    }

    #[test]
    fn observation_time_participates_in_identity() {
        let lifecycle = ready_lifecycle();
        let first = TimedBoundEmbodimentLifecycleV1::new(
            lifecycle.clone(),
            ts("host.monotonic", 1_000),
        )
        .unwrap();
        let second = TimedBoundEmbodimentLifecycleV1::new(
            lifecycle,
            ts("host.monotonic", 1_001),
        )
        .unwrap();

        assert_ne!(first.receipt_digest_hex, second.receipt_digest_hex);
    }

    #[test]
    fn semantically_valid_mutation_breaks_commitment() {
        let mut receipt = TimedBoundEmbodimentLifecycleV1::new(
            ready_lifecycle(),
            ts("host.monotonic", 1_000),
        )
        .unwrap();
        receipt.observed_at.nanoseconds = 1_001;

        assert_eq!(
            receipt.validate(),
            Err(TimedLifecycleValidationError::DigestMismatch)
        );
    }

    #[test]
    fn serde_round_trip_preserves_identity() {
        let receipt = TimedBoundEmbodimentLifecycleV1::new(
            ready_lifecycle(),
            ts("host.monotonic", 1_000),
        )
        .unwrap();
        let bytes = serde_json::to_vec(&receipt).unwrap();
        let restored: TimedBoundEmbodimentLifecycleV1 = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(restored, receipt);
        restored.validate().unwrap();
    }
}
