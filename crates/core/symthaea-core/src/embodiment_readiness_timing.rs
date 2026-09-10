// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Time-addressed provider-readiness assessments for Embodiment v2.
//!
//! [`crate::embodiment_readiness::ProviderReadinessAssessmentV1`] establishes
//! *which prerequisite states were assessed*. This module binds that exact
//! assessment to the time at which the provider performed the assessment.
//!
//! The key invariant is that a fresh lifecycle timestamp cannot re-fresh an old
//! readiness assessment. The safe bridge to
//! [`crate::embodiment_provider_timing::TimedBoundEmbodimentLifecycleV1`]
//! requires an explicit freshness gate and preserves the exact assessment
//! timestamp rather than accepting a caller-supplied replacement time.
//!
//! This does not independently validate the evidence objects referenced by each
//! readiness requirement. Product/provider qualification remains responsible for
//! evaluating those evidence objects at assessment time and marking stale or
//! unknown prerequisites accordingly.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::embodiment_evidence::{EvidenceValidationError, TimestampV1};
use crate::embodiment_provider::{
    BoundEmbodimentLifecycleStateV1, BoundEmbodimentLifecycleV1,
    ProviderBindingValidationError,
};
use crate::embodiment_provider_timing::{
    TimedBoundEmbodimentLifecycleV1, TimedLifecycleValidationError,
};
use crate::embodiment_readiness::{
    ProviderReadinessAssessmentV1, ReadinessValidationError,
};

/// Schema version for [`TimedProviderReadinessAssessmentV1`].
pub const TIMED_PROVIDER_READINESS_SCHEMA_V1: u16 = 1;
const TIMED_READINESS_DOMAIN_V1: &[u8] =
    b"symthaea.embodiment.readiness-assessment-timed.v1\0";

/// One exact readiness assessment observed/performed at an explicit time.
///
/// The timestamp is part of the content commitment. Re-wrapping the same
/// assessment at a later time creates a different evidence object; callers that
/// need to establish current readiness must perform a new assessment rather than
/// merely re-stamp an old one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimedProviderReadinessAssessmentV1 {
    /// Schema version. Must equal [`TIMED_PROVIDER_READINESS_SCHEMA_V1`].
    pub schema_version: u16,
    /// Exact provider-readiness assessment being time-addressed.
    pub assessment: ProviderReadinessAssessmentV1,
    /// Time at which the provider evaluated the prerequisite set.
    pub assessed_at: TimestampV1,
    /// Domain-separated content commitment over assessment identity and time.
    pub receipt_digest_hex: String,
}

impl TimedProviderReadinessAssessmentV1 {
    /// Construct, content-bind, and validate one timed readiness assessment.
    pub fn new(
        assessment: ProviderReadinessAssessmentV1,
        assessed_at: TimestampV1,
    ) -> Result<Self, TimedReadinessValidationError> {
        let mut value = Self {
            schema_version: TIMED_PROVIDER_READINESS_SCHEMA_V1,
            assessment,
            assessed_at,
            receipt_digest_hex: String::new(),
        };
        value.validate_without_digest()?;
        value.receipt_digest_hex = value.compute_digest_hex();
        value.validate()?;
        Ok(value)
    }

    /// Validate nested readiness evidence, timestamp, schema, and commitment.
    pub fn validate(&self) -> Result<(), TimedReadinessValidationError> {
        self.validate_without_digest()?;
        if self.receipt_digest_hex != self.compute_digest_hex() {
            return Err(TimedReadinessValidationError::DigestMismatch);
        }
        Ok(())
    }

    /// Content-addressed identity of this exact time-addressed assessment.
    pub fn receipt_id(&self) -> Result<String, TimedReadinessValidationError> {
        self.validate()?;
        Ok(format!(
            "symthaea.embodiment.readiness-assessment-timed.v1:{}",
            self.receipt_digest_hex
        ))
    }

    /// Age this assessment at `now`, in nanoseconds, only within the same clock
    /// domain and with monotonic time.
    pub fn age_at(&self, now: &TimestampV1) -> Result<u64, TimedReadinessValidationError> {
        self.validate()?;
        now.elapsed_since(&self.assessed_at)
            .map_err(TimedReadinessValidationError::Timestamp)
    }

    /// Whether this assessment is no older than `max_age_ns` at `now`.
    ///
    /// This ages the assessment as a whole. It does not replace provider-specific
    /// freshness validation of each referenced prerequisite evidence object.
    pub fn is_fresh_at(
        &self,
        now: &TimestampV1,
        max_age_ns: u64,
    ) -> Result<bool, TimedReadinessValidationError> {
        Ok(self.age_at(now)? <= max_age_ns)
    }

    /// Build a time-addressed `Ready` lifecycle proposition only when this exact
    /// readiness assessment is still fresh at `now`.
    ///
    /// The lifecycle observation time is *exactly* [`Self::assessed_at`]. There
    /// is intentionally no lifecycle timestamp parameter: callers cannot make an
    /// old assessment look fresh by attaching a newer timestamp. The lifecycle's
    /// evidence reference points to this **timed readiness receipt**, not merely
    /// the nested untimed assessment.
    ///
    /// `max_age_ns` is a caller/provider policy input. The returned lifecycle
    /// retains the original assessment timestamp so downstream consumers can
    /// independently apply their own freshness policy.
    pub fn ready_lifecycle_receipt(
        &self,
        now: &TimestampV1,
        max_age_ns: u64,
    ) -> Result<TimedBoundEmbodimentLifecycleV1, TimedReadinessValidationError> {
        self.validate()?;
        let age_ns = self.age_at(now)?;
        if age_ns > max_age_ns {
            return Err(TimedReadinessValidationError::AssessmentTooOld {
                age_ns,
                max_age_ns,
            });
        }

        // Reuse the readiness assessment's own graduation gate so a NotReady or
        // Incomplete assessment cannot reach the lifecycle constructor.
        let ready = self
            .assessment
            .ready_lifecycle()
            .map_err(TimedReadinessValidationError::Readiness)?;

        // Rebuild the descriptive lifecycle proposition so its evidence reference
        // is the timed readiness receipt, not the untimed assessment. This binds
        // the readiness reason and the assessment time into the evidence chain.
        let lifecycle = BoundEmbodimentLifecycleV1::new(
            ready.identity,
            BoundEmbodimentLifecycleStateV1::Ready,
            self.receipt_id()?,
        )
        .map_err(TimedReadinessValidationError::Provider)?;

        let receipt = TimedBoundEmbodimentLifecycleV1::new(
            lifecycle,
            self.assessed_at.clone(),
        )
        .map_err(TimedReadinessValidationError::LifecycleTiming)?;

        // Defense in depth: the lifecycle timing helper must preserve the exact
        // assessment timestamp, not merely an equivalent numeric value in a
        // different clock domain.
        if receipt.observed_at != self.assessed_at {
            return Err(TimedReadinessValidationError::TimestampMismatch);
        }
        Ok(receipt)
    }

    fn validate_without_digest(&self) -> Result<(), TimedReadinessValidationError> {
        if self.schema_version != TIMED_PROVIDER_READINESS_SCHEMA_V1 {
            return Err(TimedReadinessValidationError::UnsupportedSchemaVersion {
                found: self.schema_version,
            });
        }
        self.assessment
            .validate()
            .map_err(TimedReadinessValidationError::Readiness)?;
        self.assessed_at
            .validate()
            .map_err(TimedReadinessValidationError::Timestamp)
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(TIMED_READINESS_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.assessment.assessment_digest_hex);
        feed_str(&mut hasher, self.assessed_at.clock_domain.as_str());
        hasher.update(&self.assessed_at.nanoseconds.to_le_bytes());
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

/// Validation failure for a timed provider-readiness assessment.
#[derive(Debug)]
pub enum TimedReadinessValidationError {
    /// Receipt uses an unsupported schema version.
    UnsupportedSchemaVersion {
        /// Unsupported version encountered.
        found: u16,
    },
    /// Nested readiness assessment is invalid or not ready when a Ready lifecycle is requested.
    Readiness(ReadinessValidationError),
    /// Provider lifecycle construction failed.
    Provider(ProviderBindingValidationError),
    /// Assessment/current timestamp or same-clock age calculation is invalid.
    Timestamp(EvidenceValidationError),
    /// The readiness assessment is older than the caller/provider freshness policy allows.
    AssessmentTooOld {
        /// Observed assessment age at the attempted graduation time.
        age_ns: u64,
        /// Maximum age permitted by the caller/provider policy.
        max_age_ns: u64,
    },
    /// Construction of the timed lifecycle receipt failed.
    LifecycleTiming(TimedLifecycleValidationError),
    /// Defensive check found that lifecycle time diverged from assessment time.
    TimestampMismatch,
    /// Stored content commitment does not match receipt fields.
    DigestMismatch,
}

impl std::fmt::Display for TimedReadinessValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => {
                write!(f, "unsupported timed readiness schema version {found}")
            }
            Self::Readiness(error) => write!(f, "invalid readiness assessment: {error}"),
            Self::Provider(error) => write!(f, "invalid provider lifecycle: {error}"),
            Self::Timestamp(error) => write!(f, "invalid readiness timestamp: {error}"),
            Self::AssessmentTooOld { age_ns, max_age_ns } => write!(
                f,
                "readiness assessment age {age_ns}ns exceeds maximum {max_age_ns}ns"
            ),
            Self::LifecycleTiming(error) => write!(f, "invalid timed lifecycle receipt: {error}"),
            Self::TimestampMismatch => {
                write!(f, "ready lifecycle timestamp differs from readiness assessment time")
            }
            Self::DigestMismatch => write!(f, "timed readiness content commitment mismatch"),
        }
    }
}

impl std::error::Error for TimedReadinessValidationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embodiment_evidence::ClockDomainId;
    use crate::embodiment_provider::{
        BackendProviderId, EmbodimentBindingResultV1, EmbodimentBindingTargetV1,
        EmbodimentInstanceId, EmbodimentProfileId, PlatformFamilyId,
    };
    use crate::embodiment_readiness::{
        ProviderReadinessAssessmentV1, ProviderReadinessProfileV1,
        ProviderReadinessRequirementV1, ReadinessRequirementId,
        ReadinessRequirementStateV1,
    };

    fn ts(clock: &str, nanoseconds: u64) -> TimestampV1 {
        TimestampV1::new(ClockDomainId::new(clock).unwrap(), nanoseconds)
    }

    fn assessment(state: ReadinessRequirementStateV1) -> ProviderReadinessAssessmentV1 {
        let provider = BackendProviderId::new("org.luminous.px4.mavlink.v1").unwrap();
        let target = EmbodimentBindingTargetV1::new(
            PlatformFamilyId::new("org.luminous.multirotor").unwrap(),
            EmbodimentProfileId::new("org.luminous.multirotor.quad-x.v1").unwrap(),
            provider.clone(),
        )
        .unwrap();
        let bound = EmbodimentBindingResultV1::bound(
            target,
            EmbodimentInstanceId::new("px4:sysid-1:compid-1").unwrap(),
            "binding:px4:1",
        )
        .unwrap()
        .bound_identity()
        .unwrap()
        .unwrap();

        let requirement_id = ReadinessRequirementId::new("timesync_current").unwrap();
        let profile = ProviderReadinessProfileV1::new(
            provider,
            "org.luminous.px4.readiness.fixture.v1",
            vec![requirement_id.clone()],
        )
        .unwrap();
        let evidence_id = if matches!(state, ReadinessRequirementStateV1::Unknown) {
            None
        } else {
            Some("evidence:timesync:1".to_string())
        };
        let requirement = ProviderReadinessRequirementV1::new(
            requirement_id,
            state,
            evidence_id,
        )
        .unwrap();

        ProviderReadinessAssessmentV1::new(bound, profile, vec![requirement]).unwrap()
    }

    #[test]
    fn ready_lifecycle_preserves_exact_assessment_timestamp_and_timed_evidence() {
        let timed = TimedProviderReadinessAssessmentV1::new(
            assessment(ReadinessRequirementStateV1::Satisfied),
            ts("host.monotonic", 1_000),
        )
        .unwrap();
        let now = ts("host.monotonic", 1_100);

        let lifecycle = timed.ready_lifecycle_receipt(&now, 100).unwrap();
        assert_eq!(lifecycle.observed_at, timed.assessed_at);
        assert_eq!(lifecycle.lifecycle.evidence_id, timed.receipt_id().unwrap());
    }

    #[test]
    fn old_assessment_cannot_graduate_under_freshness_policy() {
        let timed = TimedProviderReadinessAssessmentV1::new(
            assessment(ReadinessRequirementStateV1::Satisfied),
            ts("host.monotonic", 1_000),
        )
        .unwrap();
        let now = ts("host.monotonic", 2_000);

        assert_eq!(timed.age_at(&now).unwrap(), 1_000);
        assert!(matches!(
            timed.ready_lifecycle_receipt(&now, 999),
            Err(TimedReadinessValidationError::AssessmentTooOld {
                age_ns: 1_000,
                max_age_ns: 999
            })
        ));
    }

    #[test]
    fn allowed_old_assessment_keeps_original_timestamp_in_lifecycle() {
        let timed = TimedProviderReadinessAssessmentV1::new(
            assessment(ReadinessRequirementStateV1::Satisfied),
            ts("host.monotonic", 1_000),
        )
        .unwrap();
        let now = ts("host.monotonic", 2_000);

        let lifecycle = timed.ready_lifecycle_receipt(&now, 1_000).unwrap();
        assert_eq!(lifecycle.observed_at, ts("host.monotonic", 1_000));
        assert_eq!(lifecycle.lifecycle.evidence_id, timed.receipt_id().unwrap());
    }

    #[test]
    fn non_ready_assessment_cannot_create_ready_lifecycle() {
        let timed = TimedProviderReadinessAssessmentV1::new(
            assessment(ReadinessRequirementStateV1::Unsatisfied),
            ts("host.monotonic", 1_000),
        )
        .unwrap();

        assert!(matches!(
            timed.ready_lifecycle_receipt(&ts("host.monotonic", 1_000), 0),
            Err(TimedReadinessValidationError::Readiness(_))
        ));
    }

    #[test]
    fn cross_clock_age_is_rejected() {
        let timed = TimedProviderReadinessAssessmentV1::new(
            assessment(ReadinessRequirementStateV1::Satisfied),
            ts("host.monotonic", 1_000),
        )
        .unwrap();

        assert!(matches!(
            timed.age_at(&ts("px4.hrt", 1_500)),
            Err(TimedReadinessValidationError::Timestamp(
                EvidenceValidationError::ClockDomainMismatch
            ))
        ));
        assert!(matches!(
            timed.ready_lifecycle_receipt(&ts("px4.hrt", 1_500), 1_000),
            Err(TimedReadinessValidationError::Timestamp(
                EvidenceValidationError::ClockDomainMismatch
            ))
        ));
    }

    #[test]
    fn backward_time_is_rejected() {
        let timed = TimedProviderReadinessAssessmentV1::new(
            assessment(ReadinessRequirementStateV1::Satisfied),
            ts("host.monotonic", 1_000),
        )
        .unwrap();

        assert!(matches!(
            timed.age_at(&ts("host.monotonic", 999)),
            Err(TimedReadinessValidationError::Timestamp(
                EvidenceValidationError::NonMonotonicTimestamp
            ))
        ));
    }

    #[test]
    fn assessment_time_participates_in_identity() {
        let assessment = assessment(ReadinessRequirementStateV1::Satisfied);
        let first = TimedProviderReadinessAssessmentV1::new(
            assessment.clone(),
            ts("host.monotonic", 1_000),
        )
        .unwrap();
        let second = TimedProviderReadinessAssessmentV1::new(
            assessment,
            ts("host.monotonic", 1_001),
        )
        .unwrap();

        assert_ne!(first.receipt_digest_hex, second.receipt_digest_hex);
    }

    #[test]
    fn semantically_valid_mutation_breaks_commitment() {
        let mut timed = TimedProviderReadinessAssessmentV1::new(
            assessment(ReadinessRequirementStateV1::Satisfied),
            ts("host.monotonic", 1_000),
        )
        .unwrap();
        timed.assessed_at.nanoseconds = 1_001;

        assert!(matches!(
            timed.validate(),
            Err(TimedReadinessValidationError::DigestMismatch)
        ));
    }

    #[test]
    fn serde_round_trip_preserves_identity() {
        let timed = TimedProviderReadinessAssessmentV1::new(
            assessment(ReadinessRequirementStateV1::Satisfied),
            ts("host.monotonic", 1_000),
        )
        .unwrap();
        let bytes = serde_json::to_vec(&timed).unwrap();
        let restored: TimedProviderReadinessAssessmentV1 =
            serde_json::from_slice(&bytes).unwrap();
        assert_eq!(restored, timed);
        restored.validate().unwrap();
    }
}
