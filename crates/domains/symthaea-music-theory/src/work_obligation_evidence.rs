// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generic, non-authoritative projection envelope for FORM-003 evidence.
//!
//! Native form engines keep their full evidence artifacts and semantics. This
//! module normalizes only the fields that are genuinely shared when those
//! artifacts are cited at the [`crate::work_obligation::WorkObligationPlanV2`]
//! boundary: obligation identity, source/profile identity, disposition, and
//! exact-vs-projected preservation.
//!
//! `validate_shape()` validates only this envelope. It deliberately cannot
//! establish that the source evidence exists, is canonical, or supports the
//! projected disposition. Source-specific adapters must rederive their native
//! evidence and retain that artifact alongside this projection.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[path = "work_obligation_evidence_observatory.rs"]
pub mod observatory;
#[path = "prog_suite_work_evidence_projection.rs"]
pub mod prog_suite_projection;
#[path = "sonata_work_evidence_projection.rs"]
pub mod sonata_projection;

pub const WORK_OBLIGATION_EVIDENCE_PROJECTION_VERSION: &str =
    "melothaea-work-obligation-evidence-projection-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkObligationEvidenceDispositionV1 {
    PositiveEvidenceUnderProfile,
    NegativeEvidenceUnderProfile,
    IndeterminateEvidenceUnderProfile,
    NotMeasuredUnderProfile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkObligationEvidencePreservationV1 {
    /// The adapter asserts no semantic loss between the source evidence claim
    /// and this generic obligation-level disposition.
    Exact,
    /// The adapter declares one or more explicit, namespaced semantic losses.
    Projected,
}

#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct WorkObligationEvidenceProjectionLossV1 {
    /// Stable source vocabulary namespace, for example
    /// `melothaea-sonata-work-evidence-v1`.
    pub namespace: String,
    /// Stable loss code within that namespace.
    pub code: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkObligationEvidenceSourceIdentityV1 {
    /// Evidence-family namespace. This identifies semantics, not a Rust type.
    pub namespace: String,
    /// Exact source artifact/schema version.
    pub version: String,
    /// Exact profile used to interpret the source evidence. Native evidence
    /// without a separate profile still supplies an explicit adapter profile.
    pub profile_id: String,
    /// Stable record identity inside the retained native artifact.
    pub record_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkObligationEvidenceProjectionRecordV1 {
    pub obligation_id: String,
    pub source: WorkObligationEvidenceSourceIdentityV1,
    pub disposition: WorkObligationEvidenceDispositionV1,
    pub preservation: WorkObligationEvidencePreservationV1,
    /// Strictly sorted and unique. Empty exactly when `preservation == Exact`.
    pub projection_losses: Vec<WorkObligationEvidenceProjectionLossV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkObligationEvidenceProjectionNonClaimV1 {
    ShapeValidationDoesNotEstablishSourceEvidence,
    ProjectionDoesNotResolveWorkObligation,
    ProjectionDoesNotMutateWorkObligationPlan,
    ProjectionDoesNotEstablishUniversalSatisfaction,
    ProjectionDoesNotRankEvidenceProfiles,
    ProjectionDoesNotEstablishListenerPerception,
    ProjectionDoesNotEstablishArtisticQuality,
    ProjectionDoesNotGrantProductAuthority,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkObligationEvidenceProjectionSetV1 {
    pub version: String,
    /// Canonical obligation-ID map. Source-specific adapters decide which
    /// obligations they can project; absence must never be interpreted as pass.
    pub records: BTreeMap<String, WorkObligationEvidenceProjectionRecordV1>,
    pub nonclaims: Vec<WorkObligationEvidenceProjectionNonClaimV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkObligationEvidenceProjectionErrorV1 {
    WrongVersion { found: String },
    EmptyObligationId,
    RecordKeyMismatch { key: String, record_id: String },
    EmptySourceNamespace { obligation_id: String },
    EmptySourceVersion { obligation_id: String },
    EmptySourceProfile { obligation_id: String },
    EmptySourceRecordId { obligation_id: String },
    EmptyLossNamespace { obligation_id: String },
    EmptyLossCode { obligation_id: String },
    NonCanonicalProjectionLosses { obligation_id: String },
    ExactProjectionHasLosses { obligation_id: String },
    ProjectedEvidenceHasNoLosses { obligation_id: String },
    NonCanonicalNonClaims,
}

impl WorkObligationEvidenceProjectionSetV1 {
    /// Validate only the generic projection envelope.
    ///
    /// A successful result is intentionally *not* source-evidence validation.
    /// Source-specific adapters must rederive their retained source artifact and
    /// compare the canonical projection separately.
    pub fn validate_shape(&self) -> Result<(), WorkObligationEvidenceProjectionErrorV1> {
        if self.version != WORK_OBLIGATION_EVIDENCE_PROJECTION_VERSION {
            return Err(WorkObligationEvidenceProjectionErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        if self.nonclaims != required_nonclaims() {
            return Err(WorkObligationEvidenceProjectionErrorV1::NonCanonicalNonClaims);
        }

        for (key, record) in &self.records {
            if key.trim().is_empty() || record.obligation_id.trim().is_empty() {
                return Err(WorkObligationEvidenceProjectionErrorV1::EmptyObligationId);
            }
            if key != &record.obligation_id {
                return Err(WorkObligationEvidenceProjectionErrorV1::RecordKeyMismatch {
                    key: key.clone(),
                    record_id: record.obligation_id.clone(),
                });
            }
            validate_source(record)?;
            validate_losses(record)?;
        }
        Ok(())
    }
}

fn validate_source(
    record: &WorkObligationEvidenceProjectionRecordV1,
) -> Result<(), WorkObligationEvidenceProjectionErrorV1> {
    if record.source.namespace.trim().is_empty() {
        return Err(WorkObligationEvidenceProjectionErrorV1::EmptySourceNamespace {
            obligation_id: record.obligation_id.clone(),
        });
    }
    if record.source.version.trim().is_empty() {
        return Err(WorkObligationEvidenceProjectionErrorV1::EmptySourceVersion {
            obligation_id: record.obligation_id.clone(),
        });
    }
    if record.source.profile_id.trim().is_empty() {
        return Err(WorkObligationEvidenceProjectionErrorV1::EmptySourceProfile {
            obligation_id: record.obligation_id.clone(),
        });
    }
    if record.source.record_id.trim().is_empty() {
        return Err(WorkObligationEvidenceProjectionErrorV1::EmptySourceRecordId {
            obligation_id: record.obligation_id.clone(),
        });
    }
    Ok(())
}

fn validate_losses(
    record: &WorkObligationEvidenceProjectionRecordV1,
) -> Result<(), WorkObligationEvidenceProjectionErrorV1> {
    for loss in &record.projection_losses {
        if loss.namespace.trim().is_empty() {
            return Err(WorkObligationEvidenceProjectionErrorV1::EmptyLossNamespace {
                obligation_id: record.obligation_id.clone(),
            });
        }
        if loss.code.trim().is_empty() {
            return Err(WorkObligationEvidenceProjectionErrorV1::EmptyLossCode {
                obligation_id: record.obligation_id.clone(),
            });
        }
    }
    if record
        .projection_losses
        .windows(2)
        .any(|pair| pair[0] >= pair[1])
    {
        return Err(
            WorkObligationEvidenceProjectionErrorV1::NonCanonicalProjectionLosses {
                obligation_id: record.obligation_id.clone(),
            },
        );
    }
    match record.preservation {
        WorkObligationEvidencePreservationV1::Exact if !record.projection_losses.is_empty() => {
            Err(WorkObligationEvidenceProjectionErrorV1::ExactProjectionHasLosses {
                obligation_id: record.obligation_id.clone(),
            })
        }
        WorkObligationEvidencePreservationV1::Projected if record.projection_losses.is_empty() => {
            Err(
                WorkObligationEvidenceProjectionErrorV1::ProjectedEvidenceHasNoLosses {
                    obligation_id: record.obligation_id.clone(),
                },
            )
        }
        _ => Ok(()),
    }
}

pub fn required_work_obligation_evidence_projection_nonclaims(
) -> Vec<WorkObligationEvidenceProjectionNonClaimV1> {
    required_nonclaims()
}

fn required_nonclaims() -> Vec<WorkObligationEvidenceProjectionNonClaimV1> {
    vec![
        WorkObligationEvidenceProjectionNonClaimV1::ShapeValidationDoesNotEstablishSourceEvidence,
        WorkObligationEvidenceProjectionNonClaimV1::ProjectionDoesNotResolveWorkObligation,
        WorkObligationEvidenceProjectionNonClaimV1::ProjectionDoesNotMutateWorkObligationPlan,
        WorkObligationEvidenceProjectionNonClaimV1::ProjectionDoesNotEstablishUniversalSatisfaction,
        WorkObligationEvidenceProjectionNonClaimV1::ProjectionDoesNotRankEvidenceProfiles,
        WorkObligationEvidenceProjectionNonClaimV1::ProjectionDoesNotEstablishListenerPerception,
        WorkObligationEvidenceProjectionNonClaimV1::ProjectionDoesNotEstablishArtisticQuality,
        WorkObligationEvidenceProjectionNonClaimV1::ProjectionDoesNotGrantProductAuthority,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> WorkObligationEvidenceSourceIdentityV1 {
        WorkObligationEvidenceSourceIdentityV1 {
            namespace: "example-evidence".into(),
            version: "example-v1".into(),
            profile_id: "example-profile-v1".into(),
            record_id: "native:42".into(),
        }
    }

    fn exact_record() -> WorkObligationEvidenceProjectionRecordV1 {
        WorkObligationEvidenceProjectionRecordV1 {
            obligation_id: "obligation:a".into(),
            source: source(),
            disposition: WorkObligationEvidenceDispositionV1::PositiveEvidenceUnderProfile,
            preservation: WorkObligationEvidencePreservationV1::Exact,
            projection_losses: vec![],
        }
    }

    fn set_with(
        record: WorkObligationEvidenceProjectionRecordV1,
    ) -> WorkObligationEvidenceProjectionSetV1 {
        WorkObligationEvidenceProjectionSetV1 {
            version: WORK_OBLIGATION_EVIDENCE_PROJECTION_VERSION.into(),
            records: [(record.obligation_id.clone(), record)].into_iter().collect(),
            nonclaims: required_nonclaims(),
        }
    }

    #[test]
    fn exact_projection_requires_no_losses() {
        set_with(exact_record()).validate_shape().unwrap();
        let mut record = exact_record();
        record.projection_losses.push(WorkObligationEvidenceProjectionLossV1 {
            namespace: "example".into(),
            code: "loss".into(),
        });
        assert_eq!(
            set_with(record).validate_shape(),
            Err(WorkObligationEvidenceProjectionErrorV1::ExactProjectionHasLosses {
                obligation_id: "obligation:a".into(),
            })
        );
    }

    #[test]
    fn projected_evidence_requires_explicit_canonical_losses() {
        let mut record = exact_record();
        record.preservation = WorkObligationEvidencePreservationV1::Projected;
        assert_eq!(
            set_with(record.clone()).validate_shape(),
            Err(WorkObligationEvidenceProjectionErrorV1::ProjectedEvidenceHasNoLosses {
                obligation_id: "obligation:a".into(),
            })
        );
        record.projection_losses = vec![
            WorkObligationEvidenceProjectionLossV1 {
                namespace: "z".into(),
                code: "loss".into(),
            },
            WorkObligationEvidenceProjectionLossV1 {
                namespace: "a".into(),
                code: "loss".into(),
            },
        ];
        assert_eq!(
            set_with(record).validate_shape(),
            Err(
                WorkObligationEvidenceProjectionErrorV1::NonCanonicalProjectionLosses {
                    obligation_id: "obligation:a".into(),
                }
            )
        );
    }

    #[test]
    fn obligation_key_and_record_identity_must_match() {
        let record = exact_record();
        let set = WorkObligationEvidenceProjectionSetV1 {
            version: WORK_OBLIGATION_EVIDENCE_PROJECTION_VERSION.into(),
            records: [("other".into(), record)].into_iter().collect(),
            nonclaims: required_nonclaims(),
        };
        assert_eq!(
            set.validate_shape(),
            Err(WorkObligationEvidenceProjectionErrorV1::RecordKeyMismatch {
                key: "other".into(),
                record_id: "obligation:a".into(),
            })
        );
    }

    #[test]
    fn envelope_validation_keeps_non_authority_boundary_explicit() {
        let set = set_with(exact_record());
        set.validate_shape().unwrap();
        assert!(set.nonclaims.contains(
            &WorkObligationEvidenceProjectionNonClaimV1::ShapeValidationDoesNotEstablishSourceEvidence
        ));
        assert!(set.nonclaims.contains(
            &WorkObligationEvidenceProjectionNonClaimV1::ProjectionDoesNotResolveWorkObligation
        ));
        assert!(set.nonclaims.contains(
            &WorkObligationEvidenceProjectionNonClaimV1::ProjectionDoesNotRankEvidenceProfiles
        ));
    }
}
