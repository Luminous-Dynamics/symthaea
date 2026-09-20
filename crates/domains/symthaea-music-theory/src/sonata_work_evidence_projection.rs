// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical Sonata adapter for the generic FORM-003 evidence projection.
//!
//! The adapter retains the complete [`crate::SonataRealization`] and
//! [`crate::sonata_work_evidence_coverage::SonataWorkEvidenceCoverageV1`].
//! Generic projection records are therefore an interchange/view layer over
//! canonically rederived Sonata evidence, never a replacement for it.

use crate::sonata::SonataRealization;
use crate::sonata_work_evidence::{
    NativeEvidencePreservationV1, SONATA_WORK_EVIDENCE_VERSION,
    SonataEvidenceProjectionLossV1, WorkEvidenceStatusV1,
};
use crate::sonata_work_evidence_coverage::{
    SONATA_WORK_EVIDENCE_COVERAGE_VERSION, SonataWorkEvidenceCoverageErrorV1,
    SonataWorkEvidenceCoverageV1, SonataWorkEvidenceSourceV1,
    derive_sonata_work_evidence_coverage,
};
use crate::work_obligation_evidence::{
    WORK_OBLIGATION_EVIDENCE_PROJECTION_VERSION, WorkObligationEvidenceDispositionV1,
    WorkObligationEvidencePreservationV1, WorkObligationEvidenceProjectionErrorV1,
    WorkObligationEvidenceProjectionLossV1, WorkObligationEvidenceProjectionRecordV1,
    WorkObligationEvidenceProjectionSetV1, WorkObligationEvidenceSourceIdentityV1,
    required_work_obligation_evidence_projection_nonclaims,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const SONATA_WORK_EVIDENCE_PROJECTION_VERSION: &str =
    "melothaea-sonata-work-evidence-projection-v1";
pub const SONATA_WORK_EVIDENCE_PROJECTION_PROFILE: &str =
    "melothaea-sonata-work-evidence-coverage-projection-profile-v1";
const SONATA_COVERAGE_NAMESPACE: &str = "melothaea-sonata-work-evidence-coverage";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SonataWorkObligationEvidenceProjectionV1 {
    pub version: String,
    /// Retained completed-score subject. Validation starts from this subject,
    /// not from serialized generic dispositions.
    pub realization: SonataRealization,
    /// Canonically rederived source artifact with native/thematic provenance.
    pub coverage: SonataWorkEvidenceCoverageV1,
    /// Generic FORM-level projection. This remains non-authoritative without
    /// the retained source artifact and adapter validation.
    pub projection: WorkObligationEvidenceProjectionSetV1,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SonataWorkEvidenceProjectionErrorV1 {
    WrongVersion { found: String },
    Coverage(SonataWorkEvidenceCoverageErrorV1),
    GenericShape(WorkObligationEvidenceProjectionErrorV1),
    CanonicalProjectionMismatch,
}

pub fn project_sonata_work_obligation_evidence(
    realization: &SonataRealization,
) -> Result<SonataWorkObligationEvidenceProjectionV1, SonataWorkEvidenceProjectionErrorV1> {
    let coverage = derive_sonata_work_evidence_coverage(realization)
        .map_err(SonataWorkEvidenceProjectionErrorV1::Coverage)?;
    let mut records = BTreeMap::new();

    for (obligation_id, source_record) in &coverage.records {
        let disposition = match source_record.status {
            WorkEvidenceStatusV1::Supported => {
                WorkObligationEvidenceDispositionV1::PositiveEvidenceUnderProfile
            }
            WorkEvidenceStatusV1::Failed => {
                WorkObligationEvidenceDispositionV1::NegativeEvidenceUnderProfile
            }
            WorkEvidenceStatusV1::NotMeasured => {
                WorkObligationEvidenceDispositionV1::NotMeasuredUnderProfile
            }
        };

        let (preservation, projection_losses) = projection_preservation(&source_record.source);
        let record = WorkObligationEvidenceProjectionRecordV1 {
            obligation_id: obligation_id.clone(),
            source: WorkObligationEvidenceSourceIdentityV1 {
                namespace: SONATA_COVERAGE_NAMESPACE.into(),
                version: SONATA_WORK_EVIDENCE_COVERAGE_VERSION.into(),
                profile_id: SONATA_WORK_EVIDENCE_PROJECTION_PROFILE.into(),
                record_id: obligation_id.clone(),
            },
            disposition,
            preservation,
            projection_losses,
        };
        records.insert(obligation_id.clone(), record);
    }

    let projection = WorkObligationEvidenceProjectionSetV1 {
        version: WORK_OBLIGATION_EVIDENCE_PROJECTION_VERSION.into(),
        records,
        nonclaims: required_work_obligation_evidence_projection_nonclaims(),
    };
    projection
        .validate_shape()
        .map_err(SonataWorkEvidenceProjectionErrorV1::GenericShape)?;

    Ok(SonataWorkObligationEvidenceProjectionV1 {
        version: SONATA_WORK_EVIDENCE_PROJECTION_VERSION.into(),
        realization: realization.clone(),
        coverage,
        projection,
    })
}

impl SonataWorkObligationEvidenceProjectionV1 {
    pub fn validate(&self) -> Result<(), SonataWorkEvidenceProjectionErrorV1> {
        if self.version != SONATA_WORK_EVIDENCE_PROJECTION_VERSION {
            return Err(SonataWorkEvidenceProjectionErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        self.projection
            .validate_shape()
            .map_err(SonataWorkEvidenceProjectionErrorV1::GenericShape)?;
        let canonical = project_sonata_work_obligation_evidence(&self.realization)?;
        if &canonical != self {
            return Err(SonataWorkEvidenceProjectionErrorV1::CanonicalProjectionMismatch);
        }
        Ok(())
    }
}

fn projection_preservation(
    source: &SonataWorkEvidenceSourceV1,
) -> (
    WorkObligationEvidencePreservationV1,
    Vec<WorkObligationEvidenceProjectionLossV1>,
) {
    match source {
        SonataWorkEvidenceSourceV1::NativeSonata {
            preservation:
                NativeEvidencePreservationV1::Projected {
                    loss: SonataEvidenceProjectionLossV1::CadentialSpecificityToTonalCenter,
                },
            ..
        } => (
            WorkObligationEvidencePreservationV1::Projected,
            vec![WorkObligationEvidenceProjectionLossV1 {
                namespace: SONATA_WORK_EVIDENCE_VERSION.into(),
                code: "cadential-specificity-to-tonal-center".into(),
            }],
        ),
        _ => (WorkObligationEvidencePreservationV1::Exact, vec![]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Duration, Key, Motif, MusicalIntent, PitchClass, SonataSectionKind, VoiceRole,
        realize_sonata_with_plan,
    };

    fn theme() -> Motif {
        Motif::from_degrees(&[
            (1, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
            (2, Duration::quarter()),
        ])
    }

    fn realization() -> SonataRealization {
        realize_sonata_with_plan(
            Key::major(PitchClass::C),
            96.0,
            4.0,
            &theme(),
            11,
            &MusicalIntent::default(),
        )
    }

    #[test]
    fn all_nine_sonata_promises_project_without_losing_native_artifact() {
        let projected = project_sonata_work_obligation_evidence(&realization()).unwrap();
        assert_eq!(projected.coverage.records.len(), 9);
        assert_eq!(projected.projection.records.len(), 9);
        assert!(projected.projection.records.values().all(|record| {
            record.source.namespace == SONATA_COVERAGE_NAMESPACE
                && record.source.version == SONATA_WORK_EVIDENCE_COVERAGE_VERSION
                && record.source.profile_id == SONATA_WORK_EVIDENCE_PROJECTION_PROFILE
        }));
        projected.validate().unwrap();
    }

    #[test]
    fn cadence_specificity_loss_survives_generic_projection() {
        let projected = project_sonata_work_obligation_evidence(&realization()).unwrap();
        let close = &projected.projection.records["sonata:close-home"];
        assert_eq!(
            close.preservation,
            WorkObligationEvidencePreservationV1::Projected
        );
        assert_eq!(
            close.projection_losses,
            vec![WorkObligationEvidenceProjectionLossV1 {
                namespace: SONATA_WORK_EVIDENCE_VERSION.into(),
                code: "cadential-specificity-to-tonal-center".into(),
            }]
        );
        assert!(projected
            .projection
            .records
            .iter()
            .filter(|(id, _)| id.as_str() != "sonata:close-home")
            .all(|(_, record)| record.preservation == WorkObligationEvidencePreservationV1::Exact));
    }

    #[test]
    fn completed_score_damage_projects_as_negative_evidence() {
        let mut realization = realization();
        let secondary = realization
            .plan
            .sections
            .iter()
            .find(|section| section.kind == SonataSectionKind::ExpositionSecondary)
            .unwrap()
            .clone();
        realization.score.notes.retain(|note| {
            note.role != VoiceRole::Melody
                || note.onset.beats() < secondary.start.beats()
                || note.onset.beats() >= secondary.end.beats()
        });
        let projected = project_sonata_work_obligation_evidence(&realization).unwrap();
        assert_eq!(
            projected.projection.records["sonata:establish-secondary"].disposition,
            WorkObligationEvidenceDispositionV1::NegativeEvidenceUnderProfile
        );
    }

    #[test]
    fn hand_edited_generic_disposition_cannot_acquire_sonata_authority() {
        let mut projected = project_sonata_work_obligation_evidence(&realization()).unwrap();
        projected
            .projection
            .records
            .get_mut("sonata:establish-primary")
            .unwrap()
            .disposition = WorkObligationEvidenceDispositionV1::NegativeEvidenceUnderProfile;
        assert_eq!(
            projected.validate(),
            Err(SonataWorkEvidenceProjectionErrorV1::CanonicalProjectionMismatch)
        );
    }
}
