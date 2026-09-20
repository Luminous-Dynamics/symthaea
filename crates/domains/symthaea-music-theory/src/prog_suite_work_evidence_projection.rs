// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical ProgSuite adapter for the generic FORM-003 evidence projection.
//!
//! ProgSuite currently has two complementary completed-score evidence layers:
//!
//! - [`crate::prog_suite_work_evidence::ProgSuiteWorkEvidenceV1`] covers all six
//!   translated work obligations and retains identity/tonal proxy provenance;
//! - [`crate::prog_suite_coverage_citation::ProgSuiteCoverageCitationSetV1`]
//!   provides stronger ordered-relation + whole-section evidence for the three
//!   development obligations.
//!
//! This adapter keeps both artifacts intact. Strong development citations
//! supersede the older work-evidence *view* for those exact obligation records;
//! they do not rewrite the older source artifact.

use crate::prog_suite_coverage_citation::{
    PROG_SUITE_COVERAGE_CITATION_PROFILE, PROG_SUITE_COVERAGE_CITATION_VERSION,
    ProgSuiteCoverageCitationDispositionV1, ProgSuiteCoverageCitationErrorV1,
    ProgSuiteCoverageCitationSetV1,
};
use crate::prog_suite_work_evidence::{
    PROG_SUITE_WORK_EVIDENCE_VERSION, ProgSuiteWorkEvidenceErrorV1,
    ProgSuiteWorkEvidenceSourceV1, ProgSuiteWorkEvidenceV1, derive_prog_suite_work_evidence,
};
use crate::sonata_work_evidence::WorkEvidenceStatusV1;
use crate::work_obligation_evidence::{
    WORK_OBLIGATION_EVIDENCE_PROJECTION_VERSION, WorkObligationEvidenceDispositionV1,
    WorkObligationEvidencePreservationV1, WorkObligationEvidenceProjectionErrorV1,
    WorkObligationEvidenceProjectionLossV1, WorkObligationEvidenceProjectionRecordV1,
    WorkObligationEvidenceProjectionSetV1, WorkObligationEvidenceSourceIdentityV1,
    required_work_obligation_evidence_projection_nonclaims,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const PROG_SUITE_WORK_EVIDENCE_PROJECTION_VERSION: &str =
    "melothaea-prog-suite-work-evidence-projection-v1";
pub const PROG_SUITE_WORK_EVIDENCE_PROJECTION_PROFILE: &str =
    "melothaea-prog-suite-work-evidence-projection-profile-v1";
const PROG_SUITE_WORK_EVIDENCE_NAMESPACE: &str = "melothaea-prog-suite-work-evidence";
const PROG_SUITE_STRONG_CITATION_NAMESPACE: &str = "melothaea-prog-suite-coverage-citation";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteWorkObligationEvidenceProjectionV1 {
    pub version: String,
    /// Strong development citation retained in full, including its exact
    /// subject, relation evidence, complete section coverage and losses.
    pub strong_citations: ProgSuiteCoverageCitationSetV1,
    /// Six-obligation work evidence rederived from the same native realization.
    pub work_evidence: ProgSuiteWorkEvidenceV1,
    /// Generic FORM-level projection. Non-authoritative without this adapter.
    pub projection: WorkObligationEvidenceProjectionSetV1,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgSuiteWorkEvidenceProjectionErrorV1 {
    WrongVersion { found: String },
    StrongCitation(ProgSuiteCoverageCitationErrorV1),
    WorkEvidence(ProgSuiteWorkEvidenceErrorV1),
    MissingWorkEvidenceRecord { obligation_id: String },
    GenericShape(WorkObligationEvidenceProjectionErrorV1),
    CanonicalProjectionMismatch,
}

pub fn project_prog_suite_work_obligation_evidence(
    strong_citations: &ProgSuiteCoverageCitationSetV1,
) -> Result<ProgSuiteWorkObligationEvidenceProjectionV1, ProgSuiteWorkEvidenceProjectionErrorV1> {
    strong_citations
        .validate()
        .map_err(ProgSuiteWorkEvidenceProjectionErrorV1::StrongCitation)?;

    let native_realization = &strong_citations
        .base_citations
        .adjudication
        .observed_relations
        .subject
        .realization;
    let work_evidence = derive_prog_suite_work_evidence(native_realization)
        .map_err(ProgSuiteWorkEvidenceProjectionErrorV1::WorkEvidence)?;

    let mut records = BTreeMap::new();
    for (obligation_id, work_record) in &work_evidence.records {
        let projected = if let Some(strong) = strong_citations
            .records
            .iter()
            .find(|record| record.obligation_id == *obligation_id)
        {
            project_strong_record(obligation_id, strong)
        } else {
            project_work_record(obligation_id, work_record)
        };
        records.insert(obligation_id.clone(), projected);
    }

    if records.len() != work_evidence.records.len() {
        let missing = work_evidence
            .records
            .keys()
            .find(|id| !records.contains_key(*id))
            .cloned()
            .unwrap_or_else(|| "unknown".into());
        return Err(
            ProgSuiteWorkEvidenceProjectionErrorV1::MissingWorkEvidenceRecord {
                obligation_id: missing,
            },
        );
    }

    let projection = WorkObligationEvidenceProjectionSetV1 {
        version: WORK_OBLIGATION_EVIDENCE_PROJECTION_VERSION.into(),
        records,
        nonclaims: required_work_obligation_evidence_projection_nonclaims(),
    };
    projection
        .validate_shape()
        .map_err(ProgSuiteWorkEvidenceProjectionErrorV1::GenericShape)?;

    Ok(ProgSuiteWorkObligationEvidenceProjectionV1 {
        version: PROG_SUITE_WORK_EVIDENCE_PROJECTION_VERSION.into(),
        strong_citations: strong_citations.clone(),
        work_evidence,
        projection,
    })
}

impl ProgSuiteWorkObligationEvidenceProjectionV1 {
    pub fn validate(&self) -> Result<(), ProgSuiteWorkEvidenceProjectionErrorV1> {
        if self.version != PROG_SUITE_WORK_EVIDENCE_PROJECTION_VERSION {
            return Err(ProgSuiteWorkEvidenceProjectionErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        self.projection
            .validate_shape()
            .map_err(ProgSuiteWorkEvidenceProjectionErrorV1::GenericShape)?;
        let canonical = project_prog_suite_work_obligation_evidence(&self.strong_citations)?;
        if &canonical != self {
            return Err(
                ProgSuiteWorkEvidenceProjectionErrorV1::CanonicalProjectionMismatch,
            );
        }
        Ok(())
    }
}

fn project_strong_record(
    obligation_id: &str,
    strong: &crate::prog_suite_coverage_citation::ProgSuiteCoverageCitationRecordV1,
) -> WorkObligationEvidenceProjectionRecordV1 {
    let disposition = match strong.disposition {
        ProgSuiteCoverageCitationDispositionV1::PositiveEvidenceCitationUnderWholeSectionProfile => {
            WorkObligationEvidenceDispositionV1::PositiveEvidenceUnderProfile
        }
        ProgSuiteCoverageCitationDispositionV1::NegativeEvidenceCitationUnderWholeSectionProfile => {
            WorkObligationEvidenceDispositionV1::NegativeEvidenceUnderProfile
        }
        ProgSuiteCoverageCitationDispositionV1::IndeterminateEvidenceCitationUnderWholeSectionProfile => {
            WorkObligationEvidenceDispositionV1::IndeterminateEvidenceUnderProfile
        }
    };
    WorkObligationEvidenceProjectionRecordV1 {
        obligation_id: obligation_id.into(),
        source: WorkObligationEvidenceSourceIdentityV1 {
            namespace: PROG_SUITE_STRONG_CITATION_NAMESPACE.into(),
            version: PROG_SUITE_COVERAGE_CITATION_VERSION.into(),
            profile_id: PROG_SUITE_COVERAGE_CITATION_PROFILE.into(),
            record_id: strong.stage_id.clone(),
        },
        disposition,
        preservation: WorkObligationEvidencePreservationV1::Projected,
        projection_losses: vec![
            WorkObligationEvidenceProjectionLossV1 {
                namespace: PROG_SUITE_COVERAGE_CITATION_VERSION.into(),
                code: "admission-does-not-equal-resolution".into(),
            },
            WorkObligationEvidenceProjectionLossV1 {
                namespace: PROG_SUITE_COVERAGE_CITATION_VERSION.into(),
                code: "thematic-relation-anchored-at-first-statement".into(),
            },
        ],
    }
}

fn project_work_record(
    obligation_id: &str,
    work_record: &crate::prog_suite_work_evidence::ProgSuiteWorkEvidenceRecordV1,
) -> WorkObligationEvidenceProjectionRecordV1 {
    let disposition = match work_record.status {
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

    let tonal_proxy = matches!(
        &work_record.source,
        ProgSuiteWorkEvidenceSourceV1::TonicAnchor { .. }
    );
    let (preservation, projection_losses) = if tonal_proxy {
        (
            WorkObligationEvidencePreservationV1::Projected,
            vec![WorkObligationEvidenceProjectionLossV1 {
                namespace: PROG_SUITE_WORK_EVIDENCE_VERSION.into(),
                code: "tonic-anchor-proxy-not-full-tonal-analysis".into(),
            }],
        )
    } else {
        (WorkObligationEvidencePreservationV1::Exact, vec![])
    };

    WorkObligationEvidenceProjectionRecordV1 {
        obligation_id: obligation_id.into(),
        source: WorkObligationEvidenceSourceIdentityV1 {
            namespace: PROG_SUITE_WORK_EVIDENCE_NAMESPACE.into(),
            version: PROG_SUITE_WORK_EVIDENCE_VERSION.into(),
            profile_id: PROG_SUITE_WORK_EVIDENCE_PROJECTION_PROFILE.into(),
            record_id: obligation_id.into(),
        },
        disposition,
        preservation,
        projection_losses,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Duration, Key, Motif, MusicalIntent, PitchClass, Style,
        adjudicate_prog_suite_development, bind_prog_suite_subject,
        cite_prog_suite_development_obligations, cite_prog_suite_whole_section_coverage,
        measure_prog_suite_observed_relations, measure_prog_suite_section_coverage,
        plan_prog_suite, realize_prog_suite_subject_bound,
    };

    fn motif() -> Motif {
        Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
        ])
    }

    fn strong_citations() -> ProgSuiteCoverageCitationSetV1 {
        let plan = plan_prog_suite(
            Key::major(PitchClass::C),
            100.0,
            5,
            &Style::ProgFolk.spec(),
        )
        .unwrap();
        let declaration = bind_prog_suite_subject(&plan, &motif()).unwrap();
        let subject = realize_prog_suite_subject_bound(
            &declaration,
            &MusicalIntent {
                energy: 0.73,
                seed: 9182,
                ..MusicalIntent::default()
            },
        )
        .unwrap();
        let observed = measure_prog_suite_observed_relations(&subject).unwrap();
        let adjudicated = adjudicate_prog_suite_development(&observed).unwrap();
        let base = cite_prog_suite_development_obligations(&adjudicated).unwrap();
        let coverage = measure_prog_suite_section_coverage(&observed).unwrap();
        cite_prog_suite_whole_section_coverage(&base, &coverage).unwrap()
    }

    #[test]
    fn all_six_work_obligations_receive_generic_projection_records() {
        let projected = project_prog_suite_work_obligation_evidence(&strong_citations()).unwrap();
        assert_eq!(projected.work_evidence.records.len(), 6);
        assert_eq!(projected.projection.records.len(), 6);
        projected.validate().unwrap();
    }

    #[test]
    fn ordered_b_uses_strong_citation_instead_of_older_not_measured_view() {
        let projected = project_prog_suite_work_obligation_evidence(&strong_citations()).unwrap();
        assert_eq!(
            projected.work_evidence.records["prog-suite:transform-b"].status,
            WorkEvidenceStatusV1::NotMeasured
        );
        let generic = &projected.projection.records["prog-suite:transform-b"];
        assert_eq!(
            generic.disposition,
            WorkObligationEvidenceDispositionV1::PositiveEvidenceUnderProfile
        );
        assert_eq!(generic.source.namespace, PROG_SUITE_STRONG_CITATION_NAMESPACE);
        assert_eq!(generic.preservation, WorkObligationEvidencePreservationV1::Projected);
    }

    #[test]
    fn tonic_center_promises_retain_proxy_loss() {
        let projected = project_prog_suite_work_obligation_evidence(&strong_citations()).unwrap();
        for id in ["prog-suite:reach-relative", "prog-suite:return-home"] {
            let record = &projected.projection.records[id];
            assert_eq!(record.preservation, WorkObligationEvidencePreservationV1::Projected);
            assert_eq!(
                record.projection_losses,
                vec![WorkObligationEvidenceProjectionLossV1 {
                    namespace: PROG_SUITE_WORK_EVIDENCE_VERSION.into(),
                    code: "tonic-anchor-proxy-not-full-tonal-analysis".into(),
                }]
            );
        }
    }

    #[test]
    fn strong_development_records_retain_both_profile_losses() {
        let projected = project_prog_suite_work_obligation_evidence(&strong_citations()).unwrap();
        for id in [
            "prog-suite:transform-b",
            "prog-suite:transform-c",
            "prog-suite:return-primary",
        ] {
            let record = &projected.projection.records[id];
            assert_eq!(record.source.namespace, PROG_SUITE_STRONG_CITATION_NAMESPACE);
            assert_eq!(record.projection_losses.len(), 2);
            assert_eq!(
                record.projection_losses[0].code,
                "admission-does-not-equal-resolution"
            );
            assert_eq!(
                record.projection_losses[1].code,
                "thematic-relation-anchored-at-first-statement"
            );
        }
    }

    #[test]
    fn hand_edited_generic_state_cannot_acquire_prog_suite_authority() {
        let mut projected = project_prog_suite_work_obligation_evidence(&strong_citations()).unwrap();
        projected
            .projection
            .records
            .get_mut("prog-suite:establish-primary")
            .unwrap()
            .disposition = WorkObligationEvidenceDispositionV1::NegativeEvidenceUnderProfile;
        assert_eq!(
            projected.validate(),
            Err(ProgSuiteWorkEvidenceProjectionErrorV1::CanonicalProjectionMismatch)
        );
    }
}
