// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Provenance-preserving coverage overlay for Sonata work-scale obligations.
//!
//! `sonata_work_evidence` maps the six measurements already provided by the
//! native completed-score Sonata verifier. FORM-002B adds generic observed
//! thematic identity/derivation evidence. This module combines those two
//! independent evidence sources without rewriting either contract or promoting
//! declaration knowledge into evidence.

use crate::sonata::{SonataRealization, SonataVerificationMetric};
use crate::sonata_work_bridge::{SonataWorkBridgeErrorV1, bridge_native_sonata_plan};
use crate::sonata_work_evidence::{
    NativeEvidencePreservationV1, SonataWorkEvidenceErrorV1, SonataWorkEvidenceV1,
    WorkEvidenceStatusV1, derive_sonata_work_evidence,
};
use crate::thematic_score_evidence::{
    ThematicDerivationEvidenceStatusV1, ThematicDerivationObservationV1,
    ThematicIdentityObservationStatusV1, ThematicIdentityObservationV1,
    ThematicScoreEvidenceErrorV1, ThematicScoreEvidenceV1, measure_thematic_graph,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const SONATA_WORK_EVIDENCE_COVERAGE_VERSION: &str =
    "melothaea-sonata-work-evidence-coverage-v1";

const ESTABLISH_PRIMARY: &str = "sonata:establish-primary";
const ESTABLISH_SECONDARY: &str = "sonata:establish-secondary";
const DEVELOP_PRIMARY: &str = "sonata:develop-primary";
const PRIMARY_ID: &str = "sonata:P";
const SECONDARY_ID: &str = "sonata:S";
const DEVELOPMENT_DERIVATION_ID: &str = "sonata:develop-primary";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SonataWorkEvidenceSourceV1 {
    NativeSonata {
        native_obligation_id: u64,
        preservation: NativeEvidencePreservationV1,
        metric: SonataVerificationMetric,
    },
    ThematicIdentity {
        identity_id: String,
        observation: ThematicIdentityObservationV1,
    },
    ThematicDerivation {
        derivation_id: String,
        observation: ThematicDerivationObservationV1,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SonataWorkEvidenceCoverageRecordV1 {
    pub work_obligation_id: String,
    pub status: WorkEvidenceStatusV1,
    pub source: SonataWorkEvidenceSourceV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SonataWorkEvidenceCoverageV1 {
    pub version: String,
    /// Native Sonata evidence retained intact, including cadence projection loss.
    pub native_evidence: SonataWorkEvidenceV1,
    /// Generic FORM-002B score evidence retained intact for independent audit.
    pub thematic_evidence: ThematicScoreEvidenceV1,
    /// One provenance-bearing coverage record per translated FORM-003 promise.
    pub records: BTreeMap<String, SonataWorkEvidenceCoverageRecordV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SonataWorkEvidenceCoverageErrorV1 {
    DeclarationBridge(SonataWorkBridgeErrorV1),
    NativeEvidence(SonataWorkEvidenceErrorV1),
    ThematicEvidence(ThematicScoreEvidenceErrorV1),
    MissingNativeRecord { work_obligation_id: String },
    NativeMeasuredRecordMissingMetric { work_obligation_id: String },
    MissingThematicIdentity { identity_id: String },
    MissingThematicDerivation { derivation_id: String },
    UnexpectedNativeCoverageGap { work_obligation_id: String },
    CoverageCountMismatch { expected: usize, found: usize },
}

pub fn derive_sonata_work_evidence_coverage(
    realization: &SonataRealization,
) -> Result<SonataWorkEvidenceCoverageV1, SonataWorkEvidenceCoverageErrorV1> {
    let binding = bridge_native_sonata_plan(&realization.plan)
        .map_err(SonataWorkEvidenceCoverageErrorV1::DeclarationBridge)?;
    let native_evidence = derive_sonata_work_evidence(realization)
        .map_err(SonataWorkEvidenceCoverageErrorV1::NativeEvidence)?;
    let thematic_evidence = measure_thematic_graph(
        &realization.score,
        &binding.work_plan,
        &binding.thematic_graph,
    )
    .map_err(SonataWorkEvidenceCoverageErrorV1::ThematicEvidence)?;

    let mut records = BTreeMap::new();
    for work_obligation_id in binding.obligation_plan.obligations.keys() {
        let native = native_evidence.records.get(work_obligation_id).ok_or_else(|| {
            SonataWorkEvidenceCoverageErrorV1::MissingNativeRecord {
                work_obligation_id: work_obligation_id.clone(),
            }
        })?;

        let record = if native.status != WorkEvidenceStatusV1::NotMeasured {
            let native_obligation_id = native.native_obligation_id.ok_or_else(|| {
                SonataWorkEvidenceCoverageErrorV1::NativeMeasuredRecordMissingMetric {
                    work_obligation_id: work_obligation_id.clone(),
                }
            })?;
            let metric = native.native_metric.clone().ok_or_else(|| {
                SonataWorkEvidenceCoverageErrorV1::NativeMeasuredRecordMissingMetric {
                    work_obligation_id: work_obligation_id.clone(),
                }
            })?;
            SonataWorkEvidenceCoverageRecordV1 {
                work_obligation_id: work_obligation_id.clone(),
                status: native.status,
                source: SonataWorkEvidenceSourceV1::NativeSonata {
                    native_obligation_id,
                    preservation: native.preservation,
                    metric,
                },
            }
        } else {
            thematic_gap_record(work_obligation_id, &thematic_evidence)?
        };
        records.insert(work_obligation_id.clone(), record);
    }

    if records.len() != binding.obligation_plan.obligations.len() {
        return Err(SonataWorkEvidenceCoverageErrorV1::CoverageCountMismatch {
            expected: binding.obligation_plan.obligations.len(),
            found: records.len(),
        });
    }

    Ok(SonataWorkEvidenceCoverageV1 {
        version: SONATA_WORK_EVIDENCE_COVERAGE_VERSION.into(),
        native_evidence,
        thematic_evidence,
        records,
    })
}

fn thematic_gap_record(
    work_obligation_id: &str,
    thematic_evidence: &ThematicScoreEvidenceV1,
) -> Result<SonataWorkEvidenceCoverageRecordV1, SonataWorkEvidenceCoverageErrorV1> {
    match work_obligation_id {
        ESTABLISH_PRIMARY => identity_record(work_obligation_id, PRIMARY_ID, thematic_evidence),
        ESTABLISH_SECONDARY => identity_record(work_obligation_id, SECONDARY_ID, thematic_evidence),
        DEVELOP_PRIMARY => derivation_record(
            work_obligation_id,
            DEVELOPMENT_DERIVATION_ID,
            thematic_evidence,
        ),
        other => Err(SonataWorkEvidenceCoverageErrorV1::UnexpectedNativeCoverageGap {
            work_obligation_id: other.into(),
        }),
    }
}

fn identity_record(
    work_obligation_id: &str,
    identity_id: &str,
    thematic_evidence: &ThematicScoreEvidenceV1,
) -> Result<SonataWorkEvidenceCoverageRecordV1, SonataWorkEvidenceCoverageErrorV1> {
    let observation = thematic_evidence
        .identity_observations
        .get(identity_id)
        .cloned()
        .ok_or_else(|| SonataWorkEvidenceCoverageErrorV1::MissingThematicIdentity {
            identity_id: identity_id.into(),
        })?;
    let status = match observation.status {
        ThematicIdentityObservationStatusV1::Anchored => WorkEvidenceStatusV1::Supported,
        ThematicIdentityObservationStatusV1::InsufficientMaterial => WorkEvidenceStatusV1::Failed,
    };
    Ok(SonataWorkEvidenceCoverageRecordV1 {
        work_obligation_id: work_obligation_id.into(),
        status,
        source: SonataWorkEvidenceSourceV1::ThematicIdentity {
            identity_id: identity_id.into(),
            observation,
        },
    })
}

fn derivation_record(
    work_obligation_id: &str,
    derivation_id: &str,
    thematic_evidence: &ThematicScoreEvidenceV1,
) -> Result<SonataWorkEvidenceCoverageRecordV1, SonataWorkEvidenceCoverageErrorV1> {
    let observation = thematic_evidence
        .derivation_observations
        .get(derivation_id)
        .cloned()
        .ok_or_else(|| SonataWorkEvidenceCoverageErrorV1::MissingThematicDerivation {
            derivation_id: derivation_id.into(),
        })?;
    let status = match observation.status {
        ThematicDerivationEvidenceStatusV1::Supported => WorkEvidenceStatusV1::Supported,
        ThematicDerivationEvidenceStatusV1::Failed => WorkEvidenceStatusV1::Failed,
        ThematicDerivationEvidenceStatusV1::PartiallyMeasured
        | ThematicDerivationEvidenceStatusV1::NotMeasured => WorkEvidenceStatusV1::NotMeasured,
    };
    Ok(SonataWorkEvidenceCoverageRecordV1 {
        work_obligation_id: work_obligation_id.into(),
        status,
        source: SonataWorkEvidenceSourceV1::ThematicDerivation {
            derivation_id: derivation_id.into(),
            observation,
        },
    })
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
    fn all_nine_work_promises_receive_explicit_provenance() {
        let coverage = derive_sonata_work_evidence_coverage(&realization()).unwrap();
        assert_eq!(coverage.records.len(), 9);
        let native = coverage
            .records
            .values()
            .filter(|record| matches!(&record.source, SonataWorkEvidenceSourceV1::NativeSonata { .. }))
            .count();
        let identity = coverage
            .records
            .values()
            .filter(|record| matches!(&record.source, SonataWorkEvidenceSourceV1::ThematicIdentity { .. }))
            .count();
        let derivation = coverage
            .records
            .values()
            .filter(|record| matches!(&record.source, SonataWorkEvidenceSourceV1::ThematicDerivation { .. }))
            .count();
        assert_eq!((native, identity, derivation), (6, 2, 1));
    }

    #[test]
    fn previously_unmeasured_establishment_promises_use_observed_identity_anchors() {
        let coverage = derive_sonata_work_evidence_coverage(&realization()).unwrap();
        for (obligation_id, identity_id) in [
            (ESTABLISH_PRIMARY, PRIMARY_ID),
            (ESTABLISH_SECONDARY, SECONDARY_ID),
        ] {
            let record = &coverage.records[obligation_id];
            assert_eq!(record.status, WorkEvidenceStatusV1::Supported);
            assert!(matches!(
                &record.source,
                SonataWorkEvidenceSourceV1::ThematicIdentity {
                    identity_id: found,
                    observation,
                } if found == identity_id
                    && observation.status == ThematicIdentityObservationStatusV1::Anchored
            ));
        }
    }

    #[test]
    fn development_promise_is_measured_not_inferred() {
        let coverage = derive_sonata_work_evidence_coverage(&realization()).unwrap();
        let record = &coverage.records[DEVELOP_PRIMARY];
        assert_ne!(record.status, WorkEvidenceStatusV1::NotMeasured);
        assert!(matches!(
            &record.source,
            SonataWorkEvidenceSourceV1::ThematicDerivation {
                derivation_id,
                observation,
            } if derivation_id == DEVELOPMENT_DERIVATION_ID
                && matches!(
                    observation.status,
                    ThematicDerivationEvidenceStatusV1::Supported
                        | ThematicDerivationEvidenceStatusV1::Failed
                )
        ));
    }

    #[test]
    fn native_measurements_keep_native_provenance() {
        let coverage = derive_sonata_work_evidence_coverage(&realization()).unwrap();
        for id in [
            "sonata:reach-contrast-key",
            "sonata:development-climax",
            "sonata:return-primary",
            "sonata:return-home",
            "sonata:return-secondary",
            "sonata:close-home",
        ] {
            assert!(matches!(
                &coverage.records[id].source,
                SonataWorkEvidenceSourceV1::NativeSonata { .. }
            ));
        }
    }

    #[test]
    fn removing_secondary_melody_breaks_the_observed_secondary_identity() {
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
        let coverage = derive_sonata_work_evidence_coverage(&realization).unwrap();
        assert_eq!(
            coverage.records[ESTABLISH_SECONDARY].status,
            WorkEvidenceStatusV1::Failed
        );
    }

    #[test]
    fn cadence_projection_loss_survives_the_coverage_overlay() {
        let coverage = derive_sonata_work_evidence_coverage(&realization()).unwrap();
        assert!(matches!(
            &coverage.records["sonata:close-home"].source,
            SonataWorkEvidenceSourceV1::NativeSonata {
                preservation: NativeEvidencePreservationV1::Projected { .. },
                ..
            }
        ));
    }
}
