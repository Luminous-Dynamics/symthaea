// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Completed-score evidence bridge for the native Sonata → work-scale adapter.
//!
//! The declaration bridge in [`crate::sonata_work_bridge`] says what a native
//! Sonata plan promises in FORM-001/002/003 terms. This module deliberately
//! starts over from the completed [`crate::SonataRealization::score`]: it
//! recomputes native Sonata evidence, maps only measurements that actually
//! exist, and leaves work-scale promises without a native score-side metric as
//! `NotMeasured` rather than promoting constructor knowledge into evidence.

use crate::sonata::{
    SonataObligationEvidence, SonataPlan, SonataRealization, SonataSectionKind,
    SonataVerificationMetric, verify_sonata_obligations,
};
use crate::sonata_work_bridge::{
    SONATA_WORK_BRIDGE_VERSION, SonataWorkBridgeErrorV1, bridge_native_sonata_plan,
};
use crate::obligation::ReturnTransformation;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const SONATA_WORK_EVIDENCE_VERSION: &str = "melothaea-sonata-work-evidence-v1";

const ESTABLISH_PRIMARY: &str = "sonata:establish-primary";
const ESTABLISH_SECONDARY: &str = "sonata:establish-secondary";
const REACH_CONTRAST_KEY: &str = "sonata:reach-contrast-key";
const DEVELOP_PRIMARY: &str = "sonata:develop-primary";
const DEVELOPMENT_CLIMAX: &str = "sonata:development-climax";
const RETURN_PRIMARY: &str = "sonata:return-primary";
const RETURN_HOME: &str = "sonata:return-home";
const RETURN_SECONDARY: &str = "sonata:return-secondary";
const CLOSE_HOME: &str = "sonata:close-home";

const NATIVE_MEASURED_WORK_IDS: [&str; 6] = [
    REACH_CONTRAST_KEY,
    DEVELOPMENT_CLIMAX,
    RETURN_PRIMARY,
    RETURN_HOME,
    RETURN_SECONDARY,
    CLOSE_HOME,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkEvidenceStatusV1 {
    Supported,
    Failed,
    NotMeasured,
}

/// Whether the native metric preserves the translated work promise exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NativeEvidencePreservationV1 {
    /// The work obligation currently has no corresponding native score-side metric.
    NotMeasured,
    /// Native Sonata evidence measures the same semantic promise.
    Exact,
    /// Native evidence is stronger/more specific than the translated FORM-003
    /// obligation, so the projection is usable only with an explicit loss receipt.
    Projected {
        loss: SonataEvidenceProjectionLossV1,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SonataEvidenceProjectionLossV1 {
    /// Native Sonata checks a cadential tonic arrival. FORM-003 currently stores
    /// the translated promise as the broader `ReachTonalCenter`, which does not
    /// itself retain cadential specificity.
    CadentialSpecificityToTonalCenter,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkObligationEvidenceRecordV1 {
    pub work_obligation_id: String,
    pub status: WorkEvidenceStatusV1,
    pub preservation: NativeEvidencePreservationV1,
    /// Native prospective-obligation identity used by the recomputed Sonata verifier.
    pub native_obligation_id: Option<u64>,
    /// Recomputed completed-score metric. `None` means exactly `NotMeasured`, not pass.
    pub native_metric: Option<SonataVerificationMetric>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SonataEvidenceCacheReceiptV1 {
    /// `SonataRealization` stores verifier output for convenience. This evidence
    /// bridge never trusts it; it recomputes from score+plan and records whether
    /// the cached copy happened to agree.
    pub stored_verification_matches_recomputed: bool,
    /// Same rule for the mutable legacy resolution ledger.
    pub stored_resolution_matches_recomputed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SonataWorkEvidenceV1 {
    pub version: String,
    pub declaration_bridge_version: String,
    /// Fresh native measurements recomputed from the completed score.
    pub recomputed_native_evidence: Vec<SonataObligationEvidence>,
    pub cache_receipt: SonataEvidenceCacheReceiptV1,
    /// One record for every translated work-scale obligation, including
    /// explicit `NotMeasured` records for declaration-only promises.
    pub records: BTreeMap<String, WorkObligationEvidenceRecordV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SonataWorkEvidenceErrorV1 {
    DeclarationBridge(SonataWorkBridgeErrorV1),
    WrongNativeEvidenceCount { found: usize },
    MissingTranslatedWorkObligation { work_obligation_id: String },
    DuplicateNativeEvidenceTarget { work_obligation_id: String },
    UnsupportedNativeMetric { obligation_id: u64, obligation: String },
    UnexpectedNativeMetric { obligation_id: u64, metric: String },
    MissingExpectedNativeMeasurement { work_obligation_id: String },
}

pub fn derive_sonata_work_evidence(
    realization: &SonataRealization,
) -> Result<SonataWorkEvidenceV1, SonataWorkEvidenceErrorV1> {
    let binding = bridge_native_sonata_plan(&realization.plan)
        .map_err(SonataWorkEvidenceErrorV1::DeclarationBridge)?;

    let (recomputed_resolution, recomputed_native_evidence) =
        verify_sonata_obligations(&realization.score, &realization.plan);
    if recomputed_native_evidence.len() != NATIVE_MEASURED_WORK_IDS.len() {
        return Err(SonataWorkEvidenceErrorV1::WrongNativeEvidenceCount {
            found: recomputed_native_evidence.len(),
        });
    }

    let cache_receipt = SonataEvidenceCacheReceiptV1 {
        stored_verification_matches_recomputed: realization.verification == recomputed_native_evidence,
        stored_resolution_matches_recomputed: realization.resolution == recomputed_resolution,
    };

    let mut records: BTreeMap<String, WorkObligationEvidenceRecordV1> = binding
        .obligation_plan
        .obligations
        .keys()
        .map(|work_obligation_id| {
            (
                work_obligation_id.clone(),
                WorkObligationEvidenceRecordV1 {
                    work_obligation_id: work_obligation_id.clone(),
                    status: WorkEvidenceStatusV1::NotMeasured,
                    preservation: NativeEvidencePreservationV1::NotMeasured,
                    native_obligation_id: None,
                    native_metric: None,
                },
            )
        })
        .collect();

    let mut mapped = BTreeSet::new();
    for native in &recomputed_native_evidence {
        let (work_obligation_id, preservation) =
            classify_native_metric(&native.metric, &realization.plan, native.obligation_id)?;
        if !binding
            .obligation_plan
            .obligations
            .contains_key(work_obligation_id)
        {
            return Err(SonataWorkEvidenceErrorV1::MissingTranslatedWorkObligation {
                work_obligation_id: work_obligation_id.into(),
            });
        }
        if !mapped.insert(work_obligation_id) {
            return Err(SonataWorkEvidenceErrorV1::DuplicateNativeEvidenceTarget {
                work_obligation_id: work_obligation_id.into(),
            });
        }
        let record = records
            .get_mut(work_obligation_id)
            .expect("translated work obligation checked above");
        record.status = if native.verified {
            WorkEvidenceStatusV1::Supported
        } else {
            WorkEvidenceStatusV1::Failed
        };
        record.preservation = preservation;
        record.native_obligation_id = Some(native.obligation_id);
        record.native_metric = Some(native.metric.clone());
    }

    for expected in NATIVE_MEASURED_WORK_IDS {
        if !mapped.contains(expected) {
            return Err(SonataWorkEvidenceErrorV1::MissingExpectedNativeMeasurement {
                work_obligation_id: expected.into(),
            });
        }
    }

    Ok(SonataWorkEvidenceV1 {
        version: SONATA_WORK_EVIDENCE_VERSION.into(),
        declaration_bridge_version: SONATA_WORK_BRIDGE_VERSION.into(),
        recomputed_native_evidence,
        cache_receipt,
        records,
    })
}

fn classify_native_metric(
    metric: &SonataVerificationMetric,
    plan: &SonataPlan,
    obligation_id: u64,
) -> Result<(&'static str, NativeEvidencePreservationV1), SonataWorkEvidenceErrorV1> {
    match metric {
        SonataVerificationMetric::TonalAnchor {
            section: SonataSectionKind::ExpositionSecondary,
            target,
            ..
        } if *target == plan.contrast_key.tonic => {
            Ok((REACH_CONTRAST_KEY, NativeEvidencePreservationV1::Exact))
        }
        SonataVerificationMetric::ClimaxLocation {
            section: SonataSectionKind::Development,
            ..
        } => Ok((DEVELOPMENT_CLIMAX, NativeEvidencePreservationV1::Exact)),
        SonataVerificationMetric::MotifReturn {
            source: SonataSectionKind::ExpositionPrimary,
            target: SonataSectionKind::RecapitulationPrimary,
            transformation: ReturnTransformation::Literal,
            ..
        } => Ok((RETURN_PRIMARY, NativeEvidencePreservationV1::Exact)),
        SonataVerificationMetric::TonalAnchor {
            section: SonataSectionKind::RecapitulationPrimary,
            target,
            ..
        } if *target == plan.home_key.tonic => {
            Ok((RETURN_HOME, NativeEvidencePreservationV1::Exact))
        }
        SonataVerificationMetric::MotifReturn {
            source: SonataSectionKind::ExpositionSecondary,
            target: SonataSectionKind::RecapitulationSecondary,
            transformation: ReturnTransformation::Transposed,
            ..
        } => Ok((RETURN_SECONDARY, NativeEvidencePreservationV1::Exact)),
        SonataVerificationMetric::CadentialArrival {
            section: SonataSectionKind::RecapitulationSecondary,
            target,
            ..
        } if *target == plan.home_key.tonic => Ok((
            CLOSE_HOME,
            NativeEvidencePreservationV1::Projected {
                loss: SonataEvidenceProjectionLossV1::CadentialSpecificityToTonalCenter,
            },
        )),
        SonataVerificationMetric::Unsupported { obligation } => {
            Err(SonataWorkEvidenceErrorV1::UnsupportedNativeMetric {
                obligation_id,
                obligation: obligation.clone(),
            })
        }
        other => Err(SonataWorkEvidenceErrorV1::UnexpectedNativeMetric {
            obligation_id,
            metric: format!("{other:?}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Duration, Key, Motif, MusicalIntent, PitchClass, realize_sonata_with_plan};

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
    fn native_verifier_maps_six_measurements_and_leaves_three_unmeasured() {
        let evidence = derive_sonata_work_evidence(&realization()).unwrap();
        assert_eq!(evidence.records.len(), 9);
        assert_eq!(
            evidence
                .records
                .values()
                .filter(|record| record.status != WorkEvidenceStatusV1::NotMeasured)
                .count(),
            6
        );
        for id in [ESTABLISH_PRIMARY, ESTABLISH_SECONDARY, DEVELOP_PRIMARY] {
            let record = &evidence.records[id];
            assert_eq!(record.status, WorkEvidenceStatusV1::NotMeasured);
            assert_eq!(record.preservation, NativeEvidencePreservationV1::NotMeasured);
            assert!(record.native_metric.is_none());
        }
    }

    #[test]
    fn cadence_evidence_retains_explicit_projection_loss() {
        let evidence = derive_sonata_work_evidence(&realization()).unwrap();
        assert_eq!(
            evidence.records[CLOSE_HOME].preservation,
            NativeEvidencePreservationV1::Projected {
                loss: SonataEvidenceProjectionLossV1::CadentialSpecificityToTonalCenter,
            }
        );
        assert!(matches!(
            evidence.records[CLOSE_HOME].native_metric,
            Some(SonataVerificationMetric::CadentialArrival { .. })
        ));
    }

    #[test]
    fn cached_verification_is_never_trusted_as_evidence() {
        let mut realization = realization();
        realization.verification[0].verified = !realization.verification[0].verified;
        let evidence = derive_sonata_work_evidence(&realization).unwrap();
        assert!(!evidence.cache_receipt.stored_verification_matches_recomputed);
        assert!(evidence.cache_receipt.stored_resolution_matches_recomputed);
    }

    #[test]
    fn completed_score_damage_produces_failure_not_constructor_inference() {
        let mut realization = realization();
        let secondary = realization
            .plan
            .sections
            .iter()
            .find(|section| section.kind == SonataSectionKind::ExpositionSecondary)
            .unwrap()
            .clone();
        realization.score.notes.retain(|note| {
            note.onset.beats() < secondary.start.beats()
                || note.onset.beats() >= secondary.end.beats()
        });

        let evidence = derive_sonata_work_evidence(&realization).unwrap();
        assert_eq!(
            evidence.records[REACH_CONTRAST_KEY].status,
            WorkEvidenceStatusV1::Failed
        );
        assert!(!evidence.cache_receipt.stored_verification_matches_recomputed);
        assert!(!evidence.cache_receipt.stored_resolution_matches_recomputed);
    }

    #[test]
    fn there_is_no_global_pass_that_can_hide_unmeasured_promises() {
        let evidence = derive_sonata_work_evidence(&realization()).unwrap();
        assert!(evidence
            .records
            .values()
            .any(|record| record.status == WorkEvidenceStatusV1::NotMeasured));
        assert!(!evidence.records.contains_key("all-pass"));
    }
}
