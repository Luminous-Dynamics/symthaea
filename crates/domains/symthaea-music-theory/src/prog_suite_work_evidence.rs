// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Completed-score evidence overlay for the native ProgSuite -> FORM adapter.
//!
//! Declaration authority and evidence authority stay separate. This module
//! starts from a completed `ProgSuiteRealizationV1`, rebinds it to FORM-000/001,
//! measures FORM-002B thematic observations plus the independent Retrograde
//! channel, and gives every FORM-003 promise one provenance-bearing status.
//!
//! `ReachTonalCenter` is represented by a disclosed symbolic proxy: tonic pitch
//! duration share plus tonic presence in the final onset of the due section.
//! That is useful score-side evidence, but it is not promoted into a claim of
//! full tonal analysis or listener-perceived key.

use crate::harmony::Key;
use crate::pitch::PitchClass;
use crate::prog_suite::ProgSuiteRealizationV1;
use crate::prog_suite_work_bridge::{
    ProgSuiteWorkBridgeErrorV1, ProgSuiteWorkRealizationBindingV1,
    bind_prog_suite_realization,
};
use crate::score::Score;
use crate::sonata_work_evidence::WorkEvidenceStatusV1;
use crate::thematic_identity::ThematicTransformationClassV1;
use crate::thematic_retrograde_evidence::{
    ThematicRetrogradeEvidenceSetV1, ThematicRetrogradeEvidenceV1,
    ThematicRetrogradeStatusV1, measure_thematic_retrograde_evidence,
};
use crate::thematic_score_evidence::{
    ThematicDerivationEvidenceStatusV1, ThematicDerivationObservationV1,
    ThematicIdentityObservationStatusV1, ThematicIdentityObservationV1,
    ThematicScoreEvidenceErrorV1,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const PROG_SUITE_WORK_EVIDENCE_VERSION: &str = "melothaea-prog-suite-work-evidence-v1";
pub const TONIC_ANCHOR_MIN_DURATION_SHARE: f32 = 0.02;

const NODE_C: &str = "prog-suite:C";
const NODE_RETURN_A: &str = "prog-suite:ReturnA";
const IDENTITY_P: &str = "prog-suite:P";
const DERIVE_B: &str = "prog-suite:derive-B";
const DERIVE_C: &str = "prog-suite:derive-C";
const RETURN_A: &str = "prog-suite:return-A";

const OBLIGATION_ESTABLISH: &str = "prog-suite:establish-primary";
const OBLIGATION_TRANSFORM_B: &str = "prog-suite:transform-b";
const OBLIGATION_TRANSFORM_C: &str = "prog-suite:transform-c";
const OBLIGATION_REACH_RELATIVE: &str = "prog-suite:reach-relative";
const OBLIGATION_RETURN_PRIMARY: &str = "prog-suite:return-primary";
const OBLIGATION_RETURN_HOME: &str = "prog-suite:return-home";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteTonalCenterProjectionV1 {
    /// Duration-weighted tonic presence plus tonic in the section's final
    /// simultaneous onset. This is a score-side tonal-anchor proxy only.
    TonicDurationShareAndFinalArrival,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteTonicAnchorEvidenceV1 {
    pub work_node_id: String,
    pub target_key: Key,
    pub target_tonic: PitchClass,
    pub observed_note_count: usize,
    pub total_observed_duration_beats: f64,
    pub tonic_duration_share: f32,
    pub minimum_duration_share: f32,
    pub final_event_contains_tonic: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProgSuiteWorkEvidenceSourceV1 {
    ThematicIdentity {
        observation: ThematicIdentityObservationV1,
    },
    StandardThematicDerivation {
        observation: ThematicDerivationObservationV1,
    },
    ExactRetrograde {
        /// FORM-002B itself still marks Retrograde unmeasured; retain that base
        /// observation beside the independent channel that resolves it.
        base_observation: ThematicDerivationObservationV1,
        retrograde: ThematicRetrogradeEvidenceV1,
    },
    OrderedCompositeNotMeasured {
        label: String,
        observation: ThematicDerivationObservationV1,
    },
    TonicAnchor {
        projection: ProgSuiteTonalCenterProjectionV1,
        evidence: ProgSuiteTonicAnchorEvidenceV1,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteWorkEvidenceRecordV1 {
    pub work_obligation_id: String,
    pub status: WorkEvidenceStatusV1,
    pub source: ProgSuiteWorkEvidenceSourceV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteWorkEvidenceV1 {
    pub version: String,
    /// Structural/temporal binding of the completed score to the frozen native
    /// plan and translated FORM declaration.
    pub structural_binding: ProgSuiteWorkRealizationBindingV1,
    /// Retains FORM-002B observations and any exact standalone Retrograde
    /// records computed from the same completed score.
    pub thematic_evidence: ThematicRetrogradeEvidenceSetV1,
    /// Exactly one evidence record for every ProgSuite FORM-003 obligation.
    pub records: BTreeMap<String, ProgSuiteWorkEvidenceRecordV1>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgSuiteWorkEvidenceErrorV1 {
    StructuralBinding(ProgSuiteWorkBridgeErrorV1),
    ThematicEvidence(ThematicScoreEvidenceErrorV1),
    ScoreHomeKeyMismatch,
    MissingIdentityObservation { identity_id: String },
    MissingDerivationObservation { derivation_id: String },
    MissingRetrogradeObservation { derivation_id: String },
    MissingWorkNode { node_id: String },
    CoverageCountMismatch { expected: usize, found: usize },
}

pub fn derive_prog_suite_work_evidence(
    realization: &ProgSuiteRealizationV1,
) -> Result<ProgSuiteWorkEvidenceV1, ProgSuiteWorkEvidenceErrorV1> {
    let structural_binding = bind_prog_suite_realization(realization)
        .map_err(ProgSuiteWorkEvidenceErrorV1::StructuralBinding)?;
    if realization.score.key != realization.plan.home_key {
        return Err(ProgSuiteWorkEvidenceErrorV1::ScoreHomeKeyMismatch);
    }

    let declaration = &structural_binding.declaration;
    let thematic_evidence = measure_thematic_retrograde_evidence(
        &realization.score,
        &declaration.work_plan,
        &declaration.thematic_graph,
    )
    .map_err(ProgSuiteWorkEvidenceErrorV1::ThematicEvidence)?;

    let base = &thematic_evidence.base_evidence;
    let mut records = BTreeMap::new();

    records.insert(
        OBLIGATION_ESTABLISH.into(),
        identity_record(OBLIGATION_ESTABLISH, IDENTITY_P, base)?,
    );
    records.insert(
        OBLIGATION_TRANSFORM_B.into(),
        derivation_record(
            OBLIGATION_TRANSFORM_B,
            DERIVE_B,
            declaration,
            &thematic_evidence,
        )?,
    );
    records.insert(
        OBLIGATION_TRANSFORM_C.into(),
        derivation_record(
            OBLIGATION_TRANSFORM_C,
            DERIVE_C,
            declaration,
            &thematic_evidence,
        )?,
    );

    let c_key = realization.plan.sections[2].key;
    records.insert(
        OBLIGATION_REACH_RELATIVE.into(),
        tonic_record(
            OBLIGATION_REACH_RELATIVE,
            NODE_C,
            c_key,
            &realization.score,
            declaration,
        )?,
    );
    records.insert(
        OBLIGATION_RETURN_PRIMARY.into(),
        derivation_record(
            OBLIGATION_RETURN_PRIMARY,
            RETURN_A,
            declaration,
            &thematic_evidence,
        )?,
    );
    records.insert(
        OBLIGATION_RETURN_HOME.into(),
        tonic_record(
            OBLIGATION_RETURN_HOME,
            NODE_RETURN_A,
            realization.plan.home_key,
            &realization.score,
            declaration,
        )?,
    );

    let expected = declaration.obligation_plan.obligations.len();
    if records.len() != expected {
        return Err(ProgSuiteWorkEvidenceErrorV1::CoverageCountMismatch {
            expected,
            found: records.len(),
        });
    }

    Ok(ProgSuiteWorkEvidenceV1 {
        version: PROG_SUITE_WORK_EVIDENCE_VERSION.into(),
        structural_binding,
        thematic_evidence,
        records,
    })
}

fn identity_record(
    work_obligation_id: &str,
    identity_id: &str,
    base: &crate::thematic_score_evidence::ThematicScoreEvidenceV1,
) -> Result<ProgSuiteWorkEvidenceRecordV1, ProgSuiteWorkEvidenceErrorV1> {
    let observation = base
        .identity_observations
        .get(identity_id)
        .cloned()
        .ok_or_else(|| ProgSuiteWorkEvidenceErrorV1::MissingIdentityObservation {
            identity_id: identity_id.into(),
        })?;
    let status = match observation.status {
        ThematicIdentityObservationStatusV1::Anchored => WorkEvidenceStatusV1::Supported,
        ThematicIdentityObservationStatusV1::InsufficientMaterial => WorkEvidenceStatusV1::Failed,
    };
    Ok(ProgSuiteWorkEvidenceRecordV1 {
        work_obligation_id: work_obligation_id.into(),
        status,
        source: ProgSuiteWorkEvidenceSourceV1::ThematicIdentity { observation },
    })
}

fn derivation_record(
    work_obligation_id: &str,
    derivation_id: &str,
    declaration: &crate::prog_suite_work_bridge::ProgSuiteWorkBindingV1,
    evidence: &ThematicRetrogradeEvidenceSetV1,
) -> Result<ProgSuiteWorkEvidenceRecordV1, ProgSuiteWorkEvidenceErrorV1> {
    let declared = declaration
        .thematic_graph
        .derivations
        .get(derivation_id)
        .expect("validated ProgSuite bridge retains all declared derivations");
    let observation = evidence
        .base_evidence
        .derivation_observations
        .get(derivation_id)
        .cloned()
        .ok_or_else(|| ProgSuiteWorkEvidenceErrorV1::MissingDerivationObservation {
            derivation_id: derivation_id.into(),
        })?;

    if declared.transformations.as_slice() == [ThematicTransformationClassV1::Retrograde] {
        let retrograde = evidence
            .derivations
            .get(derivation_id)
            .cloned()
            .ok_or_else(|| ProgSuiteWorkEvidenceErrorV1::MissingRetrogradeObservation {
                derivation_id: derivation_id.into(),
            })?;
        let status = match retrograde.status {
            ThematicRetrogradeStatusV1::SupportedExact => WorkEvidenceStatusV1::Supported,
            ThematicRetrogradeStatusV1::Failed => WorkEvidenceStatusV1::Failed,
            ThematicRetrogradeStatusV1::InsufficientMaterial
            | ThematicRetrogradeStatusV1::IndeterminateSymmetry => {
                WorkEvidenceStatusV1::NotMeasured
            }
        };
        return Ok(ProgSuiteWorkEvidenceRecordV1 {
            work_obligation_id: work_obligation_id.into(),
            status,
            source: ProgSuiteWorkEvidenceSourceV1::ExactRetrograde {
                base_observation: observation,
                retrograde,
            },
        });
    }

    if let [ThematicTransformationClassV1::Other(label)] = declared.transformations.as_slice() {
        return Ok(ProgSuiteWorkEvidenceRecordV1 {
            work_obligation_id: work_obligation_id.into(),
            status: WorkEvidenceStatusV1::NotMeasured,
            source: ProgSuiteWorkEvidenceSourceV1::OrderedCompositeNotMeasured {
                label: label.clone(),
                observation,
            },
        });
    }

    let status = match observation.status {
        ThematicDerivationEvidenceStatusV1::Supported => WorkEvidenceStatusV1::Supported,
        ThematicDerivationEvidenceStatusV1::Failed => WorkEvidenceStatusV1::Failed,
        ThematicDerivationEvidenceStatusV1::PartiallyMeasured
        | ThematicDerivationEvidenceStatusV1::NotMeasured => WorkEvidenceStatusV1::NotMeasured,
    };
    Ok(ProgSuiteWorkEvidenceRecordV1 {
        work_obligation_id: work_obligation_id.into(),
        status,
        source: ProgSuiteWorkEvidenceSourceV1::StandardThematicDerivation { observation },
    })
}

fn tonic_record(
    work_obligation_id: &str,
    node_id: &str,
    target_key: Key,
    score: &Score,
    declaration: &crate::prog_suite_work_bridge::ProgSuiteWorkBindingV1,
) -> Result<ProgSuiteWorkEvidenceRecordV1, ProgSuiteWorkEvidenceErrorV1> {
    let node = declaration.work_plan.nodes.get(node_id).ok_or_else(|| {
        ProgSuiteWorkEvidenceErrorV1::MissingWorkNode {
            node_id: node_id.into(),
        }
    })?;
    let evidence = tonic_anchor(score, node_id, node.start, node.end, target_key);
    let supported = evidence.observed_note_count > 0
        && evidence.final_event_contains_tonic
        && evidence.tonic_duration_share >= evidence.minimum_duration_share;
    Ok(ProgSuiteWorkEvidenceRecordV1 {
        work_obligation_id: work_obligation_id.into(),
        status: if supported {
            WorkEvidenceStatusV1::Supported
        } else {
            WorkEvidenceStatusV1::Failed
        },
        source: ProgSuiteWorkEvidenceSourceV1::TonicAnchor {
            projection: ProgSuiteTonalCenterProjectionV1::TonicDurationShareAndFinalArrival,
            evidence,
        },
    })
}

fn tonic_anchor(
    score: &Score,
    node_id: &str,
    start: crate::rhythm::Duration,
    end: crate::rhythm::Duration,
    target_key: Key,
) -> ProgSuiteTonicAnchorEvidenceV1 {
    let notes: Vec<_> = score
        .notes
        .iter()
        .filter(|note| note.onset.beats() >= start.beats() && note.onset.beats() < end.beats())
        .collect();
    let total_duration: f64 = notes.iter().map(|note| note.duration.beats()).sum();
    let tonic_duration: f64 = notes
        .iter()
        .filter(|note| note.pitch.pitch_class() == target_key.tonic)
        .map(|note| note.duration.beats())
        .sum();
    let tonic_duration_share = if total_duration <= f64::EPSILON {
        0.0
    } else {
        (tonic_duration / total_duration) as f32
    };
    let final_onset = notes
        .iter()
        .map(|note| note.onset)
        .max_by(|left, right| left.beats().total_cmp(&right.beats()));
    let final_event_contains_tonic = final_onset.is_some_and(|final_onset| {
        notes.iter().any(|note| {
            note.onset == final_onset && note.pitch.pitch_class() == target_key.tonic
        })
    });

    ProgSuiteTonicAnchorEvidenceV1 {
        work_node_id: node_id.into(),
        target_key,
        target_tonic: target_key.tonic,
        observed_note_count: notes.len(),
        total_observed_duration_beats: total_duration,
        tonic_duration_share,
        minimum_duration_share: TONIC_ANCHOR_MIN_DURATION_SHARE,
        final_event_contains_tonic,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Duration, Key, Motif, MusicalIntent, PitchClass, ProgSuiteTransformV1, Style, VoiceRole,
        plan_prog_suite, realize_prog_suite_with_plan,
    };

    fn motif() -> Motif {
        Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
        ])
    }

    fn realization() -> ProgSuiteRealizationV1 {
        let plan = plan_prog_suite(
            Key::major(PitchClass::C),
            100.0,
            5,
            &Style::ProgFolk.spec(),
        )
        .unwrap();
        realize_prog_suite_with_plan(&plan, &motif(), &MusicalIntent::default()).unwrap()
    }

    #[test]
    fn every_prog_suite_promise_receives_one_explicit_source() {
        let evidence = derive_prog_suite_work_evidence(&realization()).unwrap();
        assert_eq!(evidence.records.len(), 6);
        assert_eq!(
            evidence.records[OBLIGATION_ESTABLISH].status,
            WorkEvidenceStatusV1::Supported
        );
        assert!(matches!(
            &evidence.records[OBLIGATION_TRANSFORM_B].source,
            ProgSuiteWorkEvidenceSourceV1::OrderedCompositeNotMeasured { label, .. }
                if label == "prog-suite:retrograde-inversion-v1"
        ));
        assert_eq!(
            evidence.records[OBLIGATION_TRANSFORM_B].status,
            WorkEvidenceStatusV1::NotMeasured
        );
        assert!(matches!(
            &evidence.records[OBLIGATION_TRANSFORM_C].source,
            ProgSuiteWorkEvidenceSourceV1::StandardThematicDerivation { .. }
        ));
        assert!(matches!(
            &evidence.records[OBLIGATION_RETURN_PRIMARY].source,
            ProgSuiteWorkEvidenceSourceV1::StandardThematicDerivation { .. }
        ));
    }

    #[test]
    fn tonal_promises_retain_proxy_provenance() {
        let evidence = derive_prog_suite_work_evidence(&realization()).unwrap();
        for id in [OBLIGATION_REACH_RELATIVE, OBLIGATION_RETURN_HOME] {
            assert!(matches!(
                &evidence.records[id].source,
                ProgSuiteWorkEvidenceSourceV1::TonicAnchor {
                    projection: ProgSuiteTonalCenterProjectionV1::TonicDurationShareAndFinalArrival,
                    ..
                }
            ));
        }
    }

    #[test]
    fn standalone_retrograde_uses_only_the_exact_retrograde_channel() {
        let mut realization = realization();
        realization.plan.sections[1].transformation = ProgSuiteTransformV1::Retrograde;
        realization.plan.sections[2].transformation = ProgSuiteTransformV1::Inversion;
        realization.plan.validate().unwrap();
        realization = realize_prog_suite_with_plan(
            &realization.plan,
            &motif(),
            &MusicalIntent::default(),
        )
        .unwrap();
        let evidence = derive_prog_suite_work_evidence(&realization).unwrap();
        let record = &evidence.records[OBLIGATION_TRANSFORM_B];
        assert!(matches!(
            &record.source,
            ProgSuiteWorkEvidenceSourceV1::ExactRetrograde { .. }
        ));
        if let ProgSuiteWorkEvidenceSourceV1::ExactRetrograde { retrograde, .. } = &record.source {
            assert_ne!(
                retrograde.status,
                ThematicRetrogradeStatusV1::InsufficientMaterial
            );
        }
    }

    #[test]
    fn removing_all_c_section_notes_fails_relative_tonic_anchor() {
        let mut realization = realization();
        let c = realization.plan.sections[2].clone();
        realization.score.notes.retain(|note| {
            note.onset.beats() < c.start.beats() || note.onset.beats() >= c.end.beats()
        });
        let evidence = derive_prog_suite_work_evidence(&realization).unwrap();
        assert_eq!(
            evidence.records[OBLIGATION_REACH_RELATIVE].status,
            WorkEvidenceStatusV1::Failed
        );
        if let ProgSuiteWorkEvidenceSourceV1::TonicAnchor { evidence, .. } =
            &evidence.records[OBLIGATION_REACH_RELATIVE].source
        {
            assert_eq!(evidence.observed_note_count, 0);
            assert!(!evidence.final_event_contains_tonic);
        } else {
            panic!("relative-key promise lost tonic-anchor provenance");
        }
    }

    #[test]
    fn stale_score_home_key_is_rejected_before_evidence_authority() {
        let mut realization = realization();
        realization.score.key = Key::major(PitchClass::D);
        assert_eq!(
            derive_prog_suite_work_evidence(&realization),
            Err(ProgSuiteWorkEvidenceErrorV1::ScoreHomeKeyMismatch)
        );
    }

    #[test]
    fn deleting_primary_melody_fails_establishment_instead_of_using_plan_knowledge() {
        let mut realization = realization();
        let a = realization.plan.sections[0].clone();
        realization.score.notes.retain(|note| {
            note.role != VoiceRole::Melody
                || note.onset.beats() < a.start.beats()
                || note.onset.beats() >= a.end.beats()
        });
        let evidence = derive_prog_suite_work_evidence(&realization).unwrap();
        assert_eq!(
            evidence.records[OBLIGATION_ESTABLISH].status,
            WorkEvidenceStatusV1::Failed
        );
    }
}
