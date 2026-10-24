// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Predeclared motif-generalization lockbox for the optional ProgSuite
//! contextual-harmony intervention.
//!
//! The seed pilot fixes one motif and varies seeds. This protocol freezes a
//! second independent axis before any pilot outcome is available: eight exact
//! held-out motifs crossed with the already-predeclared eight seeds. V1 therefore
//! defines 64 motif x seed subjects while keeping the ProgFolk spec, home key,
//! tempo, expressive-intent template, intervention profile, and seed population
//! fixed.
//!
//! This file declares a future experiment. It contains no execution outcome and
//! grants no evidence merely by validating its shape.

use crate::MUSIC_THEORY_ENGINE_VERSION;
use crate::composer::MusicalIntent;
use crate::harmony::Key;
use crate::motif::{Motif, MotifNote};
use crate::pitch::PitchClass;
use crate::prog_suite_contextual_harmony::ProgSuiteContextualHarmonyProfileV1;
use crate::prog_suite_contextual_harmony_panel::{
    PROG_SUITE_CONTEXTUAL_HARMONY_PILOT_SEEDS, ProgSuiteHarmonyPilotIntentV1,
};
use crate::rhythm::Duration;
use crate::spec::CompositionSpec;
use crate::style::Style;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_VERSION: &str =
    "melothaea-prog-suite-contextual-harmony-motif-lockbox-v1";
pub const PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_MOTIF_COUNT: usize = 8;
pub const PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_SUBJECT_COUNT: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteHarmonyLockboxStatusV1 {
    /// Protocol population and analysis semantics were frozen before any
    /// lockbox result was admitted.
    PredeclaredBeforeExecution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteHarmonyLockboxPrimaryUnitV1 {
    /// One exact motif crossed with one exact seed. Event/note observations
    /// inside the subject are dependent measurements, not extra subjects.
    MotifSeedSubject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteHarmonyLockboxEndpointV1 {
    ActualProgressionIntervention,
    WholeScoreSymbolicDifference,
    SectionSymbolicDifference,
    VoiceSectionSymbolicDifference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteHarmonyLockboxAggregationV1 {
    /// V1 permits exact counts/proportions only. Inferential statistics require
    /// a separately frozen analysis contract rather than being invented after
    /// the lockbox is opened.
    DescriptiveCountsAndProportionsOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteHarmonyLockboxMotifV1 {
    pub motif_id: String,
    pub motif: Motif,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteHarmonyLockboxSubjectV1 {
    pub subject_index: usize,
    pub motif_id: String,
    pub plan_seed: u64,
    pub intent_seed: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteHarmonyLockboxAnalysisPlanV1 {
    pub primary_unit: ProgSuiteHarmonyLockboxPrimaryUnitV1,
    pub expected_subject_count: usize,
    pub endpoints: Vec<ProgSuiteHarmonyLockboxEndpointV1>,
    pub aggregation: ProgSuiteHarmonyLockboxAggregationV1,
    /// Prevents an event-rich score from being treated as hundreds of
    /// independent replicates of one motif x seed intervention.
    pub event_level_inference_allowed: bool,
    /// Results from this lockbox may not be used to alter the frozen profile
    /// and then be re-described as confirmatory evidence on the same subjects.
    pub policy_tuning_from_lockbox_allowed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteHarmonyLockboxNonClaimV1 {
    DeclarationDoesNotEstablishExecution,
    LockboxDoesNotEstablishRepresentativeMusicSampling,
    HeldOutMotifsDoNotEstablishAllMotifGeneralization,
    FixedSpecDoesNotEstablishStyleOrGenreGeneralization,
    FixedHomeKeyDoesNotEstablishKeyGeneralization,
    FixedTempoDoesNotEstablishTempoGeneralization,
    SymbolicEndpointsDoNotEstablishAudibility,
    SymbolicEndpointsDoNotEstablishListenerPreference,
    SymbolicEndpointsDoNotEstablishArtisticQuality,
    DescriptiveCountsDoNotEstablishStatisticalIndependence,
    LockboxDoesNotGrantProductAuthority,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteContextualHarmonyLockboxV1 {
    pub version: String,
    /// Package implementation identity only; not a Git/source-revision claim.
    pub engine_version: String,
    pub status: ProgSuiteHarmonyLockboxStatusV1,
    pub home_key: Key,
    pub tempo_bpm: f32,
    pub spec: CompositionSpec,
    pub intent_template: ProgSuiteHarmonyPilotIntentV1,
    pub profile: ProgSuiteContextualHarmonyProfileV1,
    pub seeds: [u64; 8],
    pub motifs: Vec<ProgSuiteHarmonyLockboxMotifV1>,
    /// Canonical motif-major, then seed-major full cross product.
    pub subjects: Vec<ProgSuiteHarmonyLockboxSubjectV1>,
    pub analysis: ProgSuiteHarmonyLockboxAnalysisPlanV1,
    pub nonclaims: Vec<ProgSuiteHarmonyLockboxNonClaimV1>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgSuiteContextualHarmonyLockboxErrorV1 {
    WrongVersion { found: String },
    WrongEngineVersion { found: String },
    WrongStatus,
    InvalidTempo,
    IntentTonicMismatch,
    NonCanonicalSeeds,
    NonCanonicalMotifs,
    EmptyMotifId,
    DuplicateMotifId { motif_id: String },
    EmptyMotif { motif_id: String },
    NonPositiveMotifDuration { motif_id: String },
    PilotMotifLeakedIntoLockbox { motif_id: String },
    DuplicateMotifMaterial { left_id: String, right_id: String },
    NonCanonicalSubjects,
    NonCanonicalAnalysisPlan,
    NonCanonicalNonClaims,
    CanonicalProtocolMismatch,
}

/// Freeze the exact V1 motif-generalization protocol. This construction is
/// intentionally parameterless: changing the spec/key/tempo/intent population
/// is a new protocol version rather than a runtime knob after outcomes exist.
pub fn predeclare_prog_suite_contextual_harmony_motif_lockbox_v1(
) -> Result<ProgSuiteContextualHarmonyLockboxV1, ProgSuiteContextualHarmonyLockboxErrorV1> {
    let home_key = Key::major(PitchClass::C);
    let tempo_bpm = 100.0_f32;
    let spec = Style::ProgFolk.spec();
    let intent_template = ProgSuiteHarmonyPilotIntentV1::from_intent(MusicalIntent {
        energy: 0.73,
        seed: 0,
        ..MusicalIntent::default()
    });
    let profile = ProgSuiteContextualHarmonyProfileV1::DirectedDepartureReturn;
    let seeds = PROG_SUITE_CONTEXTUAL_HARMONY_PILOT_SEEDS;
    let motifs = canonical_lockbox_motifs();
    validate_motifs(&motifs)?;
    let subjects = canonical_subjects(&motifs, &seeds);

    Ok(ProgSuiteContextualHarmonyLockboxV1 {
        version: PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_VERSION.into(),
        engine_version: MUSIC_THEORY_ENGINE_VERSION.into(),
        status: ProgSuiteHarmonyLockboxStatusV1::PredeclaredBeforeExecution,
        home_key,
        tempo_bpm,
        spec,
        intent_template,
        profile,
        seeds,
        motifs,
        subjects,
        analysis: canonical_analysis_plan(),
        nonclaims: required_nonclaims(),
    })
}

impl ProgSuiteContextualHarmonyLockboxV1 {
    pub fn validate(&self) -> Result<(), ProgSuiteContextualHarmonyLockboxErrorV1> {
        if self.version != PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_VERSION {
            return Err(ProgSuiteContextualHarmonyLockboxErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        if self.engine_version != MUSIC_THEORY_ENGINE_VERSION {
            return Err(ProgSuiteContextualHarmonyLockboxErrorV1::WrongEngineVersion {
                found: self.engine_version.clone(),
            });
        }
        if self.status != ProgSuiteHarmonyLockboxStatusV1::PredeclaredBeforeExecution {
            return Err(ProgSuiteContextualHarmonyLockboxErrorV1::WrongStatus);
        }
        if !self.tempo_bpm.is_finite() || self.tempo_bpm <= 0.0 {
            return Err(ProgSuiteContextualHarmonyLockboxErrorV1::InvalidTempo);
        }
        if self.intent_template.tonic != self.home_key.tonic {
            return Err(ProgSuiteContextualHarmonyLockboxErrorV1::IntentTonicMismatch);
        }
        if self.seeds != PROG_SUITE_CONTEXTUAL_HARMONY_PILOT_SEEDS {
            return Err(ProgSuiteContextualHarmonyLockboxErrorV1::NonCanonicalSeeds);
        }
        validate_motifs(&self.motifs)?;
        if self.motifs != canonical_lockbox_motifs() {
            return Err(ProgSuiteContextualHarmonyLockboxErrorV1::NonCanonicalMotifs);
        }
        if self.subjects != canonical_subjects(&self.motifs, &self.seeds) {
            return Err(ProgSuiteContextualHarmonyLockboxErrorV1::NonCanonicalSubjects);
        }
        if self.analysis != canonical_analysis_plan() {
            return Err(ProgSuiteContextualHarmonyLockboxErrorV1::NonCanonicalAnalysisPlan);
        }
        if self.nonclaims != required_nonclaims() {
            return Err(ProgSuiteContextualHarmonyLockboxErrorV1::NonCanonicalNonClaims);
        }
        let canonical = predeclare_prog_suite_contextual_harmony_motif_lockbox_v1()?;
        if &canonical != self {
            return Err(ProgSuiteContextualHarmonyLockboxErrorV1::CanonicalProtocolMismatch);
        }
        Ok(())
    }
}

fn canonical_subjects(
    motifs: &[ProgSuiteHarmonyLockboxMotifV1],
    seeds: &[u64; 8],
) -> Vec<ProgSuiteHarmonyLockboxSubjectV1> {
    let mut subjects = Vec::with_capacity(motifs.len() * seeds.len());
    for motif in motifs {
        for seed in seeds {
            let subject_index = subjects.len();
            subjects.push(ProgSuiteHarmonyLockboxSubjectV1 {
                subject_index,
                motif_id: motif.motif_id.clone(),
                plan_seed: *seed,
                intent_seed: *seed,
            });
        }
    }
    subjects
}

fn canonical_analysis_plan() -> ProgSuiteHarmonyLockboxAnalysisPlanV1 {
    ProgSuiteHarmonyLockboxAnalysisPlanV1 {
        primary_unit: ProgSuiteHarmonyLockboxPrimaryUnitV1::MotifSeedSubject,
        expected_subject_count: PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_SUBJECT_COUNT,
        endpoints: vec![
            ProgSuiteHarmonyLockboxEndpointV1::ActualProgressionIntervention,
            ProgSuiteHarmonyLockboxEndpointV1::WholeScoreSymbolicDifference,
            ProgSuiteHarmonyLockboxEndpointV1::SectionSymbolicDifference,
            ProgSuiteHarmonyLockboxEndpointV1::VoiceSectionSymbolicDifference,
        ],
        aggregation: ProgSuiteHarmonyLockboxAggregationV1::DescriptiveCountsAndProportionsOnly,
        event_level_inference_allowed: false,
        policy_tuning_from_lockbox_allowed: false,
    }
}

fn pilot_motif() -> Motif {
    Motif::from_degrees(&[
        (1, Duration::new(1, 1)),
        (2, Duration::new(1, 1)),
        (3, Duration::new(1, 1)),
        (5, Duration::new(1, 1)),
    ])
}

fn canonical_lockbox_motifs() -> Vec<ProgSuiteHarmonyLockboxMotifV1> {
    vec![
        motif(
            "descending-return",
            &[(5, 1, 1), (4, 1, 1), (2, 1, 1), (1, 1, 1)],
        ),
        motif(
            "arch",
            &[(1, 1, 1), (3, 1, 1), (5, 1, 1), (3, 1, 1)],
        ),
        motif(
            "valley",
            &[(5, 1, 1), (3, 1, 1), (1, 1, 1), (3, 1, 1)],
        ),
        motif(
            "leap-cross",
            &[(1, 1, 1), (5, 1, 1), (2, 1, 1), (6, 1, 1)],
        ),
        motif(
            "asymmetric-rhythm",
            &[(1, 1, 2), (2, 3, 2), (5, 1, 2), (4, 3, 2)],
        ),
        motif(
            "five-note-turn",
            &[(1, 1, 1), (2, 1, 2), (4, 1, 2), (3, 1, 1), (5, 1, 1)],
        ),
        ProgSuiteHarmonyLockboxMotifV1 {
            motif_id: "rest-interrupted".into(),
            motif: Motif::new(vec![
                MotifNote::new(1, Duration::new(1, 1)),
                MotifNote::rest(Duration::new(1, 2)),
                MotifNote::new(3, Duration::new(1, 2)),
                MotifNote::new(5, Duration::new(2, 1)),
            ]),
        },
        motif(
            "wide-register",
            &[(1, 1, 1), (8, 1, 1), (5, 1, 1), (10, 1, 1)],
        ),
    ]
}

fn motif(id: &str, notes: &[(i32, i64, i64)]) -> ProgSuiteHarmonyLockboxMotifV1 {
    ProgSuiteHarmonyLockboxMotifV1 {
        motif_id: id.into(),
        motif: Motif::from_degrees(
            &notes
                .iter()
                .map(|&(degree, numerator, denominator)| {
                    (degree, Duration::new(numerator, denominator))
                })
                .collect::<Vec<_>>(),
        ),
    }
}

fn validate_motifs(
    motifs: &[ProgSuiteHarmonyLockboxMotifV1],
) -> Result<(), ProgSuiteContextualHarmonyLockboxErrorV1> {
    if motifs.len() != PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_MOTIF_COUNT {
        return Err(ProgSuiteContextualHarmonyLockboxErrorV1::NonCanonicalMotifs);
    }
    let mut ids = BTreeSet::new();
    let pilot = pilot_motif();
    for motif in motifs {
        if motif.motif_id.trim().is_empty() {
            return Err(ProgSuiteContextualHarmonyLockboxErrorV1::EmptyMotifId);
        }
        if !ids.insert(motif.motif_id.clone()) {
            return Err(ProgSuiteContextualHarmonyLockboxErrorV1::DuplicateMotifId {
                motif_id: motif.motif_id.clone(),
            });
        }
        if motif.motif.is_empty() {
            return Err(ProgSuiteContextualHarmonyLockboxErrorV1::EmptyMotif {
                motif_id: motif.motif_id.clone(),
            });
        }
        if motif.motif.total_duration().num() <= 0 {
            return Err(
                ProgSuiteContextualHarmonyLockboxErrorV1::NonPositiveMotifDuration {
                    motif_id: motif.motif_id.clone(),
                },
            );
        }
        if motif.motif == pilot {
            return Err(
                ProgSuiteContextualHarmonyLockboxErrorV1::PilotMotifLeakedIntoLockbox {
                    motif_id: motif.motif_id.clone(),
                },
            );
        }
    }
    for left in 0..motifs.len() {
        for right in (left + 1)..motifs.len() {
            if motifs[left].motif == motifs[right].motif {
                return Err(
                    ProgSuiteContextualHarmonyLockboxErrorV1::DuplicateMotifMaterial {
                        left_id: motifs[left].motif_id.clone(),
                        right_id: motifs[right].motif_id.clone(),
                    },
                );
            }
        }
    }
    Ok(())
}

fn required_nonclaims() -> Vec<ProgSuiteHarmonyLockboxNonClaimV1> {
    vec![
        ProgSuiteHarmonyLockboxNonClaimV1::DeclarationDoesNotEstablishExecution,
        ProgSuiteHarmonyLockboxNonClaimV1::LockboxDoesNotEstablishRepresentativeMusicSampling,
        ProgSuiteHarmonyLockboxNonClaimV1::HeldOutMotifsDoNotEstablishAllMotifGeneralization,
        ProgSuiteHarmonyLockboxNonClaimV1::FixedSpecDoesNotEstablishStyleOrGenreGeneralization,
        ProgSuiteHarmonyLockboxNonClaimV1::FixedHomeKeyDoesNotEstablishKeyGeneralization,
        ProgSuiteHarmonyLockboxNonClaimV1::FixedTempoDoesNotEstablishTempoGeneralization,
        ProgSuiteHarmonyLockboxNonClaimV1::SymbolicEndpointsDoNotEstablishAudibility,
        ProgSuiteHarmonyLockboxNonClaimV1::SymbolicEndpointsDoNotEstablishListenerPreference,
        ProgSuiteHarmonyLockboxNonClaimV1::SymbolicEndpointsDoNotEstablishArtisticQuality,
        ProgSuiteHarmonyLockboxNonClaimV1::DescriptiveCountsDoNotEstablishStatisticalIndependence,
        ProgSuiteHarmonyLockboxNonClaimV1::LockboxDoesNotGrantProductAuthority,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lockbox_is_frozen_to_eight_held_out_motifs_crossed_with_eight_seeds() {
        let protocol = predeclare_prog_suite_contextual_harmony_motif_lockbox_v1().unwrap();
        assert_eq!(protocol.motifs.len(), 8);
        assert_eq!(protocol.seeds, [3, 11, 23, 41, 59, 79, 97, 127]);
        assert_eq!(protocol.subjects.len(), 64);
        assert!(protocol.motifs.iter().all(|entry| entry.motif != pilot_motif()));
        protocol.validate().unwrap();
    }

    #[test]
    fn subject_order_is_motif_major_then_seed_major_and_seed_is_paired() {
        let protocol = predeclare_prog_suite_contextual_harmony_motif_lockbox_v1().unwrap();
        for (index, subject) in protocol.subjects.iter().enumerate() {
            assert_eq!(subject.subject_index, index);
            let motif_index = index / 8;
            let seed_index = index % 8;
            assert_eq!(subject.motif_id, protocol.motifs[motif_index].motif_id);
            assert_eq!(subject.plan_seed, protocol.seeds[seed_index]);
            assert_eq!(subject.intent_seed, protocol.seeds[seed_index]);
        }
    }

    #[test]
    fn primary_unit_prevents_event_level_pseudoreplication() {
        let protocol = predeclare_prog_suite_contextual_harmony_motif_lockbox_v1().unwrap();
        assert_eq!(
            protocol.analysis.primary_unit,
            ProgSuiteHarmonyLockboxPrimaryUnitV1::MotifSeedSubject
        );
        assert_eq!(protocol.analysis.expected_subject_count, 64);
        assert!(!protocol.analysis.event_level_inference_allowed);
        assert!(!protocol.analysis.policy_tuning_from_lockbox_allowed);
        assert_eq!(
            protocol.analysis.aggregation,
            ProgSuiteHarmonyLockboxAggregationV1::DescriptiveCountsAndProportionsOnly
        );
    }

    #[test]
    fn exact_motif_bank_contains_pitch_rhythm_rest_and_register_variation() {
        let protocol = predeclare_prog_suite_contextual_harmony_motif_lockbox_v1().unwrap();
        assert!(protocol.motifs.iter().any(|entry| entry.motif.notes.len() == 5));
        assert!(protocol
            .motifs
            .iter()
            .any(|entry| entry.motif.notes.iter().any(|note| note.is_rest())));
        assert!(protocol.motifs.iter().any(|entry| {
            entry.motif.degrees().into_iter().any(|degree| degree > 7)
        }));
        assert!(protocol.motifs.iter().any(|entry| {
            entry.motif.notes.iter().any(|note| note.duration.den() == 2)
        }));
    }

    #[test]
    fn serialized_subject_or_analysis_tampering_fails_validation() {
        let mut protocol = predeclare_prog_suite_contextual_harmony_motif_lockbox_v1().unwrap();
        protocol.subjects[0].intent_seed = 999;
        assert!(protocol.validate().is_err());

        let mut protocol = predeclare_prog_suite_contextual_harmony_motif_lockbox_v1().unwrap();
        protocol.analysis.event_level_inference_allowed = true;
        assert!(protocol.validate().is_err());
    }

    #[test]
    fn nonclaims_keep_lockbox_scope_narrow() {
        let protocol = predeclare_prog_suite_contextual_harmony_motif_lockbox_v1().unwrap();
        assert_eq!(protocol.nonclaims, required_nonclaims());
        assert!(protocol.nonclaims.contains(
            &ProgSuiteHarmonyLockboxNonClaimV1::DeclarationDoesNotEstablishExecution
        ));
        assert!(protocol.nonclaims.contains(
            &ProgSuiteHarmonyLockboxNonClaimV1::HeldOutMotifsDoNotEstablishAllMotifGeneralization
        ));
        assert!(protocol.nonclaims.contains(
            &ProgSuiteHarmonyLockboxNonClaimV1::SymbolicEndpointsDoNotEstablishArtisticQuality
        ));
    }
}
