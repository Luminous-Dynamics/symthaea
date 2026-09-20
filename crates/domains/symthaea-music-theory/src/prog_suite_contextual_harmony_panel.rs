// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Predeclared multi-seed pilot for the optional ProgSuite contextual harmony
//! intervention.
//!
//! A single paired comparison can establish a narrow symbolic causal effect for
//! one exact subject, but it cannot establish robustness. This module freezes an
//! eight-subject seed panel *before execution results exist* and projects the
//! detailed paired comparison into descriptive incidence counts.
//!
//! V1 deliberately fixes one motif, one CompositionSpec, one expressive intent
//! template, one home key, and one tempo while varying the joint native-plan /
//! MusicalIntent seed. It therefore probes seed robustness only. It does not
//! establish motif, style, genre, perceptual, or population generalization.

use crate::MUSIC_THEORY_ENGINE_VERSION;
use crate::composer::MusicalIntent;
use crate::harmony::Key;
use crate::motif::Motif;
use crate::pitch::PitchClass;
use crate::prog_suite::{ProgSuitePlanErrorV1, plan_prog_suite};
use crate::prog_suite_contextual_harmony::ProgSuiteContextualHarmonyProfileV1;
use crate::prog_suite_contextual_harmony_comparison::{
    ProgSuiteContextualHarmonyComparisonErrorV1, ProgSuiteHarmonyComparisonAuthorityV1,
    ProgSuiteSectionHarmonyComparisonV1, derive_prog_suite_contextual_harmony_comparison,
};
use crate::prog_suite_development_context::{
    ProgSuiteDevelopmentContextErrorV1, derive_prog_suite_development_context,
};
use crate::score::VoiceRole;
use crate::spec::CompositionSpec;
use serde::{Deserialize, Serialize};

pub const PROG_SUITE_CONTEXTUAL_HARMONY_PANEL_VERSION: &str =
    "melothaea-prog-suite-contextual-harmony-panel-v1";

/// Frozen before any panel execution result. These subjects are a deterministic
/// robustness pilot, not a claim of representative sampling.
pub const PROG_SUITE_CONTEXTUAL_HARMONY_PILOT_SEEDS: [u64; 8] =
    [3, 11, 23, 41, 59, 79, 97, 127];

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteHarmonyPilotIntentV1 {
    pub valence: f32,
    pub arousal: f32,
    pub energy: f32,
    pub bars: usize,
    pub tonic: PitchClass,
}

impl ProgSuiteHarmonyPilotIntentV1 {
    pub fn from_intent(intent: MusicalIntent) -> Self {
        Self {
            valence: intent.valence,
            arousal: intent.arousal,
            energy: intent.energy,
            bars: intent.bars,
            tonic: intent.tonic,
        }
    }

    pub fn materialize(self, seed: u64) -> MusicalIntent {
        MusicalIntent {
            valence: self.valence,
            arousal: self.arousal,
            energy: self.energy,
            bars: self.bars,
            seed,
            tonic: self.tonic,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteHarmonyPilotSeedPolicyV1 {
    /// The same frozen seed selects the native ProgSuite plan and populates the
    /// paired realization's MusicalIntent. Both comparison arms receive the
    /// same materialized intent inside each subject.
    MatchPlanAndIntentSeed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteHarmonyPanelNonClaimV1 {
    FrozenSeedsDoNotEstablishRepresentativeSampling,
    FixedMotifDoesNotEstablishMotifGeneralization,
    FixedSpecDoesNotEstablishStyleOrGenreGeneralization,
    SymbolicIncidenceDoesNotEstablishAudibility,
    SymbolicIncidenceDoesNotEstablishListenerPreference,
    SymbolicIncidenceDoesNotEstablishArtisticQuality,
    DescriptiveCountsDoNotEstablishStatisticalIndependence,
    PanelDoesNotGrantProductAuthority,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteHarmonyPanelSubjectV1 {
    pub subject_index: usize,
    pub plan_seed: u64,
    pub intent_seed: u64,
    pub authority: ProgSuiteHarmonyComparisonAuthorityV1,
    pub changed_progression_section_indices: Vec<usize>,
    pub whole_score_symbolically_changed: bool,
    pub sections: Vec<ProgSuiteSectionHarmonyComparisonV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteHarmonySectionIncidenceV1 {
    pub section_index: usize,
    pub progression_changed_subject_count: usize,
    pub symbolic_changed_subject_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteHarmonyVoiceIncidenceV1 {
    pub role: VoiceRole,
    /// Number of subject x section panels whose exact event stream differs for
    /// this voice. Maximum in V1 is 8 subjects * 4 sections = 32.
    pub symbolic_changed_subject_section_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteHarmonyPanelSummaryV1 {
    pub subject_count: usize,
    pub subjects_with_actual_progression_intervention: usize,
    pub subjects_with_symbolic_difference: usize,
    pub subjects_with_progression_intervention_but_no_symbolic_difference: usize,
    pub subjects_with_no_progression_difference: usize,
    pub section_incidence: Vec<ProgSuiteHarmonySectionIncidenceV1>,
    pub voice_incidence: Vec<ProgSuiteHarmonyVoiceIncidenceV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteContextualHarmonyPanelV1 {
    pub version: String,
    /// Cargo package version under which every paired symbolic subject is
    /// rederived. This is not a Git/source revision identity.
    pub engine_version: String,
    pub seeds: [u64; 8],
    pub seed_policy: ProgSuiteHarmonyPilotSeedPolicyV1,
    pub home_key: Key,
    pub tempo_bpm: f32,
    pub spec: CompositionSpec,
    pub source_motif: Motif,
    pub intent_template: ProgSuiteHarmonyPilotIntentV1,
    pub profile: ProgSuiteContextualHarmonyProfileV1,
    pub subjects: Vec<ProgSuiteHarmonyPanelSubjectV1>,
    pub summary: ProgSuiteHarmonyPanelSummaryV1,
    pub nonclaims: Vec<ProgSuiteHarmonyPanelNonClaimV1>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgSuiteContextualHarmonyPanelErrorV1 {
    WrongVersion { found: String },
    WrongEngineVersion { found: String },
    NonCanonicalSeeds,
    IntentTonicMismatch,
    NativePlan(ProgSuitePlanErrorV1),
    DevelopmentContext(ProgSuiteDevelopmentContextErrorV1),
    Comparison(ProgSuiteContextualHarmonyComparisonErrorV1),
    NonCanonicalNonClaims,
    CanonicalPanelMismatch,
}

pub fn derive_prog_suite_contextual_harmony_pilot_panel(
    home_key: Key,
    tempo_bpm: f32,
    spec: &CompositionSpec,
    source_motif: &Motif,
    intent_template: ProgSuiteHarmonyPilotIntentV1,
    profile: ProgSuiteContextualHarmonyProfileV1,
) -> Result<ProgSuiteContextualHarmonyPanelV1, ProgSuiteContextualHarmonyPanelErrorV1> {
    if intent_template.tonic != home_key.tonic {
        return Err(ProgSuiteContextualHarmonyPanelErrorV1::IntentTonicMismatch);
    }

    let mut subjects = Vec::with_capacity(PROG_SUITE_CONTEXTUAL_HARMONY_PILOT_SEEDS.len());
    for (subject_index, seed) in PROG_SUITE_CONTEXTUAL_HARMONY_PILOT_SEEDS
        .into_iter()
        .enumerate()
    {
        let plan = plan_prog_suite(home_key, tempo_bpm, seed, spec)
            .map_err(ProgSuiteContextualHarmonyPanelErrorV1::NativePlan)?;
        let context = derive_prog_suite_development_context(&plan)
            .map_err(ProgSuiteContextualHarmonyPanelErrorV1::DevelopmentContext)?;
        let intent = intent_template.materialize(seed);
        let comparison = derive_prog_suite_contextual_harmony_comparison(
            &context,
            source_motif,
            &intent,
            profile,
        )
        .map_err(ProgSuiteContextualHarmonyPanelErrorV1::Comparison)?;
        comparison
            .validate()
            .map_err(ProgSuiteContextualHarmonyPanelErrorV1::Comparison)?;

        subjects.push(ProgSuiteHarmonyPanelSubjectV1 {
            subject_index,
            plan_seed: seed,
            intent_seed: seed,
            authority: comparison.authority,
            changed_progression_section_indices: comparison
                .changed_progression_section_indices
                .clone(),
            whole_score_symbolically_changed: !comparison.whole_score.exact_event_stream_match,
            sections: comparison.sections.clone(),
        });
    }

    let summary = summarize(&subjects);
    Ok(ProgSuiteContextualHarmonyPanelV1 {
        version: PROG_SUITE_CONTEXTUAL_HARMONY_PANEL_VERSION.into(),
        engine_version: MUSIC_THEORY_ENGINE_VERSION.into(),
        seeds: PROG_SUITE_CONTEXTUAL_HARMONY_PILOT_SEEDS,
        seed_policy: ProgSuiteHarmonyPilotSeedPolicyV1::MatchPlanAndIntentSeed,
        home_key,
        tempo_bpm,
        spec: spec.clone(),
        source_motif: source_motif.clone(),
        intent_template,
        profile,
        subjects,
        summary,
        nonclaims: required_nonclaims(),
    })
}

impl ProgSuiteContextualHarmonyPanelV1 {
    pub fn validate(&self) -> Result<(), ProgSuiteContextualHarmonyPanelErrorV1> {
        if self.version != PROG_SUITE_CONTEXTUAL_HARMONY_PANEL_VERSION {
            return Err(ProgSuiteContextualHarmonyPanelErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        if self.engine_version != MUSIC_THEORY_ENGINE_VERSION {
            return Err(ProgSuiteContextualHarmonyPanelErrorV1::WrongEngineVersion {
                found: self.engine_version.clone(),
            });
        }
        if self.seeds != PROG_SUITE_CONTEXTUAL_HARMONY_PILOT_SEEDS
            || self.seed_policy != ProgSuiteHarmonyPilotSeedPolicyV1::MatchPlanAndIntentSeed
        {
            return Err(ProgSuiteContextualHarmonyPanelErrorV1::NonCanonicalSeeds);
        }
        if self.nonclaims != required_nonclaims() {
            return Err(ProgSuiteContextualHarmonyPanelErrorV1::NonCanonicalNonClaims);
        }
        let canonical = derive_prog_suite_contextual_harmony_pilot_panel(
            self.home_key,
            self.tempo_bpm,
            &self.spec,
            &self.source_motif,
            self.intent_template,
            self.profile,
        )?;
        if &canonical != self {
            return Err(ProgSuiteContextualHarmonyPanelErrorV1::CanonicalPanelMismatch);
        }
        Ok(())
    }
}

fn summarize(subjects: &[ProgSuiteHarmonyPanelSubjectV1]) -> ProgSuiteHarmonyPanelSummaryV1 {
    let subjects_with_actual_progression_intervention = subjects
        .iter()
        .filter(|subject| {
            subject.authority
                != ProgSuiteHarmonyComparisonAuthorityV1::NoProgressionDifferenceForSubject
        })
        .count();
    let subjects_with_symbolic_difference = subjects
        .iter()
        .filter(|subject| {
            subject.authority
                == ProgSuiteHarmonyComparisonAuthorityV1::ControlledProgressionInterventionWithSymbolicDifference
        })
        .count();
    let subjects_with_progression_intervention_but_no_symbolic_difference = subjects
        .iter()
        .filter(|subject| {
            subject.authority
                == ProgSuiteHarmonyComparisonAuthorityV1::ControlledProgressionInterventionWithoutSymbolicDifference
        })
        .count();
    let subjects_with_no_progression_difference = subjects
        .iter()
        .filter(|subject| {
            subject.authority
                == ProgSuiteHarmonyComparisonAuthorityV1::NoProgressionDifferenceForSubject
        })
        .count();

    let section_incidence = (0..4usize)
        .map(|section_index| ProgSuiteHarmonySectionIncidenceV1 {
            section_index,
            progression_changed_subject_count: subjects
                .iter()
                .filter(|subject| {
                    subject
                        .sections
                        .get(section_index)
                        .is_some_and(|section| section.progression_changed)
                })
                .count(),
            symbolic_changed_subject_count: subjects
                .iter()
                .filter(|subject| {
                    subject.sections.get(section_index).is_some_and(|section| {
                        !section.all_events.exact_event_stream_match
                    })
                })
                .count(),
        })
        .collect();

    let voice_incidence = [
        VoiceRole::Melody,
        VoiceRole::Harmony,
        VoiceRole::Bass,
        VoiceRole::CounterMelody,
    ]
    .into_iter()
    .map(|role| ProgSuiteHarmonyVoiceIncidenceV1 {
        role,
        symbolic_changed_subject_section_count: subjects
            .iter()
            .flat_map(|subject| &subject.sections)
            .filter(|section| {
                section
                    .voices
                    .iter()
                    .find(|voice| voice.role == role)
                    .is_some_and(|voice| !voice.differences.exact_event_stream_match)
            })
            .count(),
    })
    .collect();

    ProgSuiteHarmonyPanelSummaryV1 {
        subject_count: subjects.len(),
        subjects_with_actual_progression_intervention,
        subjects_with_symbolic_difference,
        subjects_with_progression_intervention_but_no_symbolic_difference,
        subjects_with_no_progression_difference,
        section_incidence,
        voice_incidence,
    }
}

fn required_nonclaims() -> Vec<ProgSuiteHarmonyPanelNonClaimV1> {
    vec![
        ProgSuiteHarmonyPanelNonClaimV1::FrozenSeedsDoNotEstablishRepresentativeSampling,
        ProgSuiteHarmonyPanelNonClaimV1::FixedMotifDoesNotEstablishMotifGeneralization,
        ProgSuiteHarmonyPanelNonClaimV1::FixedSpecDoesNotEstablishStyleOrGenreGeneralization,
        ProgSuiteHarmonyPanelNonClaimV1::SymbolicIncidenceDoesNotEstablishAudibility,
        ProgSuiteHarmonyPanelNonClaimV1::SymbolicIncidenceDoesNotEstablishListenerPreference,
        ProgSuiteHarmonyPanelNonClaimV1::SymbolicIncidenceDoesNotEstablishArtisticQuality,
        ProgSuiteHarmonyPanelNonClaimV1::DescriptiveCountsDoNotEstablishStatisticalIndependence,
        ProgSuiteHarmonyPanelNonClaimV1::PanelDoesNotGrantProductAuthority,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Duration, PitchClass, Style};

    fn motif() -> Motif {
        Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
        ])
    }

    fn panel() -> ProgSuiteContextualHarmonyPanelV1 {
        let intent_template = ProgSuiteHarmonyPilotIntentV1::from_intent(MusicalIntent {
            energy: 0.73,
            seed: 999_999, // ignored by the explicit panel seed policy
            ..MusicalIntent::default()
        });
        derive_prog_suite_contextual_harmony_pilot_panel(
            Key::major(PitchClass::C),
            100.0,
            &Style::ProgFolk.spec(),
            &motif(),
            intent_template,
            ProgSuiteContextualHarmonyProfileV1::DirectedDepartureReturn,
        )
        .unwrap()
    }

    #[test]
    fn pilot_subjects_are_frozen_before_result_interpretation() {
        let panel = panel();
        assert_eq!(panel.seeds, [3, 11, 23, 41, 59, 79, 97, 127]);
        assert_eq!(panel.subjects.len(), 8);
        for (index, subject) in panel.subjects.iter().enumerate() {
            assert_eq!(subject.subject_index, index);
            assert_eq!(subject.plan_seed, panel.seeds[index]);
            assert_eq!(subject.intent_seed, panel.seeds[index]);
        }
        panel.validate().unwrap();
    }

    #[test]
    fn summary_is_descriptive_and_reconciles_to_subject_states() {
        let panel = panel();
        assert_eq!(panel.summary.subject_count, 8);
        assert_eq!(
            panel.summary.subjects_with_actual_progression_intervention,
            panel.summary.subjects_with_symbolic_difference
                + panel
                    .summary
                    .subjects_with_progression_intervention_but_no_symbolic_difference
        );
        assert_eq!(
            panel.summary.subject_count,
            panel.summary.subjects_with_actual_progression_intervention
                + panel.summary.subjects_with_no_progression_difference
        );
        assert_eq!(panel.summary.section_incidence.len(), 4);
        assert_eq!(panel.summary.voice_incidence.len(), 4);
    }

    #[test]
    fn intent_seed_is_predeclared_to_match_each_plan_seed() {
        let panel = panel();
        for subject in &panel.subjects {
            assert_eq!(subject.plan_seed, subject.intent_seed);
        }
        assert_eq!(
            panel.seed_policy,
            ProgSuiteHarmonyPilotSeedPolicyV1::MatchPlanAndIntentSeed
        );
    }

    #[test]
    fn panel_refuses_intent_tonic_different_from_planned_home_tonic() {
        let intent_template = ProgSuiteHarmonyPilotIntentV1::from_intent(MusicalIntent {
            tonic: PitchClass::D,
            ..MusicalIntent::default()
        });
        assert_eq!(
            derive_prog_suite_contextual_harmony_pilot_panel(
                Key::major(PitchClass::C),
                100.0,
                &Style::ProgFolk.spec(),
                &motif(),
                intent_template,
                ProgSuiteContextualHarmonyProfileV1::DirectedDepartureReturn,
            ),
            Err(ProgSuiteContextualHarmonyPanelErrorV1::IntentTonicMismatch)
        );
    }

    #[test]
    fn serialized_summary_tampering_fails_canonical_validation() {
        let mut panel = panel();
        panel.summary.subjects_with_symbolic_difference = panel
            .summary
            .subjects_with_symbolic_difference
            .saturating_add(1);
        assert!(panel.validate().is_err());
    }

    #[test]
    fn pilot_nonclaims_prevent_seed_incidence_from_becoming_quality_authority() {
        let panel = panel();
        assert_eq!(panel.nonclaims, required_nonclaims());
        assert!(panel.nonclaims.contains(
            &ProgSuiteHarmonyPanelNonClaimV1::FrozenSeedsDoNotEstablishRepresentativeSampling
        ));
        assert!(panel.nonclaims.contains(
            &ProgSuiteHarmonyPanelNonClaimV1::SymbolicIncidenceDoesNotEstablishArtisticQuality
        ));
    }
}
