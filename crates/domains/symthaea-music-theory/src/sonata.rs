// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sonata form (style-roadmap item, Tier 1 "high priority" — a listening
//! review named it "one of the biggest remaining architectural
//! opportunities... teaches exposition, development, recapitulation,
//! thematic conflict"): the engine's first form built around TONAL
//! CONFLICT AND RESOLUTION, not just contrast.
//!
//! Every prior multi-section form this crate has (ternary, rondo) treats
//! its B/C sections as departures that never come back transposed — the
//! home key returns, but nothing that LEFT gets brought home with it.
//! Sonata form's whole identity is the opposite: a SECOND theme is
//! deliberately introduced in a foreign key (real tension — it doesn't
//! belong yet), and the piece's entire second half exists to resolve
//! that by restating the exact same idea back in the home key. That
//! resolution is the payoff every other form in this crate lacks.
//!
//! Five sections, splicing independently-realized sub-scores onto one
//! timeline (the same technique [`crate::prog_suite`] proved out for a
//! meter change — here what moves is the KEY, and unlike prog_suite's
//! four unrelated meters, two pairs of sections here are the SAME
//! musical idea in different keys, which is the whole point):
//!
//! - **Exposition, first subject (P)**: home key, the theme as given.
//! - **Exposition, second subject (S)**: a FOREIGN key — the dominant if
//!   the home key is major, the relative major if minor (real
//!   common-practice usage — see [`crate::harmony::Key::dominant`]'s
//!   doc). [`crate::form::contrasting_transform`] of P: a genuine
//!   family relationship, not an unrelated new tune.
//! - **Development**: a THIRD key, [`crate::fugue::head_fragment`]'s
//!   compression applied to P — the same "working out" mechanism
//!   [`crate::spec::DevelopmentDna::Fragmenting`] already models,
//!   reused here for the section that IS a sonata's development by
//!   name, not just by habit.
//! - **Recapitulation, first subject**: P again, home key, unchanged —
//!   the exposition's opening restated.
//! - **Recapitulation, second subject**: the EXACT SAME transformed
//!   idea S used, and the EXACT SAME progression it walked — but now in
//!   the HOME key. This is the form's defining move: the piece proves
//!   its own resolution by literally reusing the foreign material,
//!   simply no longer foreign. Pinned by test: the two sections' scale-
//!   degree `Phrase.line`s are byte-identical; only the rendering key
//!   differs.
//!
//! Voice-leading carries continuously across every section exactly as
//! `prog_suite` and `compose()` itself already do.

use crate::MusicalIntent;
use crate::form::{Form, Section, SectionRole, contrasting_transform};
use crate::harmony::{Key, Progression};
use crate::motif::Motif;
use crate::motif_return::{MotifReturnEvidence, compare_melodic_regions};
use crate::obligation::{
    CompositionalObligation, ObligationKind, ObligationLedger, ReturnTransformation,
};
use crate::phrase::Period;
use crate::pitch::{Pitch, PitchClass};
use crate::rhythm::Duration;
use crate::score::{Emphasis, Score, ScoreNote, VoiceRole};
use crate::spelling::AlteredDegree;
use serde::{Deserialize, Serialize};

/// Stable identities for the five structural regions of this sonata plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SonataSectionKind {
    ExpositionPrimary,
    ExpositionSecondary,
    Development,
    RecapitulationPrimary,
    RecapitulationSecondary,
}

/// One fully planned sonata section before score realization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedSonataSection {
    pub kind: SonataSectionKind,
    pub role: SectionRole,
    pub key: Key,
    pub period: Period,
    pub start: Duration,
    pub end: Duration,
}

/// The explicit tonal and thematic promises of a sonata-form composition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SonataPlan {
    pub home_key: Key,
    pub contrast_key: Key,
    pub development_key: Key,
    pub sections: Vec<PlannedSonataSection>,
    pub obligations: ObligationLedger,
}

/// A realized score plus the plan it fulfilled and the resulting evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SonataRealization {
    pub score: Score,
    pub plan: SonataPlan,
    pub resolution: ObligationLedger,
    /// Independent score-side evidence used to resolve each obligation.
    #[serde(default)]
    pub verification: Vec<SonataObligationEvidence>,
}

/// Structured evidence emitted by the score-side sonata verifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SonataObligationEvidence {
    pub obligation_id: u64,
    pub verified: bool,
    pub metric: SonataVerificationMetric,
}

/// Observable metric used to decide whether a sonata obligation was fulfilled.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SonataVerificationMetric {
    TonalAnchor {
        section: SonataSectionKind,
        target: PitchClass,
        tonic_duration_share: f32,
        final_event_contains_target: bool,
    },
    MotifReturn {
        source: SonataSectionKind,
        target: SonataSectionKind,
        transformation: ReturnTransformation,
        interval_similarity: f32,
        exact_pitch_similarity: f32,
        /// Transformation-aware thematic evidence. `None` only appears in
        /// legacy serialized records produced before motif-return-v1.
        #[serde(default)]
        return_evidence: Option<MotifReturnEvidence>,
    },
    ClimaxLocation {
        section: SonataSectionKind,
        section_peak_intensity: f32,
        global_peak_intensity: f32,
        contains_climax_marker: bool,
    },
    CadentialArrival {
        section: SonataSectionKind,
        target: PitchClass,
        final_event_contains_target: bool,
    },
    Unsupported {
        obligation: String,
    },
}

const REACH_CONTRAST_KEY: u64 = 1;
const REACH_DEVELOPMENT_CLIMAX: u64 = 2;
const RETURN_PRIMARY_SUBJECT: u64 = 3;
const RETURN_HOME_KEY: u64 = 4;
const RETURN_SECONDARY_SUBJECT: u64 = 5;
const CLOSE_IN_HOME_KEY: u64 = 6;

struct SectionBlueprint {
    kind: SonataSectionKind,
    role: SectionRole,
    key_fn: fn(Key) -> Key,
    /// `None` = the theme as given (P); `Some(choice)` = the second
    /// subject's transform, shared between exposition and recap.
    transform: Option<u64>,
    /// `None` = a plain restatement of P; `Some(beats)` = the
    /// development's fragmented head, kept to this many beats.
    fragment_beats: Option<f64>,
    /// Exposition S and recap S share a progression-seed group so their
    /// degree sequence is identical; only the realization key differs.
    seed_group: u64,
}

fn identity_key(k: Key) -> Key {
    k
}

/// The exposition's second-subject key: the dominant in major, the
/// relative major in minor/modal.
fn contrast_key(home: Key) -> Key {
    match home.tonality {
        crate::harmony::Tonality::Major => home.dominant(),
        _ => home.relative(),
    }
}

/// The development's key: a third key distinct from both home and contrast.
fn development_key(home: Key) -> Key {
    contrast_key(home).relative()
}

fn section_blueprints(seed: u64, meter: f64) -> [SectionBlueprint; 5] {
    let s_choice = seed % 3;
    let bars = 4usize;
    let dev_beats = meter * bars as f64 * 0.6;
    [
        SectionBlueprint {
            kind: SonataSectionKind::ExpositionPrimary,
            role: SectionRole::A,
            key_fn: identity_key,
            transform: None,
            fragment_beats: None,
            seed_group: 0,
        },
        SectionBlueprint {
            kind: SonataSectionKind::ExpositionSecondary,
            role: SectionRole::B,
            key_fn: contrast_key,
            transform: Some(s_choice),
            fragment_beats: None,
            seed_group: 1,
        },
        SectionBlueprint {
            kind: SonataSectionKind::Development,
            role: SectionRole::C,
            key_fn: development_key,
            transform: None,
            fragment_beats: Some(dev_beats),
            seed_group: 2,
        },
        SectionBlueprint {
            kind: SonataSectionKind::RecapitulationPrimary,
            role: SectionRole::A,
            key_fn: identity_key,
            transform: None,
            fragment_beats: None,
            seed_group: 0,
        },
        SectionBlueprint {
            kind: SonataSectionKind::RecapitulationSecondary,
            role: SectionRole::ReturnA,
            key_fn: identity_key,
            transform: Some(s_choice),
            fragment_beats: None,
            seed_group: 1,
        },
    ]
}

/// Plan the sonata's exact sections, tonal regions, boundaries, and
/// prospective obligations without rendering notes or audio.
///
/// The returned ledger is intentionally pending. A cognitive planner can
/// inspect it while the piece unfolds; [`realize_sonata_with_plan`] returns a
/// separate resolution ledger containing evidence for the promises the fixed
/// realization fulfilled.
pub fn plan_sonata(home_key: Key, meter: f64, motif: &Motif, seed: u64) -> SonataPlan {
    let pivot = motif.notes.iter().find_map(|x| x.degree).unwrap_or(1);
    let bars = 4usize;
    let mut cursor = Duration::zero();
    let mut sections = Vec::with_capacity(5);

    for blueprint in section_blueprints(seed, meter) {
        let key = (blueprint.key_fn)(home_key);
        let mut section_motif = match blueprint.transform {
            Some(choice) => contrasting_transform(motif, pivot, choice),
            None => motif.clone(),
        };
        if let Some(keep) = blueprint.fragment_beats {
            section_motif = crate::fugue::head_fragment(
                &section_motif,
                Duration::new((keep.max(1.0) * 480.0) as i64, 480),
            );
        }
        let dominant = key.cadence_dominant_degree();
        let seed_variant = seed ^ (0x50DA_u64.wrapping_mul(blueprint.seed_group + 1));
        let progression = Progression::generate(bars, seed_variant);
        let period = Period::parallel_in(&section_motif, &progression.degrees, meter, dominant);
        let start = cursor;
        let end = start + period.total_duration();
        sections.push(PlannedSonataSection {
            kind: blueprint.kind,
            role: blueprint.role,
            key,
            period,
            start,
            end,
        });
        cursor = end;
    }

    let exposition_secondary = &sections[1];
    let development = &sections[2];
    let recapitulation_primary = &sections[3];
    let recapitulation_secondary = &sections[4];
    let mut obligations = ObligationLedger::new();
    obligations.add(CompositionalObligation::new(
        REACH_CONTRAST_KEY,
        Duration::zero(),
        exposition_secondary.start,
        0.75,
        ObligationKind::ReachKey {
            key: contrast_key(home_key),
        },
    ));
    obligations.add(CompositionalObligation::new(
        REACH_DEVELOPMENT_CLIMAX,
        Duration::zero(),
        development.end,
        0.8,
        ObligationKind::ReachClimax,
    ));
    obligations.add(CompositionalObligation::new(
        RETURN_PRIMARY_SUBJECT,
        Duration::zero(),
        recapitulation_primary.end,
        1.0,
        ObligationKind::ReturnMotif {
            motif_id: "sonata.primary".into(),
            transformation: ReturnTransformation::Literal,
        },
    ));
    obligations.add(CompositionalObligation::new(
        RETURN_HOME_KEY,
        Duration::zero(),
        recapitulation_primary.start,
        1.0,
        ObligationKind::ReachKey { key: home_key },
    ));
    obligations.add(CompositionalObligation::new(
        RETURN_SECONDARY_SUBJECT,
        Duration::zero(),
        recapitulation_secondary.end,
        1.0,
        ObligationKind::ReturnMotif {
            motif_id: "sonata.secondary".into(),
            transformation: ReturnTransformation::Transposed,
        },
    ));
    obligations.add(CompositionalObligation::new(
        CLOSE_IN_HOME_KEY,
        Duration::zero(),
        recapitulation_secondary.end,
        1.0,
        ObligationKind::Cadence {
            arrival_degree: AlteredDegree::diatonic(1),
        },
    ));

    SonataPlan {
        home_key,
        contrast_key: contrast_key(home_key),
        development_key: development_key(home_key),
        sections,
        obligations,
    }
}

pub(crate) fn realize_sonata(
    home_key: Key,
    tempo: f32,
    meter: f64,
    motif: &Motif,
    seed: u64,
    intent: &MusicalIntent,
) -> Score {
    realize_sonata_with_plan(home_key, tempo, meter, motif, seed, intent).score
}

/// Realize a sonata while preserving its plan and explicit obligation evidence.
pub fn realize_sonata_with_plan(
    home_key: Key,
    tempo: f32,
    meter: f64,
    motif: &Motif,
    seed: u64,
    intent: &MusicalIntent,
) -> SonataRealization {
    let plan = plan_sonata(home_key, meter, motif, seed);
    let opening_meter = meter as u8;
    let mut score = Score::new(home_key, tempo, opening_meter);
    let mut prev_upper: Vec<Pitch> = Vec::new();
    let mut prev_bass: Option<Pitch> = None;
    let mut cursor = Duration::zero();
    let pattern = crate::accompaniment::Accompaniment::Block;
    for section in &plan.sections {
        let form = Form {
            sections: vec![Section {
                role: section.role,
                key: section.key,
                period: section.period.clone(),
            }],
        };

        let mut phrase_score = Score::new(section.key, tempo, opening_meter);
        crate::composer::realize_melody(
            &mut phrase_score,
            &form,
            intent,
            Duration::zero(),
            meter,
            false,
        );
        // Bass is realized BEFORE harmony so `realize_harmony_measures` can read the
        // ACTUAL sounding bass from the score and voice the upper parts against it
        // (rootless chords + bass-vs-upper parallel fifths, both measured 2026-07-30).
        // Purely a reordering: the two use independent `prev_bass`/`prev_upper` chains
        // and never read each other's state, so the emitted NOTES are unchanged.
        crate::composer::realize_bass(
            &mut phrase_score,
            &form,
            meter,
            intent,
            &mut prev_bass,
            pattern,
            true,
            false,
        );
        crate::composer::realize_harmony(
            &mut phrase_score,
            &form,
            meter,
            intent,
            &mut prev_upper,
            pattern,
            true,
            true,
            false,
        );

        for n in &phrase_score.notes {
            let mut shifted = *n;
            shifted.onset = shifted.onset + cursor;
            score.push(shifted);
        }
        cursor = cursor + phrase_score.total_beats;
    }

    let (resolution, verification) = verify_sonata_obligations(&score, &plan);

    SonataRealization {
        score,
        plan,
        resolution,
        verification,
    }
}

/// Verify every sonata obligation from the completed score rather than from
/// constructor control flow. A promise remains pending when the observable
/// evidence does not meet its explicit threshold.
pub fn verify_sonata_obligations(
    score: &Score,
    plan: &SonataPlan,
) -> (ObligationLedger, Vec<SonataObligationEvidence>) {
    let mut resolution = plan.obligations.clone();
    let mut evidence = Vec::with_capacity(plan.obligations.items().len());

    for obligation in plan.obligations.items() {
        let metric = metric_for_obligation(score, plan, obligation);
        let verified = metric_is_verified(&metric);
        if verified {
            resolution.fulfil(obligation.id, format!("score-side verifier: {metric:?}"));
        }
        evidence.push(SonataObligationEvidence {
            obligation_id: obligation.id,
            verified,
            metric,
        });
    }

    (resolution, evidence)
}

fn metric_for_obligation(
    score: &Score,
    plan: &SonataPlan,
    obligation: &CompositionalObligation,
) -> SonataVerificationMetric {
    match &obligation.kind {
        ObligationKind::ReturnMotif {
            motif_id,
            transformation,
        } => {
            let pair = match motif_id.as_str() {
                "sonata.primary" => Some((
                    SonataSectionKind::ExpositionPrimary,
                    SonataSectionKind::RecapitulationPrimary,
                )),
                "sonata.secondary" => Some((
                    SonataSectionKind::ExpositionSecondary,
                    SonataSectionKind::RecapitulationSecondary,
                )),
                _ => None,
            };
            let Some((source_kind, target_kind)) = pair else {
                return SonataVerificationMetric::Unsupported {
                    obligation: format!("ReturnMotif({motif_id})"),
                };
            };
            let source = planned_section(plan, source_kind);
            let target = planned_section(plan, target_kind);
            let return_evidence = source.zip(target).map(|(source, target)| {
                compare_melodic_regions(
                    score,
                    source.start,
                    source.end,
                    target.start,
                    target.end,
                    *transformation,
                )
            });
            let interval_similarity = return_evidence
                .as_ref()
                .map_or(0.0, |evidence| evidence.interval_similarity);
            let exact_pitch_similarity = return_evidence
                .as_ref()
                .map_or(0.0, |evidence| evidence.literal_pitch_similarity);
            SonataVerificationMetric::MotifReturn {
                source: source_kind,
                target: target_kind,
                transformation: *transformation,
                interval_similarity,
                exact_pitch_similarity,
                return_evidence,
            }
        }
        ObligationKind::ReachKey { key } => {
            let section = section_for_key_obligation(plan, *key, obligation.due_by)
                .unwrap_or(SonataSectionKind::RecapitulationPrimary);
            let target = key.tonic;
            let (tonic_duration_share, final_event_contains_target) =
                planned_section(plan, section)
                    .map(|region| tonal_anchor(score, region, target))
                    .unwrap_or((0.0, false));
            SonataVerificationMetric::TonalAnchor {
                section,
                target,
                tonic_duration_share,
                final_event_contains_target,
            }
        }
        ObligationKind::Cadence { arrival_degree } => {
            let region = section_due_at(plan, obligation.due_by)
                .or_else(|| planned_section(plan, SonataSectionKind::RecapitulationSecondary));
            let section = region
                .map(|section| section.kind)
                .unwrap_or(SonataSectionKind::RecapitulationSecondary);
            let key = region.map(|section| section.key).unwrap_or(plan.home_key);
            let target = arrival_degree.pitch_class_in(key);
            let final_event_contains_target = region
                .map(|section| final_event_contains(score, section, target))
                .unwrap_or(false);
            SonataVerificationMetric::CadentialArrival {
                section,
                target,
                final_event_contains_target,
            }
        }
        ObligationKind::ReachClimax => {
            let section = SonataSectionKind::Development;
            let development = planned_section(plan, section);
            let section_notes = development
                .map(|region| notes_in_section(score, region, None))
                .unwrap_or_default();
            let section_peak_intensity = section_notes
                .iter()
                .map(|note| note.section_intensity)
                .fold(0.0_f32, f32::max);
            let global_peak_intensity = score
                .notes
                .iter()
                .map(|note| note.section_intensity)
                .fold(0.0_f32, f32::max);
            let contains_climax_marker = section_notes
                .iter()
                .any(|note| note.emphasis == Emphasis::Climax);
            SonataVerificationMetric::ClimaxLocation {
                section,
                section_peak_intensity,
                global_peak_intensity,
                contains_climax_marker,
            }
        }
        other => SonataVerificationMetric::Unsupported {
            obligation: format!("{other:?}"),
        },
    }
}

fn metric_is_verified(metric: &SonataVerificationMetric) -> bool {
    match metric {
        SonataVerificationMetric::TonalAnchor {
            tonic_duration_share,
            final_event_contains_target,
            ..
        } => *final_event_contains_target && *tonic_duration_share >= 0.02,
        SonataVerificationMetric::MotifReturn {
            transformation,
            interval_similarity,
            exact_pitch_similarity,
            return_evidence,
            ..
        } => return_evidence.as_ref().map_or_else(
            || match transformation {
                ReturnTransformation::Literal => *exact_pitch_similarity >= 0.999,
                ReturnTransformation::Transposed
                | ReturnTransformation::Restored
                | ReturnTransformation::Augmented
                | ReturnTransformation::Diminished => *interval_similarity >= 0.999,
                ReturnTransformation::Inverted | ReturnTransformation::Fragmented => false,
            },
            |evidence| {
                let threshold = match transformation {
                    ReturnTransformation::Literal | ReturnTransformation::Transposed => 0.95,
                    ReturnTransformation::Restored => 0.90,
                    ReturnTransformation::Inverted
                    | ReturnTransformation::Augmented
                    | ReturnTransformation::Diminished => 0.85,
                    ReturnTransformation::Fragmented => 0.75,
                };
                evidence.meets_threshold(threshold)
            },
        ),
        SonataVerificationMetric::ClimaxLocation {
            section_peak_intensity,
            global_peak_intensity,
            contains_climax_marker,
            ..
        } => {
            *contains_climax_marker
                && *section_peak_intensity + f32::EPSILON >= *global_peak_intensity
        }
        SonataVerificationMetric::CadentialArrival {
            final_event_contains_target,
            ..
        } => *final_event_contains_target,
        SonataVerificationMetric::Unsupported { .. } => false,
    }
}

fn planned_section(plan: &SonataPlan, kind: SonataSectionKind) -> Option<&PlannedSonataSection> {
    plan.sections.iter().find(|section| section.kind == kind)
}

fn section_due_at(plan: &SonataPlan, due_by: Duration) -> Option<&PlannedSonataSection> {
    plan.sections.iter().find(|section| section.end == due_by)
}

fn section_for_key_obligation(
    plan: &SonataPlan,
    key: Key,
    due_by: Duration,
) -> Option<SonataSectionKind> {
    // Key-arrival promises in the sonata plan are due at the boundary where
    // the new tonal region begins. Prefer that region over the section that
    // merely ended at the same timestamp.
    plan.sections
        .iter()
        .find(|section| section.start == due_by && section.key == key)
        .or_else(|| section_due_at(plan, due_by).filter(|section| section.key == key))
        .or_else(|| {
            plan.sections
                .iter()
                .find(|section| section.key == key && section.end.beats() <= due_by.beats())
        })
        .map(|section| section.kind)
}

fn notes_in_section(
    score: &Score,
    section: &PlannedSonataSection,
    role: Option<VoiceRole>,
) -> Vec<ScoreNote> {
    let mut notes: Vec<_> = score
        .notes
        .iter()
        .copied()
        .filter(|note| {
            note.onset.beats() >= section.start.beats()
                && note.onset.beats() < section.end.beats()
                && role.is_none_or(|role| note.role == role)
        })
        .collect();
    notes.sort_by(|left, right| left.onset.beats().total_cmp(&right.onset.beats()));
    notes
}

fn tonal_anchor(score: &Score, section: &PlannedSonataSection, target: PitchClass) -> (f32, bool) {
    let notes = notes_in_section(score, section, None);
    let total_duration: f64 = notes.iter().map(|note| note.duration.beats()).sum();
    let tonic_duration: f64 = notes
        .iter()
        .filter(|note| note.pitch.pitch_class() == target)
        .map(|note| note.duration.beats())
        .sum();
    let share = if total_duration <= f64::EPSILON {
        0.0
    } else {
        (tonic_duration / total_duration) as f32
    };
    (share, final_event_contains(score, section, target))
}

fn final_event_contains(score: &Score, section: &PlannedSonataSection, target: PitchClass) -> bool {
    let notes = notes_in_section(score, section, None);
    let Some(final_onset) = notes
        .iter()
        .map(|note| note.onset)
        .max_by(|left, right| left.beats().total_cmp(&right.beats()))
    else {
        return false;
    };
    notes
        .iter()
        .any(|note| note.onset == final_onset && note.pitch.pitch_class() == target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::PitchClass;

    fn intent() -> MusicalIntent {
        MusicalIntent::default()
    }

    fn theme() -> Motif {
        Motif::from_degrees(&[
            (1, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
            (3, Duration::quarter()),
        ])
    }

    #[test]
    fn exposition_second_subject_lands_in_a_real_foreign_key() {
        let home = Key::major(PitchClass::C);
        assert_eq!(contrast_key(home), home.dominant());
        assert_ne!(contrast_key(home), home);

        let minor_home = Key::minor(PitchClass::A);
        assert_eq!(contrast_key(minor_home), minor_home.relative());
    }

    #[test]
    fn development_key_differs_from_both_home_and_contrast() {
        let home = Key::major(PitchClass::C);
        let dev = development_key(home);
        assert_ne!(dev, home);
        assert_ne!(dev, contrast_key(home));
    }

    #[test]
    fn recap_second_subject_resolves_home_with_the_identical_idea() {
        // The form's defining move, proven directly: build exposition-S
        // and recap-S exactly as `realize_sonata` does internally, and
        // confirm they carry the SAME scale-degree Phrase.line (same
        // idea) while their KEYS differ (exposition foreign, recap home)
        // — and, critically, recap-S's key equals home, closing the loop.
        let home = Key::major(PitchClass::C);
        let motif = theme();
        let pivot = motif.notes.iter().find_map(|x| x.degree).unwrap_or(1);
        let seed = 11u64;
        let s_choice = seed % 3;
        let bars = 4usize;
        let meter = 4.0;

        let s_motif = contrasting_transform(&motif, pivot, s_choice);
        let seed_variant = seed ^ (0x50DA_u64.wrapping_mul(1 + 1)); // seed_group 1, both exposition-S and recap-S
        let progression = Progression::generate(bars, seed_variant);

        let exposition_key = contrast_key(home);
        let recap_key = home; // identity_key(home) in the real plan

        let exposition_dom = exposition_key.cadence_dominant_degree();
        let recap_dom = recap_key.cadence_dominant_degree();
        let exposition_period =
            Period::parallel_in(&s_motif, &progression.degrees, meter, exposition_dom);
        let recap_period = Period::parallel_in(&s_motif, &progression.degrees, meter, recap_dom);

        assert_ne!(
            exposition_key, home,
            "the exposition's second subject must NOT be home yet"
        );
        assert_eq!(
            recap_key, home,
            "the recap's second subject MUST be home — the resolution"
        );
        assert_eq!(
            exposition_period.antecedent.line, recap_period.antecedent.line,
            "the same idea must return — identical scale-degree content"
        );
        assert_eq!(
            exposition_period.consequent.line, recap_period.consequent.line,
            "the same idea must return — identical scale-degree content"
        );
    }

    #[test]
    fn realize_sonata_composes_a_real_five_section_piece() {
        let home = Key::major(PitchClass::C);
        let motif = theme();
        let score = realize_sonata(home, 100.0, 4.0, &motif, 5, &intent());
        assert!(!score.notes.is_empty(), "a real piece must come out");
        assert!(
            score.melody_is_monophonic(),
            "the melody voice must stay monophonic across every splice"
        );
    }

    #[test]
    fn sonata_is_deterministic() {
        let home = Key::major(PitchClass::C);
        let motif = theme();
        let a = realize_sonata(home, 100.0, 4.0, &motif, 7, &intent());
        let b = realize_sonata(home, 100.0, 4.0, &motif, 7, &intent());
        assert_eq!(a.notes.len(), b.notes.len());
    }

    #[test]
    fn sonata_plan_makes_tonal_conflict_and_return_explicit() {
        let home = Key::major(PitchClass::C);
        let plan = plan_sonata(home, 4.0, &theme(), 13);

        assert_eq!(plan.sections.len(), 5);
        assert_eq!(
            plan.sections[1].kind,
            SonataSectionKind::ExpositionSecondary
        );
        assert_eq!(plan.sections[1].key, home.dominant());
        assert_eq!(
            plan.sections[3].kind,
            SonataSectionKind::RecapitulationPrimary
        );
        assert_eq!(plan.sections[3].key, home);
        assert_eq!(plan.obligations.pending().len(), 6);
        assert!(
            plan.sections
                .windows(2)
                .all(|pair| pair[0].end == pair[1].start)
        );
    }

    #[test]
    fn planned_realization_resolves_every_form_obligation() {
        let home = Key::major(PitchClass::C);
        let realized = realize_sonata_with_plan(home, 100.0, 4.0, &theme(), 17, &intent());

        assert!(realized.resolution.all_fulfilled());
        assert!(realized.resolution.unresolved().is_empty());
        assert_eq!(realized.score.total_beats, realized.plan.sections[4].end);
    }

    #[test]
    fn sonata_resolution_is_backed_by_score_side_evidence() {
        let home = Key::major(PitchClass::C);
        let realized = realize_sonata_with_plan(home, 100.0, 4.0, &theme(), 23, &intent());

        assert_eq!(
            realized.verification.len(),
            realized.plan.obligations.items().len()
        );
        assert!(realized.verification.iter().all(|item| item.verified));
        assert!(realized.resolution.fulfilled().iter().all(|item| {
            item.resolution_note
                .as_deref()
                .is_some_and(|note| note.contains("score-side verifier"))
        }));
    }

    #[test]
    fn key_arrival_evidence_is_attributed_to_the_entered_tonal_region() {
        let home = Key::major(PitchClass::C);
        let realized = realize_sonata_with_plan(home, 100.0, 4.0, &theme(), 27, &intent());

        let contrast = realized
            .verification
            .iter()
            .find(|item| item.obligation_id == REACH_CONTRAST_KEY)
            .unwrap();
        assert!(matches!(
            &contrast.metric,
            SonataVerificationMetric::TonalAnchor {
                section: SonataSectionKind::ExpositionSecondary,
                ..
            }
        ));
        let return_home = realized
            .verification
            .iter()
            .find(|item| item.obligation_id == RETURN_HOME_KEY)
            .unwrap();
        assert!(matches!(
            &return_home.metric,
            SonataVerificationMetric::TonalAnchor {
                section: SonataSectionKind::RecapitulationPrimary,
                ..
            }
        ));
    }

    #[test]
    fn damaged_recapitulation_does_not_receive_a_false_motif_resolution() {
        let home = Key::major(PitchClass::C);
        let realized = realize_sonata_with_plan(home, 100.0, 4.0, &theme(), 29, &intent());
        let recap =
            planned_section(&realized.plan, SonataSectionKind::RecapitulationPrimary).unwrap();
        let mut damaged = realized.score.clone();
        damaged.notes.retain(|note| {
            note.role != VoiceRole::Melody
                || note.onset.beats() < recap.start.beats()
                || note.onset.beats() >= recap.end.beats()
        });

        let (resolution, evidence) = verify_sonata_obligations(&damaged, &realized.plan);
        assert!(resolution.get(RETURN_PRIMARY_SUBJECT).unwrap().is_pending());
        assert!(
            !evidence
                .iter()
                .find(|item| item.obligation_id == RETURN_PRIMARY_SUBJECT)
                .unwrap()
                .verified
        );
    }

    #[test]
    fn compatibility_realizer_preserves_the_planned_score() {
        let home = Key::major(PitchClass::C);
        let motif = theme();
        let direct = realize_sonata(home, 100.0, 4.0, &motif, 19, &intent());
        let planned = realize_sonata_with_plan(home, 100.0, 4.0, &motif, 19, &intent());
        assert_eq!(direct, planned.score);
    }
}
