// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! First-class compositional grammar routing.
//!
//! A style specializes a grammar family; it is not itself a composer. The
//! profile is intentionally symbolic and renderer-independent so downstream
//! performance and production layers can select compatible dialects without
//! guessing from a preset name.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrammarFamily {
    PeriodSentence,
    Developmental,
    Contrapuntal,
    GroundVariation,
    StrophicSong,
    BluesCallResponse,
    JazzChorus,
    GrooveCycle,
    RagaModalArc,
    ProcessAdditive,
    AmbientTextural,
    DramaticAdaptive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhraseGrammar {
    Period,
    Sentence,
    DevelopmentalObligations,
    ImitativeEntries,
    GroundVariations,
    VerseRefrain,
    ChorusCallResponse,
    JazzChoruses,
    ContinuousCycle,
    ModalUnfolding,
    AdditiveProcess,
    TexturalHorizon,
    DramaticScenes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HarmonicSyntax {
    Functional,
    Modal,
    TonalRegions,
    ContrapuntalVoiceLeading,
    GroundCycle,
    SongLoop,
    BluesChorus,
    JazzTurnaround,
    CyclicVamp,
    DronePitchHierarchy,
    StaticProcessField,
    SpectralStasis,
    NarrativeLeitmotif,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerformanceDialect {
    ClassicalRubato,
    DanceLocked,
    JazzLaidBack,
    FolkLift,
    ProcessExact,
    DroneElastic,
    DramaticTiming,
    ChoralBlend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntentAxis {
    Valence,
    Arousal,
    Energy,
    Duration,
    TonalCenter,
    NarrativeState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct GrammarProfile {
    pub family: GrammarFamily,
    pub phrase: PhraseGrammar,
    pub harmony: HarmonicSyntax,
    pub performance: PerformanceDialect,
    pub supported_intent_axes: &'static [IntentAxis],
}

/// Auditable plan emitted beside a symbolic score. `Compatibility` is honest
/// about families that still use the established period-centered composer;
/// flagship bypass engines expose their real cycle/process/modal plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "plan", rename_all = "snake_case")]
pub enum GrammarPlanEvidence {
    PeriodSentence(PeriodSentencePlan),
    Contrapuntal(ContrapuntalPlan),
    GrooveCycle(crate::groove_cycle::GrooveCyclePlan),
    AdditiveProcess(crate::process_grammar::ProcessPlan),
    ModalArc(crate::modal_arc::ModalArcPlan),
    CallResponse(crate::call_response::CallResponsePlan),
    JazzChorus(crate::jazz_chorus::JazzChorusPlan),
    Compatibility {
        family: GrammarFamily,
        form_available: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PeriodSentencePlan {
    pub phrase_model: String,
    pub section_count: usize,
    pub cadence_strategy: String,
    pub continuation_strategy: String,
    pub return_obligation: String,
    pub ending_strategy: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContrapuntalPlan {
    pub engine_version: String,
    pub answer_class: String,
    pub subject_entry_order: Vec<String>,
    pub countersubject: bool,
    pub inversion: bool,
    pub augmentation: bool,
    pub diminution: bool,
    pub stretto: bool,
    pub controlled_dissonance: bool,
    pub final_combination: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisterPlacementDisposition {
    AlreadyWithinBounds,
    OctaveRebalanced,
    BlockedByOtherInvariant,
    NoLegalOctavePlacement,
    RepairRejectedByValidator,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartRegisterPlacement {
    pub part_id: u16,
    pub before_min_midi: u8,
    pub before_max_midi: u8,
    /// Signed semitone shift; the canonical algorithm emits multiples of 12.
    pub semitone_shift: i16,
    pub after_min_midi: u8,
    pub after_max_midi: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisterPlacementReceipt {
    pub schema: String,
    pub validation_version: String,
    pub min_midi: u8,
    pub max_midi: u8,
    pub before_min_midi: Option<u8>,
    pub before_max_midi: Option<u8>,
    pub after_min_midi: Option<u8>,
    pub after_max_midi: Option<u8>,
    pub initial_fatal_universal_rules: Vec<crate::score_validation::ScoreValidationRule>,
    pub disposition: RegisterPlacementDisposition,
    pub part_placements: Vec<PartRegisterPlacement>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GrammarRealization {
    pub score: crate::score::Score,
    pub form: Option<crate::form::Form>,
    pub plan: GrammarPlanEvidence,
    pub trace: crate::grammar_trace::GrammarStructuralTrace,
    /// Proof record for the final renderer-independent octave placement pass.
    pub register_placement: Option<RegisterPlacementReceipt>,
}

/// Apply the smallest deterministic octave-only correction that makes every
/// part's notes fit the canonical MIDI bounds.
///
/// This is a last-mile score-feasibility transform, not a composer and not a
/// reason to loosen validation. It runs only when the original report's
/// fatal universal-invariant defects are exclusively NoteBounds. Every part
/// is shifted as a unit by a multiple of 12 semitones, preserving pitch
/// classes, within-part intervals, onsets, durations, velocity, emphasis and
/// section intensity. The candidate is committed only if the same independent
/// validator reports no fatal universal-invariant defects; otherwise the
/// original score is restored.
pub fn place_parts_within_validation_range(
    score: &mut crate::score::Score,
) -> RegisterPlacementReceipt {
    use crate::score_validation::{
        ScoreValidationConfig, ScoreValidationRule, ValidationSeverity,
        THEORY_VALIDATION_VERSION, is_universal_invariant, validate_score,
    };

    const SCHEMA: &str = "muse-register-placement-v1";
    let config = ScoreValidationConfig::default();
    let (before_min, before_max) = score_pitch_bounds(score);
    let initial_report = validate_score(score, &config);
    let mut initial_fatal = Vec::new();
    for issue in &initial_report.issues {
        if issue.severity == ValidationSeverity::Fatal
            && is_universal_invariant(issue.rule)
            && !initial_fatal.contains(&issue.rule)
        {
            initial_fatal.push(issue.rule);
        }
    }

    let receipt = |disposition, part_placements, after_min, after_max| {
        RegisterPlacementReceipt {
            schema: SCHEMA.into(),
            validation_version: THEORY_VALIDATION_VERSION.into(),
            min_midi: config.min_midi,
            max_midi: config.max_midi,
            before_min_midi: before_min,
            before_max_midi: before_max,
            after_min_midi: after_min,
            after_max_midi: after_max,
            initial_fatal_universal_rules: initial_fatal.clone(),
            disposition,
            part_placements,
        }
    };

    if initial_fatal.is_empty() {
        return receipt(
            RegisterPlacementDisposition::AlreadyWithinBounds,
            Vec::new(),
            before_min,
            before_max,
        );
    }

    if initial_fatal
        .iter()
        .any(|rule| *rule != ScoreValidationRule::NoteBounds)
    {
        return receipt(
            RegisterPlacementDisposition::BlockedByOtherInvariant,
            Vec::new(),
            before_min,
            before_max,
        );
    }

    let mut part_ids: Vec<u16> = score.notes.iter().map(|note| note.part.0).collect();
    part_ids.sort_unstable();
    part_ids.dedup();

    let mut placements = Vec::with_capacity(part_ids.len());
    for part_id in part_ids {
        let mut part_min = u8::MAX;
        let mut part_max = 0u8;
        for note in score.notes.iter().filter(|note| note.part.0 == part_id) {
            let midi = note.pitch.midi();
            part_min = part_min.min(midi);
            part_max = part_max.max(midi);
        }

        // Minimize absolute movement. Ties are resolved toward the lower
        // octave, making the result deterministic across platforms.
        let chosen_shift = (-10i16..=10)
            .map(|octaves| octaves * 12)
            .filter(|shift| {
                let low = i16::from(part_min) + *shift;
                let high = i16::from(part_max) + *shift;
                low >= i16::from(config.min_midi) && high <= i16::from(config.max_midi)
            })
            .min_by_key(|shift| (shift.abs(), *shift));

        let Some(semitone_shift) = chosen_shift else {
            return receipt(
                RegisterPlacementDisposition::NoLegalOctavePlacement,
                Vec::new(),
                before_min,
                before_max,
            );
        };

        placements.push(PartRegisterPlacement {
            part_id,
            before_min_midi: part_min,
            before_max_midi: part_max,
            semitone_shift,
            after_min_midi: (i16::from(part_min) + semitone_shift) as u8,
            after_max_midi: (i16::from(part_max) + semitone_shift) as u8,
        });
    }

    let original = score.clone();
    for placement in &placements {
        if placement.semitone_shift == 0 {
            continue;
        }
        for note in score
            .notes
            .iter_mut()
            .filter(|note| note.part.0 == placement.part_id)
        {
            note.pitch = note.pitch.transpose(i32::from(placement.semitone_shift));
        }
    }

    let repaired_report = validate_score(score, &config);
    let remaining_fatal_universal = repaired_report.issues.iter().any(|issue| {
        issue.severity == ValidationSeverity::Fatal && is_universal_invariant(issue.rule)
    });
    if remaining_fatal_universal {
        *score = original;
        return receipt(
            RegisterPlacementDisposition::RepairRejectedByValidator,
            placements,
            before_min,
            before_max,
        );
    }

    let (after_min, after_max) = score_pitch_bounds(score);
    let changed = placements.iter().any(|placement| placement.semitone_shift != 0);
    receipt(
        if changed {
            RegisterPlacementDisposition::OctaveRebalanced
        } else {
            RegisterPlacementDisposition::RepairRejectedByValidator
        },
        placements,
        after_min,
        after_max,
    )
}

fn score_pitch_bounds(score: &crate::score::Score) -> (Option<u8>, Option<u8>) {
    let mut min_midi = None::<u8>;
    let mut max_midi = None::<u8>;
    for note in &score.notes {
        let midi = note.pitch.midi();
        min_midi = Some(min_midi.map_or(midi, |value| value.min(midi)));
        max_midi = Some(max_midi.map_or(midi, |value| value.max(midi)));
    }
    (min_midi, max_midi)
}

/// Compose with an explicit grammar owner and retain its planning evidence.
pub fn compose_with_grammar_plan(
    profile: GrammarProfile,
    intent: &crate::composer::MusicalIntent,
    spec: &crate::spec::CompositionSpec,
) -> GrammarRealization {
    // Mirror `compose_with_spec_and_form`'s own hook-cell grafting exactly.
    // Without this, `GrammarFamily::Contrapuntal`'s fugue subject (and this
    // motif's use in the trailing structural trace below) would silently
    // diverge from what every other production path actually composes.
    let meter_beats = spec.meter as f64;
    let motif = spec.motif(intent.arousal, intent.seed);
    let motif = if spec.texture.hook_cell {
        crate::hook::graft_hook(
            &motif,
            &crate::hook::HookCell::generate_with(&spec.melody, intent.seed, meter_beats),
            meter_beats,
        )
    } else {
        motif
    };
    let mut realized = match profile.family {
        GrammarFamily::PeriodSentence => {
            let (score, form) = crate::composer::compose_with_spec_and_form_and_grammar(
                intent,
                spec,
                Some(profile),
            );
            let section_count = form.as_ref().map(|value| value.sections.len()).unwrap_or(0);
            GrammarRealization {
                score,
                form,
                plan: GrammarPlanEvidence::PeriodSentence(PeriodSentencePlan {
                    phrase_model: if crate::composer::use_sentence_for(Some(profile), intent) {
                        "sentence"
                    } else {
                        "period"
                    }
                    .into(),
                    section_count,
                    cadence_strategy: "antecedent-half-close; consequent-full-close".into(),
                    continuation_strategy: "fragmentation-and-acceleration".into(),
                    return_obligation: "primary material returns before closure".into(),
                    ending_strategy: "motif-related cadential liquidation".into(),
                }),
                trace: Default::default(),
                register_placement: None,
            }
        }
        GrammarFamily::Contrapuntal => {
            let key = spec
                .mode
                .and_then(|mode| crate::harmony::Key::modal(intent.tonic, mode))
                .unwrap_or_else(|| {
                    if intent.valence >= 0.0 {
                        crate::harmony::Key::major(intent.tonic)
                    } else {
                        crate::harmony::Key::minor(intent.tonic)
                    }
                });
            // Match `compose_with_spec_and_form`'s attitude-adjusted tempo
            // exactly (Joy lifts the pulse, Grief slows it) — the raw
            // `spec.tempo(...)` alone silently drops that adjustment for
            // any spec whose `attitude` is set.
            let tempo = spec.tempo(intent.arousal)
                * match spec.attitude {
                    Some(crate::spec::Attitude::Joy) => 1.08,
                    Some(crate::spec::Attitude::Grief) => 0.85,
                    _ => 1.0,
                };
            let score = crate::fugue::realize_fugue(key, tempo, spec.meter, &motif, intent.seed);
            GrammarRealization {
                score,
                form: None,
                plan: GrammarPlanEvidence::Contrapuntal(ContrapuntalPlan {
                    engine_version: "compact-fugue-engine-v1".into(),
                    answer_class: "real_diatonic_fifth".into(),
                    subject_entry_order: vec![
                        "soprano_subject".into(),
                        "alto_answer".into(),
                        "bass_subject".into(),
                    ],
                    countersubject: true,
                    inversion: true,
                    augmentation: true,
                    diminution: false,
                    stretto: true,
                    controlled_dissonance: true,
                    final_combination: true,
                }),
                trace: Default::default(),
                register_placement: None,
            }
        }
        GrammarFamily::GrooveCycle => {
            let realized = crate::groove_cycle::realize_groove_cycle(intent, spec);
            GrammarRealization {
                score: realized.score,
                form: None,
                plan: GrammarPlanEvidence::GrooveCycle(realized.plan),
                trace: Default::default(),
                register_placement: None,
            }
        }
        GrammarFamily::ProcessAdditive => {
            let realized = crate::process_grammar::realize_additive_process(intent, spec);
            GrammarRealization {
                score: realized.score,
                form: None,
                plan: GrammarPlanEvidence::AdditiveProcess(realized.plan),
                trace: Default::default(),
                register_placement: None,
            }
        }
        GrammarFamily::RagaModalArc => {
            let realized = crate::modal_arc::realize_modal_arc(intent, spec);
            GrammarRealization {
                score: realized.score,
                form: None,
                plan: GrammarPlanEvidence::ModalArc(realized.plan),
                trace: Default::default(),
                register_placement: None,
            }
        }
        GrammarFamily::BluesCallResponse => {
            let realized = crate::call_response::realize_call_response(intent, spec);
            GrammarRealization {
                score: realized.score,
                form: None,
                plan: GrammarPlanEvidence::CallResponse(realized.plan),
                trace: Default::default(),
                register_placement: None,
            }
        }
        GrammarFamily::JazzChorus => {
            let realized = crate::jazz_chorus::realize_jazz_chorus(intent, spec);
            GrammarRealization {
                score: realized.score,
                form: None,
                plan: GrammarPlanEvidence::JazzChorus(realized.plan),
                trace: Default::default(),
                register_placement: None,
            }
        }
        family => {
            let (score, form) = crate::composer::compose_with_spec_and_form_and_grammar(
                intent,
                spec,
                Some(profile),
            );
            GrammarRealization {
                score,
                plan: GrammarPlanEvidence::Compatibility {
                    family,
                    form_available: form.is_some(),
                },
                form,
                trace: Default::default(),
                register_placement: None,
            }
        }
    };
    // Universal safety net: every dedicated engine (GrooveCycle,
    // ProcessAdditive, RagaModalArc, BluesCallResponse, the fugue/period
    // pipeline) gets the same de-overlap/metadata-sync pass the shared
    // pipeline applies internally, so the trace below and the debug gate
    // after it both see the corrected score, not a pre-correction one.
    crate::composer::deoverlap_all_voices(&mut realized.score);
    crate::composer::sync_total_beats(&mut realized.score);
    // The period/sentence path already folds melody phrases before score
    // emission. Direct grammar engines (notably the Fugue) have their own
    // register logic and must pass through the same final feasibility gate.
    realized.register_placement = Some(place_parts_within_validation_range(
        &mut realized.score,
    ));
    realized.trace = crate::grammar_trace::build_grammar_trace(
        profile.family,
        &realized.score,
        realized.form.as_ref(),
        &realized.plan,
        &motif,
    );
    crate::score_validation::debug_assert_no_structural_defects(
        &realized.score,
        "compose_with_grammar_plan",
    );
    realized
}

/// Executable boundary between planning metadata and grammar-specific score
/// construction. New families implement this contract rather than adding a
/// top-level `Style` branch.
pub trait GrammarEngine {
    fn profile(&self) -> GrammarProfile;
    fn compose(
        &self,
        intent: &crate::composer::MusicalIntent,
        spec: &crate::spec::CompositionSpec,
    ) -> crate::score::Score;
}

pub trait PhraseGrammarEngine {
    fn phrase_grammar(&self) -> PhraseGrammar;
}

pub trait HarmonicGrammarEngine {
    fn harmonic_syntax(&self) -> HarmonicSyntax;
}

impl PhraseGrammarEngine for GrammarProfile {
    fn phrase_grammar(&self) -> PhraseGrammar {
        self.phrase
    }
}

impl HarmonicGrammarEngine for GrammarProfile {
    fn harmonic_syntax(&self) -> HarmonicSyntax {
        self.harmony
    }
}

impl GrammarEngine for GrammarProfile {
    fn profile(&self) -> GrammarProfile {
        *self
    }

    fn compose(
        &self,
        intent: &crate::composer::MusicalIntent,
        spec: &crate::spec::CompositionSpec,
    ) -> crate::score::Score {
        compose_with_grammar_plan(*self, intent, spec).score
    }
}

const STANDARD_AXES: &[IntentAxis] = &[
    IntentAxis::Valence,
    IntentAxis::Arousal,
    IntentAxis::Energy,
    IntentAxis::Duration,
    IntentAxis::TonalCenter,
];
const STRUCTURAL_AXES: &[IntentAxis] = &[
    IntentAxis::Arousal,
    IntentAxis::Energy,
    IntentAxis::Duration,
    IntentAxis::TonalCenter,
];
const CYCLE_AXES: &[IntentAxis] = &[
    IntentAxis::Arousal,
    IntentAxis::Energy,
    IntentAxis::Duration,
    IntentAxis::TonalCenter,
];
const MODAL_AXES: &[IntentAxis] = &[
    IntentAxis::Arousal,
    IntentAxis::Energy,
    IntentAxis::Duration,
    IntentAxis::TonalCenter,
];
const DRAMATIC_AXES: &[IntentAxis] = &[
    IntentAxis::Valence,
    IntentAxis::Arousal,
    IntentAxis::Energy,
    IntentAxis::Duration,
    IntentAxis::TonalCenter,
    IntentAxis::NarrativeState,
];

impl GrammarFamily {
    pub const fn profile(self) -> GrammarProfile {
        use GrammarFamily::*;
        match self {
            PeriodSentence => profile(
                self,
                PhraseGrammar::Period,
                HarmonicSyntax::Functional,
                PerformanceDialect::ClassicalRubato,
                STANDARD_AXES,
            ),
            Developmental => profile(
                self,
                PhraseGrammar::DevelopmentalObligations,
                HarmonicSyntax::TonalRegions,
                PerformanceDialect::ClassicalRubato,
                STRUCTURAL_AXES,
            ),
            Contrapuntal => profile(
                self,
                PhraseGrammar::ImitativeEntries,
                HarmonicSyntax::ContrapuntalVoiceLeading,
                PerformanceDialect::ChoralBlend,
                STRUCTURAL_AXES,
            ),
            GroundVariation => profile(
                self,
                PhraseGrammar::GroundVariations,
                HarmonicSyntax::GroundCycle,
                PerformanceDialect::ClassicalRubato,
                STRUCTURAL_AXES,
            ),
            StrophicSong => profile(
                self,
                PhraseGrammar::VerseRefrain,
                HarmonicSyntax::SongLoop,
                PerformanceDialect::FolkLift,
                STANDARD_AXES,
            ),
            BluesCallResponse => profile(
                self,
                PhraseGrammar::ChorusCallResponse,
                HarmonicSyntax::BluesChorus,
                PerformanceDialect::JazzLaidBack,
                STANDARD_AXES,
            ),
            JazzChorus => profile(
                self,
                PhraseGrammar::JazzChoruses,
                HarmonicSyntax::JazzTurnaround,
                PerformanceDialect::JazzLaidBack,
                STANDARD_AXES,
            ),
            GrooveCycle => profile(
                self,
                PhraseGrammar::ContinuousCycle,
                HarmonicSyntax::CyclicVamp,
                PerformanceDialect::DanceLocked,
                CYCLE_AXES,
            ),
            RagaModalArc => profile(
                self,
                PhraseGrammar::ModalUnfolding,
                HarmonicSyntax::DronePitchHierarchy,
                PerformanceDialect::DroneElastic,
                MODAL_AXES,
            ),
            ProcessAdditive => profile(
                self,
                PhraseGrammar::AdditiveProcess,
                HarmonicSyntax::StaticProcessField,
                PerformanceDialect::ProcessExact,
                STRUCTURAL_AXES,
            ),
            AmbientTextural => profile(
                self,
                PhraseGrammar::TexturalHorizon,
                HarmonicSyntax::SpectralStasis,
                PerformanceDialect::DroneElastic,
                STRUCTURAL_AXES,
            ),
            DramaticAdaptive => profile(
                self,
                PhraseGrammar::DramaticScenes,
                HarmonicSyntax::NarrativeLeitmotif,
                PerformanceDialect::DramaticTiming,
                DRAMATIC_AXES,
            ),
        }
    }
}

const fn profile(
    family: GrammarFamily,
    phrase: PhraseGrammar,
    harmony: HarmonicSyntax,
    performance: PerformanceDialect,
    supported_intent_axes: &'static [IntentAxis],
) -> GrammarProfile {
    GrammarProfile {
        family,
        phrase,
        harmony,
        performance,
        supported_intent_axes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flagship_families_are_structurally_distinct() {
        let groove = GrammarFamily::GrooveCycle.profile();
        let process = GrammarFamily::ProcessAdditive.profile();
        let raga = GrammarFamily::RagaModalArc.profile();
        assert_ne!(groove.phrase, process.phrase);
        assert_ne!(process.harmony, raga.harmony);
        assert_ne!(raga.performance, groove.performance);
    }

    #[test]
    fn period_sentence_plan_reports_the_same_model_the_composer_uses() {
        use crate::composer::MusicalIntent;
        use crate::style::Style;

        let style = Style::Classical;
        assert_eq!(
            style.grammar_profile().family,
            GrammarFamily::PeriodSentence
        );
        for (arousal, energy, seed) in [
            (0.0, 0.0, 1),
            (0.5, 0.5, 2),
            (0.6, 0.75, 3),
            (0.61, 0.5, 4),
            (0.5, 0.76, 5),
            (1.0, 1.0, 6),
        ] {
            let intent = MusicalIntent {
                arousal,
                energy,
                seed,
                ..MusicalIntent::default()
            };
            let expected = if crate::composer::use_sentence_for(
                Some(style.grammar_profile()),
                &intent,
            ) {
                "sentence"
            } else {
                "period"
            };
            let realization = compose_with_grammar_plan(
                style.grammar_profile(),
                &intent,
                &style.spec(),
            );
            let GrammarPlanEvidence::PeriodSentence(plan) = realization.plan else {
                panic!("Classical must emit a PeriodSentence plan");
            };
            assert_eq!(
                plan.phrase_model, expected,
                "arousal={arousal}, energy={energy}, seed={seed}"
            );
        }
    }

    #[test]
    fn flagship_styles_route_to_the_expected_family() {
        use crate::style::Style;
        assert_eq!(
            Style::AfroCuban.grammar_profile().family,
            GrammarFamily::GrooveCycle
        );
        assert_eq!(
            Style::Minimalism.grammar_profile().family,
            GrammarFamily::ProcessAdditive
        );
        assert_eq!(
            Style::HindustaniInspired.grammar_profile().family,
            GrammarFamily::RagaModalArc
        );
    }

    #[test]
    fn fugue_tonic_and_seed_grid_never_emits_out_of_range_notes() {
        use crate::composer::MusicalIntent;
        use crate::pitch::PitchClass;
        use crate::score_validation::{
            ScoreValidationConfig, ValidationSeverity, is_universal_invariant, validate_score,
        };
        use crate::style::Style;

        let style = Style::Fugue;
        let mut saw_rebalance = false;
        for tonic in [0, 3, 4, 7, 10] {
            for seed in [90_210u64, 130_363, 222_731, 918_273] {
                let intent = MusicalIntent {
                    tonic: PitchClass::new(tonic),
                    seed,
                    ..MusicalIntent::default()
                };
                let realization = compose_with_grammar_plan(
                    style.grammar_profile(),
                    &intent,
                    &style.spec(),
                );
                let receipt = realization
                    .register_placement
                    .as_ref()
                    .expect("canonical grammar composition must include placement evidence");
                if receipt.disposition == RegisterPlacementDisposition::OctaveRebalanced {
                    saw_rebalance = true;
                }

                let report = validate_score(
                    &realization.score,
                    &ScoreValidationConfig::default(),
                );
                let fatal_universal: Vec<_> = report
                    .issues
                    .iter()
                    .filter(|issue| {
                        issue.severity == ValidationSeverity::Fatal
                            && is_universal_invariant(issue.rule)
                    })
                    .collect();
                assert!(
                    fatal_universal.is_empty(),
                    "Fugue tonic={tonic} seed={seed} has fatal universal issues: {fatal_universal:?}; receipt={receipt:?}"
                );
                for note in &realization.score.notes {
                    assert!(
                        (24..=108).contains(&note.pitch.midi()),
                        "Fugue tonic={tonic} seed={seed} escaped MIDI bounds: {note:?}"
                    );
                }
            }
        }
        assert!(
            saw_rebalance,
            "the frozen regression grid must exercise at least one octave repair"
        );
    }

    #[test]
    fn octave_placement_preserves_pitch_classes_intervals_and_rhythm() {
        use crate::harmony::Key;
        use crate::pitch::{Pitch, PitchClass};
        use crate::rhythm::Duration;
        use crate::score::{Emphasis, PartId, Score, ScoreNote, VoiceRole};

        let mut score = Score::new(Key::major(PitchClass::new(0)), 90.0, 4);
        for (part, midi, onset, role) in [
            (PartId(0), 110u8, 0i64, VoiceRole::Melody),
            (PartId(0), 106u8, 1i64, VoiceRole::Melody),
            (PartId(1), 48u8, 0i64, VoiceRole::Bass),
        ] {
            score.push(ScoreNote {
                part,
                pitch: Pitch::from_midi(midi),
                onset: Duration::new(onset, 1),
                duration: Duration::new(1, 1),
                velocity: 0.7,
                role,
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
        let original: Vec<_> = score.notes.iter().map(|note| {
            (note.part, note.pitch, note.onset, note.duration, note.velocity, note.emphasis)
        }).collect();

        let receipt = place_parts_within_validation_range(&mut score);
        assert_eq!(receipt.disposition, RegisterPlacementDisposition::OctaveRebalanced);
        let melody_shift = receipt
            .part_placements
            .iter()
            .find(|placement| placement.part_id == 0)
            .expect("melody part placement")
            .semitone_shift;
        assert_eq!(melody_shift % 12, 0);
        assert_ne!(melody_shift, 0);

        for (before, after) in original.iter().zip(&score.notes) {
            assert_eq!(before.0, after.part);
            assert_eq!(before.1.midi() % 12, after.pitch.midi() % 12);
            assert_eq!(before.2, after.onset);
            assert_eq!(before.3, after.duration);
            assert_eq!(before.4, after.velocity);
            assert_eq!(before.5, after.emphasis);
            assert_eq!(
                i16::from(after.pitch.midi()) - i16::from(before.1.midi()),
                if before.0.0 == 0 { melody_shift } else { 0 },
                "each part must receive its own uniform octave offset"
            );
        }
        let before_interval = i16::from(original[0].1.midi()) - i16::from(original[1].1.midi());
        let after_interval = i16::from(score.notes[0].pitch.midi())
            - i16::from(score.notes[1].pitch.midi());
        assert_eq!(before_interval, after_interval);
        let after_report = crate::score_validation::validate_score(
            &score,
            &crate::score_validation::ScoreValidationConfig::default(),
        );
        assert!(
            !after_report.issues.iter().any(|issue| {
                issue.severity == crate::score_validation::ValidationSeverity::Fatal
                    && crate::score_validation::is_universal_invariant(issue.rule)
            }),
            "rebalanced score must pass universal structural validation"
        );
    }

    #[test]
    fn octave_placement_refuses_parts_that_cannot_fit_without_note_warping() {
        use crate::harmony::Key;
        use crate::pitch::{Pitch, PitchClass};
        use crate::rhythm::Duration;
        use crate::score::{Emphasis, PartId, Score, ScoreNote, VoiceRole};

        let mut score = Score::new(Key::major(PitchClass::new(0)), 90.0, 4);
        for (midi, onset) in [(24u8, 0i64), (120u8, 1i64)] {
            score.push(ScoreNote {
                part: PartId(7),
                pitch: Pitch::from_midi(midi),
                onset: Duration::new(onset, 1),
                duration: Duration::new(1, 1),
                velocity: 0.5,
                role: VoiceRole::Melody,
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
        let original = score.clone();
        let receipt = place_parts_within_validation_range(&mut score);
        assert_eq!(
            receipt.disposition,
            RegisterPlacementDisposition::NoLegalOctavePlacement
        );
        assert_eq!(score, original, "unrepairable scores must remain unchanged");
    }

}
