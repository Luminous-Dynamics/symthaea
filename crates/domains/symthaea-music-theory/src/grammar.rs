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

#[derive(Debug, Clone, PartialEq)]
pub struct GrammarRealization {
    pub score: crate::score::Score,
    pub form: Option<crate::form::Form>,
    pub plan: GrammarPlanEvidence,
    pub trace: crate::grammar_trace::GrammarStructuralTrace,
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
                    phrase_model: if intent.seed % 2 == 0 {
                        "period"
                    } else {
                        "sentence"
                    }
                    .into(),
                    section_count,
                    cadence_strategy: "antecedent-half-close; consequent-full-close".into(),
                    continuation_strategy: "fragmentation-and-acceleration".into(),
                    return_obligation: "primary material returns before closure".into(),
                    ending_strategy: "motif-related cadential liquidation".into(),
                }),
                trace: Default::default(),
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
            }
        }
        GrammarFamily::GrooveCycle => {
            let realized = crate::groove_cycle::realize_groove_cycle(intent, spec);
            GrammarRealization {
                score: realized.score,
                form: None,
                plan: GrammarPlanEvidence::GrooveCycle(realized.plan),
                trace: Default::default(),
            }
        }
        GrammarFamily::ProcessAdditive => {
            let realized = crate::process_grammar::realize_additive_process(intent, spec);
            GrammarRealization {
                score: realized.score,
                form: None,
                plan: GrammarPlanEvidence::AdditiveProcess(realized.plan),
                trace: Default::default(),
            }
        }
        GrammarFamily::RagaModalArc => {
            let realized = crate::modal_arc::realize_modal_arc(intent, spec);
            GrammarRealization {
                score: realized.score,
                form: None,
                plan: GrammarPlanEvidence::ModalArc(realized.plan),
                trace: Default::default(),
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
            }
        }
    };
    realized.trace = crate::grammar_trace::build_grammar_trace(
        profile.family,
        &realized.score,
        realized.form.as_ref(),
        &realized.plan,
        &motif,
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
}
