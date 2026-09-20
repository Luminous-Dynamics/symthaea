// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # symthaea-music-theory
//!
//! Symbolic music theory: the "what to play" layer. Nothing here knows what a
//! Hz is — the atom is a **pitch class**, not a frequency. `symthaea-muse`
//! consumes the symbolic `Score` this crate produces and realizes it as audio.
//!
//! See `DESIGN.md` for the full architecture and the reasoning (Symthaea's
//! music sounded aimless because every dimension was an independent random
//! draw over frequencies; real feeling needs symbolic structure — motifs,
//! functional harmony, cadences, phrase question-and-answer).
//!
//! ## Layers
//!
//! - **Layer 0** (here): [`Pitch`], [`PitchClass`], [`Interval`], [`Scale`],
//!   [`Chord`] — primitives. Every fact is unit-tested against a textbook
//!   ground truth (a major triad IS [0,4,7]).
//! - Layers 1–4 (harmony, melody/motif, form, consciousness mapping) build on
//!   these — see `DESIGN.md`.
//!
//! ## Ground-truth ethos
//!
//! Music theory has correct answers. Every rule in this crate ships a test
//! asserting a known fact. If we can't state the ground-truth property, we
//! don't ship the rule. This crate is the anti-scaffold.

#![deny(unsafe_code)]

/// Package version recorded in deterministic piece recipes and export manifests.
pub const MUSIC_THEORY_ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod accompaniment;
pub mod cadence;
pub mod call_response;
pub mod chord;
pub mod cognitive_analysis;
pub mod composer;
pub mod contrapuntal_foundry;
pub mod counterpoint;
pub mod describe;
pub mod development_program;
pub mod diversity_plan;
pub mod explorer;
pub mod fingerprint;
pub mod form;
pub mod foundry;
pub mod fugue;
pub mod grammar;
pub mod grammar_trace;
pub mod groove_cycle;
pub mod harmony;
pub mod harmony_verifier;
pub mod hook;
pub mod integration;
pub mod jazz_chorus;
pub mod live;
pub mod melodic_contour_identity;
pub mod meter;
pub mod modal_arc;
pub mod motif;
pub mod motif_family;
pub mod motif_foundry;
pub mod motif_return;
pub mod obligation;
pub mod opera;
pub mod passacaglia;
pub mod phrase;
pub mod pitch;
pub mod premise;
pub mod process_grammar;
pub mod prog_suite;
pub mod prog_suite_development_program;
pub mod prog_suite_subject_bound;
pub mod prog_suite_work_bridge;
pub mod prog_suite_work_evidence;
pub mod renaissance;
pub mod rhythm;
pub mod rhythmic_identity;
pub mod scale;
pub mod score;
pub mod score_validation;
pub mod sonata;
pub mod sonata_work_bridge;
pub mod sonata_work_evidence;
pub mod sonata_work_evidence_coverage;
pub mod spec;
pub mod spelling;
pub mod style;
pub mod temporal_map;
pub mod temporal_midi_projection;
pub mod temporal_score;
pub mod thematic_identity;
pub mod thematic_retrograde_evidence;
pub mod thematic_score_evidence;
pub mod thematic_source_material;
pub mod voicing;
pub mod work_obligation;
pub mod work_plan;

pub use accompaniment::Accompaniment;
pub use cadence::Cadence;
pub use chord::{Chord, ChordQuality};
pub use cognitive_analysis::{
    ScoreCognitiveDelta, ScoreCognitiveProfile, profile_score, profile_score_region,
};
pub use composer::{
    MusicalIntent, compose, compose_sonata_with_plan, compose_styled, compose_with_spec,
    compose_with_spec_and_form,
};
pub use counterpoint::{has_parallel_perfect, parallel_perfect_violations};
pub use development_program::{
    DEVELOPMENT_PROGRAM_VERSION, DevelopmentEvidenceRequirementV1, DevelopmentGraphProjectionV1,
    DevelopmentOperationV1, DevelopmentProgramErrorV1, DevelopmentProgramV1, DevelopmentStageV1,
};
pub use form::{Form, Section, SectionRole};
pub use grammar::{
    GrammarEngine, GrammarFamily, GrammarPlanEvidence, GrammarProfile, GrammarRealization,
    HarmonicGrammarEngine, HarmonicSyntax, IntentAxis, PerformanceDialect, PhraseGrammar,
    PhraseGrammarEngine, compose_with_grammar_plan,
};
pub use grammar_trace::{AssertedObligationState, GrammarStructuralTrace, ScoreEventRef};
pub use harmony::{HarmonicFunction, Key, Progression, Tonality};
pub use hook::{HookCell, graft_hook};
pub use integration::{MusicalPhi, musical_phi};
pub use live::LiveComposer;
pub use melodic_contour_identity::{
    MelodicContourIdentityReport, SectionedContourReport, melodic_contour_report,
};
pub use meter::{MeterError, TimeSignature};
pub use motif::{Contour, Motif, MotifNote};
pub use motif_foundry::{
    FoundryConfig, FoundryDiversityReport, canonical_fingerprint, config_for_dna,
    foundry_diversity_report, generate_candidate, generate_family, generate_with_foundry,
    is_valid_candidate,
};
pub use motif_return::{
    MOTIF_RETURN_MEASUREMENT_VERSION, MotifReturnEvidence, compare_melodic_regions,
    compare_melodic_sequences, melodic_notes_in_region,
};
pub use obligation::{
    CompositionalObligation, ObligationKind, ObligationLedger, ObligationPressure,
    ObligationStatus, ReturnTransformation,
};
pub use phrase::{Period, Phrase};
pub use pitch::{Interval, IntervalQuality, Pitch, PitchClass};
pub use prog_suite::{
    PROG_SUITE_BARS_PER_SECTION, PROG_SUITE_PLAN_VERSION, ProgSuitePlanErrorV1,
    ProgSuitePlanV1, ProgSuiteRealizationV1, ProgSuiteSectionPlanV1, ProgSuiteTransformV1,
    plan_prog_suite, realize_prog_suite_with_plan,
};
pub use prog_suite_development_program::{
    PROG_SUITE_DEVELOPMENT_PROGRAM_VERSION, ProgSuiteDevelopmentProgramErrorV1,
    ProgSuiteDevelopmentProgramV1, derive_prog_suite_development_program,
};
pub use prog_suite_subject_bound::{
    PROG_SUITE_SUBJECT_BOUND_PLAN_VERSION, PROG_SUITE_SUBJECT_BOUND_REALIZATION_VERSION,
    ProgSuiteSubjectBoundErrorV1, ProgSuiteSubjectBoundPlanV1,
    ProgSuiteSubjectBoundRealizationV1, bind_prog_suite_subject,
    realize_prog_suite_subject_bound,
};
pub use prog_suite_work_bridge::{
    PROG_SUITE_WORK_BRIDGE_VERSION, ProgSuiteMeterProjectionV1,
    ProgSuiteSectionWorkBindingV1, ProgSuiteWorkBindingV1, ProgSuiteWorkBridgeErrorV1,
    ProgSuiteWorkRealizationBindingV1, bind_prog_suite_realization, bridge_prog_suite_plan,
};
pub use prog_suite_work_evidence::{
    PROG_SUITE_WORK_EVIDENCE_VERSION, TONIC_ANCHOR_MIN_DURATION_SHARE,
    ProgSuiteTonalCenterProjectionV1, ProgSuiteTonicAnchorEvidenceV1,
    ProgSuiteWorkEvidenceErrorV1, ProgSuiteWorkEvidenceRecordV1, ProgSuiteWorkEvidenceSourceV1,
    ProgSuiteWorkEvidenceV1, derive_prog_suite_work_evidence,
};
pub use rhythm::Duration;
pub use rhythmic_identity::{RhythmicIdentityReport, rhythmic_identity_report};
pub use scale::{Mode, Scale};
pub use score::{Emphasis, PartId, Score, ScoreNote, VoiceRole};
pub use score_validation::{
    ScoreValidationConfig, ScoreValidationIssue, ScoreValidationRule, THEORY_VALIDATION_VERSION,
    TheoryValidationReport, ValidationSeverity, validate_score,
};
pub use sonata::{
    PlannedSonataSection, SonataObligationEvidence, SonataPlan, SonataRealization,
    SonataSectionKind, SonataVerificationMetric, plan_sonata, realize_sonata_with_plan,
    verify_sonata_obligations,
};
pub use sonata_work_bridge::{
    SONATA_WORK_BRIDGE_VERSION, SonataSectionWorkBindingV1, SonataWorkBindingV1,
    SonataWorkBridgeErrorV1, bridge_native_sonata_plan, validate_native_sonata_plan,
};
pub use sonata_work_evidence::{
    NativeEvidencePreservationV1, SONATA_WORK_EVIDENCE_VERSION,
    SonataEvidenceCacheReceiptV1, SonataEvidenceProjectionLossV1, SonataWorkEvidenceErrorV1,
    SonataWorkEvidenceV1, WorkEvidenceStatusV1, WorkObligationEvidenceRecordV1,
    derive_sonata_work_evidence,
};
pub use sonata_work_evidence_coverage::{
    SONATA_WORK_EVIDENCE_COVERAGE_VERSION, SonataWorkEvidenceCoverageErrorV1,
    SonataWorkEvidenceCoverageRecordV1, SonataWorkEvidenceCoverageV1,
    SonataWorkEvidenceSourceV1, derive_sonata_work_evidence_coverage,
};
pub use spec::{Attitude, CompositionSpec, DrumPolicy, FormKind, ProgressionSpec, TextureSpec};
pub use spelling::{Accidental, AlteredDegree, LetterName, SpelledPitchClass};
pub use style::Style;
pub use temporal_map::{
    TEMPORAL_MAP_VERSION, TempoV1, TemporalMapErrorV1, TemporalMapV1, TemporalPointV1,
};
pub use temporal_midi_projection::{
    DEFAULT_MIDI_TICKS_PER_QUARTER, MIDI_TEMPORAL_PROJECTION_VERSION,
    MeterMidiProjectionReceiptV1, MidiTemporalMetaEventV1, MidiTemporalMetaKindV1,
    MidiTemporalProjectionErrorV1, MidiTemporalProjectionPolicyV1, MidiTemporalProjectionV1,
    TempoMidiProjectionReceiptV1, project_temporal_score_to_midi,
};
pub use temporal_score::{TEMPORAL_SCORE_VERSION, TemporalScoreErrorV1, TemporalScoreV1};
pub use thematic_identity::{
    THEMATIC_IDENTITY_GRAPH_VERSION, ThematicDerivationV1, ThematicGraphErrorV1,
    ThematicIdentityGraphV1, ThematicIdentityV1, ThematicOriginV1,
    ThematicTransformationClassV1,
};
pub use thematic_retrograde_evidence::{
    THEMATIC_RETROGRADE_EVIDENCE_VERSION, ThematicRetrogradeEvidenceSetV1,
    ThematicRetrogradeEvidenceV1, ThematicRetrogradeStatusV1,
    measure_thematic_retrograde_evidence,
};
pub use thematic_score_evidence::{
    MIN_THEMATIC_MELODY_NOTES, THEMATIC_SCORE_EVIDENCE_VERSION,
    ThematicDerivationEvidenceStatusV1, ThematicDerivationObservationV1,
    ThematicIdentityObservationStatusV1, ThematicIdentityObservationV1,
    ThematicMelodyFingerprintV1, ThematicScoreEvidenceErrorV1, ThematicScoreEvidenceV1,
    ThematicTransformationMeasurementV1, measure_thematic_graph,
};
pub use thematic_source_material::{
    THEMATIC_SOURCE_MATERIAL_PLAN_VERSION, ThematicSourceMaterialErrorV1,
    ThematicSourceMaterialPlanV1, ThematicSourceMaterialV1,
};
pub use voicing::{lead_bass, lead_upper};
pub use work_obligation::{
    ObligationDueWindowV2, WORK_OBLIGATION_PLAN_VERSION, WorkObligationErrorV2,
    WorkObligationKindV2, WorkObligationPlanV2, WorkObligationV2,
};
pub use work_plan::{
    FormalFunctionV1, HIERARCHICAL_WORK_PLAN_VERSION, HierarchicalWorkPlanV1,
    WorkNodeKindV1, WorkNodeV1, WorkPlanErrorV1,
};
