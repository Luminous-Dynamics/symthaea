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
pub mod diversity_plan;
pub mod evidence_calibration;
pub mod explorer;
pub mod fingerprint;
pub mod form;
pub mod foundry;
pub mod fugue;
pub mod grammar;
pub mod grammar_trace;
pub mod groove_cycle;
pub mod harmony;
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
pub mod renaissance;
pub mod rhythm;
pub mod rhythmic_identity;
pub mod scale;
pub mod score;
pub mod score_validation;
pub mod sonata;
pub mod spec;
pub mod spelling;
pub mod style;
pub mod voicing;

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
pub use rhythm::Duration;
pub use rhythmic_identity::{RhythmicIdentityReport, rhythmic_identity_report};
pub use scale::{Mode, Scale};
pub use score::{Emphasis, Score, ScoreNote, VoiceRole};
pub use score_validation::{
    ScoreValidationConfig, ScoreValidationIssue, ScoreValidationRule, THEORY_VALIDATION_VERSION,
    TheoryValidationReport, ValidationSeverity, validate_score,
};
pub use sonata::{
    PlannedSonataSection, SonataObligationEvidence, SonataPlan, SonataRealization,
    SonataSectionKind, SonataVerificationMetric, plan_sonata, realize_sonata_with_plan,
    verify_sonata_obligations,
};
pub use spec::{Attitude, CompositionSpec, DrumPolicy, FormKind, ProgressionSpec, TextureSpec};
pub use spelling::{Accidental, AlteredDegree, LetterName, SpelledPitchClass};
pub use style::Style;
pub use voicing::{lead_bass, lead_upper};
