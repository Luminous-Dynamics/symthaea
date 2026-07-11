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

pub mod accompaniment;
pub mod cadence;
pub mod chord;
pub mod composer;
pub mod counterpoint;
pub mod form;
pub mod fugue;
pub mod harmony;
pub mod hook;
pub mod integration;
pub mod live;
pub mod motif;
pub mod phrase;
pub mod pitch;
pub mod rhythm;
pub mod scale;
pub mod score;
pub mod spec;
pub mod style;
pub mod voicing;

pub use accompaniment::Accompaniment;
pub use cadence::Cadence;
pub use chord::{Chord, ChordQuality};
pub use composer::{MusicalIntent, compose, compose_styled, compose_with_spec};
pub use counterpoint::{has_parallel_perfect, parallel_perfect_violations};
pub use form::{Form, Section, SectionRole};
pub use harmony::{HarmonicFunction, Key, Progression, Tonality};
pub use hook::{HookCell, graft_hook};
pub use integration::{MusicalPhi, musical_phi};
pub use live::LiveComposer;
pub use motif::{Contour, Motif, MotifNote};
pub use phrase::{Period, Phrase};
pub use pitch::{Interval, IntervalQuality, Pitch, PitchClass};
pub use rhythm::Duration;
pub use scale::{Mode, Scale};
pub use score::{Emphasis, Score, ScoreNote, VoiceRole};
pub use spec::{Attitude, CompositionSpec, DrumPolicy, FormKind, ProgressionSpec, TextureSpec};
pub use style::Style;
pub use voicing::{lead_bass, lead_upper};
