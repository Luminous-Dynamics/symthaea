// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Genre/style (Layer 5): named collections of STRUCTURAL biases — meter,
//! tempo range, motif shape bank, and progression — expressed entirely
//! through the primitives [`crate::composer::compose`] already uses.
//!
//! This is deliberately **not** a new harmonic system: [`crate::harmony::Key`]
//! and functional harmony (Tonic/Predominant/Dominant, diatonic triads,
//! cadences) are completely untouched by `Style`. A style only changes WHICH
//! motif shapes and WHICH chord progression get picked for a given
//! arousal/seed — never how a scale or a chord is built. That keeps the
//! ground-truth harmony machinery (and every existing test built on it)
//! exactly as trustworthy under any style as it is under
//! [`Style::Classical`].
//!
//! `Style::Classical` is a pure passthrough to the functions `compose()` has
//! always called (`crate::composer::classical_motif_bank`,
//! `Progression::generate`), so `compose_styled(intent, Style::Classical)`
//! is byte-identical to `compose(intent)` — adding style is additive, not a
//! behavior change to any existing caller.

use crate::harmony::Progression;
use crate::motif::Motif;
use crate::rhythm::Duration;
use serde::{Deserialize, Serialize};

/// A named genre/style bias. See the module docs for what this does and
/// does not change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Style {
    /// The original, unbiased defaults: 4/4, 60-150 BPM, the hand-picked
    /// "classical" motif bank, and the functional-grammar `Progression::generate`.
    #[default]
    Classical,
    /// 3/4, singing legato motifs, a Pachelbel-flavored cyclical progression
    /// — the "round and round" feel of a waltz.
    Waltz,
    /// 4/4, stepwise/pentatonic-flavored melodic shapes (avoids scale degrees
    /// 4 and 7 — the notes a major pentatonic scale omits) over a simple,
    /// repeating four-chord loop.
    Folk,
    /// 4/4, wide melodic leaps and sustained notes, a slower tempo ceiling,
    /// a minor-flavored repeating progression — sweeping and dramatic.
    Cinematic,
    /// 4/4, syncopated eighth-note motifs, a brisk tempo floor, the
    /// ubiquitous pop four-chord loop — bright and bouncy.
    Playful,
    /// Slow 4/4, long singing lines over a rolling arpeggio left hand,
    /// I–vi–ii–V cycles — the night piece. Added once the engine had the
    /// vocabulary a nocturne actually needs (suspensions, held arrivals).
    Nocturne,
    /// Square 4/4 with dotted figures, an authentic I–IV–V–I frame and a
    /// steady pulse — deliberately UN-syncopated (a march is insistence
    /// through regularity, not against it).
    March,
    /// Gentle 3/4 rocking, tiny range, a soft plucked palette at cradle
    /// tempo.
    Lullaby,
    /// Dorian by default with the ♭VII in the progression — the modal
    /// wanderer, productized from the "Dorian Wander" listening sessions.
    ModalFolk,
    /// A three-voice fughetta (see [`crate::fugue`]): the subject stated by
    /// every voice — exposition, episodes, inversion, stretto, augmented
    /// final entry. The one style whose form IS its texture: its pool holds
    /// only [`crate::spec::FormKind::Fugue`].
    Fugue,
}

impl Style {
    /// Beats per measure. Only [`Style::Waltz`] differs from 4/4.
    pub fn meter(self) -> u8 {
        match self {
            Style::Waltz | Style::Lullaby => 3,
            _ => 4,
        }
    }

    /// Tempo at arousal=0 and arousal=1, in BPM.
    pub fn tempo_range(self) -> (f32, f32) {
        match self {
            Style::Classical => (60.0, 150.0),
            Style::Waltz => (100.0, 180.0),
            Style::Folk => (80.0, 130.0),
            Style::Cinematic => (50.0, 110.0),
            Style::Playful => (110.0, 170.0),
            Style::Nocturne => (45.0, 80.0),
            Style::March => (100.0, 140.0),
            Style::Lullaby => (50.0, 76.0),
            Style::ModalFolk => (70.0, 110.0),
            Style::Fugue => (66.0, 120.0),
        }
    }

    /// Tempo for a given arousal (0..1, clamped), linearly interpolated
    /// across [`Self::tempo_range`].
    pub fn tempo(self, arousal: f32) -> f32 {
        let (lo, hi) = self.tempo_range();
        lo + arousal.clamp(0.0, 1.0) * (hi - lo)
    }

    /// Pick a motif shape from this style's own bank, by arousal-tier (same
    /// three tiers as `compose()` always used: calm below 0.33, medium below
    /// 0.66, busy above) and seed. Every template in every style's bank
    /// totals EXACTLY `self.meter()` beats — see the per-style bank tests.
    pub fn motif(self, arousal: f32, seed: u64) -> Motif {
        match self {
            Style::Classical => crate::composer::classical_motif_bank(arousal, seed),
            Style::Waltz => waltz_motif(arousal, seed),
            Style::Folk => folk_motif(arousal, seed),
            Style::Cinematic => cinematic_motif(arousal, seed),
            Style::Playful => playful_motif(arousal, seed),
            // The newer styles are spec-FIRST: their banks live in the
            // preset spec, and these accessors delegate — parity between
            // style and spec by construction instead of by test.
            _ => self.spec().motif(arousal, seed),
        }
    }

    /// The accompaniment pattern for this style, varied by seed. Waltz is
    /// ALWAYS oom-pah-pah (that rhythm IS the genre); Cinematic is always the
    /// sustained block pad. The others pick from a small idiomatic pool.
    /// Seed 0 maps Classical to `Block` — the pre-pattern texture — so every
    /// existing default-intent test and render stays on the texture it was
    /// written against; variety comes from nonzero seeds, exactly like the
    /// motif-bank and form choices. `seed / 2` decorrelates the pick from the
    /// ternary-vs-rondo form choice (`seed % 2`).
    pub fn accompaniment(self, seed: u64) -> crate::accompaniment::Accompaniment {
        use crate::accompaniment::Accompaniment as A;
        match self {
            Style::Classical => [A::Block, A::Alberti, A::Arpeggio][(seed as usize / 2) % 3],
            Style::Waltz => A::OomPah,
            Style::Folk => [A::Arpeggio, A::Block][(seed as usize / 2) % 2],
            Style::Cinematic => A::Block,
            Style::Playful => [A::Comp, A::Alberti][(seed as usize / 2) % 2],
            _ => self.spec().accompaniment(seed),
        }
    }

    /// A chord progression `bars` measures long. `Style::Classical` uses the
    /// same functional-grammar generator `compose()` always used
    /// ([`Progression::generate`]); the other styles cycle a single named
    /// archetype (see [`Progression`]'s archetype constructors) to exactly
    /// `bars` measures, so the length contract callers rely on
    /// (`progression.len() == bars.max(1)`) holds for every style.
    pub fn progression(self, bars: usize, seed: u64) -> Progression {
        let bars = bars.max(1);
        match self {
            Style::Classical => Progression::generate(bars, seed),
            Style::Waltz => cycle_to_length(&Progression::pachelbel().degrees, bars),
            Style::Folk => cycle_to_length(&Progression::four_chord().degrees, bars),
            Style::Cinematic => cycle_to_length(&Progression::sensitive_female().degrees, bars),
            Style::Playful => cycle_to_length(&Progression::pop_turnaround().degrees, bars),
            _ => self.spec().progression(bars, seed),
        }
    }

    /// This style as a [`CompositionSpec`] — the built-in styles are nothing
    /// more than five preset values of the open, user-editable spec type.
    /// Transcription fidelity is tested (`spec_motifs_match_the_style_banks
    /// _they_transcribe` and the compose-identity test in `composer.rs`).
    pub fn spec(self) -> crate::spec::CompositionSpec {
        use crate::accompaniment::Accompaniment as A;
        use crate::spec::*;
        let s = |v: &[[&str; 3]]| -> Vec<[String; 3]> {
            v.iter()
                .map(|t| [t[0].into(), t[1].into(), t[2].into()])
                .collect()
        };
        // Durations as (num, den) beats: q = (1,1), h = (2,1), e = (1,2).
        let texture = |drums: DrumPolicy| TextureSpec {
            thin_departure: true,
            counter_melody: true,
            climax_grace: true,
            cadential_harmonic_rhythm: true,
            coda_bars: crate::composer::CODA_BARS as u8,
            drums,
            swing: 0.5, // presets stay straight; swing is the USER's dial
            intro_bars: 2,
            staged_entrances: true,
            held_arrivals: true,
            damage: 0.5,
            hook_cell: true,
            return_color: true,
        };
        match self {
            Style::Classical => CompositionSpec {
                name: "Classical".into(),
                attitude: None,
                meter: 4,
                tempo_range: (60.0, 150.0),
                motifs_calm: vec![
                    vec![(1, 2, 1), (2, 1, 1), (3, 1, 1)],
                    vec![(5, 2, 1), (3, 1, 1), (1, 1, 1)],
                    vec![(3, 1, 1), (2, 1, 1), (1, 2, 1)],
                ],
                motifs_medium: vec![
                    vec![(1, 1, 1), (2, 1, 1), (3, 1, 1), (2, 1, 1)],
                    vec![(1, 1, 1), (3, 1, 1), (5, 1, 1), (3, 1, 1)],
                    vec![(5, 1, 1), (4, 1, 1), (3, 1, 1), (1, 1, 1)],
                ],
                motifs_busy: vec![
                    vec![(1, 1, 2), (2, 1, 2), (3, 1, 2), (4, 1, 2), (5, 2, 1)],
                    vec![(1, 1, 2), (3, 1, 2), (5, 1, 2), (3, 1, 2), (2, 2, 1)],
                    vec![(5, 1, 2), (4, 1, 2), (3, 1, 2), (2, 1, 2), (1, 2, 1)],
                ],
                progression: ProgressionSpec::Grammar,
                accompaniment_pool: vec![A::Block, A::Alberti, A::Arpeggio],
                form_pool: vec![FormKind::Ternary, FormKind::Rondo],
                ensemble_pool: s(&[
                    ["violin", "piano", "cello"],
                    ["flute", "piano", "cello"],
                    ["violin", "harp", "cello"],
                ]),
                mode: None,
                counter_instrument: None,
                texture: texture(DrumPolicy::None),
            },
            Style::Waltz => CompositionSpec {
                name: "Waltz".into(),
                attitude: None,
                meter: 3,
                tempo_range: (100.0, 180.0),
                motifs_calm: vec![
                    vec![(1, 2, 1), (2, 1, 1)],
                    vec![(3, 1, 1), (2, 1, 1), (1, 1, 1)],
                ],
                motifs_medium: vec![
                    vec![(1, 1, 1), (3, 1, 1), (5, 1, 1)],
                    vec![(5, 1, 1), (3, 1, 1), (1, 1, 1)],
                ],
                motifs_busy: vec![
                    vec![(1, 1, 2), (2, 1, 2), (3, 1, 2), (2, 1, 2), (1, 1, 1)],
                    vec![(5, 1, 2), (4, 1, 2), (3, 1, 2), (4, 1, 2), (5, 1, 1)],
                ],
                progression: ProgressionSpec::Archetype(vec![1, 5, 6, 3, 4, 1, 4, 5]),
                accompaniment_pool: vec![A::OomPah],
                form_pool: vec![FormKind::Ternary, FormKind::Rondo],
                ensemble_pool: s(&[
                    ["violin", "piano", "cello"],
                    ["flute", "piano", "cello"],
                    ["violin", "harp", "cello"],
                ]),
                mode: None,
                counter_instrument: None,
                texture: texture(DrumPolicy::None),
            },
            Style::Folk => CompositionSpec {
                name: "Folk".into(),
                attitude: None,
                meter: 4,
                tempo_range: (80.0, 130.0),
                motifs_calm: vec![
                    vec![(1, 2, 1), (2, 1, 1), (1, 1, 1)],
                    vec![(5, 2, 1), (3, 1, 1), (2, 1, 1)],
                ],
                motifs_medium: vec![
                    vec![(1, 1, 1), (2, 1, 1), (3, 1, 1), (2, 1, 1)],
                    vec![(2, 1, 1), (1, 1, 1), (2, 1, 1), (3, 1, 1)],
                ],
                motifs_busy: vec![
                    vec![(1, 1, 2), (2, 1, 2), (3, 1, 2), (2, 1, 2), (1, 2, 1)],
                    vec![(5, 1, 2), (3, 1, 2), (2, 1, 2), (1, 1, 2), (2, 2, 1)],
                ],
                progression: ProgressionSpec::Archetype(vec![1, 5, 6, 4]),
                accompaniment_pool: vec![A::Arpeggio, A::Block],
                form_pool: vec![FormKind::Ternary, FormKind::Rondo],
                ensemble_pool: s(&[
                    ["flute", "acoustic_guitar", "upright_bass"],
                    ["clarinet", "harp", "upright_bass"],
                    ["flute", "koto", "upright_bass"],
                ]),
                mode: None,
                counter_instrument: None,
                texture: texture(DrumPolicy::LightPulse),
            },
            Style::Cinematic => CompositionSpec {
                name: "Cinematic".into(),
                attitude: None,
                meter: 4,
                tempo_range: (50.0, 110.0),
                motifs_calm: vec![vec![(1, 2, 1), (5, 2, 1)], vec![(5, 2, 1), (1, 2, 1)]],
                motifs_medium: vec![
                    vec![(1, 1, 1), (5, 1, 1), (3, 1, 1), (6, 1, 1)],
                    vec![(3, 1, 1), (6, 1, 1), (2, 1, 1), (5, 1, 1)],
                ],
                motifs_busy: vec![
                    vec![(1, 1, 2), (5, 1, 2), (3, 1, 2), (6, 1, 2), (1, 2, 1)],
                    vec![(6, 1, 2), (3, 1, 2), (5, 1, 2), (1, 1, 2), (5, 2, 1)],
                ],
                progression: ProgressionSpec::Archetype(vec![6, 4, 1, 5]),
                accompaniment_pool: vec![A::Block],
                form_pool: vec![FormKind::Ternary, FormKind::Rondo],
                ensemble_pool: s(&[
                    ["violin", "organ", "cello"],
                    ["trumpet", "organ", "cello"],
                    ["violin", "pad", "cello"],
                ]),
                mode: None,
                counter_instrument: None,
                texture: texture(DrumPolicy::BarPulse),
            },
            Style::Playful => CompositionSpec {
                name: "Playful".into(),
                attitude: None,
                meter: 4,
                tempo_range: (110.0, 170.0),
                motifs_calm: vec![
                    vec![(1, 1, 2), (2, 1, 2), (1, 1, 1), (3, 2, 1)],
                    vec![(3, 1, 2), (2, 1, 2), (3, 1, 1), (5, 2, 1)],
                ],
                motifs_medium: vec![
                    vec![
                        (1, 1, 2),
                        (1, 1, 2),
                        (3, 1, 2),
                        (3, 1, 2),
                        (5, 1, 1),
                        (3, 1, 1),
                    ],
                    vec![
                        (5, 1, 2),
                        (5, 1, 2),
                        (3, 1, 2),
                        (3, 1, 2),
                        (1, 1, 1),
                        (3, 1, 1),
                    ],
                ],
                motifs_busy: vec![
                    vec![
                        (1, 1, 2),
                        (2, 1, 2),
                        (3, 1, 2),
                        (4, 1, 2),
                        (5, 1, 2),
                        (4, 1, 2),
                        (3, 1, 2),
                        (2, 1, 2),
                    ],
                    vec![
                        (5, 1, 2),
                        (4, 1, 2),
                        (3, 1, 2),
                        (2, 1, 2),
                        (1, 1, 2),
                        (2, 1, 2),
                        (3, 1, 2),
                        (4, 1, 2),
                    ],
                ],
                progression: ProgressionSpec::Archetype(vec![1, 6, 4, 5]),
                accompaniment_pool: vec![A::Comp, A::Alberti],
                form_pool: vec![FormKind::Ternary, FormKind::Rondo],
                ensemble_pool: s(&[
                    ["clarinet", "electric_piano", "acoustic_guitar"],
                    ["saxophone", "marimba", "acoustic_guitar"],
                    ["trumpet", "kalimba", "acoustic_guitar"],
                ]),
                mode: None,
                counter_instrument: None,
                texture: texture(DrumPolicy::Backbeat),
            },
            Style::Nocturne => CompositionSpec {
                name: "Nocturne".into(),
                attitude: None,
                mode: None,
                meter: 4,
                tempo_range: (45.0, 80.0),
                motifs_calm: vec![
                    vec![(3, 2, 1), (2, 1, 1), (1, 1, 1)],
                    vec![(5, 2, 1), (4, 1, 1), (3, 1, 1)],
                    vec![(1, 2, 1), (3, 1, 1), (5, 1, 1)],
                ],
                motifs_medium: vec![
                    vec![(3, 1, 1), (4, 1, 1), (5, 1, 1), (4, 1, 1)],
                    vec![(5, 1, 1), (6, 1, 1), (5, 1, 1), (3, 1, 1)],
                ],
                motifs_busy: vec![
                    vec![(3, 1, 2), (4, 1, 2), (5, 1, 1), (6, 1, 1), (5, 1, 1)],
                    vec![(5, 1, 2), (4, 1, 2), (3, 1, 2), (2, 1, 2), (1, 2, 1)],
                ],
                progression: ProgressionSpec::Archetype(vec![1, 6, 2, 5]),
                accompaniment_pool: vec![A::Arpeggio],
                // Variations belongs here: a nocturne dwelling on one idea,
                // darkening it (minore) and ornamenting it (figuration), is
                // the Chopin/Field association exactly.
                form_pool: vec![FormKind::Ternary, FormKind::Rondo, FormKind::Variations],
                ensemble_pool: s(&[["clarinet", "piano", "cello"], ["flute", "piano", "cello"]]),
                counter_instrument: None,
                texture: texture(DrumPolicy::None),
            },
            Style::March => CompositionSpec {
                name: "March".into(),
                attitude: None,
                mode: None,
                meter: 4,
                tempo_range: (100.0, 140.0),
                motifs_calm: vec![
                    vec![(1, 3, 2), (2, 1, 2), (3, 2, 1)],
                    vec![(5, 3, 2), (4, 1, 2), (3, 2, 1)],
                    vec![(1, 2, 1), (5, 1, 1), (1, 1, 1)],
                ],
                motifs_medium: vec![
                    vec![(1, 3, 2), (1, 1, 2), (3, 1, 1), (5, 1, 1)],
                    vec![(5, 3, 2), (5, 1, 2), (4, 1, 1), (3, 1, 1)],
                ],
                motifs_busy: vec![
                    vec![
                        (1, 3, 4),
                        (1, 1, 4),
                        (2, 3, 4),
                        (2, 1, 4),
                        (3, 1, 1),
                        (5, 1, 1),
                    ],
                    vec![
                        (5, 1, 2),
                        (5, 1, 2),
                        (6, 1, 2),
                        (5, 1, 2),
                        (3, 1, 1),
                        (1, 1, 1),
                    ],
                ],
                // I-V-I-IV: the march's tonic-dominant hammering — and
                // deliberately NOT the [1,4,5,1] the Classical grammar
                // generates for seed 0, which made the two styles compose
                // identical lines once the hook dominated the bar.
                progression: ProgressionSpec::Archetype(vec![1, 5, 1, 4]),
                accompaniment_pool: vec![A::Block],
                form_pool: vec![FormKind::Ternary, FormKind::Rondo],
                ensemble_pool: s(&[["trumpet", "organ", "cello"]]),
                counter_instrument: None,
                texture: texture(DrumPolicy::LightPulse),
            },
            Style::Lullaby => CompositionSpec {
                name: "Lullaby".into(),
                attitude: None,
                mode: None,
                meter: 3,
                tempo_range: (50.0, 76.0),
                motifs_calm: vec![
                    vec![(3, 2, 1), (1, 1, 1)],
                    vec![(5, 2, 1), (3, 1, 1)],
                    vec![(1, 2, 1), (2, 1, 1)],
                ],
                motifs_medium: vec![
                    vec![(3, 1, 1), (2, 1, 1), (1, 1, 1)],
                    vec![(5, 1, 1), (4, 1, 1), (3, 1, 1)],
                ],
                motifs_busy: vec![
                    vec![(3, 1, 2), (4, 1, 2), (3, 1, 1), (1, 1, 1)],
                    vec![(5, 1, 2), (4, 1, 2), (3, 1, 2), (2, 1, 2), (1, 1, 1)],
                ],
                progression: ProgressionSpec::Archetype(vec![1, 4, 1, 5]),
                accompaniment_pool: vec![A::Arpeggio],
                // The theme-and-variations lullaby is the oldest association
                // in the repertoire (Twinkle IS Mozart K.265's theme); rondo
                // stays excluded — five sections outlasts a lullaby's job.
                form_pool: vec![FormKind::Ternary, FormKind::Variations],
                ensemble_pool: s(&[["flute", "kalimba", "cello"]]),
                counter_instrument: None,
                texture: texture(DrumPolicy::None),
            },
            Style::ModalFolk => CompositionSpec {
                name: "ModalFolk".into(),
                attitude: None,
                mode: Some(crate::scale::Mode::Dorian),
                meter: 4,
                tempo_range: (70.0, 110.0),
                motifs_calm: vec![
                    vec![(1, 2, 1), (2, 1, 1), (4, 1, 1)],
                    vec![(4, 2, 1), (3, 1, 1), (1, 1, 1)],
                    vec![(1, 1, 1), (7, 1, 1), (1, 2, 1)],
                ],
                motifs_medium: vec![
                    vec![(1, 1, 1), (2, 1, 1), (4, 1, 1), (3, 1, 1)],
                    vec![(4, 1, 1), (3, 1, 1), (2, 1, 1), (1, 1, 1)],
                ],
                motifs_busy: vec![
                    vec![
                        (1, 1, 2),
                        (2, 1, 2),
                        (4, 1, 2),
                        (3, 1, 2),
                        (2, 1, 1),
                        (1, 1, 1),
                    ],
                    vec![
                        (4, 1, 2),
                        (3, 1, 2),
                        (2, 1, 2),
                        (1, 1, 2),
                        (7, 1, 1),
                        (1, 1, 1),
                    ],
                ],
                progression: ProgressionSpec::Archetype(vec![1, 7, 4, 1]),
                accompaniment_pool: vec![A::Arpeggio, A::Block],
                form_pool: vec![FormKind::Ternary, FormKind::Rondo],
                ensemble_pool: s(&[["flute", "marimba", "upright_bass"]]),
                counter_instrument: None,
                texture: texture(DrumPolicy::None),
            },
            Style::Fugue => CompositionSpec {
                name: "Fugue".into(),
                attitude: None,
                mode: None,
                meter: 4,
                tempo_range: (66.0, 120.0),
                // Subject banks: strong stepwise identities with one leap
                // and a long final tone — heads that survive fragmentation
                // (episodes sequence the first half-bar) and read clearly
                // in augmentation.
                motifs_calm: vec![
                    vec![(1, 1, 1), (3, 1, 2), (2, 1, 2), (5, 2, 1)],
                    vec![(5, 1, 1), (4, 1, 2), (3, 1, 2), (1, 2, 1)],
                    vec![(1, 2, 1), (2, 1, 1), (3, 1, 1)],
                ],
                motifs_medium: vec![
                    vec![(1, 1, 2), (2, 1, 2), (3, 1, 1), (5, 1, 1), (4, 1, 1)],
                    vec![(3, 1, 1), (1, 1, 2), (2, 1, 2), (4, 1, 1), (5, 1, 1)],
                ],
                motifs_busy: vec![
                    vec![
                        (1, 1, 2),
                        (3, 1, 2),
                        (2, 1, 2),
                        (4, 1, 2),
                        (3, 1, 2),
                        (5, 1, 2),
                        (4, 1, 2),
                        (2, 1, 2),
                    ],
                    vec![
                        (5, 1, 2),
                        (3, 1, 2),
                        (4, 1, 2),
                        (2, 1, 2),
                        (3, 1, 2),
                        (1, 1, 2),
                        (2, 1, 1),
                    ],
                ],
                // Unused by the fugue path (its texture is its counterpoint)
                // but required for a valid spec — and it keeps this preset
                // usable if a caller ever swaps the form_pool.
                progression: ProgressionSpec::Archetype(vec![1, 4, 5, 1]),
                accompaniment_pool: vec![A::Block],
                form_pool: vec![FormKind::Fugue],
                ensemble_pool: s(&[["organ", "piano", "cello"]]),
                counter_instrument: None,
                texture: texture(DrumPolicy::None),
            },
        }
    }
}

fn pick(bank: &[&[(i32, Duration)]], seed: u64) -> Motif {
    let idx = (seed % bank.len() as u64) as usize;
    let picked = Motif::from_degrees(bank[idx]);
    // Multiply the bank's effective variety with an independent seed-slice
    // orientation, same technique as `composer::classical_motif_bank`.
    let orientation = (seed / bank.len() as u64) % 4;
    crate::form::oriented(&picked, orientation)
}

/// Cycle `pattern` to exactly `bars` degrees (repeating from the start once
/// exhausted), so a fixed-length archetype can back a progression of any
/// requested length.
fn cycle_to_length(pattern: &[i32], bars: usize) -> Progression {
    let degrees: Vec<i32> = (0..bars).map(|i| pattern[i % pattern.len()]).collect();
    Progression::new(degrees)
}

fn waltz_motif(arousal: f32, seed: u64) -> Motif {
    let q = Duration::quarter();
    let h = Duration::half();
    let e = Duration::eighth();

    let calm: [&[(i32, Duration)]; 2] = [
        &[(1, h), (2, q)],         // sum = 3: a singing lean into beat 2
        &[(3, q), (2, q), (1, q)], // sum = 3: gentle stepwise descent
    ];
    let medium: [&[(i32, Duration)]; 2] = [
        &[(1, q), (3, q), (5, q)], // sum = 3: rising arpeggio
        &[(5, q), (3, q), (1, q)], // sum = 3: falling arpeggio
    ];
    let busy: [&[(i32, Duration)]; 2] = [
        &[(1, e), (2, e), (3, e), (2, e), (1, q)], // sum = 3: a lively turn
        &[(5, e), (4, e), (3, e), (4, e), (5, q)], // sum = 3: a lively neighbor figure
    ];

    let bank: &[&[(i32, Duration)]] = if arousal < 0.33 {
        &calm
    } else if arousal < 0.66 {
        &medium
    } else {
        &busy
    };
    pick(bank, seed)
}

// Folk melodies avoid scale degrees 4 and 7 (the notes a major pentatonic
// scale omits) for that open, modal folk-song color, while harmony stays
// full diatonic underneath (a deliberate, common technique — see the module
// doc's "melodic color over unchanged harmony" note).
fn folk_motif(arousal: f32, seed: u64) -> Motif {
    let q = Duration::quarter();
    let h = Duration::half();
    let e = Duration::eighth();

    let calm: [&[(i32, Duration)]; 2] = [
        &[(1, h), (2, q), (1, q)], // sum = 4
        &[(5, h), (3, q), (2, q)], // sum = 4
    ];
    let medium: [&[(i32, Duration)]; 2] = [
        &[(1, q), (2, q), (3, q), (2, q)], // sum = 4
        &[(2, q), (1, q), (2, q), (3, q)], // sum = 4
    ];
    let busy: [&[(i32, Duration)]; 2] = [
        &[(1, e), (2, e), (3, e), (2, e), (1, h)], // sum = 4
        &[(5, e), (3, e), (2, e), (1, e), (2, h)], // sum = 4
    ];

    let bank: &[&[(i32, Duration)]] = if arousal < 0.33 {
        &calm
    } else if arousal < 0.66 {
        &medium
    } else {
        &busy
    };
    pick(bank, seed)
}

fn cinematic_motif(arousal: f32, seed: u64) -> Motif {
    let q = Duration::quarter();
    let h = Duration::half();
    let e = Duration::eighth();

    let calm: [&[(i32, Duration)]; 2] = [
        &[(1, h), (5, h)], // sum = 4: a wide, sustained leap up
        &[(5, h), (1, h)], // sum = 4: a wide, sustained leap down
    ];
    let medium: [&[(i32, Duration)]; 2] = [
        &[(1, q), (5, q), (3, q), (6, q)], // sum = 4: dramatic wide leaps
        &[(3, q), (6, q), (2, q), (5, q)], // sum = 4: dramatic wide leaps
    ];
    let busy: [&[(i32, Duration)]; 2] = [
        &[(1, e), (5, e), (3, e), (6, e), (1, h)], // sum = 4
        &[(6, e), (3, e), (5, e), (1, e), (5, h)], // sum = 4
    ];

    let bank: &[&[(i32, Duration)]] = if arousal < 0.33 {
        &calm
    } else if arousal < 0.66 {
        &medium
    } else {
        &busy
    };
    pick(bank, seed)
}

fn playful_motif(arousal: f32, seed: u64) -> Motif {
    let q = Duration::quarter();
    let h = Duration::half();
    let e = Duration::eighth();

    let calm: [&[(i32, Duration)]; 2] = [
        &[(1, e), (2, e), (1, q), (3, h)], // sum = 4: a light skip
        &[(3, e), (2, e), (3, q), (5, h)], // sum = 4: a light skip
    ];
    let medium: [&[(i32, Duration)]; 2] = [
        &[(1, e), (1, e), (3, e), (3, e), (5, q), (3, q)], // sum = 4: bouncy repeated notes
        &[(5, e), (5, e), (3, e), (3, e), (1, q), (3, q)], // sum = 4: bouncy repeated notes
    ];
    let busy: [&[(i32, Duration)]; 2] = [
        &[
            (1, e),
            (2, e),
            (3, e),
            (4, e),
            (5, e),
            (4, e),
            (3, e),
            (2, e),
        ], // sum = 4: bouncy run up and back
        &[
            (5, e),
            (4, e),
            (3, e),
            (2, e),
            (1, e),
            (2, e),
            (3, e),
            (4, e),
        ], // sum = 4: bouncy run down and back
    ];

    let bank: &[&[(i32, Duration)]] = if arousal < 0.33 {
        &calm
    } else if arousal < 0.66 {
        &medium
    } else {
        &busy
    };
    pick(bank, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_STYLES: [Style; 5] = [
        Style::Classical,
        Style::Waltz,
        Style::Folk,
        Style::Cinematic,
        Style::Playful,
    ];

    #[test]
    fn classical_is_the_default() {
        assert_eq!(Style::default(), Style::Classical);
    }

    #[test]
    fn accompaniment_mapping_holds_its_contracts() {
        use crate::accompaniment::Accompaniment as A;
        // Genre-defining constants.
        for seed in 0..8 {
            assert_eq!(Style::Waltz.accompaniment(seed), A::OomPah);
            assert_eq!(Style::Cinematic.accompaniment(seed), A::Block);
        }
        // Classical seed 0 stays on the pre-pattern texture (compat).
        assert_eq!(Style::Classical.accompaniment(0), A::Block);
        // And nonzero seeds genuinely vary the texture.
        let classical: std::collections::HashSet<_> =
            (0..8).map(|s| Style::Classical.accompaniment(s)).collect();
        assert!(classical.len() > 1, "classical must vary: {classical:?}");
    }

    #[test]
    fn new_styles_validate_compose_and_stay_distinct() {
        use crate::composer::{MusicalIntent, compose_with_spec};
        // Every new style's preset must validate, compose a full piece in
        // its own meter/tempo band, and differ audibly from Classical.
        let classical = compose_with_spec(&MusicalIntent::default(), &Style::Classical.spec());
        for style in [
            Style::Nocturne,
            Style::March,
            Style::Lullaby,
            Style::ModalFolk,
            Style::Fugue,
        ] {
            let spec = style.spec();
            spec.validate()
                .unwrap_or_else(|e| panic!("{style:?} preset invalid: {e:?}"));
            let score = compose_with_spec(&MusicalIntent::default(), &spec);
            assert!(!score.notes.is_empty(), "{style:?} composes");
            assert_eq!(score.meter, spec.meter, "{style:?} keeps its meter");
            let (lo, hi) = spec.tempo_range;
            assert!(
                score.tempo_bpm >= lo * 0.8 && score.tempo_bpm <= hi * 1.1,
                "{style:?} tempo {} outside its band ({lo}..{hi}, attitude slack allowed)",
                score.tempo_bpm
            );
            // Distinct from the Classical default: different notes.
            let sig = |s: &crate::score::Score| -> Vec<(u8, i64)> {
                s.notes
                    .iter()
                    .take(24)
                    .map(|n| (n.pitch.midi(), (n.onset.beats() * 4.0) as i64))
                    .collect()
            };
            assert_ne!(
                sig(&score),
                sig(&classical),
                "{style:?} must not be Classical"
            );
        }
        // ModalFolk really is Dorian: zero raised-seventh leading tones in
        // a D-tonic piece composed with pristine texture (no wait-tone).
        let mut dorian = Style::ModalFolk.spec();
        dorian.texture.damage = 0.0;
        let score = compose_with_spec(
            &MusicalIntent {
                tonic: crate::pitch::PitchClass::D,
                valence: 0.5,
                ..Default::default()
            },
            &dorian,
        );
        assert_eq!(
            score.key.tonality,
            crate::harmony::Tonality::Modal(crate::scale::Mode::Dorian)
        );
    }

    #[test]
    fn styled_scores_realize_their_accompaniment_rhythm() {
        use crate::composer::{MusicalIntent, compose_styled, compose_with_spec};
        use crate::score::VoiceRole;
        // Waltz BODY: no harmony tone on ANY barline (beat 1 belongs to the
        // bass) and the bass is a crisp quarter note, not a whole-bar drone.
        // The plagal CODA is a deliberate texture break (held chords), so
        // both checks scope to the body. The hidden-bass hook quote is a
        // deliberate rhythm break too (augmented values mid-departure), so
        // this test — which pins the ACCOMPANIMENT rhythm — opts out of
        // the motif-memory device.
        let mut spec = Style::Waltz.spec();
        spec.texture.hook_cell = false;
        let waltz = compose_with_spec(&MusicalIntent::default(), &spec);
        let meter = waltz.meter as f64;
        let body =
            waltz.total_beats.beats() - (crate::composer::CODA_BARS * waltz.meter as i64) as f64;
        for n in waltz.voice(VoiceRole::Harmony) {
            if n.onset.beats() >= body - 1e-9 {
                continue;
            }
            let in_bar = n.onset.beats() % meter;
            assert!(in_bar > 1e-9, "oom-pah chord must not sit on the barline");
        }
        for n in waltz.voice(VoiceRole::Bass) {
            if n.onset.beats() >= body - 1e-9 {
                continue;
            }
            assert!(
                (n.duration.beats() - 1.0).abs() < 1e-9,
                "waltz bass must be quarter notes, got {} beats",
                n.duration.beats()
            );
        }
        // Classical seed 2 → Alberti: BODY harmony becomes moving eighth
        // notes, not one held block per bar (the coda's held chords are the
        // deliberate exception, as above).
        let alberti = compose_styled(
            &MusicalIntent {
                seed: 2,
                ..Default::default()
            },
            Style::Classical,
        );
        let alberti_body = alberti.total_beats.beats()
            - (crate::composer::CODA_BARS * alberti.meter as i64) as f64;
        assert!(
            alberti
                .voice(VoiceRole::Harmony)
                .iter()
                .filter(|n| n.onset.beats() < alberti_body - 1e-9)
                .all(|n| (n.duration.beats() - 0.5).abs() < 1e-9),
            "Alberti harmony must be all eighths"
        );
    }

    #[test]
    fn only_waltz_uses_a_non_four_meter() {
        for s in ALL_STYLES {
            let expected = if s == Style::Waltz { 3 } else { 4 };
            assert_eq!(s.meter(), expected, "{s:?}");
        }
    }

    #[test]
    fn every_style_every_arousal_tier_motif_sums_to_the_style_meter() {
        for s in ALL_STYLES {
            let meter_beats = s.meter() as f64;
            for arousal in [0.1_f32, 0.5, 0.9] {
                for seed in 0..4 {
                    let m = s.motif(arousal, seed);
                    let total = m.total_duration().beats();
                    assert!(
                        (total - meter_beats).abs() < 1e-9,
                        "{s:?} arousal={arousal} seed={seed}: motif sums to {total}, expected {meter_beats}"
                    );
                }
            }
        }
    }

    #[test]
    fn classical_style_matches_the_original_generate_motif_exactly() {
        // compose_styled(intent, Style::Classical) must be byte-identical to
        // compose(intent) -- this is the ground-truth check for that claim
        // at the motif level.
        for arousal in [0.1_f32, 0.5, 0.9] {
            for seed in 0..6 {
                let via_style = Style::Classical.motif(arousal, seed);
                let via_free_fn = crate::composer::classical_motif_bank(arousal, seed);
                assert_eq!(via_style, via_free_fn);
            }
        }
    }

    #[test]
    fn progression_length_always_matches_requested_bars() {
        for s in ALL_STYLES {
            for bars in [1, 3, 4, 7, 8, 11] {
                let p = s.progression(bars, 42);
                assert_eq!(p.len(), bars, "{s:?} bars={bars}");
            }
        }
    }

    #[test]
    fn non_classical_progressions_start_on_the_tonic() {
        // Every archetype used starts on I (degree 1) EXCEPT
        // sensitive_female, which deliberately starts on vi (degree 6).
        assert_eq!(Style::Waltz.progression(8, 1).degrees[0], 1);
        assert_eq!(Style::Folk.progression(8, 1).degrees[0], 1);
        assert_eq!(Style::Playful.progression(8, 1).degrees[0], 1);
        assert_eq!(Style::Cinematic.progression(8, 1).degrees[0], 6);
    }

    #[test]
    fn progression_is_deterministic() {
        for s in ALL_STYLES {
            assert_eq!(s.progression(6, 7), s.progression(6, 7));
        }
    }

    #[test]
    fn tempo_is_within_the_declared_range() {
        for s in ALL_STYLES {
            let (lo, hi) = s.tempo_range();
            assert_eq!(s.tempo(0.0), lo);
            assert_eq!(s.tempo(1.0), hi);
            let mid = s.tempo(0.5);
            assert!(mid > lo && mid < hi, "{s:?}");
        }
    }

    #[test]
    fn classical_tempo_matches_the_original_formula() {
        // compose()'s original formula was 60.0 + arousal.clamp(0,1)*90.0.
        for arousal in [0.0_f32, 0.3, 0.7, 1.0] {
            let original = 60.0 + arousal.clamp(0.0, 1.0) * 90.0;
            assert_eq!(Style::Classical.tempo(arousal), original);
        }
    }

    #[test]
    fn styles_pick_genuinely_different_shapes_at_the_same_arousal_and_seed() {
        // Not a guarantee for every pair (small banks could coincide), but
        // classical vs. cinematic at the same calm arousal/seed must differ
        // -- cinematic's calm shapes are wide leaps, classical's are stepwise.
        let a = Style::Classical.motif(0.1, 0);
        let b = Style::Cinematic.motif(0.1, 0);
        assert_ne!(a, b);
    }
}
