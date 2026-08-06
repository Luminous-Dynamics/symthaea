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
use crate::spec::{Attitude, DevelopmentDna, MelodicDna, PhraseRhetoric};
use serde::{Deserialize, Serialize};

/// `BaroqueSuite`'s progression BEFORE the 2026-07-26 harmonic-syntax pilot
/// (see `HARMONIC_SYNTAX_REWORK_SCOPE_2026-07-26.md`) — plain I-IV-V-I,
/// cycled to length. Deliberately kept as a named constant (not just a git-
/// history footnote) after the pilot's A/B evidence favored the real
/// functional-harmony generator (`ProgressionSpec::Grammar`) it replaced:
/// a compatibility baseline for debugging, an Analyst A/B control, and a
/// regression fixture proving the old behavior still composes cleanly.
/// See `composer::tests::baroque_suite_compatibility_baseline_still_composes_and_genuinely_differs_from_the_new_route`.
pub const BAROQUE_SUITE_COMPATIBILITY_PROGRESSION: [i32; 4] = [1, 4, 5, 1];

/// A named genre/style bias. See the module docs for what this does and
/// does not change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
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
    /// Variations over a remembering ground bass (see
    /// [`crate::passacaglia`]): 3/4, minor, the foundation stated, altered,
    /// restored, fragmented, and completed while the surface varies above
    /// it. Chosen by an experimental result, not the checklist: the ground
    /// attacks exactly the temporal-integration bottleneck the
    /// species-counterpoint falsification exposed.
    Passacaglia,
    /// The habanera's music: minor, 4/4, the first RHYTHM-CELL style —
    /// accompaniment and bass lock to an accent-carrying figure
    /// ([`crate::accompaniment::Accompaniment::Habanera`]) instead of a
    /// texture. Dotted, syncopated melodic identities over i-iv-V-i.
    /// The reusable habit this style taught the engine: rhythm + accent
    /// as a unit of identity (the cell), which blues shuffle, baroque
    /// dance figures, and minimalist pulses will reuse.
    Tango,
    /// The Irish/Scottish jig: Mixolydian, sextuple meter (the 3+3 lilt —
    /// see [`crate::accompaniment::Accompaniment::JigGait`]), a modal
    /// close (bVII-I — no dominant function at all, the "double tonic"
    /// real jigs and reels use) over a sustained tonic-fifth DRONE (the
    /// bass ignores the harmony above it, on purpose). Melody is
    /// decorated with quick unaccented "cuts" rather than accented leans.
    /// Two reusable habits at once: a genuinely static pedal texture
    /// (Nordic, Ambient will reuse it) and an unaccented ornament device
    /// (where appoggiaturas only had an accented one).
    Celtic,
    /// The 12-bar blues: major harmony (I-IV-I-I-IV-IV-I-I-V-IV-I-I, no
    /// mode change) colored by BLUE NOTES — melody tones on the third and
    /// seventh flattened at random, a deliberate melody/harmony scale
    /// mismatch that touches nothing underneath — over a boogie SHUFFLE
    /// bass (see [`crate::accompaniment::Accompaniment::Shuffle`], the
    /// exact "blues shuffle" the Habanera doc predicted this rhythm-cell
    /// mechanism would generalize to). Call-and-response isn't new
    /// machinery here — it falls out of the engine's existing antecedent/
    /// consequent period grammar, sharpened by the chord change landing
    /// exactly at bar 5.
    Blues,
    /// Debussy's harmonic language: Lydian (the floating, non-resolving
    /// raised fourth), a progression that avoids the dominant entirely, a
    /// rippling arpeggiated texture — and in the contrast section, PARALLEL
    /// PLANING (see [`crate::spec::TextureSpec::planing`]): the harmony
    /// stops following root motion and instead rides the melody's own
    /// contour, one struck chord shape per melody note. "Color over
    /// function" as a literal mechanism, not just a mood word.
    Impressionism,
    /// The chorale: Phrygian (the ecclesiastical mode, its half-step-above-
    /// tonic second degree giving the traditional "Phrygian cadence"
    /// color), slow homophonic block-chord texture, and SUSPENSIONS (see
    /// [`crate::spec::TextureSpec::suspension_rate`]) — the first ornament
    /// to live in the harmony voice rather than the melody: a real
    /// voice-leading candidate is tied over a chord change (a prepared
    /// dissonance) and resolves down by step. The plagal ("Amen") coda —
    /// already-built machinery, just switched on — closes every piece.
    SacredChoral,
    /// The process piece: a steady pulsing ostinato (see
    /// [`crate::accompaniment::Accompaniment::Arpeggio`]) under an
    /// ADDITIVE PROCESS theme (see
    /// [`crate::spec::TextureSpec::additive_process`]) — the hook cell
    /// grows one note at a time until it sounds whole, then shrinks back
    /// down, bouncing for as long as the section lasts. The process
    /// substitutes for melodic argument entirely: nothing here develops
    /// in the Classical sense. Static, mostly-tonic harmony; no coda —
    /// minimalist pieces don't resolve, they stop.
    Minimalism,
    /// The torch song: Aeolian (natural minor — the one ecclesiastical/
    /// modal color no prior style had claimed), a ii-V-I turnaround cycle,
    /// and SEVENTH CHORDS throughout (see
    /// [`crate::spec::TextureSpec::seventh_chords`]) — the lush maj7/
    /// min7/dom7 vocabulary that's jazz harmony's actual identity, not
    /// just an occasional cadential color. Compounds three earlier
    /// devices rather than adding only new ones: blue notes (predicted
    /// reusable here when Blues shipped), appoggiaturas, and Nocturne's
    /// Singing rhetoric all switch on together for the first time in one
    /// style.
    JazzBallad,
    /// The Sarabande: functional common-practice tonality (no exotic mode
    /// — Baroque dance identity comes from harmony and form, not scale
    /// color), a slow stately triple meter, a broken-chord continuo
    /// texture (violin/organ/cello trio), and a real HARMONIC SEQUENCE in
    /// the B section (see [`crate::spec::TextureSpec::harmonic_sequence`])
    /// — a genuine descending-fifths circle-of-fifths progression
    /// rewriting the section's harmonic plan itself, the quintessential
    /// Baroque development device (Pachelbel/Vivaldi-style), reusable by
    /// any future style wanting a real sequence episode.
    BaroqueSuite,
    /// LONG FORM via a genuine mid-piece METER CHANGE (see
    /// [`crate::spec::FormKind::ProgSuite`] / [`crate::prog_suite`]): four
    /// sections realized separately and spliced onto one timeline — the
    /// theme in 4, an asymmetric riff in 7, a modulating bridge in 5, the
    /// theme's return in 4 — with voice-leading carried continuously
    /// across every change. No prior style's `meter` was ever anything
    /// but a single constant; this is the first to actually move.
    ProgFolk,
    /// Interest with almost no events. The slowest tempo of any style,
    /// motifs that are one or two long tones per bar (never a run), zero
    /// ornamentation, and HARMONIC STASIS (see [`crate::spec::TextureSpec
    /// ::harmonic_stasis`]) — a static, repeated progression whose
    /// repeated tones TIE into long sustained notes instead of
    /// re-striking each bar, the mechanism that turns a plain repeated
    /// chord into a genuine drone. No dramatic damage, no coda: an
    /// ambient piece doesn't resolve, it just stops sounding.
    Ambient,
    /// Sonata form (see [`crate::spec::FormKind::Sonata`] /
    /// [`crate::sonata`]): the first form built around TONAL CONFLICT
    /// AND RESOLUTION rather than contrast alone. A second subject is
    /// introduced in a real foreign key (the dominant), developed
    /// through a third key, then the recapitulation brings the EXACT
    /// SAME idea home — provably the same scale-degree content, only the
    /// key differs. Alberti-figured classical trio, functional tonality.
    Sonata,
    /// Renaissance polyphony (see [`crate::spec::FormKind::Renaissance`] /
    /// [`crate::renaissance`]): three EQUAL voices, no subject-entry
    /// hierarchy — unlike Fugue, no voice's imitation is a privileged
    /// "answer." Two points of imitation entering at the OCTAVE (not a
    /// tonal transposition), voice order rotating low-to-high then
    /// high-to-low, closing on a real modal suspension-and-under-third
    /// (Landini) cadence. Dorian, no leading tone borrowed.
    RenaissancePolyphony,
    /// Afro-Cuban son montuno (see [`crate::accompaniment::Accompaniment::
    /// Montuno`]): the first rhythm cell that is not a single repeating
    /// bar — son clave (3-2) is a TWO-BAR cycle, alternating a tresillo
    /// three-side with a backbeat-adjacent two-side, and the cycle never
    /// resets at a section boundary (real clave doesn't either). Paired
    /// with a tumbao bass whose onsets are chosen to interlock with —
    /// never land on — the montuno's own onsets: layered, conversational
    /// rhythm as a checkable non-overlap property, not a vibe. Dorian, no
    /// leading tone (the "Oye Como Va" vamp idiom); the form doesn't
    /// develop because a vamp isn't supposed to — the groove is the point.
    AfroCuban,
    /// Flamenco (see [`crate::accompaniment::Accompaniment::CompasGait`]):
    /// the compás — a 12-beat cycle grouped 3+3+2+2+2, accenting only
    /// counts 3/6/8/10/12 and leaving every other beat silent. The first
    /// meter in the engine longer than a single-digit bar, and the first
    /// rhythm cell defined as much by its rests as its hits. Harmony walks
    /// the Andalusian cadence (iv-III-II-i, a descending stepwise tetrachord
    /// rather than the usual fifths-based motion) in Phrygian — the
    /// engine's first Phrygian style. Development is `Intensifying`: the
    /// dramatic arc builds toward a remate, not a formal transformation.
    Flamenco,
    /// Bossa Nova (see [`crate::accompaniment::Accompaniment::BossaComp`]):
    /// the first rhythm cell defined by an ABSENCE of silence rather than a
    /// presence of accent — syncopated onsets (0, 1.5, 3.0) whose durations
    /// chain exactly into one another, so the chord never stops ringing.
    /// "Floating" jazz-extended harmony (`seventh_chords`, every chord not
    /// just the cadential one) over a bass that holds through beat 3 then
    /// softly anticipates the next bar. Understated throughout: every
    /// accent multiplier in the roster's softest range.
    BossaNova,
    /// Opera / Art Song (see [`crate::spec::FormKind::Opera`] /
    /// [`crate::opera`]): the first form with TWO independent melodic
    /// identities in structured conversation instead of one melody voice
    /// developing alone. Theme A (Melody, confident/triadic) and Theme B
    /// (CounterMelody, searching/stepwise, its own instrument) solo
    /// separately, trade one bar at a time in a dialogue section, then B's
    /// final phrase is genuinely CUT OFF and A enters early to resolve —
    /// interruption as a checkable fact, not a mood.
    Opera,
    /// Irish Traditional (see [`crate::spec::TextureSpec::roll_ornaments`]
    /// / `apply_roll_ornaments`): the engine's first ORNAMENT CHAIN — not
    /// Celtic's single grace-note "cut" but a full five-note roll (main,
    /// upper cut, main, lower cut, main) filling exactly the decorated
    /// note's own duration. The REEL, not the jig (Celtic already owns
    /// 6/8) — a driving straight-eighth stream in Dorian, distinct modal
    /// color from Celtic's Mixolydian, over an Arpeggio's engine-room
    /// pulse. A real session trio (flute/guitar/upright bass), not
    /// Celtic's soundtrack cast.
    IrishTraditional,
    /// Hindustani-inspired (see [`crate::spec::TextureSpec::full_drone`] /
    /// `apply_full_drone`): the engine's first FULL drone — not just
    /// Celtic's bass pedal under a moving harmony, but Harmony too,
    /// replaced entirely with a static tonic-fifth-octave pad tied into
    /// one continuous sustain. No chord progression exists anywhere in
    /// the piece, so "tension without modulation" is a structural
    /// guarantee, not a mood. Dorian (a fair analogue of Kafi thaat, a
    /// common Hindustani parent scale), unhurried tempo, register/
    /// figuration/velocity arc reused for a long melodic evolution, grace
    /// ornaments as the melody's real language. Sitar's engine debut.
    HindustaniInspired,
}

impl Style {
    /// Every style, in `LISTEN_STYLES`-independent canonical declaration
    /// order. The single source of truth for "every style" -- callers that
    /// need to enumerate styles (a client wanting each style's
    /// [`Self::grammar_family`], a future style-picker UI, this crate's own
    /// exhaustiveness tests) should iterate this rather than hand-listing
    /// variants, which is exactly the drift risk `grammar_family`'s own doc
    /// comment below warns against for `HOME_STYLES`-style duplication.
    pub const ALL: [Style; 29] = [
        Style::Classical,
        Style::Waltz,
        Style::Folk,
        Style::Cinematic,
        Style::Playful,
        Style::Nocturne,
        Style::March,
        Style::Lullaby,
        Style::ModalFolk,
        Style::Fugue,
        Style::Passacaglia,
        Style::Tango,
        Style::Celtic,
        Style::Blues,
        Style::Impressionism,
        Style::SacredChoral,
        Style::Minimalism,
        Style::JazzBallad,
        Style::BaroqueSuite,
        Style::ProgFolk,
        Style::Ambient,
        Style::Sonata,
        Style::RenaissancePolyphony,
        Style::AfroCuban,
        Style::Flamenco,
        Style::BossaNova,
        Style::Opera,
        Style::IrishTraditional,
        Style::HindustaniInspired,
    ];

    /// The compositional grammar family this style specializes — the single
    /// source of truth every other module should derive from rather than
    /// hardcoding its own style list (e.g. `foundry.rs`'s `HOME_STYLES`,
    /// `contrapuntal_foundry.rs`'s compatible-styles list).
    ///
    /// `AfroCuban`/`Minimalism`/`HindustaniInspired` are the three
    /// "flagship" styles [`crate::grammar::compose_with_grammar_plan`]
    /// routes to dedicated new engines
    /// ([`crate::groove_cycle`]/[`crate::process_grammar`]/
    /// [`crate::modal_arc`]) instead of the shared period pipeline — a real
    /// behavior upgrade for those three, not just a label. Every other
    /// style maps to a family whose `compose_with_grammar_plan` arm falls
    /// through to the existing [`crate::composer::compose_with_spec_and_form`]
    /// pipeline, so this mapping is behavior-preserving for all 25 other
    /// styles. `RenaissancePolyphony` is contrapuntal in spirit (equal-voice
    /// imitative entries) but is deliberately NOT mapped to
    /// `GrammarFamily::Contrapuntal` — that arm's dispatch is hardcoded to
    /// `crate::fugue::realize_fugue`, so mapping it there would silently
    /// reroute Renaissance's own realizer through the fugue engine. Giving
    /// Renaissance (and Sonata/ProgFolk, similarly parked under
    /// `Developmental`) their own dedicated dispatch arms is real follow-up
    /// work, not attempted here.
    pub fn grammar_family(self) -> crate::grammar::GrammarFamily {
        use crate::grammar::GrammarFamily::*;
        match self {
            Style::Classical
            | Style::Waltz
            | Style::Folk
            | Style::Cinematic
            | Style::Playful
            | Style::Nocturne
            | Style::March
            | Style::Lullaby
            | Style::ModalFolk
            | Style::Impressionism
            | Style::SacredChoral
            | Style::BaroqueSuite => PeriodSentence,
            Style::Fugue => Contrapuntal,
            Style::Passacaglia => GroundVariation,
            Style::Tango
            | Style::Celtic
            | Style::Flamenco
            | Style::BossaNova
            | Style::IrishTraditional => StrophicSong,
            Style::Blues => BluesCallResponse,
            Style::JazzBallad => JazzChorus,
            Style::AfroCuban => GrooveCycle,
            Style::HindustaniInspired => RagaModalArc,
            Style::Minimalism => ProcessAdditive,
            Style::Ambient => AmbientTextural,
            Style::Opera => DramaticAdaptive,
            Style::ProgFolk | Style::Sonata | Style::RenaissancePolyphony => Developmental,
        }
    }

    /// This style's full [`crate::grammar::GrammarProfile`] (family, phrase
    /// grammar, harmonic syntax, performance dialect, supported intent
    /// axes) — `self.grammar_family().profile()`.
    pub fn grammar_profile(self) -> crate::grammar::GrammarProfile {
        self.grammar_family().profile()
    }

    /// Which [`crate::composer::MusicalIntent`] expressive axes (valence/
    /// arousal/energy/bars) actually have a measurable effect on this
    /// style's composed output — `seed` and `tonic` are excluded since
    /// they always vary the output for every style, so they're not
    /// interesting to report here.
    ///
    /// Ground truth from the Causal Diversity Validation census
    /// (`examples/diversity_census.rs`'s one-factor-at-a-time sensitivity
    /// matrix): 26 of 174 style×parameter cells showed NO measurable effect
    /// on composition-level output. Every one of the 26 is a genuine
    /// grammar limitation, not a wiring bug — verified against the actual
    /// composer code (not inferred from the census numbers alone):
    ///
    /// - `valence` selects major-vs-minor tonality only when this style's
    ///   `CompositionSpec.mode` is `None` (see `compose_with_spec_and_form`'s
    ///   key derivation). The 12 styles below deliberately pin a fixed mode
    ///   (Dorian, Phrygian, Mixolydian, …) as their harmonic identity —
    ///   valence has no key left to choose between.
    /// - `energy`/`bars` (and, for two styles, `arousal`) are consumed only
    ///   by the shared period/`Form` pipeline (the `use_sentence` phrase
    ///   choice and phrase-length building). Fugue, the three ground-music
    ///   forms (Passacaglia/Erosion/Lineage — one `Style` variant,
    ///   `Style::Passacaglia`), ProgSuite, Sonata, Renaissance, and Opera
    ///   all bypass that pipeline entirely via early-return in
    ///   `compose_with_spec_and_form`, calling their own fixed-plan
    ///   realizers (`realize_fugue`, `realize_passacaglia`,
    ///   `compose_sonata_with_plan`, etc.) whose signatures never receive
    ///   `energy`/`bars` at all (confirmed by inspection, not guessed).
    ///   Passacaglia and RenaissancePolyphony additionally show `arousal`
    ///   dead — those two also pin a fixed mode, so BOTH of arousal's and
    ///   valence's usual levers (subject/motif-bank shape and mode
    ///   selection) are foreclosed for them, alongside the shared
    ///   energy/bars limitation every bypass grammar has.
    ///
    /// Giving these styles a real energy/bars lever would mean designing
    /// genuinely new structural grammar (e.g. an energy-driven episode
    /// count in a fugue, or a length-driven number of ground-bass
    /// statements) — real, valuable future work, but deliberately NOT
    /// attempted here: inventing a new mood-selection surface from scratch
    /// for a style with no existing lever is a design decision, not a
    /// wiring fix, and out of scope for this pass.
    pub fn supported_intent_axes(self) -> &'static [&'static str] {
        const ALL: &[&str] = &["valence", "arousal", "energy", "bars"];
        // Fugue/ProgFolk/Sonata/Opera: energy+bars dead (bypass pipeline),
        // but neither pins a mode and neither shows arousal dead, so
        // valence+arousal remain live.
        const NO_ENERGY_BARS: &[&str] = &["valence", "arousal"];
        // Passacaglia/RenaissancePolyphony: energy+bars dead (bypass
        // pipeline) AND valence+arousal dead (pinned mode forecloses both
        // of their usual levers) — none of the four axes measurably affect
        // these two styles' composition-level output.
        const NONE_LIVE: &[&str] = &[];
        // Mode-pinned styles where only valence is dead (arousal/energy/
        // bars all still go through the shared period pipeline normally).
        const NO_VALENCE: &[&str] = &["arousal", "energy", "bars"];
        match self {
            Style::Fugue | Style::ProgFolk | Style::Sonata | Style::Opera => NO_ENERGY_BARS,
            Style::Passacaglia | Style::RenaissancePolyphony => NONE_LIVE,
            Style::ModalFolk
            | Style::Tango
            | Style::Celtic
            | Style::Impressionism
            | Style::SacredChoral
            | Style::JazzBallad
            | Style::AfroCuban
            | Style::Flamenco
            | Style::IrishTraditional
            | Style::HindustaniInspired => NO_VALENCE,
            _ => ALL,
        }
    }

    /// Beats per measure. Only [`Style::Waltz`]/[`Style::Flamenco`] etc.
    /// differ from 4/4.
    pub fn meter(self) -> u8 {
        match self {
            Style::Waltz | Style::Lullaby | Style::Passacaglia | Style::BaroqueSuite => 3,
            Style::Celtic => 6,    // the jig's 3+3 lilt
            Style::Flamenco => 12, // the compás's 3+3+2+2+2 cycle
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
            Style::Passacaglia => (55.0, 95.0),
            Style::Tango => (100.0, 132.0),
            Style::Celtic => (110.0, 150.0),
            Style::Blues => (70.0, 110.0),
            Style::Impressionism => (52.0, 88.0),
            Style::SacredChoral => (48.0, 72.0),
            Style::Minimalism => (108.0, 144.0),
            Style::JazzBallad => (52.0, 86.0),
            Style::BaroqueSuite => (54.0, 88.0),
            Style::ProgFolk => (92.0, 138.0),
            Style::Ambient => (32.0, 52.0),
            Style::Sonata => (100.0, 152.0),
            Style::RenaissancePolyphony => (66.0, 100.0),
            Style::AfroCuban => (92.0, 128.0),
            Style::Flamenco => (108.0, 156.0),
            Style::BossaNova => (96.0, 132.0),
            Style::Opera => (66.0, 112.0),
            Style::IrishTraditional => (112.0, 148.0),
            Style::HindustaniInspired => (56.0, 92.0),
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
    /// ([`Progression::generate`]); `Style::Waltz` cycles a single named
    /// archetype; `Folk`/`Cinematic`/`Playful` each seed-vary between two
    /// harmonically disjoint patterns (see the census-driven fix note below)
    /// to exactly `bars` measures, so the length contract callers rely on
    /// (`progression.len() == bars.max(1)`) holds for every style.
    ///
    /// The disjoint pools for `Folk`/`Cinematic`/`Playful` directly address a
    /// real finding from the diversity census (`diversity_census.rs`): all
    /// three used to hardcode ONE progression apiece — `four_chord()`
    /// [1,5,6,4], `sensitive_female()` [6,4,1,5], `pop_turnaround()`
    /// [1,6,4,5] — which are just rotations of the SAME four chords
    /// {I,IV,V,vi}, with zero seed variation. That gave these three styles
    /// (a) no within-style harmonic variety at all, and (b) an
    /// indistinguishable harmonic-trajectory fingerprint from each other,
    /// which the census measured as 40-51% cross-style nearest-neighbor
    /// bleed (far above the <10% typical of other style pairs). Each style
    /// now draws from its own degree set: Folk = {1,4,5} (plain
    /// authentic motion, no color chords — a "grounded" identity), Playful =
    /// {1,2,4,5} (adds the jazzier predominant ii — a "turn/surprise"
    /// identity), Cinematic = {1,3,4,6} with NO dominant at all (a
    /// deliberately unresolved/suspended arc — matches "harmonic
    /// suspension, delayed arrival" as a real identity marker, not just an
    /// orchestration one). `Waltz`'s existing `pachelbel()` [1,3,4,5,6] is
    /// left untouched — it wasn't part of the census's flagged bleed
    /// quartet.
    pub fn progression(self, bars: usize, seed: u64) -> Progression {
        let bars = bars.max(1);
        // Delegates entirely to the spec — the single source of truth.
        //
        // This used to hardcode five styles (Classical/Waltz/Folk/Cinematic/
        // Playful) that duplicated what their own specs already said, making it
        // a second, silently-divergable progression source: a style whose spec
        // changed would keep returning the old progression here, and `live.rs`
        // is a real consumer. Removed 2026-07-30 after proving the arms
        // redundant — legacy and spec agreed on 0 divergences across all 29
        // styles x 5 bar-lengths x 24 seeds — so this is a pure dedup, not a
        // behaviour change. `style_progression_delegates_to_the_spec` keeps it
        // that way.
        self.spec().progression(bars, seed)
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
            deceptive_close: false,
            drone: false,
            planing: false,
            suspension_rate: 0.0,
            additive_process: false,
            seventh_chords: false,
            harmonic_sequence: false,
            harmonic_stasis: false,
            roll_ornaments: false,
            full_drone: false,
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
                // Classical melodic DNA: 12 of 29 styles used to draw
                // hooks from the identical shared skeleton pool, Classical
                // among them — this is the engine's own default-preset
                // style, so it's the natural first fix. Elegant, balanced
                // shapes: equal-note insistence resolving to a long
                // arrival, and arpeggiated thirds with a clear turn.
                melody: MelodicDna {
                    hook_rhythms: vec![
                        vec![(1, 1), (1, 1), (2, 1)],
                        vec![(2, 1), (1, 1), (1, 1)],
                        vec![(1, 2), (1, 2), (1, 1)],
                        vec![(1, 2), (1, 2), (1, 2), (3, 2)],
                    ],
                    hook_contours: vec![
                        vec![1, 3, 2],
                        vec![1, 5, 3],
                        vec![3, 3, 5],
                        vec![1, 1, 5, 3],
                    ],
                    appoggiatura_rate: 0.3,
                    ornament_rate: 0.0,
                    blue_note_rate: 0.0,
                    use_procedural_foundry: false,
                },
                meter_pool: vec![4, 3],
                mode_pool: vec![
                    None,
                    Some(crate::scale::Mode::Dorian),
                    Some(crate::scale::Mode::Aeolian),
                    Some(crate::scale::Mode::Lydian),
                ],
                rhetoric: PhraseRhetoric::Classic,
                development: DevelopmentDna::Figural, // the textbook classical technique — embellish, don't transpose
                texture: {
                    let mut t = texture(DrumPolicy::None);
                    t.deceptive_close = true;
                    t
                },
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
                progression: ProgressionSpec::GrammarWithPalette(
                    crate::harmony::HarmonicPalette::waltz(),
                ),
                accompaniment_pool: vec![A::OomPah],
                form_pool: vec![FormKind::Ternary, FormKind::Rondo],
                ensemble_pool: s(&[
                    ["violin", "piano", "cello"],
                    ["flute", "piano", "cello"],
                    ["violin", "harp", "cello"],
                ]),
                mode: None,
                counter_instrument: None,
                // Waltz melodic DNA: the lilt itself — a lift up to the
                // dominant and a falling cadence gesture, sized to
                // meter=3 (unlike Classical's meter-4 pool, which would
                // rarely fit here).
                melody: MelodicDna {
                    hook_rhythms: vec![
                        vec![(2, 1), (1, 1)],
                        vec![(1, 1), (2, 1)],
                        vec![(1, 2), (1, 2), (1, 1), (1, 1)],
                    ],
                    hook_contours: vec![vec![1, 5], vec![5, 1], vec![1, 5, 3, 1]],
                    appoggiatura_rate: 0.0,
                    ornament_rate: 0.0,
                    blue_note_rate: 0.0,
                    use_procedural_foundry: false,
                },
                meter_pool: vec![],
                mode_pool: vec![],
                rhetoric: PhraseRhetoric::Classic,
                development: DevelopmentDna::Sequential, // the waltz's own circling sequence, distinct meter from Tango's
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
                // Was a single fixed archetype [1,5,6,4] (four_chord) — a
                // rotation of the SAME {I,IV,V,vi} set Cinematic/Playful
                // also used, with zero seed variation. The census measured
                // this as 40-51% cross-style nearest-neighbor bleed between
                // Folk/Cinematic/Playful/Classical. Folk's new identity:
                // plain authentic I-IV-V motion, no vi/ii/iii color chords.
                progression: ProgressionSpec::ArchetypePool(vec![
                    vec![1, 4, 5, 1],
                    vec![1, 5, 4, 1],
                ]),
                accompaniment_pool: vec![A::Arpeggio, A::Block],
                form_pool: vec![FormKind::Ternary, FormKind::Rondo],
                ensemble_pool: s(&[
                    ["flute", "acoustic_guitar", "upright_bass"],
                    ["clarinet", "harp", "upright_bass"],
                    ["flute", "koto", "upright_bass"],
                ]),
                mode: None,
                counter_instrument: None,
                // Folk melodic DNA: open, song-like shapes — settle-then-
                // reach insistence and a modal-flavored 4th-degree lean —
                // distinct from Classical's more symmetrical arpeggiation.
                melody: MelodicDna {
                    hook_rhythms: vec![
                        vec![(1, 1), (1, 1), (1, 2), (1, 2)],
                        vec![(1, 2), (1, 2), (1, 1), (1, 1)],
                        vec![(2, 1), (1, 1), (1, 1)],
                    ],
                    hook_contours: vec![vec![1, 1, 5, 3], vec![3, 1, 5], vec![1, 4, 1]],
                    appoggiatura_rate: 0.0,
                    ornament_rate: 0.0,
                    blue_note_rate: 0.0,
                    use_procedural_foundry: false,
                },
                meter_pool: vec![4, 3],
                mode_pool: vec![
                    None,
                    Some(crate::scale::Mode::Dorian),
                    Some(crate::scale::Mode::Mixolydian),
                ],
                rhetoric: PhraseRhetoric::Classic,
                development: DevelopmentDna::Wandering, // drift, not argument — the pastoral habit
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
                // Was a single fixed archetype [6,4,1,5] (sensitive_female)
                // — the same {I,IV,V,vi} rotation Folk/Playful also used,
                // zero seed variation (see Folk's identical note above).
                // Cinematic's new identity deliberately EXCLUDES V/the
                // dominant entirely: a suspended, never-authentically-
                // resolving I-vi-iii-IV arc, matching "harmonic suspension,
                // delayed arrival" as a real harmonic marker, not just an
                // orchestration one.
                progression: ProgressionSpec::ArchetypePool(vec![
                    vec![1, 6, 3, 4],
                    vec![1, 3, 6, 4],
                ]),
                accompaniment_pool: vec![A::Block],
                form_pool: vec![FormKind::Ternary, FormKind::Rondo],
                ensemble_pool: s(&[
                    ["violin", "organ", "cello"],
                    ["trumpet", "organ", "cello"],
                    ["violin", "pad", "cello"],
                ]),
                mode: None,
                counter_instrument: None,
                // Cinematic melodic DNA: sustained long-long-short arrival
                // shapes and a dramatic octave reach with recoil — the
                // "sweeping and dramatic" identity in the hook layer
                // itself, not just the orchestration/harmony devices
                // already carrying it.
                melody: MelodicDna {
                    hook_rhythms: vec![
                        vec![(2, 1), (1, 1), (1, 1)],
                        vec![(1, 1), (3, 2), (1, 2)],
                        vec![(1, 2), (1, 1), (1, 1), (1, 1)],
                    ],
                    hook_contours: vec![vec![1, 8, 5], vec![5, 5, 1], vec![1, 5, 8, 5]],
                    appoggiatura_rate: 0.15,
                    ornament_rate: 0.0,
                    blue_note_rate: 0.0,
                    use_procedural_foundry: false,
                },
                meter_pool: vec![4, 5],
                mode_pool: vec![],
                rhetoric: PhraseRhetoric::Classic,
                development: DevelopmentDna::Intensifying, // rising tension, a real dramatic arc
                texture: {
                    let mut t = texture(DrumPolicy::BarPulse);
                    t.deceptive_close = true;
                    t
                },
            },
            Style::Playful => CompositionSpec {
                name: "Playful".into(),
                // Joy's mechanism (lighter/detached articulation, +8% tempo
                // lift) reinforces what this style's own doc already claims
                // ("bright and bouncy... a brisk tempo floor") rather than
                // introducing anything new.
                attitude: Some(Attitude::Joy),
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
                // Was a single fixed archetype [1,6,4,5] (pop_turnaround) —
                // the same {I,IV,V,vi} rotation Folk/Cinematic also used,
                // zero seed variation (see Folk's identical note above).
                // Playful's new identity: the jazzier ii-V-I turnaround
                // family (the predominant ii never appears in Folk's or
                // Cinematic's pools) — a real "turn/surprise" harmonic
                // color instead of the generic four-chord loop.
                progression: ProgressionSpec::ArchetypePool(vec![vec![2, 5, 1], vec![2, 5, 1, 4]]),
                accompaniment_pool: vec![A::Comp, A::Alberti],
                form_pool: vec![FormKind::Ternary, FormKind::Rondo],
                ensemble_pool: s(&[
                    ["clarinet", "electric_piano", "acoustic_guitar"],
                    ["saxophone", "marimba", "acoustic_guitar"],
                    ["trumpet", "kalimba", "acoustic_guitar"],
                ]),
                mode: None,
                counter_instrument: None,
                // Playful melodic DNA: skip-then-hop bouncy rhythms and
                // wide, quick-recoiling leaps — "bright and bouncy" as a
                // hook identity, distinct from Classical's more balanced
                // arpeggiation and Cinematic's sustained arrivals.
                melody: MelodicDna {
                    hook_rhythms: vec![
                        vec![(1, 2), (1, 2), (1, 1)],
                        vec![(1, 1), (1, 2), (1, 2)],
                        vec![(1, 2), (1, 4), (1, 4), (1, 2)],
                    ],
                    hook_contours: vec![vec![1, 3, 1], vec![5, 1, 5], vec![1, 3, 1, 5]],
                    appoggiatura_rate: 0.0,
                    ornament_rate: 0.4,
                    blue_note_rate: 0.0,
                    use_procedural_foundry: false,
                },
                meter_pool: vec![],
                mode_pool: vec![],
                rhetoric: PhraseRhetoric::Classic,
                development: DevelopmentDna::Fragmenting, // energetic toss, compression as play
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
                progression: ProgressionSpec::GrammarWithPalette(
                    crate::harmony::HarmonicPalette::nocturne(),
                ),
                accompaniment_pool: vec![A::Arpeggio],
                // Variations belongs here: a nocturne dwelling on one idea,
                // darkening it (minore) and ornamenting it (figuration), is
                // the Chopin/Field association exactly.
                form_pool: vec![FormKind::Ternary, FormKind::Rondo, FormKind::Variations],
                ensemble_pool: s(&[["clarinet", "piano", "cello"], ["flute", "piano", "cello"]]),
                counter_instrument: None,
                // Nocturne melodic DNA: long-short sighs; stepwise
                // identities anchored by repetition or a gentle third —
                // the singing register of the night piece.
                melody: MelodicDna {
                    hook_rhythms: vec![
                        vec![(2, 1), (1, 2), (1, 2)],
                        vec![(2, 1), (1, 1), (1, 2)],
                        vec![(3, 2), (1, 2), (1, 1)],
                    ],
                    hook_contours: vec![vec![3, 3, 2], vec![5, 3, 4], vec![6, 5, 5], vec![5, 5, 4]],
                    appoggiatura_rate: 0.4, // leaning sighs
                    ornament_rate: 0.0,
                    blue_note_rate: 0.0,
                    use_procedural_foundry: false,
                },
                // Question — expansion — suspension — quiet arrival.
                meter_pool: vec![],
                mode_pool: vec![],
                rhetoric: PhraseRhetoric::Singing,
                development: DevelopmentDna::Figural, // ornament as thought
                texture: {
                    let mut t = texture(DrumPolicy::None);
                    t.deceptive_close = true;
                    t
                },
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
                progression: ProgressionSpec::GrammarWithPalette(
                    crate::harmony::HarmonicPalette::march(),
                ),
                accompaniment_pool: vec![A::Block],
                form_pool: vec![FormKind::Ternary, FormKind::Rondo],
                ensemble_pool: s(&[["trumpet", "organ", "cello"]]),
                counter_instrument: None,
                // March melodic DNA: dotted-march cells plus genuine bugle-
                // call contours — the melody-only listening test localized
                // March's failure to here. The old contours ([1,1,5] etc.)
                // passed the identity predicate only via a cheap immediate
                // repeat; they didn't actually SOUND like a march without
                // the drums that were stripped for the test. These four are
                // real bugle-call shapes (triadic reach to the octave,
                // arpeggio-and-drop, upper call-and-answer, full descent) —
                // every one still predicate-valid (recoil or closing leap
                // ≥3 semitones), but now for a musical reason, not a cheap
                // one.
                melody: MelodicDna {
                    hook_rhythms: vec![
                        vec![(3, 4), (1, 4), (1, 1)],
                        vec![(3, 2), (1, 2), (1, 1)],
                        vec![(1, 1), (1, 2), (1, 2), (2, 1)],
                        vec![(1, 2), (1, 2), (1, 2), (2, 1)], // ta-ta-ta-TAA
                    ],
                    hook_contours: vec![
                        vec![1, 3, 5, 8], // charge: ascending triad to the octave
                        vec![1, 5, 3, 1], // arpeggio call-and-drop
                        // Upper call, answered -- a 2026-07-24 melody-only
                        // listening test found a March misheard as Tango,
                        // traced (via a real symbolic contour probe, not a
                        // guess) to hook contours that leap and then leap
                        // STRAIGHT BACK (5->8->5, no stepwise recovery in
                        // between) -- exactly the leap-reversal pattern
                        // that made the confused clip measurably more
                        // angular than either a clean March control OR
                        // real Tango. Was [5, 8, 5, 3]; the answer now
                        // steps down first (8->7, a real recovery) before
                        // continuing down to 5 -- same reach-and-answer
                        // shape, no longer an immediate leap-then-leap.
                        vec![5, 8, 7, 5],
                        // Full descending call -- same fix, same reasoning:
                        // was [8, 5, 1] (two back-to-back leaps, no
                        // stepwise motion anywhere). A brief step (8->7)
                        // before the full plunge to the tonic keeps the
                        // "reach then fall" character while breaking up
                        // the double-leap.
                        vec![8, 7, 1],
                    ],
                    appoggiatura_rate: 0.0, // a march leans on nothing
                    ornament_rate: 0.0,
                    blue_note_rate: 0.0,
                    use_procedural_foundry: false,
                },
                meter_pool: vec![],
                mode_pool: vec![],
                rhetoric: PhraseRhetoric::Martial,
                development: DevelopmentDna::Fragmenting, // compression as drive
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
                // A lullaby's melodic identity is the narrowest in the
                // catalogue: stepwise, descending, and repetitive — the tune
                // settles rather than travels. Rhythms fill this style's meter
                // of 3 with an unhurried rock (long-short and short-long), and
                // the contours mirror the descending gestures its own
                // `motifs_calm` already use ((3,2,1)->(1,1,1), (5,2,1)->(3,1,1)).
                // No leaps: a lullaby that leaps is a different piece.
                melody: MelodicDna {
                    hook_rhythms: vec![
                        vec![(1, 1), (1, 1), (1, 1)],
                        vec![(2, 1), (1, 1)],
                        vec![(1, 1), (2, 1)],
                        vec![(1, 2), (1, 2), (1, 1), (1, 1)],
                    ],
                    hook_contours: vec![vec![3, 2, 1], vec![5, 3, 1], vec![1, 2, 1], vec![3, 2, 3]],
                    ..MelodicDna::default()
                },
                meter_pool: vec![],
                mode_pool: vec![],
                rhetoric: PhraseRhetoric::Classic,
                development: DevelopmentDna::Figural, // a hummed tune varies gently on repeat, never argues
                texture: texture(DrumPolicy::None),
            },
            Style::ModalFolk => CompositionSpec {
                name: "ModalFolk".into(),
                // "The modal wanderer" (this style's own doc) doesn't
                // settle -- Curiosity's unresolved melodic ending (resolves
                // UP to degree 2, "question intonation", no answer) is the
                // same idea the doc already names, not an invented one.
                attitude: Some(Attitude::Curiosity),
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
                // ModalFolk melodic DNA: leans on the Dorian-defining
                // degrees (4 and 7) rather than Folk's plainer stepwise
                // shapes — "modal cousins" in development habit, but a
                // genuinely distinct melodic surface.
                melody: MelodicDna {
                    hook_rhythms: vec![
                        vec![(1, 1), (1, 1), (1, 1), (1, 2)],
                        vec![(3, 2), (1, 2), (1, 1)],
                        vec![(1, 1), (1, 2), (1, 2), (1, 1)],
                    ],
                    hook_contours: vec![vec![1, 4, 7, 4], vec![1, 7, 4], vec![4, 4, 1]],
                    appoggiatura_rate: 0.0,
                    ornament_rate: 0.0,
                    blue_note_rate: 0.0,
                    use_procedural_foundry: false,
                },
                meter_pool: vec![],
                mode_pool: vec![],
                rhetoric: PhraseRhetoric::Classic,
                development: DevelopmentDna::Wandering, // the same drift as Folk — modal cousins, a shared habit
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
                melody: MelodicDna::default(),
                meter_pool: vec![],
                mode_pool: vec![],
                rhetoric: PhraseRhetoric::Classic,
                development: DevelopmentDna::Classic,
                texture: texture(DrumPolicy::None),
            },
            Style::Passacaglia => CompositionSpec {
                name: "Passacaglia".into(),
                attitude: None,
                // The Baroque passacaglia is a minor-mode form; harmonic
                // minor gives the ground's leading-tone pickup its real
                // pull back to the tonic at every cycle restart.
                mode: Some(crate::scale::Mode::HarmonicMinor),
                meter: 3,
                tempo_range: (55.0, 95.0),
                // 3-beat subjects with clear identities that survive both
                // augmentation (the ground is the subject slowed ×2) and
                // figuration (the restoration cycle's blaze).
                motifs_calm: vec![
                    vec![(1, 1, 1), (3, 1, 1), (2, 1, 1)],
                    vec![(5, 1, 1), (4, 1, 1), (3, 1, 1)],
                    vec![(1, 2, 1), (2, 1, 1)],
                ],
                motifs_medium: vec![
                    vec![(1, 1, 2), (2, 1, 2), (3, 1, 1), (2, 1, 1)],
                    vec![(3, 1, 1), (4, 1, 2), (3, 1, 2), (1, 1, 1)],
                ],
                motifs_busy: vec![
                    vec![
                        (1, 1, 2),
                        (2, 1, 2),
                        (3, 1, 2),
                        (4, 1, 2),
                        (3, 1, 2),
                        (2, 1, 2),
                    ],
                    vec![(5, 1, 2), (4, 1, 2), (3, 1, 2), (2, 1, 2), (1, 1, 1)],
                ],
                // Unused by the ground-music paths (the ground is the form)
                // but required for a valid spec.
                progression: ProgressionSpec::Archetype(vec![1, 6, 4, 5]),
                accompaniment_pool: vec![A::Block],
                // The three ground grammars under one style, seed-picked:
                // persistence (the ground remains), erosion (it fades —
                // what did persistence cost?), lineage (it becomes its
                // descendants). Separate primitives, shared foundation.
                form_pool: vec![FormKind::Passacaglia, FormKind::Erosion, FormKind::Lineage],
                ensemble_pool: s(&[["clarinet", "organ", "cello"]]),
                counter_instrument: None,
                melody: MelodicDna::default(),
                meter_pool: vec![],
                mode_pool: vec![],
                rhetoric: PhraseRhetoric::Classic,
                // Unused by the ground-music paths, same as `progression`
                // above — the real development IS the ground-bass variation
                // (`figuration_variation`/`contrasting_transform` in
                // `passacaglia.rs`, seed- and cycle-driven), not this field.
                development: DevelopmentDna::Classic,
                texture: texture(DrumPolicy::None),
            },
            Style::Tango => CompositionSpec {
                name: "Tango".into(),
                // This style's own doc already describes an assertive,
                // accent-carrying rhythmic identity ("dotted, syncopated
                // melodic identities... an accent-carrying figure").
                // Defiance's mechanism (further syncopation, +8% assertive
                // bass velocity) reinforces that identity rather than
                // fighting it.
                attitude: Some(Attitude::Defiance),
                // The tango's harmonic home is functional minor — the
                // raised leading tone gives V→i its bite.
                mode: Some(crate::scale::Mode::HarmonicMinor),
                meter: 4,
                tempo_range: (100.0, 132.0),
                // Dotted and syncopated identities: the melodic surface
                // carries the same snap the habanera puts underneath.
                motifs_calm: vec![
                    vec![(5, 3, 2), (4, 1, 2), (3, 2, 1)],
                    vec![(1, 3, 2), (7, 1, 2), (1, 2, 1)],
                    vec![(3, 2, 1), (2, 1, 1), (1, 1, 1)],
                ],
                motifs_medium: vec![
                    vec![(5, 1, 1), (5, 1, 2), (4, 1, 2), (3, 1, 1), (2, 1, 1)],
                    vec![
                        (1, 1, 2),
                        (2, 1, 2),
                        (3, 1, 1),
                        (4, 3, 4),
                        (5, 1, 4),
                        (5, 1, 1),
                    ],
                ],
                motifs_busy: vec![
                    vec![
                        (5, 1, 2),
                        (4, 1, 2),
                        (3, 1, 2),
                        (2, 1, 2),
                        (1, 3, 4),
                        (7, 1, 4),
                        (1, 1, 1),
                    ],
                    vec![
                        (3, 3, 4),
                        (3, 1, 4),
                        (2, 1, 2),
                        (1, 1, 2),
                        (7, 1, 1),
                        (5, 1, 1),
                    ],
                ],
                progression: ProgressionSpec::Archetype(vec![1, 4, 5, 1]),
                accompaniment_pool: vec![A::Habanera],
                form_pool: vec![FormKind::Ternary, FormKind::Rondo],
                ensemble_pool: s(&[["violin", "piano", "upright_bass"]]),
                counter_instrument: None,
                // Tango melodic DNA: the melody carries the dance's snap
                // itself — dotted calls, pickup snaps; wide reaches,
                // repeated-note insistence, descending tension.
                melody: MelodicDna {
                    hook_rhythms: vec![
                        vec![(3, 2), (1, 2), (1, 1)],
                        vec![(1, 2), (1, 1), (3, 2)],
                        vec![(1, 2), (1, 2), (1, 1), (1, 1)],
                    ],
                    hook_contours: vec![
                        vec![5, 1, 2],
                        vec![1, 5, 4],
                        vec![5, 5, 3],
                        vec![3, 3, 7],
                        vec![8, 5, 6, 5],
                    ],
                    appoggiatura_rate: 0.6, // dramatic — the review's word
                    ornament_rate: 0.0,
                    blue_note_rate: 0.0,
                    use_procedural_foundry: false,
                },
                // Statement — interruption — answer — sharp stop.
                meter_pool: vec![],
                mode_pool: vec![],
                rhetoric: PhraseRhetoric::Declamatory,
                development: DevelopmentDna::Sequential, // insistence through motion
                texture: {
                    let mut t = texture(DrumPolicy::None);
                    t.deceptive_close = true;
                    t
                },
            },
            Style::Celtic => CompositionSpec {
                name: "Celtic".into(),
                attitude: None,
                // Mixolydian, not Dorian (ModalFolk's already Dorian) —
                // the flat-seventh mode that gives the jig's bVII-I
                // "double tonic" close without any special-casing: degree
                // 7 of Mixolydian IS the flat seventh already.
                mode: Some(crate::scale::Mode::Mixolydian),
                meter: 6, // the jig's 3+3 lilt
                tempo_range: (110.0, 150.0),
                motifs_calm: vec![vec![(1, 3, 1), (5, 3, 1)], vec![(3, 3, 1), (1, 3, 1)]],
                motifs_medium: vec![
                    vec![
                        (1, 1, 1),
                        (2, 1, 1),
                        (3, 1, 1),
                        (4, 1, 1),
                        (5, 1, 1),
                        (1, 1, 1),
                    ],
                    vec![
                        (5, 1, 1),
                        (4, 1, 1),
                        (3, 1, 1),
                        (2, 1, 1),
                        (1, 1, 1),
                        (5, 1, 1),
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
                        (1, 1, 1),
                        (1, 1, 1),
                    ],
                    vec![
                        (5, 1, 2),
                        (6, 1, 2),
                        (5, 1, 2),
                        (4, 1, 2),
                        (3, 1, 2),
                        (2, 1, 2),
                        (1, 1, 1),
                        (1, 1, 1),
                        (1, 1, 1),
                    ],
                ],
                // I-IV-I-bVII-I-IV: no dominant function anywhere — the
                // modal "double tonic" real jigs and reels use instead of
                // a functional V-I close.
                progression: ProgressionSpec::Archetype(vec![1, 4, 1, 7, 1, 4]),
                accompaniment_pool: vec![A::JigGait],
                form_pool: vec![FormKind::Ternary, FormKind::Rondo],
                ensemble_pool: s(&[
                    ["violin", "harp", "cello"],
                    ["flute", "acoustic_guitar", "cello"],
                ]),
                counter_instrument: None,
                // Celtic melodic DNA: dance-tune turns — a leap with
                // recoil or a big closing reach, never a bare scale run —
                // decorated by cuts rather than leaned appoggiaturas.
                melody: MelodicDna {
                    hook_rhythms: vec![
                        vec![(1, 2), (1, 2), (1, 2), (3, 2)],
                        vec![(3, 2), (1, 2), (1, 2), (1, 1)],
                        vec![(1, 2), (1, 1), (1, 2), (1, 1)],
                        vec![(1, 1), (1, 2), (1, 2), (2, 1)],
                    ],
                    hook_contours: vec![
                        vec![1, 3, 2, 1],
                        vec![5, 3, 4, 5],
                        vec![1, 2, 1, 5],
                        vec![3, 1, 2, 3],
                    ],
                    appoggiatura_rate: 0.0, // a jig cuts, it doesn't lean
                    ornament_rate: 0.5,
                    blue_note_rate: 0.0,
                    use_procedural_foundry: false,
                },
                meter_pool: vec![],
                mode_pool: vec![],
                rhetoric: PhraseRhetoric::Classic,
                development: DevelopmentDna::Figural, // ornamental variation, the decorative practice
                texture: {
                    let mut t = texture(DrumPolicy::None);
                    t.drone = true;
                    t
                },
            },
            Style::Blues => CompositionSpec {
                name: "Blues".into(),
                attitude: None,
                mode: None, // stays major/Ionian — the blue notes do the coloring, not the mode
                meter: 4,
                tempo_range: (70.0, 110.0),
                motifs_calm: vec![
                    vec![(1, 1, 1), (3, 1, 1), (5, 1, 1), (4, 1, 1)],
                    vec![(5, 2, 1), (3, 1, 1), (1, 1, 1)],
                ],
                motifs_medium: vec![
                    vec![(1, 1, 2), (3, 1, 2), (5, 1, 1), (4, 1, 1), (3, 1, 1)],
                    vec![(5, 1, 1), (4, 1, 1), (3, 1, 1), (1, 1, 1)],
                ],
                motifs_busy: vec![
                    vec![
                        (1, 1, 2),
                        (3, 1, 2),
                        (4, 1, 2),
                        (5, 1, 2),
                        (4, 1, 2),
                        (3, 1, 2),
                        (1, 1, 1),
                    ],
                    vec![
                        (5, 1, 2),
                        (4, 1, 2),
                        (3, 1, 2),
                        (2, 1, 2),
                        (1, 1, 1),
                        (1, 1, 1),
                    ],
                ],
                // Three real, standard 12-bar-blues chord variants (purely
                // major-functional — the "blues" identity is entirely in
                // the melody's blue notes, not the chords). `ArchetypePool`
                // (not a single `Archetype`) so `realize_call_response`'s
                // already-computed per-chorus `seed_variant` — threaded
                // into `spec.progression(...)` since day one — actually
                // varies the harmony chorus to chorus, not just the
                // melody: a real gap where every chorus in a multi-chorus
                // piece got matched melodic variation but byte-identical
                // harmony every time.
                progression: ProgressionSpec::ArchetypePool(vec![
                    // Standard: I-I-I-I / IV-IV-I-I / V-IV-I-I.
                    vec![1, 1, 1, 1, 4, 4, 1, 1, 5, 4, 1, 1],
                    // Quick-change: bar 2 moves to IV and back.
                    vec![1, 4, 1, 1, 4, 4, 1, 1, 5, 4, 1, 1],
                    // Turnaround ending on V, setting up the next chorus.
                    vec![1, 1, 1, 1, 4, 4, 1, 1, 5, 4, 1, 5],
                ]),
                accompaniment_pool: vec![A::Shuffle],
                form_pool: vec![FormKind::Ternary, FormKind::Rondo],
                ensemble_pool: s(&[
                    ["saxophone", "electric_piano", "upright_bass"],
                    ["clarinet", "piano", "upright_bass"],
                ]),
                counter_instrument: None,
                // Blues melodic DNA: licks that linger on (or worry) the
                // third — the blue-note pass's favorite target.
                melody: MelodicDna {
                    hook_rhythms: vec![
                        vec![(1, 2), (1, 2), (1, 1), (1, 1)],
                        vec![(3, 2), (1, 2), (1, 1), (1, 1)],
                        vec![(1, 1), (1, 2), (1, 2), (2, 1)],
                        vec![(1, 2), (1, 1), (1, 2), (1, 1)],
                    ],
                    hook_contours: vec![
                        vec![3, 3, 1, 5], // worry the third, fall, reach
                        vec![1, 4, 3, 3], // reach, recoil, worry the third
                        vec![7, 5, 3, 3], // descend by thirds, worry the third
                        vec![1, 3, 1, 5], // leap, drop, reach
                    ],
                    appoggiatura_rate: 0.0,
                    ornament_rate: 0.0,
                    // The habit itself: melody tones on the third/seventh
                    // flatten at random, independent of the harmony.
                    blue_note_rate: 0.55,
                    use_procedural_foundry: false,
                },
                meter_pool: vec![],
                mode_pool: vec![],
                rhetoric: PhraseRhetoric::Classic,
                development: DevelopmentDna::Sequential, // the turnaround's descending schema
                texture: {
                    let mut t = texture(DrumPolicy::LightPulse);
                    t.swing = 0.63; // the shuffle's swung eighth
                    t
                },
            },
            Style::Impressionism => CompositionSpec {
                name: "Impressionism".into(),
                // This style's own progression already "avoids the
                // dominant entirely" (see below) -- Curiosity's unresolved
                // melodic ending is the same non-resolution idea applied to
                // the melody, not a new claim about the style.
                attitude: Some(Attitude::Curiosity),
                // The floating, non-resolving raised fourth — Debussy's
                // preferred mode when he wanted a passage that goes
                // nowhere in particular.
                mode: Some(crate::scale::Mode::Lydian),
                meter: 4,
                tempo_range: (52.0, 88.0),
                motifs_calm: vec![
                    vec![(1, 2, 1), (2, 1, 1), (3, 1, 1)],
                    vec![(5, 2, 1), (4, 1, 1), (3, 1, 1)],
                ],
                motifs_medium: vec![
                    vec![(1, 1, 1), (3, 1, 1), (4, 1, 1), (5, 1, 1)],
                    vec![(5, 1, 1), (4, 1, 1), (2, 1, 1), (1, 1, 1)],
                ],
                motifs_busy: vec![
                    vec![
                        (1, 1, 2),
                        (2, 1, 2),
                        (3, 1, 2),
                        (4, 1, 2),
                        (5, 1, 2),
                        (4, 1, 2),
                        (3, 1, 1),
                    ],
                    vec![
                        (5, 1, 2),
                        (6, 1, 2),
                        (5, 1, 2),
                        (4, 1, 2),
                        (3, 1, 2),
                        (2, 1, 2),
                        (1, 1, 1),
                    ],
                ],
                // No dominant, anywhere — I-IV-I-ii avoids the one chord
                // whose whole job is to pull toward resolution.
                progression: ProgressionSpec::Archetype(vec![1, 4, 1, 2]),
                accompaniment_pool: vec![A::Arpeggio], // the rippling, water-like texture
                form_pool: vec![FormKind::Ternary, FormKind::Rondo],
                ensemble_pool: s(&[["piano", "harp", "cello"], ["flute", "harp", "cello"]]),
                counter_instrument: None,
                // The identity here is harmonic, not a melodic hook — like
                // Lullaby, the classic shared pools serve it fine.
                melody: MelodicDna::default(),
                meter_pool: vec![],
                mode_pool: vec![],
                rhetoric: PhraseRhetoric::Classic,
                development: DevelopmentDna::Figural, // coloristic figuration, not argument
                texture: {
                    let mut t = texture(DrumPolicy::None);
                    t.damage = 0.0; // a wash of color has no argument to injure
                    t.planing = true;
                    t
                },
            },
            Style::SacredChoral => CompositionSpec {
                name: "SacredChoral".into(),
                attitude: None,
                // The ecclesiastical mode — the half-step-above-tonic
                // second degree gives the traditional "Phrygian cadence."
                mode: Some(crate::scale::Mode::Phrygian),
                meter: 4,
                tempo_range: (48.0, 72.0),
                motifs_calm: vec![
                    vec![(1, 1, 1), (2, 1, 1), (3, 1, 1), (2, 1, 1)],
                    vec![(5, 1, 1), (4, 1, 1), (3, 1, 1), (4, 1, 1)],
                ],
                motifs_medium: vec![
                    vec![(1, 2, 1), (2, 1, 1), (3, 1, 1)],
                    vec![(5, 2, 1), (4, 1, 1), (3, 1, 1)],
                ],
                motifs_busy: vec![
                    vec![(1, 1, 2), (2, 1, 2), (3, 1, 1), (2, 1, 1), (1, 1, 1)],
                    vec![(5, 1, 2), (4, 1, 2), (3, 1, 1), (4, 1, 1), (5, 1, 1)],
                ],
                // i-iv-i-bII-iv-i: the plagal iv-i motion twice, and the
                // Phrygian cadence's bII stepping down to i.
                progression: ProgressionSpec::Archetype(vec![1, 4, 1, 2, 4, 1]),
                accompaniment_pool: vec![A::Block], // homophonic block-chord hymn texture
                form_pool: vec![FormKind::Ternary],
                ensemble_pool: s(&[["organ", "organ", "cello"], ["organ", "bell", "cello"]]),
                counter_instrument: None,
                // The identity here is harmonic, not a melodic hook — like
                // Lullaby and Impressionism, the classic shared pools serve
                // it fine.
                melody: MelodicDna::default(),
                meter_pool: vec![],
                mode_pool: vec![],
                // A melody-only listening test (2026-07-24) found this
                // style collapsing into Nocturne's identity: both slow,
                // both stepwise, and this style had no rhetoric of its own
                // (same root cause as March's original gap). `Chorale`'s
                // held, unhurried, on-the-beat cadence is the opposite of
                // Nocturne's `Singing` (delayed, softened) on both axes.
                rhetoric: PhraseRhetoric::Chorale,
                development: DevelopmentDna::Classic, // homophonic, not imitative — nothing to develop
                texture: {
                    let mut t = texture(DrumPolicy::None);
                    t.damage = 0.0; // consonance stays pure; the suspensions ARE the tension
                    t.suspension_rate = 0.6;
                    t
                },
            },
            Style::Minimalism => CompositionSpec {
                name: "Minimalism".into(),
                attitude: None,
                mode: None, // plain major — the harmony stays still; the process is the interest
                meter: 4,
                tempo_range: (108.0, 144.0),
                motifs_calm: vec![
                    vec![(1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)],
                    vec![(5, 1, 1), (5, 1, 1), (5, 1, 1), (5, 1, 1)],
                ],
                motifs_medium: vec![
                    vec![
                        (1, 1, 2),
                        (3, 1, 2),
                        (5, 1, 2),
                        (3, 1, 2),
                        (1, 1, 1),
                        (1, 1, 1),
                    ],
                    vec![
                        (5, 1, 2),
                        (3, 1, 2),
                        (1, 1, 2),
                        (3, 1, 2),
                        (5, 1, 1),
                        (5, 1, 1),
                    ],
                ],
                motifs_busy: vec![
                    vec![
                        (1, 1, 2),
                        (2, 1, 2),
                        (3, 1, 2),
                        (2, 1, 2),
                        (1, 1, 2),
                        (2, 1, 2),
                        (3, 1, 2),
                        (2, 1, 2),
                    ],
                    vec![
                        (5, 1, 2),
                        (4, 1, 2),
                        (3, 1, 2),
                        (4, 1, 2),
                        (5, 1, 2),
                        (4, 1, 2),
                        (3, 1, 2),
                        (4, 1, 2),
                    ],
                ],
                // Mostly static tonic, with brief IV/V excursions — the
                // stasis functional harmony's job usually denies.
                progression: ProgressionSpec::Archetype(vec![1, 1, 1, 4, 1, 1, 1, 5]),
                accompaniment_pool: vec![A::Arpeggio], // the pulsing ostinato
                form_pool: vec![FormKind::Ternary],
                ensemble_pool: s(&[
                    ["marimba", "electric_piano", "upright_bass"],
                    ["piano", "marimba", "upright_bass"],
                ]),
                counter_instrument: None,
                // The additive process supplies the theme sections' real
                // identity; B still uses the classic shared pools.
                melody: MelodicDna::default(),
                meter_pool: vec![],
                mode_pool: vec![],
                rhetoric: PhraseRhetoric::Classic,
                development: DevelopmentDna::Classic, // the process IS the development
                texture: {
                    let mut t = texture(DrumPolicy::LightPulse);
                    t.damage = 0.0; // no argument to injure — the process is the whole piece
                    t.coda_bars = 0; // minimalist pieces don't resolve, they stop
                    t.additive_process = true;
                    t
                },
            },
            Style::JazzBallad => CompositionSpec {
                name: "JazzBallad".into(),
                attitude: None,
                mode: Some(crate::scale::Mode::Aeolian), // natural minor — no prior style had claimed it
                meter: 4,
                tempo_range: (52.0, 86.0),
                motifs_calm: vec![
                    vec![(1, 2, 1), (3, 1, 1), (5, 1, 1)],
                    vec![(5, 2, 1), (3, 1, 1), (1, 1, 1)],
                ],
                motifs_medium: vec![
                    vec![(1, 1, 1), (3, 1, 1), (5, 1, 1), (3, 1, 1)],
                    vec![(5, 1, 1), (3, 1, 1), (1, 1, 1), (3, 1, 1)],
                ],
                motifs_busy: vec![
                    vec![
                        (1, 1, 2),
                        (3, 1, 2),
                        (5, 1, 2),
                        (6, 1, 2),
                        (5, 1, 1),
                        (3, 1, 1),
                    ],
                    vec![
                        (5, 1, 2),
                        (3, 1, 2),
                        (1, 1, 2),
                        (7, 1, 2),
                        (1, 1, 1),
                        (3, 1, 1),
                    ],
                ],
                // ii-V-I-vi, twice: the turnaround cycle — with
                // seventh_chords on, this spells ii7-V7-i(maj7)-vi7.
                progression: ProgressionSpec::Archetype(vec![2, 5, 1, 6, 2, 5, 1, 5]),
                accompaniment_pool: vec![A::Block, A::Comp], // sustained voicings, or syncopated comping
                form_pool: vec![FormKind::Ternary],
                ensemble_pool: s(&[
                    ["saxophone", "piano", "upright_bass"],
                    ["clarinet", "electric_piano", "upright_bass"],
                ]),
                counter_instrument: None,
                // Compounds three earlier devices rather than adding only
                // new ones: blue notes (predicted reusable here when Blues
                // shipped) and appoggiaturas both switch on together for
                // the first time in one style.
                melody: MelodicDna {
                    hook_rhythms: vec![
                        vec![(3, 2), (1, 2), (1, 1), (1, 1)],
                        vec![(1, 2), (1, 2), (1, 1), (2, 1)],
                        vec![(1, 1), (1, 2), (1, 2), (2, 1)],
                        vec![(1, 2), (1, 1), (1, 2), (1, 1)],
                    ],
                    hook_contours: vec![
                        vec![5, 3, 5, 1], // dip a third, return, reach home
                        vec![1, 5, 3, 3], // big reach, drop, worry the third
                        vec![3, 3, 6, 5], // worry the third, reach the sixth's color, settle
                        vec![5, 1, 3, 3], // fall, leap up a third, worry it
                    ],
                    appoggiatura_rate: 0.35,
                    ornament_rate: 0.0,
                    blue_note_rate: 0.4,
                    use_procedural_foundry: false,
                },
                meter_pool: vec![],
                mode_pool: vec![],
                rhetoric: PhraseRhetoric::Singing, // the balladic quiet arrival, Nocturne's own device
                development: DevelopmentDna::Figural, // ornamental embellishment, jazz's own practice
                texture: {
                    let mut t = texture(DrumPolicy::None);
                    t.swing = 0.58; // a loping eighth, subtler than Blues' shuffle
                    t.seventh_chords = true;
                    t
                },
            },
            Style::BaroqueSuite => CompositionSpec {
                name: "BaroqueSuite".into(),
                attitude: None,
                mode: None, // functional common-practice tonality, not an exotic mode
                meter: 3,
                tempo_range: (54.0, 88.0),
                // Every template totals exactly 3 beats (one bar in 3/4) —
                // the calm bank walks in stately quarter notes, the busy
                // bank runs continuous eighths (6 * 1/2 = 3), both idioms
                // native to a Sarabande's continuo texture.
                motifs_calm: vec![
                    vec![(1, 1, 1), (2, 1, 1), (3, 1, 1)],
                    vec![(5, 1, 1), (4, 1, 1), (3, 1, 1)],
                ],
                motifs_medium: vec![
                    vec![(1, 1, 2), (2, 1, 2), (3, 1, 2), (2, 1, 2), (1, 1, 1)],
                    vec![(3, 1, 2), (4, 1, 2), (5, 1, 2), (4, 1, 2), (3, 1, 1)],
                ],
                motifs_busy: vec![
                    vec![
                        (1, 1, 2),
                        (2, 1, 2),
                        (3, 1, 2),
                        (4, 1, 2),
                        (5, 1, 2),
                        (4, 1, 2),
                    ],
                    vec![
                        (5, 1, 2),
                        (6, 1, 2),
                        (5, 1, 2),
                        (4, 1, 2),
                        (3, 1, 2),
                        (2, 1, 2),
                    ],
                ],
                // Harmonic-syntax pilot (2026-07-26, see
                // HARMONIC_SYNTAX_REWORK_SCOPE_2026-07-26.md): this style's
                // own doc already calls it "functional common-practice
                // tonality, not an exotic mode" -- the SAME tonal world
                // Classical's real T-PD-D-T generator lives in, with no
                // documented reason (unlike Cinematic/Folk/Playful/
                // ModalFolk/Impressionism/SacredChoral, all of which
                // deliberately avoid or restrict specific chords) for
                // staying on a fixed 4-chord loop instead of it.
                //
                // A/B evidence (2026-07-27, note-data + rendered-audio
                // analysis, not a human listening session): across 2 seeds,
                // same form/rhythm/instrumentation/tempo, the functional
                // route showed lower estimated dissonance, fewer large
                // melodic leaps, and more varied harmonic motion than
                // BAROQUE_SUITE_COMPATIBILITY_PROGRESSION (the route it
                // replaced, kept as a named constant -- a compatibility
                // baseline, an Analyst A/B control, and a regression
                // fixture, not deleted). One disclosed rough edge: seed 7's
                // counter-voice realization didn't fully adapt to the new
                // harmony (more angular leaps) -- a real follow-up target,
                // not a reason to keep the old route.
                progression: ProgressionSpec::GrammarWithPalette(
                    crate::harmony::HarmonicPalette::baroque(),
                ),
                accompaniment_pool: vec![A::Arpeggio], // broken-chord continuo
                // Diversity-review finding (2026-07-28, note-data + rendered-
                // audio analysis of 4 seeds): with a single-entry form_pool,
                // every seed produced the same macro-architecture (same
                // length/tempo/meter/texture/phrase layout), varying only
                // local harmony -- "the same template with different notes,"
                // not independently-conceived movements. `form_pool` already
                // exists precisely to let the seed choose a genuinely
                // different large-scale form (see `CompositionSpec::form_kind`);
                // it just had nothing to choose from. Each addition below is
                // a real, already-tested, historically-authentic Baroque
                // movement type, not a parameter variation on the same note
                // pipeline: `Rondo` (a rondeau, a genuine French-Baroque
                // dance-suite form) and `Variations` (theme-and-figuration,
                // e.g. a Handelian air with variations) both stay in the
                // shared period pipeline, so they're zero-risk; `Fugue`
                // (imitative counterpoint -- exposition, answer, stretto)
                // and `Passacaglia` (a ground-bass movement, e.g. a Purcell-
                // style lament) bypass that pipeline entirely for their own
                // dedicated realizers (`crate::fugue`/`crate::passacaglia`),
                // giving two seeds' worth of genuinely different note-
                // generation logic, not just different chords over the same
                // shape. `crate::fugue::realize_fugue`'s `meter: u8` is used
                // generically (no 4/4 assumption, verified against source);
                // `crate::passacaglia::realize_passacaglia`'s own tests
                // already exercise meter=3 directly. Sonata/Renaissance/Opera
                // (the other bypass forms) are deliberately left out -- their
                // tonal-conflict-and-resolution / equal-voice / dual-theme-
                // dialogue identities read as later or more specialized than
                // a Baroque suite movement, a stylistic judgment call, not a
                // technical constraint.
                form_pool: vec![
                    FormKind::Ternary,
                    FormKind::Rondo,
                    FormKind::Variations,
                    FormKind::Fugue,
                    FormKind::Passacaglia,
                ],
                ensemble_pool: s(&[["violin", "organ", "cello"]]), // trio-sonata continuo
                counter_instrument: None,
                melody: MelodicDna {
                    hook_rhythms: vec![
                        vec![(3, 4), (1, 4), (1, 1)],
                        vec![(1, 2), (1, 4), (1, 4), (1, 1)],
                    ],
                    hook_contours: vec![
                        vec![1, 3, 5, 4], // rise through the triad, one step back
                        vec![5, 4, 3, 1], // stepwise descent to rest
                    ],
                    // Baroque ornamentation is the melodic identity here:
                    // both grace-note "cuts" and accented appoggiaturas at
                    // once — trills and appoggiaturas were THE idiom's
                    // hallmark decoration.
                    appoggiatura_rate: 0.3,
                    ornament_rate: 0.3,
                    blue_note_rate: 0.0,
                    use_procedural_foundry: false,
                },
                meter_pool: vec![],
                mode_pool: vec![],
                rhetoric: PhraseRhetoric::Classic, // architectural regularity, not rubato
                development: DevelopmentDna::Sequential, // reinforces the harmonic sequence below
                texture: {
                    let mut t = texture(DrumPolicy::None);
                    t.swing = 0.5; // straight — the dance's lilt is metric, not swung
                    t.harmonic_sequence = true;
                    t
                },
            },
            Style::ProgFolk => CompositionSpec {
                name: "ProgFolk".into(),
                attitude: None,
                mode: None,
                meter: 4, // nominal/opening value — FormKind::ProgSuite moves through 4/7/5/4
                tempo_range: (92.0, 138.0),
                // The theme: an angular, pentatonic-flavored 4-bar idea
                // built to survive `contrasting_transform` twice (wide
                // leaps + a clear repeated-note anchor read even inverted
                // or retrograded). Every template totals exactly 4 beats
                // (this style's own nominal meter — `spec.motif()` picks
                // from these before `FormKind::ProgSuite` takes over).
                motifs_calm: vec![
                    vec![(1, 1, 1), (5, 1, 1), (6, 1, 1), (5, 1, 1)],
                    vec![(1, 1, 1), (3, 1, 1), (5, 1, 1), (6, 1, 1)],
                ],
                motifs_medium: vec![
                    vec![
                        (1, 1, 2),
                        (5, 1, 2),
                        (6, 1, 2),
                        (5, 1, 2),
                        (3, 1, 2),
                        (5, 1, 2),
                        (6, 1, 2),
                        (1, 1, 2),
                    ],
                    vec![
                        (1, 1, 2),
                        (3, 1, 2),
                        (5, 1, 2),
                        (6, 1, 2),
                        (5, 1, 2),
                        (3, 1, 2),
                        (1, 1, 2),
                        (1, 1, 2),
                    ],
                ],
                motifs_busy: vec![
                    vec![
                        (1, 1, 4),
                        (3, 1, 4),
                        (5, 1, 4),
                        (6, 1, 4),
                        (5, 1, 4),
                        (3, 1, 4),
                        (1, 1, 4),
                        (3, 1, 4),
                        (5, 1, 2),
                        (6, 1, 2),
                        (5, 1, 2),
                        (1, 1, 2),
                    ],
                    vec![
                        (1, 1, 4),
                        (5, 1, 4),
                        (6, 1, 4),
                        (5, 1, 4),
                        (3, 1, 4),
                        (1, 1, 4),
                        (3, 1, 4),
                        (5, 1, 4),
                        (6, 1, 2),
                        (5, 1, 2),
                        (3, 1, 2),
                        (1, 1, 2),
                    ],
                ],
                // Unused by the ProgSuite path (`realize_prog_suite`
                // generates each section's own progression internally,
                // exactly as the fugue/passacaglia family's presets
                // document theirs) but required for a valid spec.
                progression: ProgressionSpec::Archetype(vec![1, 5, 6, 4]),
                accompaniment_pool: vec![A::Comp],
                form_pool: vec![FormKind::ProgSuite],
                ensemble_pool: s(&[["saw_lead", "electric_piano", "upright_bass"]]),
                counter_instrument: None,
                melody: MelodicDna::default(),
                meter_pool: vec![],
                mode_pool: vec![],
                rhetoric: PhraseRhetoric::Classic,
                // Unused by the ProgSuite path, same as `progression` above
                // — the real per-section transformation is
                // `contrasting_transform` in `prog_suite.rs`, not this field.
                development: DevelopmentDna::Classic,
                texture: texture(DrumPolicy::Backbeat),
            },
            Style::Ambient => CompositionSpec {
                name: "Ambient".into(),
                attitude: None,
                mode: None,
                meter: 4,
                tempo_range: (32.0, 52.0),
                // Almost no events: calm is a single held tone per bar,
                // medium two half notes, and even busy — the ceiling of
                // this style's activity — never runs faster than quarter
                // notes. Every template totals exactly 4 beats.
                motifs_calm: vec![vec![(1, 4, 1)], vec![(5, 4, 1)]],
                motifs_medium: vec![vec![(1, 2, 1), (3, 2, 1)], vec![(5, 2, 1), (3, 2, 1)]],
                motifs_busy: vec![
                    vec![(1, 2, 1), (2, 1, 1), (3, 1, 1)],
                    vec![(5, 2, 1), (4, 1, 1), (3, 1, 1)],
                ],
                // A STATIC progression, deliberately repeated — the raw
                // material HARMONIC STASIS below turns into a real drone:
                // four bars of i, four bars of iv, each repeat a candidate
                // to tie into the previous bar's held tone instead of
                // re-striking.
                progression: ProgressionSpec::Archetype(vec![1, 1, 1, 1, 4, 4, 4, 4]),
                accompaniment_pool: vec![A::Block],
                form_pool: vec![FormKind::Ternary],
                ensemble_pool: s(&[["pad", "bell", "pad"]]),
                counter_instrument: None,
                melody: MelodicDna {
                    hook_rhythms: vec![],
                    hook_contours: vec![],
                    appoggiatura_rate: 0.0,
                    ornament_rate: 0.0,
                    blue_note_rate: 0.0,
                    use_procedural_foundry: false,
                },
                meter_pool: vec![],
                mode_pool: vec![],
                rhetoric: PhraseRhetoric::Classic,
                // Genuine gap, not a deliberate no-op like the ground-music/
                // ProgSuite bypass styles above: Ambient has no bespoke
                // composer module and goes through the shared Form pipeline
                // normally, so `Classic` really did mean zero development on
                // this style's departure sections. `Wandering` (undirected
                // ±1/±2-step drift, reversible) is the one development mode
                // that doesn't fight the identity — no directed argument
                // (Sequential), no rising arc (Intensifying), no ornamental
                // insistence (Figural/Fragmenting) — just the same
                // "stillness, not consequence" character `harmonic_stasis`
                // and `damage: 0.0` already commit to, applied to melody.
                development: DevelopmentDna::Wandering,
                texture: {
                    let mut t = texture(DrumPolicy::None);
                    t.harmonic_stasis = true;
                    t.damage = 0.0; // stillness, not consequence
                    t.coda_bars = 0; // ambient doesn't resolve, it stops
                    t.counter_melody = false; // a second voice would compete for the one thing there is to hear
                    t.cadential_harmonic_rhythm = false; // no accelerating into cadences — the harmonic rhythm stays flat
                    t.hook_cell = false; // no memorable identity cell — atmosphere, not a hook
                    t
                },
            },
            Style::Sonata => CompositionSpec {
                name: "Sonata".into(),
                attitude: None,
                mode: None, // functional common-practice tonality — the conflict/resolution IS the color
                meter: 4,
                tempo_range: (100.0, 152.0),
                // A clear, angular classical theme — built to survive
                // `contrasting_transform` into a real second subject
                // without losing its identity.
                motifs_calm: vec![
                    vec![(1, 1, 1), (3, 1, 1), (5, 1, 1), (3, 1, 1)],
                    vec![(5, 1, 1), (3, 1, 1), (1, 1, 1), (3, 1, 1)],
                ],
                motifs_medium: vec![
                    vec![
                        (1, 1, 2),
                        (2, 1, 2),
                        (3, 1, 2),
                        (5, 1, 2),
                        (4, 1, 2),
                        (3, 1, 2),
                        (1, 1, 1),
                    ],
                    vec![
                        (5, 1, 2),
                        (4, 1, 2),
                        (3, 1, 2),
                        (1, 1, 2),
                        (2, 1, 2),
                        (3, 1, 2),
                        (5, 1, 1),
                    ],
                ],
                motifs_busy: vec![
                    vec![
                        (1, 1, 4),
                        (2, 1, 4),
                        (3, 1, 4),
                        (4, 1, 4),
                        (5, 1, 4),
                        (4, 1, 4),
                        (3, 1, 4),
                        (2, 1, 4),
                        (1, 1, 2),
                        (3, 1, 2),
                        (5, 1, 2),
                        (3, 1, 2),
                    ],
                    vec![
                        (5, 1, 4),
                        (4, 1, 4),
                        (3, 1, 4),
                        (2, 1, 4),
                        (1, 1, 4),
                        (2, 1, 4),
                        (3, 1, 4),
                        (4, 1, 4),
                        (5, 1, 2),
                        (3, 1, 2),
                        (1, 1, 2),
                        (3, 1, 2),
                    ],
                ],
                // Unused by the Sonata path (`realize_sonata` generates
                // each section's own progression internally, exactly as
                // the fugue/passacaglia/prog-suite family's presets
                // document theirs) but required for a valid spec.
                progression: ProgressionSpec::Archetype(vec![1, 4, 5, 1]),
                accompaniment_pool: vec![A::Alberti], // the classical keyboard idiom
                form_pool: vec![FormKind::Sonata],
                ensemble_pool: s(&[["violin", "piano", "cello"]]),
                counter_instrument: None,
                melody: MelodicDna::default(),
                meter_pool: vec![],
                mode_pool: vec![],
                rhetoric: PhraseRhetoric::Classic,
                development: DevelopmentDna::Classic, // the form itself IS the development
                texture: texture(DrumPolicy::None),
            },
            Style::RenaissancePolyphony => CompositionSpec {
                name: "RenaissancePolyphony".into(),
                attitude: None,
                mode: Some(crate::scale::Mode::Dorian), // no leading tone to borrow — the mode's own color
                meter: 4,
                tempo_range: (66.0, 100.0),
                // Stepwise, conjunct points — vocal writing moves by
                // step, not leap. Every template totals exactly 4 beats.
                motifs_calm: vec![
                    vec![(1, 1, 1), (2, 1, 1), (3, 1, 1), (2, 1, 1)],
                    vec![(5, 1, 1), (4, 1, 1), (3, 1, 1), (4, 1, 1)],
                ],
                motifs_medium: vec![
                    vec![
                        (1, 1, 2),
                        (2, 1, 2),
                        (3, 1, 2),
                        (4, 1, 2),
                        (3, 1, 2),
                        (2, 1, 2),
                        (1, 1, 1),
                    ],
                    vec![
                        (5, 1, 2),
                        (4, 1, 2),
                        (3, 1, 2),
                        (2, 1, 2),
                        (3, 1, 2),
                        (4, 1, 2),
                        (5, 1, 1),
                    ],
                ],
                motifs_busy: vec![
                    vec![
                        (1, 1, 4),
                        (2, 1, 4),
                        (3, 1, 4),
                        (4, 1, 4),
                        (5, 1, 4),
                        (4, 1, 4),
                        (3, 1, 4),
                        (2, 1, 4),
                        (1, 1, 2),
                        (2, 1, 2),
                        (3, 1, 2),
                        (2, 1, 2),
                    ],
                    vec![
                        (5, 1, 4),
                        (4, 1, 4),
                        (3, 1, 4),
                        (2, 1, 4),
                        (1, 1, 4),
                        (2, 1, 4),
                        (3, 1, 4),
                        (4, 1, 4),
                        (5, 1, 2),
                        (4, 1, 2),
                        (3, 1, 2),
                        (4, 1, 2),
                    ],
                ],
                // Unused by the Renaissance path (`realize_renaissance`
                // builds the whole texture directly — three species-
                // fitted independent lines, no chordal accompaniment at
                // all) but required for a valid spec.
                progression: ProgressionSpec::Archetype(vec![1, 4, 5, 1]),
                accompaniment_pool: vec![A::Block],
                form_pool: vec![FormKind::Renaissance],
                ensemble_pool: s(&[["flute", "organ", "cello"]]), // soprano/alto/bass — organ substitutes for choir
                counter_instrument: None,
                melody: MelodicDna::default(),
                meter_pool: vec![],
                mode_pool: vec![],
                rhetoric: PhraseRhetoric::Classic,
                development: DevelopmentDna::Classic, // the form itself IS the polyphony
                texture: texture(DrumPolicy::None),
            },
            Style::AfroCuban => CompositionSpec {
                name: "AfroCuban".into(),
                attitude: None,
                // Dorian, not functional minor — the real idiom (Santana's
                // "Oye Como Va" is the textbook example: a two-chord i-IV
                // vamp in A Dorian, no raised leading tone anywhere).
                mode: Some(crate::scale::Mode::Dorian),
                meter: 4,
                tempo_range: (92.0, 128.0),
                // Melody durations shaped like the tresillo itself (3-3-2
                // as NOTE LENGTHS: dotted-quarter, dotted-quarter, quarter)
                // — the melody rhymes with the clave underneath it without
                // literally copying the accompaniment's onset math.
                motifs_calm: vec![
                    vec![(1, 3, 2), (5, 3, 2), (3, 1, 1)],
                    vec![(5, 3, 2), (1, 3, 2), (3, 1, 1)],
                    vec![(3, 3, 2), (1, 3, 2), (5, 1, 1)],
                ],
                motifs_medium: vec![
                    vec![(1, 1, 2), (3, 1, 2), (5, 3, 2), (3, 1, 2), (1, 1, 1)],
                    vec![(5, 1, 2), (3, 1, 2), (1, 3, 2), (3, 1, 2), (5, 1, 1)],
                ],
                motifs_busy: vec![
                    vec![
                        (1, 1, 4),
                        (2, 1, 4),
                        (3, 1, 4),
                        (4, 1, 4),
                        (5, 1, 4),
                        (4, 1, 4),
                        (3, 1, 4),
                        (2, 1, 4),
                        (5, 1, 2),
                        (3, 1, 2),
                        (1, 1, 1),
                    ],
                    vec![
                        (5, 1, 4),
                        (4, 1, 4),
                        (3, 1, 4),
                        (2, 1, 4),
                        (1, 1, 4),
                        (2, 1, 4),
                        (3, 1, 4),
                        (4, 1, 4),
                        (1, 1, 2),
                        (3, 1, 2),
                        (5, 1, 1),
                    ],
                ],
                // A real montuno vamp: a short cycle that REPEATS, not a
                // long-range progression — the groove is the point.
                progression: ProgressionSpec::Archetype(vec![1, 4, 5, 1]),
                accompaniment_pool: vec![A::Montuno],
                form_pool: vec![FormKind::Ternary, FormKind::Rondo],
                ensemble_pool: s(&[["saxophone", "piano", "upright_bass"]]),
                counter_instrument: None,
                melody: MelodicDna::default(),
                meter_pool: vec![],
                mode_pool: vec![],
                rhetoric: PhraseRhetoric::Classic,
                // A vamp doesn't develop — it repeats and grooves. The
                // clave/tumbao interlock carries the style's identity, not
                // a departure-section transformation.
                development: DevelopmentDna::Classic,
                texture: {
                    let mut t = texture(DrumPolicy::None);
                    // The hidden-bass-quote easter egg (`hook_cell`)
                    // overwrites a whole window of bass content with a
                    // generically-timed hook quote, ignoring whatever
                    // rhythm cell placed it there — for Montuno that means
                    // it can silently plant a bass onset back on top of
                    // one of the montuno's own stabs, undoing the interlock
                    // that IS this style's identity. Every other style can
                    // afford that easter egg; this one can't.
                    t.hook_cell = false;
                    t
                },
            },
            Style::Flamenco => CompositionSpec {
                name: "Flamenco".into(),
                attitude: None,
                mode: Some(crate::scale::Mode::Phrygian), // the engine's first Phrygian style
                meter: 12,
                tempo_range: (108.0, 156.0),
                // Cante-shaped: long declamatory notes, neighbor-tone
                // turns around the tonic, descending stepwise lines —
                // every tier sums to exactly the 12-beat compás.
                motifs_calm: vec![
                    vec![(5, 3, 1), (4, 1, 1), (3, 3, 1), (2, 1, 1), (1, 4, 1)],
                    vec![
                        (7, 2, 1),
                        (1, 1, 1),
                        (7, 1, 1),
                        (6, 2, 1),
                        (5, 1, 1),
                        (4, 1, 1),
                        (3, 4, 1),
                    ],
                    vec![
                        (3, 2, 1),
                        (2, 1, 1),
                        (1, 1, 1),
                        (2, 1, 1),
                        (1, 1, 1),
                        (7, 2, 1),
                        (1, 4, 1),
                    ],
                ],
                motifs_medium: vec![
                    vec![
                        (5, 1, 2),
                        (4, 1, 2),
                        (5, 1, 2),
                        (4, 1, 2),
                        (3, 1, 1),
                        (2, 1, 1),
                        (1, 1, 1),
                        (7, 1, 1),
                        (1, 1, 1),
                        (3, 5, 1),
                    ],
                    vec![
                        (1, 1, 2),
                        (7, 1, 2),
                        (1, 1, 2),
                        (7, 1, 2),
                        (6, 1, 1),
                        (5, 1, 1),
                        (4, 1, 1),
                        (3, 1, 1),
                        (2, 1, 1),
                        (1, 5, 1),
                    ],
                ],
                motifs_busy: vec![
                    vec![
                        (5, 1, 4),
                        (6, 1, 4),
                        (5, 1, 4),
                        (4, 1, 4),
                        (5, 1, 2),
                        (4, 1, 2),
                        (3, 1, 2),
                        (2, 1, 2),
                        (1, 1, 2),
                        (7, 1, 2),
                        (1, 1, 2),
                        (7, 1, 2),
                        (1, 1, 4),
                        (2, 1, 4),
                        (1, 1, 4),
                        (7, 1, 4),
                        (1, 1, 2),
                        (7, 1, 2),
                        (1, 5, 1),
                    ],
                    vec![
                        (1, 1, 4),
                        (7, 1, 4),
                        (1, 1, 4),
                        (2, 1, 4),
                        (3, 1, 2),
                        (4, 1, 2),
                        (5, 1, 2),
                        (6, 1, 2),
                        (7, 1, 2),
                        (1, 1, 2),
                        (7, 1, 2),
                        (6, 1, 2),
                        (5, 1, 4),
                        (4, 1, 4),
                        (3, 1, 4),
                        (2, 1, 4),
                        (1, 1, 2),
                        (2, 1, 2),
                        (1, 5, 1),
                    ],
                ],
                // The Andalusian cadence: iv-III-II-i, a descending
                // STEPWISE tetrachord (4,3,2,1) rather than the fifths-
                // based motion every other style's progression uses — one
                // chord per compás, so the harmony changes once per cycle.
                progression: ProgressionSpec::Archetype(vec![4, 3, 2, 1]),
                accompaniment_pool: vec![A::CompasGait],
                form_pool: vec![FormKind::Ternary, FormKind::Rondo],
                ensemble_pool: s(&[["oud", "acoustic_guitar", "cello"]]),
                counter_instrument: None,
                melody: MelodicDna {
                    hook_rhythms: vec![
                        vec![(3, 1), (1, 1), (4, 1)],
                        vec![(2, 1), (2, 1), (1, 1), (1, 1)],
                    ],
                    hook_contours: vec![
                        vec![5, 4, 3, 2, 1],
                        vec![1, 7, 1, 6, 5],
                        vec![3, 2, 1, 2, 1],
                    ],
                    appoggiatura_rate: 0.55, // dramatic, cante-like leans into arrivals
                    ornament_rate: 0.5,      // melismatic turns — the highest in the roster
                    blue_note_rate: 0.0,
                    use_procedural_foundry: false,
                },
                meter_pool: vec![],
                mode_pool: vec![],
                rhetoric: PhraseRhetoric::Declamatory, // statement — interruption — answer
                // Builds toward a remate (the dramatic close), not a
                // formal transformation — the same register/figuration/
                // velocity arc Cinematic uses, in service of a different
                // feeling.
                development: DevelopmentDna::Intensifying,
                texture: {
                    let mut t = texture(DrumPolicy::None);
                    // Same reasoning as AfroCuban: the hidden-bass-quote
                    // easter egg is oblivious to the compás and would
                    // silently overwrite the anchor's grounding role.
                    t.hook_cell = false;
                    t
                },
            },
            Style::BossaNova => CompositionSpec {
                name: "BossaNova".into(),
                attitude: None,
                mode: None, // functional major, jazz color comes from the 7ths not the mode
                meter: 4,
                tempo_range: (96.0, 132.0),
                // Relaxed, laid-back phrasing — a single syncopated push
                // per line, not the dense run Flamenco's cante uses.
                motifs_calm: vec![
                    vec![(1, 1, 1), (3, 1, 2), (5, 1, 2), (3, 2, 1)],
                    vec![(5, 1, 1), (3, 1, 2), (1, 1, 2), (3, 2, 1)],
                    vec![(3, 3, 2), (1, 1, 2), (5, 2, 1)],
                ],
                motifs_medium: vec![
                    vec![
                        (1, 1, 2),
                        (3, 1, 2),
                        (5, 1, 1),
                        (3, 1, 2),
                        (2, 1, 2),
                        (1, 1, 1),
                    ],
                    vec![
                        (5, 1, 2),
                        (3, 1, 2),
                        (1, 1, 1),
                        (2, 1, 2),
                        (3, 1, 2),
                        (5, 1, 1),
                    ],
                ],
                motifs_busy: vec![
                    vec![
                        (1, 1, 4),
                        (2, 1, 4),
                        (3, 1, 2),
                        (5, 1, 2),
                        (4, 1, 2),
                        (3, 1, 2),
                        (2, 1, 4),
                        (1, 1, 4),
                        (1, 1, 1),
                    ],
                    vec![
                        (5, 1, 4),
                        (4, 1, 4),
                        (3, 1, 2),
                        (1, 1, 2),
                        (2, 1, 2),
                        (3, 1, 2),
                        (4, 1, 4),
                        (5, 1, 4),
                        (5, 1, 1),
                    ],
                ],
                // I-vi-ii-V: the most famous four-chord turnaround in jazz
                // and bossa alike.
                progression: ProgressionSpec::Archetype(vec![1, 6, 2, 5]),
                accompaniment_pool: vec![A::BossaComp],
                form_pool: vec![FormKind::Ternary, FormKind::Rondo],
                ensemble_pool: s(&[["saxophone", "acoustic_guitar", "upright_bass"]]),
                counter_instrument: None,
                melody: MelodicDna::default(),
                meter_pool: vec![],
                mode_pool: vec![],
                // Leans into arrivals, overstays, resolves quietly — the
                // relaxed opposite of Flamenco's interruption-and-answer.
                rhetoric: PhraseRhetoric::Singing,
                // A gentle, ongoing variation, not a dramatic arc — the
                // "understated" identity extends to how it develops too.
                development: DevelopmentDna::Figural,
                texture: {
                    let mut t = texture(DrumPolicy::None);
                    // 7ths on every chord, not just the cadential one — the
                    // extended jazz vocabulary that makes the harmony feel
                    // like it's floating rather than resolving.
                    t.seventh_chords = true;
                    // Same reasoning as AfroCuban/Flamenco: hook_cell's
                    // hidden-bass-quote is oblivious to bossa's chained
                    // hold-then-anticipate shape and would stab a random
                    // note into the middle of a bar meant to never have one.
                    t.hook_cell = false;
                    t
                },
            },
            Style::Opera => CompositionSpec {
                name: "Opera".into(),
                attitude: None,
                mode: None, // functional major — the drama is structural (two themes), not modal
                meter: 4,
                tempo_range: (66.0, 112.0),
                // Unused by the Opera path (`realize_opera` builds its own
                // Theme A/Theme B material directly — the whole point is
                // that they're two UNRELATED ideas, not variants of a
                // single spec-picked motif) but required for a valid spec.
                motifs_calm: vec![vec![(1, 1, 1), (3, 1, 1), (5, 1, 1), (8, 1, 1)]],
                motifs_medium: vec![vec![
                    (1, 1, 2),
                    (3, 1, 2),
                    (5, 1, 1),
                    (3, 1, 2),
                    (2, 1, 2),
                    (1, 1, 1),
                ]],
                motifs_busy: vec![vec![
                    (1, 1, 4),
                    (2, 1, 4),
                    (3, 1, 4),
                    (4, 1, 4),
                    (5, 1, 2),
                    (4, 1, 2),
                    (3, 1, 1),
                    (1, 1, 1),
                ]],
                // Also unused — `realize_opera` generates each section's
                // own progression internally, exactly as the sonata/
                // prog-suite family's presets document theirs.
                progression: ProgressionSpec::Archetype(vec![1, 4, 5, 1]),
                accompaniment_pool: vec![A::Block],
                form_pool: vec![FormKind::Opera],
                ensemble_pool: s(&[["violin", "harp", "cello"]]),
                // Theme B's own voice: `contrast_counter` (the live audio
                // path's actual mechanism — `counter_instrument` only
                // feeds the separate MIDI-export/piano-roll path) picks
                // the first of Clarinet/Cello/Flute that isn't already the
                // melody or bass, which resolves to Clarinet here — a
                // genuinely different, warm reed voice against Theme A's
                // bowed strings.
                counter_instrument: None,
                melody: MelodicDna::default(),
                meter_pool: vec![],
                mode_pool: vec![],
                rhetoric: PhraseRhetoric::Classic,
                development: DevelopmentDna::Classic, // the dialogue structure itself IS the development
                texture: texture(DrumPolicy::None),
            },
            Style::IrishTraditional => CompositionSpec {
                name: "IrishTraditional".into(),
                attitude: None,
                // Dorian, not Celtic's Mixolydian — a genuinely different
                // modal color for the two "Irish-flavored" styles to share
                // the roster without sounding like variants of one idea.
                mode: Some(crate::scale::Mode::Dorian),
                meter: 4, // the REEL — Celtic already owns the jig's 6/8 lilt
                tempo_range: (112.0, 148.0), // brisk, driving session tempo
                // Reel character: a continuous straight-eighth stream, with
                // a few longer notes placed specifically to give the roll
                // ornament something to decorate.
                motifs_calm: vec![vec![(1, 1, 2), (2, 1, 2), (3, 1, 1), (5, 1, 1), (3, 1, 1)]],
                motifs_medium: vec![vec![
                    (1, 1, 2),
                    (2, 1, 2),
                    (3, 1, 2),
                    (4, 1, 2),
                    (5, 1, 1),
                    (3, 1, 1),
                ]],
                motifs_busy: vec![vec![
                    (1, 1, 2),
                    (2, 1, 2),
                    (3, 1, 2),
                    (4, 1, 2),
                    (5, 1, 2),
                    (4, 1, 2),
                    (3, 1, 1),
                ]],
                progression: ProgressionSpec::Archetype(vec![1, 4, 1, 5]),
                // A driving eighth-note pulse under the tune — the reel's
                // engine-room, staying out of the roll's way.
                accompaniment_pool: vec![A::Arpeggio],
                form_pool: vec![FormKind::Ternary, FormKind::Rondo],
                // A real session trio: flute over guitar, upright bass —
                // distinct from Celtic's fiddle/harp/cello soundtrack cast.
                ensemble_pool: s(&[["flute", "acoustic_guitar", "upright_bass"]]),
                counter_instrument: None,
                melody: MelodicDna {
                    ornament_rate: 0.4, // rolls decorate a real minority of notes, not every one
                    appoggiatura_rate: 0.0,
                    blue_note_rate: 0.0,
                    use_procedural_foundry: false,
                    ..MelodicDna::default()
                },
                meter_pool: vec![],
                mode_pool: vec![],
                rhetoric: PhraseRhetoric::Classic,
                // A tune repeats its strains; it doesn't develop them —
                // the same reasoning AfroCuban's vamp used.
                development: DevelopmentDna::Classic,
                texture: {
                    let mut t = texture(DrumPolicy::None);
                    // The engine's first ORNAMENT CHAIN (a 5-note roll)
                    // instead of Celtic's single-grace cut.
                    t.roll_ornaments = true;
                    t
                },
            },
            Style::HindustaniInspired => CompositionSpec {
                name: "HindustaniInspired".into(),
                attitude: None,
                // Dorian — reused (composition needs a stable 7-degree
                // tonic triad; a true pentatonic mode fails validation),
                // but genuinely apt here: it's a fair Western analogue of
                // Kafi thaat, one of the most common Hindustani parent
                // scales, not an arbitrary substitute.
                mode: Some(crate::scale::Mode::Dorian),
                meter: 4,
                tempo_range: (56.0, 92.0), // unhurried — the melody has room to breathe
                // Long, sustained lines centered on the tonic — minimal
                // leaps, real room for the roll/grace pass to decorate.
                motifs_calm: vec![vec![(1, 3, 1), (2, 1, 1)]],
                motifs_medium: vec![vec![(1, 2, 1), (2, 1, 1), (3, 1, 1)]],
                motifs_busy: vec![vec![(1, 1, 1), (3, 1, 1), (5, 1, 1), (3, 1, 1)]],
                // Unused — `apply_full_drone` replaces Harmony (and Bass)
                // entirely with a static tonic-fifth pad, so no chord
                // progression is ever realized — but required for a valid
                // spec, same reasoning the bypass-form presets document.
                progression: ProgressionSpec::Archetype(vec![1, 1, 1, 1]),
                accompaniment_pool: vec![A::Block], // superseded by the drone pass
                form_pool: vec![FormKind::Ternary, FormKind::Rondo],
                ensemble_pool: s(&[["sitar", "pad", "upright_bass"]]),
                counter_instrument: None,
                melody: MelodicDna {
                    ornament_rate: 0.35, // grace notes as the melody's real language, not decoration
                    appoggiatura_rate: 0.2,
                    blue_note_rate: 0.0,
                    use_procedural_foundry: false,
                    ..MelodicDna::default()
                },
                meter_pool: vec![],
                mode_pool: vec![],
                rhetoric: PhraseRhetoric::Singing, // leans into arrivals, overstays, resolves quietly
                // The register/figuration/velocity arc reused for a long
                // melodic evolution rather than a dramatic climax — a
                // raga's own "getting denser and higher over time" shape.
                development: DevelopmentDna::Intensifying,
                texture: {
                    let mut t = texture(DrumPolicy::None);
                    // The engine's first FULL drone: no chord progression
                    // survives anywhere, not even under a moving harmony
                    // (Celtic's plain `drone` still lets Harmony move).
                    t.full_drone = true;
                    t
                },
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

    /// RATCHET: how many styles ship with NO melodic DNA of their own.
    ///
    /// Measured 2026-07-30: **15 of 29** styles have `hook_contours == []` and
    /// `hook_rhythms == []`, so `HookCell::generate_with` falls back to the same
    /// shared `CONTOURS`/`RHYTHMS` defaults for all of them — Lullaby, Fugue,
    /// Passacaglia, Impressionism, SacredChoral, Minimalism, ProgFolk, Ambient,
    /// Sonata, RenaissancePolyphony, AfroCuban, BossaNova, Opera,
    /// IrishTraditional, HindustaniInspired.
    ///
    /// This is the structural cause of the 99.8% mean canonical duplicate rate
    /// `motif_foundry_style_survey` reports: over half the catalogue draws its
    /// hooks from one identical pool. A style with no melodic DNA is not
    /// expressing a design choice — it is unfinished.
    ///
    /// CORRECTION (2026-07-31), from actually authoring one: this ratchet's
    /// premise that empty DNA is "the structural cause of the 99.8% duplicate
    /// rate" is WRONG. Giving Lullaby its own DNA left its duplicate rate at
    /// 99.7% — unchanged — and made its FOUNDRY rate worse (76.2% -> 99.6%),
    /// because a lullaby's authentic vocabulary is narrower than the generic
    /// shared pool it replaced. Authoring DNA buys style IDENTITY, not variety.
    /// The duplicate rate is driven by pool SIZE (<=24 cells), which is a
    /// separate problem. This ratchet is still worth keeping — a style drawing
    /// its tune from a generic pool is genuinely unfinished — but it should not
    /// be sold as a variety fix.
    ///
    /// It also bounds what enabling `use_procedural_foundry` can achieve.
    /// `motif_foundry::config_for_dna` biases generation toward a DNA's own
    /// measured character, so for these 15 it has nothing to bias with: all
    /// fifteen score an identical 76.2% foundry duplicate rate with an identical
    /// (3,5) length range and max leap 5. Blanket-enabling the foundry would
    /// give them procedurally-varied but mutually IDENTICAL melodies — the same
    /// trap the harmonic-syntax batch hit on 2026-07-30, where converting styles
    /// to a style-agnostic generator made them bit-identical to each other (see
    /// `HARMONIC_SYNTAX_REWORK_SCOPE_2026-07-26.md`'s blocked-attempt note).
    ///
    /// The ratchet may only go DOWN. Authoring real melodic DNA for one of these
    /// styles is a genuine, taste-dependent improvement; this test makes the gap
    /// visible and stops it silently growing.
    /// Which styles' `progression` field is INERT, and therefore must not be
    /// "improved" — pinned so the set cannot drift unnoticed.
    #[test]
    fn styles_whose_progression_field_is_dead_are_known_and_documented() {
        // A style whose every form bypasses the period pipeline
        // (`FormKind::uses_progression_pipeline`) never consumes
        // `CompositionSpec::progression` at all: fugue, passacaglia, sonata,
        // renaissance, opera and prog-suite each build their own harmony
        // internally. For those styles the progression field is INERT — its
        // value cannot affect a single composed note.
        //
        // Measured 2026-07-30. This matters because the field still holds a
        // plausible-looking value (`Fugue => Archetype([1, 4, 5, 1])`), so a
        // reader naturally assumes it is the fugue's harmony. It is not, and
        // "converting" any of these to a grammar palette would be pure theatre
        // — the decorative-wiring pattern this codebase keeps having to catch.
        const KNOWN_DEAD: [Style; 6] = [
            Style::Fugue,
            Style::Passacaglia,
            Style::ProgFolk,
            Style::Sonata,
            Style::RenaissancePolyphony,
            Style::Opera,
        ];
        let dead: Vec<Style> = Style::ALL
            .iter()
            .copied()
            .filter(|s| {
                let pool = &s.spec().form_pool;
                !pool.is_empty() && !pool.iter().any(|f| f.uses_progression_pipeline())
            })
            .collect();
        assert_eq!(
            dead,
            KNOWN_DEAD.to_vec(),
            "the set of styles with an inert `progression` field changed.\n\
             If a style GAINED one: its progression value has been ignored all along and is \
             probably untuned — check it before trusting it.\n\
             If a style LOST one: its progression is now live and its value suddenly matters."
        );
    }

    /// `Style::progression` must stay a thin delegate to the spec.
    ///
    /// It used to hardcode five styles that duplicated their own specs, making
    /// it a second progression source that could silently diverge — change a
    /// style's spec and this method would keep returning the old progression,
    /// while `live.rs` (a real consumer) kept using it. The arms were provably
    /// redundant when removed (0 divergences over every style x 5 bar-lengths x
    /// 24 seeds); this stops them growing back.
    #[test]
    fn style_progression_delegates_to_the_spec() {
        for s in Style::ALL {
            for bars in [1usize, 3, 4, 8, 12] {
                for seed in 0..16u64 {
                    assert_eq!(
                        s.progression(bars, seed).degrees,
                        s.spec().progression(bars, seed).degrees,
                        "{s:?} bars={bars} seed={seed}: Style::progression diverged from its \
                         own spec — the legacy method must not special-case styles"
                    );
                }
            }
        }
    }

    /// GUARD, added 2026-07-30 with `HarmonicPalette`.
    ///
    /// Converting a style to `ProgressionSpec::Grammar` gives it real per-seed
    /// variety and simultaneously makes it produce BIT-IDENTICAL progressions to
    /// every other Grammar style, because `Progression::generate` is
    /// style-agnostic. Measured before the fix: Classical / BaroqueSuite / March
    /// / Nocturne agreed on 96/96 style-pairs × seeds. That is the cross-style
    /// bleed the diversity census fixed for Folk/Cinematic/Playful, reintroduced
    /// at 100%, and it is what blocked the harmonic-syntax rework.
    ///
    /// The fix is `ProgressionSpec::GrammarWithPalette`, which supplies the
    /// style's own vocabulary to the same universal grammar. After it, March
    /// agrees with Classical on 1/32 seeds instead of 32/32.
    ///
    /// This test asserts BOTH halves of the trade, because either alone is a
    /// regression: styles must not be interchangeable, AND each must still be
    /// varied within itself.
    #[test]
    fn grammar_generated_styles_must_not_be_harmonically_interchangeable() {
        use std::collections::BTreeSet;
        // Every style whose progressions come from the grammar generator.
        let generated: Vec<Style> = Style::ALL
            .iter()
            .copied()
            .filter(|s| {
                matches!(
                    s.spec().progression,
                    crate::spec::ProgressionSpec::Grammar
                        | crate::spec::ProgressionSpec::GrammarWithPalette(_)
                )
            })
            .collect();

        for (i, a) in generated.iter().enumerate() {
            for b in &generated[i + 1..] {
                let identical = (0..32u64)
                    .filter(|&seed| {
                        a.spec().progression(8, seed).degrees
                            == b.spec().progression(8, seed).degrees
                    })
                    .count();
                assert!(
                    identical < 24,
                    "{a:?} and {b:?} produce identical progressions on {identical}/32 seeds — \
                     they are harmonically interchangeable. Give one a HarmonicPalette; \
                     ProgressionSpec::Grammar alone is style-agnostic, so every style using it \
                     generates the SAME walk from the same seed."
                );
            }
        }

        // And each must still be varied WITHIN itself — the thing converting
        // away from a fixed archetype was supposed to buy.
        for s in &generated {
            let distinct: BTreeSet<Vec<i32>> = (0..32u64)
                .map(|seed| s.spec().progression(8, seed).degrees.clone())
                .collect();
            assert!(
                distinct.len() >= 3,
                "{s:?} generates only {} distinct progression(s) across 32 seeds — barely \
                 better than the fixed archetype it replaced",
                distinct.len()
            );
        }
    }

    #[test]
    fn styles_without_their_own_melodic_dna_may_only_decrease() {
        const KNOWN_EMPTY_MELODIC_DNA: usize = 14;
        let empty: Vec<String> = Style::ALL
            .iter()
            .filter(|s| {
                let d = &s.spec().melody;
                d.hook_contours.is_empty() && d.hook_rhythms.is_empty()
            })
            .map(|s| format!("{s:?}"))
            .collect();
        assert!(
            empty.len() <= KNOWN_EMPTY_MELODIC_DNA,
            "styles with no melodic DNA rose from {KNOWN_EMPTY_MELODIC_DNA} to {}: {empty:?}. \
             A new style must bring its own hook_contours/hook_rhythms, or it will draw from \
             the same shared default pool as every other DNA-less style.",
            empty.len()
        );
        if empty.len() < KNOWN_EMPTY_MELODIC_DNA {
            panic!(
                "GOOD NEWS, update the ratchet: only {} styles now lack melodic DNA (was \
                 {KNOWN_EMPTY_MELODIC_DNA}). Lower KNOWN_EMPTY_MELODIC_DNA to {} to lock the \
                 gain in. Remaining: {empty:?}",
                empty.len(),
                empty.len()
            );
        }
    }

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
            Style::Passacaglia,
            Style::Tango,
            Style::Celtic,
            Style::Blues,
            Style::Impressionism,
            Style::SacredChoral,
            Style::Minimalism,
            Style::JazzBallad,
            Style::BaroqueSuite,
            Style::ProgFolk,
            Style::Ambient,
            Style::Sonata,
            Style::RenaissancePolyphony,
            Style::AfroCuban,
            Style::Flamenco,
            Style::BossaNova,
            Style::Opera,
            Style::IrishTraditional,
            Style::HindustaniInspired,
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

    /// The lowest nonzero seed whose `CompositionSpec::accompaniment` selects
    /// `Alberti` from `Style::Classical`'s accompaniment pool.
    /// `accompaniment()` decorrelates from the raw seed internally
    /// (`seed / 2`), so which seed does this is not a small hardcoded
    /// number — search instead of assuming one.
    fn a_seed_that_selects_alberti() -> u64 {
        let spec = Style::Classical.spec();
        (1..1000)
            .find(|&seed| spec.accompaniment(seed) == crate::accompaniment::Accompaniment::Alberti)
            .expect("at least one seed in 1..1000 must select Alberti")
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
        // NOTE the absence of any pivot-window exemption here: the
        // modulation announcement is rendered THROUGH the accompaniment
        // pattern (re-pitched in place), so the oom-pah survives its own
        // key change — this test proves it.
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
        // Classical, some seed → Alberti: BODY harmony becomes moving
        // eighth notes, not one held block per bar (the coda's held chords
        // are the deliberate exception, as above).
        let alberti = compose_styled(
            &MusicalIntent {
                seed: a_seed_that_selects_alberti(),
                ..Default::default()
            },
            Style::Classical,
        );
        let alberti_body = alberti.total_beats.beats()
            - (crate::composer::CODA_BARS * alberti.meter as i64) as f64;
        // No pivot-window exemption: the announcement re-pitches the
        // Alberti eighths in place, so the figure survives the modulation.
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
    fn non_classical_progressions_start_on_the_tonic_or_a_documented_alternative() {
        // Folk and Cinematic both draw from a 2-member seed-varied pool
        // (fixed post-census cross-bleed fix, see `progression()`'s doc
        // comment) whose members ALL start on I; Playful's pool members all
        // start on ii (its ii-centric identity marker); Waltz's Pachelbel
        // canon always starts on I. Checked across several seeds, not one,
        // since these are no longer single hardcoded archetypes.
        for seed in 0..8u64 {
            assert_eq!(
                Style::Waltz.progression(8, seed).degrees[0],
                1,
                "seed={seed}"
            );
            assert_eq!(
                Style::Folk.progression(8, seed).degrees[0],
                1,
                "seed={seed}"
            );
            assert_eq!(
                Style::Playful.progression(8, seed).degrees[0],
                2,
                "seed={seed}"
            );
            assert_eq!(
                Style::Cinematic.progression(8, seed).degrees[0],
                1,
                "seed={seed}"
            );
        }
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

    #[test]
    fn supported_intent_axes_matches_the_causal_diversity_validation_census() {
        // Ground truth from examples/diversity_census.rs's one-factor-at-a-
        // time sensitivity matrix (26 of 174 style/parameter cells dead).
        // A default (unlisted) style supports all four axes.
        assert_eq!(
            Style::Classical.supported_intent_axes(),
            &["valence", "arousal", "energy", "bars"]
        );
        // Mode-pinned styles: only valence is dead.
        for style in [
            Style::ModalFolk,
            Style::Tango,
            Style::Celtic,
            Style::Impressionism,
            Style::SacredChoral,
            Style::JazzBallad,
            Style::AfroCuban,
            Style::Flamenco,
            Style::IrishTraditional,
            Style::HindustaniInspired,
        ] {
            assert_eq!(
                style.supported_intent_axes(),
                &["arousal", "energy", "bars"],
                "{style:?} should have only valence dead"
            );
        }
        // Bypass-grammar styles without a pinned mode: energy+bars dead.
        for style in [Style::Fugue, Style::ProgFolk, Style::Sonata, Style::Opera] {
            assert_eq!(
                style.supported_intent_axes(),
                &["valence", "arousal"],
                "{style:?} should have energy+bars dead"
            );
        }
        // Bypass-grammar styles that ALSO pin a mode: all four axes dead.
        for style in [Style::Passacaglia, Style::RenaissancePolyphony] {
            assert_eq!(
                style.supported_intent_axes(),
                &[] as &[&str],
                "{style:?} should have all four intent axes dead"
            );
        }
    }

    #[test]
    fn grammar_family_covers_every_style_exactly_once() {
        use crate::grammar::GrammarFamily;
        use std::collections::HashMap;
        let mut counts: HashMap<GrammarFamily, usize> = HashMap::new();
        for style in Style::ALL {
            *counts.entry(style.grammar_family()).or_insert(0) += 1;
        }
        // Only the fugue engine's hardcoded family; every non-flagship style
        // must NOT collide with a family whose `compose_with_grammar_plan`
        // arm hardcodes a *different* dedicated engine.
        assert_eq!(counts[&GrammarFamily::Contrapuntal], 1, "only Fugue");
        assert_eq!(counts[&GrammarFamily::GrooveCycle], 1, "only AfroCuban");
        assert_eq!(
            counts[&GrammarFamily::ProcessAdditive],
            1,
            "only Minimalism"
        );
        assert_eq!(
            counts[&GrammarFamily::RagaModalArc],
            1,
            "only HindustaniInspired"
        );
    }

    #[test]
    fn style_all_has_no_duplicates() {
        use std::collections::HashSet;
        let unique: HashSet<Style> = Style::ALL.into_iter().collect();
        assert_eq!(unique.len(), Style::ALL.len());
    }

    #[test]
    fn attitude_assignments_are_deliberate_and_narrow() {
        // Each of these 4 reinforces a fact the style's OWN doc comment
        // already states (see style.rs's doc comments for each) -- not an
        // invented mood. Every other style stays `None` on purpose:
        // Passacaglia specifically must NOT get one, because
        // `erosion_ending_for`'s `None` branch gives real per-seed ending
        // variety (Recovery/Acceptance/Elegy) that any `Some(attitude)`
        // would collapse to a single fixed choice.
        assert_eq!(Style::Playful.spec().attitude, Some(Attitude::Joy));
        assert_eq!(Style::ModalFolk.spec().attitude, Some(Attitude::Curiosity));
        assert_eq!(Style::Tango.spec().attitude, Some(Attitude::Defiance));
        assert_eq!(
            Style::Impressionism.spec().attitude,
            Some(Attitude::Curiosity)
        );
        assert_eq!(
            Style::Passacaglia.spec().attitude,
            None,
            "would silently collapse erosion_ending_for's seed-driven variety"
        );
    }

    #[test]
    fn flagship_styles_report_the_dedicated_engine_family() {
        use crate::grammar::GrammarFamily;
        assert_eq!(
            Style::AfroCuban.grammar_family(),
            GrammarFamily::GrooveCycle
        );
        assert_eq!(
            Style::Minimalism.grammar_family(),
            GrammarFamily::ProcessAdditive
        );
        assert_eq!(
            Style::HindustaniInspired.grammar_family(),
            GrammarFamily::RagaModalArc
        );
        assert_eq!(Style::Fugue.grammar_family(), GrammarFamily::Contrapuntal);
    }

    #[test]
    fn baroque_suite_form_pool_gives_the_seed_genuinely_different_premises() {
        // 2026-07-28 diversity-review fix: a single-entry form_pool meant
        // every seed produced the same macro-architecture ("the same
        // template with different notes"). Ground truth this test locks
        // in: across a real seed range, `form_kind` actually resolves to
        // more than one FormKind, AND each of those FormKinds composes a
        // valid, non-empty score under BaroqueSuite's own preset (key,
        // tempo, meter, motif bank) -- not just in each engine's own
        // dedicated test fixture.
        use crate::composer::{MusicalIntent, compose_with_spec};
        use crate::spec::FormKind;

        let spec = Style::BaroqueSuite.spec();
        assert!(
            spec.form_pool.len() >= 4,
            "expected several real movement premises, found {:?}",
            spec.form_pool
        );

        // FormKind derives Eq but not Hash, so track seen kinds in a Vec.
        let mut seen: Vec<FormKind> = Vec::new();
        for seed in 0..spec.form_pool.len() as u64 * 3 {
            let kind = spec.form_kind(seed);
            if !seen.contains(&kind) {
                seen.push(kind);
            }
            let intent = MusicalIntent {
                seed,
                ..MusicalIntent::default()
            };
            let score = compose_with_spec(&intent, &spec);
            assert!(
                !score.notes.is_empty(),
                "seed {seed} (form {kind:?}) composed an empty score"
            );
        }
        assert_eq!(
            seen.len(),
            spec.form_pool.len(),
            "expected every pool entry to be reachable within {} seeds, saw {:?}",
            spec.form_pool.len() * 3,
            seen
        );
    }
}
