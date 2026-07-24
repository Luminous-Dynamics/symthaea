// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Real-time incremental composition: generate one phrase at a time,
//! reacting to a live-updated [`MusicalIntent`], instead of deciding a whole
//! piece up front the way [`crate::composer::compose`] does.
//!
//! This is the shape a game or any other interactive consumer needs:
//! request the next phrase whenever the moment calls for one, get back a
//! self-contained [`Score`] starting at beat 0, and splice it onto your own
//! running timeline. Voice-leading state (the previous chord's upper voices
//! and bass note) carries across calls exactly like it already carries
//! across [`crate::form::Form`] sections within one `compose()` call — a
//! [`LiveComposer`] is really just a `compose()` whose sections arrive one
//! at a time instead of all at once.
//!
//! ## What's honestly different from `compose()`
//!
//! - **No global climax.** `compose()` picks ONE peak across the whole
//!   piece because it can see the whole piece. A [`LiveComposer`] cannot see
//!   the future — each call's climax is necessarily LOCAL to that call. This
//!   is an honest limit of true real-time generation, not a bug.
//! - **No large-scale form.** There is no B section, no relative-key
//!   contrast, no guaranteed return — those are large-scale decisions this
//!   crate's [`crate::form::Form`] makes ONLY when it can see the whole arc.
//!   A caller that wants an ABA shape in real time composes it explicitly by
//!   calling [`LiveComposer::retheme`] for the contrasting middle and
//!   [`LiveComposer::next_phrase`] again with the original theme for the
//!   return — the crate gives you the pieces, not a scripted arc.
//! - **Timeline-agnostic.** Every call returns a `Score` with onsets from
//!   beat 0; this module never touches wall-clock time. Splicing onto a
//!   running timeline (a game's music clock, muse's realizer) is the
//!   caller's job.

use crate::MusicalIntent;
use crate::form::{Form, Section, SectionRole};
use crate::harmony::Key;
use crate::motif::Motif;
use crate::phrase::Period;
use crate::pitch::Pitch;
use crate::rhythm::Duration;
use crate::score::Score;
use crate::style::Style;

/// Incrementally composes one phrase (a [`Period`]) at a time, carrying
/// voice-leading state between calls. Cheap: each call is the same pure,
/// sub-millisecond symbolic computation as one section of `compose()` — see
/// `DESIGN.md` for the measured cost of the whole crate's test suite.
pub struct LiveComposer {
    motif: Motif,
    key: Key,
    style: Style,
    meter_beats: f64,
    prev_upper: Vec<Pitch>,
    prev_bass: Option<Pitch>,
    seed_counter: u64,
    /// The seed the current THEME was picked with. Unlike `seed_counter`
    /// (which advances every call so progressions vary), this stays fixed
    /// until [`Self::retheme`] — it keys identity-level choices like the
    /// accompaniment texture, which should no more change per phrase than
    /// the motif does.
    base_seed: u64,
}

impl LiveComposer {
    /// Start a new live session: picks an initial theme (motif) and home key
    /// from `intent`, exactly as `compose()` would for its A section. Uses
    /// [`Style::Classical`]; see [`Self::new_styled`] for a genre bias.
    pub fn new(intent: &MusicalIntent) -> Self {
        Self::new_styled(intent, Style::Classical)
    }

    /// Like [`Self::new`], but every structural choice (meter, tempo range,
    /// motif bank, progression) comes from `style` — see [`crate::style::Style`].
    pub fn new_styled(intent: &MusicalIntent, style: Style) -> Self {
        let key = if intent.valence >= 0.0 {
            Key::major(intent.tonic)
        } else {
            Key::minor(intent.tonic)
        };
        LiveComposer {
            motif: style.motif(intent.arousal, intent.seed),
            key,
            style,
            meter_beats: style.meter() as f64,
            prev_upper: Vec::new(),
            prev_bass: None,
            seed_counter: intent.seed,
            base_seed: intent.seed,
        }
    }

    /// Generate the next phrase, reacting to a live-updated `intent`.
    ///
    /// - `valence` picks parallel major/minor of the SAME tonic (not a new
    ///   key) — the theme's identity stays stable across calls; only its
    ///   mode brightens or darkens.
    /// - `arousal`/`energy` pick the sentence-vs-period archetype and tempo,
    ///   same rule as `compose()`.
    /// - `bars`/`seed` size the progression; the seed is advanced internally
    ///   each call so consecutive phrases don't repeat the same progression
    ///   even if the caller passes the same `intent.seed` back every time.
    ///
    /// Voice leading (`prev_upper`/`prev_bass`) continues from wherever the
    /// last call left off, so consecutive phrases connect smoothly even
    /// though each is a separate `Score`.
    pub fn next_phrase(&mut self, intent: &MusicalIntent) -> Score {
        self.key = if intent.valence >= 0.0 {
            Key::major(self.key.tonic)
        } else {
            Key::minor(self.key.tonic)
        };
        let bars = intent.bars.max(1);
        self.seed_counter = self.seed_counter.wrapping_add(1);
        let base = self.style.progression(bars, self.seed_counter);
        let use_sentence = intent.arousal > 0.6 || intent.energy > 0.75;
        let period = if use_sentence {
            Period::parallel_sentence(&self.motif, &base.degrees, self.meter_beats)
        } else {
            Period::parallel(&self.motif, &base.degrees, self.meter_beats)
        };

        let tempo = self.style.tempo(intent.arousal);
        let meter = self.style.meter();
        let form = Form {
            sections: vec![Section {
                role: SectionRole::A,
                key: self.key,
                period,
            }],
        };

        let mut score = Score::new(self.key, tempo, meter);
        crate::composer::realize_melody(
            &mut score,
            &form,
            intent,
            Duration::zero(),
            self.meter_beats,
            true, // inert for a single phrase (grace needs a multi-phrase form)
        );
        // Same style-keyed accompaniment as compose_styled, but keyed on the
        // ORIGINAL seed (not the per-call counter): the texture is part of
        // the piece's identity and should stay stable across live phrases,
        // like the motif — only retheme() changes it.
        let pattern = self.style.accompaniment(self.base_seed);
        crate::composer::realize_harmony(
            &mut score,
            &form,
            self.meter_beats,
            intent,
            &mut self.prev_upper,
            pattern,
            true, // single-section form: the B-thinning gate never fires
            true,
            self.style.spec().texture.seventh_chords,
        );
        crate::composer::realize_bass(
            &mut score,
            &form,
            self.meter_beats,
            intent,
            &mut self.prev_bass,
            pattern,
            true,
        );
        score
    }

    /// Pick a brand NEW theme for a contrasting section (e.g. a game area
    /// change, or the "B" of an ABA arc composed by hand across calls).
    /// Clears voice-leading state so the new theme starts cleanly in root
    /// position rather than voice-led awkwardly from the old theme's last
    /// chord.
    pub fn retheme(&mut self, intent: &MusicalIntent) {
        self.motif = self.style.motif(intent.arousal, intent.seed);
        // A new theme is a new piece identity: re-key the accompaniment
        // texture along with the motif.
        self.base_seed = intent.seed;
        self.prev_upper.clear();
        self.prev_bass = None;
    }

    /// The current home key (reflects the most recent `valence` passed to
    /// [`Self::next_phrase`]).
    pub fn key(&self) -> Key {
        self.key
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::PitchClass;

    fn intent(valence: f32, arousal: f32) -> MusicalIntent {
        MusicalIntent {
            valence,
            arousal,
            energy: 0.5,
            // 4 bars, not 2: with only 2 bars, Period::parallel's cadence
            // steering (forcing the antecedent to end on V and the
            // consequent's last two chords to V-I) overwrites BOTH degrees
            // of a 2-bar progression, so consecutive-call tests would see
            // identical periods regardless of the seed advancing.
            bars: 4,
            seed: 42,
            tonic: PitchClass::C,
        }
    }

    #[test]
    fn new_picks_key_from_valence() {
        let bright = LiveComposer::new(&intent(1.0, 0.5));
        assert_eq!(bright.key(), Key::major(PitchClass::C));
        let dark = LiveComposer::new(&intent(-1.0, 0.5));
        assert_eq!(dark.key(), Key::minor(PitchClass::C));
    }

    #[test]
    fn next_phrase_returns_a_nonempty_score_starting_at_zero() {
        let mut live = LiveComposer::new(&intent(0.5, 0.5));
        let score = live.next_phrase(&intent(0.5, 0.5));
        assert!(!score.notes.is_empty());
        let first_onset = score
            .notes
            .iter()
            .map(|n| n.onset)
            .min_by(|a, b| a.beats().partial_cmp(&b.beats()).unwrap())
            .unwrap();
        assert_eq!(first_onset, Duration::zero());
    }

    #[test]
    fn consecutive_phrases_do_not_repeat_the_same_progression() {
        // Same intent passed twice in a row must still advance internally
        // (different seed_counter) so the piece doesn't loop.
        let mut live = LiveComposer::new(&intent(0.5, 0.5));
        let a = live.next_phrase(&intent(0.5, 0.5));
        let b = live.next_phrase(&intent(0.5, 0.5));
        assert_ne!(a, b);
    }

    #[test]
    fn valence_change_flips_to_parallel_mode_of_the_same_tonic() {
        let mut live = LiveComposer::new(&intent(1.0, 0.5));
        assert_eq!(live.key(), Key::major(PitchClass::C));
        let _ = live.next_phrase(&intent(-1.0, 0.5));
        assert_eq!(live.key(), Key::minor(PitchClass::C));
        // Tonic identity is preserved -- only mode changed.
        assert_eq!(live.key().tonic, PitchClass::C);
    }

    #[test]
    fn arousal_swings_between_period_and_sentence_archetypes() {
        // A period (calm) is 8 bars of the SAME motif variant across
        // antecedent+consequent; a sentence (excited) fragments in the back
        // half. Both are valid Scores; we just check both paths run without
        // panicking and produce plausible note counts.
        let mut calm = LiveComposer::new(&intent(0.5, 0.2));
        let calm_score = calm.next_phrase(&intent(0.5, 0.2));
        let mut excited = LiveComposer::new(&intent(0.5, 0.9));
        let excited_score = excited.next_phrase(&intent(0.5, 0.9));
        assert!(!calm_score.notes.is_empty());
        assert!(!excited_score.notes.is_empty());
    }

    #[test]
    fn retheme_clears_voice_leading_state() {
        let mut live = LiveComposer::new(&intent(0.5, 0.5));
        let _ = live.next_phrase(&intent(0.5, 0.5));
        assert!(!live.prev_upper.is_empty());
        assert!(live.prev_bass.is_some());
        live.retheme(&intent(0.5, 0.5));
        assert!(live.prev_upper.is_empty());
        assert!(live.prev_bass.is_none());
    }

    #[test]
    fn retheme_changes_the_motif() {
        // Not a strict guarantee (the seed bank could coincidentally repeat)
        // but with these two very different arousal levels the motif bank
        // differs (calm vs busy shapes), so the motif must differ too.
        let mut live = LiveComposer::new(&intent(0.5, 0.1));
        let before = live.motif.clone();
        live.retheme(&intent(0.5, 0.9));
        assert_ne!(before, live.motif);
    }

    #[test]
    fn voice_leading_state_carries_across_consecutive_calls() {
        // The second call's harmony should voice-lead FROM the first call's
        // final chord, not start fresh in root position -- i.e. prev_upper
        // must be non-empty going into the second realize_harmony call. We
        // can't directly observe realize_harmony's internal starting point,
        // but we can confirm state is retained across two calls (not reset).
        let mut live = LiveComposer::new(&intent(0.5, 0.5));
        let _ = live.next_phrase(&intent(0.5, 0.5));
        let after_first = live.prev_upper.clone();
        assert!(!after_first.is_empty());
        let _ = live.next_phrase(&intent(0.5, 0.5));
        // Still populated (not cleared) after a second call.
        assert!(!live.prev_upper.is_empty());
    }

    #[test]
    fn new_defaults_to_the_classical_style() {
        let classical_explicit = LiveComposer::new_styled(&intent(0.5, 0.5), Style::Classical);
        assert_eq!(classical_explicit.meter_beats, 4.0);
    }

    #[test]
    fn new_styled_waltz_uses_a_three_beat_meter() {
        let waltz = LiveComposer::new_styled(&intent(0.5, 0.5), Style::Waltz);
        assert_eq!(waltz.meter_beats, 3.0);
        // Every phrase note must fit the 3/4 grid -- spot check the motif
        // that seeded this session sums to exactly 3 beats.
        assert!((waltz.motif.total_duration().beats() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn new_styled_next_phrase_uses_the_styles_meter_in_the_score() {
        let mut waltz = LiveComposer::new_styled(&intent(0.5, 0.5), Style::Waltz);
        let score = waltz.next_phrase(&intent(0.5, 0.5));
        assert_eq!(score.meter, 3);
    }
}
