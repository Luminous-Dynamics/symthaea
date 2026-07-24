// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Form (Layer 3): chain multiple phrase-pairs into a real PIECE with a
//! departure and a return, instead of stopping after one period.
//!
//! Until now `compose()` produced exactly one antecedent+consequent period —
//! a single idea, stated once, was the whole piece. Real music has a larger
//! arc: TERNARY (ABA) form contrasts a middle section — a different key, a
//! transformed motif — against the opening, then RETURNS to it. That
//! departure-and-return is itself a large-scale structural narrative a
//! listener can follow, over and above any single phrase's question-and-
//! answer (Layer 2) or harmonic cadence (Layer 1).

use crate::harmony::{Key, Progression};
use crate::motif::{Motif, MotifNote};
use crate::phrase::Period;
use crate::rhythm::Duration;
use serde::{Deserialize, Serialize};

/// Pick a duration-preserving contrasting transformation of `motif`, keyed
/// by `choice` (mod 3). All three options preserve the motif's rhythm
/// exactly (only pitch direction/order changes), so they're safe drop-ins
/// anywhere a single motif statement is expected — three real classical
/// motivic transformations (inversion, retrograde, retrograde-inversion),
/// not just "always invert." Augmentation/diminution/sequence are
/// deliberately excluded here: they change total duration, which would
/// desync a per-measure development slot from the underlying harmonic
/// rhythm (one chord per measure). `pub(crate)` so [`crate::composer`] and
/// [`crate::style`] can reuse it to multiply motif-bank variety (an
/// existing bank template x 3 orientations, instead of hand-authoring more
/// templates for the same effect).
pub(crate) fn contrasting_transform(motif: &Motif, pivot: i32, choice: u64) -> Motif {
    match choice % 3 {
        0 => motif.invert(pivot),
        1 => motif.retrograde(),
        _ => motif.invert(pivot).retrograde(),
    }
}

/// Apply one of FOUR orientations to `motif`, keyed by `choice` (mod 4):
/// the motif AS-IS, or one of the three [`contrasting_transform`] variants.
/// Used to multiply a small hand-picked motif bank's effective variety
/// (one bank entry x four orientations) instead of hand-authoring more
/// entries for the same effect — every orientation preserves the motif's
/// total duration exactly, so it's always a safe drop-in wherever the
/// unmodified motif would have been used.
pub(crate) fn oriented(motif: &Motif, choice: u64) -> Motif {
    if choice.is_multiple_of(4) {
        return motif.clone();
    }
    let pivot = motif.notes.iter().find_map(|x| x.degree).unwrap_or(1);
    contrasting_transform(motif, pivot, choice - 1)
}

/// Classical "division" figuration: split each long note of `motif` into
/// two halves — the original degree, then a single connecting step — the
/// oldest variation technique there is (Renaissance divisions, Mozart's
/// K.265 first variation). The connecting tone steps one degree TOWARD the
/// next note (a passing tone) when the line is moving, or dips to the lower
/// neighbor when it isn't; `seed` flips stationary neighbors upper/lower per
/// index so two variation sets don't ornament identically. Total duration is
/// preserved EXACTLY (each split is `d -> d/2 + d/2`), so the figured motif
/// is a safe drop-in over the same harmonic rhythm — which is the whole
/// point: a variation keeps the theme's skeleton and changes its surface.
/// Only notes of at least a quarter (1 beat) are divided — halving eighths
/// into sixteenth-note chatter buries the theme instead of decorating it —
/// and the FINAL note is always left whole so the cadence stays stable.
pub(crate) fn figuration_variation(motif: &Motif, seed: u64) -> Motif {
    let notes = &motif.notes;
    let mut out = Vec::with_capacity(notes.len() * 2);
    for (i, n) in notes.iter().enumerate() {
        let is_last = i + 1 == notes.len();
        // num >= den <=> duration >= 1 beat (durations are normalized
        // positive rationals).
        let divisible = n.duration.num() >= n.duration.den();
        let Some(d) = n.degree else {
            out.push(*n); // rests pass through untouched
            continue;
        };
        if is_last || !divisible {
            out.push(*n);
            continue;
        }
        let half = n.duration.scale(1, 2);
        let next_degree = notes[i + 1..].iter().find_map(|x| x.degree);
        let connecting = match next_degree {
            Some(nd) if nd > d => d + 1,
            Some(nd) if nd < d => d - 1,
            // Stationary (repeated tone or nothing pitched ahead): a
            // neighbor tone, direction flipped per index by the seed.
            _ => {
                if (seed.wrapping_add(i as u64)).is_multiple_of(2) {
                    d - 1
                } else {
                    d + 1
                }
            }
        };
        out.push(MotifNote::new(d, half));
        out.push(MotifNote::new(connecting, half));
    }
    Motif { notes: out }
}

/// Which role a section plays in the form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SectionRole {
    /// The opening idea, in the home key.
    A,
    /// The contrasting middle: a related key and a transformed motif.
    B,
    /// The return of A — same material, back in the home key. This is what
    /// makes the form feel COMPLETE rather than merely stopping.
    ReturnA,
    /// [`Form::rondo`] only: a SECOND contrasting section, distinct from B
    /// (different key relationship, different motif transformation) —
    /// otherwise a rondo would just be ternary with an extra repeat.
    C,
}

impl SectionRole {
    /// A long-range dynamic-intensity multiplier for this section's role in
    /// the piece's overall arc — A establishes (calmer), B departs (the
    /// first real tension), ReturnA settles (a resolution, not a fade —
    /// still confident, just not straining), and C (rondo only) is the
    /// piece's genuine PEAK: the furthest harmonic departure (parallel key,
    /// not just relative) gets the highest intensity. Without this, a
    /// piece's per-phrase dynamic swells (see `realize_melody`) never add
    /// up to a large-scale shape — every section would breathe the same
    /// way regardless of its structural role.
    pub fn intensity(self) -> f32 {
        match self {
            SectionRole::A => 0.85,
            SectionRole::B => 1.0,
            SectionRole::ReturnA => 0.95,
            SectionRole::C => 1.15,
        }
    }
}

/// One section of a larger form: a period realized in a specific key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Section {
    pub role: SectionRole,
    pub key: Key,
    pub period: Period,
}

/// A full multi-section form.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Form {
    pub sections: Vec<Section>,
}

/// Extra per-piece context threaded into [`Form::ternary`]/[`Form::rondo`]/
/// [`Form::variations`] so their sections can honor a style's own declared
/// grammar instead of always falling back to generic classical functional
/// harmony. `None` (every pre-existing internal and test caller) preserves
/// today's exact behavior byte-for-byte.
///
/// Bundled into one struct rather than two independent `Option`s so the two
/// facts stay in sync: a `HarmonicSyntax` without the `spec` that declared
/// it (or vice versa) isn't a state this API can represent.
#[derive(Clone, Copy)]
pub struct FormGrammarContext<'a> {
    /// Drives cadence resolution (see [`crate::phrase::Period::parallel_in_for_grammar`]).
    pub harmony: crate::grammar::HarmonicSyntax,
    /// The style's own progression source — [`Form::ternary`]/[`Form::rondo`]'s
    /// contrasting B/C sections use THIS (with a section-specific seed) to
    /// generate their "fresh" progression instead of the generic classical
    /// functional grammar, so a 12-bar-blues piece's departure section stays
    /// in blues harmony instead of borrowing Classical's.
    pub spec: &'a crate::spec::CompositionSpec,
}

impl Form {
    /// Build a ternary (ABA) form:
    ///
    /// - **A**: `motif` over `progression`, in `home_key`.
    /// - **B**: a duration-preserving TRANSFORMATION of the same motif
    ///   (inversion, retrograde, or retrograde-inversion — chosen by `seed`,
    ///   see [`contrasting_transform`]; a contrasting character built from
    ///   the SAME idea — a "monothematic" ternary form, very common in real
    ///   practice, e.g. many minuet-and-trio movements) over a freshly
    ///   generated progression, in `home_key`'s RELATIVE key. Relative keys
    ///   share the same diatonic pitch classes, so this is a genuine
    ///   modulation needing no chromatic alteration.
    /// - **ReturnA**: A again, note-for-note, back in the home key — the
    ///   return that completes the arc. A listener who has just heard a
    ///   contrasting section and recognizes the opening coming back is
    ///   experiencing exactly the "departure and return" that makes a piece
    ///   feel like a piece, not a fragment.
    ///
    /// `seed` derives BOTH the B section's fresh progression AND which of
    /// the three contrasting transformations it uses — so two pieces with
    /// different seeds don't just modulate differently, they develop the
    /// idea differently too. The whole form stays deterministic for a given
    /// input.
    pub fn ternary(
        motif: &Motif,
        home_key: Key,
        progression: &Progression,
        meter: f64,
        seed: u64,
        use_sentence: bool,
        grammar: Option<FormGrammarContext>,
    ) -> Self {
        let bars = progression.degrees.len().max(1);
        // Each section cadences in ITS OWN key's grammar: a modal home key
        // closes ♭VII→i (see `Key::cadence_dominant_degree`) while its
        // functional B key keeps V→I — unless `grammar` declares a
        // non-functional syntax, in which case the archetype's own natural
        // close is honored instead (see `Period::parallel_in_for_grammar`).
        let harmony = grammar.map(|g| g.harmony);
        let build = |m: &Motif, prog: &[i32], key: Key| {
            let dominant = key.cadence_dominant_degree();
            match (use_sentence, harmony) {
                (true, Some(h)) => {
                    Period::parallel_sentence_in_for_grammar(m, prog, meter, dominant, h)
                }
                (true, None) => Period::parallel_sentence_in(m, prog, meter, dominant),
                (false, Some(h)) => Period::parallel_in_for_grammar(m, prog, meter, dominant, h),
                (false, None) => Period::parallel_in(m, prog, meter, dominant),
            }
        };

        let a_period = build(motif, &progression.degrees, home_key);

        let b_key = home_key.relative();
        let pivot = motif.notes.iter().find_map(|x| x.degree).unwrap_or(1);
        let b_motif = contrasting_transform(motif, pivot, seed);
        // The B section's "fresh" progression: a style with its own declared
        // harmonic vocabulary (the common case — see `FormGrammarContext`'s
        // doc) generates it from THAT vocabulary instead of borrowing
        // generic classical functional harmony, so the departure section
        // still sounds like the same piece.
        let b_progression = match grammar {
            Some(g) => g.spec.progression(bars, seed ^ 0x5EC7_104B),
            None => Progression::generate(bars, seed ^ 0x5EC7_104B),
        };
        let b_period = build(&b_motif, &b_progression.degrees, b_key);

        Form {
            sections: vec![
                Section {
                    role: SectionRole::A,
                    key: home_key,
                    period: a_period.clone(),
                },
                Section {
                    role: SectionRole::B,
                    key: b_key,
                    period: b_period,
                },
                Section {
                    role: SectionRole::ReturnA,
                    key: home_key,
                    period: a_period,
                },
            ],
        }
    }

    /// Build a RONDO (ABACA) form: the opening idea returns not once but
    /// TWICE, each time framing a DIFFERENT contrasting episode —
    /// structurally distinct from ternary (which has only one departure),
    /// and one of the most common real forms (movements, game/exploration
    /// music that returns to a "home" theme between areas).
    ///
    /// - **A**: `motif` over `progression`, in `home_key` (identical to
    ///   ternary's A).
    /// - **B**: a contrasting transformation of the motif (see
    ///   [`contrasting_transform`], same seed-chosen mechanism as ternary's
    ///   B) in `home_key`'s RELATIVE key.
    /// - **ReturnA**: A verbatim.
    /// - **C**: a DIFFERENT contrasting transformation from whichever one B
    ///   used (guaranteed distinct — see the implementation — so the two
    ///   episodes are always transformed differently from each other, not
    ///   just modulated differently) in `home_key`'s PARALLEL key
    ///   ([`Key::parallel`] — same tonic, opposite mode, a different kind of
    ///   modulation from B's relative-key move).
    /// - **ReturnA**: A verbatim again — completing the second, larger arc.
    ///
    /// `seed` derives B's and C's fresh progressions (XORed with different
    /// constants so they don't coincide) AND which transformation each
    /// uses; the whole form stays deterministic for a given input.
    pub fn rondo(
        motif: &Motif,
        home_key: Key,
        progression: &Progression,
        meter: f64,
        seed: u64,
        use_sentence: bool,
        grammar: Option<FormGrammarContext>,
    ) -> Self {
        // Per-section cadence grammar, exactly as in `ternary`.
        let harmony = grammar.map(|g| g.harmony);
        let build = |m: &Motif, prog: &[i32], key: Key| {
            let dominant = key.cadence_dominant_degree();
            match (use_sentence, harmony) {
                (true, Some(h)) => {
                    Period::parallel_sentence_in_for_grammar(m, prog, meter, dominant, h)
                }
                (true, None) => Period::parallel_sentence_in(m, prog, meter, dominant),
                (false, Some(h)) => Period::parallel_in_for_grammar(m, prog, meter, dominant, h),
                (false, None) => Period::parallel_in(m, prog, meter, dominant),
            }
        };
        let bars = progression.degrees.len().max(1);

        let a_period = build(motif, &progression.degrees, home_key);

        let b_key = home_key.relative();
        let pivot = motif.notes.iter().find_map(|x| x.degree).unwrap_or(1);
        let b_choice = seed % 3;
        // Guaranteed different from b_choice (mod 3, +1 always differs) so
        // the two episodes are never transformed identically.
        let c_choice = (b_choice + 1) % 3;
        let b_motif = contrasting_transform(motif, pivot, b_choice);
        // Same "honor the style's own vocabulary" reasoning as `ternary`'s B
        // section — see `FormGrammarContext`'s doc comment.
        let b_progression = match grammar {
            Some(g) => g.spec.progression(bars, seed ^ 0x5EC7_104B),
            None => Progression::generate(bars, seed ^ 0x5EC7_104B),
        };
        let b_period = build(&b_motif, &b_progression.degrees, b_key);

        let c_key = home_key.parallel();
        let c_motif = contrasting_transform(motif, pivot, c_choice);
        let c_progression = match grammar {
            Some(g) => g.spec.progression(bars, seed ^ 0xC0DA_15E5),
            None => Progression::generate(bars, seed ^ 0xC0DA_15E5),
        };
        let c_period = build(&c_motif, &c_progression.degrees, c_key);

        Form {
            sections: vec![
                Section {
                    role: SectionRole::A,
                    key: home_key,
                    period: a_period.clone(),
                },
                Section {
                    role: SectionRole::B,
                    key: b_key,
                    period: b_period,
                },
                Section {
                    role: SectionRole::ReturnA,
                    key: home_key,
                    period: a_period.clone(),
                },
                Section {
                    role: SectionRole::C,
                    key: c_key,
                    period: c_period,
                },
                Section {
                    role: SectionRole::ReturnA,
                    key: home_key,
                    period: a_period,
                },
            ],
        }
    }

    /// Build a THEME AND VARIATIONS form — variation is *structured
    /// remembering*: unlike ternary/rondo, where episodes get fresh
    /// progressions, EVERY section here keeps the theme's harmonic
    /// skeleton. The ear always hears the same ground beneath a changing
    /// surface — that persistence-under-transformation is what makes a
    /// variation set feel like one idea examined, not four ideas in a row.
    ///
    /// - **Theme** (role A): `motif` over `progression` in `home_key` — the
    ///   plain statement everything after refers back to.
    /// - **Minore** (role B): the classical mode-flip variation (every
    ///   Mozart/Beethoven set has one) — `home_key`'s PARALLEL key (same
    ///   tonic, opposite mode: the ground doesn't move, its color does)
    ///   with a seed-chosen [`contrasting_transform`] of the theme, over
    ///   the SAME progression degrees. The inward, darkened middle.
    /// - **Figuration** (role C): the "division" variation — the theme's own
    ///   degrees with connecting tones filled in ([`figuration_variation`]),
    ///   back in the home key, same progression. Role C's peak intensity
    ///   makes this the set's brilliant variation, the classic dramaturgy
    ///   (theme → minore darkens → final variation blazes).
    /// - **Theme return** (role ReturnA): the theme verbatim — after
    ///   hearing it darkened and ornamented, the plain statement lands as
    ///   an arrival, and the downstream judgment machinery (final return =
    ///   verbatim + lift) treats it as "finally complete."
    ///
    /// `seed` picks the minore's transformation and the figuration's
    /// neighbor directions; the whole form stays deterministic.
    pub fn variations(
        motif: &Motif,
        home_key: Key,
        progression: &Progression,
        meter: f64,
        seed: u64,
        use_sentence: bool,
        grammar: Option<FormGrammarContext>,
    ) -> Self {
        // Per-section cadence grammar, exactly as in `ternary`/`rondo`. No
        // B/C progression regeneration here (unlike `ternary`/`rondo`) — by
        // design EVERY variations section keeps the theme's own progression
        // (see this fn's doc comment), so there is no "fresh" progression
        // for `grammar.spec` to influence.
        let harmony = grammar.map(|g| g.harmony);
        let build = |m: &Motif, prog: &[i32], key: Key| {
            let dominant = key.cadence_dominant_degree();
            match (use_sentence, harmony) {
                (true, Some(h)) => {
                    Period::parallel_sentence_in_for_grammar(m, prog, meter, dominant, h)
                }
                (true, None) => Period::parallel_sentence_in(m, prog, meter, dominant),
                (false, Some(h)) => Period::parallel_in_for_grammar(m, prog, meter, dominant, h),
                (false, None) => Period::parallel_in(m, prog, meter, dominant),
            }
        };
        let prog = &progression.degrees;

        let theme_period = build(motif, prog, home_key);

        let minore_key = home_key.parallel();
        let pivot = motif.notes.iter().find_map(|x| x.degree).unwrap_or(1);
        let minore_motif = contrasting_transform(motif, pivot, seed);
        let minore_period = build(&minore_motif, prog, minore_key);

        let figured_motif = figuration_variation(motif, seed);
        let figured_period = build(&figured_motif, prog, home_key);

        Form {
            sections: vec![
                Section {
                    role: SectionRole::A,
                    key: home_key,
                    period: theme_period.clone(),
                },
                Section {
                    role: SectionRole::B,
                    key: minore_key,
                    period: minore_period,
                },
                Section {
                    role: SectionRole::C,
                    key: home_key,
                    period: figured_period,
                },
                Section {
                    role: SectionRole::ReturnA,
                    key: home_key,
                    period: theme_period,
                },
            ],
        }
    }

    /// Total length across all sections, in beats.
    pub fn total_duration(&self) -> Duration {
        self.sections.iter().fold(Duration::zero(), |acc, s| {
            acc + s.period.antecedent.total_duration() + s.period.consequent.total_duration()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motif::MotifNote;
    use crate::pitch::PitchClass;

    #[test]
    fn section_intensity_peaks_at_c_and_troughs_at_a() {
        let a = SectionRole::A.intensity();
        let b = SectionRole::B.intensity();
        let ret = SectionRole::ReturnA.intensity();
        let c = SectionRole::C.intensity();
        assert!(a < b, "A (establish) must be calmer than B (depart)");
        assert!(b < c, "C (rondo's peak) must be the most intense");
        assert!(
            a < ret,
            "the return must be more settled/confident than the opening establish"
        );
        assert!(
            ret < c,
            "the return must still be calmer than the true peak"
        );
        for v in [a, b, ret, c] {
            assert!(
                (0.5..=1.5).contains(&v),
                "intensity {v} out of a sane dynamic range"
            );
        }
    }

    #[test]
    fn contrasting_transform_choices_are_ground_truth() {
        let m = Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
        ]);
        assert_eq!(contrasting_transform(&m, 1, 0), m.invert(1));
        assert_eq!(contrasting_transform(&m, 1, 1), m.retrograde());
        assert_eq!(contrasting_transform(&m, 1, 2), m.invert(1).retrograde());
        // choice wraps mod 3
        assert_eq!(
            contrasting_transform(&m, 1, 3),
            contrasting_transform(&m, 1, 0)
        );
    }

    #[test]
    fn contrasting_transform_preserves_duration_for_every_choice() {
        let m = Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::eighth()),
            (3, Duration::eighth()),
            (5, Duration::half()),
        ]);
        for choice in 0..3u64 {
            assert_eq!(
                contrasting_transform(&m, 1, choice).total_duration(),
                m.total_duration(),
                "choice {choice} must preserve total duration"
            );
        }
    }

    #[test]
    fn rondo_b_and_c_always_use_different_transforms() {
        // For every seed, b_choice and c_choice (mod 3) must differ --
        // otherwise the two contrasting episodes could share a
        // transformation and feel less distinct from each other.
        for seed in 0..12u64 {
            let b_choice = seed % 3;
            let c_choice = (b_choice + 1) % 3;
            assert_ne!(b_choice, c_choice, "seed {seed}");
        }
    }

    #[test]
    fn oriented_choice_zero_is_identity() {
        let m = Motif::from_degrees(&[(1, Duration::quarter()), (3, Duration::quarter())]);
        assert_eq!(oriented(&m, 0), m);
        assert_eq!(oriented(&m, 4), m); // wraps mod 4
    }

    #[test]
    fn oriented_nonzero_choices_match_contrasting_transform() {
        let m = Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
        ]);
        let pivot = 1;
        assert_eq!(oriented(&m, 1), contrasting_transform(&m, pivot, 0));
        assert_eq!(oriented(&m, 2), contrasting_transform(&m, pivot, 1));
        assert_eq!(oriented(&m, 3), contrasting_transform(&m, pivot, 2));
    }

    #[test]
    fn oriented_preserves_duration_for_every_choice() {
        let m = Motif::from_degrees(&[
            (1, Duration::eighth()),
            (2, Duration::quarter()),
            (5, Duration::half()),
        ]);
        for choice in 0..4u64 {
            assert_eq!(oriented(&m, choice).total_duration(), m.total_duration());
        }
    }

    fn germ() -> Motif {
        Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
        ])
    }

    #[test]
    fn ternary_form_has_three_sections_in_order() {
        let key = Key::major(PitchClass::C);
        let prog = Progression::authentic();
        let form = Form::ternary(&germ(), key, &prog, 4.0, 1, false, None);
        assert_eq!(form.sections.len(), 3);
        assert_eq!(form.sections[0].role, SectionRole::A);
        assert_eq!(form.sections[1].role, SectionRole::B);
        assert_eq!(form.sections[2].role, SectionRole::ReturnA);
    }

    #[test]
    fn b_section_modulates_to_the_relative_key() {
        let key = Key::major(PitchClass::C);
        let prog = Progression::authentic();
        let form = Form::ternary(&germ(), key, &prog, 4.0, 1, false, None);
        assert_eq!(form.sections[0].key, key);
        assert_eq!(form.sections[1].key, key.relative()); // A minor
        assert_eq!(form.sections[2].key, key); // back home
    }

    #[test]
    fn return_a_is_identical_to_a() {
        // The return must be A verbatim (same notes) — that recognizable
        // sameness IS the "return" a listener perceives.
        let key = Key::major(PitchClass::C);
        let prog = Progression::authentic();
        let form = Form::ternary(&germ(), key, &prog, 4.0, 1, false, None);
        assert_eq!(form.sections[0].period, form.sections[2].period);
    }

    #[test]
    fn b_section_contrasts_with_a() {
        // B must NOT be identical to A — it's a genuinely different section
        // (different key AND a transformed/inverted motif).
        let key = Key::major(PitchClass::C);
        let prog = Progression::authentic();
        let form = Form::ternary(&germ(), key, &prog, 4.0, 1, false, None);
        assert_ne!(form.sections[0].period, form.sections[1].period);
    }

    #[test]
    fn ternary_is_deterministic() {
        let key = Key::major(PitchClass::C);
        let prog = Progression::authentic();
        let a = Form::ternary(&germ(), key, &prog, 4.0, 7, true, None);
        let b = Form::ternary(&germ(), key, &prog, 4.0, 7, true, None);
        assert_eq!(a, b);
    }

    #[test]
    fn total_duration_sums_all_sections() {
        let key = Key::major(PitchClass::C);
        let prog = Progression::authentic(); // 4 bars
        let form = Form::ternary(&germ(), key, &prog, 4.0, 1, false, None);
        // Each section is a period: antecedent (4 bars) + consequent (4 bars)
        // = 8 bars = 32 beats; three sections = 96 beats.
        assert_eq!(form.total_duration(), Duration::new(96, 1));
    }

    #[test]
    fn works_starting_from_a_minor_key_too() {
        // Minor -> relative major direction, exercised end-to-end.
        let key = Key::minor(PitchClass::A);
        let prog = Progression::authentic();
        let form = Form::ternary(&germ(), key, &prog, 4.0, 3, false, None);
        assert_eq!(form.sections[1].key, Key::major(PitchClass::C));
        // Sanity: no section is empty.
        for s in &form.sections {
            let notes: &[MotifNote] = &s.period.antecedent.line.notes;
            assert!(!notes.is_empty());
        }
    }

    #[test]
    fn rondo_has_five_sections_in_abaca_order() {
        let key = Key::major(PitchClass::C);
        let prog = Progression::authentic();
        let form = Form::rondo(&germ(), key, &prog, 4.0, 1, false, None);
        assert_eq!(form.sections.len(), 5);
        assert_eq!(form.sections[0].role, SectionRole::A);
        assert_eq!(form.sections[1].role, SectionRole::B);
        assert_eq!(form.sections[2].role, SectionRole::ReturnA);
        assert_eq!(form.sections[3].role, SectionRole::C);
        assert_eq!(form.sections[4].role, SectionRole::ReturnA);
    }

    #[test]
    fn rondo_b_is_relative_key_and_c_is_parallel_key() {
        let key = Key::major(PitchClass::C);
        let prog = Progression::authentic();
        let form = Form::rondo(&germ(), key, &prog, 4.0, 1, false, None);
        assert_eq!(form.sections[0].key, key);
        assert_eq!(form.sections[1].key, key.relative()); // A minor
        assert_eq!(form.sections[2].key, key);
        assert_eq!(form.sections[3].key, key.parallel()); // C minor
        assert_eq!(form.sections[4].key, key);
    }

    #[test]
    fn rondo_both_returns_are_identical_to_a() {
        let key = Key::major(PitchClass::C);
        let prog = Progression::authentic();
        let form = Form::rondo(&germ(), key, &prog, 4.0, 1, false, None);
        assert_eq!(form.sections[0].period, form.sections[2].period);
        assert_eq!(form.sections[0].period, form.sections[4].period);
    }

    #[test]
    fn rondo_c_contrasts_with_both_a_and_b() {
        let key = Key::major(PitchClass::C);
        let prog = Progression::authentic();
        let form = Form::rondo(&germ(), key, &prog, 4.0, 1, false, None);
        assert_ne!(form.sections[0].period, form.sections[3].period);
        assert_ne!(form.sections[1].period, form.sections[3].period);
    }

    #[test]
    fn rondo_is_deterministic() {
        let key = Key::major(PitchClass::C);
        let prog = Progression::authentic();
        let a = Form::rondo(&germ(), key, &prog, 4.0, 7, true, None);
        let b = Form::rondo(&germ(), key, &prog, 4.0, 7, true, None);
        assert_eq!(a, b);
    }

    #[test]
    fn variations_has_four_sections_theme_minore_figuration_return() {
        let key = Key::major(PitchClass::C);
        let prog = Progression::authentic();
        let form = Form::variations(&germ(), key, &prog, 4.0, 1, false, None);
        assert_eq!(form.sections.len(), 4);
        assert_eq!(form.sections[0].role, SectionRole::A);
        assert_eq!(form.sections[1].role, SectionRole::B);
        assert_eq!(form.sections[2].role, SectionRole::C);
        assert_eq!(form.sections[3].role, SectionRole::ReturnA);
    }

    #[test]
    fn variations_every_section_keeps_the_theme_harmonic_skeleton() {
        // THE defining invariant, and exactly what separates a variation
        // set from ternary/rondo (whose episodes get freshly generated
        // progressions): every section is built over the theme's own
        // progression degrees. The ground never moves; only the surface.
        // Compare each section against the THEME's stored progression, not
        // the raw input: period-building substitutes the antecedent's final
        // bar with the half-cadence dominant ([1,4,5,1] -> [1,4,5,5]) for
        // theme and variations alike — the skeleton is whatever the theme
        // actually carries, identically re-carried by every variation.
        let key = Key::major(PitchClass::C);
        let prog = Progression::authentic();
        let form = Form::variations(&germ(), key, &prog, 4.0, 5, false, None);
        let theme = form.sections[0].period.clone();
        for s in &form.sections {
            assert_eq!(
                s.period.antecedent.progression, theme.antecedent.progression,
                "section {:?} abandoned the theme's harmonic skeleton",
                s.role
            );
            assert_eq!(
                s.period.consequent.progression, theme.consequent.progression,
                "section {:?} abandoned the theme's harmonic skeleton (consequent)",
                s.role
            );
        }
        // Contrast with rondo on the same inputs: its episodes DO move.
        let rondo = Form::rondo(&germ(), key, &prog, 4.0, 5, false, None);
        assert_ne!(
            rondo.sections[1].period.antecedent.progression,
            rondo.sections[0].period.antecedent.progression
        );
    }

    #[test]
    fn variations_minore_is_the_parallel_key_and_the_rest_stay_home() {
        let key = Key::major(PitchClass::C);
        let prog = Progression::authentic();
        let form = Form::variations(&germ(), key, &prog, 4.0, 1, false, None);
        assert_eq!(form.sections[0].key, key);
        assert_eq!(form.sections[1].key, key.parallel()); // C minor
        assert_eq!(form.sections[2].key, key); // figuration back home
        assert_eq!(form.sections[3].key, key);
    }

    #[test]
    fn variations_final_return_is_the_theme_verbatim() {
        // The set closes with the plain statement — the judgment
        // machinery's "finally complete" treatment depends on it.
        let key = Key::major(PitchClass::C);
        let prog = Progression::authentic();
        let form = Form::variations(&germ(), key, &prog, 4.0, 9, true, None);
        assert_eq!(form.sections[0].period, form.sections[3].period);
    }

    #[test]
    fn variations_middle_sections_both_differ_from_the_theme() {
        let key = Key::major(PitchClass::C);
        let prog = Progression::authentic();
        let form = Form::variations(&germ(), key, &prog, 4.0, 2, false, None);
        assert_ne!(form.sections[0].period, form.sections[1].period);
        assert_ne!(form.sections[0].period, form.sections[2].period);
        assert_ne!(form.sections[1].period, form.sections[2].period);
    }

    #[test]
    fn variations_is_deterministic() {
        let key = Key::minor(PitchClass::A);
        let prog = Progression::authentic();
        let a = Form::variations(&germ(), key, &prog, 4.0, 7, true, None);
        let b = Form::variations(&germ(), key, &prog, 4.0, 7, true, None);
        assert_eq!(a, b);
    }

    /// Normalize a raw progression degree (which may be any integer, e.g.
    /// octave-shifted or zero-based) to 1..=7.
    fn norm_degree(d: i32) -> i32 {
        let r = d.rem_euclid(7);
        if r == 0 { 7 } else { r }
    }

    #[test]
    fn ternary_b_section_honors_the_styles_own_progression_when_grammar_context_given() {
        // Real production data: Blues's 12-bar chorus (I-I-I-I/IV-IV-I-I/
        // V-IV-I-I) uses ONLY degrees {1,4,5} -- unlike Classical's
        // functional grammar, which freely visits 2/3/6/7 too. Without a
        // grammar context, the B section is generated by the generic
        // classical grammar and can land on those foreign degrees; with a
        // grammar context, it must stay inside the style's own vocabulary.
        let spec = crate::style::Style::Blues.spec();
        let key = Key::major(PitchClass::C);
        let prog = spec.progression(12, 1);
        let grammar = FormGrammarContext {
            harmony: crate::grammar::HarmonicSyntax::BluesChorus,
            spec: &spec,
        };
        for seed in 0..12u64 {
            let form = Form::ternary(&germ(), key, &prog, 4.0, seed, false, Some(grammar));
            let b_section = &form.sections[1];
            for phrase in [&b_section.period.antecedent, &b_section.period.consequent] {
                for &d in &phrase.progression {
                    assert!(
                        [1, 4, 5].contains(&norm_degree(d)),
                        "B section degree {d} (seed {seed}) escaped Blues's own \
                         {{1,4,5}} vocabulary -- the grammar context isn't being honored"
                    );
                }
            }
        }
    }

    #[test]
    fn rondo_b_and_c_sections_honor_the_styles_own_progression_when_grammar_context_given() {
        let spec = crate::style::Style::Blues.spec();
        let key = Key::major(PitchClass::C);
        let prog = spec.progression(12, 1);
        let grammar = FormGrammarContext {
            harmony: crate::grammar::HarmonicSyntax::BluesChorus,
            spec: &spec,
        };
        for seed in 0..12u64 {
            let form = Form::rondo(&germ(), key, &prog, 4.0, seed, false, Some(grammar));
            for role_idx in [1, 3] {
                // B, then C
                let section = &form.sections[role_idx];
                for phrase in [&section.period.antecedent, &section.period.consequent] {
                    for &d in &phrase.progression {
                        assert!(
                            [1, 4, 5].contains(&norm_degree(d)),
                            "section {role_idx} degree {d} (seed {seed}) escaped \
                             Blues's own {{1,4,5}} vocabulary"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn ternary_without_grammar_context_keeps_the_plain_classical_generator() {
        // No-regression contract: `grammar: None` must still use the free
        // classical functional grammar (able to visit degrees outside a
        // narrow style vocabulary), NOT silently start respecting the A
        // section's own degree set the way `Some(grammar)` now does. Same
        // progression as the Blues archetype test above, on purpose: this
        // proves the difference is `grammar`'s presence, not the input.
        let key = Key::major(PitchClass::C);
        let prog = Progression::new(vec![1, 1, 1, 1, 4, 4, 1, 1, 5, 4, 1, 1]);
        let escaped = (0..40u64).any(|seed| {
            let form = Form::ternary(&germ(), key, &prog, 4.0, seed, false, None);
            form.sections[1]
                .period
                .antecedent
                .progression
                .iter()
                .chain(&form.sections[1].period.consequent.progression)
                .any(|&d| ![1, 4, 5].contains(&norm_degree(d)))
        });
        assert!(
            escaped,
            "expected at least one seed's B section (grammar=None) to visit a \
             degree outside {{1,4,5}} -- the free classical generator should be \
             unconstrained by the A section's own narrower vocabulary"
        );
    }

    #[test]
    fn figuration_preserves_total_duration_exactly() {
        let m = Motif::from_degrees(&[
            (1, Duration::half()),
            (3, Duration::quarter()),
            (2, Duration::eighth()),
            (5, Duration::dotted(Duration::quarter())),
            (1, Duration::whole()),
        ]);
        for seed in 0..8u64 {
            assert_eq!(
                figuration_variation(&m, seed).total_duration(),
                m.total_duration(),
                "seed {seed} broke the harmonic-rhythm safety contract"
            );
        }
    }

    #[test]
    fn figuration_divides_long_notes_but_spares_short_ones_and_the_last() {
        let m = Motif::from_degrees(&[
            (1, Duration::half()),    // divisible -> 2 notes
            (3, Duration::eighth()),  // too short -> untouched
            (2, Duration::quarter()), // divisible -> 2 notes
            (1, Duration::whole()),   // LAST: spared for cadence stability
        ]);
        let figured = figuration_variation(&m, 0);
        assert_eq!(figured.notes.len(), 6);
        // The final note is the original, whole.
        assert_eq!(*figured.notes.last().unwrap(), *m.notes.last().unwrap());
        // The eighth passed through untouched.
        assert!(
            figured
                .notes
                .iter()
                .any(|n| n.degree == Some(3) && n.duration == Duration::eighth())
        );
    }

    #[test]
    fn figuration_connecting_tones_step_toward_the_next_note() {
        // 1(half) -> 3: the moving line gets a PASSING tone (degree 2),
        // not an arbitrary neighbor.
        let m = Motif::from_degrees(&[
            (1, Duration::half()),
            (3, Duration::half()),
            (1, Duration::whole()),
        ]);
        let figured = figuration_variation(&m, 0);
        // 1 -> [1, 2] rising toward 3; 3 -> [3, 2] falling toward 1.
        let degrees: Vec<i32> = figured.notes.iter().filter_map(|n| n.degree).collect();
        assert_eq!(degrees, vec![1, 2, 3, 2, 1]);
    }

    #[test]
    fn figuration_leaves_rests_untouched() {
        let m = Motif::new(vec![
            MotifNote::new(1, Duration::half()),
            MotifNote::rest(Duration::half()),
            MotifNote::new(1, Duration::whole()),
        ]);
        let figured = figuration_variation(&m, 3);
        assert!(
            figured
                .notes
                .iter()
                .any(|n| n.is_rest() && n.duration == Duration::half())
        );
        assert_eq!(figured.total_duration(), m.total_duration());
    }

    #[test]
    fn variations_total_duration_sums_all_four_sections() {
        let key = Key::major(PitchClass::C);
        let prog = Progression::authentic(); // 4 bars
        let form = Form::variations(&germ(), key, &prog, 4.0, 1, false, None);
        // Each section: antecedent (4 bars) + consequent (4 bars) = 32
        // beats; four sections = 128 beats.
        assert_eq!(form.total_duration(), Duration::new(128, 1));
    }

    #[test]
    fn rondo_total_duration_sums_all_five_sections() {
        let key = Key::major(PitchClass::C);
        let prog = Progression::authentic(); // 4 bars
        let form = Form::rondo(&germ(), key, &prog, 4.0, 1, false, None);
        // Each section is a period: antecedent (4 bars) + consequent (4
        // bars) = 8 bars = 32 beats; five sections = 160 beats.
        assert_eq!(form.total_duration(), Duration::new(160, 1));
    }
}
