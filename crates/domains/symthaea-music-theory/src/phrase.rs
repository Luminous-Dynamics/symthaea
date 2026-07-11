// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phrases and periods: where motif (Layer 2) meets harmony (Layer 1).
//!
//! A phrase develops a motif over a chord progression, snapping strong-beat
//! notes to chord tones (so the line is heard *as* the harmony) while leaving
//! weak-beat notes as diatonic passing/neighbor tones (the stepwise motion
//! that gives a line life). It ends on a cadence.
//!
//! A [`Period`] pairs an **antecedent** (ends on a Half cadence — an
//! unresolved musical *question*) with a **consequent** built from the same
//! motif that answers it (ends on an Authentic cadence — the *resolution*).
//! That question-and-answer is the difference between "notes happening" and
//! "a phrase that means something."

use crate::cadence::Cadence;
use crate::harmony::Key;
use crate::motif::Motif;
use crate::pitch::Pitch;
use crate::rhythm::Duration;
use serde::{Deserialize, Serialize};

/// A realized melodic phrase: a developed motif over a progression, closing on
/// a cadence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Phrase {
    /// The developed melodic line (scale-degree motif; render with the key).
    pub line: Motif,
    /// The progression the line was fitted to (chord root scale-degrees).
    pub progression: Vec<i32>,
    /// The cadence this phrase closes on.
    pub cadence: Cadence,
}

impl Phrase {
    /// Build a phrase by developing `motif` over `progression` in `key`,
    /// closing on `cadence`. One motif statement per chord (one chord per
    /// `meter` beats); strong-beat notes snap to the current chord's tones,
    /// weak-beat notes stay diatonic. The final pitched note is steered to the
    /// cadence's melodic goal so the close is voiced convincingly.
    ///
    /// The motif is DEVELOPED across the phrase rather than merely repeated:
    /// the middle measure (index 2 of a 4+-bar phrase) states the motif's
    /// INVERSION, mirroring its contour (an ascending idea answered by a
    /// descending one). That statement → variation → statement arc is what a
    /// listener hears as the idea being *worked*, not looped.
    ///
    /// No `key` is needed here: the whole phrase is built in scale-degree
    /// space (key-agnostic), and the key only enters at [`Phrase::render`].
    pub fn build(motif: &Motif, progression: &[i32], cadence: Cadence, meter: f64) -> Self {
        let n = progression.len();
        // Mirror about the motif's opening degree so the first note holds.
        let pivot = motif.notes.iter().find_map(|x| x.degree).unwrap_or(1);
        let inverted = motif.invert(pivot);

        // Develop the idea: invert it in the middle of a 4+-bar phrase for a
        // contour answer; otherwise just state it every bar.
        let mut line_notes = realize_over_progression(progression, meter, |measure_idx| {
            if n >= 4 && measure_idx == 2 {
                inverted.clone()
            } else {
                motif.clone()
            }
        });
        steer_final_cadence(&mut line_notes, cadence);

        Phrase {
            line: Motif::new(line_notes),
            progression: progression.to_vec(),
            cadence,
        }
    }

    /// Build a phrase using SENTENCE structure instead of simple repetition:
    /// bars 0–1 state the motif as-is (the "basic idea" and its repetition at
    /// the next chord), and bars 2..n FRAGMENT it — the first half of its
    /// notes, rescaled to exactly half a bar and sequenced (transposed down a
    /// step) to fill each remaining bar. That break into a smaller, twice-
    /// stated cell is the classical "continuation" phase (Caplin/Schoenberg)
    /// and creates a sense of the idea being worked and driven toward the
    /// cadence that plain bar-for-bar repetition doesn't. Whether the
    /// fragment's notes end up shorter or longer than the presentation's
    /// depends on the source motif's own rhythmic profile — see
    /// `continuation_unit`'s doc for the exact guarantee.
    ///
    /// Falls back to plain per-bar statement (like [`Phrase::build`], minus
    /// the inversion) when `progression` is shorter than 4 bars — there's no
    /// room for a presentation + continuation split.
    pub fn build_sentence(
        motif: &Motif,
        progression: &[i32],
        cadence: Cadence,
        meter: f64,
    ) -> Self {
        let n = progression.len();
        let continuation = continuation_unit(motif, meter);

        let mut line_notes = realize_over_progression(progression, meter, |measure_idx| {
            if n >= 4 && measure_idx >= 2 {
                continuation.clone().unwrap_or_else(|| motif.clone())
            } else {
                motif.clone()
            }
        });
        steer_final_cadence(&mut line_notes, cadence);

        Phrase {
            line: Motif::new(line_notes),
            progression: progression.to_vec(),
            cadence,
        }
    }

    /// Render the phrase to symbolic pitches over the key's scale.
    pub fn render(&self, key: Key, tonic_octave: i32) -> Vec<(Option<Pitch>, Duration)> {
        self.line.render(key.scale(), tonic_octave)
    }

    /// Total length in beats.
    pub fn total_duration(&self) -> Duration {
        self.line.total_duration()
    }
}

/// An antecedent–consequent period: a question and its answer, from one motif.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Period {
    pub antecedent: Phrase,
    pub consequent: Phrase,
}

impl Period {
    /// Build a parallel period: both phrases develop the SAME motif over the
    /// same harmonic plan, but the antecedent ends on a Half cadence
    /// (unresolved question) and the consequent on an Authentic cadence
    /// (resolution). This shared-motif, different-ending shape is the
    /// classical period (Caplin).
    pub fn parallel(motif: &Motif, progression: &[i32], meter: f64) -> Self {
        Self::parallel_in(motif, progression, meter, 5)
    }

    /// [`Self::parallel`] with the key's own cadential "dominant" degree —
    /// 5 for functional keys, but the leading-tone-less modes close from
    /// their characteristic chord instead (♭VII for Dorian/Mixolydian/
    /// Aeolian, ♭II for Phrygian). Pass
    /// [`Key::cadence_dominant_degree`](crate::harmony::Key::cadence_dominant_degree);
    /// forcing V–I onto a modal key would erase the mode.
    pub fn parallel_in(motif: &Motif, progression: &[i32], meter: f64, dominant: i32) -> Self {
        // Antecedent: progression steered toward the dominant (Half cadence).
        let mut ante_prog = progression.to_vec();
        if let Some(last) = ante_prog.last_mut() {
            *last = dominant; // end on the tension chord — the "question"
        }
        let antecedent = Phrase::build(motif, &ante_prog, Cadence::Half, meter);

        // Consequent: same motif, progression steered to a real close.
        let mut cons_prog = progression.to_vec();
        let n = cons_prog.len();
        if n >= 2 {
            cons_prog[n - 2] = dominant; // penultimate tension chord
            cons_prog[n - 1] = 1; // final tonic
        } else if n == 1 {
            cons_prog[0] = 1;
        }
        let consequent = Phrase::build(motif, &cons_prog, Cadence::Authentic, meter);

        Period {
            antecedent,
            consequent,
        }
    }

    /// Like [`Period::parallel`], but each half is built with SENTENCE
    /// structure ([`Phrase::build_sentence`]: statement → repetition →
    /// fragmentation) instead of simple inversion-development. This is the
    /// higher-energy alternative — the fragmentation drives toward each
    /// cadence rather than the calmer statement/variation/statement arc of a
    /// plain developed phrase.
    pub fn parallel_sentence(motif: &Motif, progression: &[i32], meter: f64) -> Self {
        Self::parallel_sentence_in(motif, progression, meter, 5)
    }

    /// [`Self::parallel_sentence`] with the key's own cadential degree —
    /// see [`Self::parallel_in`] for why modal keys must not be forced to V.
    pub fn parallel_sentence_in(
        motif: &Motif,
        progression: &[i32],
        meter: f64,
        dominant: i32,
    ) -> Self {
        let mut ante_prog = progression.to_vec();
        if let Some(last) = ante_prog.last_mut() {
            *last = dominant;
        }
        let antecedent = Phrase::build_sentence(motif, &ante_prog, Cadence::Half, meter);

        let mut cons_prog = progression.to_vec();
        let n = cons_prog.len();
        if n >= 2 {
            cons_prog[n - 2] = dominant;
            cons_prog[n - 1] = 1;
        } else if n == 1 {
            cons_prog[0] = 1;
        }
        let consequent = Phrase::build_sentence(motif, &cons_prog, Cadence::Authentic, meter);

        Period {
            antecedent,
            consequent,
        }
    }

    /// The whole period as one line (antecedent then consequent).
    pub fn line(&self) -> Motif {
        self.antecedent.line.then(&self.consequent.line)
    }
}

/// Realize a sequence of per-measure motif variants over `progression`,
/// re-anchoring each to its chord's root and snapping strong beats to chord
/// tones. `variant_for(measure_idx)` supplies the (relative-degree-space)
/// motif to state in that measure, BEFORE the per-chord re-anchor. Shared by
/// [`Phrase::build`] (development-by-inversion) and
/// [`Phrase::build_sentence`] (development-by-fragmentation) so both phrase
/// archetypes share one strong-beat/chord-fitting implementation.
fn realize_over_progression(
    progression: &[i32],
    meter: f64,
    variant_for: impl Fn(usize) -> Motif,
) -> Vec<crate::motif::MotifNote> {
    let mut line_notes = Vec::new();
    for (measure_idx, &chord_deg) in progression.iter().enumerate() {
        // Re-anchor so degree-1 lands on this chord's root degree.
        let stated = variant_for(measure_idx).transpose(chord_deg - 1);

        let mut beat_pos = 0.0f64;
        for note in &stated.notes {
            let is_strong = is_strong_beat(beat_pos, meter);
            let fitted = match note.degree {
                Some(d) if is_strong => Some(nearest_chord_tone(d, chord_deg)),
                other => other, // weak beat or rest: leave as diatonic / rest
            };
            line_notes.push(crate::motif::MotifNote {
                degree: fitted,
                duration: note.duration,
            });
            beat_pos += note.duration.beats();
        }
    }
    line_notes
}

/// Steer the final pitched note to the cadence's melodic goal, in the
/// register it currently occupies (so the close resolves convincingly).
fn steer_final_cadence(line_notes: &mut [crate::motif::MotifNote], cadence: Cadence) {
    if let Some(idx) = line_notes.iter().rposition(|n| n.degree.is_some()) {
        let cur = line_notes[idx].degree.unwrap();
        line_notes[idx].degree = Some(nearest_octave_of(cadence.melodic_goal(), cur));
    }
}

/// Build the SENTENCE continuation unit: the first half of `motif`'s notes
/// (by count), diminished and rationally rescaled to fill EXACTLY half a bar
/// (`meter`/2 beats), then sequenced (transposed down one scale degree) to
/// fill a whole bar. The rescale is exact rational arithmetic (via
/// [`Duration`]'s num/den), not a float approximation, so — like every other
/// motif transform — it's reproducible regardless of the source motif's
/// rhythm (odd note counts, mixed durations, …).
///
/// Returns `None` for a 1-note (or empty) motif, which can't meaningfully
/// fragment; callers fall back to plain repetition.
fn continuation_unit(motif: &Motif, meter: f64) -> Option<Motif> {
    if motif.len() < 2 {
        return None;
    }
    let half_bar = Duration::new(meter as i64, 2);
    let frag_len = (motif.len() / 2).max(1);
    let frag = motif.fragment(0, frag_len);
    let frag_total = frag.total_duration();
    if frag_total.beats() <= 0.0 {
        return None;
    }
    // Cross-multiply to get the exact rational scale factor frag*(n/d) = half_bar.
    let scaled = frag.scale_rhythm(
        half_bar.num() * frag_total.den(),
        half_bar.den() * frag_total.num(),
    );
    Some(scaled.sequence(2, -1))
}

/// A beat position is "strong" if it is the downbeat (0) or the mid-measure
/// accent (>= meter/2), e.g. beats 1 and 3 in 4/4. Tolerant of float drift.
fn is_strong_beat(beat_pos: f64, meter: f64) -> bool {
    let within = beat_pos.rem_euclid(meter);
    within < 1e-6 || (within - meter / 2.0).abs() < 1e-6
}

/// The chord tone (as a scale degree) nearest to `raw`, where the chord is the
/// diatonic triad on `chord_root_deg` — tones at degrees {root, root+2, root+4}
/// and their octave transpositions.
fn nearest_chord_tone(raw: i32, chord_root_deg: i32) -> i32 {
    let mut best = chord_root_deg;
    let mut best_dist = i32::MAX;
    for oct in -1..=1 {
        for offset in [0, 2, 4] {
            let cand = chord_root_deg + offset + 7 * oct;
            let dist = (cand - raw).abs();
            if dist < best_dist {
                best_dist = dist;
                best = cand;
            }
        }
    }
    best
}

/// The octave transposition of scale-degree `goal` nearest to `reference`
/// (so steering the cadence note to the tonic keeps it in the local register).
fn nearest_octave_of(goal: i32, reference: i32) -> i32 {
    let mut best = goal;
    let mut best_dist = i32::MAX;
    for oct in -2..=2 {
        let cand = goal + 7 * oct;
        let dist = (cand - reference).abs();
        if dist < best_dist {
            best_dist = dist;
            best = cand;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harmony::Key;
    use crate::pitch::PitchClass;

    fn germ() -> Motif {
        // do-re-mi-sol in quarter notes (all on strong/weak beats of 4/4)
        Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
        ])
    }

    fn chord_tone_degrees(chord_root: i32) -> Vec<i32> {
        let mut v = Vec::new();
        for oct in -2..=2 {
            for off in [0, 2, 4] {
                v.push(chord_root + off + 7 * oct);
            }
        }
        v
    }

    #[test]
    fn strong_beats_land_on_chord_tones() {
        // The core correctness property: every strong-beat note in the phrase
        // is a chord tone of the chord sounding under it.
        let prog = [1, 4, 5, 1];
        let phrase = Phrase::build(&germ(), &prog, Cadence::Authentic, 4.0);

        // Walk the line, tracking chord (one per measure of 4 beats) and beat.
        let mut measure = 0usize;
        let mut measure_beat = 0.0f64;
        for note in &phrase.line.notes {
            if measure_beat >= 4.0 - 1e-6 {
                measure += 1;
                measure_beat = 0.0;
            }
            let chord_root = prog[measure.min(prog.len() - 1)];
            if let Some(d) = note.degree
                && is_strong_beat(measure_beat, 4.0)
            {
                assert!(
                    chord_tone_degrees(chord_root).contains(&d),
                    "strong-beat degree {d} not a chord tone of chord {chord_root} \
                     (measure {measure}, beat {measure_beat})"
                );
            }
            measure_beat += note.duration.beats();
        }
    }

    #[test]
    fn phrase_develops_the_motif_not_just_repeats() {
        // A 4-bar phrase inverts the motif in the middle bar, so bar 2 is a
        // contour variation of bar 0 (ascending idea → descending answer),
        // not an identical repeat. Same chord each bar isolates the develop.
        let m = germ(); // ascending 1 2 3 5
        let prog = [1, 1, 1, 1];
        let phrase = Phrase::build(&m, &prog, Cadence::Authentic, 4.0);
        let per = m.len();
        let bar0: Vec<i32> = phrase.line.notes[0..per]
            .iter()
            .filter_map(|x| x.degree)
            .collect();
        let bar2: Vec<i32> = phrase.line.notes[2 * per..3 * per]
            .iter()
            .filter_map(|x| x.degree)
            .collect();
        assert_ne!(bar0, bar2, "bar 2 must develop (invert) the motif");
        // bar 0 rises overall; bar 2 (inversion) falls overall.
        assert!(bar0.last() > bar0.first(), "statement bar rises");
        assert!(bar2.last() < bar2.first(), "inverted bar falls");
    }

    #[test]
    fn phrase_preserves_motif_rhythm_and_length() {
        // The developed line is NOT a random walk: it has one motif statement
        // per chord, so its length and total duration are exact multiples.
        let m = germ();
        let prog = [1, 4, 5, 1];
        let phrase = Phrase::build(&m, &prog, Cadence::Authentic, 4.0);
        assert_eq!(phrase.line.len(), m.len() * prog.len());
        assert_eq!(
            phrase.total_duration(),
            m.total_duration().scale(prog.len() as i64, 1)
        );
    }

    #[test]
    fn modal_period_cadences_on_the_mode_native_degree() {
        // A Dorian/Mixolydian/Aeolian key closes ♭VII→i: the consequent's
        // harmonic plan must end (7, 1) and the antecedent must end on 7 —
        // NOT the functional V those slots used to be forced to.
        let period = Period::parallel_in(&germ(), &[1, 4, 6, 1], 4.0, 7);
        assert_eq!(period.antecedent.progression.last(), Some(&7));
        let n = period.consequent.progression.len();
        assert_eq!(period.consequent.progression[n - 2], 7);
        assert_eq!(period.consequent.progression[n - 1], 1);
        // Sentence structure honors the same grammar.
        let sentence = Period::parallel_sentence_in(&germ(), &[1, 4, 6, 1], 4.0, 7);
        assert_eq!(sentence.antecedent.progression.last(), Some(&7));
        // And the default remains the functional dominant.
        let functional = Period::parallel(&germ(), &[1, 4, 6, 1], 4.0);
        assert_eq!(functional.antecedent.progression.last(), Some(&5));
    }

    #[test]
    fn parallel_period_is_question_then_answer() {
        let period = Period::parallel(&germ(), &[1, 4, 5, 1], 4.0);

        // Antecedent ends on a Half cadence (unresolved question).
        assert_eq!(period.antecedent.cadence, Cadence::Half);
        assert!(!period.antecedent.cadence.is_conclusive());

        // Consequent ends on an Authentic cadence (the resolution).
        assert_eq!(period.consequent.cadence, Cadence::Authentic);
        assert!(period.consequent.cadence.is_conclusive());

        // The consequent's final pitched note is a tonic (degree ≡ 1 mod 7).
        let last = period
            .consequent
            .line
            .notes
            .iter()
            .rev()
            .find_map(|n| n.degree)
            .unwrap();
        assert_eq!(
            last.rem_euclid(7),
            1,
            "consequent must close on tonic, got {last}"
        );

        // The antecedent does NOT close on tonic (it asks, not answers).
        let ante_last = period
            .antecedent
            .line
            .notes
            .iter()
            .rev()
            .find_map(|n| n.degree)
            .unwrap();
        assert_ne!(ante_last.rem_euclid(7), 1, "antecedent should stay open");
    }

    #[test]
    fn period_line_concatenates_both_phrases() {
        let period = Period::parallel(&germ(), &[1, 4, 5, 1], 4.0);
        assert_eq!(
            period.line().len(),
            period.antecedent.line.len() + period.consequent.line.len()
        );
    }

    /// A front-loaded motif (one long note, then two quarters) — unlike
    /// `germ()` (uniform quarters), taking "half the notes" here (1 of 3)
    /// already equals a full half-bar on its own, so the continuation
    /// naturally collapses to FEWER notes per bar than the presentation.
    fn arch_motif() -> Motif {
        Motif::from_degrees(&[
            (1, Duration::half()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
        ])
    }

    #[test]
    fn continuation_unit_fragments_and_fills_a_full_bar() {
        // Two statements of the (rescaled) fragment must exactly fill one
        // bar, and the content must differ from the plain motif — it's a
        // genuine fragment+sequence, not the original idea restated.
        let m = germ();
        let cont = continuation_unit(&m, 4.0).expect("germ has >= 2 notes");
        assert_eq!(cont.total_duration(), Duration::new(4, 1));
        assert_ne!(cont.degrees(), m.degrees());
    }

    #[test]
    fn continuation_unit_can_change_note_count() {
        // For a front-loaded motif, fragmentation genuinely changes the
        // rhythmic profile (3 notes -> 2), which is the diminution/expansion
        // effect real sentence continuations rely on. (Whether it comes out
        // as fewer/shorter or more/longer notes depends on the motif's own
        // shape — the invariant this crate can always guarantee is "the bar
        // still fills exactly," not "always shorter," since a fragment can
        // legitimately need widening as well as narrowing to fit.)
        let m = arch_motif();
        let cont = continuation_unit(&m, 4.0).unwrap();
        assert_ne!(
            cont.len(),
            m.len(),
            "3-note motif should fragment to a different count"
        );
        assert_eq!(cont.total_duration(), Duration::new(4, 1));
    }

    #[test]
    fn continuation_unit_none_for_single_note_motif() {
        let m = Motif::from_degrees(&[(1, Duration::whole())]);
        assert!(continuation_unit(&m, 4.0).is_none());
    }

    #[test]
    fn sentence_continuation_differs_from_plain_repetition() {
        // The continuation bar (index 2) must NOT equal what a plain
        // restatement of the motif at that chord would look like — proving
        // real fragmentation happened, not just the ordinary per-chord
        // transposition every phrase type does.
        let m = germ();
        let prog = [1, 4, 5, 1];
        let sentence = Phrase::build_sentence(&m, &prog, Cadence::Authentic, 4.0);
        let plain_notes = realize_over_progression(&prog, 4.0, |_| m.clone());
        // For this motif the continuation bar happens to have the same note
        // COUNT as plain repetition — which makes the content check meaningful
        // (a coincidental-length false pass is ruled out).
        assert_eq!(sentence.line.notes.len(), plain_notes.len());
        let per = m.len();
        let bar2_sentence: Vec<i32> = sentence.line.notes[2 * per..3 * per]
            .iter()
            .filter_map(|n| n.degree)
            .collect();
        let bar2_plain: Vec<i32> = plain_notes[2 * per..3 * per]
            .iter()
            .filter_map(|n| n.degree)
            .collect();
        assert_ne!(
            bar2_sentence, bar2_plain,
            "continuation bar must differ from plain repetition"
        );
    }

    #[test]
    fn sentence_continuation_fills_each_bar_exactly() {
        // Regardless of the fragmentation arithmetic, every bar must still
        // total exactly `meter` beats — the harmonic rhythm (one chord per
        // bar) must stay intact.
        let m = germ();
        let prog = [1, 4, 5, 1];
        let phrase = Phrase::build_sentence(&m, &prog, Cadence::Authentic, 4.0);
        assert_eq!(
            phrase.total_duration(),
            Duration::new(4 * prog.len() as i64, 1),
            "sentence phrase must still total exactly meter*bars beats"
        );
    }

    #[test]
    fn parallel_sentence_is_still_question_then_answer() {
        // The sentence variant preserves the core period property: antecedent
        // opens (Half cadence), consequent closes on the tonic (Authentic).
        let period = Period::parallel_sentence(&germ(), &[1, 4, 5, 1], 4.0);
        assert_eq!(period.antecedent.cadence, Cadence::Half);
        assert_eq!(period.consequent.cadence, Cadence::Authentic);
        let last = period
            .consequent
            .line
            .notes
            .iter()
            .rev()
            .find_map(|n| n.degree)
            .unwrap();
        assert_eq!(last.rem_euclid(7), 1, "consequent must close on tonic");
    }

    #[test]
    fn short_progression_sentence_falls_back_to_plain_statement() {
        // With < 4 bars there's no room for presentation+continuation; the
        // sentence builder must not panic and should behave like plain
        // per-bar repetition.
        let m = germ();
        let phrase = Phrase::build_sentence(&m, &[1, 5], Cadence::Half, 4.0);
        assert_eq!(phrase.line.len(), m.len() * 2);
    }

    #[test]
    fn phrase_renders_to_pitches() {
        let key = Key::major(PitchClass::C);
        let phrase = Phrase::build(&germ(), &[1, 4, 5, 1], Cadence::Authentic, 4.0);
        let realized = phrase.render(key, 4);
        assert_eq!(realized.len(), phrase.line.len());
        assert!(realized.iter().all(|(p, _)| p.is_some())); // no rests in this germ
    }
}
