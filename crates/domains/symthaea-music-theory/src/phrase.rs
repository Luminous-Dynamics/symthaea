// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phrases and periods: where motif (Layer 2) meets harmony (Layer 1).
//!
//! A phrase develops a motif over a chord progression, snapping strong-beat
//! notes to chord tones (so the line is heard *as* the harmony) while leaving
//! weak-beat notes as diatonic passing/neighbor tones (the stepwise motion
//! that gives a line life) -- or, on a strong beat, as a genuine SUSPENSION
//! when the dissonance is prepared (the same pitch carried from the previous
//! chord) and resolves down by step. It ends on a cadence.
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

    /// Grammar-aware cousin of [`Self::parallel_in`]. Functional harmony
    /// (or any archetype with no detectable natural close) keeps today's
    /// forced Half→Authentic (V–I) question/answer shape unchanged. Every
    /// other declared [`HarmonicSyntax`](crate::grammar::HarmonicSyntax) —
    /// `BluesChorus`/`JazzTurnaround`/`GroundCycle`/`SongLoop`/
    /// `SpectralStasis`/`NarrativeLeitmotif`/etc. — is cyclical or
    /// vamp-based rather than tonal question-and-answer, so when the
    /// archetype's own final two scale degrees already form a recognizable
    /// [`Cadence`] (via [`Cadence::detect`]), BOTH halves close on that same
    /// natural cadence, unmodified, instead of being overwritten with a
    /// borrowed V–I rhetoric that doesn't belong to the style.
    pub fn parallel_in_for_grammar(
        motif: &Motif,
        progression: &[i32],
        meter: f64,
        dominant: i32,
        harmony: crate::grammar::HarmonicSyntax,
    ) -> Self {
        if harmony == crate::grammar::HarmonicSyntax::Functional {
            return Self::parallel_in(motif, progression, meter, dominant);
        }
        let n = progression.len();
        let natural = (n >= 2)
            .then(|| Cadence::detect(progression[n - 2], progression[n - 1]))
            .flatten();
        match natural {
            Some(cadence) => Period {
                antecedent: Phrase::build(motif, progression, cadence, meter),
                consequent: Phrase::build(motif, progression, cadence, meter),
            },
            None => Self::parallel_in(motif, progression, meter, dominant),
        }
    }

    /// [`Self::parallel_in_for_grammar`], but each half uses SENTENCE
    /// structure ([`Phrase::build_sentence`]) — the sentence-flavored
    /// counterpart of [`Self::parallel_sentence_in`].
    pub fn parallel_sentence_in_for_grammar(
        motif: &Motif,
        progression: &[i32],
        meter: f64,
        dominant: i32,
        harmony: crate::grammar::HarmonicSyntax,
    ) -> Self {
        if harmony == crate::grammar::HarmonicSyntax::Functional {
            return Self::parallel_sentence_in(motif, progression, meter, dominant);
        }
        let n = progression.len();
        let natural = (n >= 2)
            .then(|| Cadence::detect(progression[n - 2], progression[n - 1]))
            .flatten();
        match natural {
            Some(cadence) => Period {
                antecedent: Phrase::build_sentence(motif, progression, cadence, meter),
                consequent: Phrase::build_sentence(motif, progression, cadence, meter),
            },
            None => Self::parallel_sentence_in(motif, progression, meter, dominant),
        }
    }

    /// Total duration of antecedent plus consequent.
    ///
    /// This is exact because both phrase durations are rational. Form planners
    /// use it to place section boundaries and prospective obligations before
    /// any audio realization occurs.
    pub fn total_duration(&self) -> Duration {
        self.antecedent.total_duration() + self.consequent.total_duration()
    }

    /// The whole period as one line (antecedent then consequent).
    pub fn line(&self) -> Motif {
        self.antecedent.line.then(&self.consequent.line)
    }
}

/// Realize a sequence of per-measure motif variants over `progression`,
/// re-anchoring each to its chord's root, snapping strong beats to chord
/// tones, and legitimizing (or correcting) weak beats as genuine
/// passing/neighbor tones. `variant_for(measure_idx)` supplies the
/// (relative-degree-space) motif to state in that measure, BEFORE the
/// per-chord re-anchor. Shared by [`Phrase::build`] (development-by-inversion)
/// and [`Phrase::build_sentence`] (development-by-fragmentation) so both
/// phrase archetypes share one strong-beat/chord-fitting implementation.
fn realize_over_progression(
    progression: &[i32],
    meter: f64,
    variant_for: impl Fn(usize) -> Motif,
) -> Vec<crate::motif::MotifNote> {
    // Pass 0: transpose each measure's stated variant onto its chord's root.
    // Nothing is snapped yet -- pass 1 (below) needs each note's RAW degree
    // to recognize suspensions, which are defined by what a note WOULD be
    // before any correction (prepared by an identical raw predecessor,
    // resolving by step to an identical raw follower).
    struct Staged {
        raw_degree: Option<i32>,
        duration: Duration,
        is_strong: bool,
        chord_deg: i32,
    }
    let mut staged = Vec::new();
    for (measure_idx, &chord_deg) in progression.iter().enumerate() {
        let stated = variant_for(measure_idx).transpose(chord_deg - 1);
        let mut beat_pos = 0.0f64;
        for note in &stated.notes {
            let is_strong = is_strong_beat(beat_pos, meter);
            staged.push(Staged {
                raw_degree: note.degree,
                duration: note.duration,
                is_strong,
                chord_deg,
            });
            beat_pos += note.duration.beats();
        }
    }

    // Pass 1: strong beats snap to the nearest chord tone -- UNLESS the raw
    // degree forms a genuine SUSPENSION: prepared by the immediately
    // preceding note carrying the identical pitch (typically consonant
    // against the previous chord), dissonant against the CURRENT chord, and
    // resolving DOWN by exactly one scale-degree step to the immediately
    // following note. That specific shape (not just "any dissonance") is
    // the textbook suspension (4-3, 7-6, 9-8, ...); an unprepared dissonance,
    // or one that doesn't resolve down by step, has no such justification
    // and is snapped exactly as a plain strong beat would be.
    let mut degrees: Vec<Option<i32>> = Vec::with_capacity(staged.len());
    for i in 0..staged.len() {
        let value = if !staged[i].is_strong {
            staged[i].raw_degree
        } else {
            match staged[i].raw_degree {
                None => None,
                Some(raw) if is_chord_tone(raw, staged[i].chord_deg) => Some(raw),
                Some(raw) => {
                    // The predecessor's ACTUAL final pitch, not its raw
                    // pre-correction value -- `degrees[i-1]` is already
                    // decided at this point in the same left-to-right pass
                    // (for a strong-beat predecessor, that's its own
                    // chord-tone-snapped or suspension-preserved value; for
                    // a weak beat, pass 2 hasn't run yet, so it's still the
                    // raw value either way). A suspension is prepared by
                    // what's really sounding beforehand, not by what an
                    // untransformed motif happened to say.
                    let prepared = i > 0 && degrees[i - 1] == Some(raw);
                    // The follower's finalized value isn't decided yet at
                    // this point in a single forward pass, so its raw value
                    // is used as a proxy -- a disclosed approximation, not a
                    // full fixed-point resolution.
                    let resolves_down = staged
                        .get(i + 1)
                        .and_then(|s| s.raw_degree)
                        .map(|next| raw - next == 1)
                        .unwrap_or(false);
                    if prepared && resolves_down {
                        Some(raw) // a genuine suspension: leave it dissonant
                    } else {
                        Some(nearest_chord_tone(raw, staged[i].chord_deg))
                    }
                }
            }
        };
        degrees.push(value);
    }

    // Pass 2: a weak-beat degree is a legitimate passing tone (moves by
    // step between two different neighbors) or neighboring tone (steps away
    // and back to the same pitch) only if EVERY neighbor it has (previous
    // and/or next -- a rest or phrase boundary just means one side isn't
    // checked) is exactly one scale-degree-step away. A weak-beat degree
    // that leaps has no such justification and gets the same chord-tone
    // snap a strong beat would -- there is no term for an unprepared,
    // unstepped dissonance on a weak beat either.
    const STEP: i32 = 1;
    for i in 0..staged.len() {
        if staged[i].is_strong {
            continue;
        }
        let Some(d) = degrees[i] else { continue };
        let steps_to = |other: Option<Option<i32>>| -> bool {
            match other.flatten() {
                Some(neighbor) => (neighbor - d).abs() == STEP,
                None => true, // no neighbor on this side (rest/boundary): not disqualifying
            }
        };
        let prev = if i > 0 { Some(degrees[i - 1]) } else { None };
        let next = if i + 1 < degrees.len() {
            Some(degrees[i + 1])
        } else {
            None
        };
        if !(steps_to(prev) && steps_to(next)) {
            degrees[i] = Some(nearest_chord_tone(d, staged[i].chord_deg));
        }
    }

    staged
        .into_iter()
        .zip(degrees)
        .map(|(s, degree)| crate::motif::MotifNote {
            degree,
            duration: s.duration,
        })
        .collect()
}

/// Steer the final pitched note to the cadence's melodic goal, choosing the
/// octave that minimizes the leap FROM THE NOTE THAT ACTUALLY LEADS INTO IT
/// (the previous pitched note, skipping rests) -- not the final note's own
/// pre-steering value. Using the final note's own prior degree as the
/// register reference (an earlier version of this function did) can pick an
/// octave close to wherever the raw motif transform happened to leave that
/// one note, while ignoring what the melody was actually doing on approach;
/// a large, unmotivated leap right into the cadence undercuts exactly the
/// "phrase destination" a cadence is supposed to deliver. Falls back to the
/// final note's own value when there's no earlier pitched note (a
/// single-note line).
fn steer_final_cadence(line_notes: &mut [crate::motif::MotifNote], cadence: Cadence) {
    let Some(idx) = line_notes.iter().rposition(|n| n.degree.is_some()) else {
        return;
    };
    let reference = line_notes[..idx]
        .iter()
        .rev()
        .find_map(|n| n.degree)
        .unwrap_or_else(|| line_notes[idx].degree.unwrap());
    line_notes[idx].degree = Some(nearest_octave_of(cadence.melodic_goal(), reference));
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
///
/// `pub(crate)` so [`crate::score_validation`] shares this definition rather
/// than reimplementing it. It previously had no metrical model at all and
/// treated EVERY integer beat as strong, which made an ordinary passing tone on
/// beat 2 of 4/4 a Fatal issue — see `validate_strong_beats`.
pub(crate) fn is_strong_beat(beat_pos: f64, meter: f64) -> bool {
    let within = beat_pos.rem_euclid(meter);
    within < 1e-6 || (within - meter / 2.0).abs() < 1e-6
}

/// True if `raw` is itself a chord tone (root, third, or fifth, or an
/// octave transposition of one) of the diatonic triad on `chord_root_deg`.
fn is_chord_tone(raw: i32, chord_root_deg: i32) -> bool {
    (-1..=1).any(|oct| {
        [0, 2, 4]
            .iter()
            .any(|&offset| chord_root_deg + offset + 7 * oct == raw)
    })
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
    fn a_weak_beat_stepwise_between_two_chord_tones_is_kept_as_a_genuine_passing_tone() {
        // Chord I (tones {1,3,5,...}). Degree 2 on beat 1 (weak) sits
        // exactly one step from both its strong-beat neighbors (1, then
        // 3) -- a textbook passing tone. It must survive unchanged; if
        // weak beats were (incorrectly) snapped like strong beats, degree
        // 2 would collapse to the nearer of {1,3} (both distance 1, ties
        // break toward the first candidate found: 1).
        let m = Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
        ]);
        let notes = realize_over_progression(&[1], 4.0, |_| m.clone());
        let degrees: Vec<Option<i32>> = notes.iter().map(|n| n.degree).collect();
        assert_eq!(
            degrees,
            vec![Some(1), Some(2), Some(3), Some(5)],
            "the passing tone (degree 2) must be preserved, not snapped"
        );
    }

    #[test]
    fn a_weak_beat_that_leaps_from_both_neighbors_is_corrected_to_the_nearest_chord_tone() {
        // Chord I again. Degree 6 on beat 1 (weak) leaps a fifth from its
        // preceding strong-beat neighbor (1) and a third from its
        // following one (3, the strong-beat-snapped value of raw degree
        // 3) -- neither side is a step, so it has no passing-tone
        // justification and must be corrected the same way a strong beat
        // would be: to its nearest chord tone (5, distance 1).
        let m = Motif::from_degrees(&[
            (1, Duration::quarter()),
            (6, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
        ]);
        let notes = realize_over_progression(&[1], 4.0, |_| m.clone());
        let degrees: Vec<Option<i32>> = notes.iter().map(|n| n.degree).collect();
        assert_eq!(
            degrees,
            vec![Some(1), Some(5), Some(3), Some(5)],
            "the unprepared leap (degree 6) must be corrected to its nearest chord tone (5)"
        );
    }

    #[test]
    fn a_weak_beat_neighboring_tone_that_steps_away_and_back_is_kept() {
        // Chord I. Degree 2 sandwiched between two statements of degree 1
        // (the tonic) is a classic neighboring tone (step away, step
        // back to the SAME pitch) -- also legitimate, not just a passing
        // tone between two DIFFERENT chord tones.
        let m = Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (1, Duration::quarter()),
            (5, Duration::quarter()),
        ]);
        let notes = realize_over_progression(&[1], 4.0, |_| m.clone());
        let degrees: Vec<Option<i32>> = notes.iter().map(|n| n.degree).collect();
        assert_eq!(
            degrees[1],
            Some(2),
            "the neighboring tone must be preserved"
        );
    }

    #[test]
    fn the_final_cadence_note_minimizes_the_leap_from_what_actually_precedes_it() {
        // A hand-built line ending high (degree 8, an octave above tonic)
        // then dropping to a far, unrelated register (degree -5) for its
        // OWN pre-steering value at the very last note -- an artificial
        // but representative stand-in for what a motif transform (e.g.
        // inversion) can leave behind. Steering to an Authentic cadence's
        // goal (tonic, degree 1) using the OLD reference (the final note's
        // own prior value, -5) would pick octave -6 -- a 14-scale-step
        // leap from the note that actually leads into it (degree 8). Using
        // the real predecessor as the reference picks octave 8 instead --
        // landing exactly on the tonic with ZERO leap.
        let mut notes = vec![
            crate::motif::MotifNote {
                degree: Some(8),
                duration: Duration::quarter(),
            },
            crate::motif::MotifNote {
                degree: Some(-5),
                duration: Duration::quarter(),
            },
        ];
        steer_final_cadence(&mut notes, Cadence::Authentic);
        assert_eq!(
            notes[1].degree,
            Some(8),
            "the cadence goal's octave must minimize the leap from the true \
             melodic predecessor (degree 8), not the final note's own prior \
             (and musically irrelevant) value"
        );
    }

    #[test]
    fn a_prepared_strong_beat_dissonance_that_resolves_down_by_step_is_a_legitimate_suspension() {
        // Bar 0 (chord I: tones {1,3,5,...}): a whole-note degree 5 -- the
        // fifth of I, consonant. Bar 1 (chord vi: tones {6,8,10,-1,1,3,...}):
        // the SAME pitch (degree 5) is prepared/carried into the new chord,
        // where it is now dissonant (5 is not among vi's tones), then
        // resolves DOWN by exactly one step to degree 4 on the very next
        // note. That is the textbook suspension shape -- it must be left
        // dissonant, not snapped to vi's nearest chord tone (6).
        let m = Motif::from_degrees(&[(5, Duration::whole())]);
        // `realize_over_progression` re-anchors whatever `variant_for`
        // returns by `transpose(chord_deg - 1)` (degree 1 = "the current
        // chord's root" in the motif's own frame) -- for chord vi (6) that's
        // `transpose(5)`, so these degrees (0, -1) become the intended final
        // degrees (5, 4) once re-anchored.
        let resolution =
            Motif::from_degrees(&[(0, Duration::quarter()), (-1, Duration::quarter())]);
        let notes = realize_over_progression(&[1, 6], 4.0, |i| {
            if i == 0 {
                m.clone()
            } else {
                resolution.clone()
            }
        });
        assert_eq!(
            notes[1].degree,
            Some(5),
            "the prepared, step-resolving dissonance (the suspension) must \
             survive unsnapped, not collapse to vi's nearest chord tone (6)"
        );
    }

    #[test]
    fn an_unprepared_strong_beat_dissonance_is_still_snapped_even_if_it_resolves_down_by_step() {
        // Same chords as above, but the bar-0 note is degree 7 (not 5) --
        // so bar 1's degree-5 note is NOT prepared (no identical predecessor
        // pitch carried over), even though it still resolves down by step
        // to degree 4. Without preparation there is no suspension -- it
        // must be snapped to vi's nearest chord tone (6), the same as any
        // other unjustified strong-beat dissonance.
        let m = Motif::from_degrees(&[(7, Duration::whole())]);
        // `realize_over_progression` re-anchors whatever `variant_for`
        // returns by `transpose(chord_deg - 1)` (degree 1 = "the current
        // chord's root" in the motif's own frame) -- for chord vi (6) that's
        // `transpose(5)`, so these degrees (0, -1) become the intended final
        // degrees (5, 4) once re-anchored.
        let resolution =
            Motif::from_degrees(&[(0, Duration::quarter()), (-1, Duration::quarter())]);
        let notes = realize_over_progression(&[1, 6], 4.0, |i| {
            if i == 0 {
                m.clone()
            } else {
                resolution.clone()
            }
        });
        assert_eq!(
            notes[1].degree,
            Some(6),
            "an unprepared dissonance has no suspension justification and \
             must be snapped to the nearest chord tone"
        );
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
    fn functional_harmony_for_grammar_matches_plain_parallel_in() {
        // HarmonicSyntax::Functional must be byte-identical to today's
        // forced-cadence behavior (the no-regression contract for
        // PeriodSentence-mapped families).
        let prog = [1, 4, 5, 1];
        let forced = Period::parallel_in(&germ(), &prog, 4.0, 5);
        let graded = Period::parallel_in_for_grammar(
            &germ(),
            &prog,
            4.0,
            5,
            crate::grammar::HarmonicSyntax::Functional,
        );
        assert_eq!(forced, graded);
    }

    #[test]
    fn non_functional_syntax_honors_a_detectable_natural_cadence() {
        // [.., 4, 1] is a real Plagal close (Cadence::detect(4, 1) ==
        // Plagal) — a non-Functional family must NOT overwrite it with a
        // forced V-I Authentic close the way `parallel_in` would.
        let prog = [1, 6, 4, 1];
        let period = Period::parallel_in_for_grammar(
            &germ(),
            &prog,
            4.0,
            5,
            crate::grammar::HarmonicSyntax::BluesChorus,
        );
        assert_eq!(period.antecedent.progression, prog);
        assert_eq!(period.consequent.progression, prog);
        assert_eq!(period.antecedent.cadence, Cadence::Plagal);
        assert_eq!(period.consequent.cadence, Cadence::Plagal);
        // Contrast: the same progression under Functional harmony still
        // gets forced to V-I.
        let forced = Period::parallel_in_for_grammar(
            &germ(),
            &prog,
            4.0,
            5,
            crate::grammar::HarmonicSyntax::Functional,
        );
        assert_eq!(forced.consequent.cadence, Cadence::Authentic);
        assert_ne!(forced.consequent.progression, prog);
    }

    #[test]
    fn non_functional_syntax_falls_back_when_no_cadence_is_detectable() {
        // [.., 3, 2] forms no recognized cadence (Cadence::detect(3, 2) ==
        // None) — the honest fallback keeps today's forced V-I close rather
        // than inventing new "no cadence" semantics.
        let prog = [1, 6, 3, 2];
        assert_eq!(Cadence::detect(3, 2), None);
        let period = Period::parallel_in_for_grammar(
            &germ(),
            &prog,
            4.0,
            5,
            crate::grammar::HarmonicSyntax::BluesChorus,
        );
        let forced = Period::parallel_in(&germ(), &prog, 4.0, 5);
        assert_eq!(period, forced);
    }

    #[test]
    fn sentence_variant_of_grammar_aware_cadence_matches_period_semantics() {
        let prog = [1, 6, 4, 1];
        let sentence = Period::parallel_sentence_in_for_grammar(
            &germ(),
            &prog,
            4.0,
            5,
            crate::grammar::HarmonicSyntax::JazzTurnaround,
        );
        assert_eq!(sentence.antecedent.cadence, Cadence::Plagal);
        assert_eq!(sentence.antecedent.progression, prog);
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

    #[test]
    fn clip14_chord_tone_snapping_hypothesis() {
        // Real-clip diagnostic (2026-07-24): confirms by direct code call
        // (not hand math) that clip 14's minor-second cluster comes from
        // nearest_chord_tone snapping a strong-beat degree DOWN to the
        // tonic, landing it a semitone from the immediately-following
        // unsnapped weak-beat degree. chord_root=1 (March's progression's
        // first chord); raw hook degrees were [2, 0, -2, -5], onsets
        // [0.0, 1.0, 1.5, 2.0] in a 4/4 bar -- onsets 0.0 and 2.0 are
        // strong beats (is_strong_beat), 1.0/1.5 are weak and stay raw.
        assert_eq!(
            nearest_chord_tone(2, 1),
            1,
            "strong-beat degree 2 snaps to the tonic"
        );
        assert_eq!(
            nearest_chord_tone(-5, 1),
            -6,
            "strong-beat degree -5 snaps to the tonic an octave down"
        );
        // Post-snap sequence: [1 (snapped), 0 (raw), -2 (raw), -6 (snapped)].
        // The first interval (1 -> 0) is a genuine minor second in
        // HarmonicMinor -- confirmed via the crate's own real Scale
        // accessor, not a hand-rolled offset table.
        let key = Key::minor(PitchClass::C).scale(); // Tonality::Minor -> HarmonicMinor
        let p1 = key.degree_pitch(1, 4);
        let p0 = key.degree_pitch(0, 4);
        let interval = p0.midi() as i32 - p1.midi() as i32;
        assert_eq!(
            interval, -1,
            "snapped degree 1 -> raw degree 0 is a real minor second in HarmonicMinor, \
             confirming the chord-tone-snapping hypothesis"
        );
    }
}
