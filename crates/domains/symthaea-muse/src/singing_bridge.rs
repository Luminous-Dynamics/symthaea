// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Singing bridge: binds Symthaea's music engine (`symthaea-muse`) to its
//! speech engine (`symthaea-voice` + `symthaea-vocal-tract`) so lyrics can be
//! **sung** to a melody, instead of merely spoken.
//!
//! ## Why this exists (2026-07-06 art/culture review, Phase 3.1)
//!
//! Before this module, music and speech were fully siloed:
//! - `symthaea-muse::compose()` produces a `Composition` of `Note`s (pitch +
//!   rhythm), with no concept of lyrics or phonemes.
//! - `symthaea-vocal-tract`'s `FormantFrame::f0` (the sung/spoken pitch) was
//!   only ever driven by *prosody* — stress, declination, valence (see
//!   `symthaea-voice/src/ltc_voice.rs`, explicitly noted there as "not LTC —
//!   LTC F0 is untrained") or by the hand-tuned contour in
//!   `symthaea-voice/src/formants.rs::phonemes_to_frames`. Neither path has
//!   ever taken a melody as input.
//!
//! This is the **first lyric+melody binding** in the codebase. It composes
//! two existing, independently-developed pieces rather than reinventing
//! either:
//! - **What is said** (vowel/consonant shape) still comes from the existing
//!   phoneme→formant-target table,
//!   [`symthaea_voice::formants::formant_target_pub`]
//!   (`symthaea-voice/src/formants.rs:99`) — reused verbatim, not
//!   reimplemented.
//! - **What pitch it's said at** comes from [`Note::frequency`] instead of
//!   from prosody. This is the actual "singing" behavior: today's speech
//!   path always *couples* F0 to prosody; this module *decouples* F0 from
//!   prosody and *couples* it to melody instead.
//!
//! ## Simplification: monosyllabic-per-note
//!
//! Symthaea has no syllabifier (nothing in `symthaea-voice::g2p` splits a
//! word into syllables — CMUdict lookups and the letter-fallback path both
//! return one flat phoneme sequence per *word*). [`sing`] therefore treats
//! each whitespace-separated **word** in the lyrics as one "syllable" bound
//! to exactly one melody note. Multi-syllable words are *not* split across
//! notes, and real melisma (one syllable sustained/ornamented across
//! multiple notes) is out of scope. See [`sing`]'s doc comment for the
//! length-mismatch policy this simplification implies, and the module-level
//! TODO below for the natural follow-up work.
//!
//! **Future extension**: a real syllabifier (e.g. sonority-sequencing over
//! the existing `Phoneme::is_vowel` flags) would let a single word span
//! multiple notes and would let `render_sung_frames` model true melisma
//! (ornamenting a single vowel across a run of notes) rather than the
//! coarser "hold the last syllable's vowel" fallback used here.
//!
//! ## Dependency direction
//!
//! `symthaea-muse` did not previously depend on `symthaea-vocal-tract` or
//! `symthaea-voice`-the-crate for its own types (only `voice_bridge.rs`
//! pulled in `symthaea_voice`'s high-level `speak_with_engine` API). Placing
//! the bridge here (rather than in `symthaea-vocal-tract` or
//! `symthaea-voice`) avoids a dependency cycle: `symthaea-voice` already
//! depends on `symthaea-vocal-tract`, so if the bridge lived in
//! `symthaea-vocal-tract` and needed `symthaea-muse::Note`, that would
//! require `vocal-tract -> muse -> voice -> vocal-tract`, a cycle. Instead,
//! `symthaea-muse` gained a new optional dependency on
//! `symthaea-vocal-tract` (for `FormantFrame`/`SourceType`), gated behind
//! the pre-existing `voice` feature alongside `symthaea-voice` (for
//! `g2p`/`formants`). `symthaea-vocal-tract` itself depends on nothing above
//! it, so this new edge (`muse -> vocal-tract`) is acyclic.

use crate::Note;
use symthaea_vocal_tract::types::FormantFrame;
use symthaea_voice::formants::formant_target_pub;
use symthaea_voice::g2p::{self, Phoneme};

/// Linear F0 attack/release ramp at each syllable's pitch boundary, in
/// milliseconds. Prevents a discontinuous step in fundamental frequency
/// where the melody moves from one note (and therefore one target pitch) to
/// the next. 15ms is short enough to be inaudible as a distinct event but
/// long enough (3+ frames at the 200Hz frame rate used elsewhere in the
/// vocal-tract pipeline) to avoid a hard digital click.
pub const F0_RAMP_MS: f32 = 15.0;

/// One syllable of lyrics bound to one musical note.
///
/// Simplification: monosyllabic-per-note (see module docs). `phonemes` is
/// the *entire* word's phoneme sequence — there is no syllabifier — and
/// `note` is the single melody note this whole word is sung on.
#[derive(Debug, Clone)]
pub struct SungSyllable {
    /// Phoneme sequence for this "syllable" (in practice: one whole word).
    pub phonemes: Vec<Phoneme>,
    /// The melody note this syllable is sung on.
    pub note: Note,
}

/// Grapheme-to-phoneme the `lyrics` and align them 1:1 with `melody`,
/// producing a sequence of sung syllables.
///
/// # Alignment policy
///
/// - `lyrics.split_whitespace()` words are treated as "syllables" (see
///   module docs — there is no real syllabifier yet).
/// - **Fewer syllables than notes**: the *last* syllable's vowel is held
///   across the remaining notes. This mirrors how singing actually works —
///   a singer sustains a vowel across held notes rather than re-articulating
///   a new word for every one. See [`sustain_last_vowel`].
/// - **More syllables than notes**: the excess trailing syllables are
///   **dropped** (with an `eprintln!` warning), rather than panicking or
///   silently truncating the melody.
/// - Empty `lyrics` or empty `melody` produce an empty result.
pub fn sing(lyrics: &str, melody: &[Note]) -> Vec<SungSyllable> {
    let words: Vec<&str> = lyrics.split_whitespace().collect();
    if words.is_empty() || melody.is_empty() {
        return Vec::new();
    }

    let mut syllables: Vec<Vec<Phoneme>> = words
        .iter()
        .map(|word| {
            g2p::text_to_phonemes(word)
                .into_iter()
                .filter(|p| p.ipa != " ")
                .collect::<Vec<_>>()
        })
        .collect();

    if syllables.len() > melody.len() {
        eprintln!(
            "symthaea-muse::singing_bridge::sing: {} syllable(s) but only {} melody note(s) \
             -- dropping {} trailing syllable(s)",
            syllables.len(),
            melody.len(),
            syllables.len() - melody.len()
        );
        syllables.truncate(melody.len());
    }

    let last_syllable = syllables
        .last()
        .cloned()
        .expect("words is non-empty, so syllables (same length, pre-truncate) is non-empty");

    let mut out = Vec::with_capacity(melody.len());
    for (i, note) in melody.iter().enumerate() {
        let phonemes = if i < syllables.len() {
            std::mem::take(&mut syllables[i])
        } else {
            sustain_last_vowel(&last_syllable)
        };
        out.push(SungSyllable {
            phonemes,
            note: *note,
        });
    }
    out
}

/// Build a held-vowel-only phoneme sequence from a syllable, used by [`sing`]
/// to sustain the last syllable across notes once the lyrics run out before
/// the melody does. Falls back to the whole syllable if it contains no
/// vowel (e.g. a consonant-only fallback word).
fn sustain_last_vowel(syllable: &[Phoneme]) -> Vec<Phoneme> {
    match syllable.iter().rev().find(|p| p.is_vowel) {
        Some(vowel) => vec![vowel.clone()],
        None => syllable.to_vec(),
    }
}

/// Approximate voiced/unvoiced classification from the IPA symbol alone.
///
/// `Phoneme` only carries `is_vowel`, not a separate voiced/unvoiced flag for
/// consonants, so this mirrors the same IPA-character-set heuristic already
/// used in production by `symthaea-voice/src/formants.rs::phonemes_to_frames`
/// rather than inventing a new one.
fn is_voiced_phoneme(phoneme: &Phoneme) -> bool {
    phoneme.is_vowel || "mnŋlɹwjvzðbdɡ".contains(phoneme.ipa)
}

/// Render `syllables` into a `FormantFrame` sequence at `frame_rate_hz`.
///
/// For each syllable, its phonemes are evenly divided across the syllable's
/// note's time span (`note.start_time` .. `note.start_time + note.duration`).
/// Formant targets (F1/F2/F3 + manner of articulation) come from the
/// existing phoneme→formant table,
/// [`symthaea_voice::formants::formant_target_pub`], reused unchanged. `f0`
/// tracks `note.frequency` — this is the actual "singing" behavior — with a
/// short linear ramp ([`F0_RAMP_MS`]) at the attack (start of syllable) and
/// release (end of syllable) so pitch doesn't step discontinuously between
/// notes. `voicing` and `energy` are derived from `note.velocity`, scaled by
/// the same ramp envelope (so the whole note fades in/out, not just pitch).
///
/// Assumes `syllables` is in chronological order (non-decreasing
/// `note.start_time`), which is what [`sing`] produces from a `Composition`'s
/// melody. Syllables with no synthesizable phonemes (a degenerate G2P
/// result) produce no frames for that note — a silence gap — rather than
/// panicking.
pub fn render_sung_frames(syllables: &[SungSyllable], frame_rate_hz: f32) -> Vec<FormantFrame> {
    let mut frames = Vec::new();
    if syllables.is_empty() || frame_rate_hz <= 0.0 {
        return frames;
    }

    let frame_dt = 1.0 / frame_rate_hz;
    let ramp_secs = (F0_RAMP_MS / 1000.0).max(0.0);

    for syllable in syllables {
        if syllable.phonemes.is_empty() {
            continue;
        }

        let note = &syllable.note;
        let note_start = note.start_time;
        let note_end = note.start_time + note.duration;
        let target_f0 = note.frequency.max(0.0);
        let level = note.velocity.clamp(0.0, 1.0);

        let n_phonemes = syllable.phonemes.len() as f32;
        let per_phoneme_dur = note.duration.max(0.0) / n_phonemes;

        for (pi, phoneme) in syllable.phonemes.iter().enumerate() {
            let (f1, f2, f3, source) = formant_target_pub(phoneme.ipa);
            let ph_start = note_start + per_phoneme_dur * pi as f32;
            let n_frames = ((per_phoneme_dur / frame_dt).round() as usize).max(1);
            let voiced = is_voiced_phoneme(phoneme);

            for frame_idx in 0..n_frames {
                let t = ph_start + frame_dt * frame_idx as f32;

                // Ramp in/out at the syllable's (i.e. the note's) own
                // boundaries, not the phoneme's -- so a multi-phoneme
                // syllable ramps once at attack and once at release, not
                // once per phoneme.
                let attack = if ramp_secs > 0.0 && t < note_start + ramp_secs {
                    ((t - note_start) / ramp_secs).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                let release = if ramp_secs > 0.0 && t > note_end - ramp_secs {
                    ((note_end - t) / ramp_secs).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                let ramp = attack.min(release).clamp(0.0, 1.0);

                frames.push(FormantFrame {
                    f1,
                    f2,
                    f3,
                    b1: 60.0,
                    b2: 90.0,
                    b3: 150.0,
                    f0: target_f0 * ramp,
                    energy: level * ramp,
                    voicing: if voiced { level * ramp } else { 0.0 },
                    time: t,
                    source_type: source,
                    nasal_zero_freq: 0.0,
                    nasal_zero_bw: 0.0,
                });
            }
        }
    }

    frames
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(frequency: f32, start_time: f32, duration: f32, velocity: f32) -> Note {
        Note {
            frequency,
            start_time,
            duration,
            velocity,
        }
    }

    #[test]
    fn monosyllabic_lyric_tracks_melody_f0() {
        let melody = [
            note(440.0, 0.0, 0.5, 0.8),
            note(880.0, 0.5, 0.5, 0.8),
            note(220.0, 1.0, 0.5, 0.8),
        ];
        let syllables = sing("la la la", &melody);
        assert_eq!(syllables.len(), 3, "1:1 word-to-note alignment");
        for (syl, m) in syllables.iter().zip(melody.iter()) {
            assert_eq!(syl.note.frequency, m.frequency);
            assert!(
                !syl.phonemes.is_empty(),
                "each syllable should have at least one phoneme"
            );
        }

        let frames = render_sung_frames(&syllables, 200.0);
        assert!(!frames.is_empty());

        // Away from the ramp regions, f0 must track the note's frequency
        // exactly (ramp == 1.0 there).
        for m in &melody {
            let mid_time = m.start_time + m.duration / 2.0;
            let closest = frames
                .iter()
                .min_by(|a, b| {
                    (a.time - mid_time)
                        .abs()
                        .partial_cmp(&(b.time - mid_time).abs())
                        .unwrap()
                })
                .expect("frames non-empty");
            assert!(
                (closest.f0 - m.frequency).abs() < 0.01,
                "mid-note f0 {} should equal note frequency {} away from ramp regions",
                closest.f0,
                m.frequency
            );
        }
    }

    #[test]
    fn fewer_syllables_than_notes_sustains_last_vowel() {
        let melody = [
            note(300.0, 0.0, 0.3, 0.7),
            note(400.0, 0.3, 0.3, 0.7),
            note(500.0, 0.6, 0.3, 0.7),
            note(600.0, 0.9, 0.3, 0.7),
        ];
        // Single word, four notes: the last three syllables must be the
        // sustained vowel from the one word we have.
        let syllables = sing("ah", &melody);
        assert_eq!(syllables.len(), 4);

        for extra in &syllables[1..] {
            assert_eq!(
                extra.phonemes.len(),
                1,
                "sustained syllables should be a single held vowel"
            );
            assert!(
                extra.phonemes[0].is_vowel,
                "sustained phoneme should be a vowel"
            );
        }

        // Rendering should still succeed and produce frames spanning all
        // four notes' worth of time.
        let frames = render_sung_frames(&syllables, 100.0);
        assert!(!frames.is_empty());
        let last_note_end = melody.last().unwrap().start_time + melody.last().unwrap().duration;
        assert!(frames.last().unwrap().time <= last_note_end + 0.05);
    }

    #[test]
    fn more_syllables_than_notes_drops_excess_without_panic() {
        let melody = [note(440.0, 0.0, 0.5, 0.8), note(550.0, 0.5, 0.5, 0.8)];
        // Five words, only two notes -- must not panic, must drop the rest.
        let syllables = sing("one two three four five", &melody);
        assert_eq!(
            syllables.len(),
            2,
            "excess syllables beyond the melody length must be dropped"
        );

        let frames = render_sung_frames(&syllables, 100.0);
        assert!(!frames.is_empty());
    }

    #[test]
    fn frame_timestamps_are_monotonic_and_span_melody_duration() {
        let melody = [
            note(261.63, 0.0, 0.25, 0.6),
            note(293.66, 0.25, 0.25, 0.6),
            note(329.63, 0.5, 0.25, 0.6),
            note(349.23, 0.75, 0.25, 0.6),
        ];
        let syllables = sing("doe ray me fa", &melody);
        let frames = render_sung_frames(&syllables, 200.0);
        assert!(!frames.is_empty());

        for pair in frames.windows(2) {
            assert!(
                pair[1].time >= pair[0].time,
                "frame timestamps must be monotonic: {} then {}",
                pair[0].time,
                pair[1].time
            );
        }

        let total_duration = melody.last().unwrap().start_time + melody.last().unwrap().duration;
        let first_time = frames.first().unwrap().time;
        let last_time = frames.last().unwrap().time;
        assert!(
            first_time <= melody[0].start_time + 0.01,
            "first frame should be at (or just after) the melody start"
        );
        assert!(
            last_time <= total_duration + 0.01 && last_time >= total_duration - 0.1,
            "last frame ({last_time}) should be near the melody's total duration ({total_duration})"
        );
    }

    #[test]
    fn empty_inputs_produce_empty_output() {
        assert!(sing("", &[note(440.0, 0.0, 1.0, 0.8)]).is_empty());
        assert!(sing("hello", &[]).is_empty());
        assert!(render_sung_frames(&[], 200.0).is_empty());
    }
}
