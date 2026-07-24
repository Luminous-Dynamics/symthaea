// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Singing bridge: binds Symthaea's music engine (`symthaea-muse`) to its
//! speech engine (`symthaea-vocal-tract`, incl. its `speech` module) so
//! lyrics can be **sung** to a melody, instead of merely spoken.
//!
//! (2026-07-15: the `symthaea-voice` crate referenced below was retired; its
//! `g2p`/`formants` moved verbatim into `symthaea_vocal_tract::speech`, which
//! also removed the dependency-cycle concern discussed in "Dependency
//! direction" — muse now needs only `symthaea-vocal-tract`.)
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
//!   [`symthaea_vocal_tract::speech::formants::formant_target_pub`] — reused
//!   verbatim, not reimplemented.
//! - **What pitch it's said at** comes from [`Note::frequency`] instead of
//!   from prosody. This is the actual "singing" behavior: today's speech
//!   path always *couples* F0 to prosody; this module *decouples* F0 from
//!   prosody and *couples* it to melody instead.
//!
//! ## Syllabification (2026-07-18, SYMTHAEA_SINGING_PLAN Phase 2)
//!
//! `symthaea_vocal_tract::speech::g2p::text_to_phonemes` returns one flat
//! phoneme sequence per *word* (CMUdict lookups and the letter-fallback path
//! both do this — neither has ever split a word into syllables). [`sing`]
//! turns that per-word sequence into one or more syllables via
//! [`syllabify`]: a **maximal-onset** split at each vowel nucleus
//! (`Phoneme::is_vowel`) — intervening consonant clusters between two
//! vowels become the ONSET of the following syllable rather than the coda
//! of the preceding one (the standard textbook rule; no phonotactic
//! legality checking beyond that — a deliberate, stated simplification, not
//! a claim of linguistic completeness). A word with N vowels produces N
//! syllables, each bound to its own melody note; a word with zero vowels
//! (a degenerate G2P result) stays one syllable, matching the old
//! monosyllabic-per-note behavior exactly for words that genuinely only
//! have one vowel. [`sing`]'s existing length-mismatch policy (sustain the
//! last vowel / drop excess trailing syllables) is unchanged, now operating
//! on the flattened per-syllable sequence across ALL words rather than
//! per-word.
//!
//! **Still out of scope**: real melisma (ornamenting a single syllable's
//! vowel across a run of held notes, as opposed to one syllable per note)
//! and any cross-syllable stress-driven note-weighting — both natural
//! follow-ups once this lands.
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
use symthaea_vocal_tract::speech::formants::formant_target_pub;
use symthaea_vocal_tract::speech::g2p::{self, Phoneme};
use symthaea_vocal_tract::types::FormantFrame;

/// Linear F0 attack/release ramp at each syllable's pitch boundary, in
/// milliseconds. Prevents a discontinuous step in fundamental frequency
/// where the melody moves from one note (and therefore one target pitch) to
/// the next. 15ms is short enough to be inaudible as a distinct event but
/// long enough (3+ frames at the 200Hz frame rate used elsewhere in the
/// vocal-tract pipeline) to avoid a hard digital click.
pub const F0_RAMP_MS: f32 = 15.0;

/// Vibrato rate in Hz (typical human vocal vibrato: 4.5-6.5 Hz).
///
/// Added 2026-07-18 (SYMTHAEA_SINGING_PLAN Phase 4, reprioritized from
/// "polish" to "likely intelligibility fix"): before this, `f0` was
/// perfectly flat for the sustained portion of every note
/// (`target_f0 * ramp`, no variation once past the attack ramp) — a
/// dead-steady pitch held for hundreds of milliseconds is the acoustic
/// signature of a synthesized tone, not a voice. `examples/
/// singing_intelligibility_gate.rs` found Whisper transcribing sung
/// lyrics as a literal `🎵` (musical-note) glyph — the same caption
/// convention YouTube transcribers use for background music — while the
/// SAME `FormantVocoder` speaking (not singing) the same lyrics with
/// natural prosodic F0 movement scores 98.6% WER
/// (`VOICE_ROUNDTRIP_BASELINE_LTC_2026-07-16.json`). That comparison
/// isolates flat F0, not the vocoder or the ASR, as the most likely
/// cause — vibrato is the direct, minimal fix for it.
pub const VIBRATO_RATE_HZ: f32 = 5.5;

/// Vibrato extent in cents (1/100 semitone) from center pitch — i.e. the
/// modulation swings `target_f0` by ±`VIBRATO_DEPTH_CENTS`. 30 cents
/// (~1.7% frequency deviation) is a moderate, natural extent (trained
/// singers commonly sit in the 25-50 cent range; wider reads as
/// exaggerated/operatic).
pub const VIBRATO_DEPTH_CENTS: f32 = 30.0;

/// One syllable of lyrics bound to one musical note.
///
/// `phonemes` is one syllable's worth of phonemes as produced by
/// [`syllabify`] (a whole word's phonemes if that word is monosyllabic, a
/// slice of them otherwise); `note` is the melody note this syllable is
/// sung on.
#[derive(Debug, Clone)]
pub struct SungSyllable {
    /// Phoneme sequence for this syllable.
    pub phonemes: Vec<Phoneme>,
    /// The melody note this syllable is sung on.
    pub note: Note,
}

/// Count how many syllables [`sing`] will produce for `lyrics`, without
/// requiring a melody. Callers that compose a melody separately (e.g. via
/// `symthaea-muse::theory_realize::compose_and_perform_melody`) can use this
/// to size the melody to the lyrics — an over-long composed melody relative
/// to the syllable count means [`sing`]'s sustain-last-vowel policy stretches
/// the FINAL syllable's vowel across most of the extra notes, which in
/// practice produces mostly-one-held-vowel audio (found via
/// `examples/singing_intelligibility_gate.rs`, SYMTHAEA_SINGING_PLAN Phase 3:
/// a 3-syllable phrase sung to a 22-note melody spent ~85% of its duration on
/// one sustained vowel and Whisper transcribed pure hallucinated filler).
pub fn syllable_count(lyrics: &str) -> usize {
    lyrics
        .split_whitespace()
        .map(|word| {
            let phonemes: Vec<Phoneme> = g2p::text_to_phonemes(word)
                .into_iter()
                .filter(|p| p.ipa != " ")
                .collect();
            syllabify(&phonemes).len()
        })
        .sum()
}

/// Grapheme-to-phoneme the `lyrics`, syllabify each word (see [`syllabify`]),
/// and align the resulting per-syllable sequence 1:1 with `melody`.
///
/// # Alignment policy
///
/// - Each word is split into one syllable per vowel nucleus via
///   [`syllabify`] (see module docs), then all words' syllables are
///   flattened in order into a single sequence, one syllable per melody
///   note.
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
        .flat_map(|word| {
            let phonemes: Vec<Phoneme> = g2p::text_to_phonemes(word)
                .into_iter()
                .filter(|p| p.ipa != " ")
                .collect();
            syllabify(&phonemes)
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

    let last_syllable = syllables.last().cloned().expect(
        "words is non-empty, and syllabify() always returns at least one syllable per word, \
         so syllables (pre-truncate) is non-empty",
    );

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

/// Split one word's flat phoneme sequence into per-syllable groups via the
/// **maximal-onset rule**: a syllable boundary falls right after each vowel
/// nucleus, so every consonant between two vowels becomes part of the
/// FOLLOWING syllable's onset rather than the preceding syllable's coda.
/// Word-initial consonants (before the first vowel) belong to the first
/// syllable; word-final consonants (after the last vowel) belong to the
/// last syllable.
///
/// A word with zero vowels (a degenerate G2P result — e.g. a bare consonant
/// cluster from the letter-fallback path) is returned as a single
/// consonant-only "syllable", matching the old monosyllabic-per-note
/// fallback exactly. A word with exactly one vowel also returns a single
/// syllable — byte-identical to the pre-syllabifier behavior for every
/// genuinely monosyllabic word (e.g. "la", "ah", "one").
///
/// Deliberately does NOT check onset-cluster phonotactic legality (e.g.
/// English disallows onsets like /tl/) — pure maximal-onset splitting, a
/// stated simplification, not a claim of linguistic completeness. See
/// module docs.
///
/// Never panics; an empty `phonemes` slice returns an empty `Vec`.
fn syllabify(phonemes: &[Phoneme]) -> Vec<Vec<Phoneme>> {
    if phonemes.is_empty() {
        return Vec::new();
    }

    let vowel_indices: Vec<usize> = phonemes
        .iter()
        .enumerate()
        .filter(|(_, p)| p.is_vowel)
        .map(|(i, _)| i)
        .collect();

    if vowel_indices.is_empty() {
        // No vowel nucleus to split around -- the whole word is one
        // consonant-only "syllable" (matches the old fallback).
        return vec![phonemes.to_vec()];
    }

    let mut syllables = Vec::with_capacity(vowel_indices.len());
    let mut start = 0;
    for (k, &vowel_idx) in vowel_indices.iter().enumerate() {
        let end = if k + 1 < vowel_indices.len() {
            // Maximal onset: this syllable ends right after its own vowel,
            // so every consonant up to (not including) the NEXT vowel
            // becomes that next syllable's onset.
            vowel_idx + 1
        } else {
            // Last syllable: absorb all remaining phonemes, including any
            // word-final coda consonants after the last vowel.
            phonemes.len()
        };
        syllables.push(phonemes[start..end].to_vec());
        start = end;
    }
    syllables
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
/// For each syllable, its phonemes are timed within the syllable's note's
/// time span (`note.start_time` .. `note.start_time + note.duration`) —
/// consonants get their natural brief duration (same formula the speaking
/// path uses) and the vowel absorbs whatever time remains, rather than
/// evenly splitting the note across every phoneme (2026-07-18,
/// SYMTHAEA_SINGING_PLAN Phase 5). Formant targets (F1/F2/F3 + manner of
/// articulation) come from the
/// existing phoneme→formant table,
/// [`symthaea_vocal_tract::speech::formants::formant_target_pub`], reused unchanged. `f0`
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

        // Per-phoneme durations: brief, natural consonants (same formula as
        // the speaking path, which scores 98.6% WER -- `(base_duration_ms *
        // 0.8).max(40.0)` ms) with the VOWEL absorbing the note's remaining
        // time, instead of evenly splitting the note across every phoneme
        // including consonants (which held e.g. a consonant like "h" for
        // 100s of ms on a typical note -- 3-6x longer than natural, in
        // speech OR singing). See SYMTHAEA_SINGING_PLAN_2026-07-18.md
        // Phase 5.
        let vowel_idx = syllable.phonemes.iter().position(|p| p.is_vowel);
        let phoneme_durs: Vec<f32> = if let Some(vi) = vowel_idx {
            const CONSONANT_FLOOR_MS: f32 = 40.0;
            const VOWEL_FLOOR_SECS: f32 = 0.06;
            let mut durs = vec![0.0f32; syllable.phonemes.len()];
            let mut consonant_total = 0.0f32;
            for (i, p) in syllable.phonemes.iter().enumerate() {
                if i != vi {
                    let d = (p.base_duration_ms * 0.8).max(CONSONANT_FLOOR_MS) / 1000.0;
                    durs[i] = d;
                    consonant_total += d;
                }
            }
            durs[vi] = (note.duration.max(0.0) - consonant_total).max(VOWEL_FLOOR_SECS);
            durs
        } else {
            // No vowel nucleus (a degenerate all-consonant syllable) --
            // fall back to the even split; there's no vowel to anchor a
            // natural timing shape on.
            let n_phonemes = syllable.phonemes.len() as f32;
            let per = note.duration.max(0.0) / n_phonemes;
            vec![per; syllable.phonemes.len()]
        };

        let mut ph_offset = 0.0f32;
        for (pi, phoneme) in syllable.phonemes.iter().enumerate() {
            let (f1, f2, f3, source) = formant_target_pub(phoneme.ipa);
            let this_dur = phoneme_durs[pi];
            let ph_start = note_start + ph_offset;
            ph_offset += this_dur;
            let n_frames = ((this_dur / frame_dt).round() as usize).max(1);
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

                // Vibrato: a sinusoidal cents-scale F0 modulation, gated by
                // the SAME ramp envelope as attack/release (so it's near-
                // zero right at a note's edges and full depth mid-note --
                // a natural "grows in, fades out" vibrato shape, not a
                // discontinuous on/off switch). Absolute time `t` (not
                // time-since-note-start) keeps the LFO phase continuous
                // across the whole utterance, avoiding any click at note
                // boundaries. See VIBRATO_RATE_HZ/VIBRATO_DEPTH_CENTS docs.
                let vibrato_cents =
                    VIBRATO_DEPTH_CENTS * (2.0 * std::f32::consts::PI * VIBRATO_RATE_HZ * t).sin();
                let vibrato_ratio = 2.0_f32.powf((vibrato_cents * ramp) / 1200.0);

                frames.push(FormantFrame {
                    f1,
                    f2,
                    f3,
                    b1: 60.0,
                    b2: 90.0,
                    b3: 150.0,
                    f0: target_f0 * ramp * vibrato_ratio,
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
    fn monosyllabic_lyric_tracks_melody_f0_within_vibrato_envelope() {
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

        // Away from the ramp regions, f0 must stay within the vibrato
        // envelope around the note's target frequency -- vibrato means it
        // deliberately no longer equals the target exactly, just
        // oscillates near it (see VIBRATO_DEPTH_CENTS).
        let max_ratio = 2.0_f32.powf(VIBRATO_DEPTH_CENTS / 1200.0);
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
            let ratio = closest.f0 / m.frequency;
            assert!(
                ratio <= max_ratio + 1e-4 && ratio >= 1.0 / max_ratio - 1e-4,
                "mid-note f0 {} should stay within the vibrato envelope of note \
                 frequency {} (ratio {ratio}, envelope [{}, {}])",
                closest.f0,
                m.frequency,
                1.0 / max_ratio,
                max_ratio
            );
        }
    }

    #[test]
    fn vibrato_produces_genuine_f0_variation_within_a_sustained_note() {
        // A single long note (well past the attack/release ramps) should
        // show real F0 movement, not a dead-flat pitch -- the whole point
        // of adding vibrato.
        let melody = [note(440.0, 0.0, 1.0, 0.8)];
        let syllables = sing("ah", &melody);
        let frames = render_sung_frames(&syllables, 200.0);
        let sustained: Vec<f32> = frames
            .iter()
            .filter(|f| f.time > 0.1 && f.time < 0.9)
            .map(|f| f.f0)
            .collect();
        assert!(
            sustained.len() > 10,
            "need enough sustained-region frames to observe variation, got {}",
            sustained.len()
        );
        let min_f0 = sustained.iter().cloned().fold(f32::MAX, f32::min);
        let max_f0 = sustained.iter().cloned().fold(f32::MIN, f32::max);
        assert!(
            max_f0 - min_f0 > 1.0,
            "vibrato should produce measurable F0 variation within a sustained \
             note (got range {min_f0} to {max_f0})"
        );
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

    fn phoneme(ipa: &'static str, is_vowel: bool) -> Phoneme {
        Phoneme {
            ipa,
            is_vowel,
            stress: 0,
            base_duration_ms: 100.0,
        }
    }

    #[test]
    fn syllabify_empty_input_is_empty() {
        assert!(syllabify(&[]).is_empty());
    }

    #[test]
    fn syllable_count_matches_hand_verified_values() {
        // "hello" has a real dict entry with 2 vowels (h,ɛ,l,oʊ -- see
        // sing_splits_a_real_multi_syllable_dict_word_across_two_notes);
        // "world" has a real dict entry with 1 vowel (w,ɜː,l,d). All other
        // words here go through the letter-fallback path and are confirmed
        // single-vowel elsewhere in this test module (monosyllabic_lyric_
        // tracks_melody_f0, fewer_syllables_than_notes_sustains_last_vowel).
        assert_eq!(syllable_count("hello world"), 3);
        assert_eq!(syllable_count("hello"), 2);
        assert_eq!(syllable_count("world"), 1);
        assert_eq!(syllable_count("la la la"), 3);
        assert_eq!(syllable_count("ah"), 1);
    }

    #[test]
    fn syllable_count_sized_melody_leaves_no_note_unused_by_a_real_syllable() {
        // The property syllable_count exists to guarantee: giving sing() a
        // melody with EXACTLY syllable_count(lyrics) notes should consume
        // every note with a genuine (non-repeated) syllable -- no note
        // beyond the first should carry the SAME phoneme sequence as its
        // predecessor (which is what the sustain-last-vowel fallback would
        // produce if the melody had MORE notes than syllables).
        let lyrics = "hello world";
        let n = syllable_count(lyrics);
        assert_eq!(n, 3);
        let melody: Vec<Note> = (0..n)
            .map(|i| note(440.0 + i as f32 * 50.0, i as f32 * 0.3, 0.3, 0.8))
            .collect();
        let syllables = sing(lyrics, &melody);
        assert_eq!(syllables.len(), n);
        for pair in syllables.windows(2) {
            let a: Vec<&str> = pair[0].phonemes.iter().map(|p| p.ipa).collect();
            let b: Vec<&str> = pair[1].phonemes.iter().map(|p| p.ipa).collect();
            assert_ne!(
                a, b,
                "consecutive syllables must differ when the melody is sized \
                 to the real syllable count -- identical phonemes would mean \
                 one of them is an unwanted sustain-repeat, not a real syllable"
            );
        }
    }

    #[test]
    fn syllable_count_empty_lyrics_is_zero() {
        assert_eq!(syllable_count(""), 0);
    }

    #[test]
    fn syllabify_zero_vowels_is_one_consonant_only_syllable() {
        let phonemes = vec![phoneme("s", false), phoneme("t", false)];
        let syllables = syllabify(&phonemes);
        assert_eq!(syllables.len(), 1);
        assert_eq!(syllables[0].len(), 2);
    }

    #[test]
    fn syllabify_single_vowel_is_one_syllable() {
        // "ah": æ(vowel), h(consonant) -- matches the real letter-fallback
        // output for the existing fewer_syllables_than_notes_sustains_last_vowel
        // test, confirming this case is unaffected by the syllabifier.
        let phonemes = vec![phoneme("æ", true), phoneme("h", false)];
        let syllables = syllabify(&phonemes);
        assert_eq!(syllables.len(), 1);
        assert_eq!(syllables[0].len(), 2);
    }

    #[test]
    fn syllabify_applies_maximal_onset_between_two_vowels() {
        // Synthetic "h-ɛ-l-oʊ" (the real dict entry for "hello"): the
        // consonant between the two vowels ('l') must land in the SECOND
        // syllable's onset, not the first syllable's coda.
        let phonemes = vec![
            phoneme("h", false),
            phoneme("ɛ", true),
            phoneme("l", false),
            phoneme("oʊ", true),
        ];
        let syllables = syllabify(&phonemes);
        assert_eq!(syllables.len(), 2, "two vowel nuclei -> two syllables");
        assert_eq!(
            syllables[0].iter().map(|p| p.ipa).collect::<Vec<_>>(),
            vec!["h", "ɛ"],
            "first syllable keeps its own onset + vowel, nothing more"
        );
        assert_eq!(
            syllables[1].iter().map(|p| p.ipa).collect::<Vec<_>>(),
            vec!["l", "oʊ"],
            "the intervocalic consonant 'l' must move to the SECOND \
             syllable's onset (maximal onset), not stay in the first \
             syllable's coda"
        );
    }

    #[test]
    fn syllabify_handles_a_word_initial_cluster_and_final_coda() {
        // Synthetic 3-consonant-onset + vowel + 2-consonant-coda, single
        // vowel: word-initial consonants belong to the (only) syllable's
        // onset, word-final consonants belong to its coda -- no splitting
        // happens since there's only one vowel nucleus.
        let phonemes = vec![
            phoneme("s", false),
            phoneme("t", false),
            phoneme("ɹ", false),
            phoneme("iː", true),
            phoneme("m", false),
            phoneme("z", false),
        ];
        let syllables = syllabify(&phonemes);
        assert_eq!(syllables.len(), 1);
        assert_eq!(
            syllables[0].len(),
            6,
            "all 6 phonemes stay in the one syllable"
        );
    }

    #[test]
    fn sing_splits_a_real_multi_syllable_dict_word_across_two_notes() {
        // "hello" has a real dict entry (h, ɛ, l, oʊ) with two vowels --
        // before the syllabifier, this whole word bound to exactly ONE
        // note; now it must span two, each carrying its own syllable's
        // phonemes and each tracking its own note's pitch.
        let melody = [note(300.0, 0.0, 0.4, 0.8), note(500.0, 0.4, 0.4, 0.8)];
        let syllables = sing("hello", &melody);
        assert_eq!(
            syllables.len(),
            2,
            "\"hello\" (2 vowels) sung to 2 notes should use BOTH notes for \
             its own two syllables, not sustain a single word-syllable"
        );
        assert_eq!(syllables[0].note.frequency, 300.0);
        assert_eq!(syllables[1].note.frequency, 500.0);
        assert_ne!(
            syllables[0]
                .phonemes
                .iter()
                .map(|p| p.ipa)
                .collect::<Vec<_>>(),
            syllables[1]
                .phonemes
                .iter()
                .map(|p| p.ipa)
                .collect::<Vec<_>>(),
            "the two syllables should carry genuinely different phonemes \
             (\"he\" vs \"llo\"), not the same word repeated"
        );

        let frames = render_sung_frames(&syllables, 200.0);
        assert!(!frames.is_empty());
        assert!(frames.iter().all(|f| f.f0.is_finite()));
    }

    #[test]
    fn render_sung_frames_gives_consonants_brief_natural_duration_not_an_even_split() {
        // A long note (1.0s) with a 2-phoneme syllable (consonant "h" +
        // vowel "ɛ") -- the OLD even-split behavior would hold "h" for
        // 0.5s (half the note); the new behavior should hold it for its
        // natural short duration and let the vowel absorb the rest.
        let syllable = SungSyllable {
            phonemes: vec![phoneme("h", false), phoneme("ɛ", true)],
            note: note(440.0, 0.0, 1.0, 0.8),
        };
        let frames = render_sung_frames(&[syllable], 200.0);
        assert!(!frames.is_empty());

        let (h_f1, _, _, _) = formant_target_pub("h");
        let consonant_frames = frames.iter().filter(|f| f.f1 == h_f1).count();
        let consonant_secs = consonant_frames as f32 / 200.0;
        // The phoneme() test helper sets base_duration_ms = 100.0, so the
        // natural consonant duration is (100.0 * 0.8).max(40.0) / 1000.0
        // = 0.08s (16 frames at 200Hz). Allow slack for frame rounding.
        assert!(
            consonant_secs < 0.15,
            "consonant should occupy well under 150ms of a 1.0s note \
             (natural duration ~80ms), got {consonant_secs}s -- the old \
             even-split behavior would have given it 0.5s"
        );

        let vowel_secs = (frames.len() - consonant_frames) as f32 / 200.0;
        assert!(
            vowel_secs > 0.8,
            "the vowel should absorb most of the note's 1.0s duration, got {vowel_secs}s"
        );
    }
}
