// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Root-crate bridge for `symthaea-muse::singing_bridge` — the first
//! lyric+melody binding in the codebase (Phase 3.1 of
//! `ART_CULTURE_REVIEW_AND_PLAN_2026-07-06.md`).
//!
//! `singing_bridge::sing()` + `render_sung_frames()` live in
//! `symthaea-muse` (gated behind its own `voice` feature) and already
//! produce real `symthaea_vocal_tract::types::FormantFrame`s — the exact
//! same type the root crate's own [`crate::voice::vocoder::FormantVocoder`]
//! consumes (re-exported here via `crate::voice::articulatory_synthesizer`).
//! Both pieces existed with zero call sites connecting them (verified
//! 2026-07-10: `symthaea-muse`'s `voice` feature was never enabled by any
//! root-level feature, so `singing_bridge` wasn't even compiled in root
//! builds). This module is the missing three-call chain: `sing` →
//! `render_sung_frames` → `FormantVocoder::synthesize`.
//!
//! Distinct from [`crate::voice::rap::RapSynthesizer`]: that system is
//! rhythm/BPM-driven with a flat base pitch plus a rhyme-emphasis lift, not
//! an actual melodic contour — genuinely different capability, not a
//! duplicate.

use symthaea_muse::Note;
use symthaea_muse::singing_bridge::{render_sung_frames, sing};

use super::vocoder::FormantVocoder;

/// Frames-per-second used to render sung formant frames, matching the
/// articulatory synthesizer's own convention (`ArticulatoryConfig::frame_rate`,
/// 200Hz = 5ms/frame).
const SUNG_FRAME_RATE_HZ: f32 = 200.0;

/// Sing `lyrics` to `melody` and return PCM audio samples.
///
/// `melody` is a sequence of [`Note`]s (pitch/rhythm) from
/// `symthaea_muse::compose()` or any other source; `lyrics` is
/// whitespace-tokenized into one "syllable" per word (see
/// `singing_bridge::sing`'s doc comment for the length-mismatch policy and
/// the monosyllabic-per-note simplification).
///
/// Returns an empty `Vec` if either input is empty — never panics.
pub fn sing_to_pcm(lyrics: &str, melody: &[Note], vocoder: &mut FormantVocoder) -> Vec<f32> {
    let syllables = sing(lyrics, melody);
    let frames = render_sung_frames(&syllables, SUNG_FRAME_RATE_HZ);
    vocoder.synthesize(&frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_melody() -> Vec<Note> {
        // A short ascending three-note melody, matching "la la la" 1:1.
        vec![
            Note {
                frequency: 261.63, // C4
                start_time: 0.0,
                duration: 0.3,
                velocity: 0.8,
            },
            Note {
                frequency: 293.66, // D4
                start_time: 0.3,
                duration: 0.3,
                velocity: 0.8,
            },
            Note {
                frequency: 329.63, // E4
                start_time: 0.6,
                duration: 0.3,
                velocity: 0.8,
            },
        ]
    }

    #[test]
    fn sing_to_pcm_produces_finite_nonempty_audio() {
        let mut vocoder = FormantVocoder::new();
        let audio = sing_to_pcm("la la la", &simple_melody(), &mut vocoder);

        assert!(
            !audio.is_empty(),
            "singing a real lyric to a real melody must produce audio"
        );
        assert!(
            audio.iter().all(|s| s.is_finite()),
            "all PCM samples must be finite"
        );
        assert!(
            audio.iter().any(|&s| s.abs() > 1e-6),
            "audio must not be silent — some sample should be audibly nonzero"
        );
    }

    #[test]
    fn sing_to_pcm_empty_lyrics_produces_empty_audio() {
        let mut vocoder = FormantVocoder::new();
        let audio = sing_to_pcm("", &simple_melody(), &mut vocoder);
        assert!(audio.is_empty(), "no lyrics means nothing to sing");
    }

    #[test]
    fn sing_to_pcm_empty_melody_produces_empty_audio() {
        let mut vocoder = FormantVocoder::new();
        let audio = sing_to_pcm("la la la", &[], &mut vocoder);
        assert!(audio.is_empty(), "no melody means nothing to sing to");
    }
}
