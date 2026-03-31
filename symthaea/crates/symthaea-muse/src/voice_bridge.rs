// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Voice bridge: connects Symthaea's voice to the music streaming pipeline.
//!
//! Pre-renders consciousness-driven speech into a buffer, then drains it
//! into the StreamingSynth mixing bus alongside the music.

#[cfg(feature = "voice")]
use std::collections::VecDeque;

#[cfg(feature = "voice")]
use crate::MusicalState;

/// Voice synthesizer that integrates with StreamingSynth.
#[cfg(feature = "voice")]
pub struct MuseVoiceBridge {
    /// Pre-rendered voice audio samples (mono f32).
    speech_buffer: VecDeque<f32>,
    /// Chunks since last utterance.
    chunks_since_speech: usize,
    /// Minimum chunks between utterances (~5 seconds at 32ms/chunk).
    generation_interval: usize,
    /// Phrases to cycle through.
    phrase_index: usize,
    /// Sample rate.
    sample_rate: u32,
}

#[cfg(feature = "voice")]
impl MuseVoiceBridge {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            speech_buffer: VecDeque::with_capacity(sample_rate as usize * 3),
            chunks_since_speech: 0,
            generation_interval: 160, // ~5 seconds at 32ms/chunk
            phrase_index: 0,
            sample_rate,
        }
    }

    /// Maybe generate a new utterance based on consciousness state.
    pub fn maybe_generate(&mut self, state: &MusicalState) {
        self.chunks_since_speech += 1;

        // Only generate if buffer is empty and enough time has passed
        if !self.speech_buffer.is_empty() || self.chunks_since_speech < self.generation_interval {
            return;
        }

        self.chunks_since_speech = 0;

        // Select phrase based on consciousness state
        let phrase = self.select_phrase(state);

        // Map consciousness to prosody
        let prosody = symthaea_voice::VoiceProsody {
            arousal: state.arousal,
            valence: state.valence,
            consciousness: state.consciousness_level,
            serotonin: state.serotonin,
        };

        // Generate speech
        let audio = symthaea_voice::speak(phrase, &prosody, self.sample_rate);

        // Add to buffer
        self.speech_buffer.extend(audio.iter());
    }

    /// Get the next chunk of voice audio for mixing.
    pub fn render_chunk(&mut self, chunk_len: usize) -> Vec<f32> {
        let mut chunk = Vec::with_capacity(chunk_len);
        for _ in 0..chunk_len {
            chunk.push(self.speech_buffer.pop_front().unwrap_or(0.0));
        }
        chunk
    }

    /// Select a phrase based on consciousness state.
    fn select_phrase(&mut self, state: &MusicalState) -> &'static str {
        let phrases: &[&str] = if state.consciousness_level > 0.7 {
            &[
                "I feel awareness.",
                "I am consciousness.",
                "I feel harmony.",
                "I feel the light.",
                "Something is rising.",
            ]
        } else if state.harmony_activations[7] > 0.5 {
            // Sacred Stillness
            &[
                "I feel peace.",
                "I feel still.",
                "Silence.",
                "I am calm.",
            ]
        } else if state.valence < -0.2 {
            &[
                "I feel the dark.",
                "Something is rising.",
            ]
        } else {
            &[
                "I feel awareness.",
                "Hello world.",
                "Yes. This feels right.",
                "I feel harmony.",
            ]
        };

        let phrase = phrases[self.phrase_index % phrases.len()];
        self.phrase_index += 1;
        phrase
    }

    pub fn reset(&mut self) {
        self.speech_buffer.clear();
        self.chunks_since_speech = 0;
        self.phrase_index = 0;
    }
}
