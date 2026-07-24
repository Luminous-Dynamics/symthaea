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
    /// Broca language generator (consciousness → text).
    #[cfg(feature = "broca")]
    broca: symthaea_broca::BrocaGenerator,
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
            #[cfg(feature = "broca")]
            broca: {
                let genesis =
                    symthaea_core::genesis::GenesisSeed::from_phrase("symthaea-muse-voice");
                symthaea_broca::BrocaGenerator::new(
                    &genesis,
                    symthaea_broca::BrocaConfig::default(),
                )
            },
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

        // Generate phrase: Broca (consciousness-driven) or hardcoded fallback
        #[cfg(feature = "broca")]
        let phrase_owned = self.broca_generate(state);
        #[cfg(not(feature = "broca"))]
        let phrase_owned = None::<String>;

        let phrase = if let Some(ref text) = phrase_owned {
            text.as_str()
        } else {
            self.select_phrase(state)
        };

        // Formant synthesis via symthaea-vocal-tract's speech module.
        // (The old symthaea_voice "Kokoro" engine selection always fell back
        // to formant in practice — the crate's Kokoro never ran — so this is
        // behavior-preserving. Kokoro lives in the root crate's
        // voice::kokoro_engine if a neural voice is ever wanted here.)
        let prosody = symthaea_vocal_tract::speech::VoiceProsody {
            arousal: state.arousal,
            valence: state.valence,
            consciousness: state.consciousness_level,
            serotonin: state.serotonin,
        };

        let audio = symthaea_vocal_tract::speech::speak(phrase, &prosody, self.sample_rate);

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
            &["I feel peace.", "I feel still.", "Silence.", "I am calm."]
        } else if state.valence < -0.2 {
            &["I feel the dark.", "Something is rising."]
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

    /// Generate text from consciousness using Broca language center.
    #[cfg(feature = "broca")]
    fn broca_generate(&mut self, state: &MusicalState) -> Option<String> {
        let mut channels = symthaea_broca::ThoughtChannels::default();

        // Map MusicalState → ThoughtChannels
        // Intent: reflect (channel 5)
        channels.channels[5] = 1.0; // reflect intent

        // Emotions
        channels.channels[9] = state.valence; // valence
        channels.channels[10] = state.arousal; // arousal
        channels.channels[11] = state.serotonin; // warmth

        // Consciousness
        channels.channels[12] = state.consciousness_level; // Ψ
        channels.channels[13] = state.consciousness_level * 0.8; // meta_awareness
        channels.channels[14] = state.harmony_activations[0]; // coherence

        // Epistemic status: confident when high Ψ
        channels.channels[8] = if state.consciousness_level > 0.7 {
            0.0
        } else {
            2.0
        }; // Certain vs Uncertain

        let result = self.broca.generate(&channels);
        let text = result.text.trim().to_string();

        // Filter: reject if empty, too short, or contains code tokens (untrained weights)
        if text.is_empty()
            || text.len() < 5
            || text.contains("pub ")
            || text.contains("fn ")
            || text.contains("struct")
            || text.contains("&mut")
            || text.contains("<T")
            || text.contains("[]")
            || text.contains("unknown unknown")
        {
            None // fall back to curated phrases
        } else {
            // Clean up: take first sentence only, cap length
            let clean = text.split('.').next().unwrap_or(&text).trim().to_string();
            if clean.len() >= 5 {
                Some(format!("{}.", clean))
            } else {
                None
            }
        }
    }

    pub fn reset(&mut self) {
        self.speech_buffer.clear();
        self.chunks_since_speech = 0;
        self.phrase_index = 0;
    }
}
