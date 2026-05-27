// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phoneme → formant frame conversion with consciousness-shaped prosody.

use crate::VoiceProsody;
use crate::g2p::Phoneme;
use symthaea_vocal_tract::types::{FormantFrame, SourceType};

/// Convert phonemes to formant frames with prosody applied.
pub fn phonemes_to_frames(
    phonemes: &[Phoneme],
    prosody: &VoiceProsody,
    sample_rate: u32,
) -> Vec<FormantFrame> {
    let mut frames = Vec::new();
    let frame_rate = 200.0; // 200 frames per second (5ms per frame)
    let total_phonemes = phonemes.len() as f32;

    // Base F0: higher for clarity. Symthaea's voice sits in alto range.
    let base_f0 = 180.0 + prosody.consciousness * 40.0; // 180-220 Hz (was 140-200)
    let f0_offset = prosody.valence * 20.0;
    let rate_factor = 0.6 + prosody.arousal * 0.8;

    for (i, phoneme) in phonemes.iter().enumerate() {
        let progress = i as f32 / total_phonemes.max(1.0);

        // Dynamic phoneme timing: vowels sustain, consonants hold enough to be heard
        let base_dur = if phoneme.is_vowel {
            phoneme.base_duration_ms * 1.4 // vowels sustain
        } else if phoneme.ipa == " " {
            phoneme.base_duration_ms
        } else {
            // Consonants need at least 40ms to be audible
            (phoneme.base_duration_ms * 0.8).max(40.0)
        };
        // Stressed syllables get more time
        let stress_stretch = 1.0 + phoneme.stress as f32 * 0.4;
        let duration_ms = base_dur * stress_stretch / rate_factor;
        let num_frames = (duration_ms / 1000.0 * frame_rate).max(1.0) as usize;

        let (f1, f2, f3, source) = formant_target(phoneme.ipa);

        // F0 contour: sentence-level intonation + stress peaks + phrase boundaries
        let stress_boost = phoneme.stress as f32 * 25.0; // bigger stress peaks
        let declination = -progress * 30.0; // steeper pitch fall
        // Phrase-final rise for questions, fall for statements
        let phrase_contour = if progress > 0.85 { -15.0 } else { 0.0 }; // falling end
        // Micro-prosody: vowels slightly higher than consonants
        let vowel_boost = if phoneme.is_vowel { 8.0 } else { -5.0 };
        let f0 = (base_f0 + f0_offset + stress_boost + declination + phrase_contour + vowel_boost)
            .max(60.0);

        // Energy from stress and prosody
        let energy = if phoneme.ipa == " " {
            0.0
        } else {
            let base_energy = if phoneme.is_vowel {
                0.7 + phoneme.stress as f32 * 0.2
            } else {
                // Consonants need strong energy for noise to be heard
                0.9
            };
            base_energy * (1.0 - prosody.serotonin * 0.15)
        };

        // Voicing (vowels and voiced consonants are voiced)
        let voicing = if phoneme.is_vowel || "mnŋlɹwjvzðbdɡ".contains(phoneme.ipa) {
            0.9 + prosody.consciousness * 0.1 // high consciousness = clearer voicing
        } else {
            0.0
        };

        for frame_idx in 0..num_frames {
            let t = frame_idx as f32 / num_frames as f32;

            frames.push(FormantFrame {
                f1: f1 + t * 5.0, // slight movement for naturalness
                f2,
                f3,
                b1: 60.0 + (1.0 - prosody.consciousness) * 40.0, // wider bandwidth at low consciousness
                b2: 80.0,
                b3: 100.0,
                f0: f0.max(50.0),
                energy,
                voicing,
                time: frames.len() as f32 / frame_rate,
                source_type: source,
                nasal_zero_freq: 0.0,
                nasal_zero_bw: 0.0,
            });
        }
    }

    frames
}

/// Map IPA phoneme to formant targets (F1, F2, F3, source type).
/// Values from Peterson & Barney (1952) and Hillenbrand et al. (1995).
pub fn formant_target_pub(ipa: &str) -> (f32, f32, f32, SourceType) {
    formant_target(ipa)
}

fn formant_target(ipa: &str) -> (f32, f32, f32, SourceType) {
    match ipa {
        // Vowels (Peterson & Barney)
        "iː" | "i" => (270.0, 2290.0, 3010.0, SourceType::Vowel),
        "ɪ" => (390.0, 1990.0, 2550.0, SourceType::Vowel),
        "ɛ" => (530.0, 1840.0, 2480.0, SourceType::Vowel),
        "æ" => (660.0, 1720.0, 2410.0, SourceType::Vowel),
        "ɑ" | "ɑː" => (730.0, 1090.0, 2440.0, SourceType::Vowel),
        "ɒ" => (570.0, 840.0, 2410.0, SourceType::Vowel),
        "ɔ" | "ɔː" => (570.0, 840.0, 2410.0, SourceType::Vowel),
        "ʌ" => (640.0, 1190.0, 2390.0, SourceType::Vowel),
        "ʊ" => (440.0, 1020.0, 2240.0, SourceType::Vowel),
        "uː" | "u" => (300.0, 870.0, 2240.0, SourceType::Vowel),
        "ə" => (500.0, 1500.0, 2500.0, SourceType::Vowel), // schwa
        "ɜː" => (580.0, 1380.0, 2500.0, SourceType::Vowel),
        // Diphthongs (approximate with midpoint)
        "eɪ" => (450.0, 2000.0, 2700.0, SourceType::Vowel),
        "aɪ" => (600.0, 1500.0, 2500.0, SourceType::Vowel),
        "oʊ" => (450.0, 1000.0, 2400.0, SourceType::Vowel),
        "aʊ" => (650.0, 1200.0, 2500.0, SourceType::Vowel),
        // Nasals
        "m" => (300.0, 1000.0, 2500.0, SourceType::Nasal),
        "n" => (300.0, 1500.0, 2500.0, SourceType::Nasal),
        "ŋ" => (300.0, 2000.0, 2700.0, SourceType::Nasal),
        // Stops
        "p" | "b" => (200.0, 1000.0, 2500.0, SourceType::Stop),
        "t" | "d" => (200.0, 1800.0, 2600.0, SourceType::Stop),
        "k" | "ɡ" => (200.0, 1300.0, 2500.0, SourceType::Stop),
        // Fricatives
        "f" | "v" => (300.0, 1200.0, 2500.0, SourceType::Fricative),
        "θ" | "ð" => (300.0, 1500.0, 2700.0, SourceType::Fricative),
        "s" | "z" => (300.0, 1800.0, 4500.0, SourceType::Fricative),
        "ʃ" | "ʒ" => (300.0, 1800.0, 3500.0, SourceType::Fricative),
        "h" => (500.0, 1500.0, 2500.0, SourceType::Fricative),
        // Affricates
        "tʃ" | "dʒ" => (300.0, 1800.0, 3500.0, SourceType::Affricate),
        // Liquids
        "l" => (350.0, 1050.0, 2800.0, SourceType::Liquid),
        "ɹ" => (350.0, 1300.0, 1600.0, SourceType::Liquid),
        // Glides
        "w" => (300.0, 800.0, 2300.0, SourceType::Liquid),
        "j" => (280.0, 2200.0, 3000.0, SourceType::Liquid),
        // Silence
        " " | "" => (0.0, 0.0, 0.0, SourceType::Silent),
        // Compound (ks)
        "ks" => (300.0, 1800.0, 4500.0, SourceType::Fricative),
        // Default
        _ => (500.0, 1500.0, 2500.0, SourceType::Vowel),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::g2p;

    #[test]
    fn produces_frames() {
        let ph = g2p::text_to_phonemes("hello");
        let frames = phonemes_to_frames(&ph, &VoiceProsody::default(), 44100);
        assert!(!frames.is_empty(), "should produce frames");
        assert!(
            frames.len() > 10,
            "should produce many frames: {}",
            frames.len()
        );
    }

    #[test]
    fn silence_has_zero_energy() {
        let ph = vec![crate::g2p::Phoneme {
            ipa: " ",
            is_vowel: false,
            stress: 0,
            base_duration_ms: 100.0,
        }];
        let frames = phonemes_to_frames(&ph, &VoiceProsody::default(), 44100);
        assert!(
            frames.iter().all(|f| f.energy < 0.01),
            "silence should have zero energy"
        );
    }

    #[test]
    fn arousal_affects_frame_count() {
        let ph = g2p::text_to_phonemes("hello");
        let calm = phonemes_to_frames(
            &ph,
            &VoiceProsody {
                arousal: 0.1,
                ..Default::default()
            },
            44100,
        );
        let excited = phonemes_to_frames(
            &ph,
            &VoiceProsody {
                arousal: 0.9,
                ..Default::default()
            },
            44100,
        );
        assert!(
            excited.len() < calm.len(),
            "excited should be faster: {} vs {}",
            excited.len(),
            calm.len()
        );
    }
}
