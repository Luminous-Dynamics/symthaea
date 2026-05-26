// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bootstrap heuristics for generating ThoughtChannel targets from speech acoustics.
//!
//! These hand-coded rules serve as **teacher signals** for training the
//! SpeechThoughtEncoder's learned probes. They map acoustic features to
//! ThoughtChannels via simple, interpretable rules. Once the encoder is
//! trained, it produces richer channels that capture temporal dynamics
//! the heuristics miss.
//!
//! Used only by `gen_speech_training` binary, not at inference time.

use crate::encoder::ThoughtChannels;

/// Acoustic features computed from a single utterance.
#[derive(Debug, Clone)]
pub struct UtteranceAcoustics {
    /// Spectral centroid in Hz (brightness proxy).
    pub spectral_centroid: f32,
    /// Speaking rate in phonemes per second.
    pub speaking_rate: f32,
    /// Fraction of utterance that is voiced (0.0–1.0).
    pub voicing_ratio: f32,
    /// Fraction of utterance that is silence (0.0–1.0).
    pub pause_ratio: f32,
    /// Articulation consistency: 1.0 - coefficient of variation of phoneme durations.
    pub articulation_consistency: f32,
    /// Mean RMS energy (log-mel scale).
    pub energy_mean: f32,
    /// Energy variance (dynamic range).
    pub energy_variance: f32,
    /// Total utterance duration in seconds.
    pub duration: f32,
}

/// A phoneme segment with timing (matches symthaea_stt::PhonemeSegment).
#[derive(Debug, Clone)]
pub struct PhonemeInfo {
    pub phoneme: String,
    pub start: f64,
    pub end: f64,
}

/// Compute spectral centroid from mel spectrogram frames.
///
/// Approximates centroid using mel-bin center frequencies as weights.
/// Higher centroid → brighter sound → maps to positive valence.
pub fn spectral_centroid(mel_frames: &[Vec<f32>]) -> f32 {
    if mel_frames.is_empty() {
        return 0.0;
    }

    // Approximate mel-bin center frequencies (80–7600 Hz, 40 bins)
    let mel_centers: Vec<f32> = (0..40)
        .map(|i| 80.0 + (7600.0 - 80.0) * (i as f32 / 39.0))
        .collect();

    let mut weighted_sum = 0.0f64;
    let mut total_energy = 0.0f64;

    for frame in mel_frames {
        for (i, &val) in frame.iter().take(40).enumerate() {
            let energy = val.max(0.0) as f64;
            weighted_sum += energy * mel_centers[i] as f64;
            total_energy += energy;
        }
    }

    if total_energy > 1e-10 {
        (weighted_sum / total_energy) as f32
    } else {
        0.0
    }
}

/// Compute speaking rate from phoneme segments (non-silence phonemes/second).
pub fn speaking_rate(phonemes: &[PhonemeInfo]) -> f32 {
    if phonemes.is_empty() {
        return 0.0;
    }

    let non_silence: usize = phonemes.iter().filter(|p| !is_silence(&p.phoneme)).count();

    let total_duration = phonemes.last().map(|p| p.end).unwrap_or(0.0)
        - phonemes.first().map(|p| p.start).unwrap_or(0.0);

    if total_duration > 0.01 {
        non_silence as f32 / total_duration as f32
    } else {
        0.0
    }
}

/// Compute voicing ratio (voiced duration / total duration).
pub fn voicing_ratio(phonemes: &[PhonemeInfo]) -> f32 {
    if phonemes.is_empty() {
        return 0.0;
    }

    let mut voiced_dur = 0.0f64;
    let mut total_dur = 0.0f64;

    for p in phonemes {
        let dur = (p.end - p.start).max(0.0);
        total_dur += dur;
        if is_voiced(&p.phoneme) {
            voiced_dur += dur;
        }
    }

    if total_dur > 1e-6 {
        (voiced_dur / total_dur) as f32
    } else {
        0.0
    }
}

/// Compute pause ratio (silence duration / total duration).
pub fn pause_ratio(phonemes: &[PhonemeInfo], total_duration: f32) -> f32 {
    if total_duration < 0.01 || phonemes.is_empty() {
        return 0.0;
    }

    let silence_dur: f64 = phonemes
        .iter()
        .filter(|p| is_silence(&p.phoneme))
        .map(|p| (p.end - p.start).max(0.0))
        .sum();

    (silence_dur as f32 / total_duration).clamp(0.0, 1.0)
}

/// Compute articulation consistency (1.0 - CV of non-silence phoneme durations).
///
/// Higher consistency → more confident, measured speech.
pub fn articulation_consistency(phonemes: &[PhonemeInfo]) -> f32 {
    let durations: Vec<f32> = phonemes
        .iter()
        .filter(|p| !is_silence(&p.phoneme))
        .map(|p| (p.end - p.start) as f32)
        .filter(|&d| d > 0.001)
        .collect();

    if durations.len() < 3 {
        return 0.5; // Insufficient data → neutral
    }

    let mean = durations.iter().sum::<f32>() / durations.len() as f32;
    if mean < 1e-6 {
        return 0.5;
    }

    let variance =
        durations.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / (durations.len() - 1) as f32;
    let cv = variance.sqrt() / mean; // Coefficient of variation

    (1.0 - cv).clamp(0.0, 1.0)
}

/// Compute mean and variance of mel frame energy.
pub fn energy_stats(mel_frames: &[Vec<f32>]) -> (f32, f32) {
    if mel_frames.is_empty() {
        return (0.0, 0.0);
    }

    let energies: Vec<f32> = mel_frames
        .iter()
        .map(|frame| frame.iter().map(|v| v * v).sum::<f32>().sqrt())
        .collect();

    let mean = energies.iter().sum::<f32>() / energies.len() as f32;
    let variance = if energies.len() > 1 {
        energies.iter().map(|e| (e - mean).powi(2)).sum::<f32>() / (energies.len() - 1) as f32
    } else {
        0.0
    };

    (mean, variance)
}

/// Classify intent from transcript text (simple rule-based).
///
/// Returns intent index for ThoughtChannels one-hot encoding:
/// 0=acknowledge, 1=answer, 2=clarify, 3=propose, 4=uncertainty,
/// 5=reflect, 6=continue, 7=unknown.
pub fn classify_intent(text: &str) -> usize {
    let trimmed = text.trim();

    if trimmed.ends_with('?') {
        return 2; // clarify
    }

    let words: Vec<&str> = trimmed.split_whitespace().collect();

    if words.len() <= 3 {
        return 0; // acknowledge (very short)
    }
    if words.len() > 30 {
        return 5; // reflect (long exposition)
    }

    // Check for imperative/proposal patterns
    let first_lower = words.first().map(|w| w.to_lowercase()).unwrap_or_default();
    if ["let", "try", "we", "shall", "should", "could"].contains(&first_lower.as_str()) {
        return 3; // propose
    }

    1 // answer (default declarative)
}

/// Map acoustic features + text to bootstrap ThoughtChannels.
///
/// These are teacher targets for training the SpeechThoughtEncoder.
pub fn heuristic_channels(acoustics: &UtteranceAcoustics, text: &str) -> ThoughtChannels {
    let intent = classify_intent(text);
    let mut channels = ThoughtChannels::with_intent(intent);

    // Valence from spectral centroid: bright (high centroid) = positive
    let valence = ((acoustics.spectral_centroid - 1500.0) / 2000.0).clamp(-1.0, 1.0);
    // Arousal from speaking rate: fast = high arousal
    let arousal = ((acoustics.speaking_rate - 5.0) / 10.0).clamp(0.0, 1.0);
    // Warmth from voicing ratio: more voicing = warmer
    let warmth = acoustics.voicing_ratio.clamp(0.0, 1.0);
    channels.set_emotion(valence, arousal, warmth);

    // Epistemic status from pause ratio: more pauses = less certain
    let epistemic = (acoustics.pause_ratio * 4.0).clamp(0.0, 4.0);
    channels.set_epistemic(epistemic);

    // Consciousness: moderate defaults, coherence from energy stability
    let coherence = (1.0 - acoustics.energy_variance.min(1.0)).clamp(0.0, 1.0);
    channels.set_consciousness(0.5, 0.5, coherence);

    // Context channels
    let time_pressure = ((acoustics.speaking_rate - 8.0) / 7.0).clamp(0.0, 1.0);
    let confidence = acoustics.articulation_consistency;
    channels.set_context(time_pressure, 0.5, 0.5, confidence);

    channels
}

/// Check if a phoneme label represents silence.
fn is_silence(phoneme: &str) -> bool {
    matches!(
        phoneme.to_uppercase().as_str(),
        "SIL" | "SPN" | "NSN" | "" | "sp" | "sil"
    )
}

/// Check if a phoneme is voiced (ARPABET classification).
fn is_voiced(phoneme: &str) -> bool {
    // Remove stress markers (e.g., "AH0" → "AH")
    let base: String = phoneme.chars().filter(|c| c.is_alphabetic()).collect();
    let upper = base.to_uppercase();

    // All vowels are voiced
    let vowels = [
        "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
    ];
    if vowels.contains(&upper.as_str()) {
        return true;
    }

    // Voiced consonants
    let voiced_consonants = [
        "B", "D", "DH", "G", "JH", "L", "M", "N", "NG", "R", "V", "W", "Y", "Z", "ZH",
    ];
    voiced_consonants.contains(&upper.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silence_detection() {
        assert!(is_silence("SIL"));
        assert!(is_silence("sil"));
        assert!(is_silence("SPN"));
        assert!(!is_silence("AH0"));
        assert!(!is_silence("T"));
    }

    #[test]
    fn test_voicing_classification() {
        assert!(is_voiced("AH0")); // Vowel
        assert!(is_voiced("B")); // Voiced stop
        assert!(is_voiced("M")); // Nasal
        assert!(!is_voiced("T")); // Voiceless stop
        assert!(!is_voiced("S")); // Voiceless fricative
        assert!(!is_voiced("SIL")); // Silence
    }

    #[test]
    fn test_intent_classification() {
        assert_eq!(classify_intent("What is this?"), 2); // Question
        assert_eq!(classify_intent("Okay."), 0); // Short acknowledge
        assert_eq!(
            classify_intent(
                "The quick brown fox jumps over the lazy dog and then does many other things across the wide open field under a bright blue sky on a warm summer day"
            ),
            5
        ); // Long → reflect
        assert_eq!(classify_intent("Let us try a different approach"), 3); // Propose
        assert_eq!(classify_intent("The cat sat on the mat"), 1); // Default answer
    }

    #[test]
    fn test_heuristic_channels_produces_valid_range() {
        let acoustics = UtteranceAcoustics {
            spectral_centroid: 2000.0,
            speaking_rate: 10.0,
            voicing_ratio: 0.7,
            pause_ratio: 0.1,
            articulation_consistency: 0.8,
            energy_mean: 0.5,
            energy_variance: 0.2,
            duration: 5.0,
        };

        let channels = heuristic_channels(&acoustics, "This is a test sentence.");
        for i in 0..channels.channels.len() {
            let v = channels.channels[i];
            assert!(
                v >= -1.5 && v <= 5.0,
                "Channel {i} out of expected range: {v}"
            );
        }
    }
}
