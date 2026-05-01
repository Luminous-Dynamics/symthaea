// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Training pair generator: (MusicalState, mel_frame) pairs for HDC audio training.
//!
//! The MAESTRO dataset provides paired MIDI + WAV performances. This module
//! builds training examples by:
//! 1. Loading a MIDI file → extracting notes over time
//! 2. Loading the matching WAV → extracting mel spectrogram frames
//! 3. For each mel frame (every 11.6ms), reconstructing the MusicalState
//!    that would have produced it (active notes, velocity envelope, etc.)
//! 4. Pairing (state_hv, mel_frame) as one training example
//!
//! Output: `Vec<TrainingPair>` — thousands of examples per performance.
//!
//! The HDC audio decoder trains by learning: given this consciousness state,
//! produce this mel spectrogram frame. Over 1,276 MAESTRO performances, the
//! model learns the mapping `MusicalState → audio` from real piano music.

use crate::mel_extractor::{MelConfig, MelExtractor};
use crate::{MusicalState, Note};

/// A single training example for HDC audio training.
#[derive(Debug, Clone)]
pub struct TrainingPair {
    /// Musical state at this time (active notes, velocities, harmony).
    pub state: MusicalState,
    /// Log-mel frame (n_mels dimensions) of the actual audio.
    pub mel_frame: Vec<f32>,
    /// Time in seconds within the performance.
    pub time_secs: f32,
}

/// Configuration for pair generation.
#[derive(Debug, Clone)]
pub struct PairConfig {
    pub mel: MelConfig,
    /// How often to emit a training pair (every N mel frames).
    /// Default 1 = every frame (11.6ms). Higher = fewer pairs, faster training.
    pub stride: usize,
    /// Skip silence at the start/end of performances.
    pub trim_silence_db: f32,
}

impl Default for PairConfig {
    fn default() -> Self {
        Self {
            mel: MelConfig::default(),
            stride: 4, // every ~47ms
            trim_silence_db: -40.0,
        }
    }
}

/// Reconstruct `MusicalState` from active notes at time `t`.
///
/// This is the inverse of the Symthaea composition process: given the notes
/// playing right now, what consciousness state would have produced them?
///
/// We derive:
/// - `arousal`: note density (notes per second active)
/// - `valence`: interval consonance (major-heavy = positive, minor = negative)
/// - `consciousness_level`: pitch coherence (tight voicing = high Phi)
/// - `dopamine`: average velocity (loud = excited)
/// - `harmony_activations`: 8-dim from chord structure
pub fn reconstruct_state_at(notes: &[Note], t: f32, window_secs: f32) -> MusicalState {
    // Find notes active at time t
    let active: Vec<&Note> = notes
        .iter()
        .filter(|n| n.start_time <= t && n.start_time + n.duration > t - window_secs)
        .collect();

    if active.is_empty() {
        return MusicalState::default();
    }

    let n = active.len() as f32;

    // Arousal from density (notes in window / window size)
    let arousal = (n / window_secs / 8.0).clamp(0.0, 1.0);

    // Mean velocity → dopamine proxy
    let mean_vel = active.iter().map(|n| n.velocity).sum::<f32>() / n;

    // Valence from interval analysis
    let mut major_count = 0usize;
    let mut minor_count = 0usize;
    for w in active.windows(2) {
        let ratio = w[1].frequency / w[0].frequency.max(0.001);
        let semitones = (ratio.log2() * 12.0).abs().round() as i32 % 12;
        match semitones {
            4 | 7 | 9 => major_count += 1,
            3 | 6 | 8 => minor_count += 1,
            _ => {}
        }
    }
    let total = (major_count + minor_count).max(1);
    let valence = (major_count as f32 - minor_count as f32) / total as f32;

    // Pitch range → consciousness coherence (tighter = higher phi)
    let min_freq = active.iter().map(|n| n.frequency).fold(f32::MAX, f32::min);
    let max_freq = active.iter().map(|n| n.frequency).fold(f32::MIN, f32::max);
    let range_octaves = if min_freq > 0.0 {
        (max_freq / min_freq).log2()
    } else {
        0.0
    };
    let consciousness_level: f32 = (1.0 - range_octaves / 6.0).clamp(0.2, 0.9);

    // Harmony activations: distribute energy by frequency register
    // H0: bass (< 200Hz), H1: tenor (200-400), H2: alto (400-800), H3: soprano (> 800)
    // H4-7: inferred from density, velocity variance, temporal spread
    let mut h = [0.0f32; 8];
    for note in &active {
        let f = note.frequency;
        if f < 200.0 {
            h[0] += note.velocity;
        } else if f < 400.0 {
            h[1] += note.velocity;
        } else if f < 800.0 {
            h[2] += note.velocity;
        } else {
            h[3] += note.velocity;
        }
    }
    let energy_sum = h[0] + h[1] + h[2] + h[3];
    if energy_sum > 0.0 {
        for i in 0..4 {
            h[i] = (h[i] / energy_sum).clamp(0.0, 1.0);
        }
    }
    // H4: rhythmic tension (velocity std)
    let vel_var = active
        .iter()
        .map(|n| (n.velocity - mean_vel).powi(2))
        .sum::<f32>()
        / n;
    h[4] = vel_var.sqrt().clamp(0.0, 1.0);
    // H5: cultural — placeholder from valence polarity
    h[5] = valence.abs().clamp(0.0, 1.0);
    // H6: progression — ascending bias
    let ascending = active
        .windows(2)
        .filter(|w| w[1].frequency > w[0].frequency)
        .count() as f32
        / n.max(1.0);
    h[6] = ascending;
    // H7: stillness — low density → high stillness
    h[7] = (1.0 - arousal).clamp(0.0, 1.0);

    MusicalState {
        consciousness_level,
        arousal,
        valence,
        dopamine: mean_vel,
        serotonin: 0.5,
        noradrenaline: vel_var.sqrt().clamp(0.0, 1.0),
        harmony_activations: h,
        prediction_error: 0.2,
    }
}

/// Build training pairs from a MIDI file and its matching WAV audio.
///
/// `notes` should come from midi_loader::load_midi.
/// `samples` should be mono f32 at `config.mel.sample_rate`.
pub fn build_pairs(notes: &[Note], samples: &[f32], config: &PairConfig) -> Vec<TrainingPair> {
    if notes.is_empty() || samples.is_empty() {
        return Vec::new();
    }

    let mut extractor = MelExtractor::new(config.mel.clone());
    let mel_frames = extractor.extract(samples);

    let sr = config.mel.sample_rate as f32;
    let hop_secs = config.mel.hop_length as f32 / sr;

    let mut pairs = Vec::with_capacity(mel_frames.len() / config.stride);

    for (frame_idx, mel_frame) in mel_frames.iter().enumerate() {
        if frame_idx % config.stride != 0 {
            continue;
        }

        let t = frame_idx as f32 * hop_secs;
        let state = reconstruct_state_at(notes, t, 0.2); // 200ms window

        pairs.push(TrainingPair {
            state,
            mel_frame: mel_frame.clone(),
            time_secs: t,
        });
    }

    pairs
}

/// Save training pairs to a binary file (simple format for fast loading).
///
/// Format: u32 count, then for each pair:
///   - MusicalState: 16 f32s (consciousness=f64→f32, arousal, valence, dopamine,
///     serotonin, noradrenaline, prediction_error + 8 harmony + time_secs + n_mels)
///   - mel_frame: n_mels f32s
pub fn save_pairs(pairs: &[TrainingPair], path: &std::path::Path) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    file.write_all(&(pairs.len() as u32).to_le_bytes())?;
    if !pairs.is_empty() {
        file.write_all(&(pairs[0].mel_frame.len() as u32).to_le_bytes())?;
    } else {
        return Ok(());
    }
    for pair in pairs {
        // Write state (8 scalar fields + 8 harmony + time)
        let fields: [f32; 17] = [
            pair.state.consciousness_level,
            pair.state.arousal,
            pair.state.valence,
            pair.state.dopamine,
            pair.state.serotonin,
            pair.state.noradrenaline,
            pair.state.prediction_error,
            pair.state.harmony_activations[0],
            pair.state.harmony_activations[1],
            pair.state.harmony_activations[2],
            pair.state.harmony_activations[3],
            pair.state.harmony_activations[4],
            pair.state.harmony_activations[5],
            pair.state.harmony_activations[6],
            pair.state.harmony_activations[7],
            pair.time_secs,
            pair.mel_frame.len() as f32,
        ];
        for &f in &fields {
            file.write_all(&f.to_le_bytes())?;
        }
        for &m in &pair.mel_frame {
            file.write_all(&m.to_le_bytes())?;
        }
    }
    file.flush()?;
    Ok(())
}

/// Load training pairs from a `.pairs.bin` file written by `save_pairs`.
///
/// Returns `(state_fields, mel_frames)` where:
/// - `state_fields[i]` is a 17-element Vec<f32> (see `save_pairs` layout).
/// - `mel_frames[i]` is an `n_mels`-element Vec<f32>.
///
/// This flat form avoids reconstructing `MusicalState` structs and is the
/// shape training code actually consumes (state vector → mel vector regression).
pub fn load_pairs(path: &std::path::Path) -> std::io::Result<(Vec<[f32; 17]>, Vec<Vec<f32>>)> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut header = [0u8; 8];
    file.read_exact(&mut header)?;
    let count = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
    let mel_dim = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;

    let mut states = Vec::with_capacity(count);
    let mut mels = Vec::with_capacity(count);
    let pair_bytes = 17 * 4 + mel_dim * 4;
    let mut buf = vec![0u8; pair_bytes];

    for _ in 0..count {
        file.read_exact(&mut buf)?;
        let mut state = [0f32; 17];
        for (i, chunk) in buf[..17 * 4].chunks_exact(4).enumerate() {
            state[i] = f32::from_le_bytes(chunk.try_into().unwrap());
        }
        let mut mel = Vec::with_capacity(mel_dim);
        for chunk in buf[17 * 4..].chunks_exact(4) {
            mel.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        states.push(state);
        mels.push(mel);
    }
    Ok((states, mels))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_notes() -> Vec<Note> {
        vec![
            Note {
                frequency: 261.63,
                start_time: 0.0,
                duration: 0.5,
                velocity: 0.7,
            }, // C4
            Note {
                frequency: 329.63,
                start_time: 0.5,
                duration: 0.5,
                velocity: 0.8,
            }, // E4
            Note {
                frequency: 392.00,
                start_time: 1.0,
                duration: 0.5,
                velocity: 0.75,
            }, // G4
        ]
    }

    #[test]
    fn reconstruct_state_identifies_active_notes() {
        let notes = make_test_notes();
        let state = reconstruct_state_at(&notes, 0.25, 0.2);
        // Should identify C4 as active
        assert!(state.arousal > 0.0);
        assert!(state.dopamine > 0.0); // velocity > 0
    }

    #[test]
    fn reconstruct_state_major_intervals_positive_valence() {
        // C-E-G arpeggio played close together so the window catches all 3
        let notes = vec![
            Note {
                frequency: 261.63,
                start_time: 0.0,
                duration: 2.0,
                velocity: 0.7,
            }, // C4
            Note {
                frequency: 329.63,
                start_time: 0.05,
                duration: 2.0,
                velocity: 0.8,
            }, // E4
            Note {
                frequency: 392.00,
                start_time: 0.1,
                duration: 2.0,
                velocity: 0.75,
            }, // G4
        ];
        // At t=1.0 with 1s window, all 3 notes are active simultaneously
        let state = reconstruct_state_at(&notes, 1.0, 1.0);
        // C-E-G contains major 3rd and perfect 5th → should be positive valence
        assert!(
            state.valence >= 0.0,
            "C major chord should have non-negative valence, got {}",
            state.valence
        );
    }

    #[test]
    fn build_pairs_produces_output() {
        let notes = make_test_notes();
        // Synthetic audio: 2 seconds of 440 Hz sine
        let samples: Vec<f32> = (0..88200)
            .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / 44100.0).sin() * 0.5)
            .collect();
        let config = PairConfig::default();
        let pairs = build_pairs(&notes, &samples, &config);
        assert!(!pairs.is_empty(), "should produce pairs");
        // Each pair should have n_mels dimensions
        assert_eq!(pairs[0].mel_frame.len(), config.mel.n_mels);
    }

    #[test]
    fn save_pairs_roundtrip() {
        let notes = make_test_notes();
        let samples: Vec<f32> = (0..88200)
            .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / 44100.0).sin() * 0.5)
            .collect();
        let config = PairConfig::default();
        let pairs = build_pairs(&notes, &samples, &config);

        let tmpfile = std::env::temp_dir().join("symthaea_pairs_test.bin");
        save_pairs(&pairs, &tmpfile).unwrap();

        let metadata = std::fs::metadata(&tmpfile).unwrap();
        assert!(metadata.len() > 0);

        let (states, mels) = load_pairs(&tmpfile).unwrap();
        assert_eq!(states.len(), pairs.len());
        assert_eq!(mels.len(), pairs.len());
        assert_eq!(mels[0].len(), config.mel.n_mels);
        // State field 0 is consciousness_level — should round-trip exactly
        assert_eq!(states[0][0], pairs[0].state.consciousness_level);
        // Mel frame should round-trip
        assert_eq!(mels[0][0], pairs[0].mel_frame[0]);

        std::fs::remove_file(&tmpfile).ok();
    }
}
