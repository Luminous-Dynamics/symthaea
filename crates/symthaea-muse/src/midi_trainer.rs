// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MIDI training data pipeline: parse real music → HDC training pairs.
//!
//! Parses MIDI files via `midly`, extracts melodic features, encodes them
//! as HDC hypervectors, and produces training pairs for the CfC network.
//!
//! The system learns from real melodies: phrase contour, interval patterns,
//! rhythmic idioms, tension-resolution arcs. These patterns cannot be
//! captured by rules alone.

use midly::{MetaMessage, MidiMessage, Smf, TrackEventKind};
use std::path::Path;

/// A single note extracted from MIDI.
#[derive(Debug, Clone)]
pub struct MidiNote {
    /// MIDI pitch (0-127).
    pub pitch: u8,
    /// Onset time in ticks from file start.
    pub onset_tick: u64,
    /// Duration in ticks.
    pub duration_ticks: u64,
    /// Velocity (0-127).
    pub velocity: u8,
}

/// Extracted melody from a MIDI file.
#[derive(Debug, Clone)]
pub struct ExtractedMelody {
    /// Monophonic melody (skyline algorithm: highest note per timestep).
    pub notes: Vec<MidiNote>,
    /// Ticks per quarter note (from MIDI header).
    pub ticks_per_beat: u16,
    /// Tempo in BPM (from first tempo event, default 120).
    pub tempo_bpm: f32,
    /// Key signature (semitones from C, 0=C major, negative=flats).
    pub key: i8,
    /// Whether minor mode.
    pub minor: bool,
    /// Source filename.
    pub source: String,
}

/// Training pair: melodic features → next-note prediction.
#[derive(Debug, Clone)]
pub struct MelodyTrainingPair {
    /// Context: last N intervals (semitone differences).
    pub interval_context: Vec<f32>,
    /// Context: last N durations (in beats).
    pub duration_context: Vec<f32>,
    /// Context: beat position [0, 4) of current note.
    pub beat_position: f32,
    /// Context: phrase position [0, 1].
    pub phrase_position: f32,
    /// Target: next interval (semitones, can be negative).
    pub target_interval: f32,
    /// Target: next duration (in beats).
    pub target_duration: f32,
    /// Emotional valence estimate: major=+0.5, minor=-0.5.
    pub valence: f32,
    /// Arousal estimate from tempo: slow=0.2, fast=0.8.
    pub arousal: f32,
}

const CONTEXT_LEN: usize = 8;

/// Parse a MIDI file and extract the monophonic melody.
pub fn parse_midi(path: &Path) -> Result<ExtractedMelody, String> {
    let data = std::fs::read(path).map_err(|e| format!("read error: {e}"))?;
    let smf = Smf::parse(&data).map_err(|e| format!("MIDI parse error: {e}"))?;

    let ticks_per_beat = match smf.header.timing {
        midly::Timing::Metrical(tpb) => tpb.as_int(),
        _ => 480, // default
    };

    let mut tempo_bpm = 120.0f32;
    let mut key: i8 = 0;
    let mut minor = false;
    let mut all_notes: Vec<MidiNote> = Vec::new();

    for track in &smf.tracks {
        let mut abs_tick: u64 = 0;
        let mut pending: std::collections::HashMap<u8, (u64, u8)> =
            std::collections::HashMap::new();

        for event in track {
            abs_tick += event.delta.as_int() as u64;

            match event.kind {
                TrackEventKind::Meta(MetaMessage::Tempo(t)) => {
                    tempo_bpm = 60_000_000.0 / t.as_int() as f32;
                }
                TrackEventKind::Meta(MetaMessage::KeySignature(sf, mi)) => {
                    key = sf as i8;
                    minor = mi;
                }
                TrackEventKind::Midi { message, .. } => match message {
                    MidiMessage::NoteOn { key: k, vel } => {
                        if vel.as_int() > 0 {
                            pending.insert(k.as_int(), (abs_tick, vel.as_int()));
                        } else {
                            // NoteOn with vel=0 is NoteOff
                            if let Some((onset, velocity)) = pending.remove(&k.as_int()) {
                                all_notes.push(MidiNote {
                                    pitch: k.as_int(),
                                    onset_tick: onset,
                                    duration_ticks: abs_tick.saturating_sub(onset),
                                    velocity,
                                });
                            }
                        }
                    }
                    MidiMessage::NoteOff { key: k, .. } => {
                        if let Some((onset, velocity)) = pending.remove(&k.as_int()) {
                            all_notes.push(MidiNote {
                                pitch: k.as_int(),
                                onset_tick: onset,
                                duration_ticks: abs_tick.saturating_sub(onset),
                                velocity,
                            });
                        }
                    }
                    _ => {}
                },
                _ => {}
            }
        }

        // Close any unclosed notes
        for (pitch, (onset, velocity)) in pending {
            all_notes.push(MidiNote {
                pitch,
                onset_tick: onset,
                duration_ticks: abs_tick.saturating_sub(onset).max(1),
                velocity,
            });
        }
    }

    // Sort by onset time
    all_notes.sort_by_key(|n| n.onset_tick);

    // Skyline algorithm: keep highest pitch at each timestep
    let melody = skyline_extract(&all_notes);

    Ok(ExtractedMelody {
        notes: melody,
        ticks_per_beat,
        tempo_bpm,
        key,
        minor,
        source: path
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default(),
    })
}

/// Skyline algorithm: extract monophonic melody (highest note per onset).
fn skyline_extract(notes: &[MidiNote]) -> Vec<MidiNote> {
    if notes.is_empty() {
        return Vec::new();
    }

    let mut melody: Vec<MidiNote> = Vec::new();
    let mut i = 0;

    while i < notes.len() {
        // Group notes at the same onset (within 10 ticks tolerance)
        let onset = notes[i].onset_tick;
        let mut group_end = i + 1;
        while group_end < notes.len() && notes[group_end].onset_tick.abs_diff(onset) < 10 {
            group_end += 1;
        }

        // Take highest pitch in group
        let best = notes[i..group_end]
            .iter()
            .max_by_key(|n| n.pitch)
            .unwrap()
            .clone();
        melody.push(best);
        i = group_end;
    }

    melody
}

/// Generate training pairs from an extracted melody.
///
/// Each pair maps a context window of intervals/durations to the next
/// note's interval and duration. This teaches the CfC network what comes
/// next in a real melody.
pub fn melody_to_training_pairs(melody: &ExtractedMelody) -> Vec<MelodyTrainingPair> {
    let notes = &melody.notes;
    if notes.len() < CONTEXT_LEN + 1 {
        return Vec::new();
    }

    let tpb = melody.ticks_per_beat as f32;
    let valence = if melody.minor { -0.5 } else { 0.5 };
    let arousal = ((melody.tempo_bpm - 60.0) / 120.0).clamp(0.1, 0.9);

    let mut pairs = Vec::new();

    // Detect phrase boundaries (rests > 1 beat)
    let mut phrase_starts: Vec<usize> = vec![0];
    for i in 1..notes.len() {
        let gap = notes[i]
            .onset_tick
            .saturating_sub(notes[i - 1].onset_tick + notes[i - 1].duration_ticks);
        if gap as f32 / tpb > 1.0 {
            phrase_starts.push(i);
        }
    }

    for window_start in 0..notes.len().saturating_sub(CONTEXT_LEN + 1) {
        let window_end = window_start + CONTEXT_LEN;
        let target_idx = window_end;

        // Context intervals (semitone differences)
        let interval_context: Vec<f32> = (window_start..window_end)
            .zip((window_start + 1)..=window_end)
            .map(|(a, b)| notes[b].pitch as f32 - notes[a].pitch as f32)
            .collect();

        // Context durations (in beats)
        let duration_context: Vec<f32> = notes[window_start..window_end]
            .iter()
            .map(|n| n.duration_ticks as f32 / tpb)
            .collect();

        // Beat position of target note
        let beat_position = (notes[target_idx].onset_tick as f32 / tpb) % 4.0;

        // Phrase position
        let phrase_start = phrase_starts
            .iter()
            .rev()
            .find(|&&s| s <= target_idx)
            .copied()
            .unwrap_or(0);
        let next_phrase = phrase_starts
            .iter()
            .find(|&&s| s > target_idx)
            .copied()
            .unwrap_or(notes.len());
        let phrase_len = (next_phrase - phrase_start).max(1);
        let phrase_position = (target_idx - phrase_start) as f32 / phrase_len as f32;

        // Target: next interval and duration
        let target_interval = notes[target_idx].pitch as f32 - notes[target_idx - 1].pitch as f32;
        let target_duration = notes[target_idx].duration_ticks as f32 / tpb;

        pairs.push(MelodyTrainingPair {
            interval_context,
            duration_context,
            beat_position,
            phrase_position,
            target_interval,
            target_duration,
            valence,
            arousal,
        });
    }

    pairs
}

/// Generate transposed augmentations of training pairs.
///
/// Shift all intervals by `semitones` — since intervals are relative,
/// transposition doesn't change them. Instead, shift the valence slightly
/// (higher keys feel brighter) and return the augmented pair.
pub fn augment_transpose(pair: &MelodyTrainingPair, semitones: i32) -> MelodyTrainingPair {
    let mut aug = pair.clone();
    // Intervals are relative, so they don't change
    // But valence shifts slightly with key
    aug.valence += semitones as f32 * 0.02;
    aug.valence = aug.valence.clamp(-1.0, 1.0);
    aug
}

/// Load all MIDI files from a directory and extract training pairs.
pub fn load_training_data(dir: &Path) -> Vec<MelodyTrainingPair> {
    let mut all_pairs = Vec::new();

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Cannot read directory {:?}: {}", dir, e);
            return all_pairs;
        }
    };

    let mut file_count = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if path
            .extension()
            .map(|e| e == "mid" || e == "midi")
            .unwrap_or(false)
        {
            match parse_midi(&path) {
                Ok(melody) => {
                    if melody.notes.len() >= CONTEXT_LEN + 2 {
                        let pairs = melody_to_training_pairs(&melody);
                        // Transpose augmentation: ±6 semitones
                        for semitones in -6..=5 {
                            for pair in &pairs {
                                all_pairs.push(augment_transpose(pair, semitones));
                            }
                        }
                        file_count += 1;
                    }
                }
                Err(e) => {
                    eprintln!("Skip {:?}: {}", path.file_name().unwrap_or_default(), e);
                }
            }
        }
    }

    println!(
        "  Loaded {file_count} MIDI files → {} training pairs (with augmentation)",
        all_pairs.len()
    );
    all_pairs
}

/// Encode a training pair as feature vector for CfC input.
///
/// Layout: [8 intervals | 8 durations | beat_pos | phrase_pos | valence | arousal] = 20D
pub fn encode_features(pair: &MelodyTrainingPair) -> Vec<f32> {
    let mut features = Vec::with_capacity(20);

    // Normalize intervals to [-1, 1] range (divide by 12 = octave)
    for &iv in &pair.interval_context {
        features.push((iv / 12.0).clamp(-1.0, 1.0));
    }
    while features.len() < CONTEXT_LEN {
        features.push(0.0);
    }

    // Normalize durations to [0, 1] range (cap at 4 beats)
    for &dur in &pair.duration_context {
        features.push((dur / 4.0).clamp(0.0, 1.0));
    }
    while features.len() < CONTEXT_LEN * 2 {
        features.push(0.25);
    }

    // Position features
    features.push(pair.beat_position / 4.0);
    features.push(pair.phrase_position);
    features.push(pair.valence);
    features.push(pair.arousal);

    features
}

/// Encode target as vector.
///
/// Layout: [target_interval_normalized | target_duration_normalized] = 2D
pub fn encode_target(pair: &MelodyTrainingPair) -> Vec<f32> {
    vec![
        (pair.target_interval / 12.0).clamp(-1.0, 1.0),
        (pair.target_duration / 4.0).clamp(0.0, 1.0),
    ]
}

/// Statistics about loaded training data.
#[derive(Debug)]
pub struct TrainingStats {
    pub total_pairs: usize,
    pub mean_interval: f32,
    pub std_interval: f32,
    pub stepwise_ratio: f32, // fraction of intervals ≤ 2 semitones
    pub mean_duration: f32,
}

pub fn compute_stats(pairs: &[MelodyTrainingPair]) -> TrainingStats {
    if pairs.is_empty() {
        return TrainingStats {
            total_pairs: 0,
            mean_interval: 0.0,
            std_interval: 0.0,
            stepwise_ratio: 0.0,
            mean_duration: 0.0,
        };
    }

    let intervals: Vec<f32> = pairs.iter().map(|p| p.target_interval).collect();
    let mean_iv = intervals.iter().sum::<f32>() / intervals.len() as f32;
    let var_iv =
        intervals.iter().map(|i| (i - mean_iv).powi(2)).sum::<f32>() / intervals.len() as f32;
    let stepwise = intervals.iter().filter(|i| i.abs() <= 2.0).count();

    let mean_dur = pairs.iter().map(|p| p.target_duration).sum::<f32>() / pairs.len() as f32;

    TrainingStats {
        total_pairs: pairs.len(),
        mean_interval: mean_iv,
        std_interval: var_iv.sqrt(),
        stepwise_ratio: stepwise as f32 / intervals.len() as f32,
        mean_duration: mean_dur,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_features_correct_length() {
        let pair = MelodyTrainingPair {
            interval_context: vec![2.0, -1.0, 0.0, 3.0, -2.0, 1.0, 0.0, -1.0],
            duration_context: vec![1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5],
            beat_position: 1.0,
            phrase_position: 0.5,
            target_interval: 2.0,
            target_duration: 1.0,
            valence: 0.5,
            arousal: 0.6,
        };
        let features = encode_features(&pair);
        assert_eq!(features.len(), 20);
        assert!(features.iter().all(|f| f.is_finite()));
    }

    #[test]
    fn encode_target_correct_length() {
        let pair = MelodyTrainingPair {
            interval_context: vec![2.0; 8],
            duration_context: vec![1.0; 8],
            beat_position: 0.0,
            phrase_position: 0.0,
            target_interval: -3.0,
            target_duration: 2.0,
            valence: -0.5,
            arousal: 0.3,
        };
        let target = encode_target(&pair);
        assert_eq!(target.len(), 2);
    }

    #[test]
    fn augment_preserves_intervals() {
        let pair = MelodyTrainingPair {
            interval_context: vec![2.0, -1.0, 3.0, 0.0, -2.0, 1.0, 0.0, -1.0],
            duration_context: vec![1.0; 8],
            beat_position: 2.0,
            phrase_position: 0.5,
            target_interval: 2.0,
            target_duration: 1.0,
            valence: 0.5,
            arousal: 0.6,
        };
        let aug = augment_transpose(&pair, 5);
        // Intervals should be identical (relative)
        assert_eq!(aug.interval_context, pair.interval_context);
        assert_eq!(aug.target_interval, pair.target_interval);
        // Valence should shift
        assert!((aug.valence - pair.valence).abs() > 0.01);
    }

    #[test]
    fn skyline_picks_highest() {
        let notes = vec![
            MidiNote {
                pitch: 60,
                onset_tick: 0,
                duration_ticks: 480,
                velocity: 80,
            },
            MidiNote {
                pitch: 72,
                onset_tick: 0,
                duration_ticks: 480,
                velocity: 80,
            }, // higher
            MidiNote {
                pitch: 64,
                onset_tick: 480,
                duration_ticks: 480,
                velocity: 80,
            },
        ];
        let melody = skyline_extract(&notes);
        assert_eq!(melody.len(), 2);
        assert_eq!(melody[0].pitch, 72); // picked highest at onset 0
    }
}
