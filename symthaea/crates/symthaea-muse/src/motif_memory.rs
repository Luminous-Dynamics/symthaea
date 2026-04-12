// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Motif memory: tracks phrases, enables repetition and development.
//!
//! The core missing piece for musicality — without repetition, there is no music.
//! This module tracks the last N phrases, scores their repeatability,
//! and decides when to replay vs generate new material.

use crate::{MusicalState, Note};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

const MAX_PHRASES: usize = 8;
const MAX_PHRASE_LEN: usize = 12;

/// A recorded melodic phrase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phrase {
    pub notes: Vec<Note>,
    pub play_count: usize,
}

/// Strategy for what to do with the next phrase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotifStrategy {
    /// Generate entirely new material.
    GenerateNew,
    /// Repeat the best phrase exactly.
    RepeatExact,
    /// Repeat with transposition (shift pitch by interval).
    RepeatTransposed(i32), // semitones
    /// Repeat with variation (alter some notes).
    RepeatVaried,
    /// Play an ostinato (shortest phrase on loop).
    Ostinato,
}

/// Motif memory: tracks phrases and decides repetition strategy.
pub struct MotifMemory {
    phrases: VecDeque<Phrase>,
    current_phrase: Vec<Note>,
    notes_since_phrase_start: usize,
    phrase_len_target: usize,
    /// Notes queued for playback from repetition.
    replay_queue: VecDeque<Note>,
}

impl MotifMemory {
    pub fn new() -> Self {
        Self {
            phrases: VecDeque::with_capacity(MAX_PHRASES),
            current_phrase: Vec::with_capacity(MAX_PHRASE_LEN),
            notes_since_phrase_start: 0,
            phrase_len_target: 4,
            replay_queue: VecDeque::new(),
        }
    }

    /// Record a note into the current phrase.
    pub fn record_note(&mut self, note: Note) {
        self.current_phrase.push(note);
        self.notes_since_phrase_start += 1;

        // End phrase at target length
        if self.notes_since_phrase_start >= self.phrase_len_target {
            self.end_phrase();
        }
    }

    /// End the current phrase and store it.
    fn end_phrase(&mut self) {
        if !self.current_phrase.is_empty() {
            let phrase = Phrase {
                notes: self.current_phrase.clone(),
                play_count: 1,
            };
            self.phrases.push_back(phrase);
            if self.phrases.len() > MAX_PHRASES {
                self.phrases.pop_front();
            }
        }
        self.current_phrase.clear();
        self.notes_since_phrase_start = 0;
    }

    /// Decide what to do next based on consciousness state.
    pub fn decide_strategy(&self, state: &MusicalState) -> MotifStrategy {
        if self.phrases.is_empty() {
            return MotifStrategy::GenerateNew;
        }

        let psi = state.consciousness_level;
        let coherence = state.harmony_activations[0]; // ResonantCoherence
        let play = state.harmony_activations[3]; // InfinitePlay
        let progress = state.harmony_activations[6]; // EvolutionaryProgression
        let stillness = state.harmony_activations[7]; // SacredStillness

        if stillness > 0.6 {
            MotifStrategy::Ostinato
        } else if coherence > 0.6 && psi > 0.5 {
            MotifStrategy::RepeatExact
        } else if play > 0.5 {
            MotifStrategy::RepeatVaried
        } else if progress > 0.5 {
            // Transpose up by 2, 4, 5, or 7 semitones (musically useful intervals)
            let intervals = [2, 4, 5, 7];
            let idx = (progress * 3.0) as usize % intervals.len();
            MotifStrategy::RepeatTransposed(intervals[idx])
        } else if psi < 0.3 {
            MotifStrategy::GenerateNew // low consciousness = searching
        } else {
            // Default: alternate between repeat and new
            if self.phrases.back().map(|p| p.play_count).unwrap_or(0) < 2 {
                MotifStrategy::RepeatExact
            } else {
                MotifStrategy::GenerateNew
            }
        }
    }

    /// Get the next note: either from replay queue or None (caller should generate new).
    pub fn next_note(&mut self, state: &MusicalState) -> Option<Note> {
        // If replay queue has notes, return from it
        if let Some(note) = self.replay_queue.pop_front() {
            return Some(note);
        }

        // Decide strategy
        let strategy = self.decide_strategy(state);

        match strategy {
            MotifStrategy::GenerateNew => None, // caller generates
            MotifStrategy::RepeatExact => {
                if let Some(phrase) = self.phrases.back_mut() {
                    phrase.play_count += 1;
                    // Queue all notes from the phrase
                    for note in &phrase.notes {
                        self.replay_queue.push_back(*note);
                    }
                    self.replay_queue.pop_front()
                } else {
                    None
                }
            }
            MotifStrategy::RepeatTransposed(semitones) => {
                if let Some(phrase) = self.phrases.back_mut() {
                    phrase.play_count += 1;
                    let ratio = 2.0f32.powf(semitones as f32 / 12.0);
                    for note in &phrase.notes {
                        let mut transposed = *note;
                        transposed.frequency *= ratio;
                        self.replay_queue.push_back(transposed);
                    }
                    self.replay_queue.pop_front()
                } else {
                    None
                }
            }
            MotifStrategy::RepeatVaried => {
                if let Some(phrase) = self.phrases.back_mut() {
                    phrase.play_count += 1;
                    for (i, note) in phrase.notes.iter().enumerate() {
                        let mut varied = *note;
                        // Vary every other note slightly (±1 scale step ≈ ±2 semitones)
                        if i % 2 == 1 {
                            let shift = if state.valence > 0.0 { 1.0 } else { -1.0 };
                            varied.frequency *= 2.0f32.powf(shift * 2.0 / 12.0);
                        }
                        self.replay_queue.push_back(varied);
                    }
                    self.replay_queue.pop_front()
                } else {
                    None
                }
            }
            MotifStrategy::Ostinato => {
                // Use shortest stored phrase
                if let Some(phrase) = self.phrases.iter().min_by_key(|p| p.notes.len()) {
                    for note in &phrase.notes {
                        self.replay_queue.push_back(*note);
                    }
                    self.replay_queue.pop_front()
                } else {
                    None
                }
            }
        }
    }

    /// Set phrase length target (consciousness-driven).
    pub fn set_phrase_length(&mut self, state: &MusicalState) {
        // Low consciousness = short fragments (2-3 notes)
        // High consciousness = longer phrases (6-8 notes)
        let psi = state.consciousness_level;
        self.phrase_len_target = (2.0 + psi * 6.0).round() as usize;
        self.phrase_len_target = self.phrase_len_target.clamp(2, MAX_PHRASE_LEN);
    }

    pub fn phrase_count(&self) -> usize {
        self.phrases.len()
    }
    pub fn replay_queue_len(&self) -> usize {
        self.replay_queue.len()
    }

    pub fn reset(&mut self) {
        self.phrases.clear();
        self.current_phrase.clear();
        self.replay_queue.clear();
        self.notes_since_phrase_start = 0;
    }

    /// Snapshot the stored phrases for persistence.
    ///
    /// Only the phrases themselves are saved — the current in-progress phrase,
    /// replay queue, and counters are ephemeral and restart fresh each session.
    pub fn to_snapshot(&self) -> MotifSnapshot {
        MotifSnapshot {
            phrases: self.phrases.iter().cloned().collect(),
        }
    }

    /// Restore phrases from a snapshot (e.g., loaded from disk).
    ///
    /// Phrases are pre-loaded so Symthaea starts with its musical vocabulary
    /// from the previous session. Play counts are preserved so frequently
    /// repeated phrases continue to be favored.
    pub fn from_snapshot(snapshot: &MotifSnapshot) -> Self {
        let mut mem = Self::new();
        for phrase in &snapshot.phrases {
            mem.phrases.push_back(phrase.clone());
        }
        // Cap at MAX_PHRASES if the snapshot is somehow larger
        while mem.phrases.len() > MAX_PHRASES {
            mem.phrases.pop_front();
        }
        mem
    }
}

/// Serializable snapshot of motif memory for cross-session persistence.
///
/// Saved alongside aesthetic memory so Symthaea's musical phrases survive restarts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifSnapshot {
    pub phrases: Vec<Phrase>,
}

impl MotifSnapshot {
    /// Load from a JSON file, returning empty on any failure.
    pub fn load(path: &std::path::Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_else(|| Self {
                phrases: Vec::new(),
            })
    }

    /// Save to a JSON file.
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(path, json).map_err(|e| e.to_string())
    }

    /// True if no phrases are stored.
    pub fn is_empty(&self) -> bool {
        self.phrases.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_note(freq: f32) -> Note {
        Note {
            frequency: freq,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.7,
        }
    }

    #[test]
    fn records_and_stores_phrases() {
        let mut mem = MotifMemory::new();
        mem.phrase_len_target = 3;
        mem.record_note(test_note(261.63));
        mem.record_note(test_note(329.63));
        mem.record_note(test_note(392.00));
        assert_eq!(mem.phrase_count(), 1);
    }

    #[test]
    fn repeat_exact_returns_same_notes() {
        let mut mem = MotifMemory::new();
        mem.phrase_len_target = 2;
        mem.record_note(test_note(261.63));
        mem.record_note(test_note(329.63));

        let state = MusicalState {
            consciousness_level: 0.7,
            harmony_activations: [0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], // high coherence
            ..Default::default()
        };

        let note = mem.next_note(&state);
        assert!(note.is_some());
        assert!((note.unwrap().frequency - 261.63).abs() < 1.0);
    }

    #[test]
    fn transposed_repeat_shifts_pitch() {
        let mut mem = MotifMemory::new();
        mem.phrase_len_target = 2;
        mem.record_note(test_note(261.63));
        mem.record_note(test_note(329.63));

        let state = MusicalState {
            consciousness_level: 0.5,
            harmony_activations: [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.2], // high progression
            ..Default::default()
        };

        let note = mem.next_note(&state);
        assert!(note.is_some());
        // Should be transposed (not exactly 261.63)
        let freq = note.unwrap().frequency;
        assert!(freq != 261.63, "should be transposed: {freq}");
    }

    #[test]
    fn low_consciousness_generates_new() {
        let mut mem = MotifMemory::new();
        mem.phrase_len_target = 2;
        mem.record_note(test_note(261.63));
        mem.record_note(test_note(329.63));

        let state = MusicalState {
            consciousness_level: 0.1,
            ..Default::default()
        };
        assert_eq!(mem.decide_strategy(&state), MotifStrategy::GenerateNew);
    }

    #[test]
    fn stillness_triggers_ostinato() {
        let mut mem = MotifMemory::new();
        mem.phrase_len_target = 2;
        mem.record_note(test_note(261.63));
        mem.record_note(test_note(329.63)); // store a phrase first
        let state = MusicalState {
            harmony_activations: [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8],
            ..Default::default()
        };
        assert_eq!(mem.decide_strategy(&state), MotifStrategy::Ostinato);
    }

    #[test]
    fn phrase_length_scales_with_consciousness() {
        let mut mem = MotifMemory::new();
        mem.set_phrase_length(&MusicalState {
            consciousness_level: 0.1,
            ..Default::default()
        });
        let low = mem.phrase_len_target;
        mem.set_phrase_length(&MusicalState {
            consciousness_level: 0.9,
            ..Default::default()
        });
        let high = mem.phrase_len_target;
        assert!(
            high > low,
            "high Psi should have longer phrases: {high} vs {low}"
        );
    }
}
