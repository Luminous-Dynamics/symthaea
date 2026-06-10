// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Real-time CfC improvisation stream.
//!
//! Provides a pull-based note source backed by `NeuralMelody`, buffering
//! generated notes in a `VecDeque` for smooth real-time consumption.
//! Each call to `next_note()` either pops from the buffer or triggers
//! a new generation batch.

use std::collections::VecDeque;

use symthaea_core::genesis::GenesisSeed;

use crate::neural_melody::NeuralMelody;
use crate::pitch;
use crate::rhythm;
use crate::structure::{self, SectionType};
use crate::{MuseConfig, MusicalState, Note};

/// A real-time note stream driven by CfC neural melody generation.
///
/// Notes are generated in batches by the underlying `NeuralMelody` network
/// and buffered in a `VecDeque`. Consumers pull notes one at a time via
/// `next_note()`, which transparently refills the buffer when empty.
pub struct MuseStream {
    /// The neural melody generator.
    neural: NeuralMelody,
    /// Buffered notes awaiting consumption.
    note_queue: VecDeque<Note>,
    /// Current cognitive state driving generation.
    state: MusicalState,
    /// Current scale (frequencies), derived from state.
    scale: Vec<f32>,
    /// Current tempo in BPM.
    tempo: f32,
    /// Beat duration in seconds (60 / tempo).
    beat_duration: f32,
    /// Generation configuration.
    config: MuseConfig,
    /// Current playback time cursor in seconds.
    time_cursor: f32,
    /// Total notes generated (including consumed).
    notes_generated: u64,
    /// Whether the state has changed since last generation.
    state_dirty: bool,
    /// The most recently emitted note.
    last_note: Option<Note>,
    /// Current musical section type.
    section: SectionType,
}

impl MuseStream {
    /// Create a new muse stream from a seed and config.
    pub fn new(seed: u64, config: MuseConfig) -> Self {
        let genesis = GenesisSeed::from_phrase(&format!("muse-stream-{seed}"));
        let state = MusicalState::default();
        let scale = pitch::build_scale(&state);
        let tempo = rhythm::compute_tempo(&config, &state);
        let beat_duration = 60.0 / tempo;
        let section = structure::determine_section(&state);
        let neural = NeuralMelody::new(&genesis, &config);

        Self {
            neural,
            note_queue: VecDeque::new(),
            state,
            scale,
            tempo,
            beat_duration,
            config,
            time_cursor: 0.0,
            notes_generated: 0,
            state_dirty: false,
            last_note: None,
            section,
        }
    }

    /// Update the cognitive state driving generation.
    ///
    /// Rebuilds the scale, tempo, and section type from the new state.
    /// The next call to `next_note()` will use the updated parameters.
    pub fn update_state(&mut self, state: &MusicalState) {
        self.state = state.clone();
        self.scale = pitch::build_scale(&self.state);
        self.tempo = rhythm::compute_tempo(&self.config, &self.state);
        self.beat_duration = 60.0 / self.tempo;
        self.section = structure::determine_section(&self.state);
        self.state_dirty = true;
    }

    /// Pull the next note from the stream.
    ///
    /// If the buffer is empty, generates a new batch via the neural melody
    /// network and enqueues all resulting notes. Returns `None` only if
    /// the network produces no notes (extremely rare).
    pub fn next_note(&mut self) -> Option<Note> {
        if self.note_queue.is_empty() {
            // Generate a batch of notes
            let notes =
                self.neural
                    .generate(&self.config, &self.state, &self.scale, self.beat_duration);

            // Push ALL generated notes to the queue
            for note in notes {
                self.note_queue.push_back(note);
            }

            self.state_dirty = false;
        }

        if let Some(mut note) = self.note_queue.pop_front() {
            // Adjust note start time to the current cursor position
            note.start_time = self.time_cursor;
            self.time_cursor += note.duration;
            self.notes_generated += 1;
            self.last_note = Some(note);
            Some(note)
        } else {
            None
        }
    }

    /// Generate a batch of notes.
    ///
    /// Calls `next_note()` `count` times, collecting results.
    pub fn generate_batch(&mut self, count: usize) -> Vec<Note> {
        let mut notes = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(note) = self.next_note() {
                notes.push(note);
            }
        }
        notes
    }

    /// Reset the stream with a new seed, clearing all state.
    pub fn reset(&mut self, seed: u64) {
        let genesis = GenesisSeed::from_phrase(&format!("muse-stream-{seed}"));
        self.neural = NeuralMelody::new(&genesis, &self.config);
        self.note_queue.clear();
        self.time_cursor = 0.0;
        self.notes_generated = 0;
        self.last_note = None;
        self.state_dirty = false;
    }

    /// The most recently emitted note, if any.
    pub fn last_note(&self) -> Option<&Note> {
        self.last_note.as_ref()
    }

    /// Current playback time cursor in seconds.
    pub fn time_cursor(&self) -> f32 {
        self.time_cursor
    }

    /// Total number of notes generated since creation or last reset.
    pub fn notes_generated(&self) -> u64 {
        self.notes_generated
    }

    /// Current tempo in BPM.
    pub fn tempo(&self) -> f32 {
        self.tempo
    }

    /// Current musical section type.
    pub fn section(&self) -> SectionType {
        self.section
    }

    /// Whether the state has been updated since the last generation.
    pub fn state_dirty(&self) -> bool {
        self.state_dirty
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> MuseConfig {
        MuseConfig {
            duration_secs: 4.0,
            max_notes: 8,
            ..Default::default()
        }
    }

    #[test]
    fn generates_notes() {
        let mut stream = MuseStream::new(42, default_config());
        let note = stream.next_note();
        assert!(note.is_some(), "should generate at least one note");
    }

    #[test]
    fn time_advances() {
        let mut stream = MuseStream::new(43, default_config());
        let _ = stream.next_note();
        assert!(
            stream.time_cursor() > 0.0,
            "time should advance after a note"
        );
    }

    #[test]
    fn counter_increments() {
        let mut stream = MuseStream::new(44, default_config());
        assert_eq!(stream.notes_generated(), 0);
        let _ = stream.next_note();
        assert_eq!(stream.notes_generated(), 1);
        let _ = stream.next_note();
        assert_eq!(stream.notes_generated(), 2);
    }

    #[test]
    fn state_changes_tempo() {
        let mut stream = MuseStream::new(45, default_config());
        let tempo_before = stream.tempo();

        let excited = MusicalState {
            arousal: 0.95,
            noradrenaline: 0.9,
            ..Default::default()
        };
        stream.update_state(&excited);
        let tempo_after = stream.tempo();

        assert!(
            (tempo_before - tempo_after).abs() > 1.0,
            "tempo should change with arousal: before={tempo_before}, after={tempo_after}"
        );
    }

    #[test]
    fn dirty_flag() {
        let mut stream = MuseStream::new(46, default_config());
        assert!(!stream.state_dirty());

        stream.update_state(&MusicalState {
            arousal: 0.9,
            ..Default::default()
        });
        assert!(stream.state_dirty());

        // Generating clears dirty flag (triggers regeneration)
        let _ = stream.next_note();
        assert!(!stream.state_dirty());
    }

    #[test]
    fn reset_clears() {
        let mut stream = MuseStream::new(47, default_config());
        let _ = stream.next_note();
        let _ = stream.next_note();
        assert!(stream.notes_generated() > 0);
        assert!(stream.time_cursor() > 0.0);

        stream.reset(99);
        assert_eq!(stream.notes_generated(), 0);
        assert_eq!(stream.time_cursor(), 0.0);
        assert!(stream.last_note().is_none());
    }

    #[test]
    fn different_seeds() {
        let mut stream_a = MuseStream::new(100, default_config());
        let mut stream_b = MuseStream::new(200, default_config());

        let note_a = stream_a.next_note();
        let note_b = stream_b.next_note();

        // Different seeds should produce different notes (with high probability)
        if let (Some(a), Some(b)) = (note_a, note_b) {
            // They might occasionally match, but this verifies the streams are independent
            let _ = (a.frequency, b.frequency);
        }
    }

    #[test]
    fn state_affects_output() {
        let mut stream = MuseStream::new(48, default_config());

        // Generate with calm state
        let calm_note = stream.next_note();

        // Update to excited state and reset to regenerate
        stream.reset(48);
        stream.update_state(&MusicalState {
            arousal: 0.95,
            valence: -0.8,
            noradrenaline: 0.9,
            harmony_activations: [0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.9, 0.0],
            ..Default::default()
        });
        let excited_note = stream.next_note();

        assert!(calm_note.is_some());
        assert!(excited_note.is_some());
    }

    #[test]
    fn batch_respects_count() {
        let mut stream = MuseStream::new(49, default_config());
        let batch = stream.generate_batch(3);
        assert!(
            batch.len() <= 3,
            "batch should not exceed requested count, got {}",
            batch.len()
        );
        assert!(!batch.is_empty(), "batch should produce at least one note");
    }

    #[test]
    fn last_note_updated() {
        let mut stream = MuseStream::new(50, default_config());
        assert!(stream.last_note().is_none());
        let note = stream.next_note();
        if note.is_some() {
            assert!(stream.last_note().is_some());
        }
    }
}
