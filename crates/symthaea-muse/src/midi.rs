// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Standard MIDI File (format 0) export for compositions.
//!
//! Converts `Composition` note sequences into binary SMF data suitable
//! for playback in any DAW or MIDI player.

use std::io;
use std::path::Path;

use crate::Composition;

/// Ticks per quarter note (MIDI resolution).
const TICKS_PER_BEAT: u16 = 480;

/// Convert a frequency in Hz to a MIDI note number.
///
/// A440 = MIDI 69 (A4). Returns nearest note, clamped to [0, 127].
pub fn freq_to_midi_note(freq: f32) -> u8 {
    if freq <= 0.0 {
        return 0;
    }
    let midi = 69.0 + 12.0 * (freq / 440.0).log2();
    midi.round().clamp(0.0, 127.0) as u8
}

/// Encode an integer as a MIDI variable-length quantity.
pub fn write_vlq(value: u32) -> Vec<u8> {
    if value == 0 {
        return vec![0];
    }
    let mut bytes = Vec::new();
    let mut v = value;
    bytes.push((v & 0x7F) as u8);
    v >>= 7;
    while v > 0 {
        bytes.push(((v & 0x7F) | 0x80) as u8);
        v >>= 7;
    }
    bytes.reverse();
    bytes
}

impl Composition {
    /// Export the composition as a Standard MIDI File (format 0, single track).
    pub fn to_midi(&self, tempo_bpm: f32) -> Vec<u8> {
        let mut data = Vec::new();

        // ── MThd header ──────────────────────────────────────────────
        data.extend_from_slice(b"MThd");
        data.extend_from_slice(&6u32.to_be_bytes()); // header length
        data.extend_from_slice(&0u16.to_be_bytes()); // format 0
        data.extend_from_slice(&1u16.to_be_bytes()); // 1 track
        data.extend_from_slice(&TICKS_PER_BEAT.to_be_bytes());

        // ── MTrk track ───────────────────────────────────────────────
        let track_data = self.build_track_data(tempo_bpm);
        data.extend_from_slice(b"MTrk");
        data.extend_from_slice(&(track_data.len() as u32).to_be_bytes());
        data.extend(track_data);

        data
    }

    /// Write the composition as a `.mid` file.
    pub fn to_midi_file(&self, path: &Path, tempo_bpm: f32) -> io::Result<()> {
        let data = self.to_midi(tempo_bpm);
        std::fs::write(path, data)
    }

    /// Build raw track data (events + end-of-track).
    fn build_track_data(&self, tempo_bpm: f32) -> Vec<u8> {
        let mut track = Vec::new();

        // Tempo meta-event: FF 51 03 <microseconds per beat>
        let uspb = (60_000_000.0 / tempo_bpm) as u32;
        track.extend(write_vlq(0)); // delta = 0
        track.push(0xFF);
        track.push(0x51);
        track.push(0x03);
        track.push(((uspb >> 16) & 0xFF) as u8);
        track.push(((uspb >> 8) & 0xFF) as u8);
        track.push((uspb & 0xFF) as u8);

        // Build NoteOn/NoteOff events sorted by absolute tick
        let ticks_per_sec = (TICKS_PER_BEAT as f32) * (tempo_bpm / 60.0);
        let mut events: Vec<(u32, u8, u8, u8)> = Vec::new(); // (tick, status, note, vel)

        for note in &self.notes {
            let on_tick = (note.start_time * ticks_per_sec) as u32;
            let off_tick = ((note.start_time + note.duration) * ticks_per_sec) as u32;
            let midi_note = freq_to_midi_note(note.frequency);
            let vel = (note.velocity * 127.0).clamp(1.0, 127.0) as u8;

            events.push((on_tick, 0x90, midi_note, vel)); // NoteOn
            events.push((off_tick, 0x80, midi_note, 0)); // NoteOff
        }

        // Sort by tick (NoteOff before NoteOn at same tick for cleanliness)
        events.sort_by(|a, b| {
            a.0.cmp(&b.0).then(a.1.cmp(&b.1)) // 0x80 < 0x90
        });

        let mut last_tick = 0u32;
        for (tick, status, note, vel) in &events {
            let delta = tick.saturating_sub(last_tick);
            track.extend(write_vlq(delta));
            track.push(*status);
            track.push(*note);
            track.push(*vel);
            last_tick = *tick;
        }

        // End of track
        track.extend(write_vlq(0));
        track.push(0xFF);
        track.push(0x2F);
        track.push(0x00);

        track
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::SectionType;
    use crate::{AudioData, Note};

    fn test_comp() -> Composition {
        Composition {
            audio: AudioData::F32(vec![0.0; 100]),
            sample_rate: 44100,
            notes: vec![
                Note {
                    frequency: 261.63,
                    start_time: 0.0,
                    duration: 0.5,
                    velocity: 0.8,
                },
                Note {
                    frequency: 440.0,
                    start_time: 0.5,
                    duration: 0.5,
                    velocity: 0.7,
                },
                Note {
                    frequency: 329.63,
                    start_time: 1.0,
                    duration: 0.25,
                    velocity: 0.6,
                },
            ],
            duration_secs: 2.0,
            section: SectionType::Developmental,
        }
    }

    #[test]
    fn mthd_magic() {
        let midi = test_comp().to_midi(120.0);
        assert_eq!(&midi[0..4], b"MThd");
    }

    #[test]
    fn mtrk_magic() {
        let midi = test_comp().to_midi(120.0);
        assert_eq!(&midi[14..18], b"MTrk");
    }

    #[test]
    fn freq_to_midi_a440() {
        assert_eq!(freq_to_midi_note(440.0), 69);
    }

    #[test]
    fn freq_to_midi_c4() {
        assert_eq!(freq_to_midi_note(261.63), 60);
    }

    #[test]
    fn vlq_single_byte() {
        assert_eq!(write_vlq(0), vec![0]);
        assert_eq!(write_vlq(127), vec![127]);
    }

    #[test]
    fn vlq_multi_byte() {
        assert_eq!(write_vlq(128), vec![0x81, 0x00]);
        assert_eq!(write_vlq(0x3FFF), vec![0xFF, 0x7F]);
    }

    #[test]
    fn velocity_mapping() {
        let comp = test_comp();
        let midi = comp.to_midi(120.0);
        // File should contain NoteOn events with non-zero velocity
        let has_note_on = midi.windows(2).any(|w| w[0] == 0x90);
        assert!(has_note_on, "MIDI should contain NoteOn events");
    }

    #[test]
    fn round_trip_compose_to_midi() {
        let config = crate::MuseConfig {
            duration_secs: 1.0,
            max_notes: 4,
            ..Default::default()
        };
        let state = crate::MusicalState::default();
        let comp = crate::compose(&config, &state, 42);
        let midi = comp.to_midi(120.0);
        assert!(midi.len() > 22, "MIDI should have header + track data");
        assert_eq!(&midi[0..4], b"MThd");
    }
}
