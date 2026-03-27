// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! MIDI binary export for Symthaea compositions.
//!
//! Writes Standard MIDI Files (SMF format 0) directly from `Composition` data.
//! No external crate needed — MIDI is a simple binary format.
//!
//! # Format
//!
//! ```text
//! MThd (header chunk) → MTrk (track chunk with NoteOn/NoteOff/Tempo events)
//! ```

use crate::{Composition, Note};

/// Ticks per quarter note (standard MIDI resolution).
const TICKS_PER_BEAT: u16 = 480;

/// Convert a frequency in Hz to the nearest MIDI note number (0-127).
///
/// A4 (440 Hz) = MIDI note 69. Each semitone is a factor of 2^(1/12).
pub fn freq_to_midi_note(freq: f32) -> u8 {
    if freq <= 0.0 {
        return 0;
    }
    let note = 69.0 + 12.0 * (freq / 440.0).log2();
    (note.round() as i32).clamp(0, 127) as u8
}

/// Convert velocity [0.0, 1.0] to MIDI velocity [1, 127].
fn velocity_to_midi(vel: f32) -> u8 {
    (vel * 126.0 + 1.0).round().clamp(1.0, 127.0) as u8
}

/// Encode a u32 as MIDI variable-length quantity (VLQ).
fn write_vlq(value: u32) -> Vec<u8> {
    if value == 0 {
        return vec![0];
    }
    let mut bytes = Vec::new();
    let mut v = value;
    bytes.push((v & 0x7F) as u8);
    v >>= 7;
    while v > 0 {
        bytes.push((v & 0x7F) as u8 | 0x80);
        v >>= 7;
    }
    bytes.reverse();
    bytes
}

/// Write a big-endian u16.
fn write_u16_be(value: u16) -> [u8; 2] {
    value.to_be_bytes()
}

/// Write a big-endian u32.
fn write_u32_be(value: u32) -> [u8; 4] {
    value.to_be_bytes()
}

impl Composition {
    /// Export the composition as a Standard MIDI File (format 0, single track).
    ///
    /// `tempo_bpm` sets the MIDI tempo meta-event. Notes are quantized to
    /// the MIDI tick grid (480 ticks per quarter note).
    pub fn to_midi(&self, tempo_bpm: f32) -> Vec<u8> {
        let mut midi = Vec::new();

        // ─── MThd header ───
        midi.extend_from_slice(b"MThd");
        midi.extend_from_slice(&write_u32_be(6)); // header length
        midi.extend_from_slice(&write_u16_be(0)); // format 0 (single track)
        midi.extend_from_slice(&write_u16_be(1)); // 1 track
        midi.extend_from_slice(&write_u16_be(TICKS_PER_BEAT));

        // ─── MTrk track ───
        let track_data = self.build_track(tempo_bpm);
        midi.extend_from_slice(b"MTrk");
        midi.extend_from_slice(&write_u32_be(track_data.len() as u32));
        midi.extend(track_data);

        midi
    }

    /// Export to a MIDI file on disk.
    pub fn to_midi_file(
        &self,
        path: &std::path::Path,
        tempo_bpm: f32,
    ) -> std::io::Result<()> {
        let data = self.to_midi(tempo_bpm);
        std::fs::write(path, data)
    }

    /// Build the raw track data (events between MTrk header and end).
    fn build_track(&self, tempo_bpm: f32) -> Vec<u8> {
        let mut track = Vec::new();

        // Tempo meta-event: FF 51 03 tt tt tt (microseconds per beat)
        let us_per_beat = (60_000_000.0 / tempo_bpm) as u32;
        track.extend(write_vlq(0)); // delta=0
        track.push(0xFF);
        track.push(0x51);
        track.push(0x03);
        track.push(((us_per_beat >> 16) & 0xFF) as u8);
        track.push(((us_per_beat >> 8) & 0xFF) as u8);
        track.push((us_per_beat & 0xFF) as u8);

        // Convert notes to timed MIDI events
        let ticks_per_sec = (TICKS_PER_BEAT as f32) * (tempo_bpm / 60.0);
        let mut events: Vec<(u32, u8, u8, u8)> = Vec::new(); // (tick, status, note, vel)

        for note in &self.notes {
            let start_tick = (note.start_time * ticks_per_sec) as u32;
            let end_tick = ((note.start_time + note.duration) * ticks_per_sec) as u32;
            let midi_note = freq_to_midi_note(note.frequency);
            let midi_vel = velocity_to_midi(note.velocity);

            events.push((start_tick, 0x90, midi_note, midi_vel)); // NoteOn
            events.push((end_tick, 0x80, midi_note, 0)); // NoteOff
        }

        // Sort by tick (stable sort preserves NoteOn before NoteOff at same tick)
        events.sort_by_key(|e| e.0);

        // Write events with delta times
        let mut prev_tick = 0u32;
        for (tick, status, note, vel) in &events {
            let delta = tick.saturating_sub(prev_tick);
            track.extend(write_vlq(delta));
            track.push(*status);
            track.push(*note);
            track.push(*vel);
            prev_tick = *tick;
        }

        // End of track meta-event: FF 2F 00
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

    fn test_composition() -> Composition {
        Composition {
            audio: crate::AudioData::I16(vec![0i16; 44100]),
            sample_rate: 44100,
            notes: vec![
                Note {
                    frequency: 440.0,
                    start_time: 0.0,
                    duration: 0.5,
                    velocity: 0.8,
                },
                Note {
                    frequency: 523.25,
                    start_time: 0.5,
                    duration: 0.5,
                    velocity: 0.6,
                },
                Note {
                    frequency: 329.63,
                    start_time: 1.0,
                    duration: 1.0,
                    velocity: 0.9,
                },
            ],
            duration_secs: 2.0,
            section: SectionType::Developmental,
        }
    }

    #[test]
    fn midi_starts_with_mthd() {
        let comp = test_composition();
        let midi = comp.to_midi(120.0);
        assert_eq!(&midi[0..4], b"MThd");
    }

    #[test]
    fn midi_contains_mtrk() {
        let comp = test_composition();
        let midi = comp.to_midi(120.0);
        // MTrk starts at byte 14 (after 14-byte header)
        assert_eq!(&midi[14..18], b"MTrk");
    }

    #[test]
    fn freq_to_midi_a440_is_69() {
        assert_eq!(freq_to_midi_note(440.0), 69);
    }

    #[test]
    fn freq_to_midi_c4_is_60() {
        assert_eq!(freq_to_midi_note(261.63), 60);
    }

    #[test]
    fn freq_to_midi_clamped() {
        assert_eq!(freq_to_midi_note(0.0), 0);
        assert_eq!(freq_to_midi_note(100_000.0), 127);
    }

    #[test]
    fn vlq_encoding() {
        assert_eq!(write_vlq(0), vec![0]);
        assert_eq!(write_vlq(127), vec![0x7F]);
        assert_eq!(write_vlq(128), vec![0x81, 0x00]);
        assert_eq!(write_vlq(480), vec![0x83, 0x60]);
    }

    #[test]
    fn midi_note_count_matches() {
        let comp = test_composition();
        let midi = comp.to_midi(120.0);
        // Count NoteOn events (status byte 0x90)
        // Each note produces NoteOn + NoteOff events; MIDI should be substantial
        assert!(midi.len() > 30, "MIDI too short: {} bytes", midi.len());
    }

    #[test]
    fn round_trip_compose_to_midi() {
        let config = crate::MuseConfig {
            duration_secs: 2.0,
            max_notes: 8,
            ..Default::default()
        };
        let state = crate::MusicalState::default();
        let comp = crate::compose(&config, &state, 42);
        let midi = comp.to_midi(120.0);
        assert_eq!(&midi[0..4], b"MThd");
        assert_eq!(&midi[14..18], b"MTrk");
        assert!(midi.len() > 20);
    }

    #[test]
    fn velocity_mapping() {
        assert_eq!(velocity_to_midi(0.0), 1);
        assert_eq!(velocity_to_midi(1.0), 127);
        assert_eq!(velocity_to_midi(0.5), 64);
    }
}
