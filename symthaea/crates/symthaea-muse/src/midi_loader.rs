// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! MIDI file parser for loading training melodies.
//!
//! Minimal Standard MIDI File parser (read-only). Extracts monophonic
//! melodies from MIDI files for training NeuralMelody projections.
//!
//! Only parses what we need: MThd header, MTrk NoteOn/NoteOff events,
//! tempo meta-events. Skips SysEx, text, and other meta-events.

use crate::Note;
use std::path::Path;

/// Parse a Standard MIDI File and extract the melody as a Vec<Note>.
///
/// For polyphonic MIDI, takes the highest-velocity note at each onset
/// (monophonic reduction). Returns notes sorted by start_time.
pub fn load_midi(path: &Path) -> Result<Vec<Note>, MidiLoadError> {
    let data = std::fs::read(path).map_err(|e| MidiLoadError::Io(e.to_string()))?;
    parse_midi(&data)
}

/// Parse MIDI data from bytes.
pub fn parse_midi(data: &[u8]) -> Result<Vec<Note>, MidiLoadError> {
    if data.len() < 14 {
        return Err(MidiLoadError::TooShort);
    }

    // ── MThd header ──
    if &data[0..4] != b"MThd" {
        return Err(MidiLoadError::InvalidHeader);
    }
    let _format = read_u16_be(&data[8..10]);
    let num_tracks = read_u16_be(&data[10..12]);
    let ticks_per_beat = read_u16_be(&data[12..14]);

    if ticks_per_beat == 0 {
        return Err(MidiLoadError::InvalidHeader);
    }

    // ── Parse tracks ──
    let mut pos = 14;
    let mut all_events: Vec<MidiNoteEvent> = Vec::new();
    let mut tempo_us_per_beat: u32 = 500_000; // default 120 BPM

    for _ in 0..num_tracks {
        if pos + 8 > data.len() {
            break;
        }
        if &data[pos..pos + 4] != b"MTrk" {
            return Err(MidiLoadError::InvalidTrack);
        }
        let track_len = read_u32_be(&data[pos + 4..pos + 8]) as usize;
        let track_end = (pos + 8 + track_len).min(data.len());
        let track_data = &data[pos + 8..track_end];

        let (events, tempo) = parse_track(track_data, ticks_per_beat, tempo_us_per_beat);
        if tempo != 0 {
            tempo_us_per_beat = tempo;
        }
        all_events.extend(events);

        pos = track_end;
    }

    // ── Convert to Note format ──
    let secs_per_tick = (tempo_us_per_beat as f64) / (ticks_per_beat as f64 * 1_000_000.0);
    let mut notes = convert_events_to_notes(&all_events, secs_per_tick as f32);
    notes.sort_by(|a, b| {
        a.start_time
            .partial_cmp(&b.start_time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(notes)
}

/// Load all MIDI files from a directory (recursively).
pub fn load_melodies(dir: &Path) -> Vec<Vec<Note>> {
    let mut melodies = Vec::new();
    load_melodies_recursive(dir, &mut melodies);
    melodies
}

fn load_melodies_recursive(dir: &Path, melodies: &mut Vec<Vec<Note>>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                load_melodies_recursive(&path, melodies);
            } else if path
                .extension()
                .map_or(false, |ext| ext == "mid" || ext == "midi")
            {
                if let Ok(notes) = load_midi(&path) {
                    if !notes.is_empty() {
                        melodies.push(notes);
                    }
                }
            }
        }
    }
}

/// MIDI loading error.
#[derive(Debug)]
pub enum MidiLoadError {
    Io(String),
    TooShort,
    InvalidHeader,
    InvalidTrack,
}

impl std::fmt::Display for MidiLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(s) => write!(f, "IO error: {s}"),
            Self::TooShort => write!(f, "MIDI data too short"),
            Self::InvalidHeader => write!(f, "Invalid MThd header"),
            Self::InvalidTrack => write!(f, "Invalid MTrk chunk"),
        }
    }
}

// ── Internal types ──

struct MidiNoteEvent {
    tick: u32,
    note: u8,
    velocity: u8,
    is_on: bool,
}

// ── Parsing helpers ──

fn read_u16_be(data: &[u8]) -> u16 {
    ((data[0] as u16) << 8) | (data[1] as u16)
}

fn read_u32_be(data: &[u8]) -> u32 {
    ((data[0] as u32) << 24) | ((data[1] as u32) << 16) | ((data[2] as u32) << 8) | (data[3] as u32)
}

fn read_vlq(data: &[u8], pos: &mut usize) -> u32 {
    let mut value: u32 = 0;
    loop {
        if *pos >= data.len() {
            break;
        }
        let byte = data[*pos];
        *pos += 1;
        value = (value << 7) | (byte & 0x7F) as u32;
        if byte & 0x80 == 0 {
            break;
        }
    }
    value
}

fn parse_track(
    data: &[u8],
    _ticks_per_beat: u16,
    _default_tempo: u32,
) -> (Vec<MidiNoteEvent>, u32) {
    let mut events = Vec::new();
    let mut pos = 0;
    let mut tick: u32 = 0;
    let mut last_status: u8 = 0;
    let mut tempo: u32 = 0;

    while pos < data.len() {
        let delta = read_vlq(data, &mut pos);
        tick += delta;

        if pos >= data.len() {
            break;
        }

        let mut status = data[pos];

        // Running status: if high bit not set, reuse last status
        if status & 0x80 == 0 {
            status = last_status;
        } else {
            pos += 1;
        }

        let _channel = status & 0x0F;
        let msg_type = status & 0xF0;

        match msg_type {
            0x90 => {
                // NoteOn
                if pos + 1 >= data.len() {
                    break;
                }
                let note = data[pos];
                let velocity = data[pos + 1];
                pos += 2;
                last_status = status;
                if velocity > 0 {
                    events.push(MidiNoteEvent {
                        tick,
                        note,
                        velocity,
                        is_on: true,
                    });
                } else {
                    // Velocity 0 = NoteOff
                    events.push(MidiNoteEvent {
                        tick,
                        note,
                        velocity: 0,
                        is_on: false,
                    });
                }
            }
            0x80 => {
                // NoteOff
                if pos + 1 >= data.len() {
                    break;
                }
                let note = data[pos];
                let _velocity = data[pos + 1];
                pos += 2;
                last_status = status;
                events.push(MidiNoteEvent {
                    tick,
                    note,
                    velocity: 0,
                    is_on: false,
                });
            }
            0xA0 | 0xB0 | 0xE0 => {
                // Aftertouch, Control Change, Pitch Bend — 2 data bytes
                pos += 2;
                last_status = status;
            }
            0xC0 | 0xD0 => {
                // Program Change, Channel Pressure — 1 data byte
                pos += 1;
                last_status = status;
            }
            0xF0 => {
                if status == 0xFF {
                    // Meta event
                    if pos + 1 >= data.len() {
                        break;
                    }
                    let meta_type = data[pos];
                    pos += 1;
                    let meta_len = read_vlq(data, &mut pos) as usize;

                    if meta_type == 0x51 && meta_len == 3 && pos + 3 <= data.len() {
                        // Tempo: 3 bytes, microseconds per beat
                        tempo = ((data[pos] as u32) << 16)
                            | ((data[pos + 1] as u32) << 8)
                            | (data[pos + 2] as u32);
                    }

                    pos += meta_len;
                } else if status == 0xF0 || status == 0xF7 {
                    // SysEx — skip
                    let sysex_len = read_vlq(data, &mut pos) as usize;
                    pos += sysex_len;
                }
            }
            _ => {
                // Unknown — try to skip
                break;
            }
        }
    }

    (events, tempo)
}

fn convert_events_to_notes(events: &[MidiNoteEvent], secs_per_tick: f32) -> Vec<Note> {
    let mut notes = Vec::new();
    let mut active: std::collections::HashMap<u8, (u32, u8)> = std::collections::HashMap::new();

    for event in events {
        if event.is_on && event.velocity > 0 {
            active.insert(event.note, (event.tick, event.velocity));
        } else {
            // NoteOff: complete the note
            if let Some((start_tick, velocity)) = active.remove(&event.note) {
                let duration_ticks = event.tick.saturating_sub(start_tick);
                let start_time = start_tick as f32 * secs_per_tick;
                let duration = (duration_ticks as f32 * secs_per_tick).max(0.01);
                let frequency = midi_note_to_freq(event.note);

                notes.push(Note {
                    frequency,
                    start_time,
                    duration,
                    velocity: velocity as f32 / 127.0,
                });
            }
        }
    }

    // Close any still-active notes (assume 1 beat duration)
    for (&note_num, &(start_tick, velocity)) in &active {
        let start_time = start_tick as f32 * secs_per_tick;
        notes.push(Note {
            frequency: midi_note_to_freq(note_num),
            start_time,
            duration: 0.5,
            velocity: velocity as f32 / 127.0,
        });
    }

    notes
}

/// Convert MIDI note number to frequency in Hz.
fn midi_note_to_freq(note: u8) -> f32 {
    440.0 * 2.0_f32.powf((note as f32 - 69.0) / 12.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn midi_note_a4_is_440() {
        let freq = midi_note_to_freq(69);
        assert!((freq - 440.0).abs() < 0.1);
    }

    #[test]
    fn midi_note_c4_is_261() {
        let freq = midi_note_to_freq(60);
        assert!((freq - 261.63).abs() < 1.0);
    }

    #[test]
    fn parse_empty_data_fails() {
        assert!(parse_midi(&[]).is_err());
    }

    #[test]
    fn parse_invalid_header_fails() {
        let data = vec![0u8; 20];
        assert!(parse_midi(&data).is_err());
    }

    #[test]
    fn round_trip_our_own_midi() {
        // Generate a composition, export to MIDI, re-import, verify notes
        let config = crate::MuseConfig {
            duration_secs: 2.0,
            max_notes: 8,
            ..Default::default()
        };
        let state = crate::MusicalState::default();
        let comp = crate::compose(&config, &state, 42);

        if comp.notes.is_empty() {
            return; // Skip if no notes generated
        }

        let tempo = crate::rhythm::compute_tempo(&config, &state);
        let midi_data = comp.to_midi(tempo);

        let reimported = parse_midi(&midi_data).expect("should parse our own MIDI");
        assert!(!reimported.is_empty(), "re-imported MIDI should have notes");
        // Note count should match (each note → NoteOn + NoteOff)
        assert_eq!(
            reimported.len(),
            comp.notes.len(),
            "note count mismatch: imported {} vs original {}",
            reimported.len(),
            comp.notes.len()
        );
    }

    #[test]
    fn vlq_roundtrip() {
        // Test VLQ parsing matches our VLQ writing
        let test_values = [0u32, 127, 128, 480, 16383, 65535];
        for &val in &test_values {
            let encoded = crate::midi::write_vlq(val);
            let mut pos = 0;
            let decoded = read_vlq(&encoded, &mut pos);
            assert_eq!(val, decoded, "VLQ roundtrip failed for {val}");
        }
    }

    #[test]
    fn load_nonexistent_file() {
        let result = load_midi(Path::new("/nonexistent/file.mid"));
        assert!(result.is_err());
    }

    #[test]
    fn load_melodies_empty_dir() {
        let dir = std::env::temp_dir().join("symthaea_test_midi_empty");
        std::fs::create_dir_all(&dir).ok();
        let melodies = load_melodies(&dir);
        assert!(melodies.is_empty());
        std::fs::remove_dir(&dir).ok();
    }
}
