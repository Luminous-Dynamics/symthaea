// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MIDI export: write compositions as Standard MIDI Files.
//!
//! Separates composition intelligence from synthesis quality.
//! Load the exported MIDI into any DAW (Ardour, Reaper, LMMS) with
//! professional virtual instruments for production-quality sound.
//!
//! Exports multi-track Type 1 MIDI with:
//! - Track 0: Tempo/time signature map
//! - Track 1: Lead melody (channel 0)
//! - Track 2: Bass (channel 1)
//! - Track 3: Harmony/chords (channel 2)
//! - Track 4: Drums (channel 9, GM standard)

use crate::Note;
use std::io::Write;
use std::path::Path;

const TICKS_PER_BEAT: u16 = 480;

/// A MIDI event in our internal representation.
#[derive(Debug, Clone)]
struct MidiEvent {
    tick: u64,
    channel: u8,
    event_type: MidiEventType,
}

#[derive(Debug, Clone)]
enum MidiEventType {
    NoteOn { note: u8, velocity: u8 },
    NoteOff { note: u8 },
    Tempo(u32),            // microseconds per beat
    TimeSignature(u8, u8), // numerator, denominator power of 2
    ProgramChange(u8),     // GM instrument number
}

/// MIDI track with events.
struct Track {
    name: String,
    channel: u8,
    events: Vec<MidiEvent>,
}

/// Export a composition as a Standard MIDI File (Type 1, multi-track).
pub fn export_midi(
    notes: &[Note],
    tempo_bpm: f32,
    time_sig_num: u8,
    time_sig_den: u8,
    output_path: &Path,
) -> Result<(), String> {
    // Separate notes into voices by frequency range
    let mut lead_notes = Vec::new();
    let mut bass_notes = Vec::new();
    let mut harmony_notes = Vec::new();

    for note in notes {
        let midi_note = freq_to_midi(note.frequency);
        if midi_note < 48 {
            bass_notes.push(note);
        } else if note.velocity < 0.3 {
            harmony_notes.push(note); // quiet notes = harmony
        } else {
            lead_notes.push(note);
        }
    }

    let usec_per_beat = (60_000_000.0 / tempo_bpm) as u32;

    // Track 0: tempo/time signature
    let tempo_track = Track {
        name: "Tempo".into(),
        channel: 0,
        events: vec![
            MidiEvent {
                tick: 0,
                channel: 0,
                event_type: MidiEventType::Tempo(usec_per_beat),
            },
            MidiEvent {
                tick: 0,
                channel: 0,
                event_type: MidiEventType::TimeSignature(time_sig_num, time_sig_den),
            },
        ],
    };

    // Track 1: Lead (GM 0 = Acoustic Grand Piano)
    let lead_track = notes_to_track("Lead", 0, 0, &lead_notes, tempo_bpm);

    // Track 2: Bass (GM 32 = Acoustic Bass)
    let bass_track = notes_to_track("Bass", 1, 32, &bass_notes, tempo_bpm);

    // Track 3: Harmony (GM 48 = String Ensemble)
    let harmony_track = notes_to_track("Harmony", 2, 48, &harmony_notes, tempo_bpm);

    let tracks = vec![tempo_track, lead_track, bass_track, harmony_track];

    // Write MIDI file
    write_midi_file(output_path, &tracks)
}

/// Export with voice separation already done (from StreamingSynth).
pub fn export_midi_voices(
    lead: &[Note],
    bass: &[Note],
    harmony: &[Note],
    drum_hits: &[(f32, u8, u8)], // (time_secs, GM_note, velocity)
    tempo_bpm: f32,
    time_sig_num: u8,
    time_sig_den: u8,
    output_path: &Path,
) -> Result<(), String> {
    let usec_per_beat = (60_000_000.0 / tempo_bpm) as u32;

    let tempo_track = Track {
        name: "Tempo".into(),
        channel: 0,
        events: vec![
            MidiEvent {
                tick: 0,
                channel: 0,
                event_type: MidiEventType::Tempo(usec_per_beat),
            },
            MidiEvent {
                tick: 0,
                channel: 0,
                event_type: MidiEventType::TimeSignature(time_sig_num, time_sig_den),
            },
        ],
    };

    let lead_refs: Vec<&Note> = lead.iter().collect();
    let bass_refs: Vec<&Note> = bass.iter().collect();
    let harmony_refs: Vec<&Note> = harmony.iter().collect();
    let lead_track = notes_to_track("Lead", 0, 0, &lead_refs, tempo_bpm);
    let bass_track = notes_to_track("Bass", 1, 32, &bass_refs, tempo_bpm);
    let harmony_track = notes_to_track("Harmony", 2, 48, &harmony_refs, tempo_bpm);

    // Drums on channel 9
    let mut drum_track = Track {
        name: "Drums".into(),
        channel: 9,
        events: Vec::new(),
    };
    drum_track.events.push(MidiEvent {
        tick: 0,
        channel: 9,
        event_type: MidiEventType::ProgramChange(0), // GM drums
    });
    for &(time, note, vel) in drum_hits {
        let tick = secs_to_ticks(time, tempo_bpm);
        drum_track.events.push(MidiEvent {
            tick,
            channel: 9,
            event_type: MidiEventType::NoteOn {
                note,
                velocity: vel,
            },
        });
        drum_track.events.push(MidiEvent {
            tick: tick + 120,
            channel: 9, // short duration
            event_type: MidiEventType::NoteOff { note },
        });
    }

    let tracks = vec![
        tempo_track,
        lead_track,
        bass_track,
        harmony_track,
        drum_track,
    ];
    write_midi_file(output_path, &tracks)
}

fn notes_to_track(name: &str, channel: u8, program: u8, notes: &[&Note], tempo_bpm: f32) -> Track {
    let mut events = Vec::new();

    // Program change
    events.push(MidiEvent {
        tick: 0,
        channel,
        event_type: MidiEventType::ProgramChange(program),
    });

    for note in notes {
        let midi_note = freq_to_midi(note.frequency).clamp(0, 127);
        let velocity = (note.velocity * 127.0).clamp(1.0, 127.0) as u8;
        let start_tick = secs_to_ticks(note.start_time, tempo_bpm);
        let duration_ticks = secs_to_ticks(note.duration, tempo_bpm).max(1);

        events.push(MidiEvent {
            tick: start_tick,
            channel,
            event_type: MidiEventType::NoteOn {
                note: midi_note,
                velocity,
            },
        });
        events.push(MidiEvent {
            tick: start_tick + duration_ticks,
            channel,
            event_type: MidiEventType::NoteOff { note: midi_note },
        });
    }

    // Sort by tick
    events.sort_by_key(|e| e.tick);

    Track {
        name: name.into(),
        channel,
        events,
    }
}

fn write_midi_file(path: &Path, tracks: &[Track]) -> Result<(), String> {
    let mut file = std::fs::File::create(path).map_err(|e| format!("create MIDI file: {e}"))?;

    // Header: MThd
    file.write_all(b"MThd").map_err(|e| e.to_string())?;
    file.write_all(&6u32.to_be_bytes())
        .map_err(|e| e.to_string())?; // chunk length
    file.write_all(&1u16.to_be_bytes())
        .map_err(|e| e.to_string())?; // format 1
    file.write_all(&(tracks.len() as u16).to_be_bytes())
        .map_err(|e| e.to_string())?;
    file.write_all(&TICKS_PER_BEAT.to_be_bytes())
        .map_err(|e| e.to_string())?;

    // Tracks
    for track in tracks {
        let track_data = encode_track(track);
        file.write_all(b"MTrk").map_err(|e| e.to_string())?;
        file.write_all(&(track_data.len() as u32).to_be_bytes())
            .map_err(|e| e.to_string())?;
        file.write_all(&track_data).map_err(|e| e.to_string())?;
    }

    Ok(())
}

fn encode_track(track: &Track) -> Vec<u8> {
    let mut data = Vec::new();
    let mut last_tick = 0u64;

    // Track name meta event
    let name_bytes = track.name.as_bytes();
    write_vlq(&mut data, 0); // delta = 0
    data.push(0xFF);
    data.push(0x03); // track name
    write_vlq(&mut data, name_bytes.len() as u32);
    data.extend_from_slice(name_bytes);

    for event in &track.events {
        let delta = event.tick.saturating_sub(last_tick);
        write_vlq(&mut data, delta as u32);
        last_tick = event.tick;

        match &event.event_type {
            MidiEventType::NoteOn { note, velocity } => {
                data.push(0x90 | (event.channel & 0x0F));
                data.push(*note & 0x7F);
                data.push(*velocity & 0x7F);
            }
            MidiEventType::NoteOff { note } => {
                data.push(0x80 | (event.channel & 0x0F));
                data.push(*note & 0x7F);
                data.push(0); // velocity 0
            }
            MidiEventType::Tempo(usec) => {
                data.push(0xFF);
                data.push(0x51);
                data.push(0x03);
                data.push((usec >> 16) as u8);
                data.push((usec >> 8) as u8);
                data.push(*usec as u8);
            }
            MidiEventType::TimeSignature(num, den) => {
                data.push(0xFF);
                data.push(0x58);
                data.push(0x04);
                data.push(*num);
                // Denominator as power of 2
                let den_pow = match den {
                    2 => 1,
                    4 => 2,
                    8 => 3,
                    16 => 4,
                    _ => 2,
                };
                data.push(den_pow);
                data.push(24); // MIDI clocks per metronome click
                data.push(8); // 32nd notes per quarter
            }
            MidiEventType::ProgramChange(program) => {
                data.push(0xC0 | (event.channel & 0x0F));
                data.push(*program & 0x7F);
            }
        }
    }

    // End of track
    write_vlq(&mut data, 0);
    data.push(0xFF);
    data.push(0x2F);
    data.push(0x00);

    data
}

fn write_vlq(data: &mut Vec<u8>, mut value: u32) {
    if value == 0 {
        data.push(0);
        return;
    }
    let mut bytes = Vec::new();
    while value > 0 {
        bytes.push((value & 0x7F) as u8);
        value >>= 7;
    }
    bytes.reverse();
    for (i, b) in bytes.iter().enumerate() {
        if i < bytes.len() - 1 {
            data.push(b | 0x80); // continuation bit
        } else {
            data.push(*b);
        }
    }
}

fn freq_to_midi(freq: f32) -> u8 {
    ((12.0 * (freq / 440.0).log2() + 69.0).round() as i32).clamp(0, 127) as u8
}

fn secs_to_ticks(secs: f32, tempo_bpm: f32) -> u64 {
    (secs * tempo_bpm / 60.0 * TICKS_PER_BEAT as f32) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freq_to_midi_a4() {
        assert_eq!(freq_to_midi(440.0), 69);
    }

    #[test]
    fn freq_to_midi_c4() {
        assert_eq!(freq_to_midi(261.63), 60);
    }

    #[test]
    fn vlq_zero() {
        let mut data = Vec::new();
        write_vlq(&mut data, 0);
        assert_eq!(data, vec![0]);
    }

    #[test]
    fn vlq_small() {
        let mut data = Vec::new();
        write_vlq(&mut data, 127);
        assert_eq!(data, vec![127]);
    }

    #[test]
    fn vlq_large() {
        let mut data = Vec::new();
        write_vlq(&mut data, 128);
        assert_eq!(data, vec![0x81, 0x00]);
    }

    #[test]
    fn export_creates_file() {
        let notes = vec![
            Note {
                frequency: 261.63,
                start_time: 0.0,
                duration: 0.5,
                velocity: 0.7,
            },
            Note {
                frequency: 329.63,
                start_time: 0.5,
                duration: 0.5,
                velocity: 0.6,
            },
            Note {
                frequency: 392.00,
                start_time: 1.0,
                duration: 0.5,
                velocity: 0.8,
            },
        ];
        let path = std::path::PathBuf::from("/tmp/test_export.mid");
        let result = export_midi(&notes, 120.0, 4, 4, &path);
        assert!(result.is_ok(), "export should succeed: {:?}", result);
        assert!(path.exists(), "file should exist");
        let size = std::fs::metadata(&path).unwrap().len();
        assert!(size > 50, "file should have content: {size} bytes");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn secs_to_ticks_correct() {
        // At 120 BPM, 1 second = 2 beats = 960 ticks
        assert_eq!(secs_to_ticks(1.0, 120.0), 960);
    }
}
