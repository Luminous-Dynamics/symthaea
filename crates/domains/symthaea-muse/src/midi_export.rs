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

/// Export a symbolic theory [`Score`](symthaea_music_theory::Score) as a
/// Standard MIDI File — the composition-intelligence escape hatch: load the
/// result into any DAW with professional virtual instruments and the
/// composed structure (voice leading, form, secondary dominants, tension
/// arc) renders at production quality, completely decoupled from muse's own
/// synthesis.
///
/// Unlike [`export_midi`], which reverse-guesses voices from frequency and
/// loudness, this export is EXACT: voice identity comes from the score's own
/// `VoiceRole`s, timing is beat-arithmetic (`beats × 480 ticks`, no
/// float-seconds round trip), the time signature is the score's real meter,
/// and each track's GM program comes from the same per-style ensemble the
/// audio renderer would use ([`crate::theory_realize::instruments_for`]).
/// Velocities are the score's written (structural) dynamics — expressive
/// jitter is deliberately NOT baked in, that's the DAW instrument's job.
/// (This doc describes [`export_score_midi`] below; the performed
/// counterpart follows.)
///
/// Export the PERFORMED piece as MIDI — the same swing∘rubato timeline,
/// learned expression, humanize jitter, and equal-loudness taper the audio
/// render bakes into its voices, written into the event ticks (fixed tempo
/// track, warped timing). Use this when the `.mid` should carry what you
/// HEAR: a listening review of a swung piece found the symbolic export
/// landing straight-grid while the WAV shuffled. [`export_score_midi`]
/// remains the clean-grid symbolic counterpart for DAW re-instrumentation.
/// Voices and instruments resolve exactly as the audio path resolves them
/// ([`crate::theory_realize::resolve_spec_ensemble`]), including the
/// climax doubling voice and the counter-melody's contrast timbre.
#[cfg(feature = "theory")]
pub fn export_performance_midi(
    score: &symthaea_music_theory::Score,
    spec: &symthaea_music_theory::CompositionSpec,
    seed: u64,
    state: &crate::MusicalState,
    output_path: &Path,
) -> Result<(), String> {
    let ensemble = crate::theory_realize::resolve_spec_ensemble(spec, seed);
    let voices = crate::theory_realize::performance_voices(
        score,
        ensemble,
        spec.texture.swing as f64,
        state,
        spec.texture.return_color,
    );
    let usec_per_beat = (60_000_000.0 / score.tempo_bpm) as u32;
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
                event_type: MidiEventType::TimeSignature(score.meter, 4),
            },
        ],
    };
    let mut tracks = vec![tempo_track];
    for (idx, voice) in voices.iter().enumerate() {
        // At most 5 performed voices — channels stay safely below the GM
        // drum channel (9).
        let channel = idx as u8;
        let mut events = vec![MidiEvent {
            tick: 0,
            channel,
            event_type: MidiEventType::ProgramChange(voice.instrument.gm_program()),
        }];
        for n in &voice.notes {
            let on = secs_to_ticks(n.start_time, score.tempo_bpm);
            let off = secs_to_ticks(n.start_time + n.duration, score.tempo_bpm).max(on + 1);
            let note = freq_to_midi(n.frequency);
            events.push(MidiEvent {
                tick: on,
                channel,
                event_type: MidiEventType::NoteOn {
                    note,
                    velocity: (n.velocity * 127.0).clamp(1.0, 127.0) as u8,
                },
            });
            events.push(MidiEvent {
                tick: off,
                channel,
                event_type: MidiEventType::NoteOff { note },
            });
        }
        events.sort_by_key(|e| e.tick);
        tracks.push(Track {
            name: voice.name.into(),
            channel,
            events,
        });
    }
    write_midi_file(output_path, &tracks)
}

/// The clean-grid SYMBOLIC export — see the doc block above
/// [`export_performance_midi`] for the full contrast between the two.
#[cfg(feature = "theory")]
pub fn export_score_midi(
    score: &symthaea_music_theory::Score,
    style: symthaea_music_theory::Style,
    seed: u64,
    output_path: &Path,
) -> Result<(), String> {
    use symthaea_music_theory::score::VoiceRole as TheoryRole;

    let (melody_i, harmony_i, bass_i) = crate::theory_realize::instruments_for(style, seed);
    let counter_i = crate::theory_realize::contrast_counter(melody_i, bass_i);
    let usec_per_beat = (60_000_000.0 / score.tempo_bpm) as u32;

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
                event_type: MidiEventType::TimeSignature(score.meter, 4),
            },
        ],
    };

    let beats_to_ticks = |beats: f64| (beats * TICKS_PER_BEAT as f64).round() as u64;
    let mut tracks = vec![tempo_track];
    for (role, name, channel, instrument) in [
        (TheoryRole::Melody, "Melody", 0u8, melody_i),
        (TheoryRole::Harmony, "Harmony", 1, harmony_i),
        (TheoryRole::Bass, "Bass", 2, bass_i),
        // The counter-melody gets its contrast timbre (same rule as the
        // audio render — see `contrast_counter`), its own channel/track.
        (TheoryRole::CounterMelody, "Counter", 3, counter_i),
    ] {
        let mut events = vec![MidiEvent {
            tick: 0,
            channel,
            event_type: MidiEventType::ProgramChange(instrument.gm_program()),
        }];
        for n in score.voice(role) {
            let on = beats_to_ticks(n.onset.beats());
            let off = beats_to_ticks((n.onset + n.duration).beats()).max(on + 1);
            let note = n.pitch.midi().clamp(0, 127);
            events.push(MidiEvent {
                tick: on,
                channel,
                event_type: MidiEventType::NoteOn {
                    note,
                    velocity: (n.velocity * 127.0).clamp(1.0, 127.0) as u8,
                },
            });
            events.push(MidiEvent {
                tick: off,
                channel,
                event_type: MidiEventType::NoteOff { note },
            });
        }
        events.sort_by_key(|e| e.tick);
        tracks.push(Track {
            name: name.into(),
            channel,
            events,
        });
    }

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

    #[cfg(feature = "theory")]
    #[test]
    fn score_export_round_trips_through_a_midi_parser() {
        use symthaea_music_theory::{MusicalIntent, Style};
        let score = symthaea_music_theory::compose(&MusicalIntent::default());
        let path =
            std::env::temp_dir().join(format!("muse_score_export_test_{}.mid", std::process::id()));
        export_score_midi(&score, Style::Classical, 0, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let smf = midly::Smf::parse(&bytes).expect("exported file must be valid SMF");
        // Tempo track + melody + harmony + bass + counter-melody.
        assert_eq!(smf.tracks.len(), 5);
        // Every score note becomes exactly one NoteOn (and its NoteOff).
        let count = |kind: fn(&midly::MidiMessage) -> bool| -> usize {
            smf.tracks
                .iter()
                .flatten()
                .filter(|e| {
                    matches!(&e.kind,
                        midly::TrackEventKind::Midi { message, .. } if kind(message))
                })
                .count()
        };
        let ons =
            count(|m| matches!(m, midly::MidiMessage::NoteOn { vel, .. } if vel.as_int() > 0));
        let offs = count(|m| {
            matches!(m, midly::MidiMessage::NoteOff { .. })
                || matches!(m, midly::MidiMessage::NoteOn { vel, .. } if vel.as_int() == 0)
        });
        assert_eq!(ons, score.notes.len());
        assert_eq!(offs, score.notes.len());
        std::fs::remove_file(&path).ok();
    }

    #[cfg(feature = "theory")]
    #[test]
    fn performance_export_bakes_swing_into_the_ticks() {
        use symthaea_music_theory::{MusicalIntent, Style};
        let intent = MusicalIntent::default();
        let straight = Style::Classical.spec();
        let mut swung = straight.clone();
        swung.texture.swing = 0.68;
        let score = symthaea_music_theory::compose_with_spec(&intent, &straight);
        let state = crate::MusicalState::default();
        let mut onsets = Vec::new();
        for (tag, spec) in [("straight", &straight), ("swung", &swung)] {
            let path = std::env::temp_dir()
                .join(format!("muse_perf_export_{tag}_{}.mid", std::process::id()));
            export_performance_midi(&score, spec, 0, &state, &path).unwrap();
            let bytes = std::fs::read(&path).unwrap();
            let smf = midly::Smf::parse(&bytes).expect("valid SMF");
            // Absolute NoteOn ticks of the melody track (named tracks are in
            // performance-voice order; melody is present in every form).
            let mut track_onsets = Vec::new();
            for track in &smf.tracks {
                let mut abs = 0u32;
                let mut name = String::new();
                let mut ons = Vec::new();
                for e in track {
                    abs += e.delta.as_int();
                    match &e.kind {
                        midly::TrackEventKind::Meta(midly::MetaMessage::TrackName(n)) => {
                            name = String::from_utf8_lossy(n).into_owned();
                        }
                        midly::TrackEventKind::Midi {
                            message: midly::MidiMessage::NoteOn { vel, .. },
                            ..
                        } if vel.as_int() > 0 => ons.push(abs),
                        _ => {}
                    }
                }
                if name == "Melody" {
                    track_onsets = ons;
                }
            }
            assert!(!track_onsets.is_empty(), "{tag}: melody track must exist");
            onsets.push(track_onsets);
        }
        assert_eq!(onsets[0].len(), onsets[1].len());
        // Swing is a within-beat warp: SOME onsets must land later in the
        // swung export (off-beats pushed toward 0.68), and none earlier by
        // more than humanize jitter (same seeds both runs → identical
        // jitter, so the difference is pure swing).
        let later = onsets[0]
            .iter()
            .zip(&onsets[1])
            .filter(|(s, w)| w > s)
            .count();
        assert!(
            later > 0,
            "a swung export must move off-beat onsets later than straight"
        );
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
