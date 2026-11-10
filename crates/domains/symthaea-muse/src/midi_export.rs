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
    ControlChange { controller: u8, value: u8 },
}

/// One discrete note scheduled on a MIDI channel, prior to sanitization.
#[derive(Debug, Clone, Copy)]
struct ScheduledNote {
    on: u64,
    off: u64,
    pitch: u8,
    velocity: u8,
}

/// Build NoteOn/NoteOff events for `notes` on `channel`, resolving the
/// hazards a real synth/DAW consuming the exported file would otherwise
/// have to guess at:
/// - a non-positive duration (defensive — callers already floor durations
///   with `.max(on + 1)`; this is the safety net for any that don't)
/// - two notes of the SAME pitch overlapping on the SAME channel (an
///   ambiguous double-trigger some synths retrigger and some silently
///   drop) — the earlier note is clipped to end exactly when the next one
///   begins, turning an overlap into a clean back-to-back retrigger
/// - an exact onset+pitch duplicate — collapses into the overlap case
///   above, which then drops the now-zero-length earlier note entirely
///
/// Does NOT sort the returned events — callers combine these with their own
/// leading meta events (ProgramChange/ControlChange) and must sort the
/// combined list with [`event_ordering_key`] so a NoteOff sharing a tick
/// with the next NoteOn is always ordered first.
fn sanitized_note_events(channel: u8, mut notes: Vec<ScheduledNote>) -> Vec<MidiEvent> {
    notes.retain(|n| n.off > n.on);
    notes.sort_by_key(|n| n.on);

    let mut sanitized: Vec<ScheduledNote> = Vec::with_capacity(notes.len());
    let mut open_idx_by_pitch: std::collections::HashMap<u8, usize> = Default::default();
    for note in notes {
        if let Some(&idx) = open_idx_by_pitch.get(&note.pitch)
            && note.on < sanitized[idx].off
        {
            // Still-sounding note of the same pitch — end it right where
            // this one begins instead of letting both ring at once.
            sanitized[idx].off = note.on;
        }
        let pitch = note.pitch;
        sanitized.push(note);
        open_idx_by_pitch.insert(pitch, sanitized.len() - 1);
    }
    // A note clipped down to (or past) its own onset by the loop above is
    // now degenerate — drop it rather than emit a zero/negative-length event.
    sanitized.retain(|n| n.off > n.on);

    let mut events = Vec::with_capacity(sanitized.len() * 2);
    for note in &sanitized {
        events.push(MidiEvent {
            tick: note.on,
            channel,
            event_type: MidiEventType::NoteOn {
                note: note.pitch,
                velocity: note.velocity,
            },
        });
        events.push(MidiEvent {
            tick: note.off,
            channel,
            event_type: MidiEventType::NoteOff { note: note.pitch },
        });
    }
    events
}

/// Sort key that deterministically breaks ties AT THE SAME TICK: NoteOff
/// before everything else before NoteOn. So a note ending and the next one
/// (same or different pitch) beginning at the identical tick never risks a
/// real-time consumer seeing the new NoteOn before the old NoteOff.
fn event_ordering_key(e: &MidiEvent) -> (u64, u8) {
    let rank = match e.event_type {
        MidiEventType::NoteOff { .. } => 0,
        MidiEventType::NoteOn { .. } => 2,
        _ => 1,
    };
    (e.tick, rank)
}

/// CC7 value for a renderer volume gain in [0, 1].
fn cc_volume(volume: f32) -> u8 {
    (volume * 127.0).round().clamp(0.0, 127.0) as u8
}

/// CC10 value for a renderer pan in [-1, 1] (64 = center).
fn cc_pan(pan: f32) -> u8 {
    (((pan + 1.0) * 0.5) * 127.0).round().clamp(0.0, 127.0) as u8
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
    let scheduled: Vec<ScheduledNote> = drum_hits
        .iter()
        .map(|&(time, note, vel)| {
            let on = secs_to_ticks(time, tempo_bpm);
            ScheduledNote {
                on,
                off: on + 120, // short duration
                pitch: note,
                velocity: vel,
            }
        })
        .collect();
    drum_track
        .events
        .extend(sanitized_note_events(9, scheduled));
    drum_track.events.sort_by_key(event_ordering_key);

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
    export_performance_midi_impl(score, spec, seed, state, None, output_path)
}

/// [`export_performance_midi`] with an explicit grammar-level
/// [`symthaea_music_theory::PerformanceDialect`] override — same
/// controlled-comparison rationale as
/// [`crate::theory_realize::realize_with_spec_and_dialect`], applied to
/// the MIDI-export path so a listening study's exported clips and its
/// audio renders agree on which dialect was used.
#[cfg(feature = "theory")]
pub fn export_performance_midi_with_dialect(
    score: &symthaea_music_theory::Score,
    spec: &symthaea_music_theory::CompositionSpec,
    seed: u64,
    state: &crate::MusicalState,
    dialect: symthaea_music_theory::PerformanceDialect,
    output_path: &Path,
) -> Result<(), String> {
    export_performance_midi_impl(
        score,
        spec,
        seed,
        state,
        Some(crate::theory_realize::map_grammar_dialect(dialect)),
        output_path,
    )
}

#[cfg(feature = "theory")]
fn export_performance_midi_impl(
    score: &symthaea_music_theory::Score,
    spec: &symthaea_music_theory::CompositionSpec,
    seed: u64,
    state: &crate::MusicalState,
    dialect_override: Option<crate::theory_realize::PerformanceDialect>,
    output_path: &Path,
) -> Result<(), String> {
    let ensemble = crate::theory_realize::resolve_spec_ensemble(spec, seed);
    let dialect = dialect_override.unwrap_or_else(|| crate::theory_realize::dialect_for_spec(spec));
    let voices = crate::theory_realize::performance_voices_with_dialect(
        score,
        ensemble,
        spec.texture.swing as f64,
        state,
        spec.texture.return_color,
        dialect,
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
        // The renderer's designed mix balance (melody 1.0 / harmony 0.5 /
        // bass 0.85…) and stereo stage (`pan_for_role`) used to be computed
        // and then DISCARDED here — every voice rendered at the GM default
        // volume, dead center. CC7/CC10 at tick 0 carry them across the
        // MIDI boundary so FluidSynth (and any DAW) hears the same balance
        // the native renderer mixes.
        let mut events = vec![
            MidiEvent {
                tick: 0,
                channel,
                event_type: MidiEventType::ProgramChange(voice.instrument.gm_program()),
            },
            MidiEvent {
                tick: 0,
                channel,
                event_type: MidiEventType::ControlChange {
                    controller: 7,
                    value: cc_volume(voice.volume),
                },
            },
            MidiEvent {
                tick: 0,
                channel,
                event_type: MidiEventType::ControlChange {
                    controller: 10,
                    value: cc_pan(voice.pan),
                },
            },
        ];
        let scheduled: Vec<ScheduledNote> = voice
            .notes
            .iter()
            .map(|n| {
                let on = secs_to_ticks(n.start_time, score.tempo_bpm);
                let off = secs_to_ticks(n.start_time + n.duration, score.tempo_bpm).max(on + 1);
                ScheduledNote {
                    on,
                    off,
                    pitch: freq_to_midi(n.frequency),
                    velocity: (n.velocity * 127.0).clamp(1.0, 127.0) as u8,
                }
            })
            .collect();
        events.extend(sanitized_note_events(channel, scheduled));
        events.sort_by_key(event_ordering_key);
        tracks.push(Track {
            name: voice.name.into(),
            channel,
            events,
        });
    }

    // The style-gated drum track, on the SAME performed timeline as the
    // pitched voices. The native renderer has had these since the Tier-2
    // percussion wave; the MIDI export silently dropped them — so under
    // FluidSynth (the live Studio default) NO style had any percussion at
    // all. Styles whose `DrumPolicy` is `None` (Classical, Waltz…) still
    // honestly get no drum track.
    let drums = crate::theory_realize::performance_drum_hits(score, spec, state, seed);
    if !drums.is_empty() {
        let mut events = vec![MidiEvent {
            tick: 0,
            channel: 9,
            // Accompaniment level, not a drum feature — mirrors the native
            // path's modest 0.35 mix gain rather than the GM default (100),
            // which would put the kit level with the melody.
            event_type: MidiEventType::ControlChange {
                controller: 7,
                value: 80,
            },
        }];
        // Percussion is one-shot: a short fixed gate is standard. A fast
        // roll on the SAME drum can still land two hits closer together
        // than that gate — `sanitized_note_events` clips the earlier hit
        // short instead of leaving two overlapping triggers on one key.
        let scheduled: Vec<ScheduledNote> = drums
            .iter()
            .map(|hit| {
                let on = secs_to_ticks(hit.time, score.tempo_bpm);
                ScheduledNote {
                    on,
                    off: on + 120,
                    pitch: hit.drum.gm_note(),
                    velocity: (hit.velocity * 127.0).clamp(1.0, 127.0) as u8,
                }
            })
            .collect();
        events.extend(sanitized_note_events(9, scheduled));
        events.sort_by_key(event_ordering_key);
        tracks.push(Track {
            name: "Drums".into(),
            channel: 9,
            events,
        });
    }
    write_midi_file(output_path, &tracks)
}

/// A short single-instrument preview phrase (a one-octave major scale up
/// to a held tonic) as a Standard MIDI File — entirely decoupled from
/// `Score`/`CompositionSpec`, so an instrument picker can be auditioned
/// without composing anything. Used by the Studio's instrument-preview
/// endpoint.
pub fn export_preview_midi(
    instrument: crate::instruments::Instrument,
    output_path: &Path,
) -> Result<(), String> {
    const TEMPO_BPM: f32 = 100.0;
    const NOTE_DUR: f32 = 0.32;
    const C4: f32 = 261.6256;
    // Major-scale semitone offsets up to the octave, then a held tonic
    // an octave above — agility and sustain color both come through.
    let semitones = [0, 2, 4, 5, 7, 9, 11, 12];
    let mut notes = Vec::new();
    let mut t = 0.0f32;
    for &s in &semitones {
        notes.push(Note {
            frequency: C4 * 2f32.powf(s as f32 / 12.0),
            start_time: t,
            duration: NOTE_DUR * 0.9,
            velocity: 0.75,
        });
        t += NOTE_DUR;
    }
    notes.push(Note {
        frequency: C4 * 2.0,
        start_time: t,
        duration: 1.2,
        velocity: 0.7,
    });

    let note_refs: Vec<&Note> = notes.iter().collect();
    let track = notes_to_track("Preview", 0, instrument.gm_program(), &note_refs, TEMPO_BPM);
    let tempo_track = Track {
        name: "Tempo".into(),
        channel: 0,
        events: vec![MidiEvent {
            tick: 0,
            channel: 0,
            event_type: MidiEventType::Tempo((60_000_000.0 / TEMPO_BPM) as u32),
        }],
    };
    write_midi_file(output_path, &[tempo_track, track])
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
        let scheduled: Vec<ScheduledNote> = score
            .voice(role)
            .iter()
            .map(|n| {
                let on = beats_to_ticks(n.onset.beats());
                let off = beats_to_ticks((n.onset + n.duration).beats()).max(on + 1);
                ScheduledNote {
                    on,
                    off,
                    pitch: n.pitch.midi(),
                    velocity: (n.velocity * 127.0).clamp(1.0, 127.0) as u8,
                }
            })
            .collect();
        events.extend(sanitized_note_events(channel, scheduled));
        events.sort_by_key(event_ordering_key);
        tracks.push(Track {
            name: name.into(),
            channel,
            events,
        });
    }

    write_midi_file(output_path, &tracks)
}

fn notes_to_track(name: &str, channel: u8, program: u8, notes: &[&Note], tempo_bpm: f32) -> Track {
    let mut events = vec![MidiEvent {
        tick: 0,
        channel,
        event_type: MidiEventType::ProgramChange(program),
    }];

    let scheduled: Vec<ScheduledNote> = notes
        .iter()
        .map(|note| {
            let on = secs_to_ticks(note.start_time, tempo_bpm);
            let off = on + secs_to_ticks(note.duration, tempo_bpm).max(1);
            ScheduledNote {
                on,
                off,
                pitch: freq_to_midi(note.frequency),
                velocity: (note.velocity * 127.0).clamp(1.0, 127.0) as u8,
            }
        })
        .collect();
    events.extend(sanitized_note_events(channel, scheduled));
    events.sort_by_key(event_ordering_key);

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
            MidiEventType::ControlChange { controller, value } => {
                data.push(0xB0 | (event.channel & 0x0F));
                data.push(*controller & 0x7F);
                data.push(*value & 0x7F);
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
    fn overlapping_same_pitch_notes_are_clipped_to_a_clean_retrigger() {
        // Two same-pitch, same-channel notes whose intervals overlap
        // (0..10 and 5..15) must never both sound at once: the earlier one
        // is clipped to end exactly where the later one begins.
        let notes = vec![
            ScheduledNote {
                on: 0,
                off: 10,
                pitch: 60,
                velocity: 100,
            },
            ScheduledNote {
                on: 5,
                off: 15,
                pitch: 60,
                velocity: 90,
            },
        ];
        let events = sanitized_note_events(0, notes);
        // Both notes survive (neither collapses to zero length), so exactly
        // 2 NoteOn + 2 NoteOff.
        let ons: Vec<&MidiEvent> = events
            .iter()
            .filter(|e| matches!(e.event_type, MidiEventType::NoteOn { .. }))
            .collect();
        let offs: Vec<&MidiEvent> = events
            .iter()
            .filter(|e| matches!(e.event_type, MidiEventType::NoteOff { .. }))
            .collect();
        assert_eq!(ons.len(), 2, "{events:?}");
        assert_eq!(offs.len(), 2, "{events:?}");
        // The first note's off must have been clipped to the second's onset.
        let first_off = offs.iter().find(|e| e.tick < 15).expect("clipped off");
        assert_eq!(first_off.tick, 5, "{events:?}");
        let second_off = offs.iter().find(|e| e.tick == 15).expect("unclipped off");
        assert_eq!(second_off.tick, 15);
    }

    #[test]
    fn exact_onset_and_pitch_duplicates_collapse_to_one_note() {
        // Two notes sharing an onset AND a pitch (the same-pitch overlap
        // case at its most extreme) must not both survive as independent
        // triggers -- the earlier one collapses to zero length and is
        // dropped entirely.
        let notes = vec![
            ScheduledNote {
                on: 0,
                off: 10,
                pitch: 60,
                velocity: 100,
            },
            ScheduledNote {
                on: 0,
                off: 8,
                pitch: 60,
                velocity: 90,
            },
        ];
        let events = sanitized_note_events(0, notes);
        let ons: Vec<&MidiEvent> = events
            .iter()
            .filter(|e| matches!(e.event_type, MidiEventType::NoteOn { .. }))
            .collect();
        assert_eq!(
            ons.len(),
            1,
            "expected exactly one surviving note: {events:?}"
        );
        let offs: Vec<&MidiEvent> = events
            .iter()
            .filter(|e| matches!(e.event_type, MidiEventType::NoteOff { .. }))
            .collect();
        assert_eq!(offs.len(), 1);
        assert_eq!(offs[0].tick, 8);
    }

    #[test]
    fn distinct_pitches_overlap_freely() {
        // The overlap guard is per-pitch, not a blanket "no simultaneous
        // notes on this channel" rule -- a real chord/harmony voice must
        // still be able to sound multiple distinct pitches at once.
        let notes = vec![
            ScheduledNote {
                on: 0,
                off: 20,
                pitch: 60,
                velocity: 100,
            },
            ScheduledNote {
                on: 5,
                off: 15,
                pitch: 64,
                velocity: 90,
            },
        ];
        let events = sanitized_note_events(0, notes);
        let offs: Vec<u64> = events
            .iter()
            .filter(|e| matches!(e.event_type, MidiEventType::NoteOff { .. }))
            .map(|e| e.tick)
            .collect();
        // Neither note was clipped -- both offs are unchanged.
        assert!(offs.contains(&20), "{offs:?}");
        assert!(offs.contains(&15), "{offs:?}");
    }

    #[test]
    fn zero_or_negative_duration_notes_are_dropped() {
        let notes = vec![
            ScheduledNote {
                on: 10,
                off: 10,
                pitch: 60,
                velocity: 100,
            },
            ScheduledNote {
                on: 20,
                off: 15,
                pitch: 62,
                velocity: 100,
            },
        ];
        let events = sanitized_note_events(0, notes);
        assert!(events.is_empty(), "{events:?}");
    }

    #[test]
    fn note_off_is_ordered_before_note_on_at_the_same_tick() {
        // A note ending and a different one beginning at the identical tick
        // must always emit the NoteOff first, regardless of push order.
        let mut events = vec![
            MidiEvent {
                tick: 100,
                channel: 0,
                event_type: MidiEventType::NoteOn {
                    note: 64,
                    velocity: 90,
                },
            },
            MidiEvent {
                tick: 100,
                channel: 0,
                event_type: MidiEventType::NoteOff { note: 60 },
            },
        ];
        events.sort_by_key(event_ordering_key);
        assert!(matches!(
            events[0].event_type,
            MidiEventType::NoteOff { .. }
        ));
        assert!(matches!(events[1].event_type, MidiEventType::NoteOn { .. }));
    }

    #[test]
    fn preview_midi_round_trips_and_uses_the_instruments_gm_program() {
        let path = std::env::temp_dir().join(format!(
            "muse_preview_export_test_{}.mid",
            std::process::id()
        ));
        export_preview_midi(crate::instruments::Instrument::Violin, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let smf = midly::Smf::parse(&bytes).expect("exported file must be valid SMF");
        // Tempo track + the single preview track.
        assert_eq!(smf.tracks.len(), 2);
        let program = smf.tracks[1].iter().find_map(|e| match &e.kind {
            midly::TrackEventKind::Midi {
                message: midly::MidiMessage::ProgramChange { program },
                ..
            } => Some(program.as_int()),
            _ => None,
        });
        assert_eq!(
            program,
            Some(crate::instruments::Instrument::Violin.gm_program())
        );
        // 8 scale notes + 1 held tonic = 9 NoteOn events.
        let note_ons = smf.tracks[1]
            .iter()
            .filter(|e| {
                matches!(&e.kind,
                    midly::TrackEventKind::Midi { message: midly::MidiMessage::NoteOn { vel, .. }, .. }
                        if vel.as_int() > 0)
            })
            .count();
        assert_eq!(note_ons, 9);
        let _ = std::fs::remove_file(&path);
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
        // Every score note becomes one NoteOn (and its NoteOff) -- EXCEPT
        // where `sanitized_note_events` correctly collapses a genuine
        // same-onset-and-pitch duplicate within one voice (confirmed present
        // in this exact score: CounterMelody has two pitch-67 notes both at
        // onset 86.5 beats) into a single clean trigger rather than an
        // ambiguous double NoteOn. So `ons`/`offs` may be slightly below
        // `score.notes.len()`, never above it, and never collapse more than
        // a handful of notes (a large gap would mean the sanitizer is
        // over-eager, not doing its job).
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
        assert_eq!(ons, offs, "every surviving NoteOn must have its NoteOff");
        assert!(
            ons <= score.notes.len(),
            "ons={ons} > notes={}",
            score.notes.len()
        );
        assert!(
            ons >= score.notes.len() - 5,
            "sanitizer collapsed too many notes: ons={ons} notes={}",
            score.notes.len()
        );
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

    #[cfg(feature = "theory")]
    #[test]
    fn performance_export_carries_mix_balance_and_pan_cc() {
        use symthaea_music_theory::{MusicalIntent, Style};
        let intent = MusicalIntent::default();
        let spec = Style::Classical.spec();
        let score = symthaea_music_theory::compose_with_spec(&intent, &spec);
        let path =
            std::env::temp_dir().join(format!("muse_perf_export_cc_{}.mid", std::process::id()));
        export_performance_midi(&score, &spec, 0, &crate::MusicalState::default(), &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let smf = midly::Smf::parse(&bytes).expect("valid SMF");
        // Collect (track name → [(controller, value)]).
        let mut ccs: std::collections::HashMap<String, Vec<(u8, u8)>> = Default::default();
        for track in &smf.tracks {
            let mut name = String::new();
            let mut list = Vec::new();
            for e in track {
                match &e.kind {
                    midly::TrackEventKind::Meta(midly::MetaMessage::TrackName(n)) => {
                        name = String::from_utf8_lossy(n).into_owned();
                    }
                    midly::TrackEventKind::Midi {
                        message: midly::MidiMessage::Controller { controller, value },
                        ..
                    } => list.push((controller.as_int(), value.as_int())),
                    _ => {}
                }
            }
            if !name.is_empty() {
                ccs.insert(name, list);
            }
        }
        // The renderer's designed balance must survive the MIDI boundary:
        // melody full volume panned slightly right, harmony at half volume
        // panned left, bass anchored center at 0.85.
        let melody = &ccs["Melody"];
        assert!(melody.contains(&(7, cc_volume(1.0))), "{melody:?}");
        assert!(melody.contains(&(10, cc_pan(0.25))), "{melody:?}");
        let harmony = &ccs["Harmony"];
        assert!(harmony.contains(&(7, cc_volume(0.5))), "{harmony:?}");
        assert!(harmony.contains(&(10, cc_pan(-0.4))), "{harmony:?}");
        let bass = &ccs["Bass"];
        assert!(bass.contains(&(7, cc_volume(0.85))), "{bass:?}");
        assert!(bass.contains(&(10, cc_pan(0.0))), "{bass:?}");
        std::fs::remove_file(&path).ok();
    }

    #[cfg(feature = "theory")]
    #[test]
    fn performance_export_writes_style_drums_only_for_drum_styles() {
        use symthaea_music_theory::{MusicalIntent, Style};
        let intent = MusicalIntent::default();
        let channel9_note_ons = |style: Style, tag: &str| -> Vec<u8> {
            let spec = style.spec();
            let score = symthaea_music_theory::compose_with_spec(&intent, &spec);
            let path = std::env::temp_dir().join(format!(
                "muse_perf_export_drums_{tag}_{}.mid",
                std::process::id()
            ));
            export_performance_midi(&score, &spec, 0, &crate::MusicalState::default(), &path)
                .unwrap();
            let bytes = std::fs::read(&path).unwrap();
            let smf = midly::Smf::parse(&bytes).expect("valid SMF");
            let keys: Vec<u8> = smf
                .tracks
                .iter()
                .flatten()
                .filter_map(|e| match &e.kind {
                    midly::TrackEventKind::Midi {
                        channel,
                        message: midly::MidiMessage::NoteOn { key, vel },
                    } if channel.as_int() == 9 && vel.as_int() > 0 => Some(key.as_int()),
                    _ => None,
                })
                .collect();
            std::fs::remove_file(&path).ok();
            keys
        };
        // Playful's Backbeat policy must reach the exported MIDI as a real
        // channel-9 kit (kick 36, snare 38, hats 42)...
        let playful = channel9_note_ons(Style::Playful, "playful");
        assert!(playful.contains(&36), "kick missing: {playful:?}");
        assert!(playful.contains(&38), "snare missing: {playful:?}");
        assert!(playful.contains(&42), "hats missing: {playful:?}");
        // ...while Classical's DrumPolicy::None honestly stays kit-free.
        let classical = channel9_note_ons(Style::Classical, "classical");
        assert!(classical.is_empty(), "unexpected drums: {classical:?}");
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


// ─── Part-aware, timeline-bound MIDI realization ────────────────────────────
//
// This path deliberately does not replace the legacy role-based exporters.
// It is the score-native realization boundary for callers that already have
// a generator-authored PartId for every note and a validated time-indexed
// orchestration timeline. It never reconstructs part identity from VoiceRole,
// pitch register, velocity, or MIDI channel.
//
// The MIDI program mapping is caller supplied. This module does not infer a
// patch from an instrument family and does not claim that a raw capability
// profile is independently evidence-qualified. Qualification remains the
// responsibility of the capability materializer and orchestration boundary.
//
// Note ordinals in the receipt are scoped to score_sha256. They are not
// persistent EventIds and must not be reused after the source score changes.

#[cfg(feature = "theory")]
pub mod part_aware_v1 {
    use super::{event_ordering_key, encode_track, MidiEvent, MidiEventType, Track};
    use serde::{Deserialize, Serialize};
    use sha2::{Digest, Sha256};
    use std::cmp::Ordering;
    use std::collections::{BTreeMap, BTreeSet};

    use symthaea_music_theory::orchestration::InstrumentCapabilityV1;
    use symthaea_music_theory::orchestration_timeline::{
        validate_orchestration_timeline, OrchestrationTimelineIssueV1,
        OrchestrationTimelineV1, RationalBeatV1, ORCHESTRATION_SCORE_DIGEST_SCHEMA_V1,
    };
    use symthaea_music_theory::rhythm::Duration;
    use symthaea_music_theory::score::{Score, ScoreNote};

    pub const PART_AWARE_MIDI_EXPORT_SCHEMA_V1: &str =
        "muse-part-aware-midi-export-v1";
    pub const PART_AWARE_MIDI_TIMELINE_DIGEST_SCHEMA_V1: &str =
        "muse-orchestration-timeline-serde-json-sha256-v1";
    pub const MIDI_TICKS_PER_QUARTER_V1: u16 = 960;

    /// Explicit caller-supplied mapping from one concrete instrument instance
    /// to a General MIDI Level 1 program number (0-based).
    ///
    /// mapping_reference identifies the external mapping decision or record.
    /// This exporter checks that the mapping is complete and unambiguous; it
    /// does not authenticate that reference or certify acoustic patch quality.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct MidiProgramBindingV1 {
        pub instrument_id: String,
        pub program: u8,
        pub mapping_reference: String,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct PartAwareMidiTrackReceiptV1 {
        pub part_id: u16,
        /// Zero-based Standard MIDI File track index. Track zero is the
        /// tempo/time-signature track.
        pub track_index: u16,
        /// Stable channel for this persistent part across instrument handoffs.
        pub channel: u8,
        pub instrument_ids: Vec<String>,
        pub mapping_references: Vec<String>,
        pub note_count: usize,
    }

    /// One source score note and its exact encoded MIDI note-on/note-off pair.
    ///
    /// score_note_ordinal is meaningful only together with score_sha256.
    /// onset_tick and offset_tick use nearest-ties-to-even quantization at
    /// MIDI_TICKS_PER_QUARTER_V1. The original rational beat values are kept
    /// so the quantization can be independently recalculated.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct PartAwareMidiEventMapV1 {
        pub score_sha256: String,
        pub score_note_ordinal: usize,
        pub part_id: u16,
        pub segment_id: String,
        pub instrument_id: String,
        pub track_index: u16,
        pub channel: u8,
        pub midi_pitch: u8,
        pub source_velocity: f32,
        pub midi_velocity: u8,
        pub source_onset: RationalBeatV1,
        pub source_duration: RationalBeatV1,
        pub onset_tick: u64,
        pub offset_tick: u64,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct PartAwareMidiProgramChangeReceiptV1 {
        pub part_id: u16,
        pub segment_id: String,
        pub instrument_id: String,
        pub track_index: u16,
        pub channel: u8,
        pub program: u8,
        pub mapping_reference: String,
        pub source_start: RationalBeatV1,
        pub start_tick: u64,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct PartAwareMidiExportReceiptV1 {
        pub schema: String,
        pub score_digest_schema: String,
        pub score_sha256: String,
        pub timeline_digest_schema: String,
        pub timeline_sha256: String,
        pub midi_sha256: String,
        pub ticks_per_quarter: u16,
        pub part_count: usize,
        pub note_count: usize,
        pub track_count: usize,
        pub tracks: Vec<PartAwareMidiTrackReceiptV1>,
        pub program_changes: Vec<PartAwareMidiProgramChangeReceiptV1>,
        /// Entries retain input Score.notes order; ordinals are digest-scoped.
        pub event_map: Vec<PartAwareMidiEventMapV1>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum PartAwareMidiExportErrorV1 {
        TimelineInvalid {
            issues: Vec<OrchestrationTimelineIssueV1>,
        },
        InvalidTempoOrMeter,
        TooManyParts {
            actual: usize,
            supported: usize,
        },
        InvalidProgramBinding {
            detail: String,
        },
        MissingProgramBinding {
            instrument_id: String,
        },
        UnusedProgramBinding {
            instrument_id: String,
        },
        InvalidRationalTime {
            score_note_ordinal: usize,
        },
        MidiTickOverflow {
            score_note_ordinal: usize,
        },
        MidiDeltaExceedsStandardMaximum {
            track_index: u16,
            delta_ticks: u64,
        },
        UnrepresentableNoteDuration {
            score_note_ordinal: usize,
        },
        InvalidVelocity {
            score_note_ordinal: usize,
        },
        MissingOrAmbiguousSegment {
            score_note_ordinal: usize,
        },
        ProgramHandoffQuantizationCollision {
            part_id: u16,
            tick: u64,
        },
        OverlappingSamePitchNotes {
            part_id: u16,
            pitch: u8,
            first_score_note_ordinal: usize,
            second_score_note_ordinal: usize,
        },
        ScoreSerialization(String),
        ScoreDigestMismatch {
            expected: String,
            actual: String,
        },
        TimelineSerialization(String),
        MidiWrite(String),
    }

    /// A ready-to-write MIDI artifact plus the receipt that binds every
    /// encoded note pair back to the exact source score and timeline.
    #[derive(Debug, Clone, PartialEq)]
    pub struct PartAwareMidiArtifactV1 {
        pub midi_bytes: Vec<u8>,
        pub receipt: PartAwareMidiExportReceiptV1,
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        format!("{:x}", hasher.finalize())
    }

    fn domain_separated_sha256(schema: &str, bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(schema.as_bytes());
        hasher.update([0]);
        hasher.update(bytes);
        format!("{:x}", hasher.finalize())
    }

    fn cmp_rational(
        left_num: i128,
        left_den: i128,
        right_num: i128,
        right_den: i128,
    ) -> Option<Ordering> {
        if left_den <= 0 || right_den <= 0 {
            return None;
        }
        let left = left_num.checked_mul(right_den)?;
        let right = right_num.checked_mul(left_den)?;
        Some(left.cmp(&right))
    }

    fn rational_to_tick(numerator: i128, denominator: i128) -> Option<u64> {
        if numerator < 0 || denominator <= 0 {
            return None;
        }
        let scaled = numerator.checked_mul(i128::from(MIDI_TICKS_PER_QUARTER_V1))?;
        let quotient = scaled / denominator;
        let remainder = scaled % denominator;
        let twice_remainder = remainder.checked_mul(2)?;
        let round_up = twice_remainder > denominator
            || (twice_remainder == denominator && quotient % 2 != 0);
        let rounded = quotient.checked_add(if round_up { 1_i128 } else { 0_i128 })?;
        u64::try_from(rounded).ok()
    }

    fn rational_beat_to_tick(beat: RationalBeatV1) -> Option<u64> {
        rational_to_tick(
            i128::from(beat.numerator),
            i128::from(beat.denominator),
        )
    }

    fn duration_pair(value: Duration) -> (i128, i128) {
        (i128::from(value.num()), i128::from(value.den()))
    }

    fn note_end_pair(note: &ScoreNote) -> Option<(i128, i128)> {
        let (on_num, on_den) = duration_pair(note.onset);
        let (duration_num, duration_den) = duration_pair(note.duration);
        if on_num < 0 || duration_num <= 0 || on_den <= 0 || duration_den <= 0 {
            return None;
        }
        let left = on_num.checked_mul(duration_den)?;
        let right = duration_num.checked_mul(on_den)?;
        let numerator = left.checked_add(right)?;
        let denominator = on_den.checked_mul(duration_den)?;
        (denominator > 0).then_some((numerator, denominator))
    }

    fn note_fits_segment(note: &ScoreNote, segment: &symthaea_music_theory::orchestration_timeline::OrchestrationSegmentV1) -> Option<bool> {
        let (on_num, on_den) = duration_pair(note.onset);
        let (end_num, end_den) = note_end_pair(note)?;
        let onset_after_start = cmp_rational(
            on_num,
            on_den,
            i128::from(segment.start.numerator),
            i128::from(segment.start.denominator),
        )?;
        let end_before_segment_end = cmp_rational(
            end_num,
            end_den,
            i128::from(segment.end.numerator),
            i128::from(segment.end.denominator),
        )?;
        Some(
            onset_after_start != Ordering::Less
                && end_before_segment_end != Ordering::Greater,
        )
    }

    fn midi_velocity(value: f32, ordinal: usize) -> Result<u8, PartAwareMidiExportErrorV1> {
        if !value.is_finite() || value <= 0.0 || value > 1.0 {
            return Err(PartAwareMidiExportErrorV1::InvalidVelocity {
                score_note_ordinal: ordinal,
            });
        }
        Ok((value * 127.0).round().clamp(1.0, 127.0) as u8)
    }

    /// Encode a Score using persistent PartId and the exact time-indexed
    /// orchestration plan. No VoiceRole grouping or pitch/register heuristics
    /// are used. The expected digest must come from the trusted score-demand
    /// boundary; the timeline validator checks the supplied score against it.
    ///
    /// The caller supplies the explicit MIDI program mapping. Capabilities
    /// passed here are the inputs to the timeline validator; this function
    /// does not independently authenticate their evidence provenance.
    pub fn encode_score_midi_part_aware_v1(
        score: &Score,
        timeline: &OrchestrationTimelineV1,
        instruments: &[InstrumentCapabilityV1],
        program_bindings: &[MidiProgramBindingV1],
        expected_score_sha256: &str,
    ) -> Result<PartAwareMidiArtifactV1, PartAwareMidiExportErrorV1> {
        // Recompute the exact score-demand digest from the score being
        // serialized. The timeline validator checks agreement with the
        // caller-supplied expected digest; this extra check prevents a stale
        // trusted digest from being paired with a changed Score.
        let serialized_score = serde_json::to_vec(score).map_err(|error| {
            PartAwareMidiExportErrorV1::ScoreSerialization(error.to_string())
        })?;
        let actual_score_sha256 = sha256_hex(&serialized_score);
        if actual_score_sha256 != expected_score_sha256 {
            return Err(PartAwareMidiExportErrorV1::ScoreDigestMismatch {
                expected: expected_score_sha256.to_owned(),
                actual: actual_score_sha256,
            });
        }

        let timeline_report = validate_orchestration_timeline(
            score,
            timeline,
            instruments,
            expected_score_sha256,
        );
        if !timeline_report.valid {
            return Err(PartAwareMidiExportErrorV1::TimelineInvalid {
                issues: timeline_report.issues,
            });
        }
        if !score.tempo_bpm.is_finite()
            || score.tempo_bpm <= 0.0
            || score.meter == 0
        {
            return Err(PartAwareMidiExportErrorV1::InvalidTempoOrMeter);
        }

        let used_instruments: BTreeSet<String> = timeline
            .segments
            .iter()
            .map(|segment| segment.instrument_id.clone())
            .collect();
        let mut binding_by_instrument = BTreeMap::<String, &MidiProgramBindingV1>::new();
        for binding in program_bindings {
            if binding.instrument_id.trim().is_empty()
                || binding.mapping_reference.trim().is_empty()
                || binding.program > 127
            {
                return Err(PartAwareMidiExportErrorV1::InvalidProgramBinding {
                    detail: "instrument ID, mapping reference and 0..=127 program are required".into(),
                });
            }
            if binding_by_instrument
                .insert(binding.instrument_id.clone(), binding)
                .is_some()
            {
                return Err(PartAwareMidiExportErrorV1::InvalidProgramBinding {
                    detail: format!("duplicate program binding for {}", binding.instrument_id),
                });
            }
        }
        for instrument_id in &used_instruments {
            if !binding_by_instrument.contains_key(instrument_id) {
                return Err(PartAwareMidiExportErrorV1::MissingProgramBinding {
                    instrument_id: instrument_id.clone(),
                });
            }
        }
        for instrument_id in binding_by_instrument.keys() {
            if !used_instruments.contains(instrument_id) {
                return Err(PartAwareMidiExportErrorV1::UnusedProgramBinding {
                    instrument_id: instrument_id.clone(),
                });
            }
        }

        // General MIDI convention reserves channel 10 (zero-based channel 9)
        // for percussion. We allocate one stable channel per PartId so
        // simultaneous same-pitch notes in independent parts remain distinct.
        const CHANNELS_V1: [u8; 15] =
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15];
        let part_ids: Vec<u16> = score
            .parts()
            .into_iter()
            .map(|part| part.0)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        if part_ids.len() > CHANNELS_V1.len() {
            return Err(PartAwareMidiExportErrorV1::TooManyParts {
                actual: part_ids.len(),
                supported: CHANNELS_V1.len(),
            });
        }
        let channel_by_part: BTreeMap<u16, u8> = part_ids
            .iter()
            .enumerate()
            .map(|(index, part_id)| (*part_id, CHANNELS_V1[index]))
            .collect();
        let track_index_by_part: BTreeMap<u16, u16> = part_ids
            .iter()
            .enumerate()
            .map(|(index, part_id)| (*part_id, (index + 1) as u16))
            .collect();

        let usec_per_beat = (60_000_000.0_f64 / f64::from(score.tempo_bpm)).round();
        if !usec_per_beat.is_finite() || usec_per_beat < 1.0 || usec_per_beat > f64::from(u32::MAX) {
            return Err(PartAwareMidiExportErrorV1::InvalidTempoOrMeter);
        }

        let tempo_track = Track {
            name: "Tempo".into(),
            channel: 0,
            events: vec![
                MidiEvent {
                    tick: 0,
                    channel: 0,
                    event_type: MidiEventType::Tempo(usec_per_beat as u32),
                },
                MidiEvent {
                    tick: 0,
                    channel: 0,
                    event_type: MidiEventType::TimeSignature(score.meter, 4),
                },
            ],
        };

        let mut events_by_part = BTreeMap::<u16, Vec<MidiEvent>>::new();
        for part_id in &part_ids {
            events_by_part.insert(*part_id, Vec::new());
        }

        // Program changes are emitted at every timeline-segment boundary.
        // A continuing PartId keeps its MIDI channel while its instrument
        // instance/patch changes explicitly at the handoff.
        let mut program_at_tick =
            BTreeMap::<(u16, u64), (u8, String)>::new();
        let mut program_change_receipts = Vec::with_capacity(timeline.segments.len());
        for segment in &timeline.segments {
            let tick = rational_beat_to_tick(segment.start).ok_or(
                PartAwareMidiExportErrorV1::InvalidRationalTime {
                    score_note_ordinal: usize::MAX,
                },
            )?;
            let binding = binding_by_instrument
                .get(&segment.instrument_id)
                .expect("all timeline instruments have validated program bindings");
            let channel = *channel_by_part
                .get(&segment.part_id)
                .expect("validated score part has a MIDI channel");
            if let Some((previous_program, previous_instrument_id)) =
                program_at_tick.insert(
                    (segment.part_id, tick),
                    (binding.program, segment.instrument_id.clone()),
                )
            {
                // Even identical GM program numbers cannot distinguish two
                // different concrete instrument instances on the same channel.
                if previous_program != binding.program
                    || previous_instrument_id != segment.instrument_id
                {
                    return Err(
                        PartAwareMidiExportErrorV1::ProgramHandoffQuantizationCollision {
                            part_id: segment.part_id,
                            tick,
                        },
                    );
                }
            }
            events_by_part
                .get_mut(&segment.part_id)
                .expect("validated score part has an event stream")
                .push(MidiEvent {
                    tick,
                    channel,
                    event_type: MidiEventType::ProgramChange(binding.program),
                });
            program_change_receipts.push(PartAwareMidiProgramChangeReceiptV1 {
                part_id: segment.part_id,
                segment_id: segment.id.clone(),
                instrument_id: segment.instrument_id.clone(),
                track_index: *track_index_by_part
                    .get(&segment.part_id)
                    .expect("validated score part has a MIDI track"),
                channel,
                program: binding.program,
                mapping_reference: binding.mapping_reference.clone(),
                source_start: segment.start,
                start_tick: tick,
            });
        }

        let mut event_map = Vec::with_capacity(score.notes.len());
        let mut same_pitch_intervals =
            BTreeMap::<(u16, u8), Vec<(i128, i128, i128, i128, usize)>>::new();

        for (ordinal, note) in score.notes.iter().enumerate() {
            let mut matched_segment = None;
            for segment in timeline
                .segments
                .iter()
                .filter(|segment| segment.part_id == note.part.0)
            {
                if note_fits_segment(note, segment).ok_or(
                    PartAwareMidiExportErrorV1::InvalidRationalTime {
                        score_note_ordinal: ordinal,
                    },
                )? {
                    if matched_segment.is_some() {
                        return Err(
                            PartAwareMidiExportErrorV1::MissingOrAmbiguousSegment {
                                score_note_ordinal: ordinal,
                            },
                        );
                    }
                    matched_segment = Some(segment);
                }
            }
            let segment = matched_segment.ok_or(
                PartAwareMidiExportErrorV1::MissingOrAmbiguousSegment {
                    score_note_ordinal: ordinal,
                },
            )?;

            let (on_num, on_den) = duration_pair(note.onset);
            let (end_num, end_den) = note_end_pair(note).ok_or(
                PartAwareMidiExportErrorV1::InvalidRationalTime {
                    score_note_ordinal: ordinal,
                },
            )?;
            let onset_tick = rational_to_tick(on_num, on_den).ok_or(
                PartAwareMidiExportErrorV1::MidiTickOverflow {
                    score_note_ordinal: ordinal,
                },
            )?;
            let offset_tick = rational_to_tick(end_num, end_den).ok_or(
                PartAwareMidiExportErrorV1::MidiTickOverflow {
                    score_note_ordinal: ordinal,
                },
            )?;
            if offset_tick <= onset_tick {
                return Err(PartAwareMidiExportErrorV1::UnrepresentableNoteDuration {
                    score_note_ordinal: ordinal,
                });
            }

            let velocity = midi_velocity(note.velocity, ordinal)?;
            let channel = *channel_by_part
                .get(&note.part.0)
                .expect("validated score part has a MIDI channel");
            let track_index = *track_index_by_part
                .get(&note.part.0)
                .expect("validated score part has a MIDI track");

            same_pitch_intervals
                .entry((note.part.0, note.pitch.midi()))
                .or_default()
                .push((on_num, on_den, end_num, end_den, ordinal));

            let events = events_by_part
                .get_mut(&note.part.0)
                .expect("validated score part has an event stream");
            events.push(MidiEvent {
                tick: onset_tick,
                channel,
                event_type: MidiEventType::NoteOn {
                    note: note.pitch.midi(),
                    velocity,
                },
            });
            events.push(MidiEvent {
                tick: offset_tick,
                channel,
                event_type: MidiEventType::NoteOff {
                    note: note.pitch.midi(),
                },
            });

            event_map.push(PartAwareMidiEventMapV1 {
                score_sha256: expected_score_sha256.to_owned(),
                score_note_ordinal: ordinal,
                part_id: note.part.0,
                segment_id: segment.id.clone(),
                instrument_id: segment.instrument_id.clone(),
                track_index,
                channel,
                midi_pitch: note.pitch.midi(),
                source_velocity: note.velocity,
                midi_velocity: velocity,
                source_onset: RationalBeatV1 {
                    numerator: note.onset.num(),
                    denominator: note.onset.den(),
                },
                source_duration: RationalBeatV1 {
                    numerator: note.duration.num(),
                    denominator: note.duration.den(),
                },
                onset_tick,
                offset_tick,
            });
        }

        // Standard MIDI NoteOff addresses channel+pitch, not a unique source
        // event. Compare exact rational source intervals, not quantized ticks:
        // two source overlaps must not become an apparently clean retrigger
        // merely because both boundaries round to the same MIDI tick.
        for ((part_id, pitch), intervals) in &mut same_pitch_intervals {
            intervals.sort_by(|left, right| {
                cmp_rational(left.0, left.1, right.0, right.1)
                    .expect("validated score rational times are inside the bounded domain")
                    .then_with(|| left.4.cmp(&right.4))
            });
            let mut previous_max_end: Option<(i128, i128, usize)> = None;
            for (onset_num, onset_den, end_num, end_den, ordinal) in
                intervals.iter().copied()
            {
                if let Some((previous_end_num, previous_end_den, previous_ordinal)) =
                    previous_max_end
                {
                    let onset_vs_previous_end = cmp_rational(
                        onset_num,
                        onset_den,
                        previous_end_num,
                        previous_end_den,
                    )
                    .ok_or(PartAwareMidiExportErrorV1::InvalidRationalTime {
                        score_note_ordinal: ordinal,
                    })?;
                    if onset_vs_previous_end == Ordering::Less {
                        return Err(
                            PartAwareMidiExportErrorV1::OverlappingSamePitchNotes {
                                part_id: *part_id,
                                pitch: *pitch,
                                first_score_note_ordinal: previous_ordinal,
                                second_score_note_ordinal: ordinal,
                            },
                        );
                    }
                    let end_vs_previous_end = cmp_rational(
                        end_num,
                        end_den,
                        previous_end_num,
                        previous_end_den,
                    )
                    .ok_or(PartAwareMidiExportErrorV1::InvalidRationalTime {
                        score_note_ordinal: ordinal,
                    })?;
                    if end_vs_previous_end == Ordering::Greater {
                        previous_max_end = Some((end_num, end_den, ordinal));
                    }
                } else {
                    previous_max_end = Some((end_num, end_den, ordinal));
                }
            }
        }

        let mut tracks = Vec::with_capacity(part_ids.len() + 1);
        for part_id in &part_ids {
            let channel = channel_by_part[part_id];
            let mut events = events_by_part.remove(part_id).unwrap_or_default();
            events.sort_by_key(event_ordering_key);
            let mut previous_tick = 0_u64;
            for event in &events {
                let delta = event.tick.saturating_sub(previous_tick);
                if delta > 0x0fff_ffff {
                    return Err(
                        PartAwareMidiExportErrorV1::MidiDeltaExceedsStandardMaximum {
                            track_index: (tracks.len() + 1) as u16,
                            delta_ticks: delta,
                        },
                    );
                }
                previous_tick = event.tick;
            }
            tracks.push(Track {
                name: format!("Part-{}", part_id),
                channel,
                events,
            });
        }
        let track_receipts: Vec<PartAwareMidiTrackReceiptV1> = part_ids
            .iter()
            .enumerate()
            .map(|(part_index, part_id)| {
                let instrument_ids: Vec<String> = timeline
                    .segments
                    .iter()
                    .filter(|segment| segment.part_id == *part_id)
                    .map(|segment| segment.instrument_id.clone())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                let mapping_references: Vec<String> = instrument_ids
                    .iter()
                    .filter_map(|instrument_id| {
                        binding_by_instrument
                            .get(instrument_id)
                            .map(|binding| binding.mapping_reference.clone())
                    })
                    .collect();
                PartAwareMidiTrackReceiptV1 {
                    part_id: *part_id,
                    track_index: (part_index + 1) as u16,
                    channel: channel_by_part[part_id],
                    instrument_ids,
                    mapping_references,
                    note_count: event_map
                        .iter()
                        .filter(|mapping| mapping.part_id == *part_id)
                        .count(),
                }
            })
            .collect();

        // Tempo track precedes all persistent-part tracks in deterministic order.
        let mut smf_tracks = Vec::with_capacity(tracks.len() + 1);
        smf_tracks.push(tempo_track);
        smf_tracks.extend(tracks);
        let midi_bytes = encode_smf_type_one(&smf_tracks);
        let midi_sha256 = sha256_hex(&midi_bytes);

        let timeline_bytes = serde_json::to_vec(timeline).map_err(|error| {
            PartAwareMidiExportErrorV1::TimelineSerialization(error.to_string())
        })?;
        let timeline_sha256 = domain_separated_sha256(
            PART_AWARE_MIDI_TIMELINE_DIGEST_SCHEMA_V1,
            &timeline_bytes,
        );

        program_change_receipts.sort_by_key(|item| {
            (item.part_id, item.start_tick, item.segment_id.clone())
        });
        let receipt = PartAwareMidiExportReceiptV1 {
            schema: PART_AWARE_MIDI_EXPORT_SCHEMA_V1.into(),
            score_digest_schema: ORCHESTRATION_SCORE_DIGEST_SCHEMA_V1.into(),
            score_sha256: expected_score_sha256.to_owned(),
            timeline_digest_schema: PART_AWARE_MIDI_TIMELINE_DIGEST_SCHEMA_V1.into(),
            timeline_sha256,
            midi_sha256,
            ticks_per_quarter: MIDI_TICKS_PER_QUARTER_V1,
            part_count: part_ids.len(),
            note_count: event_map.len(),
            track_count: smf_tracks.len(),
            tracks: track_receipts,
            program_changes: program_change_receipts,
            event_map,
        };

        Ok(PartAwareMidiArtifactV1 {
            midi_bytes,
            receipt,
        })
    }

    /// Write the part-aware MIDI bytes to disk and return the same content
    /// receipt. The file hash is over the exact bytes written.
    pub fn export_score_midi_part_aware_v1(
        score: &Score,
        timeline: &OrchestrationTimelineV1,
        instruments: &[InstrumentCapabilityV1],
        program_bindings: &[MidiProgramBindingV1],
        expected_score_sha256: &str,
        output_path: &std::path::Path,
    ) -> Result<PartAwareMidiExportReceiptV1, PartAwareMidiExportErrorV1> {
        let artifact = encode_score_midi_part_aware_v1(
            score,
            timeline,
            instruments,
            program_bindings,
            expected_score_sha256,
        )?;
        std::fs::write(output_path, &artifact.midi_bytes)
            .map_err(|error| PartAwareMidiExportErrorV1::MidiWrite(error.to_string()))?;
        let written = std::fs::read(output_path)
            .map_err(|error| PartAwareMidiExportErrorV1::MidiWrite(error.to_string()))?;
        let written_sha256 = sha256_hex(&written);
        if written_sha256 != artifact.receipt.midi_sha256 {
            return Err(PartAwareMidiExportErrorV1::MidiWrite(
                "written MIDI bytes do not match the encoded artifact digest".into(),
            ));
        }
        Ok(artifact.receipt)
    }

    fn encode_smf_type_one(tracks: &[Track]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"MThd");
        bytes.extend_from_slice(&6_u32.to_be_bytes());
        bytes.extend_from_slice(&1_u16.to_be_bytes());
        bytes.extend_from_slice(&(tracks.len() as u16).to_be_bytes());
        bytes.extend_from_slice(&MIDI_TICKS_PER_QUARTER_V1.to_be_bytes());

        for track in tracks {
            let track_bytes = encode_track(track);
            bytes.extend_from_slice(b"MTrk");
            bytes.extend_from_slice(&(track_bytes.len() as u32).to_be_bytes());
            bytes.extend_from_slice(&track_bytes);
        }
        bytes
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use symthaea_music_theory::harmony::Key;
        use symthaea_music_theory::pitch::{Pitch, PitchClass};
        use symthaea_music_theory::score::{
            Emphasis, PartId, Score, ScoreNote, VoiceRole,
        };
        use symthaea_music_theory::orchestration::{
            InstrumentKindV1, SoundingRangeV1,
        };
        use symthaea_music_theory::orchestration_timeline::{
            HandoffKindV1, OrchestrationSegmentV1, SegmentHandoffV1,
            ORCHESTRATION_TIMELINE_SCHEMA_V1,
        };

        fn score_digest(score: &Score) -> String {
            let bytes = serde_json::to_vec(score).expect("test score serializes");
            sha256_hex(&bytes)
        }

        fn note(part: u16, midi: u8, onset: Duration, duration: Duration) -> ScoreNote {
            ScoreNote {
                part: PartId(part),
                pitch: Pitch::from_midi(midi),
                onset,
                duration,
                velocity: 0.72,
                role: VoiceRole::Melody,
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            }
        }

        fn cap(id: &str, kind: InstrumentKindV1, low: u8, high: u8, polyphony: u8) -> InstrumentCapabilityV1 {
            InstrumentCapabilityV1 {
                id: id.into(),
                kind,
                sounding_range: SoundingRangeV1 {
                    lowest_midi: low,
                    highest_midi: high,
                },
                max_polyphony: polyphony,
            }
        }

        fn segment(
            id: &str,
            part_id: u16,
            instrument_id: &str,
            start_num: i64,
            start_den: i64,
            end_num: i64,
            end_den: i64,
            handoff: Option<SegmentHandoffV1>,
        ) -> OrchestrationSegmentV1 {
            OrchestrationSegmentV1 {
                id: id.into(),
                part_id,
                instrument_id: instrument_id.into(),
                start: RationalBeatV1 {
                    numerator: start_num,
                    denominator: start_den,
                },
                end: RationalBeatV1 {
                    numerator: end_num,
                    denominator: end_den,
                },
                musical_function: "authored_test_part".into(),
                handoff,
            }
        }

        fn score(notes: Vec<ScoreNote>) -> Score {
            let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
            for item in notes {
                score.push(item);
            }
            score
        }

        fn binding(id: &str, program: u8) -> MidiProgramBindingV1 {
            MidiProgramBindingV1 {
                instrument_id: id.into(),
                program,
                mapping_reference: format!("test-map:{id}:{program}"),
            }
        }

        fn plan(score_sha256: &str, segments: Vec<OrchestrationSegmentV1>) -> OrchestrationTimelineV1 {
            OrchestrationTimelineV1 {
                schema: ORCHESTRATION_TIMELINE_SCHEMA_V1.into(),
                score_digest_schema: ORCHESTRATION_SCORE_DIGEST_SCHEMA_V1.into(),
                score_sha256: score_sha256.into(),
                segments,
            }
        }

        #[test]
        fn changed_score_is_rejected_when_caller_reuses_a_stale_digest() {
            let source = score(vec![note(1, 60, Duration::zero(), Duration::quarter())]);
            let digest = score_digest(&source);
            let timeline = plan(
                &digest,
                vec![segment("p1", 1, "cello-1", 0, 1, 2, 1, None)],
            );
            let instruments = vec![cap("cello-1", InstrumentKindV1::Cello, 36, 84, 1)];
            let bindings = vec![binding("cello-1", 42)];
            let mut changed_score = source.clone();
            changed_score.notes[0].velocity = 0.93;

            let result = encode_score_midi_part_aware_v1(
                &changed_score,
                &timeline,
                &instruments,
                &bindings,
                &digest,
            );
            assert!(matches!(
                result,
                Err(PartAwareMidiExportErrorV1::ScoreDigestMismatch { .. })
            ));
        }

        #[test]
        fn different_instruments_cannot_share_a_quantized_handoff_tick() {
            let source = score(vec![note(
                7,
                60,
                Duration::new(1, 10_000),
                Duration::new(1, 2),
            )]);
            let digest = score_digest(&source);
            // These adjacent rational boundaries differ but both quantize to tick 0.
            let timeline = plan(
                &digest,
                vec![
                    segment("opening", 7, "cello-1", 0, 1, 1, 10_000, None),
                    segment(
                        "response",
                        7,
                        "flute-1",
                        1,
                        10_000,
                        2,
                        1,
                        Some(SegmentHandoffV1 {
                            from_segment_id: "opening".into(),
                            kind: HandoffKindV1::Reattack,
                        }),
                    ),
                ],
            );
            let instruments = vec![
                cap("cello-1", InstrumentKindV1::Cello, 36, 84, 1),
                cap("flute-1", InstrumentKindV1::Flute, 60, 96, 1),
            ];
            let bindings = vec![binding("cello-1", 42), binding("flute-1", 42)];

            let result = encode_score_midi_part_aware_v1(
                &source,
                &timeline,
                &instruments,
                &bindings,
                &digest,
            );
            assert!(matches!(
                result,
                Err(PartAwareMidiExportErrorV1::ProgramHandoffQuantizationCollision { .. })
            ));
        }

        #[test]
        fn independent_parts_with_the_same_voice_role_get_separate_midi_tracks() {
            let source = score(vec![
                note(1, 60, Duration::zero(), Duration::quarter()),
                note(2, 64, Duration::zero(), Duration::quarter()),
            ]);
            let digest = score_digest(&source);
            let timeline = plan(
                &digest,
                vec![
                    segment("part-1", 1, "cello-1", 0, 1, 2, 1, None),
                    segment("part-2", 2, "horn-1", 0, 1, 2, 1, None),
                ],
            );
            let instruments = vec![
                cap("cello-1", InstrumentKindV1::Cello, 36, 84, 1),
                cap("horn-1", InstrumentKindV1::FrenchHorn, 48, 91, 1),
            ];
            let bindings = vec![binding("cello-1", 42), binding("horn-1", 60)];

            let artifact = encode_score_midi_part_aware_v1(
                &source,
                &timeline,
                &instruments,
                &bindings,
                &digest,
            )
            .expect("valid part-aware score should encode");

            assert_eq!(&artifact.midi_bytes[0..4], b"MThd");
            assert_eq!(u16::from_be_bytes([artifact.midi_bytes[10], artifact.midi_bytes[11]]), 3);
            assert_eq!(artifact.receipt.part_count, 2);
            assert_eq!(artifact.receipt.track_count, 3);
            assert_eq!(artifact.receipt.event_map.len(), 2);
            assert_ne!(
                artifact.receipt.event_map[0].track_index,
                artifact.receipt.event_map[1].track_index
            );
            assert_ne!(
                artifact.receipt.event_map[0].channel,
                artifact.receipt.event_map[1].channel
            );
            assert!(artifact
                .receipt
                .event_map
                .iter()
                .all(|mapping| mapping.score_sha256 == digest));
        }

        #[test]
        fn instrument_handoff_keeps_part_channel_and_changes_program_at_boundary() {
            let source = score(vec![
                note(7, 60, Duration::zero(), Duration::quarter()),
                note(7, 62, Duration::new(2, 1), Duration::quarter()),
            ]);
            let digest = score_digest(&source);
            let timeline = plan(
                &digest,
                vec![
                    segment("opening", 7, "cello-1", 0, 1, 2, 1, None),
                    segment(
                        "response",
                        7,
                        "flute-1",
                        2,
                        1,
                        4,
                        1,
                        Some(SegmentHandoffV1 {
                            from_segment_id: "opening".into(),
                            kind: HandoffKindV1::Reattack,
                        }),
                    ),
                ],
            );
            let instruments = vec![
                cap("cello-1", InstrumentKindV1::Cello, 36, 84, 1),
                cap("flute-1", InstrumentKindV1::Flute, 60, 96, 1),
            ];
            let bindings = vec![binding("cello-1", 42), binding("flute-1", 73)];

            let artifact = encode_score_midi_part_aware_v1(
                &source,
                &timeline,
                &instruments,
                &bindings,
                &digest,
            )
            .expect("valid handoff should encode");

            assert_eq!(artifact.receipt.event_map[0].part_id, artifact.receipt.event_map[1].part_id);
            assert_eq!(artifact.receipt.event_map[0].channel, artifact.receipt.event_map[1].channel);
            assert_ne!(artifact.receipt.event_map[0].instrument_id, artifact.receipt.event_map[1].instrument_id);
            assert_eq!(artifact.receipt.event_map[1].onset_tick, 2 * u64::from(MIDI_TICKS_PER_QUARTER_V1));
            assert_eq!(artifact.receipt.tracks[0].instrument_ids.len(), 2);
            assert_eq!(artifact.receipt.program_changes.len(), 2);
            assert_eq!(artifact.receipt.program_changes[0].program, 42);
            assert_eq!(artifact.receipt.program_changes[1].program, 73);
            assert_eq!(artifact.receipt.program_changes[1].segment_id, "response");
        }

        #[test]
        fn missing_program_binding_is_rejected() {
            let source = score(vec![note(1, 60, Duration::zero(), Duration::quarter())]);
            let digest = score_digest(&source);
            let timeline = plan(
                &digest,
                vec![segment("p1", 1, "cello-1", 0, 1, 2, 1, None)],
            );
            let instruments = vec![cap("cello-1", InstrumentKindV1::Cello, 36, 84, 1)];

            let result = encode_score_midi_part_aware_v1(
                &source,
                &timeline,
                &instruments,
                &[],
                &digest,
            );
            assert!(matches!(
                result,
                Err(PartAwareMidiExportErrorV1::MissingProgramBinding { .. })
            ));
        }

        #[test]
        fn missing_part_identity_is_rejected_instead_of_grouped_by_role() {
            let mut source = score(vec![note(1, 60, Duration::zero(), Duration::quarter())]);
            source.notes[0].part = PartId::UNASSIGNED;
            let digest = score_digest(&source);
            let timeline = plan(
                &digest,
                vec![segment("p1", 1, "cello-1", 0, 1, 2, 1, None)],
            );
            let instruments = vec![cap("cello-1", InstrumentKindV1::Cello, 36, 84, 1)];
            let bindings = vec![binding("cello-1", 42)];

            let result = encode_score_midi_part_aware_v1(
                &source,
                &timeline,
                &instruments,
                &bindings,
                &digest,
            );
            assert!(matches!(
                result,
                Err(PartAwareMidiExportErrorV1::TimelineInvalid { .. })
            ));
        }

        #[test]
        fn same_pitch_overlap_is_rejected_instead_of_silently_clipped() {
            let source = score(vec![
                note(1, 60, Duration::zero(), Duration::new(2, 1)),
                note(1, 60, Duration::quarter(), Duration::new(2, 1)),
            ]);
            let digest = score_digest(&source);
            let timeline = plan(
                &digest,
                vec![segment("p1", 1, "cello-1", 0, 1, 4, 1, None)],
            );
            let instruments = vec![cap("cello-1", InstrumentKindV1::Cello, 36, 84, 2)];
            let bindings = vec![binding("cello-1", 42)];

            let result = encode_score_midi_part_aware_v1(
                &source,
                &timeline,
                &instruments,
                &bindings,
                &digest,
            );
            assert!(matches!(
                result,
                Err(PartAwareMidiExportErrorV1::OverlappingSamePitchNotes { .. })
            ));
        }

        #[test]
        fn rational_tick_quantization_uses_nearest_ties_to_even() {
            assert_eq!(rational_to_tick(1, 1_920), Some(0));
            assert_eq!(rational_to_tick(3, 1_920), Some(2));
            assert_eq!(rational_to_tick(5, 1_920), Some(2));
            assert_eq!(rational_to_tick(7, 1_920), Some(4));
        }

        #[test]
        fn source_overlap_is_rejected_even_when_tick_quantization_hides_it() {
            let source = score(vec![
                note(1, 60, Duration::zero(), Duration::new(1, 2)),
                note(
                    1,
                    60,
                    Duration::new(959, 1_920),
                    Duration::new(1, 2),
                ),
            ]);
            let digest = score_digest(&source);
            let timeline = plan(
                &digest,
                vec![segment("p1", 1, "cello-1", 0, 1, 2, 1, None)],
            );
            let instruments = vec![cap("cello-1", InstrumentKindV1::Cello, 36, 84, 2)];
            let bindings = vec![binding("cello-1", 42)];

            // The second onset is 479.5 MIDI ticks and rounds to tick 480,
            // which is also the first note's offset. The source events still
            // overlap by 1/1920 beat and must not be silently normalized.
            assert_eq!(
                rational_to_tick(959, 1_920),
                Some(480)
            );
            let result = encode_score_midi_part_aware_v1(
                &source,
                &timeline,
                &instruments,
                &bindings,
                &digest,
            );
            assert!(matches!(
                result,
                Err(PartAwareMidiExportErrorV1::OverlappingSamePitchNotes { .. })
            ));
        }

        #[test]
        fn sub_tick_duration_is_rejected_instead_of_dropped_or_stretched() {
            let source = score(vec![note(
                1,
                60,
                Duration::zero(),
                Duration::new(1, 10_000),
            )]);
            let digest = score_digest(&source);
            let timeline = plan(
                &digest,
                vec![segment("p1", 1, "cello-1", 0, 1, 1, 1, None)],
            );
            let instruments = vec![cap("cello-1", InstrumentKindV1::Cello, 36, 84, 1)];
            let bindings = vec![binding("cello-1", 42)];

            let result = encode_score_midi_part_aware_v1(
                &source,
                &timeline,
                &instruments,
                &bindings,
                &digest,
            );
            assert!(matches!(
                result,
                Err(PartAwareMidiExportErrorV1::UnrepresentableNoteDuration { .. })
            ));
        }

        #[test]
        fn midi_delta_beyond_standard_vlq_limit_is_rejected() {
            let source = score(vec![note(
                1,
                60,
                Duration::new(0x1000_0000, 1),
                Duration::quarter(),
            )]);
            let digest = score_digest(&source);
            let timeline = plan(
                &digest,
                vec![segment("p1", 1, "cello-1", 0, 1, 1_000_000_000, 1, None)],
            );
            let instruments = vec![cap("cello-1", InstrumentKindV1::Cello, 36, 84, 1)];
            let bindings = vec![binding("cello-1", 42)];

            let result = encode_score_midi_part_aware_v1(
                &source,
                &timeline,
                &instruments,
                &bindings,
                &digest,
            );
            assert!(matches!(
                result,
                Err(PartAwareMidiExportErrorV1::MidiDeltaExceedsStandardMaximum { .. })
            ));
        }

        #[test]
        fn event_mapping_and_midi_hash_are_deterministic_for_frozen_inputs() {
            let source = score(vec![
                note(1, 60, Duration::zero(), Duration::quarter()),
                note(2, 67, Duration::zero(), Duration::quarter()),
            ]);
            let digest = score_digest(&source);
            let timeline = plan(
                &digest,
                vec![
                    segment("p1", 1, "cello-1", 0, 1, 2, 1, None),
                    segment("p2", 2, "horn-1", 0, 1, 2, 1, None),
                ],
            );
            let instruments = vec![
                cap("cello-1", InstrumentKindV1::Cello, 36, 84, 1),
                cap("horn-1", InstrumentKindV1::FrenchHorn, 48, 91, 1),
            ];
            let bindings = vec![binding("cello-1", 42), binding("horn-1", 60)];

            let first = encode_score_midi_part_aware_v1(
                &source, &timeline, &instruments, &bindings, &digest,
            )
            .expect("first encoding");
            let second = encode_score_midi_part_aware_v1(
                &source, &timeline, &instruments, &bindings, &digest,
            )
            .expect("second encoding");
            assert_eq!(first.midi_bytes, second.midi_bytes);
            assert_eq!(first.receipt, second.receipt);
            assert_eq!(first.receipt.midi_sha256, sha256_hex(&first.midi_bytes));
        }
    }
}
