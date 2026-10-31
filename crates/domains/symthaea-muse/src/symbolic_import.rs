// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Private-first symbolic import for Muse Studio.
//!
//! Imported works retain a source-native interpretation. These parsers do not
//! classify a musician's work as one of Muse's territories and do not mutate
//! any shared Foundry or learning corpus.

use midly::{MetaMessage, MidiMessage, Smf, TrackEventKind};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use symthaea_muse_protocol::{
    ImportedMotifSummary, ImportedSectionSummary, ImportedWorkAnalysis, SymbolicImportFormat,
};
use symthaea_music_theory::{
    Duration, Emphasis, Key, Pitch, PitchClass, Score, ScoreNote, VoiceRole, score::PartId,
};

#[derive(Clone, Debug)]
struct RawNote {
    track: usize,
    pitch: u8,
    onset: u64,
    duration: u64,
    velocity: u8,
}

/// Convert a written time signature to the legacy Score representation.
///
/// Score::meter stores an integer number of quarter-note beats per bar. The
/// importer therefore accepts only signatures whose exact bar length is an
/// integer number of quarter notes. It preserves bar length for 6/8 (3 beats),
/// but cannot preserve its compound grouping; fractional/additive signatures
/// are rejected rather than silently reinterpreted.
fn quarter_note_meter(numerator: u8, denominator: u32, source: &str) -> Result<u8, String> {
    if numerator == 0 || denominator == 0 || !denominator.is_power_of_two() {
        return Err(format!(
            "{source} time signature has an invalid numerator/denominator: {numerator}/{denominator}"
        ));
    }
    let quarter_note_units = u32::from(numerator) * 4;
    if quarter_note_units % denominator != 0 {
        return Err(format!(
            "{source} time signature {numerator}/{denominator} has a fractional quarter-note bar length; the legacy Score meter cannot represent it"
        ));
    }
    let beats = quarter_note_units / denominator;
    u8::try_from(beats)
        .ok()
        .filter(|beats| (1..=16).contains(beats))
        .ok_or_else(|| {
            format!(
                "{source} time signature {numerator}/{denominator} is outside the supported legacy Score meter range 1..=16"
            )
        })
}

fn advance_midi_tick(current: u64, delta: u32, max_tick: u64) -> Result<u64, String> {
    current
        .checked_add(u64::from(delta))
        .filter(|value| *value <= max_tick)
        .ok_or_else(|| {
            "MIDI track event timeline exceeds the safe signed rational duration arithmetic range"
                .to_string()
        })
}

fn midi_time_signature_meter(numerator: u8, denominator_exponent: u8) -> Result<u8, String> {
    let denominator = 1_u32
        .checked_shl(u32::from(denominator_exponent))
        .ok_or_else(|| {
            format!(
                "MIDI time-signature denominator exponent {denominator_exponent} is out of range"
            )
        })?;
    quarter_note_meter(numerator, denominator, "MIDI")
}

/// The legacy Score carries one constant tempo. Preserve repeated identical
/// declarations, but do not flatten a changing tempo map to the last value.
fn musicxml_tempo_bpm(document: &roxmltree::Document<'_>) -> Result<f32, String> {
    let mut tempo = None::<f32>;
    for sound in document
        .descendants()
        .filter(|node| node.has_tag_name("sound"))
    {
        let Some(raw) = sound.attribute("tempo") else {
            continue;
        };
        let value = raw
            .parse::<f32>()
            .map_err(|_| format!("MusicXML sound tempo {raw:?} is not numeric"))?;
        if !value.is_finite() || value <= 0.0 {
            return Err(format!(
                "MusicXML sound tempo {raw:?} must be finite and positive"
            ));
        }
        if tempo.is_some_and(|previous| previous.to_bits() != value.to_bits()) {
            return Err(
                "MusicXML tempo changes are not representable by the legacy single-tempo Score"
                    .into(),
            );
        }
        tempo = Some(value);
    }
    Ok(tempo.unwrap_or(120.0))
}

fn musicxml_time_signature_meter(
    document: &roxmltree::Document<'_>,
) -> Result<u8, String> {
    let mut signatures = BTreeSet::<(u8, u8)>::new();
    for time in document.descendants().filter(|node| node.has_tag_name("time")) {
        let beats: Vec<_> = time.children().filter(|node| node.has_tag_name("beats")).collect();
        let beat_types: Vec<_> = time
            .children()
            .filter(|node| node.has_tag_name("beat-type"))
            .collect();
        if beats.len() != 1 || beat_types.len() != 1 {
            return Err(
                "MusicXML composite, additive, senza-misura, or incomplete time signatures are not representable by the legacy Score meter"
                    .into(),
            );
        }
        let numerator = beats[0]
            .text()
            .and_then(|value| value.trim().parse::<u8>().ok())
            .ok_or_else(|| {
                "MusicXML time-signature numerator must be a single integer".to_string()
            })?;
        let denominator = beat_types[0]
            .text()
            .and_then(|value| value.trim().parse::<u8>().ok())
            .ok_or_else(|| {
                "MusicXML time-signature denominator must be a single integer".to_string()
            })?;
        signatures.insert((numerator, denominator));
    }

    if signatures.len() > 1 {
        return Err(
            "MusicXML time-signature changes are not representable by the legacy Score meter"
                .into(),
        );
    }
    match signatures.into_iter().next() {
        Some((numerator, denominator)) => {
            quarter_note_meter(numerator, u32::from(denominator), "MusicXML")
        }
        None => Ok(4),
    }
}

pub fn parse_symbolic(bytes: &[u8], format: SymbolicImportFormat) -> Result<Score, String> {
    match format {
        SymbolicImportFormat::Midi => parse_midi(bytes),
        SymbolicImportFormat::MusicXml => parse_musicxml(bytes),
        SymbolicImportFormat::MuseScore => serde_json::from_slice(bytes)
            .map_err(|error| format!("Muse score parse error: {error}")),
    }
}

pub fn parse_midi(bytes: &[u8]) -> Result<Score, String> {
    let smf = Smf::parse(bytes).map_err(|error| format!("MIDI parse error: {error}"))?;
    let ticks_per_beat = match smf.header.timing {
        midly::Timing::Metrical(value) => u64::from(value.as_int()),
        midly::Timing::Timecode(_, _) => {
            return Err("SMPTE-time MIDI is not supported in the first symbolic importer".into());
        }
    };
    if ticks_per_beat == 0 {
        return Err("MIDI metrical timing declares zero ticks per beat".into());
    }
    // Score::push adds rational onset and duration by cross-multiplication.
    // Bound the complete tick timeline so that those products remain within
    // i64, not merely the raw onset/duration casts.
    let max_supported_tick = (i64::MAX as u64) / ticks_per_beat;
    let mut tempo_microseconds_per_beat = None::<u32>;
    let mut meter = 4_u8;
    let mut midi_time_signature = None::<(u8, u8)>;
    let mut fifths = 0_i8;
    let mut minor = false;
    let mut notes = Vec::new();

    for (track_index, track) in smf.tracks.iter().enumerate() {
        let mut tick = 0_u64;
        // MIDI 1.0 does not attach a unique note-instance ID to NoteOff.
        // Preserve overlapping repeated pitches and pair releases FIFO.
        let mut pending = BTreeMap::<(u8, u8), VecDeque<(u64, u8)>>::new();
        for event in track {
            tick = advance_midi_tick(tick, event.delta.as_int(), max_supported_tick)?;
            match event.kind {
                TrackEventKind::Meta(MetaMessage::Tempo(value)) => {
                    let microseconds_per_beat = value.as_int();
                    if microseconds_per_beat == 0 {
                        return Err("MIDI tempo event encodes zero microseconds per beat".into());
                    }
                    if tempo_microseconds_per_beat
                        .is_some_and(|previous| previous != microseconds_per_beat)
                    {
                        return Err(
                            "MIDI tempo changes are not representable by the legacy single-tempo Score"
                                .into(),
                        );
                    }
                    tempo_microseconds_per_beat = Some(microseconds_per_beat);
                }
                TrackEventKind::Meta(MetaMessage::TimeSignature(
                    numerator,
                    denominator_exponent,
                    _,
                    _,
                )) => {
                    let signature = (numerator, denominator_exponent);
                    if midi_time_signature.is_some_and(|previous| previous != signature) {
                        return Err(
                            "MIDI time-signature changes are not representable by the legacy Score meter"
                                .into(),
                        );
                    }
                    midi_time_signature = Some(signature);
                    meter = midi_time_signature_meter(numerator, denominator_exponent)?;
                }
                TrackEventKind::Meta(MetaMessage::KeySignature(sf, is_minor)) => {
                    fifths = sf;
                    minor = is_minor;
                }
                TrackEventKind::Midi { channel, message } if channel.as_int() != 9 => {
                    let channel = channel.as_int();
                    match message {
                        MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
                            pending
                                .entry((channel, key.as_int()))
                                .or_default()
                                .push_back((tick, vel.as_int()));
                        }
                        MidiMessage::NoteOn { key, .. } | MidiMessage::NoteOff { key, .. } => {
                            let pending_key = (channel, key.as_int());
                            let completed = pending
                                .get_mut(&pending_key)
                                .and_then(VecDeque::pop_front);
                            if let Some((onset, velocity)) = completed {
                                notes.push(RawNote {
                                    track: track_index,
                                    pitch: key.as_int(),
                                    onset,
                                    duration: tick.saturating_sub(onset).max(1),
                                    velocity,
                                });
                                if pending
                                    .get(&pending_key)
                                    .is_some_and(VecDeque::is_empty)
                                {
                                    pending.remove(&pending_key);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        for ((_, pitch), open_notes) in pending {
            for (onset, velocity) in open_notes {
                notes.push(RawNote {
                    track: track_index,
                    pitch,
                    onset,
                    duration: tick.saturating_sub(onset).max(1),
                    velocity,
                });
            }
        }
    }
    // NoteOff order is not necessarily onset order (especially with repeated
    // pitches). Return a stable chronological score independent of map order.
    notes.sort_by_key(|note| (note.onset, note.track, note.pitch, note.duration, note.velocity));
    if notes.is_empty() {
        return Err("the MIDI file contains no pitched note events".into());
    }

    let roles = roles_by_track(&notes);
    let tonic = PitchClass::new(i32::from(fifths) * 7 + if minor { 9 } else { 0 });
    let key = if minor {
        Key::minor(tonic)
    } else {
        Key::major(tonic)
    };
    let tempo_bpm = tempo_microseconds_per_beat
        .map(|value| 60_000_000.0 / value as f32)
        .unwrap_or(120.0);
    let ticks_per_beat_i64 = i64::try_from(ticks_per_beat)
        .map_err(|_| "MIDI ticks per beat exceed the score's signed rational range".to_string())?;
    let mut score = Score::new(key, tempo_bpm, meter);
    for note in notes {
        let onset = i64::try_from(note.onset)
            .map_err(|_| "MIDI note onset exceeds the score's signed rational range".to_string())?;
        let duration = i64::try_from(note.duration)
            .map_err(|_| "MIDI note duration exceeds the score's signed rational range".to_string())?;
        score.push(ScoreNote {
            // Real part identity, taken from the source format rather than
            // inferred: a MIDI track / MusicXML <part> IS the continuing line.
            // `RawNote::track` survives to here (the role lookup below uses
            // it), so this is the importer passing through what the file
            // already stated. Tracks beyond u16 are refused rather than
            // wrapped into a collision with a real part.
            part: u16::try_from(note.track)
                .map(PartId)
                .unwrap_or(PartId::UNASSIGNED),
            pitch: Pitch::from_midi(note.pitch),
            onset: Duration::new(onset, ticks_per_beat_i64),
            duration: Duration::new(duration, ticks_per_beat_i64),
            velocity: (note.velocity as f32 / 127.0).clamp(0.05, 1.0),
            role: roles
                .get(&note.track)
                .copied()
                .unwrap_or(VoiceRole::Harmony),
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        });
    }
    Ok(score)
}

fn roles_by_track(notes: &[RawNote]) -> HashMap<usize, VoiceRole> {
    let mut pitches: HashMap<usize, Vec<u8>> = HashMap::new();
    for note in notes {
        pitches.entry(note.track).or_default().push(note.pitch);
    }
    let mut ranked: Vec<(usize, f64)> = pitches
        .into_iter()
        .map(|(track, values)| {
            let mean = values.iter().map(|value| f64::from(*value)).sum::<f64>()
                / values.len().max(1) as f64;
            (track, mean)
        })
        .collect();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut roles = HashMap::new();
    if ranked.len() == 1 {
        roles.insert(ranked[0].0, VoiceRole::Melody);
        return roles;
    }
    for (index, (track, _)) in ranked.iter().enumerate() {
        let role = if index == 0 {
            VoiceRole::Bass
        } else if index + 1 == ranked.len() {
            VoiceRole::Melody
        } else if index + 2 == ranked.len() {
            VoiceRole::CounterMelody
        } else {
            VoiceRole::Harmony
        };
        roles.insert(*track, role);
    }
    roles
}

pub fn parse_musicxml(bytes: &[u8]) -> Result<Score, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| "MusicXML must be UTF-8 XML")?;
    let document = roxmltree::Document::parse(text)
        .map_err(|error| format!("MusicXML parse error: {error}"))?;
    let parts: Vec<_> = document
        .descendants()
        .filter(|node| node.has_tag_name("part"))
        .collect();
    if parts.is_empty() {
        return Err("MusicXML contains no score parts".into());
    }
    let mut divisions = 1_i64;
    let mut fifths = 0_i32;
    let mut minor = false;
    let meter = musicxml_time_signature_meter(&document)?;
    let tempo = musicxml_tempo_bpm(&document)?;
    let mut raw = Vec::<(usize, u8, i64, i64)>::new();

    for (part_index, part) in parts.iter().enumerate() {
        let mut cursor = 0_i64;
        let mut previous_onset = 0_i64;
        for child in part.descendants().filter(|node| node.is_element()) {
            if child.has_tag_name("divisions") {
                divisions = node_i64(child).unwrap_or(divisions).max(1);
            } else if child.has_tag_name("fifths") {
                fifths = node_i64(child).unwrap_or(i64::from(fifths)) as i32;
            } else if child.has_tag_name("mode") {
                minor = child.text().is_some_and(|value| value.trim() == "minor");
            } else if child.has_tag_name("backup") {
                let amount = child
                    .children()
                    .find(|node| node.has_tag_name("duration"))
                    .and_then(node_i64)
                    .unwrap_or(0);
                cursor = cursor.saturating_sub(amount);
            } else if child.has_tag_name("forward") {
                let amount = child
                    .children()
                    .find(|node| node.has_tag_name("duration"))
                    .and_then(node_i64)
                    .unwrap_or(0);
                cursor += amount;
            } else if child.has_tag_name("note") {
                let duration = child
                    .children()
                    .find(|node| node.has_tag_name("duration"))
                    .and_then(node_i64)
                    .unwrap_or(divisions)
                    .max(1);
                let chord = child.children().any(|node| node.has_tag_name("chord"));
                let rest = child.children().any(|node| node.has_tag_name("rest"));
                let onset = if chord { previous_onset } else { cursor };
                if !rest && let Some(midi) = musicxml_pitch(child) {
                    raw.push((part_index, midi, onset, duration));
                }
                previous_onset = onset;
                if !chord {
                    cursor += duration;
                }
            }
        }
    }
    if raw.is_empty() {
        return Err("MusicXML contains no pitched notes".into());
    }
    let tonic = PitchClass::new(fifths * 7 + if minor { 9 } else { 0 });
    let key = if minor {
        Key::minor(tonic)
    } else {
        Key::major(tonic)
    };
    let mut score = Score::new(key, tempo, meter);
    let part_count = parts.len();
    for (part, midi, onset, duration) in raw {
        let role = if part_count == 1 || part == 0 {
            VoiceRole::Melody
        } else if part + 1 == part_count {
            VoiceRole::Bass
        } else if part == 1 {
            VoiceRole::CounterMelody
        } else {
            VoiceRole::Harmony
        };
        score.push(ScoreNote {
            // Real part identity, taken from the source format rather than
            // inferred: a MIDI track / MusicXML <part> IS the continuing line.
            // Here that is the MusicXML part index already in scope. Indices
            // beyond u16 are refused rather than wrapped into a collision
            // with a real part.
            part: u16::try_from(part)
                .map(PartId)
                .unwrap_or(PartId::UNASSIGNED),
            pitch: Pitch::from_midi(midi),
            onset: Duration::new(onset, divisions),
            duration: Duration::new(duration, divisions),
            velocity: 0.72,
            role,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        });
    }
    Ok(score)
}

fn node_i64(node: roxmltree::Node<'_, '_>) -> Option<i64> {
    node.text()?.trim().parse().ok()
}

fn musicxml_pitch(note: roxmltree::Node<'_, '_>) -> Option<u8> {
    let pitch = note.children().find(|node| node.has_tag_name("pitch"))?;
    let step = pitch
        .children()
        .find(|node| node.has_tag_name("step"))?
        .text()?;
    let base = match step.trim() {
        "C" => 0,
        "D" => 2,
        "E" => 4,
        "F" => 5,
        "G" => 7,
        "A" => 9,
        "B" => 11,
        _ => return None,
    };
    let alter = pitch
        .children()
        .find(|node| node.has_tag_name("alter"))
        .and_then(node_i64)
        .unwrap_or(0);
    let octave = pitch
        .children()
        .find(|node| node.has_tag_name("octave"))
        .and_then(node_i64)?;
    Some(((octave + 1) * 12 + base + alter).clamp(0, 127) as u8)
}

pub fn analyze(score: &Score) -> ImportedWorkAnalysis {
    let melody = score.voice(VoiceRole::Melody);
    let motif_len = melody.len().clamp(0, 6);
    let motif_pitches: Vec<u8> = melody
        .iter()
        .take(motif_len)
        .map(|n| n.pitch.midi())
        .collect();
    let motif_intervals: Vec<i16> = motif_pitches
        .windows(2)
        .map(|pair| i16::from(pair[1]) - i16::from(pair[0]))
        .collect();
    let occurrences = if motif_intervals.is_empty() {
        0
    } else {
        melody
            .windows(motif_len)
            .filter(|window| {
                window
                    .windows(2)
                    .map(|pair| i16::from(pair[1].pitch.midi()) - i16::from(pair[0].pitch.midi()))
                    .eq(motif_intervals.iter().copied())
            })
            .count()
    };
    let motifs = (!motif_pitches.is_empty())
        .then(|| ImportedMotifSummary {
            occurrence_count: occurrences,
            midi_pitches: motif_pitches,
            identity_note:
                "Reconstructed opening interval identity; contributor confirmation required".into(),
            confidence: if occurrences > 1 { 0.72 } else { 0.42 },
        })
        .into_iter()
        .collect();

    let section_beats = f64::from(score.meter.max(1)) * 8.0;
    let total = score.total_beats.beats();
    let mut sections = Vec::new();
    let mut start = 0.0;
    let mut index = 1;
    while start < total {
        let end = (start + section_beats).min(total);
        sections.push(ImportedSectionSummary {
            label: format!("Reconstructed region {index}"),
            start_beat: start,
            end_beat: end,
            evidence: "Provisional eight-bar segmentation; not asserted as the contributor's form"
                .into(),
            confidence: 0.35,
        });
        start = end;
        index += 1;
    }
    let voices: BTreeSet<_> = score
        .notes
        .iter()
        .map(|note| format!("{:?}", note.role))
        .collect();
    ImportedWorkAnalysis {
        source_native: true,
        inferred_territory: None,
        tempo_bpm: score.tempo_bpm,
        meter: score.meter,
        tonic: score.key.tonic.name().into(),
        note_count: score.notes.len(),
        voice_count: voices.len(),
        duration_seconds: score.seconds(),
        motifs,
        sections,
        unresolved_interpretations: vec![
            "Confirm section boundaries".into(),
            "Confirm voice and instrument roles".into(),
            "Confirm reconstructed motif identity".into(),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_overlapping_repeated_same_pitch_midi_notes_fifo() {
        // Format 0, one track, 96 ticks/quarter:
        // C4 on@0, C4 on@48, C4 off@96 (NoteOn vel=0),
        // C4 off@144, end-of-track. NoteOff lacks a note-instance ID,
        // so the importer documents and tests FIFO pairing.
        let midi = [
            0x4d, 0x54, 0x68, 0x64, 0x00, 0x00, 0x00, 0x06,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x60,
            0x4d, 0x54, 0x72, 0x6b, 0x00, 0x00, 0x00, 0x14,
            0x00, 0x90, 0x3c, 0x5a,
            0x30, 0x90, 0x3c, 0x46,
            0x30, 0x90, 0x3c, 0x00,
            0x30, 0x80, 0x3c, 0x00,
            0x00, 0xff, 0x2f, 0x00,
        ];
        let score = parse_midi(&midi).expect("valid repeated-pitch MIDI");
        assert_eq!(score.notes.len(), 2);
        assert_eq!(score.notes[0].pitch.midi(), 60);
        assert_eq!(score.notes[1].pitch.midi(), 60);
        assert!((score.notes[0].onset.beats() - 0.0).abs() < 1e-12);
        assert!((score.notes[1].onset.beats() - 0.5).abs() < 1e-12);
        assert!((score.notes[0].duration.beats() - 1.0).abs() < 1e-12);
        assert!((score.notes[1].duration.beats() - 1.0).abs() < 1e-12);
        assert!(score.has_part_identity());

        let mut invalid_timing = midi;
        invalid_timing[12] = 0;
        invalid_timing[13] = 0;
        assert!(
            parse_midi(&invalid_timing).is_err(),
            "zero ticks per beat must be rejected rather than panicking in Duration::new"
        );
    }

    #[test]
    fn rejects_zero_microseconds_per_beat_tempo_event() {
        let midi = [
            0x4d, 0x54, 0x68, 0x64, 0x00, 0x00, 0x00, 0x06,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x60,
            0x4d, 0x54, 0x72, 0x6b, 0x00, 0x00, 0x00, 0x13,
            0x00, 0xff, 0x51, 0x03, 0x00, 0x00, 0x00,
            0x00, 0x90, 0x3c, 0x64,
            0x60, 0x80, 0x3c, 0x00,
            0x00, 0xff, 0x2f, 0x00,
        ];
        assert!(
            parse_midi(&midi).is_err(),
            "zero microseconds per beat must be rejected"
        );
    }

    #[test]
    fn rejects_midi_tick_advance_beyond_safe_rational_arithmetic_range() {
        let max_tick = (i64::MAX as u64) / 96;
        assert_eq!(advance_midi_tick(max_tick - 1, 1, max_tick).unwrap(), max_tick);
        assert!(advance_midi_tick(max_tick, 1, max_tick).is_err());
        assert!(advance_midi_tick(u64::MAX, 1, max_tick).is_err());
    }

    #[test]
    fn converts_compound_meters_to_exact_quarter_note_bar_length() {
        assert_eq!(midi_time_signature_meter(6, 3).unwrap(), 3);
        assert_eq!(quarter_note_meter(2, 2, "test").unwrap(), 4);
        assert_eq!(quarter_note_meter(3, 4, "test").unwrap(), 3);
    }

    #[test]
    fn rejects_time_signatures_the_legacy_score_cannot_represent() {
        assert!(midi_time_signature_meter(7, 3).is_err(), "7/8 is 3.5 quarter notes");
        assert!(midi_time_signature_meter(0, 2).is_err());
        assert!(quarter_note_meter(4, 3, "test").is_err());
    }

    #[test]
    fn midi_6_8_uses_three_quarter_note_beats_per_bar() {
        let midi = [
            0x4d, 0x54, 0x68, 0x64, 0x00, 0x00, 0x00, 0x06,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x60,
            0x4d, 0x54, 0x72, 0x6b, 0x00, 0x00, 0x00, 0x14,
            0x00, 0xff, 0x58, 0x04, 0x06, 0x03, 0x24, 0x08,
            0x00, 0x90, 0x3c, 0x64,
            0x60, 0x80, 0x3c, 0x00,
            0x00, 0xff, 0x2f, 0x00,
        ];
        let score = parse_midi(&midi).expect("valid 6/8 MIDI");
        assert_eq!(score.meter, 3);
    }

    #[test]
    fn midi_7_8_is_rejected_instead_of_becoming_seven_quarter_beats() {
        let midi = [
            0x4d, 0x54, 0x68, 0x64, 0x00, 0x00, 0x00, 0x06,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x60,
            0x4d, 0x54, 0x72, 0x6b, 0x00, 0x00, 0x00, 0x14,
            0x00, 0xff, 0x58, 0x04, 0x07, 0x03, 0x21, 0x08,
            0x00, 0x90, 0x3c, 0x64,
            0x60, 0x80, 0x3c, 0x00,
            0x00, 0xff, 0x2f, 0x00,
        ];
        assert!(parse_midi(&midi).is_err());
    }

    #[test]
    fn midi_constant_tempo_above_legacy_clamp_is_preserved() {
        // 150,000 microseconds per beat = 400 BPM.
        let midi = [
            0x4d, 0x54, 0x68, 0x64, 0x00, 0x00, 0x00, 0x06,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x60,
            0x4d, 0x54, 0x72, 0x6b, 0x00, 0x00, 0x00, 0x13,
            0x00, 0xff, 0x51, 0x03, 0x02, 0x49, 0xf0,
            0x00, 0x90, 0x3c, 0x64,
            0x3c, 0x80, 0x3c, 0x00,
            0x00, 0xff, 0x2f, 0x00,
        ];
        let score = parse_midi(&midi).expect("valid constant-tempo MIDI");
        assert!((score.tempo_bpm - 400.0).abs() < 1e-4);
    }

    #[test]
    fn midi_tempo_change_is_rejected_instead_of_applying_the_last_tempo_globally() {
        let midi = [
            0x4d, 0x54, 0x68, 0x64, 0x00, 0x00, 0x00, 0x06,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x60,
            0x4d, 0x54, 0x72, 0x6b, 0x00, 0x00, 0x00, 0x1a,
            0x00, 0xff, 0x51, 0x03, 0x07, 0xa1, 0x20,
            0x00, 0x90, 0x3c, 0x64,
            0x3c, 0xff, 0x51, 0x03, 0x03, 0xd0, 0x90,
            0x3c, 0x80, 0x3c, 0x00,
            0x00, 0xff, 0x2f, 0x00,
        ];
        assert!(parse_midi(&midi).is_err());
    }

    #[test]
    fn musicxml_constant_tempo_above_legacy_clamp_is_preserved() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <direction><sound tempo="400"/></direction>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let score = parse_musicxml(xml).unwrap();
        assert!((score.tempo_bpm - 400.0).abs() < 1e-4);
    }

    #[test]
    fn musicxml_tempo_change_is_rejected_instead_of_applying_the_last_tempo_globally() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <direction><sound tempo="120"/></direction>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
            <direction><sound tempo="90"/></direction>
            <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        assert!(parse_musicxml(xml).is_err());
    }

    #[test]
    fn musicxml_6_8_uses_three_quarter_note_beats_per_bar() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions><time><beats>6</beats><beat-type>8</beat-type></time></attributes>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let score = parse_musicxml(xml).unwrap();
        assert_eq!(score.meter, 3);
    }

    #[test]
    fn musicxml_7_8_is_rejected_instead_of_becoming_seven_quarter_beats() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions><time><beats>7</beats><beat-type>8</beat-type></time></attributes>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        assert!(parse_musicxml(xml).is_err());
    }

    #[test]
    fn parses_minimal_musicxml_without_forcing_a_territory() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions><key><fifths>0</fifths></key><time><beats>4</beats></time></attributes>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
            <note><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let score = parse_musicxml(xml).unwrap();
        assert_eq!(score.notes.len(), 2);
        let analysis = analyze(&score);
        assert!(analysis.source_native);
        assert!(analysis.inferred_territory.is_none());
    }
}
