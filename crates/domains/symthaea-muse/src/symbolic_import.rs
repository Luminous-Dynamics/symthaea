// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Private-first symbolic import for Muse Studio.
//!
//! Imported works retain a source-native interpretation. These parsers do not
//! classify a musician's work as one of Muse's territories and do not mutate
//! any shared Foundry or learning corpus.

use midly::{MetaMessage, MidiMessage, Smf, TrackEventKind};
use std::collections::{BTreeSet, HashMap};
use symthaea_muse_protocol::{
    ImportedMotifSummary, ImportedSectionSummary, ImportedWorkAnalysis, SymbolicImportFormat,
};
use symthaea_music_theory::{
    Duration, Emphasis, Key, Pitch, PitchClass, Score, ScoreNote, VoiceRole,
};

#[derive(Clone, Debug)]
struct RawNote {
    track: usize,
    pitch: u8,
    onset: u64,
    duration: u64,
    velocity: u8,
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
    let mut tempo_bpm = 120.0_f32;
    let mut meter = 4_u8;
    let mut fifths = 0_i8;
    let mut minor = false;
    let mut notes = Vec::new();

    for (track_index, track) in smf.tracks.iter().enumerate() {
        let mut tick = 0_u64;
        let mut pending: HashMap<(u8, u8), (u64, u8)> = HashMap::new();
        for event in track {
            tick = tick.saturating_add(u64::from(event.delta.as_int()));
            match event.kind {
                TrackEventKind::Meta(MetaMessage::Tempo(value)) => {
                    tempo_bpm = 60_000_000.0 / value.as_int() as f32;
                }
                TrackEventKind::Meta(MetaMessage::TimeSignature(numerator, _, _, _)) => {
                    meter = numerator.max(1);
                }
                TrackEventKind::Meta(MetaMessage::KeySignature(sf, is_minor)) => {
                    fifths = sf;
                    minor = is_minor;
                }
                TrackEventKind::Midi { channel, message } if channel.as_int() != 9 => {
                    let channel = channel.as_int();
                    match message {
                        MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
                            pending.insert((channel, key.as_int()), (tick, vel.as_int()));
                        }
                        MidiMessage::NoteOn { key, .. } | MidiMessage::NoteOff { key, .. } => {
                            if let Some((onset, velocity)) = pending.remove(&(channel, key.as_int()))
                            {
                                notes.push(RawNote {
                                    track: track_index,
                                    pitch: key.as_int(),
                                    onset,
                                    duration: tick.saturating_sub(onset).max(1),
                                    velocity,
                                });
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        for ((_, pitch), (onset, velocity)) in pending {
            notes.push(RawNote {
                track: track_index,
                pitch,
                onset,
                duration: tick.saturating_sub(onset).max(1),
                velocity,
            });
        }
    }
    if notes.is_empty() {
        return Err("the MIDI file contains no pitched note events".into());
    }

    let roles = roles_by_track(&notes);
    let tonic = PitchClass::new(i32::from(fifths) * 7 + if minor { 9 } else { 0 });
    let key = if minor { Key::minor(tonic) } else { Key::major(tonic) };
    let mut score = Score::new(key, tempo_bpm.clamp(20.0, 320.0), meter.clamp(1, 16));
    for note in notes {
        score.push(ScoreNote {
            pitch: Pitch::from_midi(note.pitch),
            onset: Duration::new(note.onset as i64, ticks_per_beat as i64),
            duration: Duration::new(note.duration as i64, ticks_per_beat as i64),
            velocity: (note.velocity as f32 / 127.0).clamp(0.05, 1.0),
            role: roles.get(&note.track).copied().unwrap_or(VoiceRole::Harmony),
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
    let mut meter = 4_u8;
    let mut tempo = 120.0_f32;
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
            } else if child.has_tag_name("beats") {
                meter = node_i64(child).unwrap_or(i64::from(meter)).clamp(1, 16) as u8;
            } else if child.has_tag_name("sound") {
                if let Some(value) = child.attribute("tempo").and_then(|v| v.parse().ok()) {
                    tempo = value;
                }
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
    let key = if minor { Key::minor(tonic) } else { Key::major(tonic) };
    let mut score = Score::new(key, tempo.clamp(20.0, 320.0), meter);
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
    let step = pitch.children().find(|node| node.has_tag_name("step"))?.text()?;
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
    let motif_pitches: Vec<u8> = melody.iter().take(motif_len).map(|n| n.pitch.midi()).collect();
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
            identity_note: "Reconstructed opening interval identity; contributor confirmation required".into(),
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
            evidence: "Provisional eight-bar segmentation; not asserted as the contributor's form".into(),
            confidence: 0.35,
        });
        start = end;
        index += 1;
    }
    let voices: BTreeSet<_> = score.notes.iter().map(|note| format!("{:?}", note.role)).collect();
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
