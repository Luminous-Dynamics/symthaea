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
    Duration, Emphasis, Key, Pitch, PitchClass, Score, ScoreNote, VoiceRole, score::PartId,
};

/// Keep parser memory and render work bounded for syntactically valid but
/// adversarial symbolic files.
pub const MAX_SYMBOLIC_IMPORT_BYTES: usize = 12 * 1024 * 1024;
pub const MAX_IMPORTED_NOTES: usize = 100_000;
pub const MAX_IMPORTED_MIDI_TRACKS: usize = 256;
pub const MAX_IMPORTED_MIDI_EVENTS: usize = 500_000;
pub const MAX_IMPORTED_SCORE_BEATS: i64 = 10_000;
/// Maximum musical content time synthesized for an import audition. The full
/// score remains stored and analyzed; only `audition.wav` uses this clip.
pub const MAX_AUDITION_CONTENT_SECONDS: f64 = 30.0;
const MAX_RECONSTRUCTED_SECTIONS: usize = 1_000;

#[derive(Clone, Debug)]
struct RawNote {
    track: usize,
    pitch: u8,
    onset: u64,
    duration: u64,
    velocity: u8,
}

pub fn parse_symbolic(bytes: &[u8], format: SymbolicImportFormat) -> Result<Score, String> {
    if bytes.len() > MAX_SYMBOLIC_IMPORT_BYTES {
        return Err(format!(
            "symbolic import exceeds the {MAX_SYMBOLIC_IMPORT_BYTES}-byte parser limit"
        ));
    }
    match format {
        SymbolicImportFormat::Midi => parse_midi(bytes),
        SymbolicImportFormat::MusicXml => parse_musicxml(bytes),
        SymbolicImportFormat::MuseScore => {
            let score: Score = serde_json::from_slice(bytes)
                .map_err(|error| format!("Muse score parse error: {error}"))?;
            validate_imported_score(&score)?;
            Ok(score)
        }
    }
}

pub fn parse_midi(bytes: &[u8]) -> Result<Score, String> {
    if bytes.len() > MAX_SYMBOLIC_IMPORT_BYTES {
        return Err(format!(
            "MIDI import exceeds the {MAX_SYMBOLIC_IMPORT_BYTES}-byte parser limit"
        ));
    }
    let smf = Smf::parse(bytes).map_err(|error| format!("MIDI parse error: {error}"))?;
    if smf.tracks.len() > MAX_IMPORTED_MIDI_TRACKS {
        return Err(format!(
            "MIDI import exceeds the {MAX_IMPORTED_MIDI_TRACKS}-track limit"
        ));
    }
    let event_count = smf
        .tracks
        .iter()
        .try_fold(0_usize, |count, track| count.checked_add(track.len()))
        .ok_or_else(|| "MIDI event count overflow".to_string())?;
    if event_count > MAX_IMPORTED_MIDI_EVENTS {
        return Err(format!(
            "MIDI import exceeds the {MAX_IMPORTED_MIDI_EVENTS}-event limit"
        ));
    }
    let ticks_per_beat = match smf.header.timing {
        midly::Timing::Metrical(value) => u64::from(value.as_int()),
        midly::Timing::Timecode(_, _) => {
            return Err("SMPTE-time MIDI is not supported in the first symbolic importer".into());
        }
    };
    if ticks_per_beat == 0 {
        return Err("MIDI metrical timing must have a non-zero ticks-per-beat value".into());
    }
    let mut tempo_bpm = 120.0_f32;
    let mut meter = 4_u8;
    let mut fifths = 0_i8;
    let mut minor = false;
    let mut notes = Vec::new();

    for (track_index, track) in smf.tracks.iter().enumerate() {
        let mut tick = 0_u64;
        let mut pending: HashMap<(u8, u8), (u64, u8)> = HashMap::new();
        for event in track {
            tick = tick
                .checked_add(u64::from(event.delta.as_int()))
                .ok_or_else(|| "MIDI absolute tick position overflowed".to_string())?;
            match event.kind {
                TrackEventKind::Meta(MetaMessage::Tempo(value)) => {
                    if value.as_int() == 0 {
                        return Err("MIDI tempo events must be non-zero".into());
                    }
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
                            if let Some((onset, velocity)) =
                                pending.remove(&(channel, key.as_int()))
                            {
                                push_raw_note(
                                    &mut notes,
                                    RawNote {
                                        track: track_index,
                                        pitch: key.as_int(),
                                        onset,
                                        duration: tick.saturating_sub(onset).max(1),
                                        velocity,
                                    },
                                )?;
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        for ((_, pitch), (onset, velocity)) in pending {
            push_raw_note(
                &mut notes,
                RawNote {
                    track: track_index,
                    pitch,
                    onset,
                    duration: tick.saturating_sub(onset).max(1),
                    velocity,
                },
            )?;
        }
    }
    if notes.is_empty() {
        return Err("the MIDI file contains no pitched note events".into());
    }

    // Unclosed notes were flushed from HashMaps above; canonical ordering
    // makes serialized import identity reproducible between process runs.
    notes.sort_by_key(|note| (note.onset, note.track, note.pitch, note.duration, note.velocity));
    let roles = roles_by_track(&notes);
    let tonic = PitchClass::new(i32::from(fifths) * 7 + if minor { 9 } else { 0 });
    let key = if minor {
        Key::minor(tonic)
    } else {
        Key::major(tonic)
    };
    let mut score = Score::new(key, tempo_bpm.clamp(20.0, 320.0), meter.clamp(1, 16));
    for note in notes {
        let onset = i64::try_from(note.onset)
            .map_err(|_| "MIDI note onset is not representable".to_string())?;
        let duration = i64::try_from(note.duration)
            .map_err(|_| "MIDI note duration is not representable".to_string())?;
        score
            .try_push(ScoreNote {
                part: u16::try_from(note.track)
                    .map(PartId)
                    .unwrap_or(PartId::UNASSIGNED),
                pitch: Pitch::from_midi(note.pitch),
                onset: Duration::new(onset, ticks_per_beat as i64),
                duration: Duration::new(duration, ticks_per_beat as i64),
                velocity: (note.velocity as f32 / 127.0).clamp(0.05, 1.0),
                role: roles
                    .get(&note.track)
                    .copied()
                    .unwrap_or(VoiceRole::Harmony),
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            })
            .map_err(|_| "MIDI note end is not exactly representable".to_string())?;
    }
    validate_imported_score(&score)?;
    Ok(score)
}

fn push_raw_note(notes: &mut Vec<RawNote>, note: RawNote) -> Result<(), String> {
    if notes.len() >= MAX_IMPORTED_NOTES {
        return Err(format!(
            "symbolic import exceeds the {MAX_IMPORTED_NOTES}-note limit"
        ));
    }
    notes.push(note);
    Ok(())
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
    if bytes.len() > MAX_SYMBOLIC_IMPORT_BYTES {
        return Err(format!(
            "MusicXML import exceeds the {MAX_SYMBOLIC_IMPORT_BYTES}-byte parser limit"
        ));
    }
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
    if parts.len() > MAX_IMPORTED_MIDI_TRACKS {
        return Err(format!(
            "MusicXML import exceeds the {MAX_IMPORTED_MIDI_TRACKS}-part limit"
        ));
    }
    let xml_note_count = document
        .descendants()
        .filter(|node| node.has_tag_name("note"))
        .count();
    if xml_note_count > MAX_IMPORTED_NOTES {
        return Err(format!(
            "MusicXML import exceeds the {MAX_IMPORTED_NOTES}-note limit"
        ));
    }
    let mut fifths = 0_i32;
    let mut minor = false;
    let mut meter = 4_u8;
    let mut tempo = 120.0_f32;
    let mut raw = Vec::<(usize, u8, Duration, Duration)>::new();

    for (part_index, part) in parts.iter().enumerate() {
        // Divisions are part-local and can change during a score. Keep the
        // cursor in exact beats so earlier notes never inherit later divisions.
        let mut divisions = 1_i64;
        let mut cursor = Duration::zero();
        let mut previous_onset = Duration::zero();
        for child in part.descendants().filter(|node| node.is_element()) {
            if child.has_tag_name("divisions") {
                let value = node_i64(child).unwrap_or(divisions);
                if value <= 0 {
                    return Err("MusicXML divisions must be positive".into());
                }
                divisions = value;
            } else if child.has_tag_name("fifths") {
                fifths = node_i64(child)
                    .unwrap_or(i64::from(fifths))
                    .clamp(-7, 7) as i32;
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
                if amount < 0 {
                    return Err("MusicXML backup duration cannot be negative".into());
                }
                let amount = Duration::new(amount, divisions);
                cursor = cursor
                    .checked_sub(amount)
                    .ok_or_else(|| "MusicXML backup timing is not exactly representable".to_string())?;
                if cursor.num() < 0 {
                    return Err("MusicXML backup moves before the start of a part".into());
                }
            } else if child.has_tag_name("forward") {
                let amount = child
                    .children()
                    .find(|node| node.has_tag_name("duration"))
                    .and_then(node_i64)
                    .unwrap_or(0);
                if amount < 0 {
                    return Err("MusicXML forward duration cannot be negative".into());
                }
                let amount = Duration::new(amount, divisions);
                cursor = cursor
                    .checked_add(amount)
                    .ok_or_else(|| "MusicXML forward timing overflowed".to_string())?;
            } else if child.has_tag_name("note") {
                let duration = child
                    .children()
                    .find(|node| node.has_tag_name("duration"))
                    .and_then(node_i64)
                    .unwrap_or(divisions);
                if duration < 0 {
                    return Err("MusicXML note duration cannot be negative".into());
                }
                let duration = Duration::new(duration.max(1), divisions);
                let chord = child.children().any(|node| node.has_tag_name("chord"));
                let rest = child.children().any(|node| node.has_tag_name("rest"));
                let onset = if chord { previous_onset } else { cursor };
                if !rest && let Some(midi) = musicxml_pitch(child) {
                    if raw.len() >= MAX_IMPORTED_NOTES {
                        return Err(format!(
                            "MusicXML import exceeds the {MAX_IMPORTED_NOTES}-note limit"
                        ));
                    }
                    raw.push((part_index, midi, onset, duration));
                }
                previous_onset = onset;
                if !chord {
                    cursor = cursor
                        .checked_add(duration)
                        .ok_or_else(|| "MusicXML note timing overflowed".to_string())?;
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
        score
            .try_push(ScoreNote {
                part: u16::try_from(part)
                    .map(PartId)
                    .unwrap_or(PartId::UNASSIGNED),
                pitch: Pitch::from_midi(midi),
                onset,
                duration,
                velocity: 0.72,
                role,
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            })
            .map_err(|_| "MusicXML note end is not exactly representable".to_string())?;
    }
    validate_imported_score(&score)?;
    Ok(score)
}

fn validate_imported_score(score: &Score) -> Result<(), String> {
    if score.notes.is_empty() {
        return Err("symbolic import contains no pitched notes".into());
    }
    if score.notes.len() > MAX_IMPORTED_NOTES {
        return Err(format!(
            "symbolic import exceeds the {MAX_IMPORTED_NOTES}-note limit"
        ));
    }
    if !score.tempo_bpm.is_finite() || !(20.0..=320.0).contains(&score.tempo_bpm) {
        return Err("symbolic score tempo must be finite and between 20 and 320 BPM".into());
    }
    if score.meter == 0 {
        return Err("symbolic score meter must be positive".into());
    }
    if score.total_beats.den() <= 0 || score.total_beats.num() <= 0 {
        return Err("symbolic score duration must be positive".into());
    }
    if i128::from(score.total_beats.num())
        > i128::from(MAX_IMPORTED_SCORE_BEATS) * i128::from(score.total_beats.den())
    {
        return Err(format!(
            "symbolic score exceeds the {MAX_IMPORTED_SCORE_BEATS}-beat import limit"
        ));
    }
    let duration_seconds = score.seconds();
    if !duration_seconds.is_finite() || duration_seconds <= 0.0 {
        return Err("symbolic score duration must be finite and positive".into());
    }

    for (index, note) in score.notes.iter().enumerate() {
        if note.onset.den() <= 0
            || note.duration.den() <= 0
            || note.onset.num() < 0
            || note.duration.num() <= 0
            || !note.velocity.is_finite()
            || !(0.0..=1.0).contains(&note.velocity)
            || !note.section_intensity.is_finite()
            || !(0.0..=1.0).contains(&note.section_intensity)
        {
            return Err(format!(
                "symbolic score note {index} has invalid timing or dynamics"
            ));
        }
        let end = note.onset.checked_add(note.duration).ok_or_else(|| {
            format!("symbolic score note {index} has an unrepresentable exact end")
        })?;
        if end
            .checked_cmp(score.total_beats)
            .is_none_or(|ordering| ordering == std::cmp::Ordering::Greater)
        {
            return Err(format!(
                "symbolic score note {index} ends after the declared score duration"
            ));
        }
    }
    Ok(())
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
    let midi = octave
        .checked_add(1)?
        .checked_mul(12)?
        .checked_add(base)?
        .checked_add(alter)?;
    Some(midi.clamp(0, 127) as u8)
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
    while start < total && sections.len() < MAX_RECONSTRUCTED_SECTIONS {
        let end = (start + section_beats).min(total);
        // Floating-point resolution may prevent progress for malformed direct
        // callers even though the public import parser rejects such scores.
        if end <= start {
            break;
        }
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
    let mut unresolved_interpretations = vec![
        "Confirm section boundaries".into(),
        "Confirm voice and instrument roles".into(),
        "Confirm reconstructed motif identity".into(),
    ];
    if start < total {
        unresolved_interpretations.push(format!(
            "Section reconstruction stopped at its {MAX_RECONSTRUCTED_SECTIONS}-region safety cap; the remaining duration is not segmented"
        ));
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
        unresolved_interpretations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_musicxml() -> &'static [u8] {
        br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions><key><fifths>0</fifths></key><time><beats>4</beats></time></attributes>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
            <note><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#
    }

    fn json_score_with_one_note() -> Score {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        score.notes.push(ScoreNote {
            part: PartId(0),
            pitch: Pitch::from_midi(60),
            onset: Duration::zero(),
            duration: Duration::quarter(),
            velocity: 0.7,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        });
        score.total_beats = Duration::quarter();
        score
    }

    #[test]
    fn parses_minimal_musicxml_without_forcing_a_territory() {
        let score = parse_musicxml(minimal_musicxml()).unwrap();
        assert_eq!(score.notes.len(), 2);
        let analysis = analyze(&score);
        assert!(analysis.source_native);
        assert!(analysis.inferred_territory.is_none());
    }

    #[test]
    fn musicxml_divisions_are_scoped_to_each_part() {
        let xml = br#"<score-partwise>
            <part id="P1"><measure number="1">
                <attributes><divisions>1</divisions></attributes>
                <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
            </measure></part>
            <part id="P2"><measure number="1">
                <attributes><divisions>2</divisions></attributes>
                <note><pitch><step>E</step><octave>4</octave></pitch><duration>2</duration></note>
            </measure></part>
        </score-partwise>"#;
        let score = parse_musicxml(xml).unwrap();
        assert_eq!(score.notes.len(), 2);
        assert_eq!(score.notes[0].duration, Duration::new(1, 1));
        assert_eq!(score.notes[1].duration, Duration::new(1, 1));
    }

    #[test]
    fn musicxml_division_changes_preserve_the_absolute_cursor() {
        let xml = br#"<score-partwise><part id="P1">
            <measure number="1">
                <attributes><divisions>1</divisions></attributes>
                <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
            </measure>
            <measure number="2">
                <attributes><divisions>2</divisions></attributes>
                <note><pitch><step>D</step><octave>4</octave></pitch><duration>2</duration></note>
            </measure>
        </part></score-partwise>"#;
        let score = parse_musicxml(xml).unwrap();
        assert_eq!(score.notes.len(), 2);
        assert_eq!(score.notes[1].onset, Duration::new(1, 1));
        assert_eq!(score.notes[1].duration, Duration::new(1, 1));
        assert_eq!(score.total_beats, Duration::new(2, 1));
    }

    #[test]
    fn oversized_musescore_duration_is_rejected_before_analysis_or_render() {
        let mut score = json_score_with_one_note();
        score.total_beats = Duration::new(MAX_IMPORTED_SCORE_BEATS + 1, 1);
        let bytes = serde_json::to_vec(&score).unwrap();
        let error = parse_symbolic(&bytes, SymbolicImportFormat::MuseScore).unwrap_err();
        assert!(error.contains("beat import limit"), "{error}");
    }

    #[test]
    fn unrepresentable_exact_note_end_is_rejected_even_if_float_timing_fits() {
        let mut score = json_score_with_one_note();
        let mut malformed = score.notes[0];
        malformed.onset = Duration::new(1, i64::MAX);
        malformed.duration = Duration::new(1, i64::MAX - 2);
        score.notes.push(malformed);
        let bytes = serde_json::to_vec(&score).unwrap();
        let error = parse_symbolic(&bytes, SymbolicImportFormat::MuseScore).unwrap_err();
        assert!(error.contains("unrepresentable exact end"), "{error}");
    }

    #[test]
    fn musicxml_cursor_overflow_is_reported_not_panicked() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>9223372036854775807</duration></note>
            <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(error.contains("timing overflowed"), "{error}");
    }

    #[test]
    fn zero_ticks_per_beat_midi_is_rejected_without_panicking() {
        let bytes = [
            b'M', b'T', b'h', b'd', 0, 0, 0, 6, 0, 0, 0, 1, 0, 0,
            b'M', b'T', b'r', b'k', 0, 0, 0, 4, 0, 0xFF, 0x2F, 0,
        ];
        let error = parse_midi(&bytes).unwrap_err();
        assert!(error.contains("non-zero ticks-per-beat"), "{error}");
    }
}
