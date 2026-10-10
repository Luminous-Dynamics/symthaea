// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Private-first symbolic import for Muse Studio.
//!
//! Imported works retain a source-native interpretation. These parsers do not
//! classify a musician's work as one of Muse's territories and do not mutate
//! any shared Foundry or learning corpus.

use midly::{MetaMessage, MidiMessage, Smf, TrackEventKind};
use std::collections::{BTreeSet, HashMap, VecDeque};
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
    let mut tempo_microseconds_per_quarter = None::<u32>;
    let mut meter = 4_u8;
    let mut seen_meter = None::<u8>;
    let mut fifths = 0_i8;
    let mut minor = false;
    let mut seen_key = None::<(i8, bool)>;
    let mut notes = Vec::new();

    for (track_index, track) in smf.tracks.iter().enumerate() {
        let mut tick = 0_u64;
        // Repeated note-ons for the same pitch/channel are valid in MIDI.
        // Queue their starts rather than overwriting an earlier still-active
        // note; each matching note-off closes the oldest outstanding onset.
        let mut pending: HashMap<(u8, u8), VecDeque<(u64, u8)>> = HashMap::new();
        for event in track {
            tick = tick
                .checked_add(u64::from(event.delta.as_int()))
                .ok_or_else(|| "MIDI absolute tick position overflowed".to_string())?;
            match event.kind {
                TrackEventKind::Meta(MetaMessage::Tempo(value)) => {
                    let micros_per_quarter = value.as_int();
                    if micros_per_quarter == 0 {
                        return Err("MIDI tempo events must be non-zero".into());
                    }
                    if tempo_microseconds_per_quarter
                        .is_some_and(|previous| previous != micros_per_quarter)
                    {
                        return Err(
                            "MIDI tempo changes cannot be represented by a single-tempo Score"
                                .into(),
                        );
                    }
                    let bpm = 60_000_000.0 / micros_per_quarter as f32;
                    if !bpm.is_finite() || !(20.0..=320.0).contains(&bpm) {
                        return Err(
                            "MIDI tempo must be finite and between 20 and 320 BPM".into(),
                        );
                    }
                    tempo_microseconds_per_quarter = Some(micros_per_quarter);
                    tempo_bpm = bpm;
                }
                TrackEventKind::Meta(MetaMessage::TimeSignature(
                    numerator,
                    denominator_power,
                    _,
                    _,
                )) => {
                    // Score stores quarter-note beats per bar, so only x/4
                    // can currently be represented without changing duration.
                    if denominator_power != 2 {
                        return Err(
                            "MIDI time-signature denominators other than quarter notes are not representable"
                                .into(),
                        );
                    }
                    if !(1..=16).contains(&numerator) {
                        return Err(
                            "MIDI time-signature numerator must be in 1..=16".into(),
                        );
                    }
                    if seen_meter.is_some_and(|previous| previous != numerator) {
                        return Err(
                            "MIDI meter changes cannot be represented by a single-meter Score"
                                .into(),
                        );
                    }
                    seen_meter = Some(numerator);
                    meter = numerator;
                }
                TrackEventKind::Meta(MetaMessage::KeySignature(sf, is_minor)) => {
                    if !(-7..=7).contains(&sf) {
                        return Err("MIDI key signature must be in -7..=7 fifths".into());
                    }
                    let signature = (sf, is_minor);
                    if seen_key.is_some_and(|previous| previous != signature) {
                        return Err(
                            "MIDI key changes cannot be represented by a single-key Score".into(),
                        );
                    }
                    seen_key = Some(signature);
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
                            let note_key = (channel, key.as_int());
                            let matched = pending
                                .get_mut(&note_key)
                                .and_then(VecDeque::pop_front);
                            if pending.get(&note_key).is_some_and(VecDeque::is_empty) {
                                pending.remove(&note_key);
                            }
                            if let Some((onset, velocity)) = matched {
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
        for ((_, pitch), starts) in pending {
            for (onset, velocity) in starts {
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
    let mut score = Score::new(key, tempo_bpm, meter);
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
    let mut seen_key = None::<(i32, bool)>;
    let mut meter = 4_u8;
    let mut seen_meter = None::<u8>;
    let mut tempo = 120.0_f32;
    let mut seen_tempo = None::<f32>;
    let mut raw = Vec::<(usize, u8, Duration, Duration)>::new();

    for (part_index, part) in parts.iter().enumerate() {
        // Divisions are part-local and can change during a score. Keep the
        // cursor in exact beats so earlier notes never inherit later divisions.
        let mut divisions = 1_i64;
        let mut cursor = Duration::zero();
        let mut previous_onset = Duration::zero();
        for child in part.descendants().filter(|node| node.is_element()) {
            if child.has_tag_name("divisions") {
                let value = node_i64(child)
                    .ok_or_else(|| "MusicXML divisions must be a positive integer".to_string())?;
                if value <= 0 {
                    return Err("MusicXML divisions must be positive".into());
                }
                divisions = value;
            } else if child.has_tag_name("key") {
                // The Score contract carries one key signature. Reject later
                // changes rather than analyzing/rendering the work in the
                // final key from the file.
                let key_fifths = child
                    .children()
                    .find(|node| node.has_tag_name("fifths"))
                    .and_then(node_i64)
                    .ok_or_else(|| {
                        "MusicXML non-traditional keys are not supported by a single-key Score"
                            .to_string()
                    })?;
                if !(-7..=7).contains(&key_fifths) {
                    return Err("MusicXML key signature fifths must be in -7..=7".into());
                }
                let key_minor = match child.children().find(|node| node.has_tag_name("mode")) {
                    None => false,
                    Some(mode) => match mode.text().unwrap_or("").trim() {
                        "major" => false,
                        "minor" => true,
                        value => {
                            return Err(format!(
                                "MusicXML key mode {value:?} is not supported by a single-key Score"
                            ));
                        }
                    },
                };
                let signature = (key_fifths as i32, key_minor);
                if seen_key.is_some_and(|previous| previous != signature) {
                    return Err(
                        "MusicXML key changes cannot be represented by a single-key Score".into(),
                    );
                }
                seen_key = Some(signature);
                fifths = signature.0;
                minor = signature.1;
            } else if child.has_tag_name("time") {
                // Score::time_signature() interprets meter as quarter-note
                // beats per bar. Compound, non-quarter and time-varying
                // signatures cannot be flattened into this field faithfully.
                let beats: Vec<_> = child
                    .children()
                    .filter(|node| node.has_tag_name("beats"))
                    .collect();
                let beat_types: Vec<_> = child
                    .children()
                    .filter(|node| node.has_tag_name("beat-type"))
                    .collect();
                if beats.len() != 1 || beat_types.len() != 1 {
                    return Err(
                        "MusicXML composite or senza-misura signatures are not supported by Score"
                            .into(),
                    );
                }
                let numerator = node_i64(beats[0]).ok_or_else(|| {
                    "MusicXML time-signature numerator must be a single integer".to_string()
                })?;
                let denominator = node_i64(beat_types[0]).ok_or_else(|| {
                    "MusicXML time-signature denominator must be an integer".to_string()
                })?;
                if denominator != 4 {
                    return Err(
                        "MusicXML time-signature denominators other than quarter notes are not representable"
                            .into(),
                    );
                }
                if !(1..=16).contains(&numerator) {
                    return Err("MusicXML time-signature numerator must be in 1..=16".into());
                }
                let numerator = numerator as u8;
                if seen_meter.is_some_and(|previous| previous != numerator) {
                    return Err(
                        "MusicXML meter changes cannot be represented by a single-meter Score"
                            .into(),
                    );
                }
                seen_meter = Some(numerator);
                meter = numerator;
            } else if child.has_tag_name("metronome") {
                let bpm = musicxml_metronome_tempo(child)?;
                register_musicxml_tempo(bpm, &mut seen_tempo, &mut tempo)?;
            } else if child.has_tag_name("sound") {
                if let Some(value) = child.attribute("tempo") {
                    let bpm = value.parse::<f32>().map_err(|_| {
                        "MusicXML tempo must be a number between 20 and 320 BPM".to_string()
                    })?;
                    register_musicxml_tempo(bpm, &mut seen_tempo, &mut tempo)?;
                }
            } else if child.has_tag_name("backup") {
                let amount = musicxml_duration(child, "backup")?;
                let amount = Duration::new(amount, divisions);
                cursor = cursor
                    .checked_sub(amount)
                    .ok_or_else(|| "MusicXML backup timing is not exactly representable".to_string())?;
                if cursor.num() < 0 {
                    return Err("MusicXML backup moves before the start of a part".into());
                }
            } else if child.has_tag_name("forward") {
                let amount = musicxml_duration(child, "forward")?;
                let amount = Duration::new(amount, divisions);
                cursor = cursor
                    .checked_add(amount)
                    .ok_or_else(|| "MusicXML forward timing overflowed".to_string())?;
            } else if child.has_tag_name("note") {
                if child.children().any(|node| node.has_tag_name("grace")) {
                    return Err(
                        "MusicXML grace notes are not yet supported; refusing to invent a duration"
                            .into(),
                    );
                }
                let duration = musicxml_duration(child, "note")?;
                let duration = Duration::new(duration, divisions);
                let chord = child.children().any(|node| node.has_tag_name("chord"));
                let rest = child.children().any(|node| node.has_tag_name("rest"));
                let onset = if chord { previous_onset } else { cursor };
                if !rest {
                    if let Some(midi) = musicxml_pitch(child)? {
                        if raw.len() >= MAX_IMPORTED_NOTES {
                            return Err(format!(
                                "MusicXML import exceeds the {MAX_IMPORTED_NOTES}-note limit"
                            ));
                        }
                        raw.push((part_index, midi, onset, duration));
                    }
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
    if score.key.tonic.value() >= 12 {
        return Err("symbolic score tonic pitch class must be in 0..=11".into());
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
        if note.pitch.midi() > 127 {
            return Err(format!(
                "symbolic score note {index} has a MIDI pitch outside 0..=127"
            ));
        }
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

/// Extract only a regular, undotted quarter-note metronome mark. Metric
/// modulations, beat-unit conversions and text/range markings require a richer
/// tempo representation than Score's single quarter-note BPM value.
fn musicxml_metronome_tempo(node: roxmltree::Node<'_, '_>) -> Result<f32, String> {
    let beat_units: Vec<_> = node
        .children()
        .filter(|child| child.has_tag_name("beat-unit"))
        .collect();
    let per_minutes: Vec<_> = node
        .children()
        .filter(|child| child.has_tag_name("per-minute"))
        .collect();
    let unsupported_form = node.children().any(|child| {
        child.has_tag_name("beat-unit-dot")
            || child.has_tag_name("beat-unit-tied")
            || child.has_tag_name("metronome-note")
            || child.has_tag_name("metronome-relation")
            || child.has_tag_name("metronome-tied")
            || child.has_tag_name("metronome-tuplet")
    });
    if unsupported_form || beat_units.len() != 1 || per_minutes.len() != 1 {
        return Err(
            "MusicXML only undotted quarter-note metronome marks with one numeric per-minute value are supported"
                .into(),
        );
    }
    if beat_units[0].text().unwrap_or("").trim() != "quarter" {
        return Err(
            "MusicXML metronome beat units other than undotted quarter notes are not supported"
                .into(),
        );
    }
    let value = per_minutes[0].text().unwrap_or("").trim();
    let bpm = value.parse::<f32>().map_err(|_| {
        "MusicXML metronome per-minute value must be a single numeric BPM".to_string()
    })?;
    if !bpm.is_finite() || !(20.0..=320.0).contains(&bpm) {
        return Err(
            "MusicXML metronome tempo must be finite and between 20 and 320 BPM".into(),
        );
    }
    Ok(bpm)
}

/// Keep all tempo indicators aligned with Score's single-tempo contract.
/// A changing or disagreeing indicator must not win merely because it appears
/// later in document traversal.
fn register_musicxml_tempo(
    bpm: f32,
    seen_tempo: &mut Option<f32>,
    tempo: &mut f32,
) -> Result<(), String> {
    if !bpm.is_finite() || !(20.0..=320.0).contains(&bpm) {
        return Err("MusicXML tempo must be finite and between 20 and 320 BPM".into());
    }
    if seen_tempo.is_some_and(|previous| previous != bpm) {
        return Err(
            "MusicXML tempo changes cannot be represented by a single-tempo Score; conflicting tempo indicators are also rejected"
                .into(),
        );
    }
    *seen_tempo = Some(bpm);
    *tempo = bpm;
    Ok(())
}

/// MusicXML durations use positive integer division units. Required values
/// must not be converted into defaults: doing so changes imported timing.
fn musicxml_duration(parent: roxmltree::Node<'_, '_>, element: &str) -> Result<i64, String> {
    let node = parent
        .children()
        .find(|node| node.has_tag_name("duration"))
        .ok_or_else(|| format!("MusicXML {element} is missing its <duration> element"))?;
    let value = node_i64(node).ok_or_else(|| {
        format!("MusicXML {element} duration must be an integer number of divisions")
    })?;
    if value <= 0 {
        return Err(format!("MusicXML {element} duration must be positive"));
    }
    Ok(value)
}

fn musicxml_pitch(note: roxmltree::Node<'_, '_>) -> Result<Option<u8>, String> {
    // Unpitched percussion may have no <pitch> element and remains outside
    // this first pitched-note importer. Once a <pitch> element is present,
    // however, malformed or unsupported values must not silently become a
    // different valid MIDI pitch (or disappear from the imported score).
    let Some(pitch) = note.children().find(|node| node.has_tag_name("pitch")) else {
        return Ok(None);
    };
    let step = pitch
        .children()
        .find(|node| node.has_tag_name("step"))
        .and_then(|node| node.text())
        .ok_or_else(|| "MusicXML pitched note is missing its pitch step".to_string())?;
    let base = match step.trim() {
        "C" => 0_i64,
        "D" => 2_i64,
        "E" => 4_i64,
        "F" => 5_i64,
        "G" => 7_i64,
        "A" => 9_i64,
        "B" => 11_i64,
        _ => return Err(format!("MusicXML pitch step is invalid: {step:?}")),
    };
    let alter = match pitch.children().find(|node| node.has_tag_name("alter")) {
        Some(node) => node_i64(node).ok_or_else(|| {
            "MusicXML pitch alteration must be an integer semitone value".to_string()
        })?,
        None => 0,
    };
    let octave = pitch
        .children()
        .find(|node| node.has_tag_name("octave"))
        .and_then(node_i64)
        .ok_or_else(|| "MusicXML pitched note is missing a valid octave".to_string())?;
    let midi = octave
        .checked_add(1)
        .and_then(|value| value.checked_mul(12))
        .and_then(|value| value.checked_add(base))
        .and_then(|value| value.checked_add(alter))
        .ok_or_else(|| "MusicXML pitch number overflowed".to_string())?;
    if !(0..=127).contains(&midi) {
        return Err(format!(
            "MusicXML pitch {midi} is outside the MIDI pitch range 0..=127"
        ));
    }
    Ok(Some(midi as u8))
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
            <attributes><divisions>1</divisions><key><fifths>0</fifths><mode>major</mode></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
            <note><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#
    }

    fn midi_file(track: &[u8]) -> Vec<u8> {
        let mut bytes = b"MThd".to_vec();
        bytes.extend_from_slice(&[0, 0, 0, 6, 0, 0, 0, 1, 0x01, 0xE0]);
        bytes.extend_from_slice(b"MTrk");
        bytes.extend_from_slice(&(track.len() as u32).to_be_bytes());
        bytes.extend_from_slice(track);
        bytes
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
    fn invalid_musescore_tonic_pitch_class_is_rejected_before_analysis() {
        let mut value = serde_json::to_value(json_score_with_one_note()).unwrap();
        value["key"]["tonic"] = serde_json::json!(255);
        let bytes = serde_json::to_vec(&value).unwrap();

        let error = parse_symbolic(&bytes, SymbolicImportFormat::MuseScore).unwrap_err();
        assert!(error.contains("tonic pitch class"), "{error}");
    }

    #[test]
    fn invalid_musescore_midi_pitch_is_rejected_before_rendering() {
        let mut value = serde_json::to_value(json_score_with_one_note()).unwrap();
        value["notes"][0]["pitch"]["midi"] = serde_json::json!(255);
        let bytes = serde_json::to_vec(&value).unwrap();

        let error = parse_symbolic(&bytes, SymbolicImportFormat::MuseScore).unwrap_err();
        assert!(error.contains("MIDI pitch outside 0..=127"), "{error}");
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
    fn musicxml_pitch_above_midi_range_is_rejected_instead_of_clipped() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <note><pitch><step>C</step><octave>10</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(error.contains("outside the MIDI pitch range 0..=127"), "{error}");
    }

    #[test]
    fn musicxml_fractional_pitch_alteration_is_rejected_explicitly() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <note><pitch><step>C</step><alter>0.5</alter><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(error.contains("integer semitone value"), "{error}");
    }

    #[test]
    fn musicxml_missing_note_duration_is_rejected_instead_of_invented() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <note><pitch><step>C</step><octave>4</octave></pitch></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(error.contains("missing its <duration> element"), "{error}");
    }

    #[test]
    fn musicxml_malformed_note_duration_is_rejected_instead_of_defaulted() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>abc</duration></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(error.contains("integer number of divisions"), "{error}");
    }

    #[test]
    fn musicxml_zero_note_duration_is_rejected_instead_of_promoted() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>0</duration></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(error.contains("duration must be positive"), "{error}");
    }

    #[test]
    fn musicxml_grace_note_is_rejected_instead_of_given_a_default_duration() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <note><grace slash="yes"/><pitch><step>B</step><octave>4</octave></pitch>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(error.contains("grace notes are not yet supported"), "{error}");
    }

    #[test]
    fn musicxml_malformed_divisions_are_rejected_instead_of_reusing_previous_value() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>abc</divisions></attributes>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(error.contains("divisions must be a positive integer"), "{error}");
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
    fn repeated_same_pitch_midi_notes_keep_both_overlapping_onsets() {
        // Two same-pitch note-ons before either note-off must not overwrite
        // the earlier onset or collapse two notes into one.
        let bytes = [
            b'M', b'T', b'h', b'd', 0, 0, 0, 6, 0, 0, 0, 1, 1, 0xE0,
            b'M', b'T', b'r', b'k', 0, 0, 0, 20,
            0, 0x90, 60, 100,
            100, 0x90, 60, 80,
            100, 0x80, 60, 0,
            100, 0x80, 60, 0,
            0, 0xFF, 0x2F, 0,
        ];
        let score = parse_midi(&bytes).unwrap();
        assert_eq!(score.notes.len(), 2);
        assert_eq!(score.notes[0].onset, Duration::new(0, 480));
        assert_eq!(score.notes[0].duration, Duration::new(200, 480));
        assert!((score.notes[0].velocity - 100.0 / 127.0).abs() < 1e-6);
        assert_eq!(score.notes[1].onset, Duration::new(100, 480));
        assert_eq!(score.notes[1].duration, Duration::new(200, 480));
        assert!((score.notes[1].velocity - 80.0 / 127.0).abs() < 1e-6);
    }

    #[test]
    fn midi_tempo_changes_are_rejected_instead_of_flattened() {
        let track = [
            0, 0xFF, 0x51, 3, 0x07, 0xA1, 0x20, // 120 BPM
            100, 0xFF, 0x51, 3, 0x06, 0x1A, 0x80, // 150 BPM
            0, 0x90, 60, 100,
            100, 0x80, 60, 0,
            0, 0xFF, 0x2F, 0,
        ];
        let error = parse_midi(&midi_file(&track)).unwrap_err();
        assert!(error.contains("tempo changes cannot be represented"), "{error}");
    }

    #[test]
    fn midi_non_quarter_meter_is_rejected_instead_of_flattened() {
        let track = [
            0, 0xFF, 0x58, 4, 4, 3, 24, 8, // 4/8, denominator power 3
            0, 0x90, 60, 100,
            100, 0x80, 60, 0,
            0, 0xFF, 0x2F, 0,
        ];
        let error = parse_midi(&midi_file(&track)).unwrap_err();
        assert!(error.contains("denominators other than quarter notes"), "{error}");
    }

    #[test]
    fn midi_key_changes_are_rejected_instead_of_flattened() {
        let track = [
            0, 0xFF, 0x59, 2, 0, 0, // C major
            100, 0xFF, 0x59, 2, 1, 0, // G major
            0, 0x90, 60, 100,
            100, 0x80, 60, 0,
            0, 0xFF, 0x2F, 0,
        ];
        let error = parse_midi(&midi_file(&track)).unwrap_err();
        assert!(error.contains("key changes cannot be represented"), "{error}");
    }

    #[test]
    fn musicxml_quarter_note_metronome_mark_sets_score_tempo() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <direction><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>96</per-minute></metronome></direction-type></direction>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let score = parse_musicxml(xml).unwrap();
        assert_eq!(score.tempo_bpm, 96.0);
    }

    #[test]
    fn musicxml_matching_metronome_and_sound_tempo_are_accepted() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <direction>
                <direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>96</per-minute></metronome></direction-type>
                <sound tempo="96"/>
            </direction>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let score = parse_musicxml(xml).unwrap();
        assert_eq!(score.tempo_bpm, 96.0);
    }

    #[test]
    fn musicxml_conflicting_metronome_and_sound_tempo_are_rejected() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <direction>
                <direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>96</per-minute></metronome></direction-type>
                <sound tempo="90"/>
            </direction>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(error.contains("conflicting tempo indicators"), "{error}");
    }

    #[test]
    fn musicxml_non_quarter_metronome_mark_is_rejected() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <direction><direction-type><metronome><beat-unit>eighth</beat-unit><per-minute>180</per-minute></metronome></direction-type></direction>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(error.contains("beat units other than undotted quarter notes"), "{error}");
    }

    #[test]
    fn musicxml_dotted_metronome_mark_is_rejected() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <direction><direction-type><metronome><beat-unit>quarter</beat-unit><beat-unit-dot/><per-minute>72</per-minute></metronome></direction-type></direction>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(error.contains("only undotted quarter-note metronome marks"), "{error}");
    }

    #[test]
    fn musicxml_metric_modulation_metronome_is_rejected() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <direction><direction-type><metronome><metronome-note><metronome-type>eighth</metronome-type></metronome-note><metronome-relation>equals</metronome-relation><metronome-note><metronome-type>quarter</metronome-type><metronome-dot/></metronome-note></metronome></direction-type></direction>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(error.contains("only undotted quarter-note metronome marks"), "{error}");
    }

    #[test]
    fn musicxml_tempo_changes_are_rejected_instead_of_flattened() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <direction><sound tempo="120"/></direction>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
            <direction><sound tempo="90"/></direction>
            <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(error.contains("tempo changes cannot be represented"), "{error}");
    }

    #[test]
    fn musicxml_meter_changes_are_rejected_instead_of_flattened() {
        let xml = br#"<score-partwise><part id="P1">
            <measure number="1"><attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
                <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
            </measure>
            <measure number="2"><attributes><time><beats>3</beats><beat-type>4</beat-type></time></attributes>
                <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration></note>
            </measure>
        </part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(error.contains("meter changes cannot be represented"), "{error}");
    }

    #[test]
    fn musicxml_non_quarter_meter_is_rejected_instead_of_flattened() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions><time><beats>6</beats><beat-type>8</beat-type></time></attributes>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(error.contains("denominators other than quarter notes"), "{error}");
    }

    #[test]
    fn musicxml_key_changes_are_rejected_instead_of_flattened() {
        let xml = br#"<score-partwise><part id="P1">
            <measure number="1"><attributes><divisions>1</divisions><key><fifths>0</fifths><mode>major</mode></key></attributes>
                <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
            </measure>
            <measure number="2"><attributes><key><fifths>1</fifths><mode>major</mode></key></attributes>
                <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration></note>
            </measure>
        </part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(error.contains("key changes cannot be represented"), "{error}");
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
