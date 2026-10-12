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
    ImportedBeatRational, ImportedMotifNearMatchOccurrence, ImportedMotifNearMatchSummary,
    ImportedMotifOccurrence, ImportedMotifSummary, ImportedNoteReference, ImportedSectionSummary,
    ImportedWorkAnalysis, SymbolicImportFormat,
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

/// One note edge on the globally merged SMF timeline. A velocity means
/// note-on; None represents note-off (including velocity-zero note-on).
#[derive(Clone, Copy, Debug)]
struct RawMidiNoteEvent {
    tick: u64,
    track: usize,
    event_index: usize,
    channel: u8,
    pitch: u8,
    velocity: Option<u8>,
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
    match smf.header.format {
        midly::Format::Sequential => {
            return Err(
                "MIDI format 2 contains sequential independent patterns and cannot be represented as one simultaneous Score"
                    .into(),
            );
        }
        midly::Format::SingleTrack if smf.tracks.len() != 1 => {
            return Err(
                "MIDI format 0 requires exactly one track; refusing to merge malformed single-track input"
                    .into(),
            );
        }
        midly::Format::SingleTrack | midly::Format::Parallel => {}
    }
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
    // Collect absolute tick positions before judging whether these global
    // metadata streams fit Score's single-value fields. Track traversal order
    // is not a global timeline: another track may contain an earlier event.
    let mut tempo_events = Vec::<(u64, u32)>::new();
    let mut meter_events = Vec::<(u64, (u8, u8))>::new();
    let mut key_events = Vec::<(u64, (i8, bool))>::new();
    let mut midi_note_events = Vec::<RawMidiNoteEvent>::new();
    let mut channels_by_track = vec![BTreeSet::<u8>::new(); smf.tracks.len()];
    let mut notes = Vec::new();

    for (track_index, track) in smf.tracks.iter().enumerate() {
        let mut tick = 0_u64;
        for (event_index, event) in track.iter().enumerate() {
            tick = tick
                .checked_add(u64::from(event.delta.as_int()))
                .ok_or_else(|| "MIDI absolute tick position overflowed".to_string())?;
            match event.kind {
                TrackEventKind::Meta(MetaMessage::Tempo(value)) => {
                    let micros_per_quarter = value.as_int();
                    if micros_per_quarter == 0 {
                        return Err("MIDI tempo events must be non-zero".into());
                    }
                    let bpm = 60_000_000.0 / micros_per_quarter as f32;
                    if !bpm.is_finite() || !(20.0..=320.0).contains(&bpm) {
                        return Err("MIDI tempo must be finite and between 20 and 320 BPM".into());
                    }
                    tempo_events.push((tick, micros_per_quarter));
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
                        return Err("MIDI time-signature numerator must be in 1..=16".into());
                    }
                    meter_events.push((tick, (numerator, denominator_power)));
                }
                TrackEventKind::Meta(MetaMessage::KeySignature(sf, is_minor)) => {
                    if !(-7..=7).contains(&sf) {
                        return Err("MIDI key signature must be in -7..=7 fifths".into());
                    }
                    key_events.push((tick, (sf, is_minor)));
                }
                TrackEventKind::Meta(MetaMessage::MidiPort(_)) => {
                    return Err(
                        "MIDI port-routing metadata is not supported; Score cannot preserve output-port identity or safely pair cross-track notes"
                            .into(),
                    );
                }
                TrackEventKind::Meta(MetaMessage::MidiChannel(_)) => {
                    return Err(
                        "MIDI channel-routing metadata is not supported by the pitched-note Score importer"
                            .into(),
                    );
                }
                TrackEventKind::Meta(MetaMessage::Lyric(_)) => {
                    return Err(
                        "MIDI lyric events are not represented by Score; refusing to discard sung text during symbolic import"
                            .into(),
                    );
                }
                TrackEventKind::Meta(MetaMessage::SmpteOffset(_)) => {
                    return Err(
                        "MIDI SMPTE start offsets are not supported by the beat-relative Score importer"
                            .into(),
                    );
                }
                TrackEventKind::Meta(MetaMessage::SequencerSpecific(_)) => {
                    return Err(
                        "MIDI sequencer-specific metadata is not interpreted by the Score importer"
                            .into(),
                    );
                }
                TrackEventKind::Meta(MetaMessage::Unknown(kind, _)) => {
                    return Err(format!(
                        "MIDI unknown meta-event type 0x{kind:02X} is not supported by the Score importer"
                    ));
                }
                TrackEventKind::Midi { channel, message } => {
                    let channel = channel.as_int();
                    let note_event = match message {
                        MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
                            Some((key.as_int(), Some(vel.as_int())))
                        }
                        MidiMessage::NoteOn { key, .. } | MidiMessage::NoteOff { key, .. } => {
                            Some((key.as_int(), None))
                        }
                        unsupported => {
                            return Err(format!(
                                "MIDI channel {} uses unsupported playback event {unsupported:?}; Score cannot preserve controller, instrument, aftertouch, or pitch-bend semantics",
                                channel + 1
                            ));
                        }
                    };
                    if let Some((pitch, velocity)) = note_event {
                        if channel == 9 {
                            return Err(
                                "MIDI channel 10 note events are outside the pitched-note importer; refusing to silently discard them"
                                    .into(),
                            );
                        }
                        channels_by_track[track_index].insert(channel);
                        midi_note_events.push(RawMidiNoteEvent {
                            tick,
                            track: track_index,
                            event_index,
                            channel,
                            pitch,
                            velocity,
                        });
                    }
                }
                TrackEventKind::SysEx(_) => {
                    return Err(
                        "MIDI System Exclusive events are not supported; Score cannot preserve device-specific instrument or playback state"
                            .into(),
                    );
                }
                TrackEventKind::Escape(_) => {
                    return Err(
                        "MIDI escape events are not supported; Score cannot preserve arbitrary synthesizer commands"
                            .into(),
                    );
                }
                _ => {}
            }
        }
    }

    // Score assigns one PartId per MIDI track. A single track that carries
    // multiple note channels would collapse distinct logical parts into one
    // identity, so reject it until Score can preserve channel-level parts.
    if let Some((track, channels)) = channels_by_track
        .iter()
        .enumerate()
        .find(|(_, channels)| channels.len() > 1)
    {
        return Err(format!(
            "MIDI track {track} contains multiple note channels {channels:?}; split channels into separate tracks before import"
        ));
    }

    // In Format 1, tracks are simultaneous streams. Pair by absolute tick,
    // channel and pitch across tracks, retaining the note-on track as the
    // score part. For equal-tick cross-track events use track/event order as
    // a deterministic tie-break; zero-length pairs still fail closed.
    midi_note_events.sort_by_key(|event| (event.tick, event.track, event.event_index));
    let mut pending: HashMap<(u8, u8), VecDeque<(u64, u8, usize)>> = HashMap::new();
    for event in midi_note_events {
        let note_key = (event.channel, event.pitch);
        if let Some(velocity) = event.velocity {
            pending
                .entry(note_key)
                .or_default()
                .push_back((event.tick, velocity, event.track));
            continue;
        }

        let (onset, velocity, onset_track) = pending
            .get_mut(&note_key)
            .and_then(VecDeque::pop_front)
            .ok_or_else(|| {
                "MIDI note-off has no matching active note-on in the merged track timeline"
                    .to_string()
            })?;
        if pending.get(&note_key).is_some_and(VecDeque::is_empty) {
            pending.remove(&note_key);
        }
        let duration = event
            .tick
            .checked_sub(onset)
            .filter(|duration| *duration > 0)
            .ok_or_else(|| "MIDI note-on/off pair must span at least one tick".to_string())?;
        push_raw_note(
            &mut notes,
            RawNote {
                track: onset_track,
                pitch: event.pitch,
                onset,
                duration,
                velocity,
            },
        )?;
    }
    if pending.values().any(|starts| !starts.is_empty()) {
        return Err(
            "MIDI file ends with unmatched note-on events; refusing to infer note durations".into(),
        );
    }

    let tempo_microseconds_per_quarter =
        stable_midi_metadata(&tempo_events, 500_000_u32, "tempo", "single-tempo")?;
    let tempo_bpm = 60_000_000.0 / tempo_microseconds_per_quarter as f32;
    if !tempo_bpm.is_finite() || !(20.0..=320.0).contains(&tempo_bpm) {
        return Err("MIDI tempo must be finite and between 20 and 320 BPM".into());
    }
    let (meter, _denominator_power) =
        stable_midi_metadata(&meter_events, (4_u8, 2_u8), "meter", "single-meter")?;
    let (fifths, minor) = stable_midi_key_metadata(&key_events)?;

    if notes.is_empty() {
        return Err("the MIDI file contains no pitched note events".into());
    }

    // Canonical ordering makes serialized import identity reproducible
    // between process runs.
    notes.sort_by_key(|note| {
        (
            note.onset,
            note.track,
            note.pitch,
            note.duration,
            note.velocity,
        )
    });
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
                velocity: note.velocity as f32 / 127.0,
                role: VoiceRole::Unspecified,
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

pub fn parse_musicxml(bytes: &[u8]) -> Result<Score, String> {
    if bytes.len() > MAX_SYMBOLIC_IMPORT_BYTES {
        return Err(format!(
            "MusicXML import exceeds the {MAX_SYMBOLIC_IMPORT_BYTES}-byte parser limit"
        ));
    }
    let text = std::str::from_utf8(bytes).map_err(|_| "MusicXML must be UTF-8 XML")?;
    let document = roxmltree::Document::parse(text)
        .map_err(|error| format!("MusicXML parse error: {error}"))?;
    let root = document.root_element();
    if root.has_tag_name("score-timewise") {
        return Err(
            "MusicXML score-timewise organization is not supported; convert to score-partwise before import"
                .into(),
        );
    }
    if !root.has_tag_name("score-partwise") {
        return Err(format!(
            "MusicXML root element <{}> is not a supported score-partwise document",
            root.tag_name().name()
        ));
    }
    // Score models note pitch, onset, duration, velocity, and structural
    // emphasis, but not conventional articulations, ornaments, expressive
    // pauses, octave shifts, or written crescendo/diminuendo wedges. Refuse
    // source markings whose performance semantics would otherwise disappear.
    let unsupported_performance_tags = [
        "accent",
        "arpeggiate",
        "bend",
        "breath-mark",
        "caesura",
        "detached-legato",
        "doit",
        "fall",
        "falloff",
        "fermata",
        "glissando",
        "inverted-mordent",
        "inverted-turn",
        "mordent",
        "non-arpeggiate",
        "octave-shift",
        "plop",
        "scoop",
        "slur",
        "spiccato",
        "staccatissimo",
        "staccato",
        "strong-accent",
        "tenuto",
        "tremolo",
        "trill-mark",
        "turn",
        "wedge",
    ];
    if let Some(marking) = document.descendants().find(|node| {
        unsupported_performance_tags
            .iter()
            .any(|tag| node.has_tag_name(*tag))
    }) {
        return Err(format!(
            "MusicXML <{}> performance marking is not represented by Score; refusing to flatten articulation or expressive timing semantics",
            marking.tag_name().name()
        ));
    }
    if document
        .descendants()
        .any(|node| node.has_tag_name("dynamics"))
    {
        return Err(
            "MusicXML notated <dynamics> markings are not mapped to note velocity; only supported <sound dynamics> playback metadata can be imported"
                .into(),
        );
    }
    if document
        .descendants()
        .any(|node| node.has_tag_name("midi-instrument"))
    {
        return Err(
            "MusicXML <midi-instrument> routing/program metadata is not represented by Score"
                .into(),
        );
    }
    // A linear note list cannot preserve repeat traversal, alternate endings,
    // or the elapsed span of a multiple-measure rest. Reject these structures
    // before parsing note positions rather than silently compressing the score.
    if let Some(structure) = document.descendants().find(|node| {
        ["repeat", "ending", "multiple-rest"]
            .iter()
            .any(|tag| node.has_tag_name(*tag))
    }) {
        let name = structure.tag_name().name();
        return Err(format!(
            "MusicXML <{name}> repeat/measure structure cannot be represented by the linear Score importer; refusing to flatten source timing"
        ));
    }
    // The importer has no separate written-pitch/sounding-pitch layer. A
    // transposing-instrument directive must not be silently ignored.
    if document
        .descendants()
        .any(|node| node.has_tag_name("transpose"))
    {
        return Err(
            "MusicXML part transposition is not supported; convert to concert pitch before import"
                .into(),
        );
    }
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
    // Score assigns one persistent PartId to each MusicXML <part>. Distinct
    // voices or staves inside one part would therefore be flattened into one
    // line and could fabricate line-motion/voice relationships.
    for part in &parts {
        let voices: BTreeSet<_> = part
            .descendants()
            .filter(|node| node.has_tag_name("voice"))
            .filter_map(|node| node.text())
            .map(str::trim)
            .filter(|voice| !voice.is_empty())
            .collect();
        if voices.len() > 1 {
            return Err(format!(
                "MusicXML part {:?} contains multiple voices {voices:?}; separate the voices into individual parts before import",
                part.attribute("id").unwrap_or("unknown")
            ));
        }

        let staves: BTreeSet<_> = part
            .descendants()
            .filter(|node| node.has_tag_name("staff"))
            .filter_map(|node| node.text())
            .map(str::trim)
            .filter(|staff| !staff.is_empty())
            .collect();
        if staves.len() > 1 {
            return Err(format!(
                "MusicXML part {:?} contains multiple staves {staves:?}; split staves into individual parts before import",
                part.attribute("id").unwrap_or("unknown")
            ));
        }
    }
    if document
        .descendants()
        .any(|node| node.has_tag_name("lyric"))
    {
        return Err(
            "MusicXML lyrics are not represented by Score; refusing to discard sung text during symbolic import"
                .into(),
        );
    }
    // Collect metadata before reducing it to Score's single-value fields.
    // Part traversal order is not musical order across the whole score.
    let mut key_events = Vec::<(Duration, (i32, bool))>::new();
    let mut meter_events = Vec::<(Duration, u8)>::new();
    let mut tempo_events = Vec::<(Duration, f32)>::new();
    let mut raw = Vec::<(usize, u8, Duration, Duration, f32)>::new();

    for (part_index, part) in parts.iter().enumerate() {
        // Divisions are part-local and can change during a score. Keep the
        // cursor in exact beats so earlier notes never inherit later divisions.
        let mut divisions = 1_i64;
        let mut cursor = Duration::zero();
        let mut previous_onset = Duration::zero();
        let mut current_velocity = 90.0_f32 / 127.0;
        for child in part.descendants().filter(|node| node.is_element()) {
            if child.has_tag_name("divisions") {
                let value = node_i64(child)
                    .ok_or_else(|| "MusicXML divisions must be a positive integer".to_string())?;
                if value <= 0 {
                    return Err("MusicXML divisions must be positive".into());
                }
                divisions = value;
            } else if child.has_tag_name("key") {
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
                key_events.push((cursor, (key_fifths as i32, key_minor)));
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
                meter_events.push((cursor, numerator));
            } else if child.has_tag_name("metronome") {
                let bpm = musicxml_metronome_tempo(child)?;
                let position = musicxml_sound_position(child, cursor)?;
                tempo_events.push((position, bpm));
            } else if child.has_tag_name("sound") {
                if [
                    "coda",
                    "dacapo",
                    "dalsegno",
                    "damper-pedal",
                    "divisions",
                    "elevation",
                    "fine",
                    "forward-repeat",
                    "pan",
                    "pizzicato",
                    "segno",
                    "soft-pedal",
                    "sostenuto-pedal",
                    "time-only",
                    "tocoda",
                ]
                .iter()
                .any(|attribute| child.attribute(*attribute).is_some())
                    || child
                        .children()
                        .any(|node| node.is_element() && !node.has_tag_name("offset"))
                {
                    return Err(
                        "MusicXML sound uses unsupported playback attributes or children; tempo/dynamics-only sound directives are supported"
                            .into(),
                    );
                }
                if let Some(value) = child.attribute("tempo") {
                    let bpm = value.parse::<f32>().map_err(|_| {
                        "MusicXML tempo must be a number between 20 and 320 BPM".to_string()
                    })?;
                    if !bpm.is_finite() || !(20.0..=320.0).contains(&bpm) {
                        return Err(
                            "MusicXML tempo must be finite and between 20 and 320 BPM".into()
                        );
                    }
                    let position = musicxml_sound_position(child, cursor)?;
                    tempo_events.push((position, bpm));
                }
                if let Some(value) = child.attribute("dynamics") {
                    // Non-zero effective offsets require a positioned dynamics
                    // map. Score stores velocity per note, so supported
                    // unoffset changes can update this part's later notes.
                    musicxml_sound_position(child, cursor)?;
                    current_velocity = musicxml_dynamics_velocity(value, "sound dynamics")?;
                }
            } else if child.has_tag_name("backup") {
                let amount = musicxml_duration(child, "backup")?;
                let amount = Duration::new(amount, divisions);
                cursor = cursor.checked_sub(amount).ok_or_else(|| {
                    "MusicXML backup timing is not exactly representable".to_string()
                })?;
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
                // A tied note segment would be imported as a fresh attack
                // because Score currently has no tie/sustain-link metadata.
                if child
                    .descendants()
                    .any(|node| node.has_tag_name("tie") || node.has_tag_name("tied"))
                {
                    return Err(
                        "MusicXML ties are not supported by the single-note-event importer; refusing to turn tied segments into reattacks"
                            .into(),
                    );
                }
                if [
                    "end-dynamics",
                    "attack",
                    "release",
                    "pizzicato",
                    "time-only",
                ]
                .iter()
                .any(|attribute| child.attribute(*attribute).is_some())
                {
                    return Err(
                        "MusicXML note uses end-dynamics, attack, release, pizzicato, or time-only, which the current Score model cannot represent"
                            .into(),
                    );
                }
                if child
                    .children()
                    .any(|node| node.has_tag_name("instrument") || node.has_tag_name("play"))
                {
                    return Err(
                        "MusicXML note uses an unsupported instrument or playback-technique child"
                            .into(),
                    );
                }
                let note_velocity = match child.attribute("dynamics") {
                    Some(value) => musicxml_dynamics_velocity(value, "note dynamics")?,
                    None => current_velocity,
                };
                let duration = musicxml_duration(child, "note")?;
                let duration = Duration::new(duration, divisions);
                let chord = child.children().any(|node| node.has_tag_name("chord"));
                let pitch_count = child
                    .children()
                    .filter(|node| node.has_tag_name("pitch"))
                    .count();
                let rest_count = child
                    .children()
                    .filter(|node| node.has_tag_name("rest"))
                    .count();
                let unpitched_count = child
                    .children()
                    .filter(|node| node.has_tag_name("unpitched"))
                    .count();
                if pitch_count + rest_count + unpitched_count != 1 {
                    return Err(
                        "MusicXML note must contain exactly one of <pitch>, <unpitched>, or <rest>"
                            .into(),
                    );
                }
                if unpitched_count != 0 {
                    return Err(
                        "MusicXML unpitched notes are outside the pitched-note importer; refusing to silently discard percussion or speaking-voice events"
                            .into(),
                    );
                }
                let rest = rest_count == 1;
                let onset = if chord { previous_onset } else { cursor };
                if !rest {
                    let midi = musicxml_pitch(child)?.ok_or_else(|| {
                        "MusicXML note must contain exactly one of <pitch>, <unpitched>, or <rest>"
                            .to_string()
                    })?;
                    if raw.len() >= MAX_IMPORTED_NOTES {
                        return Err(format!(
                            "MusicXML import exceeds the {MAX_IMPORTED_NOTES}-note limit"
                        ));
                    }
                    raw.push((part_index, midi, onset, duration, note_velocity));
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
    let (fifths, minor) =
        stable_musicxml_metadata(&key_events, (0_i32, false), "key", "single-key")?;
    let meter = stable_musicxml_metadata(&meter_events, 4_u8, "meter", "single-meter")?;
    let tempo = stable_musicxml_metadata(&tempo_events, 120.0_f32, "tempo", "single-tempo")?;

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
    for (part, midi, onset, duration, velocity) in raw {
        score
            .try_push(ScoreNote {
                part: u16::try_from(part)
                    .map(PartId)
                    .unwrap_or(PartId::UNASSIGNED),
                pitch: Pitch::from_midi(midi),
                onset,
                duration,
                velocity,
                role: VoiceRole::Unspecified,
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

/// Validate a MIDI tempo or meter stream against a documented format default
/// and Score's single-value field. Tempo and time signature have SMF defaults;
/// key signature is handled separately because an absent key event is not a
/// reliable statement about the notated key of the opening.
fn stable_midi_metadata<T: Copy + Eq>(
    events: &[(u64, T)],
    default: T,
    name: &str,
    schema_field: &str,
) -> Result<T, String> {
    let Some((first_tick, first_value)) = events.iter().min_by_key(|event| event.0) else {
        return Ok(default);
    };
    if *first_tick > 0 && *first_value != default {
        return Err(format!(
            "MIDI {name} changes from the default at a non-zero tick and cannot be represented by a {schema_field} Score"
        ));
    }
    if events.iter().any(|(_, value)| value != first_value) {
        return Err(format!(
            "MIDI {name} changes cannot be represented by a {schema_field} Score"
        ));
    }
    Ok(*first_value)
}

/// Resolve MIDI key-signature events without treating the internal C-major
/// fallback as proof of opening context. A key event first seen after tick zero
/// only establishes later metadata and cannot describe the opening in a
/// single-key Score.
fn stable_midi_key_metadata(events: &[(u64, (i8, bool))]) -> Result<(i8, bool), String> {
    let Some((first_tick, first_value)) = events.iter().min_by_key(|event| event.0).copied() else {
        return Ok((0, false));
    };
    if first_tick > 0 {
        return Err(
            "MIDI key signature first appears at a non-zero tick; opening key is unspecified and cannot be represented by a single-key Score"
                .into(),
        );
    }
    if events.iter().any(|(_, value)| *value != first_value) {
        return Err("MIDI key changes cannot be represented by a single-key Score".into());
    }
    Ok(first_value)
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
        return Err("MusicXML metronome tempo must be finite and between 20 and 320 BPM".into());
    }
    Ok(bpm)
}

/// Resolve one metadata stream against Score's single-value field.
///
/// Events are collected across all parts before temporal validation. That way,
/// a late event encountered in an earlier XML part cannot override or
/// incorrectly precede a beat-zero event in a later part.
fn stable_musicxml_metadata<T: Copy + PartialEq>(
    events: &[(Duration, T)],
    default: T,
    name: &str,
    schema_field: &str,
) -> Result<T, String> {
    let Some((mut earliest_position, first_value)) = events.first().copied() else {
        return Ok(default);
    };

    for &(position, value) in &events[1..] {
        if value != first_value {
            // Two different tempo declarations at the same score position are
            // a conflict, not merely a later tempo change. Preserve that
            // distinction in diagnostics while still failing closed.
            let conflicting_tempo_at_same_position = name == "tempo"
                && events.iter().any(|(other_position, other_value)| {
                    *other_value != value
                        && other_position.checked_cmp(position)
                            == Some(std::cmp::Ordering::Equal)
                });
            if conflicting_tempo_at_same_position {
                return Err(
                    "MusicXML conflicting tempo indicators at the same score position cannot be represented by a single-tempo Score"
                        .into(),
                );
            }
            return Err(format!(
                "MusicXML {name} changes cannot be represented by a {schema_field} Score"
            ));
        }
        let order = position
            .checked_cmp(earliest_position)
            .ok_or_else(|| format!("MusicXML {name} event positions cannot be compared exactly"))?;
        if order == std::cmp::Ordering::Less {
            earliest_position = position;
        }
    }

    if earliest_position.num() > 0 {
        return Err(format!(
            "MusicXML {name} first appears at a non-zero beat; opening metadata cannot be represented by a single-value {schema_field} Score"
        ));
    }
    Ok(first_value)
}

/// Return the current beat position for a sound/dynamics indicator only when
/// its effective sound/direction offset is zero. A sound-level offset
/// overrides a containing direction offset, so inspect it first.
fn musicxml_sound_position(
    node: roxmltree::Node<'_, '_>,
    position: Duration,
) -> Result<Duration, String> {
    if let Some(offset) = node.children().find(|child| child.has_tag_name("offset")) {
        let value = node_i64(offset)
            .ok_or_else(|| "MusicXML sound tempo offset must be an integer".to_string())?;
        if value != 0 {
            return Err(
                "MusicXML non-zero sound or direction tempo offsets are not supported by the single-tempo Score importer"
                    .into(),
            );
        }
        // A sound-level offset takes precedence, including an explicit zero.
        return Ok(position);
    }

    let mut ancestor = node.parent();
    while let Some(parent) = ancestor {
        if parent.has_tag_name("direction") {
            if let Some(offset) = parent.children().find(|child| child.has_tag_name("offset")) {
                let value = node_i64(offset).ok_or_else(|| {
                    "MusicXML tempo direction offset must be an integer".to_string()
                })?;
                if value != 0 {
                    return Err(
                        "MusicXML non-zero sound or direction tempo offsets are not supported by the single-tempo Score importer"
                            .into(),
                    );
                }
            }
            break;
        }
        ancestor = parent.parent();
    }
    Ok(position)
}

/// Map MusicXML playback dynamics (percent of forte MIDI velocity 90) onto
/// the representable MIDI velocity domain. Reject values that cannot be
/// encoded rather than clipping or silently changing the source value.
fn musicxml_dynamics_velocity(value: &str, field: &str) -> Result<f32, String> {
    let percent = value
        .trim()
        .parse::<f32>()
        .map_err(|_| format!("MusicXML {field} must be a non-negative decimal"))?;
    if !percent.is_finite() || percent < 0.0 {
        return Err(format!(
            "MusicXML {field} must be a finite non-negative decimal"
        ));
    }
    let midi_velocity = (percent * 0.9).round();
    if midi_velocity > 127.0 {
        return Err(format!(
            "MusicXML {field} exceeds the representable MIDI velocity range"
        ));
    }
    Ok(midi_velocity / 127.0)
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

const MAX_MOTIF_OCCURRENCE_EVIDENCE: usize = 128;
const MAX_MOTIF_NEAR_MATCH_OCCURRENCE_EVIDENCE: usize = 128;
const MOTIF_NEAR_MATCH_MAX_INTERVAL_DEVIATION_SEMITONES: i16 = 1;
const MOTIF_NEAR_MATCH_MAX_NORMALIZED_ONSET_GAP_ERROR: f64 = 0.15;

/// Normalize exact inter-onset gaps to shares of the motif's total onset span.
/// This makes the shape invariant to uniform tempo scaling while keeping the
/// tolerance definition explicit. The five-slot array is sufficient for the
/// importer's fixed 3–6-note motif window.
fn normalized_onset_gap_shape(notes: &[(usize, &ScoreNote)]) -> Option<[f64; 5]> {
    if !(3..=6).contains(&notes.len()) {
        return None;
    }
    let gap_count = notes.len() - 1;
    let mut gaps = [0.0_f64; 5];
    let mut total = 0.0_f64;
    for (index, pair) in notes.windows(2).enumerate() {
        let delta = pair[1].1.onset.checked_sub(pair[0].1.onset)?;
        if delta.num() <= 0 || delta.den() <= 0 {
            return None;
        }
        let beats = delta.beats();
        if !beats.is_finite() || beats <= 0.0 {
            return None;
        }
        gaps[index] = beats;
        total += beats;
    }
    if !total.is_finite() || total <= 0.0 {
        return None;
    }
    for gap in &mut gaps[..gap_count] {
        *gap /= total;
    }
    Some(gaps)
}

/// Compare equal-length part-local windows against a representative contour.
/// A near match must preserve every interval's direction, stay within one
/// semitone of each signed interval, and keep every normalized onset-gap share
/// within 0.15 of the representative. Exact interval + rhythm-shape matches
/// remain solely in the exact evidence layer. Counts include overlapping
/// windows; independent counts use deterministic earliest-first non-overlap.
fn rhythm_near_match_summary(
    notes: &[(usize, &ScoreNote)],
    target_pitch_intervals: &[i16],
    target_gap_shape: [f64; 5],
) -> ImportedMotifNearMatchSummary {
    let note_len = target_pitch_intervals.len() + 1;
    let mut summary = ImportedMotifNearMatchSummary {
        occurrence_count: 0,
        independent_occurrence_count: 0,
        max_allowed_pitch_interval_deviation_semitones:
            MOTIF_NEAR_MATCH_MAX_INTERVAL_DEVIATION_SEMITONES as u8,
        max_allowed_normalized_onset_gap_error: MOTIF_NEAR_MATCH_MAX_NORMALIZED_ONSET_GAP_ERROR,
        occurrence_evidence_complete: true,
        occurrences: Vec::new(),
    };
    if !(3..=6).contains(&note_len) || notes.len() < note_len {
        return summary;
    }

    let mut next_non_overlapping_start = 0_usize;
    for window_start in 0..=(notes.len() - note_len) {
        let candidate = &notes[window_start..window_start + note_len];
        let mut interval_pattern_is_compatible = true;
        let mut interval_pattern_is_exact = true;
        let mut max_interval_deviation = 0_u8;

        for (pair, &target_interval) in candidate.windows(2).zip(target_pitch_intervals) {
            let candidate_interval =
                i16::from(pair[1].1.pitch.midi()) - i16::from(pair[0].1.pitch.midi());
            if candidate_interval.signum() != target_interval.signum() {
                interval_pattern_is_compatible = false;
                break;
            }
            let deviation = (candidate_interval - target_interval).abs();
            if deviation > MOTIF_NEAR_MATCH_MAX_INTERVAL_DEVIATION_SEMITONES {
                interval_pattern_is_compatible = false;
                break;
            }
            if deviation != 0 {
                interval_pattern_is_exact = false;
            }
            max_interval_deviation = max_interval_deviation.max(deviation as u8);
        }
        if !interval_pattern_is_compatible {
            continue;
        }
        // Exact pitch-interval contours belong to the exact layer, regardless
        // of rhythmic variation; do not double-count them as near matches.
        if interval_pattern_is_exact {
            continue;
        }

        let Some(candidate_gap_shape) = normalized_onset_gap_shape(candidate) else {
            continue;
        };
        let gap_count = note_len - 1;
        let mut max_gap_error = 0.0_f64;
        let mut total_gap_error = 0.0_f64;
        for index in 0..gap_count {
            let error = (candidate_gap_shape[index] - target_gap_shape[index]).abs();
            max_gap_error = max_gap_error.max(error);
            total_gap_error += error;
        }
        if max_gap_error > MOTIF_NEAR_MATCH_MAX_NORMALIZED_ONSET_GAP_ERROR {
            continue;
        }
        summary.occurrence_count += 1;
        if window_start >= next_non_overlapping_start {
            summary.independent_occurrence_count += 1;
            next_non_overlapping_start = window_start + note_len;
        }
        if summary.occurrences.len() >= MAX_MOTIF_NEAR_MATCH_OCCURRENCE_EVIDENCE {
            continue;
        }

        let references = candidate
            .iter()
            .enumerate()
            .map(|(offset, (score_note_index, note))| ImportedNoteReference {
                score_note_index: *score_note_index,
                part_note_index: window_start + offset,
                midi_pitch: note.pitch.midi(),
                onset_numerator: note.onset.num(),
                onset_denominator: note.onset.den(),
                duration_numerator: note.duration.num(),
                duration_denominator: note.duration.den(),
            })
            .collect();

        summary.occurrences.push(ImportedMotifNearMatchOccurrence {
            notes: references,
            observed_max_pitch_interval_deviation_semitones: max_interval_deviation,
            observed_max_normalized_onset_gap_error: max_gap_error,
            observed_mean_normalized_onset_gap_error: total_gap_error / gap_count as f64,
        });
    }
    summary.occurrence_evidence_complete =
        summary.occurrence_count <= MAX_MOTIF_NEAR_MATCH_OCCURRENCE_EVIDENCE;
    summary
}

fn part_local_motif_summary(score: &Score, part: PartId) -> Option<ImportedMotifSummary> {
    // Keep original Score indices and explicit within-part ordering together.
    // The index is relative to this exact score snapshot, not a persistent ID.
    let mut notes: Vec<_> = score
        .notes
        .iter()
        .enumerate()
        .filter(|(_, note)| note.part == part)
        .map(|(score_note_index, note)| (score_note_index, note))
        .collect();
    for (_, note) in &notes {
        if note.onset.den() <= 0
            || note.duration.den() <= 0
            || note.onset.num() < 0
            || note.duration.num() <= 0
            || note.pitch.midi() > 127
            || note.onset.checked_add(note.duration).is_none()
        {
            return None;
        }
    }
    notes.sort_by(|(left_index, left), (right_index, right)| {
        left.onset
            .checked_cmp(right.onset)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left_index.cmp(right_index))
    });
    if notes.len() < 2 {
        return None;
    }

    // A PartId preserves source-line identity, but one part may contain chords
    // or overlapping voices. Do not invent a monophonic contour in that case.
    for pair in notes.windows(2) {
        let previous_end = pair[0].1.onset.checked_add(pair[0].1.duration)?;
        let ordering = previous_end.checked_cmp(pair[1].1.onset)?;
        if ordering == std::cmp::Ordering::Greater {
            return None;
        }
    }

    // Pack up to five signed semitone intervals into one key so candidate
    // discovery is linear-ish and does not allocate a Vec for every window.
    // MIDI pitches bound each interval to -127..=127, which fits one byte
    // after adding 127. The top byte records note count to avoid length clashes.
    let signature_for = |start: usize, note_len: usize| -> u64 {
        let mut signature = (note_len as u64) << 56;
        for (offset, pair) in notes[start..start + note_len].windows(2).enumerate() {
            let interval = i16::from(pair[1].1.pitch.midi()) - i16::from(pair[0].1.pitch.midi());
            signature |= ((interval + 127) as u64) << (offset * 8);
        }
        signature
    };

    #[derive(Clone, Copy)]
    struct PatternStats {
        note_len: usize,
        first_start: usize,
        occurrence_count: usize,
        independent_occurrence_count: usize,
        next_non_overlapping_start: usize,
    }

    let mut patterns = HashMap::<u64, PatternStats>::new();
    for start in 0..notes.len() {
        let max_len = (notes.len() - start).min(6);
        if max_len < 3 {
            continue;
        }
        for note_len in 3..=max_len {
            let signature = signature_for(start, note_len);
            let stats = patterns.entry(signature).or_insert(PatternStats {
                note_len,
                first_start: start,
                occurrence_count: 0,
                independent_occurrence_count: 0,
                next_non_overlapping_start: start,
            });
            // Preserve every exact match, including overlapping windows, for
            // transparent evidence. Count independent occurrences separately.
            stats.occurrence_count += 1;
            if start >= stats.next_non_overlapping_start {
                stats.independent_occurrence_count += 1;
                stats.next_non_overlapping_start = start + note_len;
            }
        }
    }

    // Prefer longer repeated contours, then the contour with more instances,
    // then the earliest first instance. Signature is a final stable tiebreak.
    let mut selected: Option<(u64, PatternStats)> = None;
    for (&signature, &stats) in &patterns {
        if stats.independent_occurrence_count < 2 {
            continue;
        }
        let better = match selected {
            None => true,
            Some((best_signature, best)) => {
                stats.note_len > best.note_len
                    || (stats.note_len == best.note_len
                        && (stats.independent_occurrence_count > best.independent_occurrence_count
                            || (stats.independent_occurrence_count
                                == best.independent_occurrence_count
                                && (stats.first_start < best.first_start
                                    || (stats.first_start == best.first_start
                                        && signature < best_signature)))))
            }
        };
        if better {
            selected = Some((signature, stats));
        }
    }
    // Preserve a useful provisional summary when no repeated contour exists.
    // A singleton is evidence of the opening pattern, not a repetition claim.
    let (selected_signature, selected_stats) = selected.unwrap_or_else(|| {
        let note_len = notes.len().min(6);
        let opening_signature = signature_for(0, note_len);
        let stats = patterns
            .get(&opening_signature)
            .copied()
            .unwrap_or(PatternStats {
                note_len,
                first_start: 0,
                occurrence_count: 1,
                independent_occurrence_count: 1,
                next_non_overlapping_start: note_len,
            });
        (opening_signature, stats)
    });
    let motif_len = selected_stats.note_len;
    let midi_pitches: Vec<u8> = notes
        .iter()
        .skip(selected_stats.first_start)
        .take(motif_len)
        .map(|(_, note)| note.pitch.midi())
        .collect();
    let pitch_interval_pattern: Vec<i16> = midi_pitches
        .windows(2)
        .map(|pair| i16::from(pair[1]) - i16::from(pair[0]))
        .collect();
    if pitch_interval_pattern.is_empty() {
        return None;
    }

    let representative_notes =
        &notes[selected_stats.first_start..selected_stats.first_start + motif_len];
    let mut onset_deltas = Vec::with_capacity(motif_len.saturating_sub(1));
    for pair in representative_notes.windows(2) {
        let delta = pair[1].1.onset.checked_sub(pair[0].1.onset)?;
        if delta.num() < 0 || delta.den() <= 0 {
            return None;
        }
        onset_deltas.push(ImportedBeatRational {
            numerator: delta.num(),
            denominator: delta.den(),
        });
    }

    let rhythm_near_matches = normalized_onset_gap_shape(representative_notes)
        .map(|shape| rhythm_near_match_summary(&notes, &pitch_interval_pattern, shape));

    let mut occurrence_count = 0_usize;
    let mut independent_occurrence_count = 0_usize;
    let mut next_non_overlapping_start = 0_usize;
    let mut matched_occurrences = Vec::new();
    for window_start in 0..=notes.len().saturating_sub(motif_len) {
        if signature_for(window_start, motif_len) != selected_signature {
            continue;
        }
        occurrence_count += 1;
        if window_start >= next_non_overlapping_start {
            independent_occurrence_count += 1;
            next_non_overlapping_start = window_start + motif_len;
        }
        if matched_occurrences.len() >= MAX_MOTIF_OCCURRENCE_EVIDENCE {
            continue;
        }

        let mut references = Vec::with_capacity(motif_len);
        for (offset, (score_note_index, note)) in
            notes.iter().skip(window_start).take(motif_len).enumerate()
        {
            references.push(ImportedNoteReference {
                score_note_index: *score_note_index,
                part_note_index: window_start + offset,
                midi_pitch: note.pitch.midi(),
                onset_numerator: note.onset.num(),
                onset_denominator: note.onset.den(),
                duration_numerator: note.duration.num(),
                duration_denominator: note.duration.den(),
            });
        }
        matched_occurrences.push(ImportedMotifOccurrence { notes: references });
    }
    debug_assert_eq!(occurrence_count, selected_stats.occurrence_count);
    debug_assert_eq!(
        independent_occurrence_count,
        selected_stats.independent_occurrence_count
    );

    let identity_note = if independent_occurrence_count > 1 {
        "Longest repeated non-overlapping part-local pitch-interval contour (3–6 notes); musical function remains unspecified; contributor confirmation required"
    } else {
        "Opening part-local pitch-interval contour; no non-overlapping repeat of this contour observed; musical function remains unspecified; contributor confirmation required"
    };
    Some(ImportedMotifSummary {
        source_part_id: Some(part.0),
        occurrence_count,
        independent_occurrence_count: Some(independent_occurrence_count),
        midi_pitches,
        identity_note: identity_note.into(),
        confidence: if independent_occurrence_count > 1 {
            0.72
        } else {
            0.42
        },
        pitch_interval_pattern,
        onset_deltas,
        matched_occurrences,
        occurrence_evidence_complete: Some(occurrence_count <= MAX_MOTIF_OCCURRENCE_EVIDENCE),
        rhythm_near_matches,
    })
}

pub fn analyze(score: &Score) -> ImportedWorkAnalysis {
    let parts = score.parts();
    let voice_count = parts.len();
    let motifs: Vec<_> = parts
        .iter()
        .filter_map(|part| part_local_motif_summary(score, *part))
        .collect();
    let skipped_part_count = voice_count.saturating_sub(motifs.len());
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
    ];
    if motifs.is_empty() {
        unresolved_interpretations.push(
            "No role-neutral motif summary could be reconstructed from a stable monophonic source part".into(),
        );
    } else {
        unresolved_interpretations.push(
            "Confirm source-part motif identities; no audio or listener-recognition claim is made"
                .into(),
        );
    }
    if skipped_part_count > 0 {
        unresolved_interpretations.push(format!(
            "Part-local motif extraction was withheld for {skipped_part_count} source part(s) with fewer than two notes or overlapping/polyphonic events"
        ));
    }
    let unassigned_note_count = score
        .notes
        .iter()
        .filter(|note| !note.part.is_assigned())
        .count();
    if unassigned_note_count > 0 {
        unresolved_interpretations.push(format!(
            "Part-local motif extraction did not combine {unassigned_note_count} note(s) without persistent PartId identity"
        ));
    }
    if start < total {
        unresolved_interpretations.push(format!(
            "Section reconstruction stopped at its {MAX_RECONSTRUCTED_SECTIONS}-region safety cap; the remaining duration is not segmented"
        ));
    }
    ImportedWorkAnalysis {
        source_native: true,
        inferred_territory: None,
        tempo_bpm: score.tempo_bpm,
        meter: score.meter,
        tonic: score.key.tonic.name().into(),
        note_count: score.notes.len(),
        voice_count,
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
        midi_file_with_tracks(&[track])
    }

    fn monophonic_midi_track(pitches: &[u8], extra_rest_before_note: &[u8]) -> Vec<u8> {
        let mut track = Vec::new();
        for (index, pitch) in pitches.iter().copied().enumerate() {
            let rest = extra_rest_before_note
                .get(index)
                .copied()
                .unwrap_or_default();
            track.extend_from_slice(&[rest, 0x90, pitch, 100, 100, 0x80, pitch, 0]);
        }
        track.extend_from_slice(&[0, 0xFF, 0x2F, 0]);
        track
    }

    fn midi_file_with_tracks(tracks: &[&[u8]]) -> Vec<u8> {
        let format = if tracks.len() > 1 { 1_u16 } else { 0_u16 };
        midi_file_with_format(format, tracks)
    }

    fn midi_file_with_format(format: u16, tracks: &[&[u8]]) -> Vec<u8> {
        let mut bytes = b"MThd".to_vec();
        bytes.extend_from_slice(&[0, 0, 0, 6]);
        bytes.extend_from_slice(&format.to_be_bytes());
        bytes.extend_from_slice(&(tracks.len() as u16).to_be_bytes());
        bytes.extend_from_slice(&[0x01, 0xE0]);
        for track in tracks {
            bytes.extend_from_slice(b"MTrk");
            bytes.extend_from_slice(&(track.len() as u32).to_be_bytes());
            bytes.extend_from_slice(track);
        }
        bytes
    }

    #[test]
    fn midi_low_note_velocity_is_preserved_without_amplifying_floor() {
        let track = [
            0, 0x90, 60, 1, // lowest positive MIDI note-on velocity
            100, 0x80, 60, 0, 0, 0xFF, 0x2F, 0,
        ];
        let score = parse_midi(&midi_file(&track)).unwrap();
        assert_eq!(score.notes.len(), 1);
        assert!(
            (score.notes[0].velocity - 1.0_f32 / 127.0).abs() < 1e-7,
            "expected the source velocity ratio, got {}",
            score.notes[0].velocity
        );
    }

    #[test]
    fn midi_import_preserves_part_identity_without_guessing_functional_role() {
        let track = [0, 0x90, 72, 100, 100, 0x80, 72, 0, 0, 0xFF, 0x2F, 0];
        let score = parse_midi(&midi_file(&track)).unwrap();
        assert_eq!(score.notes.len(), 1);
        assert_eq!(score.notes[0].part, PartId(0));
        assert_eq!(score.notes[0].role, VoiceRole::Unspecified);
    }

    #[test]
    fn motif_evidence_records_transposed_interval_matches_and_exact_timing() {
        let pitches = [60_u8, 62, 64, 65, 67, 69, 62, 64, 66, 67, 69, 71];
        let mut track = Vec::new();
        for pitch in pitches {
            track.extend_from_slice(&[0, 0x90, pitch, 100, 100, 0x80, pitch, 0]);
        }
        track.extend_from_slice(&[0, 0xFF, 0x2F, 0]);
        let score = parse_midi(&midi_file(&track)).unwrap();
        let analysis = analyze(&score);
        assert_eq!(analysis.motifs.len(), 1);
        let motif = &analysis.motifs[0];

        assert_eq!(motif.pitch_interval_pattern, vec![2, 2, 1, 2, 2]);
        assert_eq!(motif.occurrence_count, 2);
        assert_eq!(motif.occurrence_evidence_complete, Some(true));
        assert_eq!(motif.matched_occurrences.len(), 2);
        assert_eq!(motif.onset_deltas.len(), 5);
        assert!(
            motif
                .onset_deltas
                .iter()
                .all(|delta| delta.numerator == 5 && delta.denominator == 24)
        );

        let first = &motif.matched_occurrences[0].notes;
        let transposed = &motif.matched_occurrences[1].notes;
        assert_eq!(
            first
                .iter()
                .map(|note| note.score_note_index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 5]
        );
        assert_eq!(
            transposed
                .iter()
                .map(|note| note.score_note_index)
                .collect::<Vec<_>>(),
            vec![6, 7, 8, 9, 10, 11]
        );
        assert_eq!(first[0].midi_pitch, 60);
        assert_eq!(transposed[0].midi_pitch, 62);
        assert_eq!(first[0].onset_numerator, 0);
        assert_eq!(first[0].onset_denominator, 1);
        assert_eq!(first[0].duration_numerator, 5);
        assert_eq!(first[0].duration_denominator, 24);
    }

    #[test]
    fn near_matches_are_unavailable_for_two_note_provisional_contours() {
        let track = monophonic_midi_track(&[60_u8, 62], &[]);
        let score = parse_midi(&midi_file(&track)).unwrap();
        let analysis = analyze(&score);
        assert_eq!(analysis.motifs.len(), 1);
        assert_eq!(analysis.motifs[0].midi_pitches, vec![60, 62]);
        assert!(analysis.motifs[0].rhythm_near_matches.is_none());
    }

    #[test]
    fn rhythm_aware_near_matches_expose_pitch_tolerance_and_reject_direction_changes() {
        let pitches = [
            60_u8, 62, 64, 65, 60, 62, 64, 65, // exact contour repeated twice
            70, 72, 75, 76, // +2,+3,+1: one interval differs by one semitone
            70, 72, 75, 74, // +2,+3,-1: direction changes and must be rejected
        ];
        let track = monophonic_midi_track(&pitches, &[]);
        let score = parse_midi(&midi_file(&track)).unwrap();
        let analysis = analyze(&score);
        assert_eq!(analysis.motifs.len(), 1);
        let motif = &analysis.motifs[0];
        assert_eq!(motif.pitch_interval_pattern, vec![2, 2, 1]);

        let near = motif
            .rhythm_near_matches
            .as_ref()
            .expect("valid timing yields a near-match scan");
        assert_eq!(near.max_allowed_pitch_interval_deviation_semitones, 1);
        assert!((near.max_allowed_normalized_onset_gap_error - 0.15).abs() < 1e-12);
        assert_eq!(near.occurrence_count, 1);
        assert_eq!(near.independent_occurrence_count, 1);
        assert!(near.occurrence_evidence_complete);
        assert_eq!(
            near.occurrences[0]
                .notes
                .iter()
                .map(|note| note.score_note_index)
                .collect::<Vec<_>>(),
            vec![8, 9, 10, 11]
        );
        assert_eq!(
            near.occurrences[0].observed_max_pitch_interval_deviation_semitones,
            1
        );
        assert!(near.occurrences[0].observed_max_normalized_onset_gap_error < 1e-12);
    }

    #[test]
    fn rhythm_near_matches_do_not_duplicate_exact_contours_with_changed_rhythm() {
        let pitches = [60_u8, 62, 64, 65, 60, 62, 64, 65, 70, 72, 74, 75];
        let mut extra_rest_before_note = vec![0_u8; pitches.len()];
        extra_rest_before_note[9] = 100;
        let track = monophonic_midi_track(&pitches, &extra_rest_before_note);
        let score = parse_midi(&midi_file(&track)).unwrap();
        let analysis = analyze(&score);
        assert_eq!(analysis.motifs.len(), 1);
        let motif = &analysis.motifs[0];

        // Exact contour matching intentionally ignores rhythmic shape.
        assert_eq!(motif.pitch_interval_pattern, vec![2, 2, 1]);
        assert_eq!(motif.occurrence_count, 3);
        let near = motif
            .rhythm_near_matches
            .as_ref()
            .expect("valid timing yields a near-match scan");
        assert_eq!(near.occurrence_count, 0);
        assert!(near.occurrences.is_empty());
    }

    #[test]
    fn rhythm_aware_near_matches_reject_gap_shape_outside_explicit_tolerance() {
        let pitches = [
            60_u8, 62, 64, 65, 60, 62, 64, 65, // exact contour repeated twice
            70, 72, 75, 76, // pitch-near, but 70->72 is rhythmically stretched
        ];
        let mut extra_rest_before_note = vec![0_u8; pitches.len()];
        // Adds 100 ticks to the second candidate's first internal onset gap:
        // normalized gaps become [1/2, 1/4, 1/4]; max share error is 1/6 (~0.167), above 0.15.
        extra_rest_before_note[9] = 100;
        let track = monophonic_midi_track(&pitches, &extra_rest_before_note);
        let score = parse_midi(&midi_file(&track)).unwrap();
        let analysis = analyze(&score);
        assert_eq!(analysis.motifs.len(), 1);
        let near = analysis.motifs[0]
            .rhythm_near_matches
            .as_ref()
            .expect("valid timing yields a near-match scan");
        assert_eq!(near.max_allowed_normalized_onset_gap_error, 0.15);
        assert_eq!(near.occurrence_count, 0);
        assert_eq!(near.independent_occurrence_count, 0);
        assert!(near.occurrence_evidence_complete);
        assert!(near.occurrences.is_empty());
    }

    #[test]
    fn motif_search_finds_a_repeated_contour_after_an_unrelated_opening() {
        let pitches = [60_u8, 61, 64, 66, 63, 62, 60, 62, 64, 65, 67, 69];
        let mut track = Vec::new();
        for pitch in pitches {
            track.extend_from_slice(&[0, 0x90, pitch, 100, 100, 0x80, pitch, 0]);
        }
        track.extend_from_slice(&[0, 0xFF, 0x2F, 0]);
        let score = parse_midi(&midi_file(&track)).unwrap();
        let analysis = analyze(&score);
        assert_eq!(analysis.motifs.len(), 1);
        let summary = &analysis.motifs[0];

        assert_eq!(summary.midi_pitches, vec![60, 62, 64]);
        assert_eq!(summary.pitch_interval_pattern, vec![2, 2]);
        assert_eq!(summary.occurrence_count, 2);
        assert_eq!(summary.matched_occurrences.len(), 2);
        assert_eq!(
            summary.matched_occurrences[0]
                .notes
                .iter()
                .map(|note| note.score_note_index)
                .collect::<Vec<_>>(),
            vec![6, 7, 8]
        );
        assert_eq!(
            summary.matched_occurrences[1]
                .notes
                .iter()
                .map(|note| note.score_note_index)
                .collect::<Vec<_>>(),
            vec![9, 10, 11]
        );
        assert!(summary.identity_note.contains("Longest repeated"));
    }

    #[test]
    fn overlapping_contour_windows_do_not_count_as_repetitions() {
        // A five-note ascending run has several matching sliding windows, but
        // no two candidates can be independent because every candidate overlaps.
        let pitches = [60_u8, 61, 62, 63, 64];
        let mut track = Vec::new();
        for pitch in pitches {
            track.extend_from_slice(&[0, 0x90, pitch, 100, 100, 0x80, pitch, 0]);
        }
        track.extend_from_slice(&[0, 0xFF, 0x2F, 0]);
        let score = parse_midi(&midi_file(&track)).unwrap();
        let analysis = analyze(&score);
        assert_eq!(analysis.motifs.len(), 1);
        let summary = &analysis.motifs[0];

        assert_eq!(summary.midi_pitches, vec![60, 61, 62, 63, 64]);
        assert_eq!(summary.occurrence_count, 1);
        assert_eq!(summary.independent_occurrence_count, Some(1));
        assert_eq!(summary.matched_occurrences.len(), 1);
        assert!(summary.identity_note.contains("no non-overlapping repeat"));
    }

    #[test]
    fn overlapping_matches_remain_visible_but_are_counted_separately() {
        let pitches = [60_u8, 61, 62, 63, 64, 65, 66, 67];
        let mut track = Vec::new();
        for pitch in pitches {
            track.extend_from_slice(&[0, 0x90, pitch, 100, 100, 0x80, pitch, 0]);
        }
        track.extend_from_slice(&[0, 0xFF, 0x2F, 0]);
        let score = parse_midi(&midi_file(&track)).unwrap();
        let analysis = analyze(&score);
        assert_eq!(analysis.motifs.len(), 1);
        let summary = &analysis.motifs[0];

        // Five exact four-note contour windows overlap; only starts 0 and 4
        // are independent occurrences. The raw evidence count is not discarded.
        assert_eq!(summary.midi_pitches, vec![60, 61, 62, 63]);
        assert_eq!(summary.occurrence_count, 5);
        assert_eq!(summary.independent_occurrence_count, Some(2));
        assert_eq!(summary.matched_occurrences.len(), 5);
    }

    #[test]
    fn motif_occurrence_evidence_is_bounded_and_reports_truncation() {
        let motif = [60_u8, 62, 64, 65, 67, 69];
        let mut track = Vec::new();
        for _ in 0..130 {
            for pitch in motif {
                track.extend_from_slice(&[0, 0x90, pitch, 100, 100, 0x80, pitch, 0]);
            }
        }
        track.extend_from_slice(&[0, 0xFF, 0x2F, 0]);
        let score = parse_midi(&midi_file(&track)).unwrap();
        let analysis = analyze(&score);
        assert_eq!(analysis.motifs.len(), 1);
        let summary = &analysis.motifs[0];
        assert_eq!(summary.occurrence_count, 130);
        assert_eq!(
            summary.matched_occurrences.len(),
            MAX_MOTIF_OCCURRENCE_EVIDENCE
        );
        assert_eq!(summary.occurrence_evidence_complete, Some(false));
    }

    #[test]
    fn analysis_withholds_motif_for_overlapping_notes_in_one_source_part() {
        let track = [
            0, 0x90, 60, 100, // first note starts at tick zero
            0, 0x90, 64, 100, // second note overlaps it in the same part
            100, 0x80, 60, 0, 0, 0x80, 64, 0, 0, 0xFF, 0x2F, 0,
        ];
        let score = parse_midi(&midi_file(&track)).unwrap();
        let analysis = analyze(&score);
        assert_eq!(analysis.voice_count, 1);
        assert!(analysis.motifs.is_empty());
        assert!(
            analysis
                .unresolved_interpretations
                .iter()
                .any(|item| { item.contains("overlapping/polyphonic events") })
        );
    }

    #[test]
    fn midi_same_tick_note_on_and_note_off_are_rejected_not_lengthened() {
        let track = [0, 0x90, 60, 100, 0, 0x80, 60, 0, 0, 0xFF, 0x2F, 0];
        let error = parse_midi(&midi_file(&track)).unwrap_err();
        assert!(error.contains("must span at least one tick"), "{error}");
    }

    #[test]
    fn midi_unmatched_note_on_at_end_of_track_is_rejected() {
        let track = [0, 0x90, 60, 100, 0, 0xFF, 0x2F, 0];
        let error = parse_midi(&midi_file(&track)).unwrap_err();
        assert!(
            error.contains("ends with unmatched note-on events"),
            "{error}"
        );
    }

    #[test]
    fn midi_channel_ten_notes_are_rejected_instead_of_silently_dropped() {
        let pitched_track = [0, 0x90, 60, 100, 100, 0x80, 60, 0, 0, 0xFF, 0x2F, 0];
        let channel_ten_track = [0, 0x99, 35, 100, 100, 0x89, 35, 0, 0, 0xFF, 0x2F, 0];
        let tracks: [&[u8]; 2] = [&pitched_track, &channel_ten_track];
        let error = parse_midi(&midi_file_with_tracks(&tracks)).unwrap_err();
        assert!(
            error.contains("channel 10 note events are outside the pitched-note importer"),
            "{error}"
        );
    }

    #[test]
    fn midi_program_change_is_rejected_instead_of_silently_dropped() {
        let track = [
            0, 0xC0, 5, // program 5 changes the channel instrument
            0, 0x90, 60, 100, 1, 0x80, 60, 0, 0, 0xFF, 0x2F, 0,
        ];
        let error = parse_midi(&midi_file(&track)).unwrap_err();
        assert!(error.contains("unsupported playback event"), "{error}");
    }

    #[test]
    fn midi_sustain_controller_is_rejected_instead_of_shortening_notes() {
        let track = [
            0, 0xB0, 64, 127, // sustain pedal down
            0, 0x90, 60, 100, 1, 0x80, 60, 0, 0, 0xFF, 0x2F, 0,
        ];
        let error = parse_midi(&midi_file(&track)).unwrap_err();
        assert!(error.contains("unsupported playback event"), "{error}");
    }

    #[test]
    fn midi_pitch_bend_is_rejected_instead_of_flattening_sounding_pitch() {
        let track = [
            0, 0xE0, 0, 127, // non-neutral pitch bend
            0, 0x90, 60, 100, 1, 0x80, 60, 0, 0, 0xFF, 0x2F, 0,
        ];
        let error = parse_midi(&midi_file(&track)).unwrap_err();
        assert!(error.contains("unsupported playback event"), "{error}");
    }

    #[test]
    fn midi_port_routing_is_rejected_before_cross_track_note_pairing() {
        let track = [
            0, 0xFF, 0x21, 1, 1, // select non-default MIDI port
            0, 0x90, 60, 100, 1, 0x80, 60, 0, 0, 0xFF, 0x2F, 0,
        ];
        let error = parse_midi(&midi_file(&track)).unwrap_err();
        assert!(
            error.contains("MIDI port-routing metadata is not supported"),
            "{error}"
        );
    }

    #[test]
    fn midi_lyrics_are_rejected_until_score_preserves_sung_text() {
        let track = [
            0, 0xFF, 0x05, 4, b'H', b'e', b'l', b'o', // lyric meta-event
            0, 0x90, 60, 100, 1, 0x80, 60, 0, 0, 0xFF, 0x2F, 0,
        ];
        let error = parse_midi(&midi_file(&track)).unwrap_err();
        assert!(
            error.contains("lyric events are not represented by Score"),
            "{error}"
        );
    }

    #[test]
    fn midi_smpte_start_offset_is_rejected_in_beat_relative_score() {
        let track = [
            0, 0xFF, 0x54, 5, 0, 0, 0, 0, 0, // zero-valued valid SMPTE offset
            0, 0x90, 60, 100, 1, 0x80, 60, 0, 0, 0xFF, 0x2F, 0,
        ];
        let error = parse_midi(&midi_file(&track)).unwrap_err();
        assert!(
            error.contains("MIDI SMPTE start offsets are not supported"),
            "{error}"
        );
    }

    #[test]
    fn midi_unknown_meta_event_is_rejected_instead_of_dropped() {
        let track = [
            0, 0xFF, 0x70, 1, 0x7D, // unknown meta-event 0x70
            0, 0x90, 60, 100, 1, 0x80, 60, 0, 0, 0xFF, 0x2F, 0,
        ];
        let error = parse_midi(&midi_file(&track)).unwrap_err();
        assert!(
            error.contains("MIDI unknown meta-event type 0x70"),
            "{error}"
        );
    }

    #[test]
    fn midi_system_exclusive_events_are_rejected_instead_of_dropped() {
        let track = [
            0, 0xF0, 2, 0x7D, 0xF7, // vendor-specific SysEx payload
            0, 0x90, 60, 100, 1, 0x80, 60, 0, 0, 0xFF, 0x2F, 0,
        ];
        let error = parse_midi(&midi_file(&track)).unwrap_err();
        assert!(
            error.contains("MIDI System Exclusive events are not supported"),
            "{error}"
        );
    }

    #[test]
    fn midi_escape_events_are_rejected_instead_of_dropped() {
        let track = [
            0, 0xF7, 1, 0x01, // arbitrary synthesizer escape payload
            0, 0x90, 60, 100, 1, 0x80, 60, 0, 0, 0xFF, 0x2F, 0,
        ];
        let error = parse_midi(&midi_file(&track)).unwrap_err();
        assert!(
            error.contains("MIDI escape events are not supported"),
            "{error}"
        );
    }

    #[test]
    fn midi_multiple_note_channels_in_one_track_are_rejected_instead_of_merged() {
        let track = [
            0, 0x90, 60, 100, // channel 1 note-on
            0, 0x91, 67, 100, // channel 2 note-on
            1, 0x80, 60, 0, 0, 0x81, 67, 0, 0, 0xFF, 0x2F, 0,
        ];
        let error = parse_midi(&midi_file(&track)).unwrap_err();
        assert!(error.contains("contains multiple note channels"), "{error}");
    }

    #[test]
    fn midi_note_on_and_note_off_pair_across_format_one_tracks() {
        let note_on_track = [0, 0x90, 60, 100, 0, 0xFF, 0x2F, 0];
        let note_off_track = [100, 0x80, 60, 0, 0, 0xFF, 0x2F, 0];
        let score = parse_midi(&midi_file_with_tracks(&[&note_off_track, &note_on_track])).unwrap();
        assert_eq!(score.notes.len(), 1);
        assert_eq!(score.notes[0].part, PartId(1));
        assert_eq!(score.notes[0].onset, Duration::zero());
        assert_eq!(score.notes[0].duration, Duration::new(100, 480));
    }

    #[test]
    fn midi_note_off_without_active_matching_note_is_rejected() {
        let track = [0, 0x80, 60, 0, 0, 0xFF, 0x2F, 0];
        let error = parse_midi(&midi_file(&track)).unwrap_err();
        assert!(error.contains("no matching active note-on"), "{error}");
    }

    #[test]
    fn midi_format_two_sequential_patterns_are_not_flattened_into_one_score() {
        let first = [0, 0x90, 60, 100, 100, 0x80, 60, 0, 0, 0xFF, 0x2F, 0];
        let second = [0, 0x90, 67, 100, 100, 0x80, 67, 0, 0, 0xFF, 0x2F, 0];
        let tracks: [&[u8]; 2] = [&first, &second];
        let error = parse_midi(&midi_file_with_format(2, &tracks)).unwrap_err();
        assert!(error.contains("sequential independent patterns"), "{error}");
    }

    #[test]
    fn malformed_midi_format_zero_with_multiple_tracks_is_rejected() {
        let first = [0, 0x90, 60, 100, 100, 0x80, 60, 0, 0, 0xFF, 0x2F, 0];
        let second = [0, 0x90, 67, 100, 100, 0x80, 67, 0, 0, 0xFF, 0x2F, 0];
        let tracks: [&[u8]; 2] = [&first, &second];
        let error = parse_midi(&midi_file_with_format(0, &tracks)).unwrap_err();
        assert!(error.contains("requires exactly one track"), "{error}");
    }

    #[test]
    fn musicxml_sound_dynamics_apply_to_subsequent_notes() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <direction><sound dynamics="71"/></direction>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
            <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let score = parse_musicxml(xml).unwrap();
        assert_eq!(score.notes.len(), 2);
        assert!(
            score
                .notes
                .iter()
                .all(|note| { (note.velocity - 64.0_f32 / 127.0).abs() < 1e-7 })
        );
    }

    #[test]
    fn musicxml_note_dynamics_override_part_playback_dynamics() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <direction><sound dynamics="50"/></direction>
            <note dynamics="71"><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
            <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let score = parse_musicxml(xml).unwrap();
        assert_eq!(score.notes.len(), 2);
        assert!((score.notes[0].velocity - 64.0_f32 / 127.0).abs() < 1e-7);
        assert!((score.notes[1].velocity - 45.0_f32 / 127.0).abs() < 1e-7);
    }

    #[test]
    fn musicxml_sound_dynamics_are_scoped_to_their_part() {
        let xml = br#"<score-partwise>
            <part id="P1"><measure number="1">
                <attributes><divisions>1</divisions></attributes>
                <direction><sound dynamics="71"/></direction>
                <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
            </measure></part>
            <part id="P2"><measure number="1">
                <attributes><divisions>1</divisions></attributes>
                <note><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration></note>
            </measure></part>
        </score-partwise>"#;
        let score = parse_musicxml(xml).unwrap();
        assert_eq!(score.notes.len(), 2);
        assert!((score.notes[0].velocity - 64.0_f32 / 127.0).abs() < 1e-7);
        assert!((score.notes[1].velocity - 90.0_f32 / 127.0).abs() < 1e-7);
    }

    #[test]
    fn musicxml_invalid_sound_dynamics_are_rejected() {
        for value in ["-1", "NaN", "inf", "not-a-number"] {
            let xml = format!(
                r#"<score-partwise><part id="P1"><measure number="1">
                    <attributes><divisions>1</divisions></attributes>
                    <direction><sound dynamics="{value}"/></direction>
                    <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
                </measure></part></score-partwise>"#
            );
            let error = parse_musicxml(xml.as_bytes()).unwrap_err();
            assert!(
                error.contains("MusicXML sound dynamics must be"),
                "{value}: {error}"
            );
        }
    }

    #[test]
    fn musicxml_sound_dynamics_outside_velocity_range_are_rejected() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <direction><sound dynamics="150"/></direction>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(
            error.contains("exceeds the representable MIDI velocity range"),
            "{error}"
        );
    }

    #[test]
    fn musicxml_unsupported_sound_playback_semantics_are_rejected() {
        for directive in [r#"pizzicato="yes""#, r#"pan="45""#, r#"damper-pedal="yes""#] {
            let xml = format!(
                r#"<score-partwise><part id="P1"><measure number="1">
                    <attributes><divisions>1</divisions></attributes>
                    <direction><sound {directive}/></direction>
                    <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
                </measure></part></score-partwise>"#
            );
            let error = parse_musicxml(xml.as_bytes()).unwrap_err();
            assert!(
                error.contains("tempo/dynamics-only sound directives are supported"),
                "{directive}: {error}"
            );
        }
    }

    #[test]
    fn musicxml_sound_instrument_or_play_child_is_rejected() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <direction><sound><midi-instrument id="P1-I1"><midi-program>1</midi-program></midi-instrument></sound></direction>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(
            error.contains("<midi-instrument> routing/program metadata is not represented by Score"),
            "{error}"
        );
    }

    #[test]
    fn musicxml_note_time_only_repeat_selection_is_rejected() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <note time-only="2"><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(
            error.contains("end-dynamics, attack, release, pizzicato, or time-only"),
            "{error}"
        );
    }

    #[test]
    fn musicxml_note_instrument_child_is_rejected() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <note><instrument id="P1-I1"/><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(
            error.contains("unsupported instrument or playback-technique child"),
            "{error}"
        );
    }

    #[test]
    fn musicxml_unsupported_note_playback_attributes_are_rejected() {
        for attribute in [
            r#"end-dynamics="71""#,
            r#"attack="1""#,
            r#"release="1""#,
            r#"pizzicato="yes""#,
        ] {
            let xml = format!(
                r#"<score-partwise><part id="P1"><measure number="1">
                    <attributes><divisions>1</divisions></attributes>
                    <note {attribute}><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
                </measure></part></score-partwise>"#
            );
            let error = parse_musicxml(xml.as_bytes()).unwrap_err();
            assert!(
                error.contains("current Score model cannot represent"),
                "{attribute}: {error}"
            );
        }
    }

    #[test]
    fn musicxml_default_note_velocity_matches_midi_forte_90() {
        let score = parse_musicxml(minimal_musicxml()).unwrap();
        assert!(
            score
                .notes
                .iter()
                .all(|note| (note.velocity - 90.0_f32 / 127.0).abs() < 1e-7),
            "MusicXML default velocity should map to MIDI forte 90"
        );
    }

    #[test]
    fn musicxml_sound_level_nonzero_tempo_offset_is_rejected() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <direction>
                <offset>0</offset>
                <sound tempo="90"><offset>1</offset></sound>
            </direction>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(
            error.contains("non-zero sound or direction tempo offsets are not supported"),
            "{error}"
        );
    }

    #[test]
    fn musicxml_zero_sound_tempo_offset_overrides_direction_offset() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <direction>
                <offset>1</offset>
                <sound tempo="90"><offset>0</offset></sound>
            </direction>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let score = parse_musicxml(xml).unwrap();
        assert_eq!(score.tempo_bpm, 90.0);
    }

    #[test]
    fn musicxml_unpitched_note_is_rejected_in_mixed_pitched_score() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
            <note><unpitched><display-step>F</display-step><display-octave>4</display-octave></unpitched><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(
            error.contains("unpitched notes are outside the pitched-note importer"),
            "{error}"
        );
    }

    #[test]
    fn musicxml_note_with_both_pitch_and_rest_is_rejected() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <note>
                <pitch><step>C</step><octave>4</octave></pitch>
                <rest/><duration>1</duration>
            </note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(
            error.contains("exactly one of <pitch>, <unpitched>, or <rest>"),
            "{error}"
        );
    }

    #[test]
    fn musicxml_note_with_multiple_pitch_elements_is_rejected() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <note>
                <pitch><step>C</step><octave>4</octave></pitch>
                <pitch><step>D</step><octave>4</octave></pitch>
                <duration>1</duration>
            </note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(
            error.contains("exactly one of <pitch>, <unpitched>, or <rest>"),
            "{error}"
        );
    }

    #[test]
    fn musicxml_non_rest_note_without_pitch_is_rejected() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <note><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(
            error.contains("must contain exactly one of <pitch>, <unpitched>, or <rest>"),
            "{error}"
        );
    }

    #[test]
    fn musicxml_part_transposition_is_rejected_instead_of_ignored() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes>
                <divisions>1</divisions>
                <transpose><diatonic>-1</diatonic><chromatic>-2</chromatic></transpose>
            </attributes>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(
            error.contains("part transposition is not supported"),
            "{error}"
        );
    }

    #[test]
    fn musicxml_tied_note_segments_are_rejected_instead_of_reattacked() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <note>
                <pitch><step>C</step><octave>4</octave></pitch><duration>1</duration>
                <tie type="start"/>
                <notations><tied type="start"/></notations>
            </note>
            <note>
                <pitch><step>C</step><octave>4</octave></pitch><duration>1</duration>
                <tie type="stop"/>
                <notations><tied type="stop"/></notations>
            </note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(
            error.contains("ties are not supported by the single-note-event importer"),
            "{error}"
        );
    }

    #[test]
    fn musicxml_repeat_markers_are_rejected_instead_of_flattened() {
        let xml = String::from_utf8(minimal_musicxml().to_vec())
            .unwrap()
            .replace(
                "</measure>",
                r#"<barline location="right"><repeat direction="backward"/></barline></measure>"#,
            );
        let error = parse_musicxml(xml.as_bytes()).unwrap_err();
        assert!(error.contains("<repeat>"), "{error}");
    }

    #[test]
    fn musicxml_alternate_endings_are_rejected_instead_of_flattened() {
        let xml = String::from_utf8(minimal_musicxml().to_vec())
            .unwrap()
            .replace(
                "</measure>",
                r#"<barline location="left"><ending number="1" type="start">1.</ending></barline></measure>"#,
            );
        let error = parse_musicxml(xml.as_bytes()).unwrap_err();
        assert!(error.contains("<ending>"), "{error}");
    }

    #[test]
    fn musicxml_multiple_measure_rests_are_rejected_to_preserve_timing() {
        let xml = String::from_utf8(minimal_musicxml().to_vec())
            .unwrap()
            .replace(
                "</attributes>",
                "<measure-style><multiple-rest>2</multiple-rest></measure-style></attributes>",
            );
        let error = parse_musicxml(xml.as_bytes()).unwrap_err();
        assert!(error.contains("<multiple-rest>"), "{error}");
    }

    #[test]
    fn musicxml_articulation_is_rejected_instead_of_flattened() {
        let xml = String::from_utf8(minimal_musicxml().to_vec())
            .unwrap()
            .replace(
                "</note>",
                "<notations><articulations><staccato/></articulations></notations></note>",
            );
        let error = parse_musicxml(xml.as_bytes()).unwrap_err();
        assert!(error.contains("<staccato>"), "{error}");
    }

    #[test]
    fn musicxml_notated_dynamics_are_rejected_until_velocity_mapping_exists() {
        let xml = String::from_utf8(minimal_musicxml().to_vec())
            .unwrap()
            .replace(
                "<note><pitch>",
                "<direction><direction-type><dynamics><f/></dynamics></direction-type></direction><note><pitch>",
            );
        let error = parse_musicxml(xml.as_bytes()).unwrap_err();
        assert!(
            error.contains("notated <dynamics> markings are not mapped"),
            "{error}"
        );
    }

    #[test]
    fn musicxml_midi_instrument_mapping_is_rejected_until_score_can_preserve_it() {
        let xml = String::from_utf8(minimal_musicxml().to_vec())
            .unwrap()
            .replace(
                "<part id=\"P1\">",
                "<part-list><score-part id=\"P1\"><part-name>Solo</part-name><midi-instrument id=\"P1-I1\"><midi-channel>1</midi-channel><midi-program>41</midi-program></midi-instrument></score-part></part-list><part id=\"P1\">",
            );
        let error = parse_musicxml(xml.as_bytes()).unwrap_err();
        assert!(
            error.contains("<midi-instrument> routing/program metadata"),
            "{error}"
        );
    }

    #[test]
    fn musicxml_multiple_voices_in_one_part_are_rejected_instead_of_merged() {
        let xml = String::from_utf8(minimal_musicxml().to_vec())
            .unwrap()
            .replace("<note><pitch>", "<note><voice>1</voice><pitch>")
            .replacen("<voice>1</voice><pitch>", "<voice>2</voice><pitch>", 1);
        let error = parse_musicxml(xml.as_bytes()).unwrap_err();
        assert!(error.contains("contains multiple voices"), "{error}");
    }

    #[test]
    fn musicxml_multiple_staves_in_one_part_are_rejected_instead_of_merged() {
        let xml = String::from_utf8(minimal_musicxml().to_vec())
            .unwrap()
            .replace(
                "</pitch><duration>1</duration>",
                "</pitch><duration>1</duration><staff>1</staff>",
            )
            .replacen("<staff>1</staff>", "<staff>2</staff>", 1);
        let error = parse_musicxml(xml.as_bytes()).unwrap_err();
        assert!(error.contains("contains multiple staves"), "{error}");
    }

    #[test]
    fn musicxml_lyrics_are_rejected_until_score_preserves_sung_text() {
        let xml = String::from_utf8(minimal_musicxml().to_vec())
            .unwrap()
            .replace(
                "</note>",
                "<lyric><syllabic>single</syllabic><text>Hello</text></lyric></note>",
            );
        let error = parse_musicxml(xml.as_bytes()).unwrap_err();
        assert!(
            error.contains("lyrics are not represented by Score"),
            "{error}"
        );
    }

    #[test]
    fn musicxml_timewise_root_is_rejected_before_parts_are_flattened() {
        let xml = br#"<score-timewise>
            <part-list><score-part id="P1"><part-name>Solo</part-name></score-part></part-list>
            <measure number="1"><part id="P1">
                <attributes><divisions>1</divisions></attributes>
                <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
            </part></measure>
            <measure number="2"><part id="P1">
                <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration></note>
            </part></measure>
        </score-timewise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(
            error.contains("score-timewise organization is not supported"),
            "{error}"
        );
    }

    #[test]
    fn non_score_musicxml_root_is_rejected_explicitly() {
        let error = parse_musicxml(br#"<root><part id="P1"/></root>"#).unwrap_err();
        assert!(
            error.contains("is not a supported score-partwise document"),
            "{error}"
        );
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
        assert!(
            score
                .notes
                .iter()
                .all(|note| note.role == VoiceRole::Unspecified)
        );
        let analysis = analyze(&score);
        assert!(analysis.source_native);
        assert!(analysis.inferred_territory.is_none());
        assert_eq!(analysis.voice_count, 1);
        assert_eq!(analysis.motifs.len(), 1);
        let motif = &analysis.motifs[0];
        assert_eq!(motif.source_part_id, Some(0));
        assert!(motif.identity_note.contains("part-local"));
        assert_eq!(motif.pitch_interval_pattern, vec![4]);
        assert_eq!(motif.occurrence_count, 1);
        assert_eq!(motif.occurrence_evidence_complete, Some(true));
        assert_eq!(motif.matched_occurrences.len(), 1);
        assert_eq!(motif.matched_occurrences[0].notes.len(), 2);
        assert_eq!(motif.matched_occurrences[0].notes[0].score_note_index, 0);
        assert_eq!(motif.matched_occurrences[0].notes[1].score_note_index, 1);
        assert_eq!(motif.onset_deltas.len(), 1);
        assert_eq!(motif.onset_deltas[0].numerator, 1);
        assert_eq!(motif.onset_deltas[0].denominator, 1);
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
        assert!(
            error.contains("outside the MIDI pitch range 0..=127"),
            "{error}"
        );
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
            <note><grace slash="yes"/><pitch><step>B</step><octave>4</octave></pitch></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(
            error.contains("grace notes are not yet supported"),
            "{error}"
        );
    }

    #[test]
    fn musicxml_malformed_divisions_are_rejected_instead_of_reusing_previous_value() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>abc</divisions></attributes>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(
            error.contains("divisions must be a positive integer"),
            "{error}"
        );
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
            b'M', b'T', b'h', b'd', 0, 0, 0, 6, 0, 0, 0, 1, 1, 0xE0, b'M', b'T', b'r', b'k', 0, 0,
            0, 20, 0, 0x90, 60, 100, 100, 0x90, 60, 80, 100, 0x80, 60, 0, 100, 0x80, 60, 0, 0,
            0xFF, 0x2F, 0,
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
    fn midi_tempo_validation_uses_absolute_tick_order_across_tracks() {
        // Track 0 is traversed first but contains the later tempo event.
        // Track 1 supplies the same tempo at tick zero, so the score is
        // constant-tempo despite traversal order.
        let later_track = [
            100, 0xFF, 0x51, 3, 0x06, 0x1A, 0x80, // 150 BPM at tick 100
            0, 0x90, 62, 100, 100, 0x80, 62, 0, 0, 0xFF, 0x2F, 0,
        ];
        let initial_track = [
            0, 0xFF, 0x51, 3, 0x06, 0x1A, 0x80, // 150 BPM at tick 0
            0, 0x90, 60, 100, 100, 0x80, 60, 0, 0, 0xFF, 0x2F, 0,
        ];
        let tracks: [&[u8]; 2] = [&later_track, &initial_track];
        let score = parse_midi(&midi_file_with_tracks(&tracks)).unwrap();
        assert_eq!(score.tempo_bpm, 150.0);
        assert_eq!(score.notes.len(), 2);
    }

    #[test]
    fn midi_late_nondefault_tempo_is_rejected_against_default_120_bpm() {
        let track = [
            100, 0xFF, 0x51, 3, 0x06, 0x1A, 0x80, // 150 BPM at tick 100
            0, 0x90, 60, 100, 100, 0x80, 60, 0, 0, 0xFF, 0x2F, 0,
        ];
        let error = parse_midi(&midi_file(&track)).unwrap_err();
        assert!(
            error.contains("changes from the default at a non-zero tick"),
            "{error}"
        );
    }

    #[test]
    fn midi_late_nondefault_key_is_rejected_without_opening_context() {
        let track = [
            100, 0xFF, 0x59, 2, 1, 0, // G major at tick 100
            0, 0x90, 60, 100, 100, 0x80, 60, 0, 0, 0xFF, 0x2F, 0,
        ];
        let error = parse_midi(&midi_file(&track)).unwrap_err();
        assert!(
            error.contains("first appears at a non-zero tick"),
            "{error}"
        );
    }

    #[test]
    fn midi_late_default_key_does_not_prove_opening_key_context() {
        let track = [
            100, 0xFF, 0x59, 2, 0, 0, // C major appears only at tick 100
            0, 0x90, 60, 100, 100, 0x80, 60, 0, 0, 0xFF, 0x2F, 0,
        ];
        let error = parse_midi(&midi_file(&track)).unwrap_err();
        assert!(
            error.contains("first appears at a non-zero tick"),
            "{error}"
        );
    }

    #[test]
    fn midi_late_nondefault_meter_is_rejected_against_default_four_four() {
        let track = [
            100, 0xFF, 0x58, 4, 3, 2, 24, 8, // 3/4 at tick 100
            0, 0x90, 60, 100, 100, 0x80, 60, 0, 0, 0xFF, 0x2F, 0,
        ];
        let error = parse_midi(&midi_file(&track)).unwrap_err();
        assert!(
            error.contains("changes from the default at a non-zero tick"),
            "{error}"
        );
    }

    #[test]
    fn midi_equal_tempo_events_at_zero_and_later_are_accepted() {
        let track = [
            0, 0xFF, 0x51, 3, 0x06, 0x1A, 0x80, // 150 BPM at tick 0
            100, 0xFF, 0x51, 3, 0x06, 0x1A, 0x80, // same value later
            0, 0x90, 60, 100, 100, 0x80, 60, 0, 0, 0xFF, 0x2F, 0,
        ];
        let score = parse_midi(&midi_file(&track)).unwrap();
        assert_eq!(score.tempo_bpm, 150.0);
    }

    #[test]
    fn midi_tempo_changes_are_rejected_instead_of_flattened() {
        let track = [
            0, 0xFF, 0x51, 3, 0x07, 0xA1, 0x20, // 120 BPM
            100, 0xFF, 0x51, 3, 0x06, 0x1A, 0x80, // 150 BPM
            0, 0x90, 60, 100, 100, 0x80, 60, 0, 0, 0xFF, 0x2F, 0,
        ];
        let error = parse_midi(&midi_file(&track)).unwrap_err();
        assert!(
            error.contains("tempo changes cannot be represented"),
            "{error}"
        );
    }

    #[test]
    fn midi_non_quarter_meter_is_rejected_instead_of_flattened() {
        let track = [
            0, 0xFF, 0x58, 4, 4, 3, 24, 8, // 4/8, denominator power 3
            0, 0x90, 60, 100, 100, 0x80, 60, 0, 0, 0xFF, 0x2F, 0,
        ];
        let error = parse_midi(&midi_file(&track)).unwrap_err();
        assert!(
            error.contains("denominators other than quarter notes"),
            "{error}"
        );
    }

    #[test]
    fn midi_key_changes_are_rejected_instead_of_flattened() {
        let track = [
            0, 0xFF, 0x59, 2, 0, 0, // C major
            100, 0xFF, 0x59, 2, 1, 0, // G major
            0, 0x90, 60, 100, 100, 0x80, 60, 0, 0, 0xFF, 0x2F, 0,
        ];
        let error = parse_midi(&midi_file(&track)).unwrap_err();
        assert!(
            error.contains("key changes cannot be represented"),
            "{error}"
        );
    }

    #[test]
    fn musicxml_sound_tempo_outside_supported_range_is_rejected() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <direction><direction-type><words>Tempo</words></direction-type><sound tempo="10"/></direction>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(
            error.contains("finite and between 20 and 320 BPM"),
            "{error}"
        );
    }

    #[test]
    fn musicxml_tempo_resolution_uses_earliest_position_across_parts() {
        // The first XML part has its only tempo mark at beat 1. A later part
        // supplies the same value at beat 0. Temporal resolution must not
        // mistake the earlier document traversal for the earlier music event.
        let xml = br#"<score-partwise>
            <part id="P1">
                <measure number="1">
                    <attributes><divisions>1</divisions></attributes>
                    <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
                </measure>
                <measure number="2">
                    <direction><direction-type><words>Tempo</words></direction-type><sound tempo="96"/></direction>
                    <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration></note>
                </measure>
            </part>
            <part id="P2">
                <measure number="1">
                    <attributes><divisions>1</divisions></attributes>
                    <direction><direction-type><words>Tempo</words></direction-type><sound tempo="96"/></direction>
                    <note><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration></note>
                </measure>
            </part>
        </score-partwise>"#;
        let score = parse_musicxml(xml).unwrap();
        assert_eq!(score.tempo_bpm, 96.0);
    }

    #[test]
    fn musicxml_late_default_tempo_does_not_prove_opening_tempo() {
        let xml = br#"<score-partwise><part id="P1">
            <measure number="1">
                <attributes><divisions>1</divisions></attributes>
                <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
            </measure>
            <measure number="2">
                <direction><direction-type><words>Tempo</words></direction-type><sound tempo="120"/></direction>
                <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration></note>
            </measure>
        </part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(
            error.contains("first appears at a non-zero beat"),
            "{error}"
        );
    }

    #[test]
    fn musicxml_late_default_key_does_not_prove_opening_key() {
        let xml = br#"<score-partwise><part id="P1">
            <measure number="1">
                <attributes><divisions>1</divisions></attributes>
                <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
            </measure>
            <measure number="2">
                <attributes><key><fifths>0</fifths><mode>major</mode></key></attributes>
                <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration></note>
            </measure>
        </part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(
            error.contains("first appears at a non-zero beat"),
            "{error}"
        );
    }

    #[test]
    fn musicxml_late_nondefault_tempo_is_rejected_against_default_120_bpm() {
        let xml = br#"<score-partwise><part id="P1">
            <measure number="1">
                <attributes><divisions>1</divisions></attributes>
                <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
            </measure>
            <measure number="2">
                <direction><sound tempo="90"/></direction>
                <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration></note>
            </measure>
        </part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(
            error.contains("first appears at a non-zero beat"),
            "{error}"
        );
    }

    #[test]
    fn musicxml_late_nondefault_key_is_rejected_against_default_c_major() {
        let xml = br#"<score-partwise><part id="P1">
            <measure number="1">
                <attributes><divisions>1</divisions></attributes>
                <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
            </measure>
            <measure number="2">
                <attributes><key><fifths>1</fifths><mode>major</mode></key></attributes>
                <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration></note>
            </measure>
        </part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(
            error.contains("first appears at a non-zero beat"),
            "{error}"
        );
    }

    #[test]
    fn musicxml_late_nondefault_meter_is_rejected_against_default_four_four() {
        let xml = br#"<score-partwise><part id="P1">
            <measure number="1">
                <attributes><divisions>1</divisions></attributes>
                <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
            </measure>
            <measure number="2">
                <attributes><time><beats>3</beats><beat-type>4</beat-type></time></attributes>
                <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration></note>
            </measure>
        </part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(
            error.contains("first appears at a non-zero beat"),
            "{error}"
        );
    }

    #[test]
    fn musicxml_nonzero_tempo_direction_offset_is_rejected() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <direction><offset>1</offset><sound tempo="90"/></direction>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(
            error.contains("non-zero sound or direction tempo offsets are not supported"),
            "{error}"
        );
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
        assert!(
            error.contains("beat units other than undotted quarter notes"),
            "{error}"
        );
    }

    #[test]
    fn musicxml_dotted_metronome_mark_is_rejected() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <direction><direction-type><metronome><beat-unit>quarter</beat-unit><beat-unit-dot/><per-minute>72</per-minute></metronome></direction-type></direction>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(
            error.contains("only undotted quarter-note metronome marks"),
            "{error}"
        );
    }

    #[test]
    fn musicxml_metric_modulation_metronome_is_rejected() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions></attributes>
            <direction><direction-type><metronome><metronome-note><metronome-type>eighth</metronome-type></metronome-note><metronome-relation>equals</metronome-relation><metronome-note><metronome-type>quarter</metronome-type><metronome-dot/></metronome-note></metronome></direction-type></direction>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(
            error.contains("only undotted quarter-note metronome marks"),
            "{error}"
        );
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
        assert!(
            error.contains("tempo changes cannot be represented"),
            "{error}"
        );
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
        assert!(
            error.contains("meter changes cannot be represented"),
            "{error}"
        );
    }

    #[test]
    fn musicxml_non_quarter_meter_is_rejected_instead_of_flattened() {
        let xml = br#"<score-partwise><part id="P1"><measure number="1">
            <attributes><divisions>1</divisions><time><beats>6</beats><beat-type>8</beat-type></time></attributes>
            <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note>
        </measure></part></score-partwise>"#;
        let error = parse_musicxml(xml).unwrap_err();
        assert!(
            error.contains("denominators other than quarter notes"),
            "{error}"
        );
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
        assert!(
            error.contains("key changes cannot be represented"),
            "{error}"
        );
    }

    #[test]
    fn zero_ticks_per_beat_midi_is_rejected_without_panicking() {
        let bytes = [
            b'M', b'T', b'h', b'd', 0, 0, 0, 6, 0, 0, 0, 1, 0, 0, b'M', b'T', b'r', b'k', 0, 0, 0,
            4, 0, 0xFF, 0x2F, 0,
        ];
        let error = parse_midi(&bytes).unwrap_err();
        assert!(error.contains("non-zero ticks-per-beat"), "{error}");
    }
}
