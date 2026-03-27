// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Musical score notation: MusicXML export and SVG staff rendering.
//!
//! Converts `Composition` note data into standard notation formats:
//! - **MusicXML**: Industry-standard interchange for notation software
//! - **Score SVG**: Visual staff notation rendered as SVG

use crate::{midi::freq_to_midi_note, Composition, Note};

/// Note name and octave from MIDI note number.
fn midi_to_name(midi: u8) -> (&'static str, i32) {
    let names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    let name = names[(midi % 12) as usize];
    let octave = (midi as i32 / 12) - 1;
    (name, octave)
}

/// MusicXML step and alter from MIDI note.
fn midi_to_musicxml_pitch(midi: u8) -> (char, i8, i32) {
    let steps = ['C', 'C', 'D', 'D', 'E', 'F', 'F', 'G', 'G', 'A', 'A', 'B'];
    let alters = [0i8, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0];
    let pc = (midi % 12) as usize;
    let octave = (midi as i32 / 12) - 1;
    (steps[pc], alters[pc], octave)
}

/// Convert duration in seconds to a MusicXML note type at a given tempo.
fn duration_to_type(dur_secs: f32, tempo_bpm: f32) -> (&'static str, u32) {
    let beat_secs = 60.0 / tempo_bpm;
    let beats = dur_secs / beat_secs;

    if beats >= 3.5 {
        ("whole", 4)
    } else if beats >= 1.75 {
        ("half", 2)
    } else if beats >= 0.875 {
        ("quarter", 1)
    } else if beats >= 0.4375 {
        ("eighth", 1)
    } else {
        ("16th", 1)
    }
}

/// Export a composition as MusicXML 4.0 (partwise format).
pub fn to_musicxml(comp: &Composition, tempo_bpm: f32) -> String {
    let mut xml = String::new();
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<!DOCTYPE score-partwise PUBLIC \"-//Recordare//DTD MusicXML 4.0 Partwise//EN\" \"http://www.musicxml.org/dtds/partwise.dtd\">\n");
    xml.push_str("<score-partwise version=\"4.0\">\n");

    // Part list
    xml.push_str("  <part-list>\n");
    xml.push_str("    <score-part id=\"P1\">\n");
    xml.push_str("      <part-name>Symthaea</part-name>\n");
    xml.push_str("    </score-part>\n");
    xml.push_str("  </part-list>\n");

    // Part with measures
    xml.push_str("  <part id=\"P1\">\n");

    let beat_secs = 60.0 / tempo_bpm;
    let beats_per_measure = 4;
    let measure_duration = beats_per_measure as f32 * beat_secs;

    // Group notes into measures
    let num_measures = ((comp.duration_secs / measure_duration).ceil() as usize).max(1);

    for measure_idx in 0..num_measures {
        let measure_start = measure_idx as f32 * measure_duration;
        let measure_end = measure_start + measure_duration;

        xml.push_str(&format!("    <measure number=\"{}\">\n", measure_idx + 1));

        // First measure: add attributes and tempo
        if measure_idx == 0 {
            xml.push_str("      <attributes>\n");
            xml.push_str("        <divisions>1</divisions>\n");
            xml.push_str("        <time>\n");
            xml.push_str("          <beats>4</beats>\n");
            xml.push_str("          <beat-type>4</beat-type>\n");
            xml.push_str("        </time>\n");
            xml.push_str("        <clef>\n");
            xml.push_str("          <sign>G</sign>\n");
            xml.push_str("          <line>2</line>\n");
            xml.push_str("        </clef>\n");
            xml.push_str("      </attributes>\n");
            xml.push_str("      <direction placement=\"above\">\n");
            xml.push_str("        <sound tempo=\"");
            xml.push_str(&format!("{:.0}", tempo_bpm));
            xml.push_str("\"/>\n");
            xml.push_str("      </direction>\n");
        }

        // Notes in this measure
        let measure_notes: Vec<&Note> = comp
            .notes
            .iter()
            .filter(|n| n.start_time >= measure_start && n.start_time < measure_end)
            .collect();

        for note in &measure_notes {
            let midi = freq_to_midi_note(note.frequency);
            let (step, alter, octave) = midi_to_musicxml_pitch(midi);
            let (note_type, duration) = duration_to_type(note.duration, tempo_bpm);

            xml.push_str("      <note>\n");
            xml.push_str("        <pitch>\n");
            xml.push_str(&format!("          <step>{step}</step>\n"));
            if alter != 0 {
                xml.push_str(&format!("          <alter>{alter}</alter>\n"));
            }
            xml.push_str(&format!("          <octave>{octave}</octave>\n"));
            xml.push_str("        </pitch>\n");
            xml.push_str(&format!("        <duration>{duration}</duration>\n"));
            xml.push_str(&format!("        <type>{note_type}</type>\n"));
            xml.push_str("      </note>\n");
        }

        // Add a rest if measure is empty
        if measure_notes.is_empty() {
            xml.push_str("      <note>\n");
            xml.push_str("        <rest/>\n");
            xml.push_str("        <duration>4</duration>\n");
            xml.push_str("        <type>whole</type>\n");
            xml.push_str("      </note>\n");
        }

        xml.push_str("    </measure>\n");
    }

    xml.push_str("  </part>\n");
    xml.push_str("</score-partwise>\n");
    xml
}

/// Render a simple SVG staff notation of the composition.
///
/// Produces a basic treble clef staff with note heads positioned by pitch.
/// Not a full notation renderer, but a visual artifact for the gallery.
pub fn to_score_svg(comp: &Composition, tempo_bpm: f32) -> String {
    let width = 800.0f32;
    let height = 200.0f32;
    let margin = 40.0;
    let staff_top = 60.0;
    let line_spacing = 10.0;

    let mut svg = String::new();
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">\n"
    ));
    svg.push_str(&format!(
        "<rect width=\"{width}\" height=\"{height}\" fill=\"#fffef0\"/>\n"
    ));

    // Draw 5 staff lines
    for i in 0..5 {
        let y = staff_top + i as f32 * line_spacing;
        svg.push_str(&format!(
            "<line x1=\"{margin}\" y1=\"{y}\" x2=\"{}\" y2=\"{y}\" stroke=\"#333\" stroke-width=\"0.8\"/>\n",
            width - margin
        ));
    }

    // Draw treble clef (simplified text glyph)
    svg.push_str(&format!(
        "<text x=\"{}\" y=\"{}\" font-size=\"40\" font-family=\"serif\" fill=\"#333\">&#x1D11E;</text>\n",
        margin + 5.0,
        staff_top + 35.0
    ));

    // Draw notes
    let note_start_x = margin + 50.0;
    let note_area = width - margin * 2.0 - 60.0;

    for note in &comp.notes {
        if comp.duration_secs <= 0.0 {
            continue;
        }

        // X position: proportional to time
        let x = note_start_x + (note.start_time / comp.duration_secs) * note_area;

        // Y position: map MIDI note to staff position
        // Middle C (60) = first ledger line below staff
        // Each semitone moves ~half a line_spacing
        let midi = freq_to_midi_note(note.frequency);
        let staff_position = (midi as f32 - 64.0) * (line_spacing / 2.0);
        let y = staff_top + 4.0 * line_spacing - staff_position;
        let y = y.clamp(staff_top - 20.0, staff_top + 60.0);

        // Note head (filled ellipse)
        let (note_type, _) = duration_to_type(note.duration, tempo_bpm);
        let fill = if note_type == "whole" || note_type == "half" {
            "none"
        } else {
            "#333"
        };
        svg.push_str(&format!(
            "<ellipse cx=\"{x:.1}\" cy=\"{y:.1}\" rx=\"5\" ry=\"3.5\" fill=\"{fill}\" stroke=\"#333\" stroke-width=\"1\"/>\n"
        ));

        // Stem (for non-whole notes)
        if note_type != "whole" {
            let stem_dir = if y > staff_top + 2.0 * line_spacing {
                -1.0 // stem up
            } else {
                1.0 // stem down
            };
            let stem_x = if stem_dir < 0.0 { x + 5.0 } else { x - 5.0 };
            let stem_y2 = y + stem_dir * 30.0;
            svg.push_str(&format!(
                "<line x1=\"{stem_x:.1}\" y1=\"{y:.1}\" x2=\"{stem_x:.1}\" y2=\"{stem_y2:.1}\" stroke=\"#333\" stroke-width=\"1\"/>\n"
            ));
        }
    }

    svg.push_str("</svg>\n");
    svg
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::SectionType;

    fn test_comp() -> Composition {
        Composition {
            audio: crate::AudioData::I16(vec![]),
            sample_rate: 44100,
            notes: vec![
                Note { frequency: 261.63, start_time: 0.0, duration: 0.5, velocity: 0.8 },
                Note { frequency: 329.63, start_time: 0.5, duration: 0.5, velocity: 0.7 },
                Note { frequency: 392.00, start_time: 1.0, duration: 1.0, velocity: 0.9 },
                Note { frequency: 523.25, start_time: 2.0, duration: 0.5, velocity: 0.6 },
            ],
            duration_secs: 3.0,
            section: SectionType::Developmental,
        }
    }

    #[test]
    fn musicxml_well_formed() {
        let xml = to_musicxml(&test_comp(), 120.0);
        assert!(xml.contains("<?xml"));
        assert!(xml.contains("<score-partwise"));
        assert!(xml.contains("</score-partwise>"));
    }

    #[test]
    fn musicxml_has_part() {
        let xml = to_musicxml(&test_comp(), 120.0);
        assert!(xml.contains("<part id=\"P1\">"));
        assert!(xml.contains("<part-name>Symthaea</part-name>"));
    }

    #[test]
    fn musicxml_has_notes() {
        let xml = to_musicxml(&test_comp(), 120.0);
        assert!(xml.contains("<note>"));
        assert!(xml.contains("<pitch>"));
        assert!(xml.contains("<step>"));
    }

    #[test]
    fn musicxml_has_tempo() {
        let xml = to_musicxml(&test_comp(), 120.0);
        assert!(xml.contains("tempo=\"120\""));
    }

    #[test]
    fn score_svg_valid() {
        let svg = to_score_svg(&test_comp(), 120.0);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn score_svg_has_staff_lines() {
        let svg = to_score_svg(&test_comp(), 120.0);
        // 5 staff lines
        let line_count = svg.matches("<line ").count();
        assert!(line_count >= 5, "should have at least 5 staff lines, got {line_count}");
    }

    #[test]
    fn score_svg_has_note_heads() {
        let svg = to_score_svg(&test_comp(), 120.0);
        let ellipse_count = svg.matches("<ellipse").count();
        assert_eq!(
            ellipse_count,
            test_comp().notes.len(),
            "should have one note head per note"
        );
    }

    #[test]
    fn midi_to_name_works() {
        let (name, octave) = midi_to_name(60);
        assert_eq!(name, "C");
        assert_eq!(octave, 4);

        let (name, octave) = midi_to_name(69);
        assert_eq!(name, "A");
        assert_eq!(octave, 4);
    }

    #[test]
    fn duration_types() {
        let (t, _) = duration_to_type(2.0, 120.0); // 2s at 120bpm = 4 beats = whole
        assert_eq!(t, "whole");

        let (t, _) = duration_to_type(0.5, 120.0); // 0.5s at 120bpm = 1 beat = quarter
        assert_eq!(t, "quarter");

        let (t, _) = duration_to_type(0.25, 120.0); // 0.25s at 120bpm = 0.5 beat = eighth
        assert_eq!(t, "eighth");
    }

    #[test]
    fn round_trip_compose_to_notation() {
        let config = crate::MuseConfig {
            duration_secs: 2.0,
            max_notes: 8,
            ..Default::default()
        };
        let state = crate::MusicalState::default();
        let comp = crate::compose(&config, &state, 42);
        let xml = to_musicxml(&comp, 120.0);
        let svg = to_score_svg(&comp, 120.0);
        assert!(xml.contains("<score-partwise"));
        assert!(svg.contains("<svg"));
    }
}
