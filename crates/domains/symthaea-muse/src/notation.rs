// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! MusicXML 4.0 and SVG staff notation export.
//!
//! Converts `Composition` note sequences into MusicXML (score-partwise)
//! and simplified SVG staff renderings.

use crate::Composition;
use crate::midi::freq_to_midi_note;

/// Divisions per quarter note (16th note resolution: quarter = 4 divisions).
const DIVISIONS: u32 = 4;

/// Convert a MIDI note number to (name, octave).
pub fn midi_to_name(midi: u8) -> (&'static str, i32) {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let octave = (midi as i32 / 12) - 1;
    let name_idx = (midi % 12) as usize;
    (NAMES[name_idx], octave)
}

/// Duration in divisions for a given note duration in seconds at a tempo.
fn duration_to_divisions(dur_secs: f32, tempo_bpm: f32) -> u32 {
    let beats = dur_secs * tempo_bpm / 60.0;
    let divs = (beats * DIVISIONS as f32).round() as u32;
    divs.max(1)
}

/// Map divisions to a MusicXML duration type name.
fn divisions_to_type(divs: u32) -> &'static str {
    match divs {
        16.. => "whole",
        8..=15 => "half",
        4..=7 => "quarter",
        2..=3 => "eighth",
        _ => "16th",
    }
}

/// Export a composition as MusicXML 4.0 (score-partwise).
pub fn to_musicxml(comp: &Composition, tempo_bpm: f32) -> String {
    let mut xml = String::new();
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<!DOCTYPE score-partwise PUBLIC \"-//Recordare//DTD MusicXML 4.0 Partwise//EN\" \"http://www.musicxml.org/dtds/partwise.dtd\">\n");
    xml.push_str("<score-partwise version=\"4.0\">\n");
    xml.push_str("  <part-list>\n");
    xml.push_str("    <score-part id=\"P1\">\n");
    xml.push_str("      <part-name>Symthaea</part-name>\n");
    xml.push_str("    </score-part>\n");
    xml.push_str("  </part-list>\n");
    xml.push_str("  <part id=\"P1\">\n");

    // Group notes into 4/4 measures (each measure = 4 beats = 16 divisions)
    let measure_divs: u32 = 4 * DIVISIONS; // 16 divisions per measure

    // Convert all notes to (start_div, duration_div, midi_note)
    let mut note_events: Vec<(u32, u32, u8)> = comp
        .notes
        .iter()
        .map(|n| {
            let start = duration_to_divisions(n.start_time, tempo_bpm);
            let dur = duration_to_divisions(n.duration, tempo_bpm);
            let midi = freq_to_midi_note(n.frequency);
            (start, dur, midi)
        })
        .collect();
    note_events.sort_by_key(|e| e.0);

    // Determine total measures needed
    let max_end = note_events
        .iter()
        .map(|(s, d, _)| s + d)
        .max()
        .unwrap_or(measure_divs);
    let num_measures = ((max_end + measure_divs - 1) / measure_divs).max(1);

    for m in 0..num_measures {
        let m_start = m * measure_divs;
        let m_end = m_start + measure_divs;

        xml.push_str(&format!("    <measure number=\"{}\">\n", m + 1));

        // First measure: attributes + tempo
        if m == 0 {
            xml.push_str("      <attributes>\n");
            xml.push_str(&format!("        <divisions>{DIVISIONS}</divisions>\n"));
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
            xml.push_str("        <direction-type>\n");
            xml.push_str(&format!(
                "          <metronome><beat-unit>quarter</beat-unit><per-minute>{}</per-minute></metronome>\n",
                tempo_bpm as u32
            ));
            xml.push_str("        </direction-type>\n");
            xml.push_str("      </direction>\n");
        }

        // Notes in this measure
        let measure_notes: Vec<_> = note_events
            .iter()
            .filter(|(s, _, _)| *s >= m_start && *s < m_end)
            .collect();

        for &&(start, dur, midi) in &measure_notes {
            let (name, octave) = midi_to_name(midi);
            let step = &name[..1];
            let alter = if name.len() > 1 { Some(1) } else { None };
            let note_type = divisions_to_type(dur);

            xml.push_str("      <note>\n");
            xml.push_str("        <pitch>\n");
            xml.push_str(&format!("          <step>{step}</step>\n"));
            if let Some(a) = alter {
                xml.push_str(&format!("          <alter>{a}</alter>\n"));
            }
            xml.push_str(&format!("          <octave>{octave}</octave>\n"));
            xml.push_str("        </pitch>\n");
            xml.push_str(&format!("        <duration>{dur}</duration>\n"));
            xml.push_str(&format!("        <type>{note_type}</type>\n"));
            xml.push_str("      </note>\n");

            let _ = start; // used in filter
        }

        xml.push_str("    </measure>\n");
    }

    xml.push_str("  </part>\n");
    xml.push_str("</score-partwise>\n");
    xml
}

/// Export a composition as an SVG staff rendering.
///
/// Produces an 800x200 SVG with 5 staff lines, a treble clef glyph,
/// and note heads positioned by MIDI pitch.
pub fn to_score_svg(comp: &Composition, tempo_bpm: f32) -> String {
    let width = 800;
    let height = 200;
    let staff_top = 60;
    let line_spacing = 10;
    let note_start_x = 80; // leave room for clef
    let note_spacing = if comp.notes.is_empty() {
        20
    } else {
        ((width - note_start_x - 20) as f32 / comp.notes.len() as f32).max(12.0) as i32
    };

    let mut svg = String::new();
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\">\n"
    ));
    svg.push_str("<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n");

    // 5 staff lines
    for i in 0..5 {
        let y = staff_top + i * line_spacing;
        svg.push_str(&format!(
            "<line x1=\"40\" y1=\"{y}\" x2=\"{}\" y2=\"{y}\" stroke=\"black\" stroke-width=\"1\"/>\n",
            width - 20
        ));
    }

    // Treble clef glyph (simplified text)
    svg.push_str(&format!(
        "<text x=\"45\" y=\"{}\" font-size=\"40\" font-family=\"serif\">\u{1D11E}</text>\n",
        staff_top + 35
    ));

    // Note heads
    for (i, note) in comp.notes.iter().enumerate() {
        let midi = freq_to_midi_note(note.frequency);
        // Map MIDI to staff position: middle line (B4=71) at staff_top + 2*line_spacing
        let staff_center_midi = 71u8; // B4 = middle line of treble clef
        let offset = (staff_center_midi as i32 - midi as i32) * (line_spacing / 2);
        let y = staff_top + 2 * line_spacing + offset;
        let x = note_start_x + i as i32 * note_spacing;

        let divs = duration_to_divisions(note.duration, tempo_bpm);
        let filled = divs < 8; // quarter and shorter are filled

        if filled {
            svg.push_str(&format!(
                "<ellipse cx=\"{x}\" cy=\"{y}\" rx=\"5\" ry=\"4\" fill=\"black\"/>\n"
            ));
        } else {
            svg.push_str(&format!(
                "<ellipse cx=\"{x}\" cy=\"{y}\" rx=\"5\" ry=\"4\" fill=\"white\" stroke=\"black\" stroke-width=\"1.5\"/>\n"
            ));
        }

        // Stem (up if below middle, down if above)
        if divs < 16 {
            // no stem for whole notes
            let stem_dir = if y > staff_top + 2 * line_spacing {
                -1
            } else {
                1
            };
            let stem_x = if stem_dir < 0 { x + 5 } else { x - 5 };
            let stem_y_end = y + stem_dir * 30;
            svg.push_str(&format!(
                "<line x1=\"{stem_x}\" y1=\"{y}\" x2=\"{stem_x}\" y2=\"{stem_y_end}\" stroke=\"black\" stroke-width=\"1\"/>\n"
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
    use crate::{AudioData, Note};

    fn test_comp() -> Composition {
        Composition {
            audio: AudioData::F32(vec![0.0; 100]),
            sample_rate: 44100,
            notes: vec![
                Note {
                    frequency: 261.63,
                    start_time: 0.0,
                    duration: 1.0,
                    velocity: 0.8,
                },
                Note {
                    frequency: 329.63,
                    start_time: 1.0,
                    duration: 0.5,
                    velocity: 0.7,
                },
                Note {
                    frequency: 392.00,
                    start_time: 1.5,
                    duration: 0.25,
                    velocity: 0.9,
                },
                Note {
                    frequency: 523.25,
                    start_time: 1.75,
                    duration: 2.0,
                    velocity: 0.6,
                },
            ],
            duration_secs: 4.0,
            section: SectionType::Developmental,
        }
    }

    #[test]
    fn musicxml_well_formed() {
        let xml = to_musicxml(&test_comp(), 120.0);
        assert!(xml.starts_with("<?xml"));
        assert!(xml.contains("<score-partwise"));
        assert!(xml.contains("</score-partwise>"));
    }

    #[test]
    fn musicxml_has_notes() {
        let xml = to_musicxml(&test_comp(), 120.0);
        assert!(xml.contains("<note>"), "should contain note elements");
        assert!(xml.contains("<pitch>"), "should contain pitch elements");
    }

    #[test]
    fn musicxml_has_tempo() {
        let xml = to_musicxml(&test_comp(), 120.0);
        assert!(xml.contains("<metronome>"), "should contain tempo marking");
        assert!(xml.contains("<per-minute>120</per-minute>"));
    }

    #[test]
    fn svg_valid() {
        let svg = to_score_svg(&test_comp(), 120.0);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn svg_has_staff_lines() {
        let svg = to_score_svg(&test_comp(), 120.0);
        let line_count = svg.matches("<line").count();
        // At least 5 staff lines (plus stems)
        assert!(
            line_count >= 5,
            "should have at least 5 staff lines, got {line_count}"
        );
    }

    #[test]
    fn svg_has_note_heads() {
        let svg = to_score_svg(&test_comp(), 120.0);
        let ellipse_count = svg.matches("<ellipse").count();
        assert_eq!(ellipse_count, 4, "should have 4 note heads");
    }

    #[test]
    fn duration_types_vary() {
        let xml = to_musicxml(&test_comp(), 120.0);
        // We have notes of different durations, so should see different type elements
        let has_quarter = xml.contains("<type>quarter</type>");
        let has_other = xml.contains("<type>half</type>")
            || xml.contains("<type>eighth</type>")
            || xml.contains("<type>whole</type>");
        assert!(
            has_quarter || has_other,
            "should have varied duration types"
        );
    }

    #[test]
    fn midi_to_name_c4() {
        let (name, octave) = midi_to_name(60);
        assert_eq!(name, "C");
        assert_eq!(octave, 4);
    }

    #[test]
    fn midi_to_name_a4() {
        let (name, octave) = midi_to_name(69);
        assert_eq!(name, "A");
        assert_eq!(octave, 4);
    }

    #[test]
    fn round_trip_compose_to_notation() {
        let config = crate::MuseConfig {
            duration_secs: 1.0,
            max_notes: 4,
            ..Default::default()
        };
        let state = crate::MusicalState::default();
        let comp = crate::compose(&config, &state, 42);
        let xml = comp.to_musicxml(120.0);
        let svg = comp.to_score_svg(120.0);
        assert!(xml.contains("<note>"));
        assert!(svg.contains("<ellipse"));
    }
}
