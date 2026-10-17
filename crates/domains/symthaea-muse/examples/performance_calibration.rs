// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Calibrate the PERFORMANCE layer against real recorded performances.
//!
//! # Why
//!
//! `validator_calibration` showed the three voice-leading validators cannot
//! distinguish our scores from Chopin — our symbolic output is structurally
//! comparable to real repertoire. That points the remaining quality gap at
//! realization: how the notes are PLAYED, not which notes they are. The standing
//! listening verdict ("still not good enough") and the code's own comments
//! ("the MIDI preview sound", "a note PARKED, not played") point the same way.
//!
//! MAESTRO is ground truth for exactly this. It is not transcription — it is
//! virtuoso performance captured on a Disklavier, with real rubato, real
//! dynamics and real articulation. Measuring our performed MIDI against it says
//! whether our playing is humanly variable or mechanically flat, with no
//! listening required.
//!
//! # Method
//!
//! Both sides go through the SAME MIDI parser and the SAME statistics.
//! `export_performance_midi` bakes in swing, rubato, the learned expressive
//! model and humanize jitter (`midi_export.rs:288`), so ours is a genuine
//! performance rendering rather than the written score.
//!
//! Three measures, each chosen because a mechanical performance and a human one
//! differ on it in a known direction:
//!
//! - **Timing spread** — |onset − nearest 16th| in beats. Zero means quantized;
//!   humans are never zero.
//! - **Velocity spread** — std dev of note velocity. Zero means every note is
//!   equally loud.
//! - **Articulation spread** — std dev of (duration ÷ inter-onset interval).
//!   Constant means every note is held the same proportion of its slot.
//!
//! # Confounds, stated
//!
//! 1. MAESTRO is solo piano; ours is ensemble, and per-voice statistics differ
//!    from a single player's. Measured over all notes on both sides.
//! 2. MAESTRO tempo varies continuously; beats here come from the file's PPQ
//!    with no tempo-map following, so slow rubato inflates its timing spread.
//!    That biases the comparison IN OUR FAVOUR on timing — so "ours ≪ theirs"
//!    is the conservative direction, and "ours ≈ theirs" would be suspicious.
//! 3. Sustain pedal is ignored, which shortens repertoire articulation.
//!
//! ```sh
//! cargo run --release -p symthaea-muse --features theory \
//!   --example performance_calibration -- /opt/datasets/maestro 40
//! ```

use std::collections::BTreeMap;
use std::path::Path;
use symthaea_muse::MusicalState;
use symthaea_music_theory::{MusicalIntent, Style, compose_styled};

/// (onset beats, duration beats, midi, velocity 0-127)
type Ev = (f64, f64, u8, u8);

fn parse(path: &Path) -> Option<Vec<Ev>> {
    let bytes = std::fs::read(path).ok()?;
    let smf = midly::Smf::parse(&bytes).ok()?;
    let ppq = match smf.header.timing {
        midly::Timing::Metrical(t) => t.as_int() as f64,
        _ => return None,
    };
    let mut out = Vec::new();
    for track in &smf.tracks {
        out.extend(parse_track(track, ppq)?);
    }
    (!out.is_empty()).then_some(out)
}

/// Parse one MIDI track with channel-and-pitch-specific note pairing.
///
/// The calibration must not overwrite an earlier same-pitch note-on, pair
/// different channels, or silently discard unmatched events: any such file is
/// excluded as ambiguous rather than allowed to distort performance statistics.
fn parse_track(track: &[midly::TrackEvent<'_>], ppq: f64) -> Option<Vec<Ev>> {
    let mut open: BTreeMap<(u8, u8), (u32, u8)> = BTreeMap::new();
    let mut out = Vec::new();
    let mut tick = 0_u32;

    for event in track {
        tick = tick.checked_add(event.delta.as_int())?;
        let midly::TrackEventKind::Midi { channel, message } = &event.kind else {
            continue;
        };
        let channel = channel.as_int();
        match message {
            midly::MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
                let identity = (channel, key.as_int());
                if open.contains_key(&identity) {
                    return None;
                }
                open.insert(identity, (tick, vel.as_int()));
            }
            midly::MidiMessage::NoteOff { key, .. }
            | midly::MidiMessage::NoteOn { key, .. } => {
                let identity = (channel, key.as_int());
                let (start, velocity) = open.remove(&identity)?;
                let duration_ticks = tick.checked_sub(start)?;
                let duration_beats = f64::from(duration_ticks) / ppq;
                if duration_beats <= 1.0e-4 {
                    return None;
                }
                out.push((
                    f64::from(start) / ppq,
                    duration_beats,
                    key.as_int(),
                    velocity,
                ));
            }
            _ => {}
        }
    }

    if !open.is_empty() {
        return None;
    }
    Some(out)
}

#[cfg(test)]
mod midi_pairing_tests {
    use super::*;
    use midly::{
        MidiMessage, TrackEvent, TrackEventKind,
        num::{u4, u7, u28},
    };

    fn event(delta: u32, channel: u8, message: MidiMessage) -> TrackEvent<'static> {
        TrackEvent {
            delta: u28::new(delta),
            kind: TrackEventKind::Midi {
                channel: u4::new(channel),
                message,
            },
        }
    }

    fn note_on(key: u8, velocity: u8) -> MidiMessage {
        MidiMessage::NoteOn {
            key: u7::new(key),
            vel: u7::new(velocity),
        }
    }

    fn note_off(key: u8) -> MidiMessage {
        MidiMessage::NoteOff {
            key: u7::new(key),
            vel: u7::new(0),
        }
    }

    #[test]
    fn pairs_same_pitch_independently_on_different_channels() {
        let track = vec![
            event(0, 0, note_on(60, 100)),
            event(0, 1, note_on(60, 80)),
            event(480, 0, note_off(60)),
            event(0, 1, note_off(60)),
        ];
        let parsed = parse_track(&track, 480.0).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].0, 0.0);
        assert_eq!(parsed[1].0, 0.0);
        assert_eq!(parsed[0].1, 1.0);
        assert_eq!(parsed[1].1, 1.0);
        assert_eq!(parsed[0].3, 100);
        assert_eq!(parsed[1].3, 80);
    }

    #[test]
    fn rejects_retriggered_same_channel_pitch_before_note_off() {
        let track = vec![
            event(0, 0, note_on(60, 100)),
            event(120, 0, note_on(60, 90)),
            event(120, 0, note_off(60)),
            event(120, 0, note_off(60)),
        ];
        assert!(parse_track(&track, 480.0).is_none());
    }

    #[test]
    fn rejects_orphan_note_off_and_unmatched_note_on() {
        let orphan = vec![event(0, 0, note_off(60))];
        assert!(parse_track(&orphan, 480.0).is_none());

        let unmatched = vec![event(0, 0, note_on(60, 100))];
        assert!(parse_track(&unmatched, 480.0).is_none());
    }

    #[test]
    fn rejects_tick_overflow_instead_of_wrapping() {
        let track = vec![
            event(0x0fff_ffff, 0, note_on(60, 100)),
            event(0x0fff_ffff, 0, note_off(60)),
            event(0x0fff_ffff, 0, note_on(62, 100)),
            event(0x0fff_ffff, 0, note_off(62)),
        ];
        assert!(parse_track(&track, 480.0).is_none());
    }
}

fn std_dev(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = xs.iter().sum::<f64>() / xs.len() as f64;
    (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64).sqrt()
}

#[derive(Default)]
struct Acc {
    timing: Vec<f64>,
    velocity: Vec<f64>,
    articulation: Vec<f64>,
    notes: usize,
}

fn accumulate(evs: &[Ev], a: &mut Acc) {
    let mut sorted = evs.to_vec();
    sorted.sort_by(|x, y| x.0.total_cmp(&y.0));
    a.notes += sorted.len();
    for e in &sorted {
        // Distance to the nearest 16th-note grid position.
        a.timing.push((e.0 * 4.0 - (e.0 * 4.0).round()).abs() / 4.0);
        a.velocity.push(e.2 as f64 * 0.0 + e.3 as f64);
    }
    // Articulation needs the NEXT onset, so walk pairs.
    for w in sorted.windows(2) {
        let ioi = w[1].0 - w[0].0;
        if ioi > 1e-6 {
            a.articulation.push((w[0].1 / ioi).min(4.0));
        }
    }
}

fn report(label: &str, a: &Acc) {
    println!(
        "  {label:<32} {:>9} {:>12.4} {:>12.2} {:>14.3}",
        a.notes,
        std_dev(&a.timing),
        std_dev(&a.velocity),
        std_dev(&a.articulation)
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = args
        .get(1)
        .cloned()
        .unwrap_or("/opt/datasets/maestro".into());
    let limit: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(40);

    let mut files: Vec<std::path::PathBuf> = Vec::new();
    let mut stack = vec![std::path::PathBuf::from(&root)];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "midi" || x == "mid") {
                files.push(p);
            }
        }
    }
    files.sort();
    if files.is_empty() {
        eprintln!(
            "ABORT: no MIDI under {root}. Without real performances this would \
             report only our own numbers and prove nothing."
        );
        std::process::exit(2);
    }

    let mut human = Acc::default();
    let mut parsed = 0;
    for f in files.iter().take(limit) {
        if let Some(evs) = parse(f) {
            accumulate(&evs, &mut human);
            parsed += 1;
        }
    }

    // Ours, exported through the real performance path and re-parsed with the
    // identical parser so no asymmetry creeps in.
    let mut ours = Acc::default();
    let dir = std::env::temp_dir().join(format!("perf_calib_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let state = MusicalState::default();
    let mut exported = 0;
    for style in Style::ALL {
        for seed in 0..4u64 {
            let intent = MusicalIntent {
                seed,
                ..Default::default()
            };
            let score = compose_styled(&intent, style);
            let spec = style.spec();
            let path = dir.join(format!("{style:?}_{seed}.mid"));
            if symthaea_muse::midi_export::export_performance_midi(
                &score, &spec, seed, &state, &path,
            )
            .is_err()
            {
                continue;
            }
            if let Some(evs) = parse(&path) {
                accumulate(&evs, &mut ours);
                exported += 1;
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);

    println!("Real performances parsed: {parsed}   ours exported: {exported}\n");
    println!(
        "  {:<32} {:>9} {:>12} {:>12} {:>14}",
        "", "notes", "timing sd", "velocity sd", "articul. sd"
    );
    report("MAESTRO (human performance)", &human);
    report("symthaea-muse (ours)", &ours);

    let r = |o: f64, h: f64| {
        if h.abs() < 1e-12 {
            f64::INFINITY
        } else {
            o / h
        }
    };
    println!(
        "\n  ours / human:  timing {:.2}x   velocity {:.2}x   articulation {:.2}x",
        r(std_dev(&ours.timing), std_dev(&human.timing)),
        r(std_dev(&ours.velocity), std_dev(&human.velocity)),
        r(std_dev(&ours.articulation), std_dev(&human.articulation))
    );
    println!(
        "\n  Well below 1x means mechanically flat on that axis -- the measurable\n  \
         shape of \"sounds like MIDI\". Note confound 2 inflates the HUMAN timing\n  \
         figure (no tempo-map following), so a low timing ratio is the weaker\n  \
         claim; velocity and articulation are not affected by it."
    );
}
