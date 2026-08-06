// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Calibrate the score validators against REAL repertoire.
//!
//! # Why
//!
//! `score_validation` reports Fatal voice-leading issues on our generated
//! music — currently ~432 parallel fifths, ~2714 strong-beat dissonances and
//! ~941 voice crossings across 29 styles × 12 seeds. Nobody knows whether those
//! numbers are BAD. They are unvalidated proxies: no one has ever shown they
//! predict anything a listener notices.
//!
//! Real music answers that without asking anyone to listen. Run the identical
//! validators over MAESTRO (virtuoso performances of classical repertoire) and
//! compare rates:
//!
//! - Chopin scoring ~the same rate as us ⇒ the metric measures noise, and every
//!   conclusion drawn from it in this crate should be discounted.
//! - Chopin scoring near zero while we do not ⇒ our music is genuinely
//!   defective and the number is a real target.
//!
//! # Fairness, which is the whole methodology
//!
//! MIDI has no `VoiceRole`. Roles are therefore assigned HEURISTICALLY by
//! register (lowest sounding = Bass, highest = Melody, rest = Harmony) — and
//! the SAME heuristic is re-applied to our own generated scores, discarding
//! their true roles. Otherwise this would compare true-role analysis against
//! heuristic-role analysis and measure the heuristic, not the music.
//!
//! # Known confounds, stated rather than buried
//!
//! 1. **Quantization.** MAESTRO is unquantized human performance; our scores
//!    are exact. `validate_strong_beats` only inspects integer beats, so raw
//!    performance data would land almost nothing on a beat and score a false
//!    zero. Onsets are therefore quantized to a 16th-note grid. That is a real
//!    intervention and it flatters the repertoire's strong-beat numbers.
//! 2. **Pedal.** MAESTRO has sustain-pedal data this ignores, so held
//!    sonorities are under-represented — biasing AGAINST finding dissonance in
//!    the repertoire.
//! 3. **Tempo/meter.** Assumed 4/4 at the file's own PPQ. MAESTRO includes
//!    triple-meter works, which will mis-locate their strong beats.
//! 4. **Piano only.** MAESTRO is solo piano; our output is ensemble. Register
//!    separation between "voices" is therefore tighter in the repertoire.
//!
//! Confounds 1-3 all push the repertoire's counts DOWN relative to ours, so a
//! finding of "repertoire ≈ us" is robust (the metric is noise) while
//! "repertoire ≪ us" is the weaker direction and should be read as suggestive.
//!
//! ```sh
//! cargo run --release -p symthaea-muse --features theory \
//!   --example validator_calibration -- /opt/datasets/maestro 60
//! ```

use std::collections::BTreeMap;
use std::path::Path;
use symthaea_music_theory::score::{PartId, Score, ScoreNote, VoiceRole};
use symthaea_music_theory::score_validation::{
    ScoreValidationConfig, ValidationSeverity, validate_score,
};
use symthaea_music_theory::{
    Duration, Emphasis, Key, MusicalIntent, PitchClass, Style, compose_styled, pitch::Pitch,
};

/// One note, before roles are assigned: (onset beats, duration beats, midi).
type RawNote = (f64, f64, u8);

/// Quantize to a 16th-note grid — see confound 1.
fn quantize(beats: f64) -> f64 {
    (beats * 4.0).round() / 4.0
}

/// Parse a MIDI file into beat-relative notes. Merges all tracks: we want the
/// full polyphonic texture, not a monophonic melody line (which is what
/// `midi_loader` extracts and why it is not reused here).
fn parse(path: &Path) -> Option<Vec<RawNote>> {
    let bytes = std::fs::read(path).ok()?;
    let smf = midly::Smf::parse(&bytes).ok()?;
    let ppq = match smf.header.timing {
        midly::Timing::Metrical(t) => t.as_int() as f64,
        _ => return None, // SMPTE timing: not beat-relative, skip
    };
    let mut open: BTreeMap<u8, (u32, u8)> = BTreeMap::new();
    let mut out: Vec<RawNote> = Vec::new();
    for track in smf.tracks {
        let mut tick: u32 = 0;
        for ev in track {
            tick += ev.delta.as_int();
            let midly::TrackEventKind::Midi { message, .. } = ev.kind else {
                continue;
            };
            match message {
                midly::MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
                    open.insert(key.as_int(), (tick, vel.as_int()));
                }
                midly::MidiMessage::NoteOff { key, .. }
                | midly::MidiMessage::NoteOn { key, .. } => {
                    if let Some((start, _)) = open.remove(&key.as_int()) {
                        let on = start as f64 / ppq;
                        let dur = (tick.saturating_sub(start)) as f64 / ppq;
                        if dur > 1e-3 {
                            out.push((quantize(on), quantize(dur).max(0.25), key.as_int()));
                        }
                    }
                }
                _ => {}
            }
        }
    }
    (!out.is_empty()).then_some(out)
}

/// Assign roles by REGISTER at each onset — the one heuristic, applied
/// identically to repertoire and to our own output.
fn assign_roles(raw: &[RawNote]) -> Score {
    let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
    // Group by quantized onset; within each group the extremes take Bass and
    // Melody and everything between becomes Harmony.
    let mut by_onset: BTreeMap<i64, Vec<&RawNote>> = BTreeMap::new();
    for n in raw {
        by_onset
            .entry((n.0 * 4.0).round() as i64)
            .or_default()
            .push(n);
    }
    for (_, mut group) in by_onset {
        group.sort_by_key(|n| n.2);
        let last = group.len().saturating_sub(1);
        for (i, n) in group.iter().enumerate() {
            let role = if group.len() == 1 {
                VoiceRole::Melody
            } else if i == 0 {
                VoiceRole::Bass
            } else if i == last {
                VoiceRole::Melody
            } else {
                VoiceRole::Harmony
            };
            score.push(ScoreNote {
                part: PartId::UNASSIGNED,
                pitch: Pitch::from_midi(n.2),
                onset: Duration::new((n.0 * 4.0).round() as i64, 4),
                duration: Duration::new((n.1 * 4.0).round().max(1.0) as i64, 4),
                velocity: 0.7,
                role,
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
    }
    score
}

#[derive(Default, Clone, Copy)]
struct Rates {
    bars: f64,
    parallel: usize,
    strong_beat: usize,
    crossing: usize,
    final_tonic: usize,
}

fn tally(score: &Score, r: &mut Rates) {
    let cfg = ScoreValidationConfig::default();
    r.bars += (score.total_beats.beats() / f64::from(score.meter.max(1))).max(1.0);
    for i in validate_score(score, &cfg)
        .issues
        .iter()
        .filter(|i| i.severity == ValidationSeverity::Fatal)
    {
        match format!("{:?}", i.rule).as_str() {
            "ParallelPerfectMotion" => r.parallel += 1,
            "StrongBeatConsonance" => r.strong_beat += 1,
            "VoiceCrossing" => r.crossing += 1,
            "FinalTonicArrival" => r.final_tonic += 1,
            _ => {}
        }
    }
}

fn report(label: &str, r: Rates) {
    let per = |n: usize| n as f64 / r.bars.max(1.0);
    println!(
        "  {label:<34} {:>9.0} {:>10.3} {:>12.3} {:>10.3}",
        r.bars,
        per(r.parallel),
        per(r.strong_beat),
        per(r.crossing)
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = args
        .get(1)
        .cloned()
        .unwrap_or("/opt/datasets/maestro".into());
    let limit: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(60);

    // ── Repertoire ────────────────────────────────────────────────────────
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
    files.sort(); // deterministic sample, not a lucky one
    if files.is_empty() {
        eprintln!(
            "ABORT: no MIDI under {root}. Without repertoire this harness \
                   would report only our own numbers and prove nothing."
        );
        std::process::exit(2);
    }
    let mut rep = Rates::default();
    let mut parsed = 0usize;
    for f in files.iter().take(limit) {
        if let Some(raw) = parse(f) {
            tally(&assign_roles(&raw), &mut rep);
            parsed += 1;
        }
    }

    // ── Ours, through the IDENTICAL heuristic ─────────────────────────────
    let mut ours = Rates::default();
    for style in Style::ALL {
        for seed in 0..12u64 {
            let intent = MusicalIntent {
                seed,
                ..Default::default()
            };
            let sc = compose_styled(&intent, style);
            let raw: Vec<RawNote> = sc
                .notes
                .iter()
                .map(|n| (n.onset.beats(), n.duration.beats(), n.pitch.midi()))
                .collect();
            tally(&assign_roles(&raw), &mut ours);
        }
    }

    println!(
        "Repertoire: {parsed} MAESTRO files parsed (of {} found, limit {limit})",
        files.len()
    );
    println!("Roles assigned by REGISTER on both sides; onsets quantized to 1/16.\n");
    println!(
        "  {:<34} {:>9} {:>10} {:>12} {:>10}",
        "", "bars", "parallel/bar", "strongbeat/bar", "cross/bar"
    );
    report("MAESTRO (real repertoire)", rep);
    report("symthaea-music-theory (ours)", ours);

    let ratio = |o: usize, r: usize, ob: f64, rb: f64| {
        let (a, b) = (o as f64 / ob.max(1.0), r as f64 / rb.max(1.0));
        if b <= 1e-9 { f64::INFINITY } else { a / b }
    };
    println!("\n  ours ÷ repertoire:");
    println!(
        "    parallel   {:.2}x\n    strongbeat {:.2}x\n    crossing   {:.2}x",
        ratio(ours.parallel, rep.parallel, ours.bars, rep.bars),
        ratio(ours.strong_beat, rep.strong_beat, ours.bars, rep.bars),
        ratio(ours.crossing, rep.crossing, ours.bars, rep.bars)
    );
    println!(
        "\n  Read ~1x as \"this validator does not distinguish us from real music\".\n  \
         Quantization, missing pedal and assumed 4/4 all push the repertoire's\n  \
         numbers DOWN, so ~1x is the robust finding and \"ours >> repertoire\" is\n  \
         the weaker direction."
    );
}
