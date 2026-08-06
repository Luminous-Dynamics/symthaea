// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Resolve the PARKED rhythmic-regularity gate by calibrating it against real
//! performance.
//!
//! # This is not a fourth proxy
//!
//! `creative_regression.rs` gates four metrics and *parks* a fifth. The parked
//! one, `onset_evenness`, is `1 − CV(inter-onset intervals)`, so **1.0
//! means perfectly metronomic**. Its baseline of 0.78 is therefore a demand for
//! rigidity, and the doc comment on the parked test says outright that
//! resolving it "is a musical-intent decision, not a threshold tweak".
//!
//! That decision does not need taste. It needs a number from real music, which
//! is the same move the three sibling calibration harnesses make. Nothing new is
//! being measured here — an existing, already-shipped gate is being pointed at
//! repertoire so it can be un-parked with a defensible band instead of a guess.
//!
//! # Method: the production scorer, not a re-implementation
//!
//! Repertoire notes are packed into a real `Composition` and scored by
//! `CreativeQualityScore::evaluate`, the same public entry point
//! `creative_bench::run_benchmark` uses. Melody extraction (highest pitch per
//! 30 ms onset group) and the median×3 section-gap filter therefore run as
//! production code on both sides, with no chance of a replication error.
//!
//! Repertoire is chopped into windows of the same note count the benchmark
//! generates (`MuseConfig::max_notes`), because CV over a 20-minute recital and
//! CV over four seconds are not the same statistic.
//!
//! Note timings come from a real tempo map (all `MetaMessage::Tempo` events
//! accumulated), not the initial tempo, so seconds are exact rather than
//! nominal — this metric reads onsets in seconds and would otherwise inherit a
//! tempo confound.
//!
//! # Confounds
//!
//! 1. MAESTRO is solo piano; our compositions are ensemble. Melody extraction
//!    is applied identically, but a wandering top line in dense piano writing
//!    picks up onsets from inner voices, which RAISES the repertoire's apparent
//!    onset density and lowers its CV. Biases the repertoire toward looking
//!    more regular than it is.
//! 2. Real rubato is genuine timing variance; ours is generated humanization.
//!    Both are legitimately "performance", which is why this comparison is fair
//!    at all — but they are not the same mechanism.
//! 3. MAESTRO includes pedalled passages and free cadenzas whose onsets have no
//!    metric pulse. The median×3 filter removes section gaps, not those.
//! 4. The gate's unit is a mean over k compositions, so the repertoire is
//!    grouped into k *consecutive* windows to match. Consecutive windows come
//!    from the same piece and are correlated, so those group means are more
//!    dispersed than k independent draws would be — which makes the derived
//!    band WIDER, i.e. the gate more permissive. For a CI gate that is the safe
//!    direction (it will not flake); it also means the band is not a tight
//!    bound and should not be read as one.
//!
//! ```sh
//! cargo run --release -p symthaea-muse --features theory \
//!   --example rhythm_gate_calibration -- /opt/datasets/maestro 60
//! ```

use std::collections::BTreeMap;
use std::path::Path;
use symthaea_aesthetic::ValenceArousal;
use symthaea_muse::creative_bench::CreativeQualityScore;
use symthaea_muse::{AudioData, Composition, MuseConfig, MusicalState, Note, creative_bench};

/// Build the tempo map, then convert every note-on to absolute seconds.
fn parse_seconds(path: &Path) -> Option<Vec<Note>> {
    let bytes = std::fs::read(path).ok()?;
    let smf = midly::Smf::parse(&bytes).ok()?;
    let ppq = match smf.header.timing {
        midly::Timing::Metrical(t) => f64::from(t.as_int()),
        _ => return None,
    };

    // Tempo map: (tick, microseconds per quarter note), in tick order.
    let mut tempos: Vec<(u32, u32)> = Vec::new();
    for track in &smf.tracks {
        let mut tick: u32 = 0;
        for ev in track {
            tick += ev.delta.as_int();
            if let midly::TrackEventKind::Meta(midly::MetaMessage::Tempo(us)) = ev.kind {
                tempos.push((tick, us.as_int()));
            }
        }
    }
    tempos.sort_by_key(|t| t.0);
    if tempos.is_empty() {
        tempos.push((0, 500_000)); // MIDI default: 120 bpm
    }

    let to_secs = |tick: u32| -> f32 {
        let mut secs = 0.0f64;
        let mut prev_tick = 0u32;
        let mut prev_us = f64::from(tempos[0].1);
        for &(t, us) in &tempos {
            if t >= tick {
                break;
            }
            secs += f64::from(t - prev_tick) / ppq * prev_us / 1e6;
            prev_tick = t;
            prev_us = f64::from(us);
        }
        (secs + f64::from(tick - prev_tick) / ppq * prev_us / 1e6) as f32
    };

    let mut open: BTreeMap<u8, (u32, u8)> = BTreeMap::new();
    let mut out: Vec<Note> = Vec::new();
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
                    if let Some((start, vel)) = open.remove(&key.as_int()) {
                        let s = to_secs(start);
                        let e = to_secs(tick);
                        if e > s {
                            out.push(Note {
                                // MIDI note -> Hz; the scorer compares frequencies
                                // only to pick the highest of an onset group.
                                frequency: 440.0
                                    * 2f32.powf((f32::from(key.as_int()) - 69.0) / 12.0),
                                start_time: s,
                                duration: e - s,
                                velocity: f32::from(vel) / 127.0,
                            });
                        }
                    }
                }
                _ => {}
            }
        }
    }
    out.sort_by(|a, b| a.start_time.total_cmp(&b.start_time));
    (!out.is_empty()).then_some(out)
}

/// Score one window of notes through the real production path.
fn score(notes: &[Note]) -> f32 {
    let duration = notes
        .last()
        .map_or(0.0, |n| n.start_time + n.duration)
        .max(0.001);
    let comp = Composition {
        audio: AudioData::F32(Vec::new()),
        sample_rate: 48_000,
        notes: notes.to_vec(),
        duration_secs: duration,
        section: symthaea_muse::structure::SectionType::Developmental,
    };
    // Only `onset_evenness` is read; the VA target affects a different field.
    CreativeQualityScore::evaluate(&comp, ValenceArousal::new(0.0, 0.5)).onset_evenness
}

fn stats(xs: &mut Vec<f32>) -> (f32, f32, f32, f32) {
    xs.sort_by(|a, b| a.total_cmp(b));
    let mean = xs.iter().sum::<f32>() / xs.len() as f32;
    let pct = |p: f32| xs[((xs.len() - 1) as f32 * p).round() as usize];
    (mean, pct(0.05), pct(0.5), pct(0.95))
}

/// Means of consecutive `k`-window groups.
///
/// This is the statistic the gate actually reads: `mean_onset_evenness` is
/// an average over the benchmark's `k` compositions, not a single window. A
/// per-window percentile band would be far too wide for it — comparing a mean
/// against a distribution of singletons is a units error, and it would make the
/// gate unable to catch anything.
fn group_means(xs: &[f32], k: usize) -> Vec<f32> {
    xs.chunks(k)
        .filter(|c| c.len() == k)
        .map(|c| c.iter().sum::<f32>() / k as f32)
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = args
        .get(1)
        .cloned()
        .unwrap_or("/opt/datasets/maestro".into());
    let limit: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(60);

    let cfg = MuseConfig {
        duration_secs: 4.0,
        max_notes: 16,
        ..MuseConfig::default()
    };

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
            "ABORT: no MIDI under {root}. Without repertoire this reports only our \
             own number, which is exactly the failure that parked the gate."
        );
        std::process::exit(2);
    }

    // Repertoire, windowed to the benchmark's own note count.
    let window = cfg.max_notes.max(3);
    let mut human: Vec<f32> = Vec::new();
    let mut parsed = 0;
    for f in files.iter().take(limit) {
        if let Some(notes) = parse_seconds(f) {
            parsed += 1;
            for chunk in notes.chunks(window) {
                if chunk.len() >= 3 {
                    human.push(score(chunk));
                }
            }
        }
    }
    if human.len() < 100 {
        eprintln!("ABORT: only {} repertoire windows; too few.", human.len());
        std::process::exit(1);
    }

    // Ours, straight from the benchmark the gate actually guards.
    let result = creative_bench::run_benchmark(&cfg, &MusicalState::default());

    // The benchmark averages over this many compositions, so the comparable
    // repertoire statistic is the mean of that many windows, not one window.
    let k = result.n_compositions.max(1);
    let mut grouped = group_means(&human, k);
    let (g_mean, g_p05, g_p50, g_p95) = stats(&mut grouped);
    let (h_mean, h_p05, h_p50, h_p95) = stats(&mut human);
    let below =
        100.0 * grouped.iter().filter(|&&x| x < 0.780).count() as f32 / grouped.len() as f32;

    println!(
        "Repertoire: {parsed} files -> {} windows of <= {window} notes\n\
         Scored by CreativeQualityScore::evaluate -- production extraction and\n\
         production metric, on both sides.\n",
        human.len()
    );
    println!(
        "  {:<38} {:>7} {:>7} {:>7} {:>7}",
        "", "p05", "median", "mean", "p95"
    );
    println!(
        "  {:<38} {h_p05:>7.3} {h_p50:>7.3} {h_mean:>7.3} {h_p95:>7.3}",
        "MAESTRO, single window"
    );
    println!(
        "  {:<38} {g_p05:>7.3} {g_p50:>7.3} {g_mean:>7.3} {g_p95:>7.3}",
        format!("MAESTRO, mean of {k} (gate's unit) [n={}]", grouped.len())
    );
    println!(
        "  {:<38} {:>7} {:>7} {:>7.3} {:>7}",
        "symthaea-muse (ours)", "-", "-", result.mean_onset_evenness, "-"
    );
    println!(
        "\n  Parked baseline was 0.780, a one-sided FLOOR -- i.e. a demand for\n  \
         metronomic playing. {below:.0}% of real-music samples would FAIL it.",
    );
    println!(
        "\n  Gate band = repertoire's own 5th-95th percentile at the gate's unit:\n  \
         [{g_p05:.2}, {g_p95:.2}]. Two-sided, so it rejects mechanical rigidity as\n  \
         well as incoherence. Confound 1 makes the repertoire look MORE regular\n  \
         than it is, so the UPPER edge is the conservative one."
    );
}
