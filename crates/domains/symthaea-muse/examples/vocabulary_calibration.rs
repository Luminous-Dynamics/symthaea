// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Calibrate MELODIC VOCABULARY diversity against real repertoire.
//!
//! # Why this one
//!
//! `validator_calibration` and `performance_calibration` between them showed our
//! scores are structurally comparable to real repertoire and our performance
//! dispersion matches human performance. The 99.8% canonical duplicate rate that
//! `motif_foundry_style_survey` reports is now the ONLY measure on which we
//! still look anomalous — and it is the only one never calibrated.
//!
//! It is also the direct test of the standing strategic thesis, held by this
//! crate's own docs and by an external review, that the remaining quality gap is
//! "memorable melodic identity". If real repertoire is similarly repetitive by
//! this measure, that thesis is another proxy artefact. If it is not, the thesis
//! is confirmed with a number attached.
//!
//! # Why interval n-grams rather than the existing canonical fingerprint
//!
//! `motif_foundry::canonical_fingerprint` takes a `HookCell`, so applying it to
//! repertoire would mean segmenting Chopin into motifs first — a musicological
//! judgement with no clean answer, and exactly the kind of buried assumption
//! that would contaminate the result.
//!
//! Interval n-grams need no phrase boundaries. Intervals are
//! transposition-invariant by construction, and each n-gram is canonicalised
//! under the same four symmetries the existing fingerprint folds out
//! (identity, inversion, retrograde, retrograde-inversion). Same statistic,
//! same symmetries, no segmentation.
//!
//! # The methodological trap this avoids
//!
//! Distinct-ratio SATURATES with sample size: 200k n-grams will always look less
//! diverse than 5k, whatever the music. Comparing our ~10k against MAESTRO's
//! ~200k would manufacture a difference out of nothing. Both sides are therefore
//! deterministically subsampled to the SAME n-gram count before counting, and
//! the harness refuses to report if it cannot.
//!
//! # Confounds
//!
//! 1. Melody extraction is "highest sounding pitch at each onset", applied
//!    identically to both. In dense piano writing that line wanders between
//!    real voices; in our ensemble output it tracks the Melody role closely.
//!    This biases the repertoire's apparent diversity UP.
//! 2. MAESTRO is unquantized, so repeated rhythmic figures are not collapsed.
//!    Only pitch intervals are measured here, so this affects little.
//! 3. Solo piano versus ensemble, as in the sibling harnesses.
//!
//! ```sh
//! cargo run --release -p symthaea-muse --features theory \
//!   --example vocabulary_calibration -- /opt/datasets/maestro 40
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use symthaea_music_theory::{MusicalIntent, Style, compose_styled};

/// Length of the interval window. 4 intervals = 5 notes, which brackets the
/// 3-5 note hooks `hook_contours` actually declares.
const N: usize = 4;

fn parse_melody(path: &Path) -> Option<Vec<u8>> {
    let bytes = std::fs::read(path).ok()?;
    let smf = midly::Smf::parse(&bytes).ok()?;
    let ppq = match smf.header.timing {
        midly::Timing::Metrical(t) => t.as_int() as f64,
        _ => return None,
    };
    let mut open: BTreeMap<u8, u32> = BTreeMap::new();
    let mut notes: Vec<(f64, u8)> = Vec::new();
    for track in smf.tracks {
        let mut tick: u32 = 0;
        for ev in track {
            tick += ev.delta.as_int();
            let midly::TrackEventKind::Midi { message, .. } = ev.kind else {
                continue;
            };
            match message {
                midly::MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
                    open.insert(key.as_int(), tick);
                }
                midly::MidiMessage::NoteOff { key, .. }
                | midly::MidiMessage::NoteOn { key, .. } => {
                    if let Some(start) = open.remove(&key.as_int()) {
                        notes.push((start as f64 / ppq, key.as_int()));
                    }
                }
                _ => {}
            }
        }
    }
    Some(top_line(&notes))
}

/// Highest sounding pitch per quantized onset — the one extraction heuristic,
/// applied identically to repertoire and to our own output.
fn top_line(notes: &[(f64, u8)]) -> Vec<u8> {
    let mut by_onset: BTreeMap<i64, u8> = BTreeMap::new();
    for (t, p) in notes {
        let slot = (t * 4.0).round() as i64;
        by_onset
            .entry(slot)
            .and_modify(|e| *e = (*e).max(*p))
            .or_insert(*p);
    }
    by_onset.into_values().collect()
}

/// Canonical form of one interval n-gram under the four symmetries the existing
/// canonical fingerprint folds out. Returns the lexicographically smallest.
fn canonical(window: &[i32]) -> Vec<i32> {
    let inv: Vec<i32> = window.iter().map(|x| -x).collect();
    let retro: Vec<i32> = window.iter().rev().map(|x| -x).collect();
    let retro_inv: Vec<i32> = window.iter().rev().copied().collect();
    [window.to_vec(), inv, retro, retro_inv]
        .into_iter()
        .min()
        .expect("four candidates")
}

fn ngrams(line: &[u8]) -> Vec<Vec<i32>> {
    if line.len() < N + 1 {
        return Vec::new();
    }
    let iv: Vec<i32> = line
        .windows(2)
        .map(|w| w[1] as i32 - w[0] as i32)
        // Fold octave displacement: a leap of a 9th and a 2nd are the same
        // melodic gesture for vocabulary purposes, as the existing contour
        // model also treats scale degrees rather than absolute leaps.
        .map(|d| d.clamp(-12, 12))
        .collect();
    iv.windows(N).map(canonical).collect()
}

/// Deterministic even-stride subsample to exactly `target` items. No RNG, so the
/// result is reproducible and cannot be a lucky draw.
fn subsample(v: &[Vec<i32>], target: usize) -> Vec<&Vec<i32>> {
    if v.len() <= target {
        return v.iter().collect();
    }
    let stride = v.len() as f64 / target as f64;
    (0..target)
        .map(|i| &v[((i as f64 * stride) as usize).min(v.len() - 1)])
        .collect()
}

fn report(label: &str, sample: &[&Vec<i32>]) -> f64 {
    let distinct: BTreeSet<&Vec<i32>> = sample.iter().copied().collect();
    let dup = 1.0 - distinct.len() as f64 / sample.len().max(1) as f64;
    println!(
        "  {label:<32} {:>9} {:>11} {:>13.1}%",
        sample.len(),
        distinct.len(),
        dup * 100.0
    );
    dup
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
            "ABORT: no MIDI under {root}. Without repertoire this would report \
             only our own number, which is what the whole exercise exists to avoid."
        );
        std::process::exit(2);
    }

    let mut human: Vec<Vec<i32>> = Vec::new();
    let mut parsed = 0;
    for f in files.iter().take(limit) {
        if let Some(line) = parse_melody(f) {
            human.extend(ngrams(&line));
            parsed += 1;
        }
    }

    let mut ours: Vec<Vec<i32>> = Vec::new();
    for style in Style::ALL {
        for seed in 0..12u64 {
            let intent = MusicalIntent {
                seed,
                ..Default::default()
            };
            let sc = compose_styled(&intent, style);
            let notes: Vec<(f64, u8)> = sc
                .notes
                .iter()
                .map(|n| (n.onset.beats(), n.pitch.midi()))
                .collect();
            ours.extend(ngrams(&top_line(&notes)));
        }
    }

    let target = human.len().min(ours.len());
    if target < 500 {
        eprintln!("ABORT: only {target} comparable n-grams; too few to conclude anything.");
        std::process::exit(1);
    }

    println!(
        "Repertoire files parsed: {parsed}. {N}-interval windows, canonicalised under\n\
         inversion/retrograde/retrograde-inversion. Both sides subsampled to {target}\n\
         (raw: repertoire {}, ours {}) because distinct-ratio saturates with size.\n",
        human.len(),
        ours.len()
    );
    println!(
        "  {:<32} {:>9} {:>11} {:>14}",
        "", "n-grams", "distinct", "duplicate %"
    );
    let h = report("MAESTRO (real repertoire)", &subsample(&human, target));
    let o = report("symthaea (ours)", &subsample(&ours, target));

    println!(
        "\n  ours - repertoire: {:+.1} percentage points",
        (o - h) * 100.0
    );
    println!(
        "\n  Similar duplicate rates ⇒ the \"memorable melodic identity\" thesis is\n  \
         another proxy artefact and melodic repetition is normal. Ours much higher\n  \
         ⇒ the thesis holds and this is the target. Confound 1 (top-line\n  \
         extraction wanders between voices in dense piano writing) inflates the\n  \
         REPERTOIRE's diversity, so \"ours ≈ theirs\" is the robust direction."
    );
}
