// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Does a Passacaglia actually sound like a passacaglia? A premise detector.
//!
//! A generic feature vector cannot answer that. A passacaglia's identity is one
//! specific, checkable thing: **a repeating bass ground, perceptible as a
//! repetition, with variation happening above it.** This measures exactly that
//! and nothing else, so a change can be judged against the premise rather than
//! against "does the table look different".
//!
//! # It measures PARTS, not roles
//!
//! Gated on [`Score::has_part_identity`]. `VoiceRole` would be the wrong tool:
//! it answers "what is this line doing", not "which line is it", and grouping
//! by it silently merges two lines that share a role. This probe declines to
//! answer rather than report a role-based number as if it were a voice-based
//! one — the substitution that produced a false "textbook staggered exposition"
//! claim on 2026-07-29.
//!
//! # Baseline first, change second
//!
//! Run this and freeze the numbers BEFORE altering the generator. Then make one
//! change (e.g. exposing the ground alone for its first cycle) and rerun the
//! identical seed cohort. Without a frozen baseline a "better" reading is
//! unfalsifiable.
//!
//! Usage:
//!   cargo run --release -p symthaea-muse --features theory \
//!       --example passacaglia_premise_probe -- [bars]

use std::collections::BTreeMap;

use symthaea_music_theory::composer::compose_styled;
use symthaea_music_theory::score::{PartId, Score, ScoreNote};
use symthaea_music_theory::{MusicalIntent, PitchClass, Style};

/// The frozen cohort. Fixed so a later run is comparable by construction.
const SEEDS: &[u64] = &[1, 2, 3, 5, 7, 11, 13, 17, 19, 23];

/// The ground is the lowest-sounding declared part. Chosen by mean pitch rather
/// than by assuming `PartId(0)`, so this keeps working if the generator
/// renumbers its parts — but still never falls back to `VoiceRole`.
fn ground_part(score: &Score) -> Option<PartId> {
    let mut mean: BTreeMap<u16, (f64, usize)> = BTreeMap::new();
    for n in &score.notes {
        if !n.part.is_assigned() {
            continue;
        }
        let e = mean.entry(n.part.0).or_insert((0.0, 0));
        e.0 += n.pitch.midi() as f64;
        e.1 += 1;
    }
    mean.into_iter()
        .min_by(|a, b| {
            (a.1.0 / a.1.1 as f64)
                .partial_cmp(&(b.1.0 / b.1.1 as f64))
                .unwrap()
        })
        .map(|(id, _)| PartId(id))
}

fn part_notes(score: &Score, part: PartId) -> Vec<ScoreNote> {
    let mut v: Vec<ScoreNote> = score
        .notes
        .iter()
        .copied()
        .filter(|n| n.part == part)
        .collect();
    v.sort_by(|a, b| a.onset.beats().partial_cmp(&b.onset.beats()).unwrap());
    v
}

/// How alike are two note runs, in pitch and in rhythm, reported separately?
///
/// Separate because they fail differently: a ground whose pitches repeat but
/// whose rhythm drifts is still recognisable; one whose rhythm repeats over
/// changing pitches is not a ground at all.
fn similarity(a: &[ScoreNote], b: &[ScoreNote]) -> (f64, f64) {
    let n = a.len().min(b.len());
    if n == 0 {
        return (0.0, 0.0);
    }
    let pitch = (0..n)
        .filter(|&i| a[i].pitch.midi() == b[i].pitch.midi())
        .count();
    let rhythm = (0..n)
        .filter(|&i| (a[i].duration.beats() - b[i].duration.beats()).abs() < 1e-6)
        .count();
    (pitch as f64 / n as f64, rhythm as f64 / n as f64)
}

struct GroundReport {
    seed: u64,
    has_parts: bool,
    ground_notes: usize,
    /// Length of one ground statement, in notes, inferred by finding the
    /// shortest prefix that repeats.
    period_notes: Option<usize>,
    period_beats: f64,
    repetitions: usize,
    pitch_sim: f64,
    rhythm_sim: f64,
    /// Seconds until the ground has stated itself twice — the earliest a
    /// listener could possibly hear it AS a repetition. A ground you cannot
    /// yet know is repeating is not yet doing its job.
    perceptible_at_s: Option<f64>,
    /// Do the upper parts sound during the FIRST ground cycle?
    /// `None` = NOT MEASURED (no period was inferred, so there is no first
    /// cycle to test). Distinct from `Some(false)` = measured, not obscured.
    upper_obscures_first_cycle: Option<bool>,
    /// Does material above the ground actually change between cycles? A
    /// passacaglia whose upper voices repeat identically is a loop, not a
    /// variation set.
    /// `None` = NOT MEASURED. Absence of a measurement is not a negative
    /// result, and collapsing the two is how an unmeasured seed becomes
    /// evidence for a change nobody verified was needed.
    upper_varies: Option<bool>,
}

/// Infer the ground period: the shortest prefix length whose repeat matches
/// well in both pitch and rhythm. Returns None when nothing repeats — which is
/// itself the finding, not an error.
fn infer_period_at(g: &[ScoreNote], thresh: f64) -> Option<usize> {
    let max = g.len() / 2;
    (2..=max).find(|&p| {
        let (ps, rs) = similarity(&g[..p], &g[p..(2 * p).min(g.len())]);
        ps > thresh && rs > thresh
    })
}

/// How strongly does period `p` hold across the WHOLE bass, not just its first
/// repeat? Mean pitch similarity of every later statement against the first.
///
/// The old detector compared cycle 1 to cycle 2 only and gated on a cutoff I
/// chose. Both were wrong: a ground whose second statement is the ornamented
/// one failed outright, and the cutoff turned a continuous quantity into a
/// yes/no that 8 of 10 seeds fell the wrong side of.
fn period_score(g: &[ScoreNote], p: usize) -> (f64, usize) {
    let reps = g.len() / p;
    if reps < 2 {
        return (0.0, reps);
    }
    let sims: Vec<f64> = (1..reps)
        .map(|k| similarity(&g[..p], &g[k * p..(k + 1) * p]).0)
        .collect();
    (sims.iter().sum::<f64>() / sims.len() as f64, reps)
}

/// A ground is defined by RECURRENCE, and two statements are not recurrence —
/// they are one thing happening twice. You cannot hear a pattern as a returning
/// ground until it returns a second time, so three statements is the floor.
///
/// This is not a tuning knob: at `reps >= 2` the search happily returned
/// "period 33, repeats twice" for a 73-note bass, i.e. half the line matching
/// its other half. That scores high and means nothing a listener could track.
const MIN_GROUND_REPETITIONS: usize = 3;

/// The best-supported period, by score rather than by threshold. Prefers more
/// repetitions at equal similarity — a 6-note cell recurring eight times is
/// stronger evidence of a ground than a 30-note span recurring three times.
///
/// Selected by MARGIN OVER THE NULL, not by any similarity/repetition product.
///
/// Both weightings tried before were wrong, in opposite directions. `sqrt(reps)`
/// let "period 33, repeats twice" — half the bass matching its other half — win.
/// Linear `reps` then let period-2 noise win, because 0.16x36 beats a real
/// 6-note ground's 0.80x6; mean margin over control collapsed to +0.00 and only
/// 1 of 10 seeds stayed above it. There is no principled exchange rate between
/// similarity and repetition count, so optimising their product is guesswork.
///
/// The control already solves this and I had failed to use it for selection: a
/// period-2 "ground" scores high on a SHUFFLED bass too, so its margin is
/// small, while a genuine 6-note cell beats its shuffle by a wide margin.
/// Maximising margin needs no exchange rate — chance is measured, not assumed.
fn best_period(g: &[ScoreNote], seed: u64) -> Option<(usize, f64, usize)> {
    let shuffled = shuffled_bass(g, seed);
    (2..=g.len() / 2)
        .map(|p| {
            let (score, reps) = period_score(g, p);
            let (null, _) = period_score(&shuffled, p);
            (p, score, reps, score - null)
        })
        .filter(|&(_, _, reps, _)| reps >= MIN_GROUND_REPETITIONS)
        .max_by(|a, b| a.3.partial_cmp(&b.3).unwrap())
        .map(|(p, score, reps, _)| (p, score, reps))
}

/// What score does this bass get with its pitches shuffled? The control that
/// says whether a similarity of 0.7 is a ground or just what any bass of this
/// length and pitch distribution scores by chance. Deterministic per call.
fn shuffled_bass(g: &[ScoreNote], seed: u64) -> Vec<ScoreNote> {
    let mut pitches: Vec<u8> = g.iter().map(|n| n.pitch.midi()).collect();
    let mut st = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    for i in (1..pitches.len()).rev() {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        pitches.swap(i, (st >> 33) as usize % (i + 1));
    }
    let mut shuffled = g.to_vec();
    for (n, &m) in shuffled.iter_mut().zip(pitches.iter()) {
        n.pitch = symthaea_music_theory::pitch::Pitch::from_midi(m);
    }
    shuffled
}

/// The control's own best score, for reporting alongside the real one.
fn null_score(g: &[ScoreNote], seed: u64) -> f64 {
    let sh = shuffled_bass(g, seed);
    (2..=sh.len() / 2)
        .map(|p| period_score(&sh, p))
        .filter(|&(_, reps)| reps >= MIN_GROUND_REPETITIONS)
        .map(|(s, _)| s)
        .fold(0.0f64, f64::max)
}

fn infer_period(g: &[ScoreNote]) -> Option<usize> {
    infer_period_at(g, 0.9)
}

/// Threshold sensitivity: is "no ground" a property of the music or of my
/// 0.9 cutoff? A ground that is ornamented between statements is still a
/// ground; a cutoff that rejects it is a detector defect, not a finding.
fn threshold_sweep(g: &[ScoreNote]) -> Vec<(f64, Option<usize>)> {
    [0.9, 0.8, 0.7, 0.6, 0.5]
        .into_iter()
        .map(|t| (t, infer_period_at(g, t)))
        .collect()
}

fn probe(seed: u64, bars: usize) -> GroundReport {
    let intent = MusicalIntent {
        valence: 0.0,
        arousal: 0.5,
        energy: 0.5,
        bars,
        seed,
        tonic: PitchClass::C,
    };
    let score = compose_styled(&intent, Style::Passacaglia);
    let spb = 60.0 / score.tempo_bpm as f64;

    let mut r = GroundReport {
        seed,
        has_parts: score.has_part_identity(),
        ground_notes: 0,
        period_notes: None,
        period_beats: 0.0,
        repetitions: 0,
        pitch_sim: 0.0,
        rhythm_sim: 0.0,
        perceptible_at_s: None,
        upper_obscures_first_cycle: None,
        upper_varies: None,
    };
    if !r.has_parts {
        return r; // decline rather than answer from roles
    }
    let Some(gp) = ground_part(&score) else {
        return r;
    };
    let g = part_notes(&score, gp);
    r.ground_notes = g.len();

    let Some(p) = infer_period(&g) else {
        return r;
    };
    r.period_notes = Some(p);
    r.period_beats = g[..p].iter().map(|n| n.duration.beats()).sum();
    r.repetitions = g.len() / p;

    // Average similarity of every later cycle against the first.
    let mut ps = Vec::new();
    let mut rs = Vec::new();
    for k in 1..r.repetitions {
        let (a, b) = similarity(&g[..p], &g[k * p..(k + 1) * p]);
        ps.push(a);
        rs.push(b);
    }
    r.pitch_sim = if ps.is_empty() {
        0.0
    } else {
        ps.iter().sum::<f64>() / ps.len() as f64
    };
    r.rhythm_sim = if rs.is_empty() {
        0.0
    } else {
        rs.iter().sum::<f64>() / rs.len() as f64
    };

    if r.repetitions >= 2 {
        // Earliest a repetition can be heard: the end of the second statement.
        let end2 = g[2 * p - 1].onset.beats() + g[2 * p - 1].duration.beats();
        r.perceptible_at_s = Some(end2 * spb);
    }

    let cycle1_end = g[p - 1].onset.beats() + g[p - 1].duration.beats();
    let upper: Vec<&ScoreNote> = score.notes.iter().filter(|n| n.part != gp).collect();
    r.upper_obscures_first_cycle = Some(upper.iter().any(|n| n.onset.beats() < cycle1_end));

    // Compare upper material in cycle 1 vs cycle 2 windows.
    if r.repetitions >= 2 {
        let c2_end = g[2 * p - 1].onset.beats() + g[2 * p - 1].duration.beats();
        let w1: Vec<u8> = upper
            .iter()
            .filter(|n| n.onset.beats() < cycle1_end)
            .map(|n| n.pitch.midi())
            .collect();
        let w2: Vec<u8> = upper
            .iter()
            .filter(|n| n.onset.beats() >= cycle1_end && n.onset.beats() < c2_end)
            .map(|n| n.pitch.midi())
            .collect();
        r.upper_varies = Some(w1 != w2);
    }
    r
}

fn main() {
    let bars: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    println!(
        "PASSACAGLIA PREMISE BASELINE — bars={bars}, {} frozen seeds\n\
         Measures the premise itself: a repeating bass ground, perceptible AS a\n\
         repetition, with variation above it. Parts, not roles.\n",
        SEEDS.len()
    );
    println!(
        "{:>5} {:>6} {:>7} {:>7} {:>6} {:>7} {:>7} {:>9} {:>8} {:>7}",
        "seed",
        "parts",
        "gnotes",
        "period",
        "reps",
        "pitchS",
        "rhythS",
        "percep_s",
        "obscured",
        "upperVar"
    );

    let reports: Vec<GroundReport> = SEEDS.iter().map(|&s| probe(s, bars)).collect();
    for r in &reports {
        println!(
            "{:>5} {:>6} {:>7} {:>7} {:>6} {:>7.2} {:>7.2} {:>9} {:>8} {:>7}",
            r.seed,
            r.has_parts,
            r.ground_notes,
            r.period_notes
                .map(|p| p.to_string())
                .unwrap_or("none".into()),
            r.repetitions,
            r.pitch_sim,
            r.rhythm_sim,
            r.perceptible_at_s
                .map(|s| format!("{s:.1}"))
                .unwrap_or("never".into()),
            r.upper_obscures_first_cycle
                .map(|b| b.to_string())
                .unwrap_or("n/m".into()),
            r.upper_varies
                .map(|b| b.to_string())
                .unwrap_or("n/m".into())
        );
    }

    // ── The premise verdict, stated as criteria rather than a score ─────────
    let n = reports.len() as f64;
    let with_period = reports.iter().filter(|r| r.period_notes.is_some()).count();
    let ge3 = reports.iter().filter(|r| r.repetitions >= 3).count();
    let obs_measured = reports
        .iter()
        .filter(|r| r.upper_obscures_first_cycle.is_some())
        .count();
    let obscured = reports
        .iter()
        .filter(|r| r.upper_obscures_first_cycle == Some(true))
        .count();
    let var_measured = reports.iter().filter(|r| r.upper_varies.is_some()).count();
    let varies = reports
        .iter()
        .filter(|r| r.upper_varies == Some(true))
        .count();
    let mean_percep: Vec<f64> = reports.iter().filter_map(|r| r.perceptible_at_s).collect();

    println!("\nPREMISE CRITERIA ({} seeds)", reports.len());
    println!(
        "  detectable repeating ground      {with_period}/{}",
        reports.len()
    );
    println!("  ground repeats >= 3 times        {ge3}/{}", reports.len());
    println!(
        "  ground exposed alone first cycle  {}/{obs_measured} MEASURED (obscured in \
         {obscured}; {} seeds NOT MEASURABLE — no period inferred, so there is no \
         first cycle to test)",
        obs_measured - obscured,
        reports.len() - obs_measured
    );
    println!(
        "  upper material varies per cycle  {varies}/{var_measured} MEASURED ({} NOT MEASURABLE)",
        reports.len() - var_measured
    );
    if !mean_percep.is_empty() {
        println!(
            "  mean time-to-perceptible         {:.1}s",
            mean_percep.iter().sum::<f64>() / mean_percep.len() as f64
        );
    }

    // Detector validation, printed before any conclusion is drawn from the
    // table above.
    // The scored detector: every seed gets an answer, with a null control so
    // the number means something.
    println!("\nSCORED GROUND DETECTION — no threshold, with a shuffled-bass control");
    println!(
        "  {:>5} {:>7} {:>6} {:>7} {:>7} {:>8}",
        "seed", "period", "reps", "score", "null", "margin"
    );
    let mut margins = Vec::new();
    for &seed in SEEDS {
        let intent = MusicalIntent {
            valence: 0.0,
            arousal: 0.5,
            energy: 0.5,
            bars,
            seed,
            tonic: PitchClass::C,
        };
        let score_ = compose_styled(&intent, Style::Passacaglia);
        let Some(gp) = ground_part(&score_) else {
            continue;
        };
        let g = part_notes(&score_, gp);
        match best_period(&g, seed) {
            Some((p, sc, reps)) => {
                let nl = null_score(&g, seed);
                margins.push(sc - nl);
                println!(
                    "  {seed:>5} {p:>7} {reps:>6} {sc:>7.2} {nl:>7.2} {:>8.2}",
                    sc - nl
                );
            }
            None => println!(
                "  {seed:>5} {:>7} {:>6} {:>7} {:>7} {:>8}",
                "none", 0, "-", "-", "-"
            ),
        }
    }
    if !margins.is_empty() {
        let mean = margins.iter().sum::<f64>() / margins.len() as f64;
        let above = margins.iter().filter(|&&m| m > 0.1).count();
        println!(
            "  mean margin over shuffled control: {mean:+.2}   \
             seeds clearly above control (>0.10): {above}/{}",
            margins.len()
        );
        println!(
            "  A margin near zero means the detected period is no better than\n\
             \x20 what a shuffled bass of the same notes scores — i.e. not a ground."
        );
    }

    println!("\nTHRESHOLD SENSITIVITY — does relaxing the match cutoff find grounds?");
    println!(
        "  {:>5} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "seed", "t=0.9", "t=0.8", "t=0.7", "t=0.6", "t=0.5"
    );
    for &seed in SEEDS {
        let intent = MusicalIntent {
            valence: 0.0,
            arousal: 0.5,
            energy: 0.5,
            bars,
            seed,
            tonic: PitchClass::C,
        };
        let score = compose_styled(&intent, Style::Passacaglia);
        let Some(gp) = ground_part(&score) else {
            continue;
        };
        let g = part_notes(&score, gp);
        let cells: Vec<String> = threshold_sweep(&g)
            .into_iter()
            .map(|(_, p)| p.map(|v| v.to_string()).unwrap_or("-".into()))
            .collect();
        println!(
            "  {:>5} {:>8} {:>8} {:>8} {:>8} {:>8}",
            seed, cells[0], cells[1], cells[2], cells[3], cells[4]
        );
    }
    println!(
        "  A column that stays '-' all the way to 0.5 means the bass genuinely\n\
         \x20 does not restate. Periods appearing as the cutoff drops mean the\n\
         \x20 ground IS repeating but varied, and 0.9 was too strict."
    );

    println!(
        "\nNOT MEASURED, deliberately: whether any of this is AUDIBLE. These are\n\
         symbolic properties of the score. A ground that repeats perfectly on\n\
         paper can still be buried by register, dynamics or timbre in the render.\n\
         That is what the A/B pack is for."
    );
    let _ = n;
}
