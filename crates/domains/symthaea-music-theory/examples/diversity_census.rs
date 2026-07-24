// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Diversity census: measures whether Muse's generated pieces collapse into a
//! small number of perceptually-similar clusters, or are already reasonably
//! diverse — using ONLY data `compose_with_spec_and_form` already produces
//! (`Score`, `Option<Form>`, `Key`). Symbolic/composition-level only: no audio
//! synthesis, no embeddings, no human judgment baked in as ground truth —
//! those are explicitly out of scope (see the "Census v1 — scope" design and
//! the "Causal Diversity Validation" follow-up). This is a measurement tool;
//! it does not change generation.
//!
//! ## Two fingerprint tiers
//! - **Exact**: a hash of the canonicalized note-event stream (role, pitch,
//!   onset, duration only — velocity/emphasis/section_intensity stripped as
//!   cosmetic) — catches byte-for-byte-equivalent pieces.
//! - **Structural**: a fixed 40-dim feature vector, in five named LAYERS (see
//!   `LAYERS` below): form shape (12d), harmonic trajectory (8d), a
//!   per-voice-role register/count profile as an orchestration proxy (12d), a
//!   rhythmic-duration histogram (5d), and melodic contour (3d).
//!
//! ## A note on "performance realization"
//! `symthaea-muse::theory_realize` (a DOWNSTREAM crate — it depends on this
//! one, not the reverse) has a genuine performance-realization layer distinct
//! from the symbolic `Score`: swing, structure-driven rubato, and a learned
//! velocity/articulation model (see `theory_realize.rs`'s `humanize_seed`,
//! `swing_beat`, `Rubato`, and the velocity/articulation predictor). This
//! census cannot reach that layer without either a reverse dependency or
//! duplicating its fingerprint code in `symthaea-muse` — out of scope here.
//! Every fingerprint in this file is therefore COMPOSITION-level; the
//! "same score, different performance" invariance bucket is reported as
//! "not measurable at this layer", not fabricated.
//!
//! ## Two passes
//! 1. **Pilot** (unchanged from the original Phase A run): 2,900 pieces,
//!    tonic derived from seed (`(seed*7)%12`) — kept for continuity/
//!    reproducibility of the original result.
//! 2. **Causal Diversity Validation** (this follow-up): a larger, DECORRELATED
//!    corpus (tonic and seed independently swept, not derived from each
//!    other), a per-style/per-parameter one-factor-at-a-time causal
//!    sensitivity matrix, an invariance-bucket breakdown of near-duplicate
//!    pairs, a distance-threshold sensitivity curve, cluster/concentration
//!    metrics, and a candidate pair-labeling artifact for future real human
//!    review (explicitly marked as unvalidated AI-assisted scaffolding, not
//!    ground truth).
//!
//! Run: `cargo run --release -p symthaea-music-theory --example diversity_census`
//! Env overrides (for smoke-testing before a full run):
//!   CENSUS_V2_TONICS_PER_STYLE, CENSUS_V2_SEEDS_PER_TONIC (defaults below)

use std::collections::HashMap;

use symthaea_music_theory::composer::compose_with_spec_and_form;
use symthaea_music_theory::fingerprint::{
    self, LAYERS as FP_LAYERS, STRUCT_DIMS as FP_STRUCT_DIMS,
};
use symthaea_music_theory::{MusicalIntent, PitchClass, Style};

const STYLES: &[Style] = &[
    Style::Classical,
    Style::Waltz,
    Style::Folk,
    Style::Cinematic,
    Style::Playful,
    Style::Nocturne,
    Style::March,
    Style::Lullaby,
    Style::ModalFolk,
    Style::Fugue,
    Style::Passacaglia,
    Style::Tango,
    Style::Celtic,
    Style::Blues,
    Style::Impressionism,
    Style::SacredChoral,
    Style::Minimalism,
    Style::JazzBallad,
    Style::BaroqueSuite,
    Style::ProgFolk,
    Style::Ambient,
    Style::Sonata,
    Style::RenaissancePolyphony,
    Style::AfroCuban,
    Style::Flamenco,
    Style::BossaNova,
    Style::Opera,
    Style::IrishTraditional,
    Style::HindustaniInspired,
];

/// (valence, arousal, energy, bars) — a small spread of expressive intents,
/// not an exhaustive sweep.
const VARIATIONS: &[(f32, f32, f32, usize)] = &[
    (0.0, 0.5, 0.5, 16),
    (0.8, 0.7, 0.7, 16),
    (-0.7, 0.3, 0.4, 16),
    (0.2, 0.5, 0.5, 24),
];

const SEEDS_PER_STYLE: u64 = 25;

// Fingerprint dimensionality/layers/exact+structural-hash/distance logic all
// now live in `symthaea_music_theory::fingerprint` (extracted so both this
// census and the Muse Atlas endpoint call the same code) — thin local
// aliases below keep the rest of this file's call sites unchanged.
const STRUCT_DIMS: usize = FP_STRUCT_DIMS;
const NUM_LAYERS: usize = symthaea_music_theory::fingerprint::NUM_LAYERS;
const LAYERS: [(&str, usize, usize); NUM_LAYERS] = FP_LAYERS;

use fingerprint::exact_fingerprint;
use fingerprint::structural_fingerprint;

fn dist(a: &[f64; STRUCT_DIMS], b: &[f64; STRUCT_DIMS]) -> f64 {
    fingerprint::dist(a, b)
}

/// Per-layer L2 distances, in `LAYERS` order. Array size bumped from 5 to
/// `NUM_LAYERS` for fingerprint v2 (5 new layers appended after the
/// original 5) — mechanical only, indices 0..=4 (form/harmony/
/// orchestration/rhythm/contour) keep their original meaning, so every
/// existing `ld[0..=4]` use below is unaffected.
fn layer_dists(a: &[f64; STRUCT_DIMS], b: &[f64; STRUCT_DIMS]) -> [f64; NUM_LAYERS] {
    fingerprint::layer_dists(a, b)
}

#[derive(Clone)]
struct Piece {
    style: Style,
    seed: u64,
    tonic: i32,
    variation_idx: usize,
    exact_hash: u64,
    structural: [f64; STRUCT_DIMS],
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.total_cmp(b));
    percentile(&xs, 0.5)
}

fn compose_piece(style: Style, tonic: i32, seed: u64, variation_idx: usize) -> Piece {
    let (valence, arousal, energy, bars) = VARIATIONS[variation_idx];
    let intent = MusicalIntent {
        valence,
        arousal,
        energy,
        bars,
        seed,
        tonic: PitchClass::new(tonic),
    };
    let (score, form) = compose_with_spec_and_form(&intent, &style.spec());
    Piece {
        style,
        seed,
        tonic,
        variation_idx,
        exact_hash: exact_fingerprint(&score),
        structural: structural_fingerprint(&score, &form),
    }
}

/// Sensitivity class for one (style, parameter) cell, based on the WORST-CASE
/// (max over tested perturbations) overall structural distance from a fixed
/// base configuration. "Worst-case" rather than "average" because a knob
/// that only matters at some settings (e.g. only affects minor-mode pieces)
/// should still show up as non-dead.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Sensitivity {
    None,
    Low,
    Medium,
    High,
}

impl Sensitivity {
    fn classify(max_overall_dist: f64, exact_hash_ever_changed: bool) -> Self {
        // Thresholds are fractions of the pilot's own same-style median
        // distance scale (~0.19 in the original run) — reused here as the
        // empirical "normal variation" yardstick rather than an arbitrary
        // absolute cutoff.
        const NORMAL_SCALE: f64 = 0.19;
        if !exact_hash_ever_changed && max_overall_dist < 0.01 {
            Sensitivity::None
        } else if max_overall_dist < 0.25 * NORMAL_SCALE {
            Sensitivity::Low
        } else if max_overall_dist < 0.75 * NORMAL_SCALE {
            Sensitivity::Medium
        } else {
            Sensitivity::High
        }
    }
    fn label(self) -> &'static str {
        match self {
            Sensitivity::None => "None",
            Sensitivity::Low => "Low",
            Sensitivity::Medium => "Medium",
            Sensitivity::High => "High",
        }
    }
}

/// One-factor-at-a-time causal sensitivity matrix: for each style, hold a
/// base config fixed and vary ONE parameter across several test values,
/// measuring whether the exact hash and each structural layer actually
/// change. Returns rows of (style, param_name, Sensitivity, layers_touched).
fn causal_sensitivity_matrix() -> Vec<(Style, &'static str, Sensitivity, Vec<&'static str>)> {
    const BASE_SEED: u64 = 90_210;
    const BASE_TONIC: i32 = 0;
    const BASE_VALENCE: f32 = 0.0;
    const BASE_AROUSAL: f32 = 0.5;
    const BASE_ENERGY: f32 = 0.5;
    const BASE_BARS: usize = 16;

    fn compose_raw(
        style: Style,
        tonic: i32,
        seed: u64,
        valence: f32,
        arousal: f32,
        energy: f32,
        bars: usize,
    ) -> (u64, [f64; STRUCT_DIMS]) {
        let intent = MusicalIntent {
            valence,
            arousal,
            energy,
            bars,
            seed,
            tonic: PitchClass::new(tonic),
        };
        let (score, form) = compose_with_spec_and_form(&intent, &style.spec());
        (
            exact_fingerprint(&score),
            structural_fingerprint(&score, &form),
        )
    }

    let mut rows = Vec::new();
    for &style in STYLES {
        let (base_hash, base_struct) = compose_raw(
            style,
            BASE_TONIC,
            BASE_SEED,
            BASE_VALENCE,
            BASE_AROUSAL,
            BASE_ENERGY,
            BASE_BARS,
        );

        // (param name, list of (tonic, seed, valence, arousal, energy, bars) test points)
        let params: [(&str, Vec<(i32, u64, f32, f32, f32, usize)>); 6] = [
            (
                "tonic",
                vec![
                    (
                        4,
                        BASE_SEED,
                        BASE_VALENCE,
                        BASE_AROUSAL,
                        BASE_ENERGY,
                        BASE_BARS,
                    ),
                    (
                        7,
                        BASE_SEED,
                        BASE_VALENCE,
                        BASE_AROUSAL,
                        BASE_ENERGY,
                        BASE_BARS,
                    ),
                    (
                        10,
                        BASE_SEED,
                        BASE_VALENCE,
                        BASE_AROUSAL,
                        BASE_ENERGY,
                        BASE_BARS,
                    ),
                ],
            ),
            (
                "valence",
                vec![
                    (
                        BASE_TONIC,
                        BASE_SEED,
                        -0.7,
                        BASE_AROUSAL,
                        BASE_ENERGY,
                        BASE_BARS,
                    ),
                    (
                        BASE_TONIC,
                        BASE_SEED,
                        0.8,
                        BASE_AROUSAL,
                        BASE_ENERGY,
                        BASE_BARS,
                    ),
                ],
            ),
            (
                "arousal",
                vec![
                    (
                        BASE_TONIC,
                        BASE_SEED,
                        BASE_VALENCE,
                        0.1,
                        BASE_ENERGY,
                        BASE_BARS,
                    ),
                    (
                        BASE_TONIC,
                        BASE_SEED,
                        BASE_VALENCE,
                        0.9,
                        BASE_ENERGY,
                        BASE_BARS,
                    ),
                ],
            ),
            (
                "energy",
                vec![
                    (
                        BASE_TONIC,
                        BASE_SEED,
                        BASE_VALENCE,
                        BASE_AROUSAL,
                        0.1,
                        BASE_BARS,
                    ),
                    (
                        BASE_TONIC,
                        BASE_SEED,
                        BASE_VALENCE,
                        BASE_AROUSAL,
                        0.9,
                        BASE_BARS,
                    ),
                ],
            ),
            (
                "bars",
                vec![
                    (
                        BASE_TONIC,
                        BASE_SEED,
                        BASE_VALENCE,
                        BASE_AROUSAL,
                        BASE_ENERGY,
                        8,
                    ),
                    (
                        BASE_TONIC,
                        BASE_SEED,
                        BASE_VALENCE,
                        BASE_AROUSAL,
                        BASE_ENERGY,
                        24,
                    ),
                ],
            ),
            (
                "seed",
                vec![
                    (
                        BASE_TONIC,
                        BASE_SEED + 1,
                        BASE_VALENCE,
                        BASE_AROUSAL,
                        BASE_ENERGY,
                        BASE_BARS,
                    ),
                    (
                        BASE_TONIC,
                        BASE_SEED + 2,
                        BASE_VALENCE,
                        BASE_AROUSAL,
                        BASE_ENERGY,
                        BASE_BARS,
                    ),
                    (
                        BASE_TONIC,
                        BASE_SEED + 3,
                        BASE_VALENCE,
                        BASE_AROUSAL,
                        BASE_ENERGY,
                        BASE_BARS,
                    ),
                ],
            ),
        ];

        for (pname, points) in params {
            let mut max_overall = 0.0f64;
            let mut hash_changed = false;
            let mut layers_touched: [bool; NUM_LAYERS] = [false; NUM_LAYERS];
            for (tonic, seed, valence, arousal, energy, bars) in points {
                let (h, s) = compose_raw(style, tonic, seed, valence, arousal, energy, bars);
                if h != base_hash {
                    hash_changed = true;
                }
                let overall = dist(&base_struct, &s);
                if overall > max_overall {
                    max_overall = overall;
                }
                let ld = layer_dists(&base_struct, &s);
                for (li, &d) in ld.iter().enumerate() {
                    if d > 0.02 {
                        layers_touched[li] = true;
                    }
                }
            }
            let sens = Sensitivity::classify(max_overall, hash_changed);
            let touched: Vec<&'static str> = LAYERS
                .iter()
                .zip(layers_touched.iter())
                .filter(|&(_, &t)| t)
                .map(|(&(name, _, _), _)| name)
                .collect();
            rows.push((style, pname, sens, touched));
        }
    }
    rows
}

/// Union-find for connected-components clustering at a fixed distance
/// threshold — cheap and exact given we already have all pairwise distances.
struct UnionFind {
    parent: Vec<usize>,
}
impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }
    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

struct CorpusReport {
    label: String,
    n: usize,
    exact_dup_rate: f64,
    largest_exact_group: usize,
    same_style_median: f64,
    diff_style_median: f64,
    near_dup_threshold: f64,
    near_dup_rate: f64,
    nn_dists: Vec<f64>,
    bleed_rows: Vec<(Style, f64, usize)>,
    cross_pairs: Vec<((Style, Style), usize)>,
    nearest: Vec<(f64, usize)>,
}

fn analyze_corpus(pieces: &[Piece], label: &str) -> CorpusReport {
    let n = pieces.len();

    let mut exact_groups: HashMap<u64, usize> = HashMap::new();
    for p in pieces {
        *exact_groups.entry(p.exact_hash).or_insert(0) += 1;
    }
    let exact_dup_pieces: usize = exact_groups.values().filter(|&&c| c > 1).map(|&c| c).sum();
    let exact_dup_rate = exact_dup_pieces as f64 / n as f64;
    let largest_exact_group = exact_groups.values().cloned().max().unwrap_or(1);

    if std::env::var("CENSUS_DEBUG_DUPS").is_ok() {
        let mut by_hash: HashMap<u64, Vec<usize>> = HashMap::new();
        for (idx, p) in pieces.iter().enumerate() {
            by_hash.entry(p.exact_hash).or_default().push(idx);
        }
        let mut groups: Vec<&Vec<usize>> = by_hash.values().filter(|g| g.len() > 1).collect();
        groups.sort_by(|a, b| b.len().cmp(&a.len()));
        eprintln!("\n--- CENSUS_DEBUG_DUPS ({label}): top exact-duplicate groups ---");
        for g in groups.iter().take(8) {
            let members: Vec<String> = g
                .iter()
                .map(|&i| {
                    format!(
                        "{:?}/tonic{}/seed{}/var{}",
                        pieces[i].style, pieces[i].tonic, pieces[i].seed, pieces[i].variation_idx
                    )
                })
                .collect();
            eprintln!("group of {}: {}", g.len(), members.join(", "));
        }
    }

    let mut nearest: Vec<(f64, usize)> = vec![(f64::MAX, usize::MAX); n];
    let mut same_style_dists: Vec<f64> = Vec::new();
    let mut diff_style_dists: Vec<f64> = Vec::new();
    let mut nn_style_tally: HashMap<(Style, Style), usize> = HashMap::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist(&pieces[i].structural, &pieces[j].structural);
            if pieces[i].style == pieces[j].style {
                same_style_dists.push(d);
            } else {
                diff_style_dists.push(d);
            }
            if d < nearest[i].0 {
                nearest[i] = (d, j);
            }
            if d < nearest[j].0 {
                nearest[j] = (d, i);
            }
        }
    }
    for (i, &(_, j)) in nearest.iter().enumerate() {
        if j != usize::MAX {
            *nn_style_tally
                .entry((pieces[i].style, pieces[j].style))
                .or_insert(0) += 1;
        }
    }

    let same_style_median = median(same_style_dists.clone());
    let diff_style_median = median(diff_style_dists.clone());
    let near_dup_threshold = 0.25 * same_style_median;

    let near_dup_count = nearest
        .iter()
        .filter(|&&(d, _)| d < near_dup_threshold)
        .count();
    let near_dup_rate = near_dup_count as f64 / n as f64;

    let mut nn_dists: Vec<f64> = nearest.iter().map(|&(d, _)| d).collect();
    nn_dists.sort_by(|a, b| a.total_cmp(b));

    let mut style_total: HashMap<Style, usize> = HashMap::new();
    let mut style_bleed: HashMap<Style, usize> = HashMap::new();
    for (i, &(_, j)) in nearest.iter().enumerate() {
        *style_total.entry(pieces[i].style).or_insert(0) += 1;
        if j != usize::MAX && pieces[j].style != pieces[i].style {
            *style_bleed.entry(pieces[i].style).or_insert(0) += 1;
        }
    }
    let mut bleed_rows: Vec<(Style, f64, usize)> = style_total
        .iter()
        .map(|(&s, &total)| {
            let bleed = *style_bleed.get(&s).unwrap_or(&0);
            (s, bleed as f64 / total as f64, total)
        })
        .collect();
    bleed_rows.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut cross_pairs: Vec<((Style, Style), usize)> = nn_style_tally
        .iter()
        .filter(|((a, b), _)| a != b)
        .map(|(&k, &v)| (k, v))
        .collect();
    cross_pairs.sort_by(|a, b| b.1.cmp(&a.1));

    CorpusReport {
        label: label.to_string(),
        n,
        exact_dup_rate,
        largest_exact_group,
        same_style_median,
        diff_style_median,
        near_dup_threshold,
        near_dup_rate,
        nn_dists,
        bleed_rows,
        cross_pairs,
        nearest,
    }
}

fn print_pilot_style_report(r: &CorpusReport) {
    println!("Corpus: {} pieces\n", r.n);
    println!("## Exact duplicates");
    println!(
        "- exact-duplicate rate: {:.4}% ({}/{} pieces share a hash with at least one other piece)",
        r.exact_dup_rate * 100.0,
        (r.exact_dup_rate * r.n as f64).round() as usize,
        r.n
    );
    println!(
        "- largest exact-duplicate group: {}\n",
        r.largest_exact_group
    );

    println!("## Structural near-duplicates");
    println!(
        "- median same-style pairwise distance: {:.4}",
        r.same_style_median
    );
    println!(
        "- median cross-style pairwise distance: {:.4}",
        r.diff_style_median
    );
    println!("- near-duplicate threshold: {:.4}", r.near_dup_threshold);
    println!(
        "- near-duplicate rate: {:.4}% ({}/{} pieces have a nearest neighbor closer than the threshold)\n",
        r.near_dup_rate * 100.0,
        (r.near_dup_rate * r.n as f64).round() as usize,
        r.n
    );

    println!("## Nearest-neighbor distance distribution (per piece, over the whole corpus)");
    println!("- min: {:.4}", r.nn_dists.first().copied().unwrap_or(0.0));
    println!("- p10: {:.4}", percentile(&r.nn_dists, 0.10));
    println!("- median: {:.4}", percentile(&r.nn_dists, 0.50));
    println!("- p90: {:.4}", percentile(&r.nn_dists, 0.90));
    println!("- max: {:.4}\n", r.nn_dists.last().copied().unwrap_or(0.0));

    println!(
        "## Per-style bleed (fraction of a style's pieces whose overall nearest neighbor is a DIFFERENT style)"
    );
    for (style, rate, total) in &r.bleed_rows {
        println!(
            "- {style:?}: {:.1}% ({}/{total})",
            rate * 100.0,
            (*rate * *total as f64).round() as usize
        );
    }

    println!("\n## Top cross-style nearest-neighbor pairs (possible bleed hot spots)");
    for ((a, b), count) in r.cross_pairs.iter().take(10) {
        println!("- {a:?} <-> {b:?}: {count}");
    }
}

fn main() {
    // ============================================================
    // PASS 1: original Phase A pilot, UNCHANGED — kept for
    // continuity/reproducibility of the original result.
    // ============================================================
    let mut pilot_pieces: Vec<Piece> = Vec::new();
    for &style in STYLES {
        for seed in 0..SEEDS_PER_STYLE {
            for variation_idx in 0..VARIATIONS.len() {
                let tonic = ((seed * 7) % 12) as i32;
                pilot_pieces.push(compose_piece(style, tonic, seed, variation_idx));
            }
        }
    }
    eprintln!(
        "[pilot] Composed {} pieces ({} styles x {SEEDS_PER_STYLE} seeds x {} variations).",
        pilot_pieces.len(),
        STYLES.len(),
        VARIATIONS.len()
    );
    let pilot_report = analyze_corpus(&pilot_pieces, "Phase A pilot (tonic derived from seed)");

    println!("# Muse diversity census\n");
    println!("## Pass 1 — Phase A pilot (unchanged, for continuity)\n");
    print_pilot_style_report(&pilot_report);

    // ============================================================
    // PASS 2: Causal Diversity Validation follow-up.
    // ============================================================
    let tonics_per_style: usize = std::env::var("CENSUS_V2_TONICS_PER_STYLE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12); // all 12 pitch classes
    let seeds_per_tonic: u64 = std::env::var("CENSUS_V2_SEEDS_PER_TONIC")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(11);

    let mut v2_pieces: Vec<Piece> = Vec::new();
    for &style in STYLES {
        for tonic in 0..tonics_per_style as i32 {
            for seed in 0..seeds_per_tonic {
                // Seeds are offset well clear of the pilot's own seed range
                // (0..25) and the causal matrix's base seed (90_210+) so
                // none of the three passes accidentally share a seed.
                let decorrelated_seed = 500_000 + tonic as u64 * 100_000 + seed;
                for variation_idx in 0..VARIATIONS.len() {
                    v2_pieces.push(compose_piece(
                        style,
                        tonic,
                        decorrelated_seed,
                        variation_idx,
                    ));
                }
            }
        }
    }
    eprintln!(
        "[v2] Composed {} pieces ({} styles x {tonics_per_style} tonics x {seeds_per_tonic} seeds x {} variations, tonic and seed fully DECORRELATED).",
        v2_pieces.len(),
        STYLES.len(),
        VARIATIONS.len()
    );
    let v2_report = analyze_corpus(
        &v2_pieces,
        "Causal Diversity Validation (decorrelated tonic/seed)",
    );

    println!("\n\n## Pass 2 — Causal Diversity Validation (decorrelated tonic/seed)\n");
    print_pilot_style_report(&v2_report);

    // --- 4. Distance-threshold sensitivity curve ---
    println!("\n## Threshold sensitivity curve (near-duplicate rate at fixed absolute distances)");
    let thresholds = [0.05, 0.10, 0.15, 0.1888, 0.25, 0.30];
    for &t in &thresholds {
        let count = v2_report.nn_dists.iter().filter(|&&d| d < t).count();
        println!(
            "- < {t:.4}: {:.2}% ({count}/{})",
            100.0 * count as f64 / v2_report.n as f64,
            v2_report.n
        );
    }

    // --- 5. Cluster / concentration metrics (connected components at the
    // corpus's own near-dup threshold) ---
    let n = v2_pieces.len();
    let mut uf = UnionFind::new(n);
    // Reuse the same O(n^2) pairwise pass structure as analyze_corpus, but
    // only need the threshold decision this time (cheaper per pair).
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist(&v2_pieces[i].structural, &v2_pieces[j].structural);
            if d < v2_report.near_dup_threshold {
                uf.union(i, j);
            }
        }
    }
    let mut cluster_sizes: HashMap<usize, usize> = HashMap::new();
    for i in 0..n {
        let root = uf.find(i);
        *cluster_sizes.entry(root).or_insert(0) += 1;
    }
    let mut sizes: Vec<usize> = cluster_sizes.values().cloned().collect();
    sizes.sort_by(|a, b| b.cmp(a));
    let occupied_clusters = sizes.len();
    let largest_cluster_share = sizes.first().copied().unwrap_or(0) as f64 / n as f64;
    // Shannon entropy (nats) of the cluster-size distribution -> effective
    // cluster count = exp(entropy). A corpus split evenly across K clusters
    // has effective count == K; heavy concentration in one cluster drives it
    // toward 1 regardless of how many tiny clusters also exist.
    let probs: Vec<f64> = sizes.iter().map(|&s| s as f64 / n as f64).collect();
    let entropy: f64 = -probs.iter().map(|&p| p * p.ln()).sum::<f64>();
    let effective_clusters = entropy.exp();

    println!(
        "\n## Cluster / concentration metrics (Pass 2 corpus, connected components at near-dup threshold {:.4})",
        v2_report.near_dup_threshold
    );
    println!("- occupied clusters: {occupied_clusters} (out of {n} pieces)");
    println!(
        "- largest cluster share: {:.2}% ({} pieces)",
        largest_cluster_share * 100.0,
        sizes.first().copied().unwrap_or(0)
    );
    println!("- effective cluster count (exp(entropy)): {effective_clusters:.1}");
    println!(
        "- cluster size distribution (top 15): {:?}",
        &sizes[..sizes.len().min(15)]
    );
    println!(
        "- median nearest-neighbor distance overall: {:.4}",
        percentile(&v2_report.nn_dists, 0.5)
    );
    println!("- median nearest-neighbor distance, per style:");
    let mut per_style_nn: HashMap<Style, Vec<f64>> = HashMap::new();
    for (i, &(d, _)) in v2_report.nearest.iter().enumerate() {
        per_style_nn.entry(v2_pieces[i].style).or_default().push(d);
    }
    let mut per_style_nn_rows: Vec<(Style, f64)> = per_style_nn
        .into_iter()
        .map(|(s, ds)| (s, median(ds)))
        .collect();
    per_style_nn_rows.sort_by(|a, b| a.1.total_cmp(&b.1));
    for (style, med) in &per_style_nn_rows {
        println!("  - {style:?}: {med:.4}");
    }

    // --- 2. Per-style causal sensitivity matrix ---
    println!(
        "\n## Per-style causal sensitivity matrix (one-factor-at-a-time, base seed=90210/tonic=C/valence=0/arousal=0.5/energy=0.5/bars=16)"
    );
    println!(
        "Sensitivity is the WORST CASE over tested perturbations of that parameter (a knob that matters only sometimes still counts as non-dead)."
    );
    let matrix = causal_sensitivity_matrix();
    let mut dead_cells: Vec<(Style, &str)> = Vec::new();
    for (style, param, sens, layers) in &matrix {
        println!(
            "- {style:?} / {param}: {} (layers touched: {})",
            sens.label(),
            if layers.is_empty() {
                "none".to_string()
            } else {
                layers.join(", ")
            }
        );
        if *sens == Sensitivity::None {
            dead_cells.push((*style, param));
        }
    }
    println!(
        "\n### Dead controls found ({} of {} style/parameter cells)",
        dead_cells.len(),
        matrix.len()
    );
    for (style, param) in &dead_cells {
        println!(
            "- {style:?}: `{param}` has NO measurable effect on composition-level output at these test points"
        );
    }

    // --- 3. Invariance-bucket categorization for Pass-2 near-duplicate pairs ---
    println!(
        "\n## Invariance-bucket breakdown of near-duplicate pairs (Pass 2, threshold {:.4})",
        v2_report.near_dup_threshold
    );
    let mut buckets: HashMap<&'static str, usize> = HashMap::new();
    let mut near_dup_pair_count = 0usize;
    for (i, &(d, j)) in v2_report.nearest.iter().enumerate() {
        if j == usize::MAX || d >= v2_report.near_dup_threshold || j < i {
            continue; // count each pair once, in the lower-index direction
        }
        near_dup_pair_count += 1;
        let a = &v2_pieces[i];
        let b = &v2_pieces[j];
        let bucket = if a.exact_hash == b.exact_hash {
            "same_symbolic_composition (exact hash match)"
        } else {
            let ld = layer_dists(&a.structural, &b.structural);
            let form_same = ld[0] < 0.02 && ld[4] < 0.02; // form + contour
            let harmony_diff = ld[1] > 0.02;
            let orch_diff = ld[2] > 0.02;
            if form_same && harmony_diff && !orch_diff {
                "same_structure_diff_harmony"
            } else if form_same && orch_diff {
                "same_motif_form_diff_orchestration"
            } else {
                "fully_distinct (below threshold on overall distance only)"
            }
        };
        *buckets.entry(bucket).or_insert(0) += 1;
    }
    println!(
        "(Note: \"same score, different performance\" is NOT measurable at this layer — performance realization lives downstream in symthaea-muse::theory_realize, which this composition-level crate cannot depend on. Not fabricated as a bucket.)"
    );
    for (bucket, count) in &buckets {
        println!(
            "- {bucket}: {count} ({:.1}% of {near_dup_pair_count} near-dup pairs)",
            100.0 * *count as f64 / near_dup_pair_count.max(1) as f64
        );
    }

    // --- 6. Candidate pair-labeling artifact (PENDING REAL HUMAN REVIEW) ---
    if let Ok(path) = std::env::var("CENSUS_LABELING_OUT") {
        let bins: [(f64, f64); 7] = [
            (0.0, 0.05),
            (0.05, 0.10),
            (0.10, 0.15),
            (0.15, 0.1888),
            (0.1888, 0.25),
            (0.25, 0.30),
            (0.30, f64::MAX),
        ];
        let mut out = String::new();
        out.push_str("# Muse diversity census — candidate pair-labeling set\n\n");
        out.push_str("**STATUS: PENDING REAL HUMAN/MUSICIAN REVIEW. NOT YET VALIDATED.**\n");
        out.push_str("Every \"AI heuristic\" line below is this script's own best-effort guess, NOT human ground truth — do not treat it as validated perceptual similarity.\n\n");
        for (lo, hi) in bins {
            out.push_str(&format!("## Distance band [{lo:.4}, {hi:.4})\n\n"));
            let mut shown = 0;
            for (i, &(d, j)) in v2_report.nearest.iter().enumerate() {
                if shown >= 25 || j == usize::MAX || j < i {
                    continue;
                }
                if d >= lo && d < hi {
                    let a = &v2_pieces[i];
                    let b = &v2_pieces[j];
                    let ld = layer_dists(&a.structural, &b.structural);
                    let heuristic = if a.exact_hash == b.exact_hash {
                        "byte-identical composition"
                    } else if ld.iter().all(|&x| x < 0.05) {
                        "likely near-identical to a listener"
                    } else if ld[0] < 0.05 {
                        "same form/shape, probably distinguishable on harmony/orchestration"
                    } else {
                        "probably distinguishable"
                    };
                    out.push_str(&format!(
                        "- pair (dist={d:.4}): {:?}(tonic={},seed={},var={}) vs {:?}(tonic={},seed={},var={}) — layer dists [form={:.3},harmony={:.3},orch={:.3},rhythm={:.3},contour={:.3}] — AI heuristic: {heuristic}\n",
                        a.style, a.tonic, a.seed, a.variation_idx,
                        b.style, b.tonic, b.seed, b.variation_idx,
                        ld[0], ld[1], ld[2], ld[3], ld[4]
                    ));
                    shown += 1;
                }
            }
            if shown == 0 {
                out.push_str("(no pairs found in this band)\n");
            }
            out.push('\n');
        }
        std::fs::write(&path, &out).expect("write labeling set");
        eprintln!("Wrote candidate labeling set ({path}) — PENDING REAL HUMAN REVIEW");
    }

    // JSON dump for programmatic follow-up.
    let json = serde_json::json!({
        "pilot": {
            "n_pieces": pilot_report.n,
            "exact_dup_rate": pilot_report.exact_dup_rate,
            "near_dup_rate": pilot_report.near_dup_rate,
        },
        "v2": {
            "n_pieces": v2_report.n,
            "tonics_per_style": tonics_per_style,
            "seeds_per_tonic": seeds_per_tonic,
            "exact_dup_rate": v2_report.exact_dup_rate,
            "near_dup_rate": v2_report.near_dup_rate,
            "near_dup_threshold": v2_report.near_dup_threshold,
            "occupied_clusters": occupied_clusters,
            "largest_cluster_share": largest_cluster_share,
            "effective_cluster_count": effective_clusters,
            "threshold_curve": thresholds.iter().map(|&t| {
                let count = v2_report.nn_dists.iter().filter(|&&d| d < t).count();
                serde_json::json!({"threshold": t, "rate": count as f64 / v2_report.n as f64})
            }).collect::<Vec<_>>(),
            "dead_controls": dead_cells.iter().map(|(s,p)| serde_json::json!({"style": format!("{s:?}"), "param": p})).collect::<Vec<_>>(),
            "invariance_buckets": buckets.iter().map(|(b,c)| serde_json::json!({"bucket": b, "count": c})).collect::<Vec<_>>(),
        },
    });
    if let Ok(path) = std::env::var("CENSUS_JSON_OUT") {
        std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap())
            .expect("write census JSON");
        eprintln!("Wrote {path}");
    }
}
