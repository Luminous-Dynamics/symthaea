// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Style Contract Benchmark v1.
//!
//! This is an executable audit of the real public composition path, not a
//! music-quality oracle. It measures mechanical validity, deterministic
//! replay, declared-vs-observed intent sensitivity, seed variation, and
//! exact/structural cross-style collisions. Human recognition, preference,
//! cultural authenticity, and artistic quality remain explicitly unmeasured.
//!
//! Run:
//!   cargo run --release -p symthaea-music-theory --example style_contract_benchmark
//!   cargo run --release -p symthaea-music-theory --example style_contract_benchmark -- /tmp/style-contract.json

use std::collections::HashSet;

use serde_json::{Value, json};
use symthaea_music_theory::fingerprint::{
    STRUCT_DIMS, dist, exact_fingerprint, structural_fingerprint,
};
use symthaea_music_theory::{
    MUSIC_THEORY_ENGINE_VERSION, MusicalIntent, PitchClass, Style, compose_with_grammar_plan,
};

const SEEDS: [u64; 4] = [0, 1, 7, 42];
const INTENTS: [(f32, f32, f32, usize); 4] = [
    (0.0, 0.5, 0.5, 16),
    (0.8, 0.7, 0.7, 16),
    (-0.7, 0.3, 0.4, 16),
    (0.2, 0.5, 0.5, 24),
];
const AXIS_SEEDS: [u64; 3] = [13, 29, 73];
const AXES: [&str; 4] = ["valence", "arousal", "energy", "bars"];

#[derive(Clone)]
struct Sample {
    style_index: usize,
    style_name: String,
    config_index: usize,
    exact_hash: u64,
    structural: [f64; STRUCT_DIMS],
}

struct StyleCounters {
    style_name: String,
    spec_valid: bool,
    samples: usize,
    empty_scores: usize,
    invalid_timeline: usize,
    meter_mismatches: usize,
    tempo_out_of_band: usize,
    deterministic_replay_failures: usize,
    seed_hashes: HashSet<u64>,
}

impl StyleCounters {
    fn new(style: Style) -> Self {
        Self {
            style_name: format!("{style:?}"),
            spec_valid: style.spec().validate().is_ok(),
            samples: 0,
            empty_scores: 0,
            invalid_timeline: 0,
            meter_mismatches: 0,
            tempo_out_of_band: 0,
            deterministic_replay_failures: 0,
            seed_hashes: HashSet::new(),
        }
    }
}

fn intent(
    valence: f32,
    arousal: f32,
    energy: f32,
    bars: usize,
    seed: u64,
) -> MusicalIntent {
    MusicalIntent {
        valence,
        arousal,
        energy,
        bars,
        seed,
        tonic: PitchClass::new(0),
    }
}

fn realize(style: Style, intent: &MusicalIntent) -> symthaea_music_theory::GrammarRealization {
    let spec = style.spec();
    compose_with_grammar_plan(style.grammar_profile(), intent, &spec)
}

fn axis_intent(axis: &str, level: bool, seed: u64) -> MusicalIntent {
    let mut value = intent(0.0, 0.5, 0.5, 16, seed);
    match axis {
        "valence" => value.valence = if level { 0.8 } else { -0.8 },
        "arousal" => value.arousal = if level { 0.9 } else { 0.1 },
        "energy" => value.energy = if level { 0.9 } else { 0.1 },
        "bars" => value.bars = if level { 24 } else { 8 },
        _ => unreachable!("AXES contains only supported intent parameter names"),
    }
    value
}

fn sensitivity_rows(style: Style) -> Vec<Value> {
    let advertised = style.supported_intent_axes();
    AXES
        .iter()
        .map(|axis| {
            let mut observed_seed_count = 0usize;
            let mut max_distance = 0.0_f64;
            let mut mean_distance = 0.0_f64;
            for seed in AXIS_SEEDS {
                let low = realize(style, &axis_intent(axis, false, seed));
                let high = realize(style, &axis_intent(axis, true, seed));
                let low_fp =
                    structural_fingerprint(&low.score, low.form.as_ref());
                let high_fp =
                    structural_fingerprint(&high.score, high.form.as_ref());
                let distance = dist(&low_fp, &high_fp);
                if distance > 1e-9 {
                    observed_seed_count += 1;
                }
                max_distance = max_distance.max(distance);
                mean_distance += distance;
            }
            mean_distance /= AXIS_SEEDS.len() as f64;

            let declared_live = advertised.contains(axis);
            let status = match (declared_live, observed_seed_count > 0) {
                (true, true) => "declared_live_observed_effect",
                (true, false) => "declared_live_no_effect_at_test_points",
                (false, true) => "declared_inert_but_effect_observed",
                (false, false) => "declared_inert_no_effect_at_test_points",
            };
            json!({
                "axis": axis,
                "declared_live": declared_live,
                "observed_seed_count": observed_seed_count,
                "tested_seed_count": AXIS_SEEDS.len(),
                "max_structural_distance": max_distance,
                "mean_structural_distance": mean_distance,
                "status": status,
            })
        })
        .collect()
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn main() {
    let mut counters = Vec::with_capacity(Style::ALL.len());
    let mut samples = Vec::with_capacity(Style::ALL.len() * SEEDS.len() * INTENTS.len());
    let config_count = SEEDS.len() * INTENTS.len();

    for (style_index, style) in Style::ALL.iter().copied().enumerate() {
        let spec = style.spec();
        let mut stats = StyleCounters::new(style);
        for (seed_index, seed) in SEEDS.iter().copied().enumerate() {
            for (intent_index, &(valence, arousal, energy, bars)) in
                INTENTS.iter().enumerate()
            {
                let config_index = seed_index * INTENTS.len() + intent_index;
                let request = intent(valence, arousal, energy, bars, seed);
                let first = realize(style, &request);
                let first_hash = exact_fingerprint(&first.score);
                let first_structural =
                    structural_fingerprint(&first.score, first.form.as_ref());
                let replay = realize(style, &request);
                let replay_hash = exact_fingerprint(&replay.score);

                stats.samples += 1;
                if first.score.notes.is_empty() {
                    stats.empty_scores += 1;
                }
                let total_beats = first.score.total_beats.beats();
                if !total_beats.is_finite()
                    || total_beats <= 0.0
                    || !first.score.tempo_bpm.is_finite()
                {
                    stats.invalid_timeline += 1;
                }
                if first.score.meter != spec.meter {
                    stats.meter_mismatches += 1;
                }
                let (tempo_min, tempo_max) = spec.tempo_range;
                if first.score.tempo_bpm < tempo_min * 0.8
                    || first.score.tempo_bpm > tempo_max * 1.1
                {
                    stats.tempo_out_of_band += 1;
                }
                if first_hash != replay_hash {
                    stats.deterministic_replay_failures += 1;
                }
                if intent_index == 0 {
                    stats.seed_hashes.insert(first_hash);
                }
                samples.push(Sample {
                    style_index,
                    style_name: format!("{style:?}"),
                    config_index,
                    exact_hash: first_hash,
                    structural: first_structural,
                });
            }
        }
        counters.push(stats);
    }

    // Compare only matched intent/seed cells. Raw pair counts across different
    // prompts would confuse prompt sensitivity with style identity.
    let mut exact_pair_rows = Vec::new();
    let mut exact_collision_pairs = 0usize;
    let style_count = Style::ALL.len();
    for left in 0..style_count {
        for right in (left + 1)..style_count {
            let mut equal_cells = 0usize;
            for config_index in 0..config_count {
                let left_sample = &samples[left * config_count + config_index];
                let right_sample = &samples[right * config_count + config_index];
                if left_sample.exact_hash == right_sample.exact_hash {
                    equal_cells += 1;
                }
            }
            if equal_cells > 0 {
                exact_collision_pairs += 1;
                exact_pair_rows.push(json!({
                    "style_a": counters[left].style_name,
                    "style_b": counters[right].style_name,
                    "matching_exact_cells": equal_cells,
                    "compared_cells": config_count,
                    "exact_collision_rate": equal_cells as f64 / config_count as f64,
                }));
            }
        }
    }
    exact_pair_rows.sort_by(|a, b| {
        let ar = a["exact_collision_rate"].as_f64().unwrap_or(0.0);
        let br = b["exact_collision_rate"].as_f64().unwrap_or(0.0);
        br.total_cmp(&ar)
    });

    // Nearest different-style structural neighbor, matched on the same
    // config cell. This is a diagnostic for candidate label bleed, not a
    // listener-validated measure of perceived style.
    let mut nearest_distances = vec![Vec::<f64>::new(); style_count];
    for sample in &samples {
        let nearest = samples
            .iter()
            .filter(|other| {
                other.style_index != sample.style_index
                    && other.config_index == sample.config_index
            })
            .map(|other| dist(&sample.structural, &other.structural))
            .min_by(f64::total_cmp)
            .unwrap_or(0.0);
        nearest_distances[sample.style_index].push(nearest);
    }

    let mut style_rows = Vec::with_capacity(style_count);
    for (index, style) in Style::ALL.iter().copied().enumerate() {
        let stats = &counters[index];
        style_rows.push(json!({
            "style": stats.style_name,
            "grammar_family": format!("{:?}", style.grammar_family()),
            "spec_valid": stats.spec_valid,
            "samples": stats.samples,
            "empty_scores": stats.empty_scores,
            "invalid_timeline": stats.invalid_timeline,
            "meter_mismatches": stats.meter_mismatches,
            "tempo_out_of_band": stats.tempo_out_of_band,
            "deterministic_replay_failures": stats.deterministic_replay_failures,
            "distinct_seed_fingerprints": stats.seed_hashes.len(),
            "seed_diversity_test_seeds": SEEDS.len(),
            "mean_nearest_cross_style_structural_distance":
                mean(&nearest_distances[index]),
            "intent_sensitivity": sensitivity_rows(style),
        }));
    }

    let spec_invalid_count = counters.iter().filter(|row| !row.spec_valid).count();
    let empty_score_count: usize = counters.iter().map(|row| row.empty_scores).sum();
    let invalid_timeline_count: usize =
        counters.iter().map(|row| row.invalid_timeline).sum();
    let meter_mismatch_count: usize =
        counters.iter().map(|row| row.meter_mismatches).sum();
    let tempo_out_of_band_count: usize =
        counters.iter().map(|row| row.tempo_out_of_band).sum();
    let replay_failure_count: usize = counters
        .iter()
        .map(|row| row.deterministic_replay_failures)
        .sum();

    let source_commit = std::env::var("GIT_COMMIT").ok();
    let report = json!({
        "schema_version": 1,
        "engine_version": MUSIC_THEORY_ENGINE_VERSION,
        "provenance": {
            "git_commit": source_commit,
            "git_commit_source": "GIT_COMMIT environment variable; null means the caller did not provide it",
            "corpus": "none; deterministic synthetic intents only",
        },
        "registry": {
            "style_count": style_count,
            "styles": Style::ALL.iter().map(|style| format!("{style:?}")).collect::<Vec<_>>(),
        },
        "sampling": {
            "seeds": SEEDS,
            "intent_cases": INTENTS.iter().map(|(valence, arousal, energy, bars)| {
                json!({
                    "valence": valence,
                    "arousal": arousal,
                    "energy": energy,
                    "bars": bars,
                    "tonic": "C",
                })
            }).collect::<Vec<_>>(),
            "samples_per_style": config_count,
            "total_samples": samples.len(),
            "route": "compose_with_grammar_plan(style.grammar_profile(), intent, style.spec())",
        },
        "mechanical_audit": {
            "spec_invalid_styles": spec_invalid_count,
            "empty_scores": empty_score_count,
            "invalid_timeline_scores": invalid_timeline_count,
            "meter_mismatches": meter_mismatch_count,
            "tempo_out_of_band": tempo_out_of_band_count,
            "deterministic_replay_failures": replay_failure_count,
            "exact_cross_style_collision_pairs": exact_collision_pairs,
            "all_mechanical_checks_passed": spec_invalid_count == 0
                && empty_score_count == 0
                && invalid_timeline_count == 0
                && meter_mismatch_count == 0
                && tempo_out_of_band_count == 0
                && replay_failure_count == 0,
        },
        "per_style": style_rows,
        "exact_cross_style_collisions": exact_pair_rows,
        "human_evaluation": {
            "status": "not_run",
            "quality_qualified": false,
            "style_recognition": null,
            "pairwise_preference": null,
            "expert_cultural_review": null,
        },
        "interpretation": {
            "scope": "symbolic composition and grammar-dispatch audit",
            "not_claimed": [
                "perceptual style authenticity",
                "human preference",
                "artistic superiority",
                "cultural authenticity",
                "audio/rendering quality",
            ],
            "warning": "Structural distance and exact fingerprint collisions are diagnostics, not aesthetic ground truth.",
        },
    });

    let rendered = serde_json::to_string_pretty(&report)
        .expect("the style benchmark report should serialize as JSON");
    if let Some(path) = std::env::args().nth(1) {
        std::fs::write(&path, rendered)
            .unwrap_or_else(|error| panic!("write {path}: {error}"));
        eprintln!("Wrote style contract benchmark to {path}");
    } else {
        println!("{rendered}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_covers_all_unique_canonical_styles() {
        assert!(!Style::ALL.is_empty());
        assert_eq!(SEEDS.len() * INTENTS.len(), 16);
        let unique: HashSet<String> =
            Style::ALL.iter().map(|style| format!("{style:?}")).collect();
        assert_eq!(unique.len(), Style::ALL.len());
    }

    #[test]
    fn sensitivity_probe_axes_match_public_intent_axis_names() {
        for style in Style::ALL {
            let advertised = style.supported_intent_axes();
            let unique: HashSet<&str> = advertised.iter().copied().collect();
            assert_eq!(unique.len(), advertised.len(), "{style:?} repeats an axis");
            assert!(
                advertised.iter().all(|axis| AXES.contains(axis)),
                "{style:?} advertises an axis the benchmark does not probe"
            );
        }
    }
}
