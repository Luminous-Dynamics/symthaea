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

use std::collections::{HashMap, HashSet};

use serde_json::{Value, json};
use symthaea_music_theory::fingerprint::{
    LAYERS, STRUCT_DIMS, dist, exact_fingerprint, layer_dists, structural_fingerprint,
};
use symthaea_music_theory::rhythm::Duration;
use symthaea_music_theory::pitch::Pitch;
use symthaea_music_theory::score::{Score, VoiceRole};
use symthaea_music_theory::{
    GrammarPlanEvidence, MUSIC_THEORY_ENGINE_VERSION, MusicalIntent, PitchClass, Style,
    compose_with_grammar_plan,
    score_validation::{
        ScoreValidationConfig, ScoreValidationRule, ValidationSeverity,
        is_universal_invariant, validate_score,
    },
};

// Eight fixed seeds give each matched between-style pair 32 intent/config cells,
// and 28 within-style seed pairs per intent. Keep this list frozen for comparable reports.
const SEEDS: [u64; 8] = [0, 1, 7, 13, 29, 42, 73, 101];
const INTENTS: [(f32, f32, f32, usize); 4] = [
    (0.0, 0.5, 0.5, 16),
    (0.8, 0.7, 0.7, 16),
    (-0.7, 0.3, 0.4, 16),
    (0.2, 0.5, 0.5, 24),
];
const AXIS_SEEDS: [u64; 3] = [13, 29, 73];
const AXES: [&str; 4] = ["valence", "arousal", "energy", "bars"];
const REPORT_SCHEMA_VERSION: u32 = 6;

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
    deterministic_replay_structural_failures: usize,
    universal_structural_defects: usize,
    style_dependent_fatal_findings: usize,
    validation_warnings: usize,
    form_available_samples: usize,
    plan_kinds: HashSet<String>,
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
            deterministic_replay_structural_failures: 0,
            universal_structural_defects: 0,
            style_dependent_fatal_findings: 0,
            validation_warnings: 0,
            form_available_samples: 0,
            plan_kinds: HashSet::new(),
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


/// Event-level contract for the Minuet-and-Trio's central contrast:
/// A and ReturnA keep the quarter-note Minuet gait; B must actually contain
/// offbeat harmony attacks from the Trio's eighth-note arpeggiation.
fn minuet_trio_texture_audit() -> Value {
    let spec = Style::MinuetAndTrio.spec();
    let request = intent(0.2, 0.45, 0.4, 16, 42);
    let generated = compose_with_grammar_plan(
        Style::MinuetAndTrio.grammar_profile(),
        &request,
        &spec,
    );
    let Some(form) = generated.form.as_ref() else {
        return json!({
            "passed": false,
            "failure": "missing_form",
            "sections": [],
            "is_qualification_gate": true
        });
    };

    let score = &generated.score;
    let meter = f64::from(score.meter);
    let intro_beats = spec.texture.intro_bars as f64 * meter;
    let mut start_bar = 0_i64;
    let mut metrics = Vec::<(String, usize, usize)>::new();
    for section in &form.sections {
        let section_bars = (section.period.antecedent.progression.len()
            + section.period.consequent.progression.len()) as i64;
        let start = intro_beats + start_bar as f64 * meter;
        let end = intro_beats + (start_bar + section_bars) as f64 * meter;
        let harmony_onsets: Vec<f64> = score
            .notes
            .iter()
            .filter(|note| note.role == VoiceRole::Harmony)
            .filter(|note| {
                note.onset.beats() >= start - 1e-9
                    && note.onset.beats() < end - 1e-9
            })
            .map(|note| note.onset.beats())
            .collect();
        let offbeat_count = harmony_onsets
            .iter()
            .filter(|onset| (**onset - onset.round()).abs() > 1e-6)
            .count();
        metrics.push((format!("{:?}", section.role), harmony_onsets.len(), offbeat_count));
        start_bar += section_bars;
    }

    let expected = ["A", "B", "ReturnA"];
    let roles_match = metrics.len() == expected.len()
        && metrics.iter().zip(expected).all(|(row, name)| row.0 == name);
    let passed = roles_match
        && metrics[0].1 > 0
        && metrics[0].2 == 0
        && metrics[1].2 > 0
        && metrics[2].1 > 0
        && metrics[2].2 == 0;
    json!({
        "passed": passed,
        "contract": "A and ReturnA use the Minuet's on-beat quarter-note harmony; B has actual offbeat eighth-note arpeggio attacks",
        "is_qualification_gate": true,
        "sections": metrics.iter().map(|(role, total, offbeat)| {
            json!({
                "role": role,
                "harmony_event_count": total,
                "offbeat_harmony_event_count": offbeat,
                "offbeat_harmony_event_rate": if *total == 0 {
                    Value::Null
                } else {
                    json!(*offbeat as f64 / *total as f64)
                }
            })
        }).collect::<Vec<_>>()
    })
}


fn plan_name(plan: &GrammarPlanEvidence) -> String {
    match plan {
        GrammarPlanEvidence::PeriodSentence(_) => "period_sentence".into(),
        GrammarPlanEvidence::Contrapuntal(_) => "contrapuntal".into(),
        GrammarPlanEvidence::GrooveCycle(_) => "groove_cycle".into(),
        GrammarPlanEvidence::AdditiveProcess(_) => "additive_process".into(),
        GrammarPlanEvidence::ModalArc(_) => "modal_arc".into(),
        GrammarPlanEvidence::CallResponse(_) => "call_response".into(),
        GrammarPlanEvidence::JazzChorus(_) => "jazz_chorus".into(),
        GrammarPlanEvidence::Compatibility { family, .. } => {
            format!("compatibility:{family:?}")
        }
    }
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
            let mut exact_effect_seed_count = 0usize;
            let mut structural_effect_seed_count = 0usize;
            let mut max_distance = 0.0_f64;
            let mut mean_distance = 0.0_f64;
            for seed in AXIS_SEEDS {
                let low = realize(style, &axis_intent(axis, false, seed));
                let high = realize(style, &axis_intent(axis, true, seed));
                let low_hash = exact_fingerprint(&low.score);
                let high_hash = exact_fingerprint(&high.score);
                let low_fp = structural_fingerprint(&low.score, low.form.as_ref());
                let high_fp = structural_fingerprint(&high.score, high.form.as_ref());
                let distance = dist(&low_fp, &high_fp);
                let exact_effect = low_hash != high_hash;
                let structural_effect = distance > 1e-9;
                if exact_effect {
                    exact_effect_seed_count += 1;
                }
                if structural_effect {
                    structural_effect_seed_count += 1;
                }
                if exact_effect || structural_effect {
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
                "observed_exact_effect_seed_count": exact_effect_seed_count,
                "observed_structural_effect_seed_count": structural_effect_seed_count,
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

/// Linearly interpolated quantile over an already sorted finite sample.
fn quantile_sorted(values: &[f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let position = quantile.clamp(0.0, 1.0) * (values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    Some(values[lower] + (values[upper] - values[lower]) * fraction)
}

/// Calculate a quantile for an arbitrary sample while excluding, but not
/// silently losing visibility of, non-finite values in the emitted summary.
fn sample_quantile(values: &[f64], quantile: f64) -> Option<f64> {
    let mut finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    finite.sort_by(f64::total_cmp);
    quantile_sorted(&finite, quantile)
}

/// A distribution summary keeps central tendency and tails visible together.
/// Non-finite values are excluded from statistics but explicitly counted.
fn distribution_summary(values: &[f64]) -> Value {
    let mut finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let non_finite_count = values.len() - finite.len();
    finite.sort_by(f64::total_cmp);
    if finite.is_empty() {
        return json!({
            "sample_count": values.len(),
            "finite_count": 0,
            "non_finite_count": non_finite_count,
            "mean": Value::Null,
            "min": Value::Null,
            "p10": Value::Null,
            "p25": Value::Null,
            "median": Value::Null,
            "p75": Value::Null,
            "p90": Value::Null,
            "max": Value::Null
        });
    }

    json!({
        "sample_count": values.len(),
        "finite_count": finite.len(),
        "non_finite_count": non_finite_count,
        "mean": mean(&finite),
        "min": finite.first().copied(),
        "p10": quantile_sorted(&finite, 0.10),
        "p25": quantile_sorted(&finite, 0.25),
        "median": quantile_sorted(&finite, 0.50),
        "p75": quantile_sorted(&finite, 0.75),
        "p90": quantile_sorted(&finite, 0.90),
        "max": finite.last().copied()
    })
}

/// Equalizes the contribution of the 11 named fingerprint layers by comparing
/// each layer's RMS feature difference, rather than letting long layers
/// dominate simply because they contain more dimensions. This is still a
/// structural diagnostic, not a learned or perceptually calibrated metric.
fn layer_balanced_distance(
    left: &[f64; STRUCT_DIMS],
    right: &[f64; STRUCT_DIMS],
) -> f64 {
    let layer_distances = layer_dists(left, right);
    let mean_squared_layer_rms = layer_distances
        .iter()
        .zip(LAYERS.iter())
        .map(|(distance, (_, _, length))| {
            if *length == 0 {
                0.0
            } else {
                (distance / (*length as f64).sqrt()).powi(2)
            }
        })
        .sum::<f64>()
        / LAYERS.len() as f64;
    mean_squared_layer_rms.sqrt()
}

/// A deliberately transposition-invariant representation: for each role, keep
/// exact onset/duration and the interval from its previous note, not absolute
/// pitch. It checks whether changing tonic preserves the score's rhythmic and
/// interval skeleton without pretending that absolute register is irrelevant.
fn fatal_rule_detected(score: &Score, expected: ScoreValidationRule) -> bool {
    validate_score(score, &ScoreValidationConfig::default())
        .issues
        .iter()
        .any(|issue| {
            issue.rule == expected && issue.severity == ValidationSeverity::Fatal
        })
}

/// Seed known-invalid mutations to check the benchmark's validator oracle.
/// These are synthetic tests of rule sensitivity, not music-quality tests.
fn validator_mutation_audit() -> Value {
    let baseline = realize(Style::Classical, &intent(0.0, 0.5, 0.5, 16, 42)).score;

    let mut nonfinite_tempo = baseline.clone();
    nonfinite_tempo.tempo_bpm = f32::NAN;
    let metadata_detected =
        fatal_rule_detected(&nonfinite_tempo, ScoreValidationRule::ScoreMetadata);

    let mut zero_duration = baseline.clone();
    zero_duration.notes[0].duration = Duration::zero();
    let zero_duration_detected =
        fatal_rule_detected(&zero_duration, ScoreValidationRule::NoteBounds);

    let mut out_of_range_pitch = baseline.clone();
    out_of_range_pitch.notes[0].pitch = Pitch::from_midi(0);
    let pitch_bound_detected =
        fatal_rule_detected(&out_of_range_pitch, ScoreValidationRule::NoteBounds);

    let mut overlapping_melody = baseline.clone();
    let mut melody_indices: Vec<usize> = overlapping_melody
        .notes
        .iter()
        .enumerate()
        .filter(|(_, note)| note.role == VoiceRole::Melody)
        .map(|(index, _)| index)
        .collect();
    melody_indices.sort_by(|left, right| {
        overlapping_melody.notes[*left]
            .onset
            .beats()
            .total_cmp(&overlapping_melody.notes[*right].onset.beats())
    });
    let monophony_detected = if melody_indices.len() >= 2 {
        let first_onset = overlapping_melody.notes[melody_indices[0]].onset;
        overlapping_melody.notes[melody_indices[1]].onset = first_onset;
        fatal_rule_detected(&overlapping_melody, ScoreValidationRule::VoiceMonophony)
    } else {
        false
    };

    let passed = metadata_detected
        && zero_duration_detected
        && pitch_bound_detected
        && monophony_detected;
    json!({
        "passed": passed,
        "scope": "validator self-test using synthetic fault injection; not a composition-quality test",
        "cases": [
            {
                "mutation": "nonfinite_tempo",
                "expected_fatal_rule": "ScoreMetadata",
                "detected": metadata_detected
            },
            {
                "mutation": "zero_note_duration",
                "expected_fatal_rule": "NoteBounds",
                "detected": zero_duration_detected
            },
            {
                "mutation": "pitch_below_default_midi_floor",
                "expected_fatal_rule": "NoteBounds",
                "detected": pitch_bound_detected
            },
            {
                "mutation": "overlapping_melody_events",
                "expected_fatal_rule": "VoiceMonophony",
                "detected": monophony_detected,
                "eligible_melody_notes": melody_indices.len()
            }
        ]
    })
}

fn transposition_shape_signature(score: &Score) -> Vec<(String, Duration, Duration, i32)> {
    let mut previous_pitch_by_role = HashMap::<String, i32>::new();
    score
        .events()
        .into_iter()
        .map(|note| {
            let role = format!("{:?}", note.role);
            let pitch = i32::from(note.pitch.midi());
            let interval_from_previous = previous_pitch_by_role
                .get(&role)
                .map_or(0, |previous| pitch - previous);
            previous_pitch_by_role.insert(role.clone(), pitch);
            (role, note.onset, note.duration, interval_from_previous)
        })
        .collect()
}

/// Metamorphic probe: ask for the same seeded intent in C and G, then measure
/// whether the rhythm/interval skeleton survives transposition. This is
/// diagnostic rather than a universal pass/fail rule: a style may make
/// register-sensitive orchestration or range choices, but the deviation must
/// be visible rather than silently conflated with ordinary diversity.
fn transposition_audit(style: Style) -> Value {
    let mut invariant_pairs = 0usize;
    let mut exact_event_stream_changed_pairs = 0usize;
    let mut structural_distances = Vec::new();
    let mut total_pairs = 0usize;

    for seed in SEEDS {
        for &(valence, arousal, energy, bars) in &INTENTS {
            let c_intent = intent(valence, arousal, energy, bars, seed);
            let mut g_intent = intent(valence, arousal, energy, bars, seed);
            g_intent.tonic = PitchClass::new(7);

            let c_output = realize(style, &c_intent);
            let g_output = realize(style, &g_intent);
            total_pairs += 1;

            if transposition_shape_signature(&c_output.score)
                == transposition_shape_signature(&g_output.score)
            {
                invariant_pairs += 1;
            }
            if exact_fingerprint(&c_output.score) != exact_fingerprint(&g_output.score) {
                exact_event_stream_changed_pairs += 1;
            }
            let c_structure =
                structural_fingerprint(&c_output.score, c_output.form.as_ref());
            let g_structure =
                structural_fingerprint(&g_output.score, g_output.form.as_ref());
            structural_distances.push(dist(&c_structure, &g_structure));
        }
    }

    json!({
        "source_tonic": "C",
        "target_tonic": "G",
        "paired_samples": total_pairs,
        "rhythm_and_interval_signature_preserved": invariant_pairs,
        "signature_invariance_rate": if total_pairs == 0 {
            0.0
        } else {
            invariant_pairs as f64 / total_pairs as f64
        },
        "exact_event_stream_changed": exact_event_stream_changed_pairs,
        "mean_structural_distance": mean(&structural_distances),
        "interpretation": "diagnostic only; deviations can reveal tonic sensitivity, register effects, or grammar branches and must be reviewed before setting a universal gate"
    })
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
                let replay_structural =
                    structural_fingerprint(&replay.score, replay.form.as_ref());

                stats.samples += 1;
                stats.plan_kinds.insert(plan_name(&first.plan));
                if first.form.is_some() {
                    stats.form_available_samples += 1;
                }
                if first.score.notes.is_empty() {
                    stats.empty_scores += 1;
                }
                let validation =
                    validate_score(&first.score, &ScoreValidationConfig::default());
                for issue in &validation.issues {
                    match issue.severity {
                        ValidationSeverity::Fatal if is_universal_invariant(issue.rule) => {
                            stats.universal_structural_defects += 1;
                        }
                        ValidationSeverity::Fatal => {
                            // Keep style-dependent tonal/voice-leading findings visible
                            // without treating one idiom's rules as universal law.
                            stats.style_dependent_fatal_findings += 1;
                        }
                        ValidationSeverity::Warning => {
                            stats.validation_warnings += 1;
                        }
                    }
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
                if first_structural != replay_structural {
                    stats.deterministic_replay_structural_failures += 1;
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

    // Estimate whether a style's signature survives random-seed variation.
    // Compare different seeds for the SAME style and intent against different
    // styles under the SAME seed and intent. This is a structural diagnostic,
    // not a perceptual classifier or a quality score.
    let mut within_seed_distances = vec![Vec::<f64>::new(); style_count];
    let mut within_layer_distances = vec![Vec::<f64>::new(); LAYERS.len()];
    let mut within_layer_distances_by_style =
        vec![vec![Vec::<f64>::new(); LAYERS.len()]; style_count];
    for style_index in 0..style_count {
        for intent_index in 0..INTENTS.len() {
            for seed_left in 0..SEEDS.len() {
                for seed_right in (seed_left + 1)..SEEDS.len() {
                    let left_config = seed_left * INTENTS.len() + intent_index;
                    let right_config = seed_right * INTENTS.len() + intent_index;
                    let left_sample = &samples[style_index * config_count + left_config];
                    let right_sample = &samples[style_index * config_count + right_config];
                    let layer_values =
                        layer_dists(&left_sample.structural, &right_sample.structural);
                    for (layer_index, (distance, (_, _, length))) in
                        layer_values.iter().zip(LAYERS.iter()).enumerate()
                    {
                        if *length > 0 {
                            let normalized = distance / (*length as f64).sqrt();
                            within_layer_distances[layer_index].push(normalized);
                            within_layer_distances_by_style[style_index][layer_index]
                                .push(normalized);
                        }
                    }
                    within_seed_distances[style_index].push(layer_balanced_distance(
                        &left_sample.structural,
                        &right_sample.structural,
                    ));
                }
            }
        }
    }

    let mut between_style_distances = Vec::<f64>::new();
    let mut between_layer_distances = vec![Vec::<f64>::new(); LAYERS.len()];
    let mut between_style_by_style = vec![Vec::<f64>::new(); style_count];
    let mut style_pair_rows = Vec::new();
    let mut style_pairs_above_seed_variation = 0usize;
    let mut style_pairs_with_positive_tail_margin = 0usize;
    for left in 0..style_count {
        for right in (left + 1)..style_count {
            let mut matched_distances = Vec::with_capacity(config_count);
            let mut matched_layer_distances = vec![Vec::<f64>::new(); LAYERS.len()];
            for config_index in 0..config_count {
                let left_sample = &samples[left * config_count + config_index];
                let right_sample = &samples[right * config_count + config_index];
                let layer_values =
                    layer_dists(&left_sample.structural, &right_sample.structural);
                for (layer_index, (layer_distance, (_, _, length))) in
                    layer_values.iter().zip(LAYERS.iter()).enumerate()
                {
                    if *length > 0 {
                        let normalized = layer_distance / (*length as f64).sqrt();
                        matched_layer_distances[layer_index].push(normalized);
                        between_layer_distances[layer_index].push(normalized);
                    }
                }
                let distance = layer_balanced_distance(
                    &left_sample.structural,
                    &right_sample.structural,
                );
                matched_distances.push(distance);
                between_style_distances.push(distance);
            }
            let mean_between = mean(&matched_distances);
            let left_within = mean(&within_seed_distances[left]);
            let right_within = mean(&within_seed_distances[right]);
            let mean_within_pair = (left_within + right_within) / 2.0;
            let between_p10 = sample_quantile(&matched_distances, 0.10);
            let left_within_p90 = sample_quantile(&within_seed_distances[left], 0.90);
            let right_within_p90 = sample_quantile(&within_seed_distances[right], 0.90);
            let robust_tail_margin = match (between_p10, left_within_p90, right_within_p90) {
                (Some(between), Some(within_a), Some(within_b)) => {
                    Some(between - within_a.max(within_b))
                }
                _ => None,
            };
            let positive_tail_margin = robust_tail_margin.is_some_and(|margin| margin > 0.0);
            let margin = mean_between - mean_within_pair;
            let ratio = if mean_within_pair > 1e-9 {
                json!(mean_between / mean_within_pair)
            } else {
                Value::Null
            };
            let above_seed_variation = mean_between > mean_within_pair;
            if above_seed_variation {
                style_pairs_above_seed_variation += 1;
            }
            if positive_tail_margin {
                style_pairs_with_positive_tail_margin += 1;
            }
            let per_layer_diagnostics = LAYERS
                .iter()
                .enumerate()
                .map(|(layer_index, (layer_name, _, layer_length))| {
                    let between = mean(&matched_layer_distances[layer_index]);
                    let within_a =
                        mean(&within_layer_distances_by_style[left][layer_index]);
                    let within_b =
                        mean(&within_layer_distances_by_style[right][layer_index]);
                    let within_pair_mean = (within_a + within_b) / 2.0;
                    let between_p10 =
                        sample_quantile(&matched_layer_distances[layer_index], 0.10);
                    let within_a_p90 =
                        sample_quantile(&within_layer_distances_by_style[left][layer_index], 0.90);
                    let within_b_p90 =
                        sample_quantile(&within_layer_distances_by_style[right][layer_index], 0.90);
                    let robust_tail_margin = match (between_p10, within_a_p90, within_b_p90) {
                        (Some(between), Some(within_a), Some(within_b)) => {
                            Some(between - within_a.max(within_b))
                        }
                        _ => None,
                    };
                    let ratio_a = if within_a > 1e-9 {
                        json!(between / within_a)
                    } else {
                        Value::Null
                    };
                    let ratio_b = if within_b > 1e-9 {
                        json!(between / within_b)
                    } else {
                        Value::Null
                    };
                    let pair_ratio = if within_pair_mean > 1e-9 {
                        json!(between / within_pair_mean)
                    } else {
                        Value::Null
                    };
                    json!({
                        "layer": layer_name,
                        "dimensions": layer_length,
                        "mean_matched_between_style_rms": between,
                        "matched_between_style_rms_distribution":
                            distribution_summary(&matched_layer_distances[layer_index]),
                        "mean_within_style_seed_rms_a": within_a,
                        "within_style_seed_rms_distribution_a":
                            distribution_summary(&within_layer_distances_by_style[left][layer_index]),
                        "mean_within_style_seed_rms_b": within_b,
                        "within_style_seed_rms_distribution_b":
                            distribution_summary(&within_layer_distances_by_style[right][layer_index]),
                        "between_minus_within_margin_a": between - within_a,
                        "between_minus_within_margin_b": between - within_b,
                        "between_minus_mean_within_pair_margin": between - within_pair_mean,
                        "between_to_within_ratio_a": ratio_a,
                        "between_to_within_ratio_b": ratio_b,
                        "between_to_mean_within_pair_ratio": pair_ratio,
                        "observed_tail_separation": {
                            "between_style_p10": between_p10,
                            "within_style_seed_p90_a": within_a_p90,
                            "within_style_seed_p90_b": within_b_p90,
                            "margin": robust_tail_margin,
                            "positive_margin": robust_tail_margin.is_some_and(|margin| margin > 0.0),
                            "is_qualification_gate": false
                        }
                    })
                })
                .collect::<Vec<_>>();
            between_style_by_style[left].push(mean_between);
            between_style_by_style[right].push(mean_between);
            style_pair_rows.push(json!({
                "style_a": counters[left].style_name,
                "style_b": counters[right].style_name,
                "mean_matched_between_style_layer_balanced_distance": mean_between,
                "matched_between_style_distance_distribution": distribution_summary(&matched_distances),
                "mean_seed_conditioned_within_style_layer_balanced_distance_a": left_within,
                "within_style_seed_distance_distribution_a": distribution_summary(&within_seed_distances[left]),
                "mean_seed_conditioned_within_style_layer_balanced_distance_b": right_within,
                "within_style_seed_distance_distribution_b": distribution_summary(&within_seed_distances[right]),
                "between_minus_within_margin": margin,
                "between_to_within_ratio": ratio,
                "between_exceeds_mean_seed_variation": above_seed_variation,
                "observed_tail_separation": {
                    "between_style_p10": between_p10,
                    "within_style_seed_p90_a": left_within_p90,
                    "within_style_seed_p90_b": right_within_p90,
                    "margin": robust_tail_margin,
                    "positive_margin": positive_tail_margin,
                    "is_qualification_gate": false,
                    "interpretation": "positive margin means the observed lower tail of between-style distance exceeds both observed within-style upper tails; this small fixed sample is not a confidence interval"
                },
                "per_layer_diagnostics": per_layer_diagnostics,
                "matched_config_count": config_count
            }));
        }
    }
    let all_within_seed_distances: Vec<f64> =
        within_seed_distances.iter().flatten().copied().collect();
    let mean_within_seed_distance = mean(&all_within_seed_distances);
    let mean_between_style_distance = mean(&between_style_distances);
    let global_style_separation_ratio = if mean_within_seed_distance > 1e-9 {
        json!(mean_between_style_distance / mean_within_seed_distance)
    } else {
        Value::Null
    };
    let per_layer_separation = LAYERS
        .iter()
        .enumerate()
        .map(|(layer_index, (layer_name, _, layer_length))| {
            let within = mean(&within_layer_distances[layer_index]);
            let between = mean(&between_layer_distances[layer_index]);
            json!({
                "layer": layer_name,
                "dimensions": layer_length,
                "mean_within_style_seed_rms": within,
                "within_style_seed_rms_distribution":
                    distribution_summary(&within_layer_distances[layer_index]),
                "mean_matched_between_style_rms": between,
                "matched_between_style_rms_distribution":
                    distribution_summary(&between_layer_distances[layer_index]),
                "between_minus_within_margin": between - within,
                "between_to_within_ratio": if within > 1e-9 {
                    json!(between / within)
                } else {
                    Value::Null
                }
            })
        })
        .collect::<Vec<_>>();
    style_pair_rows.sort_by(|a, b| {
        let ar = a["between_minus_within_margin"].as_f64().unwrap_or(0.0);
        let br = b["between_minus_within_margin"].as_f64().unwrap_or(0.0);
        ar.total_cmp(&br)
    });

    let mut style_rows = Vec::with_capacity(style_count);
    let mut declared_live_no_effect_count = 0usize;
    let mut declared_inert_with_effect_count = 0usize;
    for (index, style) in Style::ALL.iter().copied().enumerate() {
        let stats = &counters[index];
        let mut plan_kinds = stats.plan_kinds.iter().cloned().collect::<Vec<_>>();
        plan_kinds.sort();
        let spec = style.spec();
        let form_pool = spec.form_pool.iter().map(|form| format!("{form:?}")).collect::<Vec<_>>();
        let accompaniment_pool = spec
            .accompaniment_pool
            .iter()
            .map(|pattern| format!("{pattern:?}"))
            .collect::<Vec<_>>();
        let sensitivities = sensitivity_rows(style);
        for row in &sensitivities {
            match row["status"].as_str() {
                Some("declared_live_no_effect_at_test_points") => {
                    declared_live_no_effect_count += 1;
                }
                Some("declared_inert_but_effect_observed") => {
                    declared_inert_with_effect_count += 1;
                }
                _ => {}
            }
        }
        let transposition = transposition_audit(style);
        style_rows.push(json!({
            "style": stats.style_name,
            "grammar_family": format!("{:?}", style.grammar_family()),
            "selected_plan_kinds": plan_kinds,
            "form_available_samples": stats.form_available_samples,
            "declared_form_pool": form_pool,
            "declared_accompaniment_pool": accompaniment_pool,
            "spec_valid": stats.spec_valid,
            "samples": stats.samples,
            "empty_scores": stats.empty_scores,
            "invalid_timeline": stats.invalid_timeline,
            "meter_mismatches": stats.meter_mismatches,
            "tempo_out_of_band": stats.tempo_out_of_band,
            "deterministic_replay_failures": stats.deterministic_replay_failures,
            "deterministic_replay_structural_failures":
                stats.deterministic_replay_structural_failures,
            "universal_structural_defects": stats.universal_structural_defects,
            "style_dependent_fatal_findings": stats.style_dependent_fatal_findings,
            "validation_warnings": stats.validation_warnings,
            "distinct_seed_fingerprints": stats.seed_hashes.len(),
            "seed_diversity_test_seeds": SEEDS.len(),
            "mean_nearest_cross_style_structural_distance":
                mean(&nearest_distances[index]),
            "nearest_cross_style_structural_distance_distribution":
                distribution_summary(&nearest_distances[index]),
            "mean_seed_conditioned_within_style_layer_balanced_distance":
                mean(&within_seed_distances[index]),
            "within_style_seed_distance_distribution":
                distribution_summary(&within_seed_distances[index]),
            "mean_matched_between_style_layer_balanced_distance":
                mean(&between_style_by_style[index]),
            "between_to_within_seed_variation_ratio": {
                let within = mean(&within_seed_distances[index]);
                if within > 1e-9 {
                    json!(mean(&between_style_by_style[index]) / within)
                } else {
                    Value::Null
                }
            },
            "metamorphic_audits": {
                "tonic_transposition": transposition,
            },
            "intent_sensitivity": sensitivities,
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
    let transposition_audits: Vec<&Value> = style_rows
        .iter()
        .map(|row| &row["metamorphic_audits"]["tonic_transposition"])
        .collect();
    let transposition_pair_count: u64 = transposition_audits
        .iter()
        .map(|audit| audit["paired_samples"].as_u64().unwrap_or(0))
        .sum();
    let transposition_invariant_pairs: u64 = transposition_audits
        .iter()
        .map(|audit| {
            audit["rhythm_and_interval_signature_preserved"]
                .as_u64()
                .unwrap_or(0)
        })
        .sum();
    let transposition_exact_changed_pairs: u64 = transposition_audits
        .iter()
        .map(|audit| audit["exact_event_stream_changed"].as_u64().unwrap_or(0))
        .sum();
    let transposition_mean_distances: Vec<f64> = transposition_audits
        .iter()
        .filter_map(|audit| audit["mean_structural_distance"].as_f64())
        .collect();

    let replay_failure_count: usize = counters
        .iter()
        .map(|row| row.deterministic_replay_failures)
        .sum();
    let replay_structural_failure_count: usize = counters
        .iter()
        .map(|row| row.deterministic_replay_structural_failures)
        .sum();
    let universal_structural_defect_count: usize = counters
        .iter()
        .map(|row| row.universal_structural_defects)
        .sum();
    let style_dependent_fatal_finding_count: usize = counters
        .iter()
        .map(|row| row.style_dependent_fatal_findings)
        .sum();
    let validation_warning_count: usize = counters
        .iter()
        .map(|row| row.validation_warnings)
        .sum();
    let control_contract_violation_count =
        declared_live_no_effect_count + declared_inert_with_effect_count;

    let source_commit = std::env::var("GIT_COMMIT").ok();
    let validator_self_test = validator_mutation_audit();
    let minuet_trio_contract = minuet_trio_texture_audit();
    let report = json!({
        "schema_version": REPORT_SCHEMA_VERSION,
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
            "score_validation_version": symthaea_music_theory::score_validation::THEORY_VALIDATION_VERSION,
            "universal_structural_defects": universal_structural_defect_count,
            "style_dependent_fatal_findings": style_dependent_fatal_finding_count,
            "validation_warnings": validation_warning_count,
            "spec_invalid_styles": spec_invalid_count,
            "empty_scores": empty_score_count,
            "invalid_timeline_scores": invalid_timeline_count,
            "meter_mismatches": meter_mismatch_count,
            "tempo_out_of_band": tempo_out_of_band_count,
            "deterministic_replay_failures": replay_failure_count,
            "deterministic_replay_structural_failures": replay_structural_failure_count,
            "exact_cross_style_collision_pairs": exact_collision_pairs,
            "named_style_contract_violations": if minuet_trio_contract["passed"].as_bool().unwrap_or(false) { 0 } else { 1 },
            "all_mechanical_checks_passed": minuet_trio_contract["passed"].as_bool().unwrap_or(false)
                && spec_invalid_count == 0
                && empty_score_count == 0
                && invalid_timeline_count == 0
                && meter_mismatch_count == 0
                && tempo_out_of_band_count == 0
                && replay_failure_count == 0
                && replay_structural_failure_count == 0
                && universal_structural_defect_count == 0
                && control_contract_violation_count == 0
                && validator_self_test["passed"].as_bool().unwrap_or(false),
        },
        "control_contract_audit": {
            "declared_live_no_effect_at_test_points": declared_live_no_effect_count,
            "declared_inert_but_effect_observed": declared_inert_with_effect_count,
            "violation_count": control_contract_violation_count,
            "passed_at_test_points": control_contract_violation_count == 0,
            "scope": "paired high/low probes across fixed seeds; no effect at tested points is not proof of global inactivity",
        },
        "validator_mutation_audit": validator_self_test,
        "named_style_contract_audit": {
            "minuet_and_trio_texture": minuet_trio_contract
        },
        "style_identifiability_audit": {
            "method": "compare matched between-style distances against same-style, same-intent distances across different seeds",
            "distance_metric": "layer-balanced RMS over the 11 named structural-fingerprint layers; each layer contributes equally regardless of dimensionality",
            "within_style_seed_comparison_count": all_within_seed_distances.len(),
            "mean_seed_conditioned_within_style_layer_balanced_distance": mean_within_seed_distance,
            "within_style_distance_distribution": distribution_summary(&all_within_seed_distances),
            "matched_between_style_comparison_count": between_style_distances.len(),
            "mean_matched_between_style_layer_balanced_distance": mean_between_style_distance,
            "between_style_distance_distribution": distribution_summary(&between_style_distances),
            "global_between_to_within_distance_ratio": global_style_separation_ratio,
            "style_pairs_above_both_seed_variation_baselines": style_pairs_above_seed_variation,
            "style_pairs_with_positive_observed_tail_margin": style_pairs_with_positive_tail_margin,
            "style_pair_count": style_pair_rows.len(),
            "per_layer_separation": per_layer_separation,
            "per_style_pair": style_pair_rows,
            "is_qualification_gate": false,
            "interpretation": "diagnostic only; structural Euclidean distances are neither human recognition nor cultural authenticity, and aggregate separation may hide individual style confusions"
        },
        "metamorphic_audit_summary": {
            "probe": "same seed and intent in C vs G",
            "style_count": style_count,
            "paired_samples": transposition_pair_count,
            "rhythm_and_interval_signature_preserved": transposition_invariant_pairs,
            "signature_invariance_rate": if transposition_pair_count == 0 {
                0.0
            } else {
                transposition_invariant_pairs as f64 / transposition_pair_count as f64
            },
            "exact_event_stream_changed": transposition_exact_changed_pairs,
            "mean_structural_distance_across_styles":
                mean(&transposition_mean_distances),
            "is_qualification_gate": false,
            "reason": "first establish a measured baseline and review legitimate range-sensitive exceptions before declaring transposition a universal contract",
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
        assert_eq!(SEEDS.len() * INTENTS.len(), 32);
        let unique: HashSet<String> =
            Style::ALL.iter().map(|style| format!("{style:?}")).collect();
        assert_eq!(unique.len(), Style::ALL.len());
    }

    #[test]
    fn quantile_interpolation_is_stable_at_boundaries_and_between_samples() {
        let sorted = [0.0, 10.0, 20.0, 30.0];
        assert_eq!(quantile_sorted(&sorted, 0.0), Some(0.0));
        assert_eq!(quantile_sorted(&sorted, 0.5), Some(15.0));
        assert_eq!(quantile_sorted(&sorted, 0.9), Some(27.0));
        assert_eq!(quantile_sorted(&sorted, 1.0), Some(30.0));
        assert_eq!(quantile_sorted(&[], 0.5), None);
    }

    #[test]
    fn sampled_quantiles_detect_robust_tail_margin_without_claiming_confidence() {
        let between = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let within_a = [0.0, 1.0, 1.0, 2.0];
        let within_b = [0.0, 1.0, 2.0, 3.0];
        let between_p10 = sample_quantile(&between, 0.10).unwrap();
        let within_a_p90 = sample_quantile(&within_a, 0.90).unwrap();
        let within_b_p90 = sample_quantile(&within_b, 0.90).unwrap();
        assert!(between_p10 - within_a_p90.max(within_b_p90) > 0.0);
        assert_eq!(sample_quantile(&[f64::NAN, f64::INFINITY], 0.5), None);
    }

    #[test]
    fn distribution_summary_exposes_tails_and_counts_non_finite_values() {
        let summary = distribution_summary(&[1.0, 1.0, 1.0, 1_000.0, f64::NAN]);
        assert_eq!(summary["sample_count"].as_u64(), Some(5));
        assert_eq!(summary["finite_count"].as_u64(), Some(4));
        assert_eq!(summary["non_finite_count"].as_u64(), Some(1));
        assert_eq!(summary["median"].as_f64(), Some(1.0));
        assert_eq!(summary["max"].as_f64(), Some(1_000.0));
        assert!(summary["p90"].as_f64().unwrap_or(0.0) > 1.0);
        assert!(summary["mean"].as_f64().unwrap_or(0.0) > 100.0);
    }

    #[test]
    fn empty_distribution_reports_null_statistics_instead_of_fabricating_zeroes() {
        let summary = distribution_summary(&[]);
        assert_eq!(summary["sample_count"].as_u64(), Some(0));
        assert_eq!(summary["finite_count"].as_u64(), Some(0));
        assert!(summary["median"].is_null());
        assert!(summary["mean"].is_null());
    }

    #[test]
    fn report_schema_version_tracks_the_named_style_contract_fields() {
        assert_eq!(REPORT_SCHEMA_VERSION, 6);
    }

    #[test]
    fn fixed_seed_sweep_has_unique_seeds_and_32_matched_config_cells() {
        let unique: HashSet<u64> = SEEDS.iter().copied().collect();
        assert_eq!(unique.len(), SEEDS.len());
        assert_eq!(SEEDS.len() * INTENTS.len(), 32);
    }

    #[test]
    fn minuet_trio_texture_audit_detects_event_level_contrast() {
        let report = minuet_trio_texture_audit();
        assert!(
            report["passed"].as_bool().unwrap_or(false),
            "Minuet-and-Trio texture contract failed: {report}"
        );
        assert_eq!(report["sections"][0]["role"], "A");
        assert_eq!(report["sections"][1]["role"], "B");
        assert_eq!(report["sections"][2]["role"], "ReturnA");
        assert_eq!(report["sections"][0]["offbeat_harmony_event_count"], 0);
        assert!(
            report["sections"][1]["offbeat_harmony_event_count"]
                .as_u64().unwrap_or(0) > 0
        );
        assert_eq!(report["sections"][2]["offbeat_harmony_event_count"], 0);
    }

    #[test]
    fn layer_balanced_distance_is_symmetric_and_zero_for_identical_vectors() {
        let baseline = [0.0; STRUCT_DIMS];
        assert_eq!(layer_balanced_distance(&baseline, &baseline), 0.0);

        let mut changed = baseline;
        let (_, start, _) = LAYERS[0];
        changed[start] = 1.0;
        assert!(layer_balanced_distance(&baseline, &changed) > 0.0);
        assert_eq!(
            layer_balanced_distance(&baseline, &changed),
            layer_balanced_distance(&changed, &baseline)
        );
    }

    #[test]
    fn layer_balanced_distance_weights_uniform_layer_shifts_equally() {
        let baseline = [0.0; STRUCT_DIMS];
        let mut form_shift = baseline;
        let mut orchestration_shift = baseline;
        let (_, form_start, form_len) = LAYERS[0];
        let (_, orchestration_start, orchestration_len) = LAYERS[2];

        form_shift[form_start..form_start + form_len].fill(1.0);
        orchestration_shift
            [orchestration_start..orchestration_start + orchestration_len]
            .fill(1.0);

        let form_distance = layer_balanced_distance(&baseline, &form_shift);
        let orchestration_distance =
            layer_balanced_distance(&baseline, &orchestration_shift);
        assert!(
            (form_distance - orchestration_distance).abs() < 1e-12,
            "each uniformly changed fingerprint layer should carry equal aggregate weight"
        );
    }

    #[test]
    fn validator_mutation_audit_detects_seeded_universal_faults() {
        let report = validator_mutation_audit();
        assert!(
            report["passed"].as_bool().unwrap_or(false),
            "validator failed to detect an intentionally seeded universal invariant defect: {report}"
        );
    }

    #[test]
    fn transposition_signature_ignores_uniform_pitch_shift() {
        let generated = realize(Style::Classical, &intent(0.0, 0.5, 0.5, 16, 42));
        assert!(!generated.score.notes.is_empty());
        let original = transposition_shape_signature(&generated.score);

        let mut shifted = generated.score.clone();
        for note in &mut shifted.notes {
            note.pitch = note.pitch.transpose(7);
        }

        assert_eq!(original, transposition_shape_signature(&shifted));
    }

    #[test]
    fn benchmark_gates_only_style_independent_score_invariants() {
        assert!(is_universal_invariant(ScoreValidationRule::ScoreMetadata));
        assert!(is_universal_invariant(ScoreValidationRule::NoteBounds));
        assert!(is_universal_invariant(ScoreValidationRule::VoiceMonophony));
        assert!(!is_universal_invariant(ScoreValidationRule::FinalTonicArrival));
        assert!(!is_universal_invariant(ScoreValidationRule::StrongBeatConsonance));
        assert!(!is_universal_invariant(ScoreValidationRule::MelodicLeap));
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
