// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic response curves for Muse's real public style-composition path.
//!
//! Unlike the composition-only diversity census, this probe calls
//! `compose_with_grammar_plan(style.grammar_profile(), ...)`, the same
//! dispatch path used by `compose_styled`. It changes one intent control at a
//! time, holding the other inputs and the seed fixed, and records every tested
//! point rather than reducing the curve to a single sensitivity label.
//!
//! Run:
//!   cargo run --release -p symthaea-music-theory --example style_control_response_curves
//!
//! Write the JSON artifact:
//!   MUSE_STYLE_RESPONSE_JSON_OUT=/tmp/muse-style-response.json \
//!     cargo run --release -p symthaea-music-theory --example style_control_response_curves
//!
//! Output is symbolic evidence only. It does not verify audio rendering,
//! perceptual distinctness, or cultural authenticity.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use symthaea_music_theory::composer::MusicalIntent;
use symthaea_music_theory::fingerprint::{self, LAYERS, STRUCT_DIMS};
use symthaea_music_theory::{
    MUSIC_THEORY_ENGINE_VERSION, PitchClass, Style, compose_with_grammar_plan,
};

const BASE_SEED: u64 = 90_210;
const BASE_SEED_SET: [u64; 4] = [90_210, 130_363, 222_731, 918_273];
const BASE_TONIC: i32 = 0; // C
const BASE_VALENCE: f32 = 0.0;
const BASE_AROUSAL: f32 = 0.5;
const BASE_ENERGY: f32 = 0.5;
const BASE_BARS: usize = 16;
const LAYER_CHANGE_THRESHOLD: f64 = 0.02;
const MONOTONIC_EPSILON: f64 = 1.0e-6;

#[derive(Clone, Copy, Debug, PartialEq)]
struct InputConfig {
    tonic: i32,
    seed: u64,
    valence: f32,
    arousal: f32,
    energy: f32,
    bars: usize,
}

impl InputConfig {
    const BASE: Self = Self {
        tonic: BASE_TONIC,
        seed: BASE_SEED,
        valence: BASE_VALENCE,
        arousal: BASE_AROUSAL,
        energy: BASE_ENERGY,
        bars: BASE_BARS,
    };

    fn intent(self) -> MusicalIntent {
        MusicalIntent {
            valence: self.valence,
            arousal: self.arousal,
            energy: self.energy,
            bars: self.bars,
            seed: self.seed,
            tonic: PitchClass::new(self.tonic),
        }
    }

    fn json(self) -> serde_json::Value {
        serde_json::json!({
            "tonic_semitones_from_c": self.tonic,
            "seed": self.seed,
            "valence": self.valence,
            "arousal": self.arousal,
            "energy": self.energy,
            "bars": self.bars,
        })
    }
}

#[derive(Clone, Debug)]
struct ControlPoint {
    value_label: String,
    numeric_value: Option<f64>,
    input: InputConfig,
}

#[derive(Debug)]
struct Observation {
    value_label: String,
    numeric_value: Option<f64>,
    input: InputConfig,
    exact_hash: u64,
    exact_hash_changed: bool,
    overall_structural_distance: f64,
    layer_distances: [f64; LAYERS.len()],
    tempo_bpm: f32,
    meter_beats: u8,
    total_duration_beats: f64,
    score_note_count: usize,
    distinct_midi_pitches: usize,
    pitch_span_semitones: i32,
    score_key_tonic_pc: i32,
    score_key_tonality: String,
    phrase_model: Option<String>,
    mean_velocity: f64,
    velocity_standard_deviation: f64,
    mean_note_duration_beats: f64,
    mean_section_intensity: f64,
    section_intensity_standard_deviation: f64,
    emphasis_counts: [usize; 4],
    form_section_count: Option<usize>,
    plan_requested_bars: Option<usize>,
    plan_realized_bars: Option<usize>,
    plan_bars_per_chorus: Option<usize>,
    observed_score_span_bars: f64,
}

fn control_points(control: &str, base_seed: u64) -> Vec<ControlPoint> {
    let baseline = InputConfig {
        seed: base_seed,
        ..InputConfig::BASE
    };
    match control {
        "tonic" => [(0, "C"), (3, "Eb"), (4, "E"), (7, "G"), (10, "Bb")]
            .into_iter()
            .map(|(tonic, label)| ControlPoint {
                value_label: label.into(),
                numeric_value: None,
                input: InputConfig { tonic, ..baseline },
            })
            .collect(),
        "valence" => [-1.0_f32, -0.5, 0.0, 0.5, 1.0]
            .into_iter()
            .map(|value| ControlPoint {
                value_label: format!("{value:.2}"),
                numeric_value: Some(value as f64),
                input: InputConfig { valence: value, ..baseline },
            })
            .collect(),
        "arousal" => [0.0_f32, 0.25, 0.5, 0.75, 1.0]
            .into_iter()
            .map(|value| ControlPoint {
                value_label: format!("{value:.2}"),
                numeric_value: Some(value as f64),
                input: InputConfig { arousal: value, ..baseline },
            })
            .collect(),
        "energy" => [0.0_f32, 0.25, 0.5, 0.75, 1.0]
            .into_iter()
            .map(|value| ControlPoint {
                value_label: format!("{value:.2}"),
                numeric_value: Some(value as f64),
                input: InputConfig { energy: value, ..baseline },
            })
            .collect(),
        "bars" => [8_usize, 12, 16, 20, 24]
            .into_iter()
            .map(|value| ControlPoint {
                value_label: value.to_string(),
                numeric_value: Some(value as f64),
                input: InputConfig { bars: value, ..baseline },
            })
            .collect(),
        "seed" => [base_seed, base_seed + 1, base_seed + 2, base_seed + 3]
            .into_iter()
            .map(|value| ControlPoint {
                value_label: value.to_string(),
                numeric_value: None,
                input: InputConfig { seed: value, ..baseline },
            })
            .collect(),
        other => panic!("unknown control {other}"),
    }
}

fn observe(
    style: Style,
    control_point: &ControlPoint,
    base_hash: u64,
    base_structure: &[f64; STRUCT_DIMS],
) -> Observation {
    let intent = control_point.input.intent();
    let spec = style.spec();
    let realization = compose_with_grammar_plan(style.grammar_profile(), &intent, &spec);
    let phrase_model = match &realization.plan {
        symthaea_music_theory::grammar::GrammarPlanEvidence::PeriodSentence(plan) => {
            Some(plan.phrase_model.clone())
        }
        _ => None,
    };
    // Only these dedicated engines currently publish an explicit request-to-
    // realized-bar quantization contract. Do not infer such a contract from
    // a style name or from section count.
    let (plan_requested_bars, plan_realized_bars, plan_bars_per_chorus) =
        match &realization.plan {
            symthaea_music_theory::grammar::GrammarPlanEvidence::CallResponse(plan) => (
                Some(plan.requested_bars),
                Some(plan.realized_bars),
                Some(plan.bars_per_chorus),
            ),
            symthaea_music_theory::grammar::GrammarPlanEvidence::JazzChorus(plan) => (
                Some(plan.requested_bars),
                Some(plan.realized_bars),
                Some(plan.bars_per_chorus),
            ),
            _ => (None, None, None),
        };
    let score = realization.score;
    let note_span_beats = score
        .notes
        .iter()
        .map(|note| (note.onset + note.duration).beats())
        .fold(0.0_f64, f64::max);
    let observed_score_span_bars = note_span_beats / f64::from(score.meter.max(1));
    let structural = fingerprint::structural_fingerprint(&score, &realization.form);
    let exact_hash = fingerprint::exact_fingerprint(&score);
    let layer_distances = fingerprint::layer_dists(base_structure, &structural);
    let midi_pitches: BTreeSet<i32> = score
        .notes
        .iter()
        .map(|note| i32::from(note.pitch.midi()))
        .collect();
    let pitch_span_semitones = midi_pitches
        .first()
        .zip(midi_pitches.last())
        .map_or(0, |(low, high)| high - low);
    let note_count = score.notes.len();
    let mean_velocity = if note_count == 0 {
        0.0
    } else {
        score.notes.iter().map(|note| note.velocity as f64).sum::<f64>()
            / note_count as f64
    };
    let velocity_standard_deviation = if note_count == 0 {
        0.0
    } else {
        (score
            .notes
            .iter()
            .map(|note| (note.velocity as f64 - mean_velocity).powi(2))
            .sum::<f64>()
            / note_count as f64)
            .sqrt()
    };
    let mean_note_duration_beats = if note_count == 0 {
        0.0
    } else {
        score
            .notes
            .iter()
            .map(|note| note.duration.beats())
            .sum::<f64>()
            / note_count as f64
    };
    let mean_section_intensity = if note_count == 0 {
        0.0
    } else {
        score
            .notes
            .iter()
            .map(|note| note.section_intensity as f64)
            .sum::<f64>()
            / note_count as f64
    };
    let section_intensity_standard_deviation = if note_count == 0 {
        0.0
    } else {
        (score
            .notes
            .iter()
            .map(|note| (note.section_intensity as f64 - mean_section_intensity).powi(2))
            .sum::<f64>()
            / note_count as f64)
            .sqrt()
    };
    let mut emphasis_counts = [0_usize; 4];
    for note in &score.notes {
        let index = match note.emphasis {
            symthaea_music_theory::score::Emphasis::Normal => 0,
            symthaea_music_theory::score::Emphasis::Climax => 1,
            symthaea_music_theory::score::Emphasis::Cadential => 2,
            symthaea_music_theory::score::Emphasis::PhraseStart => 3,
        };
        emphasis_counts[index] += 1;
    }

    Observation {
        value_label: control_point.value_label.clone(),
        numeric_value: control_point.numeric_value,
        input: control_point.input,
        exact_hash,
        exact_hash_changed: exact_hash != base_hash,
        overall_structural_distance: fingerprint::dist(base_structure, &structural),
        layer_distances,
        tempo_bpm: score.tempo_bpm,
        meter_beats: score.meter,
        total_duration_beats: score.total_beats.beats(),
        score_note_count: score.notes.len(),
        distinct_midi_pitches: midi_pitches.len(),
        pitch_span_semitones,
        score_key_tonic_pc: score.key.tonic.value() as i32,
        score_key_tonality: format!("{:?}", score.key.tonality),
        phrase_model,
        mean_velocity,
        velocity_standard_deviation,
        mean_note_duration_beats,
        mean_section_intensity,
        section_intensity_standard_deviation,
        emphasis_counts,
        form_section_count: realization.form.as_ref().map(|form| form.sections.len()),
        plan_requested_bars,
        plan_realized_bars,
        plan_bars_per_chorus,
        observed_score_span_bars,
    }
}

/// Assign average ranks so ties do not receive arbitrary ordering.
fn average_ranks(values: &[f64]) -> Option<Vec<f64>> {
    if values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&left, &right| values[left].total_cmp(&values[right]));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        // Positions are one-based: [start + 1, end].
        let average_rank = (start + 1 + end) as f64 / 2.0;
        for index in start..end {
            ranks[order[index]] = average_rank;
        }
        start = end;
    }
    Some(ranks)
}

/// Rank association is descriptive response evidence, never target fidelity.
/// Constant outputs and non-numeric control axes produce explicit null metrics.
fn spearman_rank_response(points: &[(f64, f64)]) -> serde_json::Value {
    if points.is_empty() {
        return serde_json::json!({
            "status": "not_applicable_non_numeric_control",
            "rho": null,
            "point_count": 0,
        });
    }
    if points.len() < 3 {
        return serde_json::json!({
            "status": "insufficient_points",
            "rho": null,
            "point_count": points.len(),
        });
    }
    let x: Vec<f64> = points.iter().map(|(control, _)| *control).collect();
    let y: Vec<f64> = points.iter().map(|(_, output)| *output).collect();
    let Some(x_ranks) = average_ranks(&x) else {
        return serde_json::json!({
            "status": "invalid_values",
            "rho": null,
            "point_count": points.len(),
        });
    };
    let Some(y_ranks) = average_ranks(&y) else {
        return serde_json::json!({
            "status": "invalid_values",
            "rho": null,
            "point_count": points.len(),
        });
    };
    let x_mean = x_ranks.iter().sum::<f64>() / x_ranks.len() as f64;
    let y_mean = y_ranks.iter().sum::<f64>() / y_ranks.len() as f64;
    let mut covariance = 0.0;
    let mut x_variance = 0.0;
    let mut y_variance = 0.0;
    for (x_rank, y_rank) in x_ranks.iter().zip(y_ranks.iter()) {
        let x_delta = x_rank - x_mean;
        let y_delta = y_rank - y_mean;
        covariance += x_delta * y_delta;
        x_variance += x_delta * x_delta;
        y_variance += y_delta * y_delta;
    }
    if x_variance <= f64::EPSILON {
        return serde_json::json!({
            "status": "constant_control",
            "rho": null,
            "point_count": points.len(),
        });
    }
    if y_variance <= f64::EPSILON {
        return serde_json::json!({
            "status": "constant_output",
            "rho": null,
            "point_count": points.len(),
        });
    }
    let rho = (covariance / (x_variance * y_variance).sqrt()).clamp(-1.0, 1.0);
    serde_json::json!({
        "status": "measured",
        "method": "spearman_rho_average_ranks_for_ties",
        "rho": rho,
        "point_count": points.len(),
        "interpretation": "Rank association describes response ordering, not target correctness, causality beyond this intervention, perceived affect, or quality.",
    })
}

fn monotonicity(values: &[(f64, f64)]) -> &'static str {
    if values.len() < 2 {
        return "insufficient_points";
    }
    let nondecreasing = values.windows(2).all(|pair| pair[1].1 >= pair[0].1 - MONOTONIC_EPSILON);
    let nonincreasing = values.windows(2).all(|pair| pair[1].1 <= pair[0].1 + MONOTONIC_EPSILON);
    if nondecreasing && nonincreasing {
        "flat"
    } else if nondecreasing {
        "nondecreasing"
    } else if nonincreasing {
        "nonincreasing"
    } else {
        "non_monotonic"
    }
}

fn observation_json(observation: &Observation, base_hash: u64) -> serde_json::Value {
    let layer_distances: Vec<serde_json::Value> = LAYERS
        .iter()
        .zip(observation.layer_distances.iter())
        .map(|(&(name, _, _), &distance)| {
            serde_json::json!({
                "layer": name,
                "distance_from_baseline": distance,
                "changed_over_inventory_threshold": distance > LAYER_CHANGE_THRESHOLD,
            })
        })
        .collect();

    serde_json::json!({
        "control_value": observation.value_label,
        "numeric_control_value": observation.numeric_value,
        "input": observation.input.json(),
        "exact_fingerprint_hex": format!("{:016x}", observation.exact_hash),
        "exact_fingerprint_changed": observation.exact_hash_changed,
        "baseline_exact_fingerprint_hex": format!("{:016x}", base_hash),
        "overall_structural_distance_from_baseline": observation.overall_structural_distance,
        "layer_distances": layer_distances,
        "descriptors": {
            "tempo_bpm": observation.tempo_bpm,
            "meter_beats": observation.meter_beats,
            "total_duration_beats": observation.total_duration_beats,
            "score_note_count": observation.score_note_count,
            "distinct_midi_pitches": observation.distinct_midi_pitches,
            "pitch_span_semitones": observation.pitch_span_semitones,
            "score_key_tonic_pc": observation.score_key_tonic_pc,
            "score_key_tonality": observation.score_key_tonality,
            "phrase_model": observation.phrase_model,
            "mean_velocity": observation.mean_velocity,
            "velocity_standard_deviation": observation.velocity_standard_deviation,
            "mean_note_duration_beats": observation.mean_note_duration_beats,
            "mean_section_intensity": observation.mean_section_intensity,
            "section_intensity_standard_deviation": observation
                .section_intensity_standard_deviation,
            "emphasis_counts": {
                "normal": observation.emphasis_counts[0],
                "climax": observation.emphasis_counts[1],
                "cadential": observation.emphasis_counts[2],
                "phrase_start": observation.emphasis_counts[3],
            },
            "form_available": observation.form_section_count.is_some(),
            "form_section_count": observation.form_section_count,
            "plan_requested_bars": observation.plan_requested_bars,
            "plan_realized_bars": observation.plan_realized_bars,
            "plan_bars_per_chorus": observation.plan_bars_per_chorus,
            "observed_score_span_bars": observation.observed_score_span_bars,
        },
    })
}

const TONIC_CANDIDATE_INVARIANT_LAYERS: &[&str] = &[
    "form",
    "rhythm",
    "contour",
    "tempo_meter",
    "pitch_class",
    "interval",
    "chord_quality",
    "cadence",
    "motif",
];

/// Evaluate only source-grounded candidate invariants while a single control varies.
/// These are diagnostics, not quality scores; the controlled target channel is excluded.
fn non_target_invariance_diagnostics(
    style: Style,
    control: &str,
    observations: &[Observation],
) -> serde_json::Value {
    let family = style.grammar_profile().family;
    let valence_tonality_is_controlled = control == "valence"
        && style.spec().mode.is_none()
        && family != symthaea_music_theory::grammar::GrammarFamily::RagaModalArc;
    let baseline_seed = observations.first().map(|observation| observation.input.seed);
    let baseline = baseline_seed.and_then(|seed| {
        observations.iter().find(|observation| {
            observation.input
                == InputConfig {
                    seed,
                    ..InputConfig::BASE
                }
        })
    });
    let Some(baseline) = baseline else {
        return serde_json::json!({
            "status": "incomplete",
            "reason": "The unchanged seeded baseline is missing; candidate invariance cannot be evaluated.",
            "checked_channels": [],
            "coverage_complete": false,
        });
    };

    let points: Vec<&Observation> = observations
        .iter()
        .filter(|observation| observation.input != baseline.input)
        .collect();

    // A seed intervention intentionally resamples surface and structure.
    // Without a registered per-channel stochastic invariance hypothesis,
    // do not label tonal/meter/tempo outputs as non-target invariants.
    if control == "seed" {
        return serde_json::json!({
            "status": "not_applicable",
            "reason": "Seed variation is a robustness dimension; musical output channels may vary, and no per-channel invariance target is predeclared.",
            "declaration_basis": "seed_robustness_summary",
            "controlled_target_excluded": "seed_dependent_surface_and_structure",
            "baseline_included_in_grid": true,
            "nonbaseline_point_count": points.len(),
            "checked_channels": [],
            "coverage_complete": true,
            "interpretation": "Use the fixed-seed coverage and response summaries for robustness; do not score tonal, meter, or tempo invariance under seed variation without a separate registered hypothesis.",
        });
    }

    fn channel(
        name: &str,
        baseline: &Observation,
        points: &[&Observation],
        value: impl Fn(&Observation) -> serde_json::Value,
        changed: impl Fn(&Observation, &Observation) -> bool,
    ) -> serde_json::Value {
        let baseline_value = value(baseline);
        let mut changes = Vec::new();
        for point in points {
            if changed(point, baseline) {
                changes.push(serde_json::json!({
                    "control_value": point.value_label,
                    "baseline_value": baseline_value,
                    "observed_value": value(point),
                }));
            }
        }
        serde_json::json!({
            "channel": name,
            "baseline_value": baseline_value,
            "checked_point_count": points.len(),
            "changed_point_count": changes.len(),
            "changes": changes,
        })
    }

    let mut checked_channels = Vec::new();
    if control != "tonic" {
        checked_channels.push(channel(
            "absolute_key_tonic_pc",
            baseline,
            &points,
            |observation| serde_json::json!(observation.score_key_tonic_pc.rem_euclid(12)),
            |observation, baseline| {
                observation.score_key_tonic_pc.rem_euclid(12)
                    != baseline.score_key_tonic_pc.rem_euclid(12)
            },
        ));
    }
    if !valence_tonality_is_controlled {
        checked_channels.push(channel(
            "key_tonality",
            baseline,
            &points,
            |observation| serde_json::json!(observation.score_key_tonality),
            |observation, baseline| observation.score_key_tonality != baseline.score_key_tonality,
        ));
    }
    checked_channels.push(channel(
        "meter_beats",
        baseline,
        &points,
        |observation| serde_json::json!(observation.meter_beats),
        |observation, baseline| observation.meter_beats != baseline.meter_beats,
    ));
    if control != "arousal" {
        checked_channels.push(channel(
            "tempo_bpm",
            baseline,
            &points,
            |observation| serde_json::json!(observation.tempo_bpm),
            |observation, baseline| {
                (observation.tempo_bpm - baseline.tempo_bpm).abs() > 0.001
            },
        ));
    }

    serde_json::json!({
        "status": "diagnostic",
        "declaration_basis": "single_factor_intervention_and_source_declared_key_meter_tempo_roles",
        "controlled_target_excluded": match control {
            "tonic" => "absolute_key_tonic_pc",
            "valence" => "key_tonality",
            "arousal" => "tempo_bpm",
            "energy" => "phrase_model_and_other_energy_sensitive_structure",
            "bars" => "observed_score_span_bars",
            "seed" => "seed_dependent_surface_and_structure",
            _ => "unknown",
        },
        "baseline_included_in_grid": true,
        "nonbaseline_point_count": points.len(),
        "checked_channels": checked_channels,
        "interpretation": "Candidate non-target invariants only. Deviations are surfaced for investigation and are not automatically interpreted as quality failures.",
    })
}

/// Reports target fidelity separately from observed displacement. A control
/// is scored only when the active grammar path exposes a source-grounded
/// target contract; unknown target semantics remain explicitly unscored.
fn control_target_evaluation(
    style: Style,
    control: &str,
    observations: &[Observation],
) -> serde_json::Value {
    let family = style.grammar_profile().family;
    let spec = style.spec();

    if control == "valence" {
        if let Some(mode) = spec.mode {
            return serde_json::json!({
                "control": control,
                "target_fidelity": {
                    "status": "preset_mode_override",
                    "pinned_mode": format!("{mode:?}"),
                    "reason": "The style preset pins mode; valence-to-major/minor adherence is not scored.",
                },
                "non_target_invariance": non_target_invariance_diagnostics(style, control, observations),
            });
        }
        if family == symthaea_music_theory::grammar::GrammarFamily::RagaModalArc {
            return serde_json::json!({
                "control": control,
                "target_fidelity": {
                    "status": "engine_owned_default",
                    "reason": "RagaModalArc defaults to minor when no mode override exists; valence is not its tonality target.",
                },
                "non_target_invariance": non_target_invariance_diagnostics(style, control, observations),
            });
        }

        let points: Vec<serde_json::Value> = observations
            .iter()
            .map(|observation| {
                // Zero is neutral, not a positive-valence target. Major/minor
                // is only a tonal proxy for directional valence, not a claim
                // that key mode determines perceived emotion.
                let expected_tonality = if observation.input.valence > 0.0 {
                    Some("Major")
                } else if observation.input.valence < 0.0 {
                    Some("Minor")
                } else {
                    None
                };
                let matches_target = expected_tonality
                    .map(|expected| observation.score_key_tonality == expected);
                serde_json::json!({
                    "control_value": observation.value_label,
                    "target_applicability": if expected_tonality.is_some() {
                        "directional_tonal_proxy"
                    } else {
                        "neutral_unscored"
                    },
                    "expected_tonality": expected_tonality,
                    "realized_tonality": observation.score_key_tonality,
                    "matches_target": matches_target,
                })
            })
            .collect();
        let matched = points
            .iter()
            .filter(|point| point["matches_target"].as_bool() == Some(true))
            .count();
        let scored_count = points
            .iter()
            .filter(|point| point["target_applicability"] == "directional_tonal_proxy")
            .count();
        let count = points.len();
        return serde_json::json!({
            "control": control,
            "target_fidelity": {
                "status": "measured",
                "target_metric": "negative_valence_minor_positive_valence_major_exact_zero_neutral_unscored",
                "tested_point_count": count,
                "scored_point_count": scored_count,
                "unscored_neutral_point_count": count - scored_count,
                "matching_point_count": matched,
                "mismatching_point_count": scored_count - matched,
                "adherence_fraction": matched as f64 / scored_count.max(1) as f64,
                "interpretation": "Directional tonal proxy only; this is not a perceptual-valence measurement.",
                "points": points,
            },
            "non_target_invariance": non_target_invariance_diagnostics(style, control, observations),
        });
    }

    if control == "arousal" {
        let attitude_multiplier = if matches!(
            family,
            symthaea_music_theory::grammar::GrammarFamily::BluesCallResponse
                | symthaea_music_theory::grammar::GrammarFamily::JazzChorus
        ) {
            // These dedicated engines use spec.tempo(arousal) directly.
            1.0_f32
        } else {
            match spec.attitude {
                Some(symthaea_music_theory::spec::Attitude::Joy) => 1.08,
                Some(symthaea_music_theory::spec::Attitude::Grief) => 0.85,
                _ => 1.0,
            }
        };
        let points: Vec<serde_json::Value> = observations
            .iter()
            .map(|observation| {
                let expected_bpm = spec.tempo(observation.input.arousal) * attitude_multiplier;
                let actual_bpm = observation.tempo_bpm;
                let absolute_error = (actual_bpm - expected_bpm).abs();
                serde_json::json!({
                    "control_value": observation.value_label,
                    "expected_tempo_bpm": expected_bpm,
                    "realized_tempo_bpm": actual_bpm,
                    "absolute_error_bpm": absolute_error,
                    "matches_target_within_tolerance": absolute_error <= 0.001,
                })
            })
            .collect();
        let matched = points
            .iter()
            .filter(|point| point["matches_target_within_tolerance"].as_bool() == Some(true))
            .count();
        let max_error = points
            .iter()
            .filter_map(|point| point["absolute_error_bpm"].as_f64())
            .fold(0.0_f64, f64::max);
        let count = points.len();
        return serde_json::json!({
            "control": control,
            "target_fidelity": {
                "status": "measured",
                "target_metric": "realized_tempo_matches_spec_linear_arousal_curve",
                "tolerance_bpm": 0.001,
                "tested_point_count": count,
                "matching_point_count": matched,
                "mismatching_point_count": count - matched,
                "adherence_fraction": matched as f64 / count.max(1) as f64,
                "max_absolute_error_bpm": max_error,
                "points": points,
            },
            "interpretation": "This is fidelity to the source-declared tempo mapping, not a perceptual-arousal measurement.",
            "non_target_invariance": non_target_invariance_diagnostics(style, control, observations),
        });
    }

    if control == "energy"
        && family == symthaea_music_theory::grammar::GrammarFamily::PeriodSentence
    {
        let points: Vec<serde_json::Value> = observations
            .iter()
            .map(|observation| {
                let expected_model = if observation.input.arousal > 0.6
                    || observation.input.energy > 0.75
                {
                    "sentence"
                } else {
                    "period"
                };
                let realized_model = observation.phrase_model.as_deref().unwrap_or("missing");
                serde_json::json!({
                    "control_value": observation.value_label,
                    "expected_phrase_model": expected_model,
                    "realized_phrase_model": realized_model,
                    "matches_target": realized_model == expected_model,
                })
            })
            .collect();
        let matched = points
            .iter()
            .filter(|point| point["matches_target"].as_bool() == Some(true))
            .count();
        let count = points.len();
        return serde_json::json!({
            "control": control,
            "target_fidelity": {
                "status": "measured",
                "target_metric": "period_sentence_model_matches_arousal_energy_threshold_rule",
                "tested_point_count": count,
                "matching_point_count": matched,
                "mismatching_point_count": count - matched,
                "adherence_fraction": matched as f64 / count.max(1) as f64,
                "points": points,
            },
            "interpretation": "This checks the PeriodSentence threshold policy, not perceived musical energy.",
            "non_target_invariance": non_target_invariance_diagnostics(style, control, observations),
        });
    }

    if control == "bars" {
        let has_plan_contract = observations.iter().all(|observation| {
            observation.plan_requested_bars.is_some()
                && observation.plan_realized_bars.is_some()
                && observation.plan_bars_per_chorus.unwrap_or(0) > 0
        });
        if !has_plan_contract {
            return serde_json::json!({
                "control": control,
                "target_fidelity": {
                    "status": "not_predeclared",
                    "reason": "This grammar route does not yet expose a typed requested-to-realized bar quantization contract.",
                },
                "non_target_invariance": non_target_invariance_diagnostics(style, control, observations),
            });
        }

        let points: Vec<serde_json::Value> = observations
            .iter()
            .map(|observation| {
                let plan_requested_bars = observation.plan_requested_bars.unwrap();
                let plan_realized_bars = observation.plan_realized_bars.unwrap();
                let bars_per_chorus = observation.plan_bars_per_chorus.unwrap();
                let expected_realized_bars =
                    observation.input.bars.max(1).div_ceil(bars_per_chorus) * bars_per_chorus;
                let plan_request_matches_input = plan_requested_bars == observation.input.bars;
                let plan_quantization_matches_policy =
                    plan_realized_bars == expected_realized_bars;
                let score_span_matches_plan =
                    (observation.observed_score_span_bars - plan_realized_bars as f64).abs()
                        <= 0.001;
                serde_json::json!({
                    "control_value": observation.value_label,
                    "requested_bars": observation.input.bars,
                    "plan_requested_bars": plan_requested_bars,
                    "bars_per_chorus": bars_per_chorus,
                    "expected_realized_bars": expected_realized_bars,
                    "plan_realized_bars": plan_realized_bars,
                    "observed_score_span_bars": observation.observed_score_span_bars,
                    "plan_request_matches_input": plan_request_matches_input,
                    "plan_quantization_matches_policy": plan_quantization_matches_policy,
                    "score_span_matches_plan": score_span_matches_plan,
                    "matches_target": plan_request_matches_input
                        && plan_quantization_matches_policy
                        && score_span_matches_plan,
                })
            })
            .collect();
        let matched = points
            .iter()
            .filter(|point| point["matches_target"].as_bool() == Some(true))
            .count();
        let count = points.len();
        return serde_json::json!({
            "control": control,
            "target_fidelity": {
                "status": "measured",
                "target_metric": "chorus_quantized_plan_and_observed_score_span_match_requested_bars",
                "quantization_policy": "round_up_to_whole_declared_chorus",
                "observed_span_metric": "max(note.onset + note.duration) / meter",
                "tolerance_bars": 0.001,
                "tested_point_count": count,
                "matching_point_count": matched,
                "mismatching_point_count": count - matched,
                "adherence_fraction": matched as f64 / count.max(1) as f64,
                "points": points,
            },
            "non_target_invariance": non_target_invariance_diagnostics(style, control, observations),
        });
    }

    if control != "tonic" {
        return serde_json::json!({
            "control": control,
            "target_fidelity": {
                "status": "not_predeclared",
                "reason": "No style-valid target metric is yet declared for this control and grammar route.",
            },
            "non_target_invariance": non_target_invariance_diagnostics(style, control, observations),
        });
    }

    let target_points: Vec<serde_json::Value> = observations
        .iter()
        .map(|observation| {
            let requested_tonic_pc = observation.input.tonic.rem_euclid(12);
            let realized_tonic_pc = observation.score_key_tonic_pc.rem_euclid(12);
            serde_json::json!({
                "control_value": observation.value_label,
                "requested_tonic_pc": requested_tonic_pc,
                "realized_tonic_pc": realized_tonic_pc,
                "matches_target": requested_tonic_pc == realized_tonic_pc,
            })
        })
        .collect();
    let matching_points = target_points
        .iter()
        .filter(|point| point["matches_target"].as_bool() == Some(true))
        .count();
    let target_point_count = target_points.len();

    // These layers are candidate invariants under global transposition. The
    // absolute-key harmony and register-sensitive orchestration layers are
    // intentionally excluded. Violations are diagnostic, not an automatic
    // failure: individual styles may expose constraints needing investigation.
    let nonbaseline_points: Vec<&Observation> = observations
        .iter()
        .filter(|observation| observation.input.tonic.rem_euclid(12) != BASE_TONIC.rem_euclid(12))
        .collect();
    let mut invariant_violations = Vec::new();
    let mut points_without_violations = 0;
    for observation in &nonbaseline_points {
        let mut point_violation_count = 0;
        for (&(layer, _, _), &distance) in
            LAYERS.iter().zip(observation.layer_distances.iter())
        {
            if TONIC_CANDIDATE_INVARIANT_LAYERS.contains(&layer)
                && distance > LAYER_CHANGE_THRESHOLD
            {
                point_violation_count += 1;
                invariant_violations.push(serde_json::json!({
                    "control_value": observation.value_label,
                    "layer": layer,
                    "distance_from_baseline": distance,
                }));
            }
        }
        if point_violation_count == 0 {
            points_without_violations += 1;
        }
    }

    serde_json::json!({
        "control": control,
        "target_fidelity": {
            "status": "measured",
            "target_metric": "realized_key_tonic_pc_matches_requested_pitch_class",
            "tested_point_count": target_point_count,
            "matching_point_count": matching_points,
            "mismatching_point_count": target_point_count - matching_points,
            "adherence_fraction": matching_points as f64
                / target_point_count.max(1) as f64,
            "points": target_points,
        },
        "non_target_invariance": {
            "status": "diagnostic",
            "candidate_invariant_layers": TONIC_CANDIDATE_INVARIANT_LAYERS,
            "layer_change_threshold": LAYER_CHANGE_THRESHOLD,
            "nonbaseline_point_count": nonbaseline_points.len(),
            "points_without_candidate_invariance_violations": points_without_violations,
            "points_with_candidate_invariance_violations":
                nonbaseline_points.len() - points_without_violations,
            "violations": invariant_violations,
            "interpretation": concat!(
                "Candidate transposition invariants are reported separately from key-target ",
                "fidelity. Violations are diagnostic and are not quality or authenticity scores."
            ),
        },
    })
}

fn curve_json(style: Style, control: &str, base_seed: u64) -> serde_json::Value {
    let base = ControlPoint {
        value_label: "baseline".into(),
        numeric_value: None,
        input: InputConfig {
            seed: base_seed,
            ..InputConfig::BASE
        },
    };
    let baseline_intent = base.input.intent();
    let spec = style.spec();
    let baseline_realization = compose_with_grammar_plan(
        style.grammar_profile(),
        &baseline_intent,
        &spec,
    );
    let baseline_score = baseline_realization.score;
    let base_hash = fingerprint::exact_fingerprint(&baseline_score);
    let base_structure = fingerprint::structural_fingerprint(
        &baseline_score,
        &baseline_realization.form,
    );

    let points = control_points(control, base_seed);
    let observations: Vec<Observation> = points
        .iter()
        .map(|point| observe(style, point, base_hash, &base_structure))
        .collect();

    let mut unique_fingerprints = BTreeSet::new();
    let mut layers_changed = BTreeSet::new();
    let mut max_structural_distance = 0.0_f64;
    for observation in &observations {
        unique_fingerprints.insert(observation.exact_hash);
        max_structural_distance =
            max_structural_distance.max(observation.overall_structural_distance);
        for (&(name, _, _), &distance) in LAYERS.iter().zip(observation.layer_distances.iter()) {
            if distance > LAYER_CHANGE_THRESHOLD {
                layers_changed.insert(name.to_string());
            }
        }
    }

    let tempo_points: Vec<(f64, f64)> = observations
        .iter()
        .filter_map(|point| point.numeric_value.map(|value| (value, point.tempo_bpm as f64)))
        .collect();
    let duration_points: Vec<(f64, f64)> = observations
        .iter()
        .filter_map(|point| point.numeric_value.map(|value| (value, point.total_duration_beats)))
        .collect();
    let note_count_points: Vec<(f64, f64)> = observations
        .iter()
        .filter_map(|point| {
            point
                .numeric_value
                .map(|value| (value, point.score_note_count as f64))
        })
        .collect();
    let velocity_points: Vec<(f64, f64)> = observations
        .iter()
        .filter_map(|point| point.numeric_value.map(|value| (value, point.mean_velocity)))
        .collect();
    let duration_mean_points: Vec<(f64, f64)> = observations
        .iter()
        .filter_map(|point| {
            point
                .numeric_value
                .map(|value| (value, point.mean_note_duration_beats))
        })
        .collect();
    let intensity_points: Vec<(f64, f64)> = observations
        .iter()
        .filter_map(|point| {
            point
                .numeric_value
                .map(|value| (value, point.mean_section_intensity))
        })
        .collect();
    let velocity_spread_points: Vec<(f64, f64)> = observations
        .iter()
        .filter_map(|point| {
            point
                .numeric_value
                .map(|value| (value, point.velocity_standard_deviation))
        })
        .collect();
    let intensity_spread_points: Vec<(f64, f64)> = observations
        .iter()
        .filter_map(|point| {
            point.numeric_value.map(|value| {
                (value, point.section_intensity_standard_deviation)
            })
        })
        .collect();

    let baseline_input = InputConfig {
        seed: base_seed,
        ..InputConfig::BASE
    };
    let baseline_observation = observations
        .iter()
        .find(|point| point.input == baseline_input)
        .expect("every control grid includes the unchanged seeded baseline");
    let exact_events_changed = unique_fingerprints.len() > 1;
    let structural_distance_changed = max_structural_distance > MONOTONIC_EPSILON;
    let structural_layers_changed = !layers_changed.is_empty();
    let tempo_changed = observations
        .iter()
        .any(|point| {
            (point.tempo_bpm - baseline_observation.tempo_bpm).abs() > MONOTONIC_EPSILON as f32
        });
    let meter_changed = observations
        .iter()
        .any(|point| point.meter_beats != baseline_observation.meter_beats);
    let duration_changed = observations.iter().any(|point| {
        (point.total_duration_beats - baseline_observation.total_duration_beats).abs()
            > MONOTONIC_EPSILON
    });
    let note_count_changed = observations
        .iter()
        .any(|point| point.score_note_count != baseline_observation.score_note_count);
    let pitch_diversity_changed = observations
        .iter()
        .any(|point| point.distinct_midi_pitches != baseline_observation.distinct_midi_pitches);
    let pitch_span_changed = observations
        .iter()
        .any(|point| point.pitch_span_semitones != baseline_observation.pitch_span_semitones);
    let tonic_changed = observations
        .iter()
        .any(|point| point.score_key_tonic_pc != baseline_observation.score_key_tonic_pc);
    let mean_velocity_changed = observations.iter().any(|point| {
        (point.mean_velocity - baseline_observation.mean_velocity).abs() > MONOTONIC_EPSILON
    });
    let velocity_spread_changed = observations.iter().any(|point| {
        (point.velocity_standard_deviation - baseline_observation.velocity_standard_deviation)
            .abs()
            > MONOTONIC_EPSILON
    });
    let note_duration_changed = observations.iter().any(|point| {
        (point.mean_note_duration_beats - baseline_observation.mean_note_duration_beats).abs()
            > MONOTONIC_EPSILON
    });
    let section_intensity_changed = observations.iter().any(|point| {
        (point.mean_section_intensity - baseline_observation.mean_section_intensity).abs()
            > MONOTONIC_EPSILON
    });
    let intensity_spread_changed = observations.iter().any(|point| {
        (point.section_intensity_standard_deviation
            - baseline_observation.section_intensity_standard_deviation)
            .abs()
            > MONOTONIC_EPSILON
    });
    let emphasis_changed = observations
        .iter()
        .any(|point| point.emphasis_counts != baseline_observation.emphasis_counts);
    let form_availability_changed = observations.iter().any(|point| {
        point.form_section_count != baseline_observation.form_section_count
    });
    let no_observed_change_at_tested_points = !(
        exact_events_changed
            || structural_distance_changed
            || structural_layers_changed
            || tempo_changed
            || meter_changed
            || duration_changed
            || note_count_changed
            || pitch_diversity_changed
            || pitch_span_changed
            || tonic_changed
            || mean_velocity_changed
            || velocity_spread_changed
            || note_duration_changed
            || section_intensity_changed
            || intensity_spread_changed
            || emphasis_changed
            || form_availability_changed
    );

    serde_json::json!({
        "control": control,
        "baseline_seed": base_seed,
        "baseline_value": base.input.json(),
        "point_count": observations.len(),
        "unique_exact_fingerprints": unique_fingerprints.len(),
        "distinct_exact_output_fraction": unique_fingerprints.len() as f64
            / observations.len().max(1) as f64,
        "max_structural_distance": max_structural_distance,
        "layers_changed_at_least_once": layers_changed,
        "control_target_evaluation": control_target_evaluation(style, control, &observations),
        "observed_changes_from_baseline": {
            "exact_note_event_fingerprint": exact_events_changed,
            "overall_structural_distance": structural_distance_changed,
            "structural_layers": structural_layers_changed,
            "tempo_bpm": tempo_changed,
            "meter_beats": meter_changed,
            "total_duration_beats": duration_changed,
            "score_note_count": note_count_changed,
            "distinct_midi_pitches": pitch_diversity_changed,
            "pitch_span_semitones": pitch_span_changed,
            "absolute_key_tonic": tonic_changed,
            "mean_velocity": mean_velocity_changed,
            "velocity_spread": velocity_spread_changed,
            "mean_note_duration_beats": note_duration_changed,
            "mean_section_intensity": section_intensity_changed,
            "section_intensity_spread": intensity_spread_changed,
            "emphasis_counts": emphasis_changed,
            "form_availability_or_section_count": form_availability_changed,
            "no_observed_change_at_tested_points": no_observed_change_at_tested_points,
        },
        "response_rank": {
            "method": "spearman_rho_average_ranks_for_ties",
            "tempo_bpm_vs_control": spearman_rank_response(&tempo_points),
            "total_duration_beats_vs_control": spearman_rank_response(&duration_points),
            "score_note_count_vs_control": spearman_rank_response(&note_count_points),
            "mean_velocity_vs_control": spearman_rank_response(&velocity_points),
            "mean_note_duration_beats_vs_control": spearman_rank_response(&duration_mean_points),
            "mean_section_intensity_vs_control": spearman_rank_response(&intensity_points),
            "velocity_spread_vs_control": spearman_rank_response(&velocity_spread_points),
            "section_intensity_spread_vs_control": spearman_rank_response(&intensity_spread_points),
            "note": "Rank associations are reported independently from monotonicity labels and target-fidelity contracts.",
        },
        "response_shape": {
            "tempo_bpm_vs_control": monotonicity(&tempo_points),
            "total_duration_beats_vs_control": monotonicity(&duration_points),
            "score_note_count_vs_control": monotonicity(&note_count_points),
            "mean_velocity_vs_control": monotonicity(&velocity_points),
            "mean_note_duration_beats_vs_control": monotonicity(&duration_mean_points),
            "mean_section_intensity_vs_control": monotonicity(&intensity_points),
            "velocity_spread_vs_control": monotonicity(&velocity_spread_points),
            "section_intensity_spread_vs_control": monotonicity(&intensity_spread_points),
            "structural_distance_note": concat!(
                "Per-point distance from baseline is recorded below. It is a ",
                "displacement magnitude, not a directional target metric, so no ",
                "monotonicity class is assigned to it."
            ),
            "note": concat!(
                "Shape labels describe observed points only; they are not correctness ",
                "or quality scores. Tonic and seed are not treated as ordinal axes. ",
                "Dynamics and key are reported separately because the exact fingerprint ",
                "omits velocity/emphasis and pitch-class structural features are ",
                "transposition-invariant."
            )
        },
        "points": observations
            .iter()
            .map(|observation| observation_json(observation, base_hash))
            .collect::<Vec<_>>(),
    })
}

const OBSERVED_CHANGE_CHANNELS: &[&str] = &[
    "exact_note_event_fingerprint",
    "overall_structural_distance",
    "structural_layers",
    "tempo_bpm",
    "meter_beats",
    "total_duration_beats",
    "score_note_count",
    "distinct_midi_pitches",
    "pitch_span_semitones",
    "absolute_key_tonic",
    "mean_velocity",
    "velocity_spread",
    "mean_note_duration_beats",
    "mean_section_intensity",
    "section_intensity_spread",
    "emphasis_counts",
    "form_availability_or_section_count",
];

fn seed_robustness_summary(curves: &[serde_json::Value]) -> Vec<serde_json::Value> {
    let mut summaries = Vec::new();
    for control in ["tonic", "valence", "arousal", "energy", "bars", "seed"] {
        let matching: Vec<&serde_json::Value> = curves
            .iter()
            .filter(|curve| curve["control"].as_str() == Some(control))
            .collect();
        let expected_seed_set: BTreeSet<u64> = BASE_SEED_SET.into_iter().collect();
        let observed_seed_values: Vec<u64> = matching
            .iter()
            .filter_map(|curve| curve["baseline_seed"].as_u64())
            .collect();
        let observed_seed_set: BTreeSet<u64> = observed_seed_values.iter().copied().collect();
        let invalid_seed_record_count = matching.len() - observed_seed_values.len();
        let duplicate_seed_record_count = observed_seed_values.len() - observed_seed_set.len();
        let missing_seed_ids: Vec<serde_json::Value> = expected_seed_set
            .difference(&observed_seed_set)
            .copied()
            .map(serde_json::Value::from)
            .collect();
        let unexpected_seed_ids: Vec<serde_json::Value> = observed_seed_set
            .difference(&expected_seed_set)
            .copied()
            .map(serde_json::Value::from)
            .collect();
        let changed_count = matching
            .iter()
            .filter(|curve| {
                curve["observed_changes_from_baseline"]
                    ["no_observed_change_at_tested_points"]
                    .as_bool()
                    == Some(false)
            })
            .count();
        let unchanged_seeds: Vec<serde_json::Value> = matching
            .iter()
            .filter(|curve| {
                curve["observed_changes_from_baseline"]
                    ["no_observed_change_at_tested_points"]
                    .as_bool()
                    == Some(true)
            })
            .map(|curve| curve["baseline_seed"].clone())
            .collect();
        let channel_seed_counts: Vec<serde_json::Value> = OBSERVED_CHANGE_CHANNELS
            .iter()
            .map(|&channel| {
                let changed_seed_count = matching
                    .iter()
                    .filter(|curve| {
                        curve["observed_changes_from_baseline"][channel].as_bool()
                            == Some(true)
                    })
                    .count();
                serde_json::json!({
                    "channel": channel,
                    "seeds_with_observed_change": changed_seed_count,
                    "seed_count": matching.len(),
                })
            })
            .collect();

        let expected_seed_count = BASE_SEED_SET.len();
        let coverage_complete = matching.len() == expected_seed_count
            && invalid_seed_record_count == 0
            && duplicate_seed_record_count == 0
            && observed_seed_set == expected_seed_set;
        summaries.push(serde_json::json!({
            "control": control,
            "seed_count": matching.len(),
            "unique_seed_count": observed_seed_set.len(),
            "expected_seed_count": expected_seed_count,
            "expected_seed_ids": expected_seed_set,
            "observed_seed_ids": observed_seed_set,
            "missing_seed_ids": missing_seed_ids,
            "unexpected_seed_ids": unexpected_seed_ids,
            "duplicate_seed_record_count": duplicate_seed_record_count,
            "invalid_seed_record_count": invalid_seed_record_count,
            "coverage_complete": coverage_complete,
            "seeds_with_any_observed_change": changed_count,
            "seeds_with_no_observed_change": unchanged_seeds,
            "all_seeds_changed_at_least_one_recorded_channel":
                coverage_complete && changed_count == expected_seed_count,
            "per_channel_seed_counts": channel_seed_counts,
            "interpretation": concat!(
                "Coverage is complete only when there is exactly one valid record for each ",
                "expected baseline seed. Counts describe measured output changes, not ",
                "target-correctness, quality, or authenticity scores."
            ),
        }));
    }
    summaries
}

fn main() {
    let profiles: Vec<serde_json::Value> = Style::ALL
        .iter()
        .copied()
        .map(|style| {
            let mut curves = Vec::new();
            for base_seed in BASE_SEED_SET {
                for control in ["tonic", "valence", "arousal", "energy", "bars", "seed"] {
                    eprintln!(
                        "[muse-response-curve] style={style:?} control={control} baseline_seed={base_seed}"
                    );
                    curves.push(curve_json(style, control, base_seed));
                }
            }

            let seed_robustness_summary = seed_robustness_summary(&curves);
            serde_json::json!({
                "style": format!("{style:?}"),
                "grammar_family": format!("{:?}", style.grammar_family()),
                "dispatch_path": "compose_with_grammar_plan (same path used by compose_styled)",
                "seed_robustness_summary": seed_robustness_summary,
                "curves": curves,
            })
        })
        .collect();

    let report = serde_json::json!({
        "schema": "muse-style-control-response-curves-v1",
        "engine_package_version": MUSIC_THEORY_ENGINE_VERSION,
        "measurement_scope": "symbolic composition; not audio or perceptual validation",
        "protocol": {
            "fixed_base": InputConfig::BASE.json(),
            "one_factor_at_a_time": true,
            "style_count": profiles.len(),
            "curve_count": profiles.len() * BASE_SEED_SET.len() * 6,
            "point_count": profiles.iter().map(|profile| {
                profile["curves"].as_array().unwrap().iter().map(|curve| {
                    curve["point_count"].as_u64().unwrap_or(0)
                }).sum::<u64>()
            }).sum::<u64>(),
            "layer_change_threshold": LAYER_CHANGE_THRESHOLD,
            "control_evaluation_schema": "muse-control-evaluation-v2",
            "fingerprint_layers": LAYERS.iter().map(|(name, start, len)| {
                serde_json::json!({"name": name, "start": start, "length": len})
            }).collect::<Vec<_>>(),
            "control_grids": {
                "tonic": ["C", "Eb", "E", "G", "Bb"],
                "valence": [-1.0, -0.5, 0.0, 0.5, 1.0],
                "arousal": [0.0, 0.25, 0.5, 0.75, 1.0],
                "energy": [0.0, 0.25, 0.5, 0.75, 1.0],
                "bars": [8, 12, 16, 20, 24],
                "baseline_seed_set": BASE_SEED_SET,
                "seed_offsets_within_each_curve": [0, 1, 2, 3],
            },
        },
        "interpretation": concat!(
            "Observed deterministic symbolic response curves only. A structural or ",
            "exact-output change does not itself imply improved quality, a correct target ",
            "effect, audio-render success, style distinctness, or cultural authenticity. ",
            "Compare per-layer effects and non-target invariance before making capability ",
            "claims."
        ),
        "profiles": profiles,
    });

    let rendered =
        serde_json::to_string_pretty(&report).expect("serialize response curve report");
    if let Ok(path) = std::env::var("MUSE_STYLE_RESPONSE_JSON_OUT") {
        let path = Path::new(&path);
        fs::write(path, &rendered).expect("write response curve report");
        eprintln!(
            "Wrote {} style response curves to {}",
            report["protocol"]["curve_count"],
            path.display()
        );
        eprintln!(
            concat!(
                "This artifact is symbolic evidence only; it does not assert render ",
                "success or perceptual distinctness."
            )
        );
    }
    println!("{rendered}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_control_grid_contains_its_unchanged_seeded_baseline() {
        for base_seed in BASE_SEED_SET {
            let baseline = InputConfig {
                seed: base_seed,
                ..InputConfig::BASE
            };
            for control in ["tonic", "valence", "arousal", "energy", "bars", "seed"] {
                let points = control_points(control, base_seed);
                assert!(
                    points.iter().any(|point| point.input == baseline),
                    "control={control}, seed={base_seed}"
                );
            }
        }
    }

    #[test]
    fn seed_robustness_set_is_unique_and_seed_offsets_are_reproducible() {
        let unique_seeds: BTreeSet<u64> = BASE_SEED_SET.into_iter().collect();
        assert_eq!(unique_seeds.len(), BASE_SEED_SET.len());

        for base_seed in BASE_SEED_SET {
            let points = control_points("seed", base_seed);
            let observed_seeds: Vec<u64> =
                points.iter().map(|point| point.input.seed).collect();
            assert_eq!(
                observed_seeds,
                vec![base_seed, base_seed + 1, base_seed + 2, base_seed + 3]
            );
        }
    }

    #[test]
    fn continuous_controls_have_five_ordered_levels() {
        for control in ["valence", "arousal", "energy", "bars"] {
            let points = control_points(control, BASE_SEED);
            assert_eq!(points.len(), 5, "{control}");
            let values: Vec<f64> = points
                .iter()
                .map(|point| point.numeric_value.unwrap())
                .collect();
            assert!(values.windows(2).all(|pair| pair[0] < pair[1]), "{control}: {values:?}");
        }
    }

    #[test]
    fn spearman_rank_response_handles_direction_ties_and_flat_outputs() {
        let increasing = spearman_rank_response(&[(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)]);
        assert_eq!(increasing["status"].as_str(), Some("measured"));
        assert!((increasing["rho"].as_f64().unwrap() - 1.0).abs() < 1.0e-12);

        let decreasing = spearman_rank_response(&[(1.0, 30.0), (2.0, 20.0), (3.0, 10.0)]);
        assert!((decreasing["rho"].as_f64().unwrap() + 1.0).abs() < 1.0e-12);

        let tied = spearman_rank_response(&[(1.0, 2.0), (2.0, 2.0), (3.0, 4.0)]);
        assert!((tied["rho"].as_f64().unwrap() - 0.866_025_403_784_438_6).abs() < 1.0e-9);

        let flat = spearman_rank_response(&[(1.0, 4.0), (2.0, 4.0), (3.0, 4.0)]);
        assert_eq!(flat["status"].as_str(), Some("constant_output"));
        assert!(flat["rho"].is_null());

        let non_numeric = spearman_rank_response(&[]);
        assert_eq!(non_numeric["status"].as_str(), Some("not_applicable_non_numeric_control"));
        assert!(non_numeric["rho"].is_null());
    }

    #[test]
    fn response_rank_is_available_for_numeric_control_but_not_tonic_labels() {
        let arousal = curve_json(Style::Classical, "arousal", BASE_SEED);
        let tempo_rank = &arousal["response_rank"]["tempo_bpm_vs_control"];
        assert_eq!(tempo_rank["status"].as_str(), Some("measured"));
        assert_eq!(tempo_rank["point_count"].as_u64(), Some(5));
        assert!(tempo_rank["rho"].as_f64().is_some());

        let tonic = curve_json(Style::Classical, "tonic", BASE_SEED);
        let tonic_tempo_rank = &tonic["response_rank"]["tempo_bpm_vs_control"];
        assert_eq!(
            tonic_tempo_rank["status"].as_str(),
            Some("not_applicable_non_numeric_control")
        );
        assert!(tonic_tempo_rank["rho"].is_null());
    }

    #[test]
    fn monotonicity_classifier_does_not_invent_a_direction() {
        assert_eq!(monotonicity(&[(0.0, 1.0), (1.0, 2.0)]), "nondecreasing");
        assert_eq!(monotonicity(&[(0.0, 2.0), (1.0, 1.0)]), "nonincreasing");
        assert_eq!(monotonicity(&[(0.0, 1.0), (1.0, 1.0)]), "flat");
        assert_eq!(monotonicity(&[(0.0, 1.0), (1.0, 2.0), (2.0, 1.0)]), "non_monotonic");
    }

    #[test]
    fn seed_summary_counts_any_and_per_channel_changes_without_scoring_quality() {
        let make_curve = |seed: u64, no_change: bool, velocity_changed: bool| {
            serde_json::json!({
                "control": "arousal",
                "baseline_seed": seed,
                "observed_changes_from_baseline": {
                    "no_observed_change_at_tested_points": no_change,
                    "mean_velocity": velocity_changed,
                    "exact_note_event_fingerprint": !no_change,
                    "overall_structural_distance": !no_change,
                    "structural_layers": !no_change,
                    "tempo_bpm": !no_change,
                    "meter_beats": false,
                    "total_duration_beats": !no_change,
                    "score_note_count": !no_change,
                    "distinct_midi_pitches": !no_change,
                    "pitch_span_semitones": !no_change,
                    "absolute_key_tonic": false,
                    "velocity_spread": velocity_changed,
                    "mean_note_duration_beats": false,
                    "mean_section_intensity": false,
                    "section_intensity_spread": false,
                    "emphasis_counts": false,
                    "form_availability_or_section_count": false,
                },
            })
        };
        let curves = vec![
            make_curve(90_210, false, true),
            make_curve(130_363, true, false),
            make_curve(222_731, false, false),
            make_curve(918_273, false, true),
        ];
        let summaries = seed_robustness_summary(&curves);
        let arousal = summaries
            .iter()
            .find(|summary| summary["control"] == "arousal")
            .expect("arousal summary");
        assert_eq!(arousal["seed_count"].as_u64(), Some(4));
        assert_eq!(arousal["expected_seed_count"].as_u64(), Some(4));
        assert_eq!(arousal["coverage_complete"].as_bool(), Some(true));
        assert_eq!(arousal["seeds_with_any_observed_change"].as_u64(), Some(3));
        assert_eq!(arousal["seeds_with_no_observed_change"][0].as_u64(), Some(130_363));
        let velocity = arousal["per_channel_seed_counts"]
            .as_array()
            .unwrap()
            .iter()
            .find(|channel| channel["channel"] == "mean_velocity")
            .unwrap();
        assert_eq!(velocity["seeds_with_observed_change"].as_u64(), Some(2));
        assert!(arousal["interpretation"].as_str().unwrap().contains("not target-correctness"));

        let incomplete = seed_robustness_summary(&curves[..3]);
        let incomplete_arousal = incomplete
            .iter()
            .find(|summary| summary["control"] == "arousal")
            .expect("incomplete arousal summary");
        assert_eq!(incomplete_arousal["seed_count"].as_u64(), Some(3));
        assert_eq!(incomplete_arousal["expected_seed_count"].as_u64(), Some(4));
        assert_eq!(incomplete_arousal["coverage_complete"].as_bool(), Some(false));
        assert_eq!(
            incomplete_arousal["all_seeds_changed_at_least_one_recorded_channel"].as_bool(),
            Some(false)
        );
    }

    #[test]
    fn seed_coverage_rejects_duplicate_seed_records_even_when_count_matches() {
        let make_curve = |seed: u64| {
            serde_json::json!({
                "control": "arousal",
                "baseline_seed": seed,
                "observed_changes_from_baseline": {
                    "no_observed_change_at_tested_points": false,
                },
            })
        };
        let curves = vec![
            make_curve(90_210),
            make_curve(90_210),
            make_curve(222_731),
            make_curve(918_273),
        ];
        let summary = seed_robustness_summary(&curves)
            .into_iter()
            .find(|row| row["control"] == "arousal")
            .expect("arousal summary");
        assert_eq!(summary["seed_count"].as_u64(), Some(4));
        assert_eq!(summary["unique_seed_count"].as_u64(), Some(3));
        assert_eq!(summary["duplicate_seed_record_count"].as_u64(), Some(1));
        assert_eq!(summary["missing_seed_ids"][0].as_u64(), Some(130_363));
        assert_eq!(summary["coverage_complete"].as_bool(), Some(false));
        assert_eq!(
            summary["all_seeds_changed_at_least_one_recorded_channel"].as_bool(),
            Some(false)
        );
    }

    #[test]
    fn seed_coverage_rejects_unexpected_seed_ids() {
        let make_curve = |seed: u64| {
            serde_json::json!({
                "control": "arousal",
                "baseline_seed": seed,
                "observed_changes_from_baseline": {
                    "no_observed_change_at_tested_points": false,
                },
            })
        };
        let curves = vec![
            make_curve(90_210),
            make_curve(123_456),
            make_curve(222_731),
            make_curve(918_273),
        ];
        let summary = seed_robustness_summary(&curves)
            .into_iter()
            .find(|row| row["control"] == "arousal")
            .expect("arousal summary");
        assert_eq!(summary["missing_seed_ids"][0].as_u64(), Some(130_363));
        assert_eq!(summary["unexpected_seed_ids"][0].as_u64(), Some(123_456));
        assert_eq!(summary["coverage_complete"].as_bool(), Some(false));
    }

    #[test]
    fn seed_coverage_rejects_invalid_seed_identity() {
        let make_curve = |seed: serde_json::Value| {
            serde_json::json!({
                "control": "arousal",
                "baseline_seed": seed,
                "observed_changes_from_baseline": {
                    "no_observed_change_at_tested_points": false,
                },
            })
        };
        let curves = vec![
            make_curve(serde_json::json!(90_210)),
            make_curve(serde_json::json!("130363")),
            make_curve(serde_json::json!(222_731)),
            make_curve(serde_json::json!(918_273)),
        ];
        let summary = seed_robustness_summary(&curves)
            .into_iter()
            .find(|row| row["control"] == "arousal")
            .expect("arousal summary");
        assert_eq!(summary["invalid_seed_record_count"].as_u64(), Some(1));
        assert_eq!(summary["missing_seed_ids"][0].as_u64(), Some(130_363));
        assert_eq!(summary["coverage_complete"].as_bool(), Some(false));
    }

    #[test]
    fn sweep_cardinality_matches_the_published_seeded_protocol() {
        assert_eq!(Style::ALL.len(), 29);
        assert_eq!(BASE_SEED_SET.len(), 4);
        let points_per_seed: usize = [
            "tonic", "valence", "arousal", "energy", "bars", "seed",
        ]
        .into_iter()
        .map(|control| control_points(control, BASE_SEED).len())
        .sum();
        assert_eq!(points_per_seed, 29);
        assert_eq!(Style::ALL.len() * BASE_SEED_SET.len() * 6, 696);
        assert_eq!(Style::ALL.len() * BASE_SEED_SET.len() * points_per_seed, 3_364);
    }

    #[test]
    fn tonic_curve_reports_target_fidelity_separately_from_invariance() {
        let curve = curve_json(Style::Classical, "tonic", BASE_SEED);
        let evaluation = &curve["control_target_evaluation"];
        let target = &evaluation["target_fidelity"];
        let invariance = &evaluation["non_target_invariance"];
        assert_eq!(target["status"].as_str(), Some("measured"));
        assert_eq!(target["tested_point_count"].as_u64(), Some(5));
        assert_eq!(target["points"].as_array().unwrap().len(), 5);
        let independently_counted = target["points"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|point| {
                point["requested_tonic_pc"].as_i64()
                    == point["realized_tonic_pc"].as_i64()
            })
            .count();
        assert_eq!(
            target["matching_point_count"].as_u64(),
            Some(independently_counted as u64)
        );
        assert_eq!(
            target["mismatching_point_count"].as_u64(),
            Some((5 - independently_counted) as u64)
        );
        assert_eq!(invariance["status"].as_str(), Some("diagnostic"));
        assert_eq!(invariance["nonbaseline_point_count"].as_u64(), Some(4));
        assert!(invariance["candidate_invariant_layers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|layer| layer.as_str() == Some("rhythm")));
        assert!(invariance["interpretation"]
            .as_str()
            .unwrap()
            .contains("reported separately"));
    }

    #[test]
    fn seed_control_target_is_explicitly_not_predeclared() {
        let curve = curve_json(Style::Classical, "seed", BASE_SEED);
        assert_eq!(
            curve["control_target_evaluation"]["target_fidelity"]["status"].as_str(),
            Some("not_predeclared")
        );
        let invariance = &curve["control_target_evaluation"]["non_target_invariance"];
        assert_eq!(invariance["status"].as_str(), Some("not_applicable"));
        assert_eq!(
            invariance["controlled_target_excluded"].as_str(),
            Some("seed_dependent_surface_and_structure")
        );
        assert_eq!(invariance["nonbaseline_point_count"].as_u64(), Some(3));
        assert_eq!(invariance["checked_channels"].as_array().unwrap().len(), 0);
        assert!(invariance["reason"]
            .as_str()
            .unwrap()
            .contains("no per-channel invariance target is predeclared"));
    }

    #[test]
    fn non_target_invariants_exclude_the_controlled_channel_and_report_deviations() {
        let valence_curve = curve_json(Style::Classical, "valence", BASE_SEED);
        let valence = &valence_curve["control_target_evaluation"]["non_target_invariance"];
        let valence_channels = valence["checked_channels"].as_array().unwrap();
        assert!(valence_channels.iter().any(|item| item["channel"] == "absolute_key_tonic_pc"));
        assert!(valence_channels.iter().any(|item| item["channel"] == "meter_beats"));
        assert!(valence_channels.iter().any(|item| item["channel"] == "tempo_bpm"));
        assert!(!valence_channels.iter().any(|item| item["channel"] == "key_tonality"));
        for channel_name in ["absolute_key_tonic_pc", "meter_beats", "tempo_bpm"] {
            let channel = valence_channels
                .iter()
                .find(|item| item["channel"] == channel_name)
                .expect("declared valence non-target invariant");
            assert_eq!(channel["checked_point_count"].as_u64(), Some(4));
            assert_eq!(
                channel["changed_point_count"].as_u64(),
                Some(0),
                "unexpected non-target change: {:?}",
                channel
            );
        }

        let arousal_curve = curve_json(Style::Classical, "arousal", BASE_SEED);
        let arousal = &arousal_curve["control_target_evaluation"]["non_target_invariance"];
        let arousal_channels = arousal["checked_channels"].as_array().unwrap();
        assert!(arousal_channels.iter().any(|item| item["channel"] == "absolute_key_tonic_pc"));
        assert!(arousal_channels.iter().any(|item| item["channel"] == "key_tonality"));
        assert!(arousal_channels.iter().any(|item| item["channel"] == "meter_beats"));
        assert!(!arousal_channels.iter().any(|item| item["channel"] == "tempo_bpm"));
        for channel_name in ["absolute_key_tonic_pc", "key_tonality", "meter_beats"] {
            let channel = arousal_channels
                .iter()
                .find(|item| item["channel"] == channel_name)
                .expect("declared arousal non-target invariant");
            assert_eq!(channel["checked_point_count"].as_u64(), Some(4));
            assert_eq!(
                channel["changed_point_count"].as_u64(),
                Some(0),
                "unexpected non-target change: {:?}",
                channel
            );
        }

        let tempo_channel = arousal_curve["control_target_evaluation"]["target_fidelity"]["points"]
            .as_array()
            .unwrap();
        assert_eq!(tempo_channel.len(), 5);
    }

    #[test]
    fn arousal_tempo_target_matches_the_declared_spec_curve() {
        for style in [Style::Classical, Style::AfroCuban, Style::Minimalism] {
            let curve = curve_json(style, "arousal", BASE_SEED);
            let target = &curve["control_target_evaluation"]["target_fidelity"];
            assert_eq!(
                target["status"].as_str(),
                Some("measured"),
                "{style:?}"
            );
            assert_eq!(target["tested_point_count"].as_u64(), Some(5));
            assert_eq!(
                target["matching_point_count"].as_u64(),
                Some(5),
                "{style:?}: {:?}",
                target["points"]
            );
            assert_eq!(target["mismatching_point_count"].as_u64(), Some(0));
        }
    }

    #[test]
    fn valence_target_checks_major_minor_mapping_when_not_preset_pinned() {
        let style = Style::Classical;
        let curve = curve_json(style, "valence", BASE_SEED);
        let target = &curve["control_target_evaluation"]["target_fidelity"];
        if style.spec().mode.is_some() {
            assert_eq!(target["status"].as_str(), Some("preset_mode_override"));
        } else {
            assert_eq!(target["status"].as_str(), Some("measured"));
            assert_eq!(target["tested_point_count"].as_u64(), Some(5));
            assert_eq!(target["scored_point_count"].as_u64(), Some(4));
            assert_eq!(target["unscored_neutral_point_count"].as_u64(), Some(1));
            assert_eq!(
                target["matching_point_count"].as_u64(),
                Some(4),
                "{:?}",
                target["points"]
            );
            assert_eq!(target["mismatching_point_count"].as_u64(), Some(0));
            let points = target["points"].as_array().unwrap();
            assert!(points.iter().any(|point| {
                point["control_value"] == "-1.00"
                    && point["expected_tonality"] == "Minor"
                    && point["matches_target"] == true
            }));
            assert!(points.iter().any(|point| {
                point["control_value"] == "0.00"
                    && point["target_applicability"] == "neutral_unscored"
                    && point["expected_tonality"].is_null()
                    && point["matches_target"].is_null()
            }));
            assert!(points.iter().any(|point| {
                point["control_value"] == "1.00"
                    && point["expected_tonality"] == "Major"
                    && point["matches_target"] == true
            }));
        }
    }

    #[test]
    fn chorus_bar_targets_check_plan_quantization_and_observed_score_span() {
        for style in [Style::Blues, Style::JazzBallad] {
            let curve = curve_json(style, "bars", BASE_SEED);
            let target = &curve["control_target_evaluation"]["target_fidelity"];
            assert_eq!(target["status"].as_str(), Some("measured"), "{style:?}");
            assert_eq!(target["tested_point_count"].as_u64(), Some(5), "{style:?}");
            assert_eq!(
                target["matching_point_count"].as_u64(),
                Some(5),
                "{style:?}: {:?}",
                target["points"]
            );
            assert_eq!(target["mismatching_point_count"].as_u64(), Some(0), "{style:?}");
            assert!(target["points"].as_array().unwrap().iter().all(|point| {
                point["plan_request_matches_input"].as_bool() == Some(true)
                    && point["plan_quantization_matches_policy"].as_bool() == Some(true)
                    && point["score_span_matches_plan"].as_bool() == Some(true)
            }));
        }
    }

    #[test]
    fn valence_target_is_not_scored_when_style_owns_the_mode() {
        let style = Style::HindustaniInspired;
        let curve = curve_json(style, "valence", BASE_SEED);
        let target = &curve["control_target_evaluation"]["target_fidelity"];
        assert_ne!(target["status"].as_str(), Some("measured"));
        assert!(matches!(
            target["status"].as_str(),
            Some("preset_mode_override" | "engine_owned_default")
        ));
        let invariance = &curve["control_target_evaluation"]["non_target_invariance"];
        let mode = invariance["checked_channels"]
            .as_array()
            .unwrap()
            .iter()
            .find(|channel| channel["channel"] == "key_tonality")
            .expect("a preset/fixed modal route should keep tonality invariant");
        assert_eq!(mode["changed_point_count"].as_u64(), Some(0));
    }

    #[test]
    fn energy_phrase_model_target_tracks_the_composers_threshold_rule() {
        let curve = curve_json(Style::Classical, "energy", BASE_SEED);
        let target = &curve["control_target_evaluation"]["target_fidelity"];
        assert_eq!(target["status"].as_str(), Some("measured"));
        assert_eq!(target["tested_point_count"].as_u64(), Some(5));
        assert_eq!(
            target["matching_point_count"].as_u64(),
            Some(5),
            "{:?}",
            target["points"]
        );
        assert_eq!(target["mismatching_point_count"].as_u64(), Some(0));
        let points = target["points"].as_array().unwrap();
        let sentence_count = points
            .iter()
            .filter(|point| point["expected_phrase_model"].as_str() == Some("sentence"))
            .count();
        assert_eq!(sentence_count, 1);
    }

    #[test]
    fn routed_curve_reports_dynamic_channels_and_observed_change_flags() {
        let curve = curve_json(Style::Classical, "arousal", BASE_SEED);
        assert_eq!(curve["point_count"].as_u64(), Some(5));
        assert_eq!(
            curve["observed_changes_from_baseline"]
                ["no_observed_change_at_tested_points"]
                .as_bool(),
            Some(false),
        );
        let points = curve["points"].as_array().expect("response points");
        assert!(points.iter().all(|point| {
            let exact = point["exact_fingerprint_hex"].as_str().unwrap_or_default();
            let baseline = point["baseline_exact_fingerprint_hex"].as_str().unwrap_or_default();
            exact.len() == 16
                && baseline.len() == 16
                && exact.bytes().all(|byte| byte.is_ascii_hexdigit())
                && baseline.bytes().all(|byte| byte.is_ascii_hexdigit())
                && point["descriptors"].get("mean_velocity").is_some()
                && point["descriptors"].get("velocity_standard_deviation").is_some()
                && point["descriptors"].get("mean_section_intensity").is_some()
                && point["descriptors"].get("score_key_tonic_pc").is_some()
                && point["layer_distances"].as_array().is_some()
        }));
        assert!(points.iter().any(|point| {
            point["input"]["arousal"].as_f64() == Some(0.5)
                && point["overall_structural_distance_from_baseline"].as_f64() == Some(0.0)
        }));
    }
}
