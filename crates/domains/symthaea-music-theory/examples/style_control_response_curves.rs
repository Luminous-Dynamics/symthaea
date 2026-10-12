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
    mean_velocity: f64,
    velocity_standard_deviation: f64,
    mean_note_duration_beats: f64,
    mean_section_intensity: f64,
    section_intensity_standard_deviation: f64,
    emphasis_counts: [usize; 4],
    form_section_count: Option<usize>,
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
    let score = realization.score;
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
        mean_velocity,
        velocity_standard_deviation,
        mean_note_duration_beats,
        mean_section_intensity,
        section_intensity_standard_deviation,
        emphasis_counts,
        form_section_count: realization.form.as_ref().map(|form| form.sections.len()),
    }
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

    let baseline_observation = observations
        .iter()
        .find(|point| point.input == InputConfig::BASE)
        .expect("every control grid includes the unchanged baseline");
    let exact_events_changed = unique_fingerprints.len() > 1;
    let structural_distance_changed = max_structural_distance > MONOTONIC_EPSILON;
    let structural_layers_changed = !layers_changed.is_empty();
    let tempo_changed = observations
        .iter()
        .any(|point| {
            (point.tempo_bpm - baseline_observation.tempo_bpm).abs() > MONOTONIC_EPSILON
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

fn main() {
    let profiles: Vec<serde_json::Value> = Style::ALL
        .iter()
        .copied()
        .map(|style| {
            let mut curves = Vec::new();
            for base_seed in BASE_SEED_SET {
                for control in ["tonic", "valence", "arousal", "energy", "bars", "seed"] {
                    curves.push(curve_json(style, control, base_seed));
                }
            }

            serde_json::json!({
                "style": format!("{style:?}"),
                "grammar_family": format!("{:?}", style.grammar_family()),
                "dispatch_path": "compose_with_grammar_plan (same path used by compose_styled)",
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
    fn monotonicity_classifier_does_not_invent_a_direction() {
        assert_eq!(monotonicity(&[(0.0, 1.0), (1.0, 2.0)]), "nondecreasing");
        assert_eq!(monotonicity(&[(0.0, 2.0), (1.0, 1.0)]), "nonincreasing");
        assert_eq!(monotonicity(&[(0.0, 1.0), (1.0, 1.0)]), "flat");
        assert_eq!(monotonicity(&[(0.0, 1.0), (1.0, 2.0), (2.0, 1.0)]), "non_monotonic");
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
