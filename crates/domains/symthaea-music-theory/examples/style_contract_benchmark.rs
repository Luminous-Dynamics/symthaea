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
    STRUCT_DIMS, dist, exact_fingerprint, structural_fingerprint,
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
    let mut exact_score_changed_pairs = 0usize;
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
                exact_score_changed_pairs += 1;
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
        "exact_score_changed": exact_score_changed_pairs,
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
        .map(|audit| audit["exact_score_changed"].as_u64().unwrap_or(0))
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
    let report = json!({
        "schema_version": 2,
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
            "all_mechanical_checks_passed": spec_invalid_count == 0
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
            "exact_score_changed": transposition_exact_changed_pairs,
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
        assert_eq!(SEEDS.len() * INTENTS.len(), 16);
        let unique: HashSet<String> =
            Style::ALL.iter().map(|style| format!("{style:?}")).collect();
        assert_eq!(unique.len(), Style::ALL.len());
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
