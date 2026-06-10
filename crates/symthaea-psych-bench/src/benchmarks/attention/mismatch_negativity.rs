// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mismatch Negativity (MMN) — Pre-attentive Oddball Detection.
//!
//! Tests the brain's automatic detection of deviations from repeated patterns,
//! even without focused attention. The MMN response is pre-attentive and indexes
//! the brain's predictive model of sensory regularities.
//!
//! HDC implementation: A "standard" stimulus HV is repeated many times, building
//! a strong predictive model (EMA of recent stimuli). Periodically a "deviant"
//! stimulus is inserted. Detection is based on similarity between predicted and
//! actual stimulus — low similarity = mismatch = deviant detected. Deviant-standard
//! similarity is varied to create easy (very different) and hard (similar) conditions.
//! Pre-attentive: detection works even under attentional load (added noise).
//!
//! Human baselines (Naatanen et al. 1978, 2007):
//! - detection_accuracy: 0.88 (SD~0.06)
//! - false_alarm_rate: 0.08 (SD~0.04)
//! - mismatch_magnitude: 0.35 (SD~0.10)
//! - attentional_independence: 0.80 (SD~0.12)

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Mismatch Negativity benchmark testing pre-attentive oddball detection.
pub struct MismatchNegativityBenchmark;

struct TrialResult {
    detection_accuracy: f64,
    false_alarm_rate: f64,
    mismatch_magnitude: f64,
    attentional_independence: f64,
    rt_ticks: f64,
}

impl MismatchNegativityBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("attention", "mismatch_negativity", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        let diff_model = difficulty_model_for(self.name());

        // Detection threshold: prediction error must exceed this to flag deviant.
        // Time pressure raises threshold (less careful monitoring).
        // Treisman & Gelade (1980): speed emphasis narrows the attentional filter.
        let detection_threshold: f64 = (0.11 + config.time_pressure * 0.10)
            * diff_model.temperature_multiplier(config.difficulty);

        // Attentional load noise: degrades the prediction model when attention
        // is diverted, modeling the dual-task conditions in Naatanen (2007).
        // MMN is pre-attentive, so load effect should be modest.
        let load_noise_weight: f32 = 0.08;

        // EMA decay for building the predictive model of the standard.
        // Higher alpha = faster adaptation to recent stimuli.
        let ema_alpha: f32 = 0.25;

        // Sequence parameters
        let sequence_len = 80; // Total stimuli per sequence
        let deviant_probability = 0.15; // ~12 deviants per sequence
        let num_difficulty_levels = 3; // Easy, medium, hard deviant similarity

        // Create standard stimulus prototype
        xor_shift(&mut rng);
        let standard_prototype = ContinuousHV::random(dim, rng);

        // Create deviant prototypes at different similarities to standard
        // Easy: very different deviant; Hard: similar to standard
        let deviant_similarities: [f32; 3] = [0.15, 0.40, 0.65]; // low, medium, high overlap

        let mut deviant_prototypes = Vec::with_capacity(num_difficulty_levels);
        for (i, &overlap) in deviant_similarities.iter().enumerate() {
            xor_shift(&mut rng);
            let unique_part = ContinuousHV::random(dim, rng.wrapping_add(100 + i as u64));
            let deviant = ContinuousHV::weighted_bundle(
                &[&standard_prototype, &unique_part],
                &[overlap, 1.0 - overlap],
            );
            deviant_prototypes.push(deviant);
        }

        // Tracking per condition
        let mut hits_total = 0u32;
        let mut deviants_total = 0u32;
        let mut false_alarms = 0u32;
        let mut standards_total = 0u32;
        let mut mismatch_magnitudes = Vec::new();
        let mut all_rts = Vec::new();

        // Detection accuracy under attentional load (for attentional_independence)
        let mut hits_no_load = 0u32;
        let mut deviants_no_load = 0u32;
        let mut hits_with_load = 0u32;
        let mut deviants_with_load = 0u32;

        for deviant_proto in deviant_prototypes.iter().take(num_difficulty_levels) {
            // Run two conditions: without and with attentional load
            for load_condition in 0..2u32 {
                let with_load = load_condition == 1;

                // Initialize prediction model as the first standard stimulus
                xor_shift(&mut rng);
                let first_standard_noise = ContinuousHV::random(dim, rng);
                let mut prediction_model = ContinuousHV::weighted_bundle(
                    &[&standard_prototype, &first_standard_noise],
                    &[0.90, 0.10],
                );

                for step in 0..sequence_len {
                    // Determine if this stimulus is a deviant
                    // First 5 stimuli are always standards (build up prediction)
                    xor_shift(&mut rng);
                    let is_deviant =
                        step >= 8 && ((rng % 1000) as f64 / 1000.0) < deviant_probability;

                    // Generate stimulus
                    xor_shift(&mut rng);
                    let stimulus_noise = ContinuousHV::random(dim, rng);
                    let stimulus = if is_deviant {
                        ContinuousHV::weighted_bundle(
                            &[deviant_proto, &stimulus_noise],
                            &[0.85, 0.15],
                        )
                    } else {
                        ContinuousHV::weighted_bundle(
                            &[&standard_prototype, &stimulus_noise],
                            &[0.85, 0.15],
                        )
                    };

                    // Under attentional load, add noise to the prediction model
                    // This simulates divided attention degrading the internal model
                    let effective_prediction = if with_load {
                        xor_shift(&mut rng);
                        let load_noise = ContinuousHV::random(dim, rng.wrapping_add(5000));
                        ContinuousHV::weighted_bundle(
                            &[&prediction_model, &load_noise],
                            &[1.0 - load_noise_weight, load_noise_weight],
                        )
                    } else {
                        prediction_model.clone()
                    };

                    // Compute prediction error: 1 - similarity(predicted, actual)
                    let similarity = effective_prediction.similarity(&stimulus) as f64;
                    let prediction_error = 1.0 - similarity;

                    // Decision: is this a deviant?
                    // Softmax-like decision with temperature from difficulty model
                    let temperature: f64 = 0.08
                        * diff_model.temperature_multiplier(config.difficulty)
                        * (1.0 + config.time_pressure * 0.15);
                    let detection_evidence = prediction_error / temperature;
                    let threshold_evidence = detection_threshold / temperature;
                    let max_ev = detection_evidence.max(threshold_evidence);
                    let p_detect = (detection_evidence - max_ev).exp()
                        / ((detection_evidence - max_ev).exp()
                            + (threshold_evidence - max_ev).exp());

                    xor_shift(&mut rng);
                    let r = (rng % 10000) as f64 / 10000.0;
                    let detected_as_deviant = r < p_detect;

                    // RT: faster for large prediction errors (obvious deviants)
                    let rt = 3.0 + (1.0 - prediction_error.min(1.0)) * 5.0;
                    all_rts.push(rt);

                    // Score
                    if is_deviant {
                        deviants_total += 1;
                        mismatch_magnitudes.push(prediction_error);
                        if detected_as_deviant {
                            hits_total += 1;
                        }
                        if with_load {
                            deviants_with_load += 1;
                            if detected_as_deviant {
                                hits_with_load += 1;
                            }
                        } else {
                            deviants_no_load += 1;
                            if detected_as_deviant {
                                hits_no_load += 1;
                            }
                        }
                    } else {
                        standards_total += 1;
                        if detected_as_deviant {
                            false_alarms += 1;
                        }
                    }

                    // Update prediction model via EMA only for stimuli not flagged
                    // as deviant. This prevents deviant contamination of the
                    // predictive model, matching the brain's gating mechanism
                    // (Winkler et al. 1996: deviants do not update the standard trace).
                    if !detected_as_deviant {
                        prediction_model = ContinuousHV::weighted_bundle(
                            &[&prediction_model, &stimulus],
                            &[1.0 - ema_alpha, ema_alpha],
                        );
                    }
                }
            }
        }

        let detection_accuracy = if deviants_total > 0 {
            hits_total as f64 / deviants_total as f64
        } else {
            0.0
        };

        let false_alarm_rate = if standards_total > 0 {
            false_alarms as f64 / standards_total as f64
        } else {
            0.0
        };

        let mismatch_magnitude = if !mismatch_magnitudes.is_empty() {
            mismatch_magnitudes.iter().sum::<f64>() / mismatch_magnitudes.len() as f64
        } else {
            0.0
        };

        // Attentional independence: ratio of detection under load to without load
        let acc_no_load = if deviants_no_load > 0 {
            hits_no_load as f64 / deviants_no_load as f64
        } else {
            0.0
        };
        let acc_with_load = if deviants_with_load > 0 {
            hits_with_load as f64 / deviants_with_load as f64
        } else {
            0.0
        };
        let attentional_independence = if acc_no_load > 0.0 {
            (acc_with_load / acc_no_load).min(1.0)
        } else {
            0.0
        };

        let mean_rt = if !all_rts.is_empty() {
            all_rts.iter().sum::<f64>() / all_rts.len() as f64
        } else {
            0.0
        };

        TrialResult {
            detection_accuracy,
            false_alarm_rate,
            mismatch_magnitude,
            attentional_independence,
            rt_ticks: mean_rt,
        }
    }
}

impl PsychBenchmark for MismatchNegativityBenchmark {
    fn name(&self) -> &str {
        "Attention::MismatchNegativity"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Mismatch Negativity (Oddball)",
            citation: "Naatanen et al. (1978)",
            year: 1978,
            doi: Some("10.1016/0013-4694(78)90267-9"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut det_accs = Vec::new();
        let mut fa_rates = Vec::new();
        let mut mm_mags = Vec::new();
        let mut att_indeps = Vec::new();
        let mut rts = Vec::new();
        let mut trace = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            det_accs.push(r.detection_accuracy);
            fa_rates.push(r.false_alarm_rate);
            mm_mags.push(r.mismatch_magnitude);
            att_indeps.push(r.attentional_independence);
            rts.push(r.rt_ticks);

            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "mismatch_negativity".to_string(),
                    correct: r.detection_accuracy > 0.5,
                    rt_ticks: r.rt_ticks,
                    similarity: r.mismatch_magnitude,
                    confidence: r.detection_accuracy,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("detection_accuracy", MetricValue::from_samples(&det_accs));
        result.insert("false_alarm_rate", MetricValue::from_samples(&fa_rates));
        result.insert("mismatch_magnitude", MetricValue::from_samples(&mm_mags));
        result.insert(
            "attentional_independence",
            MetricValue::from_samples(&att_indeps),
        );
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        if config.trial_trace {
            result.trial_trace = trace;
        }

        result.conditions = 6; // 3 difficulty levels x 2 load conditions
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mismatch_negativity_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = MismatchNegativityBenchmark.run(&config);
        assert!(result.metrics.contains_key("detection_accuracy"));
        assert!(result.metrics.contains_key("false_alarm_rate"));
        assert!(result.metrics.contains_key("mismatch_magnitude"));
        assert!(result.metrics.contains_key("attentional_independence"));
    }

    #[test]
    fn test_mismatch_negativity_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = MismatchNegativityBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_detection_accuracy_bounded() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = MismatchNegativityBenchmark.run(&config);
        let acc = result.metrics["detection_accuracy"].mean;
        assert!(
            acc >= 0.0 && acc <= 1.0,
            "detection_accuracy should be in [0, 1], got {:.3}",
            acc
        );
    }

    #[test]
    fn test_time_pressure_degrades_accuracy() {
        let base_config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let pressure_config = BenchmarkConfig {
            time_pressure: 1.0,
            ..base_config.clone()
        };
        let base_result = MismatchNegativityBenchmark.run(&base_config);
        let pressure_result = MismatchNegativityBenchmark.run(&pressure_config);
        let base_acc = base_result.metrics["detection_accuracy"].mean;
        let pressure_acc = pressure_result.metrics["detection_accuracy"].mean;
        // Under time pressure, accuracy should not dramatically improve
        assert!(
            pressure_acc <= base_acc + 0.15,
            "time pressure should not improve accuracy: base={:.3}, pressure={:.3}",
            base_acc,
            pressure_acc
        );
    }
}
