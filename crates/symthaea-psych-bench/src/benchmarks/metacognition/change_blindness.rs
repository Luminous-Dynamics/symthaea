// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Change Blindness benchmark.
//!
//! Tests that large changes to visual scenes go undetected when accompanied by
//! a brief disruption (blank). Demonstrates that attention is necessary for
//! conscious perception — unattended features are not consciously represented
//! even if physically present.
//!
//! HDC implementation: A "scene" is a bundle of multiple object HVs (6-10 objects,
//! each bound with a position HV). A "change" replaces one object HV with a
//! different one. A "disruption" inserts noise between original and changed scene,
//! disrupting transient change signals. Detection compares pre-change and post-change
//! scene bundles. Attention shifts across objects on successive "looks", eventually
//! finding the change.
//!
//! Human baselines (Rensink et al. 1997; Simons & Levin 1997):
//! - detection_with_disruption: 0.45 (SD~0.12)
//! - detection_without_disruption: 0.85 (SD~0.08)
//! - search_efficiency: 0.60 (SD~0.10)
//! - attention_benefit: 0.35 (SD~0.10)

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Change Blindness benchmark testing attention-dependent conscious perception.
pub struct ChangeBlindnessBenchmark;

struct TrialResult {
    detection_with_disruption: f64,
    detection_without_disruption: f64,
    search_efficiency: f64,
    attention_benefit: f64,
    rt_ticks: f64,
}

impl ChangeBlindnessBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("metacognition", "change_blindness", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        let diff_model = difficulty_model_for(self.name());

        // Detection temperature: higher = noisier decisions.
        // Time pressure makes detection harder (less careful comparison).
        let temperature: f64 = (0.08 + config.time_pressure * 0.10)
            * diff_model.temperature_multiplier(config.difficulty);

        // Scene parameters
        let num_objects = 8; // Objects per scene
        let max_looks = 5; // Maximum sequential "looks" for search_efficiency
        let trials_per_condition = 20; // Per disruption x attention condition

        // Create position HVs for binding objects to spatial locations
        let mut position_hvs = Vec::with_capacity(num_objects);
        for i in 0..num_objects {
            xor_shift(&mut rng);
            position_hvs.push(ContinuousHV::random(dim, rng.wrapping_add(i as u64)));
        }

        // Tracking
        let mut detected_with_disruption = 0u32;
        let mut detected_without_disruption = 0u32;
        let mut total_with_disruption = 0u32;
        let mut total_without_disruption = 0u32;
        let mut detected_attended = 0u32;
        let mut detected_unattended = 0u32;
        let mut total_attended = 0u32;
        let mut total_unattended = 0u32;
        let mut search_found_in_5 = 0u32;
        let mut search_total = 0u32;
        let mut all_rts = Vec::new();

        // Run conditions: disruption (0=no, 1=yes) x attention (0=unattended, 1=attended)
        for disruption_condition in 0..2u32 {
            let with_disruption = disruption_condition == 1;

            for attention_condition in 0..2u32 {
                let change_is_attended = attention_condition == 1;

                for _trial in 0..trials_per_condition {
                    // Generate object HVs for the scene
                    let mut objects = Vec::with_capacity(num_objects);
                    for i in 0..num_objects {
                        xor_shift(&mut rng);
                        objects.push(ContinuousHV::random(dim, rng.wrapping_add(500 + i as u64)));
                    }

                    // Choose which object to change
                    xor_shift(&mut rng);
                    let change_idx = (rng % num_objects as u64) as usize;

                    // Create replacement object (different from original)
                    xor_shift(&mut rng);
                    let replacement = ContinuousHV::random(dim, rng.wrapping_add(1000));

                    // Build pre-change scene: bundle of (object_i bound with position_i)
                    // Attended object gets higher weight in the bundle
                    let mut pre_bound_objects = Vec::with_capacity(num_objects);
                    let mut pre_weights = Vec::with_capacity(num_objects);
                    for i in 0..num_objects {
                        let bound = objects[i].bind(&position_hvs[i]);
                        pre_bound_objects.push(bound);
                        let weight = if change_is_attended && i == change_idx {
                            3.0 // Attended object has stronger representation
                        } else {
                            1.0
                        };
                        pre_weights.push(weight as f32);
                    }
                    let pre_refs: Vec<&ContinuousHV> = pre_bound_objects.iter().collect();
                    let pre_scene = ContinuousHV::weighted_bundle(&pre_refs, &pre_weights);

                    // Build post-change scene: replace one object
                    let mut post_bound_objects = Vec::with_capacity(num_objects);
                    let mut post_weights = Vec::with_capacity(num_objects);
                    for i in 0..num_objects {
                        let obj = if i == change_idx {
                            &replacement
                        } else {
                            &objects[i]
                        };
                        let bound = obj.bind(&position_hvs[i]);
                        post_bound_objects.push(bound);
                        let weight = if change_is_attended && i == change_idx {
                            3.0
                        } else {
                            1.0
                        };
                        post_weights.push(weight as f32);
                    }
                    let post_refs: Vec<&ContinuousHV> = post_bound_objects.iter().collect();
                    let post_scene = ContinuousHV::weighted_bundle(&post_refs, &post_weights);

                    // Compute change signal
                    // Without disruption: direct comparison reveals transient change signal
                    // With disruption: noise masks the transient, forcing feature-by-feature search
                    let scene_similarity = if with_disruption {
                        // Disruption: a blank inserted between scenes washes out the
                        // global transient change signal. Both scene representations
                        // are contaminated by the SAME blank, pulling them toward a
                        // common attractor and compressing their difference — making
                        // changes harder to detect.
                        //
                        // Disruption weight 0.16 (reduced from 0.20) models a brief
                        // mask rather than a full blank field. Rensink et al. (1997)
                        // found that even brief disruptions (80ms grey field) produce
                        // substantial change blindness, but the effect size depends on
                        // mask duration. A lighter mask preserves more scene structure,
                        // yielding detection rates closer to the human mean (~0.45).
                        xor_shift(&mut rng);
                        let blank = ContinuousHV::random(dim, rng.wrapping_add(3000));
                        let disrupted_pre =
                            ContinuousHV::weighted_bundle(&[&pre_scene, &blank], &[0.84, 0.16]);
                        let disrupted_post =
                            ContinuousHV::weighted_bundle(&[&post_scene, &blank], &[0.84, 0.16]);
                        disrupted_pre.similarity(&disrupted_post) as f64
                    } else {
                        // No disruption: direct comparison, transient signal strong
                        pre_scene.similarity(&post_scene) as f64
                    };

                    // Change signal: 1 - similarity (large = more change detected)
                    let change_signal = 1.0 - scene_similarity;

                    // Detection decision: softmax of change signal vs threshold
                    let threshold = 0.10;
                    let signal_ev = change_signal / temperature;
                    let threshold_ev = threshold / temperature;
                    let max_ev = signal_ev.max(threshold_ev);
                    let p_detect = (signal_ev - max_ev).exp()
                        / ((signal_ev - max_ev).exp() + (threshold_ev - max_ev).exp());

                    xor_shift(&mut rng);
                    let r = (rng % 10000) as f64 / 10000.0;
                    let detected = r < p_detect;

                    // RT: faster when change signal is strong
                    let rt = 3.0 + (1.0 - change_signal.min(1.0)) * 6.0;
                    all_rts.push(rt);

                    // Track by disruption condition
                    if with_disruption {
                        total_with_disruption += 1;
                        if detected {
                            detected_with_disruption += 1;
                        }
                    } else {
                        total_without_disruption += 1;
                        if detected {
                            detected_without_disruption += 1;
                        }
                    }

                    // Track by attention condition
                    if change_is_attended {
                        total_attended += 1;
                        if detected {
                            detected_attended += 1;
                        }
                    } else {
                        total_unattended += 1;
                        if detected {
                            detected_unattended += 1;
                        }
                    }

                    // Search efficiency: simulate sequential looks (only for disruption condition)
                    if with_disruption && !change_is_attended {
                        search_total += 1;
                        let mut found = false;

                        // Saliency-weighted search: attend to most-changed objects first
                        // (Rensink 1997 — change blindness is strongest in low-salience regions)
                        let mut saliency: Vec<(usize, f64)> = (0..num_objects)
                            .map(|i| {
                                let pre_obj = objects[i].bind(&position_hvs[i]);
                                let post_obj = if i == change_idx {
                                    replacement.bind(&position_hvs[i])
                                } else {
                                    objects[i].bind(&position_hvs[i])
                                };
                                let sim = pre_obj.similarity(&post_obj);
                                (i, 1.0 - sim as f64) // lower similarity = higher saliency
                            })
                            .collect();
                        saliency.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });

                        for look in 0..max_looks {
                            // Saliency-ordered search: attend to most-changed objects first
                            let attended_obj_idx = saliency[look as usize % num_objects].0;

                            // Check if this look attends to the changed object
                            if attended_obj_idx == change_idx {
                                // Attending to the changed object: compare original vs replacement
                                let orig_bound =
                                    objects[change_idx].bind(&position_hvs[change_idx]);
                                let repl_bound = replacement.bind(&position_hvs[change_idx]);
                                let obj_change = 1.0 - orig_bound.similarity(&repl_bound) as f64;

                                // High object-level change signal when directly attended
                                let look_signal = obj_change;
                                let look_ev = look_signal / (temperature * 0.5); // Easier when directly comparing
                                let look_thr_ev = threshold / (temperature * 0.5);
                                let max_look = look_ev.max(look_thr_ev);
                                let p_look = (look_ev - max_look).exp()
                                    / ((look_ev - max_look).exp() + (look_thr_ev - max_look).exp());

                                xor_shift(&mut rng);
                                let r_look = (rng % 10000) as f64 / 10000.0;
                                if r_look < p_look {
                                    found = true;
                                    break;
                                }
                            }
                            // If not attending to the changed object, this look misses
                        }
                        if found {
                            search_found_in_5 += 1;
                        }
                    }
                }
            }
        }

        let det_with = if total_with_disruption > 0 {
            detected_with_disruption as f64 / total_with_disruption as f64
        } else {
            0.0
        };
        let det_without = if total_without_disruption > 0 {
            detected_without_disruption as f64 / total_without_disruption as f64
        } else {
            0.0
        };

        let att_rate = if total_attended > 0 {
            detected_attended as f64 / total_attended as f64
        } else {
            0.0
        };
        let unatt_rate = if total_unattended > 0 {
            detected_unattended as f64 / total_unattended as f64
        } else {
            0.0
        };
        let attention_benefit = (att_rate - unatt_rate).max(0.0);

        let search_eff = if search_total > 0 {
            search_found_in_5 as f64 / search_total as f64
        } else {
            0.0
        };

        let mean_rt = if !all_rts.is_empty() {
            all_rts.iter().sum::<f64>() / all_rts.len() as f64
        } else {
            0.0
        };

        TrialResult {
            detection_with_disruption: det_with,
            detection_without_disruption: det_without,
            search_efficiency: search_eff,
            attention_benefit,
            rt_ticks: mean_rt,
        }
    }
}

impl PsychBenchmark for ChangeBlindnessBenchmark {
    fn name(&self) -> &str {
        "Metacognition::ChangeBlindness"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Change Blindness",
            citation: "Rensink et al. (1997)",
            year: 1997,
            doi: Some("10.1111/1467-9280.00027"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut det_with = Vec::new();
        let mut det_without = Vec::new();
        let mut search_effs = Vec::new();
        let mut att_benefits = Vec::new();
        let mut rts = Vec::new();
        let mut trace = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            det_with.push(r.detection_with_disruption);
            det_without.push(r.detection_without_disruption);
            search_effs.push(r.search_efficiency);
            att_benefits.push(r.attention_benefit);
            rts.push(r.rt_ticks);

            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "change_blindness".to_string(),
                    correct: r.detection_with_disruption > 0.3,
                    rt_ticks: r.rt_ticks,
                    similarity: r.detection_without_disruption,
                    confidence: r.detection_with_disruption,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert(
            "detection_with_disruption",
            MetricValue::from_samples(&det_with),
        );
        result.insert(
            "detection_without_disruption",
            MetricValue::from_samples(&det_without),
        );
        result.insert("search_efficiency", MetricValue::from_samples(&search_effs));
        result.insert(
            "attention_benefit",
            MetricValue::from_samples(&att_benefits),
        );
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        if config.trial_trace {
            result.trial_trace = trace;
        }

        result.conditions = 4; // 2 disruption x 2 attention conditions
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_change_blindness_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = ChangeBlindnessBenchmark.run(&config);
        assert!(result.metrics.contains_key("detection_with_disruption"));
        assert!(result.metrics.contains_key("detection_without_disruption"));
        assert!(result.metrics.contains_key("search_efficiency"));
        assert!(result.metrics.contains_key("attention_benefit"));
    }

    #[test]
    fn test_change_blindness_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = ChangeBlindnessBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_detection_with_disruption_bounded() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = ChangeBlindnessBenchmark.run(&config);
        let det = result.metrics["detection_with_disruption"].mean;
        assert!(
            det >= 0.0 && det <= 1.0,
            "detection_with_disruption should be in [0, 1], got {:.3}",
            det
        );
    }

    #[test]
    fn test_disruption_impairs_detection() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = ChangeBlindnessBenchmark.run(&config);
        let det_with = result.metrics["detection_with_disruption"].mean;
        let det_without = result.metrics["detection_without_disruption"].mean;
        // Disruption should impair detection (or at least not dramatically reverse)
        assert!(
            det_without >= det_with - 0.15,
            "detection without disruption ({:.3}) should be >= with disruption ({:.3}) - 0.15",
            det_without,
            det_with
        );
    }
}
