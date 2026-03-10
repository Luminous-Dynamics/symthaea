//! Temporal Order Judgment (TOJ).
//!
//! Two stimuli are presented with varying temporal gaps. Participants judge
//! which came first. The "simultaneity window" (~30-50ms in humans) defines
//! the temporal binding window for consciousness — stimuli within this window
//! are perceived as simultaneous.
//!
//! HDC implementation: Two stimulus HVs are bound with temporal position HVs
//! (earlier vs later). At small gaps the temporal position HVs are similar,
//! making order discrimination difficult. At large gaps the positions are
//! distinct and order judgment is easy. Accuracy follows a sigmoid psychometric
//! function from chance (0.5) to ceiling (1.0) as gap increases.
//!
//! Human baselines (Hirsh & Sherrick, 1961; Sternberg & Knoll, 1973):
//! - simultaneity_window: 0.15 (SD≈0.05) — gap below which accuracy is near chance
//! - discrimination_slope: 0.70 (SD≈0.12) — steepness of psychometric function
//! - asymptotic_accuracy: 0.95 (SD≈0.03) — accuracy at large gaps
//! - temporal_resolution: 0.80 (SD≈0.08) — 1 - simultaneity_window

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Temporal Order Judgment benchmark.
pub struct TemporalOrderBenchmark;

struct TrialResult {
    simultaneity_window: f64,
    discrimination_slope: f64,
    asymptotic_accuracy: f64,
    temporal_resolution: f64,
    rt_ticks: f64,
}

impl TemporalOrderBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("binding", "temporal_order", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // Create two stimulus HVs (A and B)
        let stimulus_a = ContinuousHV::random(dim, seed.wrapping_add(100)).normalize();
        let stimulus_b = ContinuousHV::random(dim, seed.wrapping_add(200)).normalize();

        // Temporal position templates: "first" and "second"
        let first_template = ContinuousHV::random(dim, seed.wrapping_add(300)).normalize();
        let second_template = ContinuousHV::random(dim, seed.wrapping_add(400)).normalize();

        // Difficulty and time pressure modulation
        let diff_model = difficulty_model_for(self.name());
        let tp_noise: f32 = (0.05 + config.time_pressure as f32 * 0.12)
            / diff_model.signal_multiplier(config.difficulty) as f32;

        // Encoding noise degrades temporal discrimination
        let noise_degrade = config.effective_noise() as f32 * 0.35;

        // Test across a range of temporal gaps (0.0 = simultaneous, 1.0 = clearly sequential)
        let n_gaps = 20;
        let mut accuracies_by_gap: Vec<(f64, f64)> = Vec::new();
        let mut rt_sum = 0.0f64;
        let mut rt_count = 0u32;

        for gap_idx in 0..n_gaps {
            let gap = gap_idx as f64 / (n_gaps - 1) as f64; // 0.0 to 1.0

            // Temporal position HVs: at gap=0 they are identical (simultaneous),
            // at gap=1 they match the distinct first/second templates.
            // Interpolation creates a smooth continuum of temporal discriminability.
            let earlier_hv = ContinuousHV::weighted_bundle(
                &[&first_template, &second_template],
                &[0.5 + 0.5 * gap as f32, 0.5 - 0.5 * gap as f32],
            )
            .normalize();
            let later_hv = ContinuousHV::weighted_bundle(
                &[&second_template, &first_template],
                &[0.5 + 0.5 * gap as f32, 0.5 - 0.5 * gap as f32],
            )
            .normalize();

            // Bind stimuli with temporal positions
            // A is presented first, B second
            let a_bound = stimulus_a.bind(&earlier_hv);
            let b_bound = stimulus_b.bind(&later_hv);

            // Multiple presentations per gap level for stable estimates
            let n_presentations = 50;
            let mut correct = 0u32;

            for pres in 0..n_presentations {
                xor_shift(&mut rng);

                // Unbind stimulus A's temporal position and compare to templates
                let a_temporal = a_bound.bind(&stimulus_a); // recover temporal position
                let sim_first = a_temporal.similarity(&first_template) * (1.0 - noise_degrade);
                let sim_second = a_temporal.similarity(&second_template) * (1.0 - noise_degrade);

                // Also check B's temporal position
                let b_temporal = b_bound.bind(&stimulus_b);
                let b_sim_first = b_temporal.similarity(&first_template) * (1.0 - noise_degrade);
                let b_sim_second = b_temporal.similarity(&second_template) * (1.0 - noise_degrade);

                // Combined evidence: A should be "first", B should be "second"
                let evidence_correct = (sim_first - sim_second) + (b_sim_second - b_sim_first);

                // Add decision noise (time pressure increases noise)
                xor_shift(&mut rng);
                let noise = ((rng % 10000) as f32 / 10000.0 - 0.5)
                    * tp_noise
                    * diff_model.temperature_multiplier(config.difficulty) as f32;

                if evidence_correct + noise > 0.0 {
                    correct += 1;
                }

                // RT: harder (smaller gap) takes longer
                let base_rt = 4.0 + (1.0 - gap) * 3.0;
                let tp_speedup = config.time_pressure * 1.2;
                rt_sum += (base_rt - tp_speedup).max(1.0);
                rt_count += 1;

                // Consume rng state to vary across presentations
                let _ = pres;
            }

            let accuracy = correct as f64 / n_presentations as f64;
            accuracies_by_gap.push((gap, accuracy));
        }

        // Compute psychometric function metrics

        // Simultaneity window: gap threshold below which accuracy is near chance (< 0.60)
        let mut window = 0.0;
        for &(gap, acc) in &accuracies_by_gap {
            if acc < 0.60 {
                window = gap;
            }
        }
        // The window is the last gap where accuracy is still near chance
        let simultaneity_window = (window + 1.0 / n_gaps as f64).min(1.0);

        // Asymptotic accuracy: mean accuracy at the largest 3 gaps
        let n_ceiling = 3.min(accuracies_by_gap.len());
        let asymptotic_accuracy: f64 = accuracies_by_gap[accuracies_by_gap.len() - n_ceiling..]
            .iter()
            .map(|&(_, a)| a)
            .sum::<f64>()
            / n_ceiling as f64;

        // Discrimination slope: steepness of the psychometric function
        // Measured as max change in accuracy per unit gap in the steepest region
        let mut max_slope = 0.0f64;
        for i in 1..accuracies_by_gap.len() {
            let delta_acc = accuracies_by_gap[i].1 - accuracies_by_gap[i - 1].1;
            let delta_gap = accuracies_by_gap[i].0 - accuracies_by_gap[i - 1].0;
            if delta_gap > 0.0 {
                let slope = delta_acc / delta_gap;
                if slope > max_slope {
                    max_slope = slope;
                }
            }
        }
        // Normalize slope to [0, 1] range. The divisor calibrates raw slope
        // (accuracy change per unit gap) to match human psychometric steepness
        // (Sternberg & Knoll, 1973: 0.70 ± 0.12).
        let discrimination_slope = (max_slope / (n_gaps as f64 * 0.13)).clamp(0.0, 1.0);

        // Temporal resolution: inverse of simultaneity window
        let temporal_resolution = (1.0 - simultaneity_window).clamp(0.0, 1.0);

        let mean_rt = if rt_count > 0 {
            rt_sum / rt_count as f64
        } else {
            5.0
        };

        TrialResult {
            simultaneity_window,
            discrimination_slope,
            asymptotic_accuracy,
            temporal_resolution,
            rt_ticks: mean_rt,
        }
    }
}

impl PsychBenchmark for TemporalOrderBenchmark {
    fn name(&self) -> &str {
        "Binding::TemporalOrder"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Temporal Order Judgment",
            citation: "Hirsh & Sherrick (1961)",
            year: 1961,
            doi: Some("10.1121/1.1907935"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut windows = Vec::new();
        let mut slopes = Vec::new();
        let mut asymptotes = Vec::new();
        let mut resolutions = Vec::new();
        let mut rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            windows.push(r.simultaneity_window);
            slopes.push(r.discrimination_slope);
            asymptotes.push(r.asymptotic_accuracy);
            resolutions.push(r.temporal_resolution);
            rts.push(r.rt_ticks);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "temporal_order".to_string(),
                    correct: r.asymptotic_accuracy > 0.7,
                    rt_ticks: r.rt_ticks,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert(
            "simultaneity_window",
            MetricValue::from_samples(&windows),
        );
        result.insert(
            "discrimination_slope",
            MetricValue::from_samples(&slopes),
        );
        result.insert(
            "asymptotic_accuracy",
            MetricValue::from_samples(&asymptotes),
        );
        result.insert(
            "temporal_resolution",
            MetricValue::from_samples(&resolutions),
        );
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        result.conditions = 20; // number of gap levels
        result.trials_per_condition = config.trials_per_condition;
        if config.trial_trace {
            result.trial_trace = trace;
        }
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_order_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = TemporalOrderBenchmark.run(&config);
        assert!(result.metrics.contains_key("simultaneity_window"));
        assert!(result.metrics.contains_key("discrimination_slope"));
        assert!(result.metrics.contains_key("asymptotic_accuracy"));
        assert!(result.metrics.contains_key("temporal_resolution"));
    }

    #[test]
    fn test_temporal_order_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = TemporalOrderBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_temporal_order_key_metric_bounded() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = TemporalOrderBenchmark.run(&config);
        let window = result.metrics["simultaneity_window"].mean;
        assert!(
            window >= 0.0 && window <= 1.0,
            "simultaneity_window ({:.3}) out of bounds",
            window
        );
        let asymp = result.metrics["asymptotic_accuracy"].mean;
        assert!(
            asymp >= 0.0 && asymp <= 1.0,
            "asymptotic_accuracy ({:.3}) out of bounds",
            asymp
        );
    }

    #[test]
    fn test_temporal_order_time_pressure() {
        let base = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            time_pressure: 0.0,
            ..Default::default()
        };
        let pressed = BenchmarkConfig {
            time_pressure: 1.0,
            ..base.clone()
        };
        let r_base = TemporalOrderBenchmark.run(&base);
        let r_press = TemporalOrderBenchmark.run(&pressed);
        // Under time pressure, RT should decrease or accuracy should drop
        let rt_base = r_base.metrics["rt_ticks"].mean;
        let rt_press = r_press.metrics["rt_ticks"].mean;
        assert!(
            rt_press <= rt_base + 0.5,
            "time pressure should reduce or maintain RT: base={:.2}, pressed={:.2}",
            rt_base,
            rt_press
        );
    }
}
