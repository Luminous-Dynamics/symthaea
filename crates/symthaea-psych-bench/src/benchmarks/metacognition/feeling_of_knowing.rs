//! Feeling of Knowing (FOK) benchmark.
//!
//! Tests metamemory accuracy: after failed recall, can the system predict
//! whether it would recognize the correct answer? Good FOK = high gamma
//! correlation between FOK rating and subsequent recognition success.
//! Complements the Calibration benchmark (confidence during retrieval).
//!
//! Human baselines (Hart 1965; Metcalfe et al. 1993; Schwartz 1994):
//! - fok_gamma: 0.65 (SD≈0.10) — gamma(FOK, recognition)
//! - recognition_hit_rate: 0.75 (SD≈0.10)
//! - fok_resolution: 0.60 (SD≈0.12) — AUC of FOK predicting recognition
//! - recall_rate: 0.40 (SD≈0.12)

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;
use crate::wm::{WmConfig, WorkingMemory};
use symthaea_core::hdc::ContinuousHV;

/// Feeling of Knowing benchmark testing metamemory accuracy.
pub struct FeelingOfKnowingBenchmark;

impl FeelingOfKnowingBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> FokResult {
        let dim = config.dimension;
        let seed = config.trial_seed("metacognition", "fok", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        let num_items = 30;
        // Time pressure: higher recall threshold produces more recall failures (needed for FOK).
        // Wickelgren (1977): speed emphasis raises response criterion.
        let recall_threshold: f64 = 0.45 + config.time_pressure * 0.10;
        // Metacognitive noise: imperfect FOK introspection.
        // Lichtenstein et al. (1982): confidence calibration degrades under time pressure.
        let fok_noise_range: f64 = 0.30 + config.time_pressure * 0.15;

        let mut wm = WorkingMemory::new(WmConfig {
            dimension: dim,
            capacity: config.working_memory_capacity,
            ..Default::default()
        });

        // Study phase: generate cue-target pairs and store targets in WM
        let mut cues = Vec::with_capacity(num_items);
        let mut targets = Vec::with_capacity(num_items);
        for i in 0..num_items {
            xor_shift(&mut rng);
            let cue = ContinuousHV::random(dim, rng.wrapping_add(i as u64));
            xor_shift(&mut rng);
            let target = ContinuousHV::random(dim, rng.wrapping_add(1000 + i as u64));

            // Store bound cue-target pair in WM (with encoding noise)
            let pair = cue.bind(&target);
            xor_shift(&mut rng);
            let noise = ContinuousHV::random(dim, rng.wrapping_add(2000));
            let noise_weight = 0.10 + 0.03 * (i as f32 / num_items as f32);
            let noisy_pair =
                ContinuousHV::weighted_bundle(&[&pair, &noise], &[1.0 - noise_weight, noise_weight]);
            wm.perceive(noisy_pair);

            cues.push(cue);
            targets.push(target);
        }

        let mut recall_successes = 0u32;
        let mut recall_total = 0u32;
        let mut fok_ratings = Vec::new();
        let mut recognition_outcomes = Vec::new();
        let mut all_rts = Vec::new();

        let contents = wm.contents();

        for i in 0..num_items {
            recall_total += 1;

            // Recall attempt: unbind cue from WM items, find best match to target
            let mut best_recall_sim = f64::NEG_INFINITY;
            for wm_item in contents {
                // Unbind: cue ⊗ stored_pair ≈ target (bind is self-inverse for ContinuousHV)
                let unbound = cues[i].bind(wm_item);
                let sim = unbound.similarity(&targets[i]) as f64;
                if sim > best_recall_sim {
                    best_recall_sim = sim;
                }
            }

            // RT: recall deliberation time based on match quality
            let rt = 5.0 + (1.0 - best_recall_sim.min(1.0).max(0.0)) * 6.0;
            all_rts.push(rt);

            if best_recall_sim > recall_threshold {
                // Successful recall — no FOK needed
                recall_successes += 1;
                continue;
            }

            // Failed recall → FOK judgment
            // FOK based on partial activation (cue familiarity + residual match)
            // Hart (1965): FOK reflects accessibility of target information
            // even when full recall fails.
            let partial_activation = best_recall_sim.max(0.0);

            // Cue familiarity: how well does the cue match any WM contents?
            let cue_familiarity = if !contents.is_empty() {
                contents
                    .iter()
                    .map(|item| cues[i].similarity(item) as f64)
                    .fold(f64::NEG_INFINITY, f64::max)
                    .max(0.0)
            } else {
                0.0
            };

            // FOK = weighted combination of partial activation + cue familiarity
            // (Koriat, 1993 — accessibility heuristic model).
            let raw_fok = partial_activation * 0.60 + cue_familiarity * 0.40;

            // Logistic scaling to [0,1] range
            let scaled_fok = 1.0 / (1.0 + (-((raw_fok - 0.25) * 4.0)).exp());

            // Add metacognitive noise
            xor_shift(&mut rng);
            let noise = ((rng % 1000) as f64 / 1000.0 - 0.5) * fok_noise_range;
            let fok = (scaled_fok + noise).clamp(0.0, 1.0);

            // Recognition test: present target + 3 foils, select most familiar
            xor_shift(&mut rng);
            let mut options = vec![targets[i].clone()];
            for f in 0..3 {
                xor_shift(&mut rng);
                let foil = ContinuousHV::random(dim, rng.wrapping_add(5000 + f));
                options.push(foil);
            }

            // Recognition: compare each option to unbound WM contents
            let mut best_option_idx = 0;
            let mut best_option_sim = f64::NEG_INFINITY;
            for (oi, option) in options.iter().enumerate() {
                let mut max_sim = f64::NEG_INFINITY;
                for wm_item in contents {
                    let unbound = cues[i].bind(wm_item);
                    let sim = unbound.similarity(option) as f64;
                    if sim > max_sim {
                        max_sim = sim;
                    }
                }
                if max_sim > best_option_sim {
                    best_option_sim = max_sim;
                    best_option_idx = oi;
                }
            }

            let recognized = best_option_idx == 0; // target was first option
            fok_ratings.push(fok);
            recognition_outcomes.push(if recognized { 1.0 } else { 0.0 });
        }

        // Compute gamma (Goodman-Kruskal) between FOK ratings and recognition
        let gamma = compute_gamma(&fok_ratings, &recognition_outcomes);

        // Recognition hit rate (among FOK-judged items only)
        let recognition_hr = if !recognition_outcomes.is_empty() {
            recognition_outcomes.iter().sum::<f64>() / recognition_outcomes.len() as f64
        } else {
            0.0
        };

        // FOK resolution: area under ROC approximation
        // (higher FOK → higher P(recognition))
        let resolution = compute_fok_resolution(&fok_ratings, &recognition_outcomes);

        let recall_rate = if recall_total > 0 {
            recall_successes as f64 / recall_total as f64
        } else {
            0.0
        };

        let mean_rt = if !all_rts.is_empty() {
            all_rts.iter().sum::<f64>() / all_rts.len() as f64
        } else {
            0.0
        };

        FokResult {
            fok_gamma: gamma,
            recognition_hit_rate: recognition_hr,
            fok_resolution: resolution,
            recall_rate,
            rt_ticks: mean_rt,
        }
    }
}

/// Goodman-Kruskal gamma between two parallel vectors.
fn compute_gamma(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let mut concordant = 0u64;
    let mut discordant = 0u64;
    let max_pairs = 5000u64;
    let step = if n > 100 { n / 100 } else { 1 };
    for i in (0..n).step_by(step) {
        for j in (i + 1..n).step_by(step) {
            let product = (x[i] - x[j]) * (y[i] - y[j]);
            if product > 0.0 {
                concordant += 1;
            } else if product < 0.0 {
                discordant += 1;
            }
            if concordant + discordant > max_pairs {
                break;
            }
        }
        if concordant + discordant > max_pairs {
            break;
        }
    }
    if concordant + discordant > 0 {
        (concordant as f64 - discordant as f64) / (concordant as f64 + discordant as f64)
    } else {
        0.0
    }
}

/// Simple FOK resolution: split FOK into above/below median, compare recognition rates.
fn compute_fok_resolution(fok: &[f64], outcomes: &[f64]) -> f64 {
    if fok.len() < 4 {
        return 0.0;
    }
    let mut sorted_fok: Vec<f64> = fok.to_vec();
    sorted_fok.sort_by(|a, b| a.total_cmp(b));
    let median = sorted_fok[sorted_fok.len() / 2];

    let mut high_sum = 0.0f64;
    let mut high_count = 0u32;
    let mut low_sum = 0.0f64;
    let mut low_count = 0u32;

    for (f, o) in fok.iter().zip(outcomes) {
        if *f >= median {
            high_sum += o;
            high_count += 1;
        } else {
            low_sum += o;
            low_count += 1;
        }
    }

    let high_rate = if high_count > 0 {
        high_sum / high_count as f64
    } else {
        0.0
    };
    let low_rate = if low_count > 0 {
        low_sum / low_count as f64
    } else {
        0.0
    };

    // Resolution = difference in recognition rates (clamped to [0,1])
    (high_rate - low_rate).clamp(0.0, 1.0)
}

struct FokResult {
    fok_gamma: f64,
    recognition_hit_rate: f64,
    fok_resolution: f64,
    recall_rate: f64,
    rt_ticks: f64,
}

impl PsychBenchmark for FeelingOfKnowingBenchmark {
    fn name(&self) -> &str {
        "Metacognition::FeelingOfKnowing"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut gammas = Vec::new();
        let mut hit_rates = Vec::new();
        let mut resolutions = Vec::new();
        let mut recall_rates = Vec::new();
        let mut rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            gammas.push(r.fok_gamma);
            hit_rates.push(r.recognition_hit_rate);
            resolutions.push(r.fok_resolution);
            recall_rates.push(r.recall_rate);
            rts.push(r.rt_ticks);
        }

        result.insert("fok_gamma", MetricValue::from_samples(&gammas));
        result.insert(
            "recognition_hit_rate",
            MetricValue::from_samples(&hit_rates),
        );
        result.insert("fok_resolution", MetricValue::from_samples(&resolutions));
        result.insert("recall_rate", MetricValue::from_samples(&recall_rates));
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        result.conditions = 1;
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fok_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = FeelingOfKnowingBenchmark.run(&config);
        assert!(result.metrics.contains_key("fok_gamma"));
        assert!(result.metrics.contains_key("recognition_hit_rate"));
        assert!(result.metrics.contains_key("fok_resolution"));
        assert!(result.metrics.contains_key("recall_rate"));
        assert!(result.metrics.contains_key("rt_ticks"));
    }

    #[test]
    fn test_fok_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = FeelingOfKnowingBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_fok_gamma_bounded() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = FeelingOfKnowingBenchmark.run(&config);
        let gamma = result.metrics["fok_gamma"].mean;
        assert!(
            gamma >= -1.0 && gamma <= 1.0,
            "gamma should be in [-1, 1], got {}",
            gamma
        );
    }
}
