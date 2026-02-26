//! Feeling of Knowing (FOK) benchmark.
//!
//! Tests metamemory accuracy: after failed recall, can the system predict
//! whether it would recognize the correct answer? Good FOK = high gamma
//! correlation between FOK rating and subsequent recognition success.
//! Complements the Calibration benchmark (confidence during retrieval).
//!
//! Paradigm: Hart (1965) — study cue-target pairs, test recall, then
//! FOK judgment + forced-choice recognition for failed items.
//!
//! Model: Encoding quality varies by serial position (Murdock 1962) and
//! attention fluctuation. Both FOK and recognition depend monotonically
//! on encoding quality, producing the gamma correlation.
//!
//! Human baselines (Hart 1965; Metcalfe et al. 1993; Schwartz 1994):
//! - fok_gamma: 0.65 (SD≈0.10) — gamma(FOK, recognition)
//! - recognition_hit_rate: 0.75 (SD≈0.10)
//! - fok_resolution: 0.60 (SD≈0.12) — AUC of FOK predicting recognition
//! - recall_rate: 0.40 (SD≈0.12)

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;
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

        let num_items = 24;

        // Recall uses logistic threshold: recall_sensitivity controls sharpness.
        // Softer sensitivity (6.0) lets medium-encoding items also fail recall,
        // widening the encoding range in the FOK pool → stronger gamma.
        // Wickelgren (1977): speed emphasis raises response criterion.
        let recall_threshold: f64 = 0.55 + config.time_pressure * 0.08;
        let recall_sensitivity: f64 = 6.0;

        // Metacognitive noise: imperfect FOK introspection.
        // Lichtenstein et al. (1982): confidence calibration degrades under time pressure.
        let fok_noise_range: f64 = 0.12 + config.time_pressure * 0.08;

        // Recognition softmax temperature: controls 4AFC difficulty.
        // Lower = more deterministic → sharper discrimination between
        // well-encoded and poorly-encoded items → better FOK resolution.
        // At 0.12: low-encoding P≈0.61, high-encoding P≈0.88 → resolution≈0.27.
        let recognition_temp: f64 = 0.12;

        // ── Study phase ──
        // Encode cue-target pairs into memory traces with varying quality.
        // This models long-term memory encoding, not WM storage.
        // Encoding quality varies by serial position (Murdock 1962)
        // and random attention fluctuation.
        let mut cues = Vec::with_capacity(num_items);
        let mut targets = Vec::with_capacity(num_items);
        let mut traces = Vec::with_capacity(num_items);
        let mut encoding_strengths = Vec::with_capacity(num_items);

        for i in 0..num_items {
            xor_shift(&mut rng);
            let cue = ContinuousHV::random(dim, rng.wrapping_add(i as u64));
            xor_shift(&mut rng);
            let target = ContinuousHV::random(dim, rng.wrapping_add(1000 + i as u64));

            // Serial position curve: primacy + recency advantages.
            // Atkinson & Shiffrin (1968): rehearsal-based primacy gradient.
            // Glanzer & Cunitz (1966): recency from short-term buffer.
            let serial_pos = i as f64 / (num_items - 1).max(1) as f64;
            let primacy = (-serial_pos * 3.0).exp() * 0.12;
            let recency = (-(1.0 - serial_pos) * 3.0).exp() * 0.12;
            xor_shift(&mut rng);
            let attention_noise = ((rng % 1000) as f64 / 1000.0 - 0.5) * 0.15;
            let encoding = (0.60 + primacy + recency + attention_noise).clamp(0.25, 0.95);

            // Create memory trace: bound cue-target pair degraded by (1 - encoding) noise.
            let pair = cue.bind(&target);
            xor_shift(&mut rng);
            let noise_hv = ContinuousHV::random(dim, rng.wrapping_add(2000 + i as u64));
            let enc_f32 = encoding as f32;
            let trace = ContinuousHV::weighted_bundle(
                &[&pair, &noise_hv],
                &[enc_f32, 1.0 - enc_f32],
            );

            cues.push(cue);
            targets.push(target);
            traces.push(trace);
            encoding_strengths.push(encoding);
        }

        // ── Test phase ──
        let mut recall_successes = 0u32;
        let mut fok_ratings = Vec::new();
        let mut recognition_outcomes = Vec::new();
        let mut all_rts = Vec::new();

        for i in 0..num_items {
            // Recall: unbind cue from this item's trace → approximation of target.
            // Quality depends on encoding strength (trace fidelity).
            let unbound = cues[i].bind(&traces[i]);
            let recall_sim = unbound.similarity(&targets[i]) as f64;

            // RT: deliberation time inversely proportional to match quality.
            let rt = 5.0 + (1.0 - recall_sim.clamp(0.0, 1.0)) * 6.0;
            all_rts.push(rt);

            // Probabilistic recall: logistic function of similarity.
            // Gradual falloff rather than hard threshold — some well-encoded items
            // fail recall and some poorly-encoded items succeed, matching human
            // variability (Koriat & Goldsmith 1996).
            let recall_prob =
                1.0 / (1.0 + (-(recall_sim - recall_threshold) * recall_sensitivity).exp());
            xor_shift(&mut rng);
            let u = (rng as f64) / u64::MAX as f64;
            if u < recall_prob {
                recall_successes += 1;
                continue;
            }

            // ── Failed recall → FOK judgment ──
            // Koriat (1993): accessibility heuristic — FOK reflects partial
            // retrieval strength even when full recall fails.
            let partial_activation = recall_sim.max(0.0);

            // Raw FOK combines partial retrieval strength + implicit
            // encoding-quality awareness (Nelson & Narens 1990).
            // Note: cue-trace similarity is near-zero in high dimensions
            // (bind = element-wise multiply, so cosine(cue, cue*target) ≈ 0),
            // so we rely on partial activation + encoding strength.
            let raw_fok = partial_activation * 0.60 + encoding_strengths[i] * 0.40;

            // Logistic scaling to spread [0,1] range with centered expansion.
            let scaled_fok = 1.0 / (1.0 + (-((raw_fok - 0.40) * 5.0)).exp());

            // Metacognitive noise.
            xor_shift(&mut rng);
            let noise = ((rng % 1000) as f64 / 1000.0 - 0.5) * fok_noise_range;
            let fok = (scaled_fok + noise).clamp(0.0, 1.0);

            // ── Recognition test: 4AFC via softmax ──
            // Present target + 3 category-neighbor foils. Foils share 30%
            // similarity with the target, modeling within-category lures
            // (Tulving 1985). This prevents trivially easy recognition in
            // high dimensions where pure random foils have ~0 similarity.
            let target_sim = unbound.similarity(&targets[i]) as f64;
            let mut sims = vec![target_sim];

            for f in 0..3u64 {
                xor_shift(&mut rng);
                let foil_base = ContinuousHV::random(dim, rng.wrapping_add(5000 + i as u64 * 10 + f));
                let foil = ContinuousHV::weighted_bundle(
                    &[&targets[i], &foil_base],
                    &[0.30, 0.70],
                );
                sims.push(unbound.similarity(&foil) as f64);
            }

            // Softmax response selection over 4AFC.
            // Temperature calibrated so mean recognition ≈ 0.75 for FOK items.
            let max_sim = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_sims: Vec<f64> = sims
                .iter()
                .map(|s| ((s - max_sim) / recognition_temp).exp())
                .collect();
            let sum_exp: f64 = exp_sims.iter().sum();

            xor_shift(&mut rng);
            let pick = (rng as f64) / u64::MAX as f64;
            let mut cumulative = 0.0;
            let mut selected = 0;
            for (idx, &e) in exp_sims.iter().enumerate() {
                cumulative += e / sum_exp;
                if pick < cumulative {
                    selected = idx;
                    break;
                }
            }

            let recognized = selected == 0; // target was option 0
            fok_ratings.push(fok);
            recognition_outcomes.push(if recognized { 1.0 } else { 0.0 });
        }

        // Compute gamma (Goodman-Kruskal) between FOK ratings and recognition.
        let gamma = compute_gamma(&fok_ratings, &recognition_outcomes);

        // Recognition hit rate (among FOK-judged items only).
        let recognition_hr = if !recognition_outcomes.is_empty() {
            recognition_outcomes.iter().sum::<f64>() / recognition_outcomes.len() as f64
        } else {
            0.0
        };

        // FOK resolution: median-split recognition rate difference.
        let resolution = compute_fok_resolution(&fok_ratings, &recognition_outcomes);

        let recall_rate = recall_successes as f64 / num_items as f64;

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

    // Resolution = difference in recognition rates (clamped to [0,1]).
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

    #[test]
    fn test_fok_gamma_positive() {
        // With per-item traces, FOK should positively predict recognition.
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = FeelingOfKnowingBenchmark.run(&config);
        let gamma = result.metrics["fok_gamma"].mean;
        assert!(
            gamma > 0.0,
            "gamma should be positive with proper encoding gradient, got {:.3}",
            gamma
        );
    }
}
