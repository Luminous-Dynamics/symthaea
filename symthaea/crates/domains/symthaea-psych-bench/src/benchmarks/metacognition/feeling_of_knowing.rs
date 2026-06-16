// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Feeling of Knowing benchmark testing metamemory accuracy.
pub struct FeelingOfKnowingBenchmark;

impl FeelingOfKnowingBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> FokResult {
        let diff_model = difficulty_model_for(self.name());
        let dim = config.dimension;
        let seed = config.trial_seed("metacognition", "fok", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // More items gives more failed-recall trials for the FOK judgment,
        // increasing statistical power in the gamma computation (Nelson 1984 —
        // gamma reliability increases with number of (FOK, recognition) pairs).
        let num_items = 80;

        // Recall uses logistic threshold: recall_sensitivity controls sharpness.
        // Sharper sensitivity (10.0) ensures high-encoding items mostly succeed,
        // keeping the FOK pool concentrated at lower encodings for better
        // recognition discrimination. Threshold 0.62 targets ~40% recall rate.
        // Wickelgren (1977): speed emphasis raises response criterion.
        // Lapse_rate lowers recall threshold — models premature recall termination
        // under reduced cognitive control (Koriat 2007).
        let recall_threshold: f64 = 0.62 - config.lapse_rate * 0.06 + config.time_pressure * 0.08;
        let recall_sensitivity: f64 = 10.0;

        // Metacognitive noise: imperfect FOK introspection.
        // Base noise 0.008 (reduced from 0.010) preserves the encoding gradient
        // in FOK rankings more faithfully. The encoding range (~0.20-0.95) is
        // the primary discriminative signal — Koriat (1997) showed that trace
        // accessibility is the dominant cue for FOK judgments, and narrower
        // metacognitive noise better preserves this signal's rank ordering.
        // Human gamma ~0.65 (Hart 1965; Metcalfe et al. 1993; Schwartz 1994).
        // Lichtenstein et al. (1982): calibration degrades under time pressure.
        // Lapse_rate degrades metacognitive precision — models reduced
        // introspective accuracy under attentional lapses (Reder & Ritter 1992;
        // individual differences in metacognitive monitoring quality).
        let fok_noise_range: f64 = (0.008 + config.lapse_rate * 0.04 + config.time_pressure * 0.08)
            / diff_model.signal_multiplier(config.difficulty);

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
            let attention_noise = ((rng % 1000) as f64 / 1000.0 - 0.5) * 0.30;
            // Wider encoding range (0.20 - 0.95) increases the variance of
            // encoding quality across items, producing more discriminable FOK
            // ratings for the gamma computation (Nelson 1984 — restriction of
            // range attenuates gamma). Base 0.55 (down from 0.60) lowers the
            // floor for mid-list items, creating more failed-recall items with
            // diverse encoding strengths.
            // Difficulty degrades encoding via SNR reduction (signal_multiplier < 1.0
            // at high difficulty). This compresses encoding toward the midpoint,
            // reducing FOK discriminability — matching human FOK degradation under
            // cognitive load (Lichtenstein et al. 1982).
            let sig_mult = diff_model.signal_multiplier(config.difficulty);
            // Lapse_rate degrades encoding quality — models reduced depth
            // of processing under attentional lapses (Craik & Lockhart 1972).
            let lapse_encoding_penalty = config.lapse_rate * 0.15;
            let raw_enc = 0.55 + primacy + recency + attention_noise - lapse_encoding_penalty;
            let encoding = (0.5 + (raw_enc - 0.5) * sig_mult).clamp(0.20, 0.95);

            // Create memory trace: bound cue-target pair degraded by (1 - encoding) noise.
            let pair = cue.bind(&target);
            xor_shift(&mut rng);
            let noise_hv = ContinuousHV::random(dim, rng.wrapping_add(2000 + i as u64));
            let enc_f32 = encoding as f32;
            let trace =
                ContinuousHV::weighted_bundle(&[&pair, &noise_hv], &[enc_f32, 1.0 - enc_f32]);

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
            // Encoding-weighted FOK: for failed-recall items, partial activation
            // is noisy and compressed, so encoding strength is the primary
            // discriminative signal (Koriat 1997 — trace-based accessibility).
            // The 40/60 split lets encoding drive FOK rankings while partial
            // activation adds realistic variability from retrieval attempts.
            let raw_fok = partial_activation * 0.40 + encoding_strengths[i] * 0.60;

            // No logistic — preserve linear relationship with encoding.
            // Raw FOK is already in a natural [0, 1] range. Adding noise
            // models metacognitive imperfection (Lichtenstein et al. 1982).
            xor_shift(&mut rng);
            let noise = ((rng % 1000) as f64 / 1000.0 - 0.5) * fok_noise_range;
            let fok_noisy = (raw_fok + noise).clamp(0.0, 1.0);

            // Metacognitive calibration: humans compress FOK ratings toward
            // moderate values (Metcalfe 2000). Temperature 3.5 matches the
            // empirical compression from raw encoding signals to FOK ratings.
            let fok = 1.0 / (1.0 + (-((fok_noisy - 0.5) * 3.5) as f64).exp());

            // ── Recognition test: familiarity-based signal detection ──
            // Dual-process theory (Yonelinas 2002): recognition relies on
            // familiarity (encoding strength) more than recollection (retrieval).
            // Criterion 0.58 centers the transition on failed-recall items.
            // Sensitivity 20.0 creates sharp discrimination that produces
            // the large resolution (high/low FOK recognition rate difference)
            // observed in human studies (Hart 1965; Schwartz 1994):
            //   encoding=0.48 → P≈0.12, encoding=0.58 → P≈0.50, encoding=0.68 → P≈0.88
            let rec_criterion: f64 = 0.58;
            let rec_sensitivity: f64 = 20.0;
            let recognition_prob =
                1.0 / (1.0 + (-(encoding_strengths[i] - rec_criterion) * rec_sensitivity).exp());
            xor_shift(&mut rng);
            let rec_u = (rng as f64) / u64::MAX as f64;
            let recognized = rec_u < recognition_prob;
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

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Feeling of Knowing",
            citation: "Hart (1965)",
            year: 1965,
            doi: Some("10.1037/h0022263"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut gammas = Vec::new();
        let mut hit_rates = Vec::new();
        let mut resolutions = Vec::new();
        let mut recall_rates = Vec::new();
        let mut rts = Vec::new();
        let mut trace = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            gammas.push(r.fok_gamma);
            hit_rates.push(r.recognition_hit_rate);
            resolutions.push(r.fok_resolution);
            recall_rates.push(r.recall_rate);
            rts.push(r.rt_ticks);

            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "fok".to_string(),
                    correct: r.fok_gamma > 0.0,
                    rt_ticks: r.rt_ticks,
                    similarity: r.fok_gamma,
                    confidence: r.recognition_hit_rate,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("fok_gamma", MetricValue::from_samples(&gammas));
        result.insert(
            "recognition_hit_rate",
            MetricValue::from_samples(&hit_rates),
        );
        result.insert("fok_resolution", MetricValue::from_samples(&resolutions));
        result.insert("recall_rate", MetricValue::from_samples(&recall_rates));
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        if config.trial_trace {
            result.trial_trace = trace;
        }

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
