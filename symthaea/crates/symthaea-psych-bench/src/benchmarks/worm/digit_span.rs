// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Digit Span working memory benchmark.
//!
//! Tests raw WM capacity via forward and backward digit recall.
//! The system is presented increasingly long digit sequences and must
//! recall them in order (forward) or reversed (backward).
//!
//! Human baselines (Wechsler, 2008; Woods et al., 2011):
//! - forward_span: 6.8 (longest sequence recalled forward)
//! - backward_span: 5.1 (longest sequence recalled backward)
//! - forward_accuracy_at_7: 0.82

use crate::adapter::StimulusAdapter;
use crate::adapter::sequence::{SequenceAdapter, SequenceItem};
use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use crate::wm::ssm_temporal::SsmTemporalBackend;
use crate::wm::{WmConfig, WorkingMemory};
use std::collections::BTreeMap;

/// Digit Span benchmark testing WM capacity via serial recall.
pub struct DigitSpanBenchmark;

impl DigitSpanBenchmark {
    /// Run a single trial. Returns (forward_span, backward_span, fwd_acc_at_7, fwd_rt, bwd_rt).
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("worm", "digit_span", trial_idx);
        let adapter = SequenceAdapter;

        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        // Test span lengths from 3 to 10
        let mut forward_span = 0u32;
        let mut backward_span = 0u32;
        let mut fwd_accuracy_at_7 = 0.0f64;
        let mut fwd_rt_sum = 0.0f64;
        let mut bwd_rt_sum = 0.0f64;
        let mut fwd_rt_count = 0u32;
        let mut bwd_rt_count = 0u32;

        for span_len in 3..=10usize {
            // Generate a unique digit sequence (no repeats) via Fisher-Yates shuffle.
            // Standard digit span uses non-repeating sequences (Wechsler, 2008).
            let mut digits: Vec<u64> = (0..10).collect();
            for i in (1..10).rev() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let j = (rng % (i as u64 + 1)) as usize;
                digits.swap(i, j);
            }
            let sequence: Vec<SequenceItem> = digits[..span_len]
                .iter()
                .map(|&d| SequenceItem(d))
                .collect();

            // ── Forward recall ──
            let (fwd_correct, fwd_rt) = self.test_recall(
                &sequence,
                &sequence, // recall in presentation order
                dim,
                config.working_memory_capacity,
                &adapter,
                false, // not backward
                config.time_pressure,
                config.difficulty,
                config.ssm_backend,
                config.effective_noise(),
            );
            fwd_rt_sum += fwd_rt;
            fwd_rt_count += 1;

            // Digit span criterion: ALL items must be correctly recalled
            // (standard Wechsler administration; Woods et al., 2011)
            if fwd_correct == span_len as u32 {
                forward_span = span_len as u32;
            }

            if span_len == 7 {
                fwd_accuracy_at_7 = fwd_correct as f64 / span_len as f64;
            }

            // ── Backward recall ──
            let reversed: Vec<SequenceItem> = sequence.iter().rev().copied().collect();
            let (bwd_correct, bwd_rt) = self.test_recall(
                &sequence,
                &reversed, // recall in reverse order
                dim,
                config.working_memory_capacity,
                &adapter,
                true, // backward: adds output interference
                config.time_pressure,
                config.difficulty,
                config.ssm_backend,
                config.effective_noise(),
            );
            bwd_rt_sum += bwd_rt;
            bwd_rt_count += 1;

            if bwd_correct == span_len as u32 {
                backward_span = span_len as u32;
            }
        }

        let mean_fwd_rt = if fwd_rt_count > 0 {
            fwd_rt_sum / fwd_rt_count as f64
        } else {
            0.0
        };
        let mean_bwd_rt = if bwd_rt_count > 0 {
            bwd_rt_sum / bwd_rt_count as f64
        } else {
            0.0
        };

        TrialResult {
            forward_span,
            backward_span,
            fwd_accuracy_at_7,
            forward_rt: mean_fwd_rt,
            backward_rt: mean_bwd_rt,
        }
    }

    /// Present a sequence to WM, then test recall of expected items.
    /// Returns (count of correctly recalled positions, mean_rt_ticks).
    ///
    /// For backward recall, each retrieval incurs a tick (output interference)
    /// and requires higher similarity, modeling the cognitive cost of
    /// maintaining reversed order (Gathercole et al., 2004).
    ///
    /// When `ssm_backend` is true, per-item memory strength is modeled by an
    /// `SsmTemporalBackend` (A=-0.50, d_state=4) instead of the WM's built-in
    /// `0.95^age` activation decay. Each digit encoding steps the SSM with
    /// the item's activation (1.0), and during recall the SSM's decayed output
    /// modulates the raw similarity score.
    #[allow(clippy::too_many_arguments)]
    fn test_recall(
        &self,
        presentation: &[SequenceItem],
        expected_recall: &[SequenceItem],
        dim: usize,
        capacity: usize,
        adapter: &SequenceAdapter,
        is_backward: bool,
        time_pressure: f64,
        difficulty: f64,
        ssm_backend: bool,
        encoding_noise: f64,
    ) -> (u32, f64) {
        let mut wm = WorkingMemory::new(WmConfig {
            dimension: dim,
            capacity,
            ..Default::default()
        });

        // SSM temporal backend: A=-0.50 for per-item decay. With 7+ items encoded
        // (each pumping activation), decay must be fast enough that by recall time
        // the last items have significantly weaker traces, producing the expected
        // backward < forward span difference.
        let mut ssm = SsmTemporalBackend::new(-0.50, 4);

        // Encode and present each digit to WM
        for item in presentation {
            let hv = adapter.encode(item, dim);
            wm.perceive(hv);
            wm.tick();
            // Step SSM with full activation for each encoded digit.
            // The SSM accumulates per-item trace that decays across ticks.
            if ssm_backend {
                ssm.step(1.0);
            }
        }

        // Recall phase: forward uses raw similarity (items are fresh);
        // backward uses activation_weighted_similarity (Gathercole et al., 2004)
        // — items presented early have decayed activation, making them harder
        // to retrieve when probed last (reversed order). Output interference
        // (1 tick per retrieval) further degrades earlier items.
        let mut correct = 0u32;
        let mut rt_sum = 0.0f64;
        for expected in expected_recall {
            let target_hv = adapter.encode(expected, dim);

            // Encoding noise degrades retrieval similarity
            let noise_degrade = encoding_noise as f32 * 0.4;
            let recall_score = if ssm_backend {
                // SSM-driven recall: raw cosine similarity modulated by the
                // SSM's current memory strength. Step with 0.0 (no new input)
                // to advance decay, then read the decayed activation.
                let ssm_activation = ssm.step(0.0);
                let raw_sim = wm
                    .contents()
                    .iter()
                    .map(|item| target_hv.similarity(item) * (1.0 - noise_degrade))
                    .fold(f32::NEG_INFINITY, f32::max);
                // Blend raw similarity with SSM decay; backward recall gets
                // stronger SSM modulation (items decay more by retrieval time).
                let ssm_weight = if is_backward { 0.5 } else { 0.3 };
                raw_sim * (1.0 - ssm_weight + ssm_weight * ssm_activation)
            } else if is_backward {
                // Legacy: activation-weighted — early items have lower activation
                wm.activation_weighted_similarity(&target_hv) * (1.0 - noise_degrade)
            } else {
                // Legacy: raw similarity — all items in capacity are fresh
                wm.contents()
                    .iter()
                    .map(|item| target_hv.similarity(item) * (1.0 - noise_degrade))
                    .fold(f32::NEG_INFINITY, f32::max)
            };

            // Time pressure: +0.10/unit raises recall threshold, modeling reduced retrieval search
            // under deadline (Woods et al., 2011 digit span norms; Wickelgren, 1977 SAT framework).
            let diff_model = difficulty_model_for("WorM::DigitSpan");
            let temp_mult = diff_model.temperature_multiplier(difficulty) as f32;
            let tp_penalty = time_pressure as f32 * 0.10;
            let threshold = if is_backward {
                0.60 * temp_mult + tp_penalty
            } else {
                0.5 * temp_mult + tp_penalty
            };
            if recall_score > threshold {
                correct += 1;
            }

            // RT proxy: deliberation ticks based on retrieval difficulty.
            // Margin between recall_score and threshold → harder retrieval → longer RT
            // (Sternberg, 1966 serial search; Wickelgren, 1977 SAT).
            let decision_margin = ((recall_score as f64 - threshold as f64).abs()).min(1.0);
            let rt_ticks = 3.0 + (1.0 - decision_margin) * 5.0;
            rt_sum += rt_ticks;

            // Output interference: backward recall takes time per item
            if is_backward {
                wm.tick();
            }
        }

        let mean_rt = if expected_recall.is_empty() {
            0.0
        } else {
            rt_sum / expected_recall.len() as f64
        };
        (correct, mean_rt)
    }
}

struct TrialResult {
    forward_span: u32,
    backward_span: u32,
    fwd_accuracy_at_7: f64,
    forward_rt: f64,
    backward_rt: f64,
}

impl PsychBenchmark for DigitSpanBenchmark {
    fn name(&self) -> &str {
        "WorM::DigitSpan"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Digit Span (Forward/Backward)",
            citation: "Wechsler (1997)",
            year: 1997,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut fwd_spans = Vec::new();
        let mut bwd_spans = Vec::new();
        let mut fwd_acc_7 = Vec::new();
        let mut fwd_rts = Vec::new();
        let mut bwd_rts = Vec::new();
        let mut trace = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            fwd_spans.push(r.forward_span as f64);
            bwd_spans.push(r.backward_span as f64);
            fwd_acc_7.push(r.fwd_accuracy_at_7);
            fwd_rts.push(r.forward_rt);
            bwd_rts.push(r.backward_rt);

            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trial * 2,
                    condition: "forward".to_string(),
                    correct: r.forward_span >= 5,
                    rt_ticks: r.forward_rt,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
                trace.push(TrialOutcome {
                    trial_idx: trial * 2 + 1,
                    condition: "backward".to_string(),
                    correct: r.backward_span >= 4,
                    rt_ticks: r.backward_rt,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("forward_span", MetricValue::from_samples(&fwd_spans));
        result.insert("backward_span", MetricValue::from_samples(&bwd_spans));
        result.insert(
            "forward_accuracy_at_7",
            MetricValue::from_samples(&fwd_acc_7),
        );
        result.insert("forward::rt_ticks", MetricValue::from_samples(&fwd_rts));
        result.insert("backward::rt_ticks", MetricValue::from_samples(&bwd_rts));

        if config.trial_trace {
            result.trial_trace = trace;
        }

        result.conditions = 2; // forward + backward
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_digit_span_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = DigitSpanBenchmark.run(&config);
        assert!(result.metrics.contains_key("forward_span"));
        assert!(result.metrics.contains_key("backward_span"));
        assert!(result.metrics.contains_key("forward_accuracy_at_7"));
    }

    #[test]
    fn test_digit_span_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = DigitSpanBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_forward_span_geq_backward() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = DigitSpanBenchmark.run(&config);
        let fwd = result.metrics["forward_span"].mean;
        let bwd = result.metrics["backward_span"].mean;
        // Forward span should generally be >= backward span
        // (allow some slack since both are stochastic)
        assert!(
            fwd >= bwd - 1.0,
            "forward span ({}) should be >= backward span ({}) - 1",
            fwd,
            bwd
        );
    }

    #[test]
    fn test_digit_span_ssm_matches_baseline() {
        let base_config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 20,
            ..Default::default()
        };
        let ssm_config = BenchmarkConfig {
            ssm_backend: true,
            ..base_config.clone()
        };
        let base_result = DigitSpanBenchmark.run(&base_config);
        let ssm_result = DigitSpanBenchmark.run(&ssm_config);
        let base_bwd = base_result.metrics["backward_span"].mean;
        let ssm_bwd = ssm_result.metrics["backward_span"].mean;
        let diff = (base_bwd - ssm_bwd).abs();
        // backward_span is integer-valued (span lengths), so allow up to 1.5 tolerance
        assert!(
            diff < 1.5,
            "SSM backward_span ({:.3}) too far from baseline ({:.3}), diff={:.3}",
            ssm_bwd,
            base_bwd,
            diff
        );
    }
}
