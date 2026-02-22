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

use crate::adapter::sequence::{SequenceAdapter, SequenceItem};
use crate::adapter::StimulusAdapter;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::PsychBenchmark;
use crate::wm::{WmConfig, WorkingMemory};

/// Digit Span benchmark testing WM capacity via serial recall.
pub struct DigitSpanBenchmark;

impl DigitSpanBenchmark {
    /// Run a single trial. Returns (forward_span, backward_span, fwd_acc_at_7).
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("worm", "digit_span", trial_idx);
        let adapter = SequenceAdapter;

        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        // Test span lengths from 3 to 10
        let mut forward_span = 0u32;
        let mut backward_span = 0u32;
        let mut fwd_accuracy_at_7 = 0.0f64;

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
            let fwd_correct = self.test_recall(
                &sequence,
                &sequence, // recall in presentation order
                dim,
                config.working_memory_capacity,
                &adapter,
                false, // not backward
            );

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
            let bwd_correct = self.test_recall(
                &sequence,
                &reversed, // recall in reverse order
                dim,
                config.working_memory_capacity,
                &adapter,
                true, // backward: adds output interference
            );

            if bwd_correct == span_len as u32 {
                backward_span = span_len as u32;
            }
        }

        TrialResult {
            forward_span,
            backward_span,
            fwd_accuracy_at_7,
        }
    }

    /// Present a sequence to WM, then test recall of expected items.
    /// Returns count of correctly recalled positions.
    ///
    /// For backward recall, each retrieval incurs a tick (output interference)
    /// and requires higher similarity, modeling the cognitive cost of
    /// maintaining reversed order (Gathercole et al., 2004).
    fn test_recall(
        &self,
        presentation: &[SequenceItem],
        expected_recall: &[SequenceItem],
        dim: usize,
        capacity: usize,
        adapter: &SequenceAdapter,
        is_backward: bool,
    ) -> u32 {
        let mut wm = WorkingMemory::new(WmConfig {
            dimension: dim,
            capacity,
            ..Default::default()
        });

        // Encode and present each digit to WM
        for item in presentation {
            let hv = adapter.encode(item, dim);
            wm.perceive(hv);
            wm.tick();
        }

        // Recall phase: forward uses raw similarity (items are fresh);
        // backward uses activation_weighted_similarity (Gathercole et al., 2004)
        // — items presented early have decayed activation, making them harder
        // to retrieve when probed last (reversed order). Output interference
        // (1 tick per retrieval) further degrades earlier items.
        let mut correct = 0u32;
        for expected in expected_recall {
            let target_hv = adapter.encode(expected, dim);

            let recall_score = if is_backward {
                // Activation-weighted: early items have lower activation
                wm.activation_weighted_similarity(&target_hv)
            } else {
                // Raw similarity: all items in capacity are fresh
                wm.contents()
                    .iter()
                    .map(|item| target_hv.similarity(item))
                    .fold(f32::NEG_INFINITY, f32::max)
            };

            let threshold = if is_backward { 0.60 } else { 0.5 };
            if recall_score > threshold {
                correct += 1;
            }

            // Output interference: backward recall takes time per item
            if is_backward {
                wm.tick();
            }
        }

        correct
    }
}

struct TrialResult {
    forward_span: u32,
    backward_span: u32,
    fwd_accuracy_at_7: f64,
}

impl PsychBenchmark for DigitSpanBenchmark {
    fn name(&self) -> &str {
        "WorM::DigitSpan"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut fwd_spans = Vec::new();
        let mut bwd_spans = Vec::new();
        let mut fwd_acc_7 = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            fwd_spans.push(r.forward_span as f64);
            bwd_spans.push(r.backward_span as f64);
            fwd_acc_7.push(r.fwd_accuracy_at_7);
        }

        result.insert("forward_span", MetricValue::from_samples(&fwd_spans));
        result.insert("backward_span", MetricValue::from_samples(&bwd_spans));
        result.insert(
            "forward_accuracy_at_7",
            MetricValue::from_samples(&fwd_acc_7),
        );

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
}
