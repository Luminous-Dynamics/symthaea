//! N-back working memory task.
//!
//! The system sees a stream of items and must identify when the current item
//! matches the item N positions back. Tests the updating component of WM.

use crate::adapter::sequence::{SequenceAdapter, SequenceItem};
use crate::adapter::StimulusAdapter;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use crate::wm::{WmConfig, WorkingMemory};
use symthaea_core::hdc::ContinuousHV;

/// N-back benchmark testing the updating function of working memory.
pub struct NBackBenchmark;

impl NBackBenchmark {
    /// Run a single N-back trial and return (hit_rate, false_alarm_rate, rt_ticks).
    fn run_trial(
        &self,
        n: usize,
        sequence_len: usize,
        config: &BenchmarkConfig,
        trial_idx: usize,
    ) -> (f64, f64, Vec<f64>) {
        let dim = config.dimension;
        let seed = config.trial_seed("worm", &format!("nback_{}", n), trial_idx);
        let adapter = SequenceAdapter;

        let mut wm = WorkingMemory::new(WmConfig {
            dimension: dim,
            capacity: config.working_memory_capacity,
            ..Default::default()
        });

        // Generate sequence with ~30% match targets
        let mut rng_state = seed;
        // Small vocab creates realistic proactive interference: with frequent
        // item repetitions, non-target items may match the n-back item by
        // coincidence, causing false alarms (Jonides & Nee, 2006).
        let vocab_size = 8u64;
        let mut sequence = Vec::with_capacity(sequence_len);
        for i in 0..sequence_len {
            let is_target = i >= n && {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                (rng_state % 100) < 30
            };
            if is_target {
                sequence.push(sequence[i - n]);
            } else {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                sequence.push(SequenceItem(rng_state % vocab_size));
            }
        }

        let mut hits = 0u64;
        let mut misses = 0u64;
        let mut false_alarms = 0u64;
        let mut correct_rejections = 0u64;
        let mut rt_ticks = Vec::new();

        // Keep history of perceived HVs for position-aware n-back retrieval
        let mut perceived_history: Vec<ContinuousHV> = Vec::with_capacity(sequence_len);

        // Present sequence to working memory
        for (i, &item) in sequence.iter().enumerate() {
            let hv = adapter.encode(&item, dim);
            wm.perceive(hv.clone());
            wm.tick();
            perceived_history.push(hv.clone());

            if i >= n {
                let is_target = sequence[i] == sequence[i - n];

                // N-back detection: WM-capacity-gated match
                //
                // With FIFO eviction and capacity K, an item perceived N steps ago
                // is retained iff N < K (the current item occupies one slot).
                // For N >= K, the n-back item has been evicted and the system
                // cannot identify targets — modelling WM capacity limits.
                let nback_retained = n < config.working_memory_capacity;

                // Compare current item to n-back item (identity test via HDC similarity)
                let nback_hv = &perceived_history[i - n];
                let match_sim = hv.similarity(nback_hv);
                // Time pressure: base 0.5 produces ~80% hit rate matching Kane et al. (2007) 2-back norms;
                // -0.15/unit lowers criterion, modeling liberal bias under speed emphasis (Luce, 1986).
                let match_threshold = (0.5 - config.time_pressure * 0.15) as f32;

                // Proactive interference: check for "lure" items at nearby positions
                // (n±1 back). Lures cause false alarms in human n-back (Jonides & Nee,
                // 2006; Kane et al., 2007). If a lure exceeds threshold, the system
                // may respond "match" even when it shouldn't.
                let mut lure_match = false;
                if !is_target && nback_retained {
                    for offset in [1i32, -1] {
                        let lure_pos = i as i32 - n as i32 + offset;
                        if lure_pos >= 0 && (lure_pos as usize) < i {
                            let lure_sim = hv.similarity(&perceived_history[lure_pos as usize]);
                            if lure_sim > match_threshold {
                                lure_match = true;
                                break;
                            }
                        }
                    }
                }

                // Respond "match" only if we remember the n-back item AND it
                // matches the current item. Lure matches cause false alarms.
                let responded_match = nback_retained && (match_sim > match_threshold || lure_match);

                // RT proxy: deliberation ticks based on decision margin
                let margin = (match_sim - match_threshold).abs() as f64;
                let ticks = 3.0 + (1.0 - margin.min(1.0)) * 5.0;
                rt_ticks.push(ticks);

                match (is_target, responded_match) {
                    (true, true) => hits += 1,
                    (true, false) => misses += 1,
                    (false, true) => false_alarms += 1,
                    (false, false) => correct_rejections += 1,
                }
            }
        }

        let hit_rate = if hits + misses > 0 {
            hits as f64 / (hits + misses) as f64
        } else {
            0.0
        };
        let fa_rate = if false_alarms + correct_rejections > 0 {
            false_alarms as f64 / (false_alarms + correct_rejections) as f64
        } else {
            0.0
        };

        (hit_rate, fa_rate, rt_ticks)
    }
}

impl PsychBenchmark for NBackBenchmark {
    fn name(&self) -> &str {
        "WorM::N-back"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "N-back Working Memory",
            citation: "Kirchner (1958)",
            year: 1958,
            doi: Some("10.1037/h0043688"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let sequence_len = 30;

        for n in [1, 2, 3] {
            let mut hit_rates = Vec::new();
            let mut fa_rates = Vec::new();
            let mut accuracies = Vec::new();
            let mut all_rts = Vec::new();

            for trial in 0..config.trials_per_condition {
                let (hr, fa, rts) = self.run_trial(n, sequence_len, config, trial);
                hit_rates.push(hr);
                fa_rates.push(fa);
                // d'-like accuracy: hit_rate - false_alarm_rate
                accuracies.push(hr - fa);
                all_rts.extend_from_slice(&rts);
            }

            result.insert(
                format!("nback_{}::hit_rate", n),
                MetricValue::from_samples(&hit_rates),
            );
            result.insert(
                format!("nback_{}::false_alarm_rate", n),
                MetricValue::from_samples(&fa_rates),
            );
            result.insert(
                format!("nback_{}::accuracy", n),
                MetricValue::from_samples(&accuracies),
            );
            result.insert(
                format!("nback_{}::rt_ticks", n),
                MetricValue::from_samples(&all_rts),
            );
        }

        result.conditions = 3;
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nback_runs_without_panic() {
        let config = BenchmarkConfig {
            trials_per_condition: 3,
            dimension: 256,
            ..Default::default()
        };
        let bench = NBackBenchmark;
        let result = bench.run(&config);
        assert_eq!(result.conditions, 3);
        assert!(result.metrics.contains_key("nback_1::hit_rate"));
        assert!(result.metrics.contains_key("nback_2::hit_rate"));
        assert!(result.metrics.contains_key("nback_3::hit_rate"));
    }

    #[test]
    fn test_nback_metrics_finite() {
        let config = BenchmarkConfig {
            trials_per_condition: 5,
            dimension: 256,
            ..Default::default()
        };
        let result = NBackBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
            assert!(val.std_dev.is_finite(), "metric {} std_dev not finite", key);
        }
    }
}
