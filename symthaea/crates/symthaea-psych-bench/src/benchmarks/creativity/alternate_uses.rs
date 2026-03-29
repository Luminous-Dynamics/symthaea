// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Alternate Uses Task (AUT).
//!
//! Encode common objects as feature bundles, then generate "uses" by unbinding
//! features and binding with random contexts. A use is accepted if its similarity
//! to a `useful_action_proto` falls in a moderate range (not too close = copying,
//! not too far = nonsense). Measures fluency, originality, and flexibility.
//! Human baseline: fluency ~8.0 (Torrance 1974).

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Alternate Uses Task benchmark.
pub struct AlternateUsesBenchmark;

fn next_seed(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

struct ObjectDef {
    name: &'static str,
    feature_count: usize,
}

impl AlternateUsesBenchmark {
    fn objects() -> Vec<ObjectDef> {
        vec![
            ObjectDef {
                name: "brick",
                feature_count: 4,
            },
            ObjectDef {
                name: "paperclip",
                feature_count: 3,
            },
            ObjectDef {
                name: "newspaper",
                feature_count: 4,
            },
            ObjectDef {
                name: "shoe",
                feature_count: 3,
            },
            ObjectDef {
                name: "blanket",
                feature_count: 4,
            },
        ]
    }

    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> (f64, f64, f64, f64) {
        let dim = config.dimension;
        let objects = Self::objects();
        let obj = &objects[trial_idx % objects.len()];
        let seed = config.trial_seed("creativity", obj.name, trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        // Create dimension HVs for features
        let feature_dims: Vec<ContinuousHV> = (0..obj.feature_count)
            .map(|_| ContinuousHV::random(dim, next_seed(&mut rng)))
            .collect();

        // Create feature value HVs for this object
        let feature_vals: Vec<ContinuousHV> = (0..obj.feature_count)
            .map(|_| ContinuousHV::random(dim, next_seed(&mut rng)))
            .collect();

        // Object = bundle of (dim_i ⊗ val_i) pairs
        let bound_features: Vec<ContinuousHV> = feature_dims
            .iter()
            .zip(feature_vals.iter())
            .map(|(d, v)| d.bind(v))
            .collect();
        // Create "useful action" prototype
        let useful_action_proto = ContinuousHV::random(dim, next_seed(&mut rng));

        // Generate candidate uses by unbinding features and binding with random contexts
        // Cognitive generation budget: humans produce ~8 uses in typical 2-minute
        // window (Torrance 1974, mean). Creative individuals with high openness
        // produce 12+ uses (Silvia et al., 2008). With the ~14% acceptance rate
        // from the semantic plausibility filter, 75 attempts models persistent
        // associative search — near the 75th percentile of fluency distributions.
        let max_attempts = 75;
        let mut accepted_uses = Vec::new();
        let mut use_sims = Vec::new(); // similarity of each use to object (for originality)

        // Track "categories" for flexibility: group by which feature was unbound
        let mut categories_used = std::collections::HashSet::new();

        // RT tracking: attempts between each accepted use (search time per idea)
        let mut attempts_since_last = 0u32;
        let mut search_rts = Vec::new();

        for attempt in 0..max_attempts {
            // Cycle through features first (guarantees coverage), then random
            let feat_idx = if attempt < obj.feature_count {
                attempt % obj.feature_count
            } else {
                (next_seed(&mut rng) as usize) % obj.feature_count
            };

            // Create a random context
            let context = ContinuousHV::random(dim, next_seed(&mut rng));

            // New use = unbind old feature, bind with new context
            let unbound = feature_dims[feat_idx].bind(&context);

            // Replace that feature in the bundle
            let mut new_features = bound_features.clone();
            new_features[feat_idx] = unbound;
            let new_refs: Vec<&ContinuousHV> = new_features.iter().collect();
            let use_hv = ContinuousHV::bundle(&new_refs);

            // Check similarity to useful_action_proto
            let sim = use_hv.similarity(&useful_action_proto);

            // Accept if in moderate range: not copying original, not nonsense.
            // Tighter band (vs 0.01-0.70) models the semantic plausibility
            // constraint: uses must be functionally grounded (Silvia et al., 2008).
            // Time pressure: -0.02/unit lowers minimum originality, +0.10/unit raises max similarity;
            // models relaxed quality criteria under deadline (Silvia et al., 2008; Beaty et al., 2014 AUT).
            let lower_bound = (0.04 - config.time_pressure * 0.02) as f32;
            let upper_bound = (0.62 + config.time_pressure * 0.10) as f32;
            attempts_since_last += 1;
            if sim > lower_bound && sim < upper_bound {
                accepted_uses.push(use_hv);
                use_sims.push(sim);
                categories_used.insert(feat_idx);
                // RT: number of attempts to find this use (search ticks)
                search_rts.push(attempts_since_last as f64);
                attempts_since_last = 0;
            }

            let _ = attempt; // used for loop control
        }

        // Fluency: number of accepted uses
        let fluency = accepted_uses.len() as f64;

        // Originality: mean distance from useful_action_proto (higher = more original)
        let originality = if use_sims.is_empty() {
            0.0
        } else {
            // Invert: lower similarity = more original
            let mean_sim = use_sims.iter().sum::<f32>() / use_sims.len() as f32;
            (1.0 - mean_sim as f64).max(0.0)
        };

        // Flexibility: number of distinct feature categories explored
        let flexibility = categories_used.len() as f64;

        // RT: mean search ticks per accepted use
        let mean_search_rt = if search_rts.is_empty() {
            max_attempts as f64
        } else {
            search_rts.iter().sum::<f64>() / search_rts.len() as f64
        };

        (fluency, originality, flexibility, mean_search_rt)
    }
}

impl PsychBenchmark for AlternateUsesBenchmark {
    fn name(&self) -> &str {
        "Creativity::AlternateUses"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Alternate Uses Task",
            citation: "Guilford (1967)",
            year: 1967,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut fluencies = Vec::new();
        let mut originalities = Vec::new();
        let mut flexibilities = Vec::new();
        let mut rt_ticks = Vec::new();

        for trial in 0..config.trials_per_condition {
            let (f, o, flex, rt) = self.run_trial(config, trial);
            fluencies.push(f);
            originalities.push(o);
            flexibilities.push(flex);
            rt_ticks.push(rt);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "alternate_uses".to_string(),
                    correct: f > 0.0,
                    rt_ticks: rt,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("fluency", MetricValue::from_samples(&fluencies));
        result.insert("originality", MetricValue::from_samples(&originalities));
        result.insert("flexibility", MetricValue::from_samples(&flexibilities));
        result.insert("rt_ticks", MetricValue::from_samples(&rt_ticks));

        result.conditions = 1;
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
    fn test_alternate_uses_runs() {
        let config = BenchmarkConfig {
            trials_per_condition: 5,
            dimension: 256,
            ..Default::default()
        };
        let result = AlternateUsesBenchmark.run(&config);
        assert!(result.metrics.contains_key("fluency"));
        assert!(result.metrics.contains_key("originality"));
        assert!(result.metrics.contains_key("flexibility"));
        for val in result.metrics.values() {
            assert!(val.mean.is_finite());
        }
    }

    #[test]
    fn test_alternate_uses_values() {
        let config = BenchmarkConfig::default();
        let result = AlternateUsesBenchmark.run(&config);
        for (key, val) in &result.metrics {
            eprintln!("AUT {key}: mean={:.4}, sd={:.4}", val.mean, val.std_dev);
        }
    }
}
