// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Online acquisition/resource measurement primitives for SYM-ARCH-002A6.
//!
//! This module is architecture-agnostic. It records prequential correctness
//! before each update, inference/update latency for the same step, and post-update
//! resource state. It intentionally does not decide whether any model "wins".

use serde::{Deserialize, Serialize};

pub const ONLINE_MEASUREMENT_TRACE_SCHEMA_V1: &str = "symthaea.online-measurement-trace/v1";
const TRACE_HASH_DOMAIN: &[u8] = b"symthaea.online-measurement-trace.hash/v1";

fn canonical_hash<T: Serialize>(domain: &[u8], value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&[0]);
    hasher.update(&bytes);
    Ok(hasher.finalize().to_hex().to_string())
}

fn looks_like_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LearningCriterion {
    /// Rolling prequential window size in examples.
    pub window_size: usize,
    /// Required rolling accuracy in [0,1].
    pub accuracy_threshold: f64,
    /// Number of consecutive qualifying windows required before criterion is met.
    pub consecutive_windows: usize,
}

impl LearningCriterion {
    pub fn validate(&self) -> Result<(), String> {
        if self.window_size == 0 {
            return Err("learning criterion window_size must be positive".into());
        }
        if self.consecutive_windows == 0 {
            return Err("learning criterion consecutive_windows must be positive".into());
        }
        if !self.accuracy_threshold.is_finite()
            || self.accuracy_threshold < 0.0
            || self.accuracy_threshold > 1.0
        {
            return Err("learning criterion accuracy_threshold must be finite in [0,1]".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    /// Number of trainable scalar parameters after the update.
    pub trainable_parameters: usize,
    /// Total persistent model state retained across examples.
    pub persistent_state_bytes: u64,
    /// Portion of persistent state attributable to replay/examples, when present.
    pub replay_bytes: u64,
    /// Portion of persistent state attributable to temporal/recurrent state.
    pub temporal_state_bytes: u64,
    /// Process resident-set size sampled after the update, when available.
    pub rss_bytes: Option<u64>,
}

impl ResourceSnapshot {
    pub fn validate(&self) -> Result<(), String> {
        if self.replay_bytes > self.persistent_state_bytes {
            return Err("replay bytes cannot exceed total persistent state bytes".into());
        }
        if self.temporal_state_bytes > self.persistent_state_bytes {
            return Err("temporal state bytes cannot exceed total persistent state bytes".into());
        }
        Ok(())
    }
}

/// One prequential online-learning step.
///
/// `correct_before_update` must reflect a prediction made before the label is
/// consumed by the learner. Latencies are observational measurements and must not
/// be used as hidden tuning signals in CONFIRM/REPL streams.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OnlineStepMeasurement {
    pub correct_before_update: bool,
    pub inference_latency_ns: u64,
    pub update_latency_ns: u64,
    pub resource_after_update: ResourceSnapshot,
}

impl OnlineStepMeasurement {
    pub fn validate(&self) -> Result<(), String> {
        self.resource_after_update.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OnlineMeasurementTrace {
    pub schema: String,
    /// Digest of the experiment manifest controlling this trace.
    pub manifest_digest: String,
    /// Optional digest of the exact learner/spec state before the first step.
    pub initial_state_digest: Option<String>,
    pub steps: Vec<OnlineStepMeasurement>,
}

impl OnlineMeasurementTrace {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != ONLINE_MEASUREMENT_TRACE_SCHEMA_V1 {
            return Err(format!("unsupported online-measurement schema: {}", self.schema));
        }
        if !looks_like_digest(&self.manifest_digest) {
            return Err("measurement trace manifest_digest must be a 32-byte hex digest".into());
        }
        if let Some(digest) = &self.initial_state_digest {
            if !looks_like_digest(digest) {
                return Err("measurement trace initial_state_digest must be a 32-byte hex digest".into());
            }
        }
        if self.steps.is_empty() {
            return Err("online measurement trace must contain at least one step".into());
        }
        for step in &self.steps {
            step.validate()?;
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, String> {
        self.validate()?;
        canonical_hash(TRACE_HASH_DOMAIN, self)
    }

    pub fn correctness(&self) -> Vec<bool> {
        self.steps
            .iter()
            .map(|step| step.correct_before_update)
            .collect()
    }

    /// Number of examples observed when the frozen rolling criterion is first met.
    /// Returns `None` if the criterion is never sustained.
    pub fn examples_to_criterion(
        &self,
        criterion: &LearningCriterion,
    ) -> Result<Option<usize>, String> {
        self.validate()?;
        criterion.validate()?;
        if self.steps.len() < criterion.window_size {
            return Ok(None);
        }

        let mut prefix = Vec::with_capacity(self.steps.len() + 1);
        prefix.push(0usize);
        for step in &self.steps {
            let next = prefix.last().copied().unwrap_or(0)
                + usize::from(step.correct_before_update);
            prefix.push(next);
        }

        let mut consecutive = 0usize;
        for end in criterion.window_size..=self.steps.len() {
            let start = end - criterion.window_size;
            let correct = prefix[end] - prefix[start];
            let accuracy = correct as f64 / criterion.window_size as f64;
            if accuracy + 1e-12 >= criterion.accuracy_threshold {
                consecutive += 1;
                if consecutive >= criterion.consecutive_windows {
                    return Ok(Some(end));
                }
            } else {
                consecutive = 0;
            }
        }
        Ok(None)
    }

    /// Mean cumulative prequential accuracy across learning steps.
    ///
    /// This is a normalized right-rectangle area under the cumulative-accuracy
    /// learning curve. Two runs with the same final accuracy can differ here when
    /// one acquires useful behavior earlier.
    pub fn cumulative_accuracy_auc(&self) -> Result<f64, String> {
        self.validate()?;
        let mut correct = 0usize;
        let mut area = 0.0;
        for (index, step) in self.steps.iter().enumerate() {
            correct += usize::from(step.correct_before_update);
            area += correct as f64 / (index + 1) as f64;
        }
        Ok(area / self.steps.len() as f64)
    }

    pub fn final_prequential_accuracy(&self) -> Result<f64, String> {
        self.validate()?;
        let correct = self
            .steps
            .iter()
            .filter(|step| step.correct_before_update)
            .count();
        Ok(correct as f64 / self.steps.len() as f64)
    }

    pub fn summarize(
        &self,
        criterion: &LearningCriterion,
    ) -> Result<OnlineMeasurementSummary, String> {
        self.validate()?;
        criterion.validate()?;
        let inference: Vec<u64> = self.steps.iter().map(|step| step.inference_latency_ns).collect();
        let update: Vec<u64> = self.steps.iter().map(|step| step.update_latency_ns).collect();
        let resources: Vec<ResourceSnapshot> = self
            .steps
            .iter()
            .map(|step| step.resource_after_update.clone())
            .collect();
        Ok(OnlineMeasurementSummary {
            observations: self.steps.len(),
            examples_to_criterion: self.examples_to_criterion(criterion)?,
            final_prequential_accuracy: self.final_prequential_accuracy()?,
            cumulative_accuracy_auc: self.cumulative_accuracy_auc()?,
            inference_latency: LatencySummary::from_samples(&inference)?,
            update_latency: LatencySummary::from_samples(&update)?,
            resources: ResourceTraceSummary::from_snapshots(&resources)?,
            trace_digest: self.digest()?,
        })
    }
}

fn percentile_linear(sorted: &[u64], probability: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    let position = probability.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower] as f64
    } else {
        let weight = position - lower as f64;
        sorted[lower] as f64 * (1.0 - weight) + sorted[upper] as f64 * weight
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatencySummary {
    pub samples: usize,
    pub total_ns: u64,
    pub mean_ns: f64,
    pub p50_ns: f64,
    pub p95_ns: f64,
    /// None only when every measured latency sample is zero.
    pub throughput_per_second: Option<f64>,
}

impl LatencySummary {
    pub fn from_samples(samples: &[u64]) -> Result<Self, String> {
        if samples.is_empty() {
            return Err("latency summary requires at least one sample".into());
        }
        let total_ns = samples.iter().try_fold(0u64, |total, sample| {
            total
                .checked_add(*sample)
                .ok_or_else(|| "latency total overflow".to_string())
        })?;
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        let throughput_per_second = if total_ns == 0 {
            None
        } else {
            Some(samples.len() as f64 * 1_000_000_000.0 / total_ns as f64)
        };
        Ok(Self {
            samples: samples.len(),
            total_ns,
            mean_ns: total_ns as f64 / samples.len() as f64,
            p50_ns: percentile_linear(&sorted, 0.50),
            p95_ns: percentile_linear(&sorted, 0.95),
            throughput_per_second,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceTraceSummary {
    pub samples: usize,
    pub final_trainable_parameters: usize,
    pub peak_trainable_parameters: usize,
    pub final_persistent_state_bytes: u64,
    pub peak_persistent_state_bytes: u64,
    pub peak_replay_bytes: u64,
    pub peak_temporal_state_bytes: u64,
    pub peak_rss_bytes: Option<u64>,
}

impl ResourceTraceSummary {
    pub fn from_snapshots(snapshots: &[ResourceSnapshot]) -> Result<Self, String> {
        if snapshots.is_empty() {
            return Err("resource summary requires at least one snapshot".into());
        }
        for snapshot in snapshots {
            snapshot.validate()?;
        }
        let last = snapshots.last().expect("non-empty snapshots");
        Ok(Self {
            samples: snapshots.len(),
            final_trainable_parameters: last.trainable_parameters,
            peak_trainable_parameters: snapshots
                .iter()
                .map(|snapshot| snapshot.trainable_parameters)
                .max()
                .unwrap_or(0),
            final_persistent_state_bytes: last.persistent_state_bytes,
            peak_persistent_state_bytes: snapshots
                .iter()
                .map(|snapshot| snapshot.persistent_state_bytes)
                .max()
                .unwrap_or(0),
            peak_replay_bytes: snapshots
                .iter()
                .map(|snapshot| snapshot.replay_bytes)
                .max()
                .unwrap_or(0),
            peak_temporal_state_bytes: snapshots
                .iter()
                .map(|snapshot| snapshot.temporal_state_bytes)
                .max()
                .unwrap_or(0),
            peak_rss_bytes: snapshots.iter().filter_map(|snapshot| snapshot.rss_bytes).max(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnlineMeasurementSummary {
    pub observations: usize,
    pub examples_to_criterion: Option<usize>,
    pub final_prequential_accuracy: f64,
    pub cumulative_accuracy_auc: f64,
    pub inference_latency: LatencySummary,
    pub update_latency: LatencySummary,
    pub resources: ResourceTraceSummary,
    pub trace_digest: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: &str) -> String {
        byte.repeat(64)
    }

    fn resource(persistent: u64, replay: u64, temporal: u64, rss: Option<u64>) -> ResourceSnapshot {
        ResourceSnapshot {
            trainable_parameters: 10,
            persistent_state_bytes: persistent,
            replay_bytes: replay,
            temporal_state_bytes: temporal,
            rss_bytes: rss,
        }
    }

    fn trace(correctness: &[bool]) -> OnlineMeasurementTrace {
        OnlineMeasurementTrace {
            schema: ONLINE_MEASUREMENT_TRACE_SCHEMA_V1.into(),
            manifest_digest: digest("a"),
            initial_state_digest: Some(digest("b")),
            steps: correctness
                .iter()
                .enumerate()
                .map(|(index, correct)| OnlineStepMeasurement {
                    correct_before_update: *correct,
                    inference_latency_ns: 10 + index as u64,
                    update_latency_ns: 20 + index as u64,
                    resource_after_update: resource(
                        1_000 + index as u64,
                        100,
                        50,
                        Some(5_000 + index as u64),
                    ),
                })
                .collect(),
        }
    }

    #[test]
    fn criterion_requires_sustained_rolling_accuracy() {
        let trace = trace(&[false, false, true, true, true, true]);
        let criterion = LearningCriterion {
            window_size: 3,
            accuracy_threshold: 2.0 / 3.0,
            consecutive_windows: 2,
        };
        assert_eq!(trace.examples_to_criterion(&criterion).unwrap(), Some(5));
    }

    #[test]
    fn criterion_returns_none_when_not_reached() {
        let trace = trace(&[false, true, false, true, false, true]);
        let criterion = LearningCriterion {
            window_size: 3,
            accuracy_threshold: 1.0,
            consecutive_windows: 2,
        };
        assert_eq!(trace.examples_to_criterion(&criterion).unwrap(), None);
    }

    #[test]
    fn cumulative_auc_rewards_earlier_acquisition_at_same_final_accuracy() {
        let early = trace(&[true, true, false, false]);
        let late = trace(&[false, false, true, true]);
        assert_eq!(early.final_prequential_accuracy().unwrap(), 0.5);
        assert_eq!(late.final_prequential_accuracy().unwrap(), 0.5);
        assert!(early.cumulative_accuracy_auc().unwrap() > late.cumulative_accuracy_auc().unwrap());
    }

    #[test]
    fn latency_summary_reports_deterministic_percentiles_and_throughput() {
        let summary = LatencySummary::from_samples(&[10, 20, 30, 40]).unwrap();
        assert_eq!(summary.samples, 4);
        assert_eq!(summary.total_ns, 100);
        assert!((summary.mean_ns - 25.0).abs() < 1e-12);
        assert!((summary.p50_ns - 25.0).abs() < 1e-12);
        assert!((summary.p95_ns - 38.5).abs() < 1e-12);
        assert!((summary.throughput_per_second.unwrap() - 40_000_000.0).abs() < 1e-6);
    }

    #[test]
    fn zero_latency_does_not_invent_infinite_throughput() {
        let summary = LatencySummary::from_samples(&[0, 0, 0]).unwrap();
        assert_eq!(summary.throughput_per_second, None);
    }

    #[test]
    fn resource_summary_tracks_final_and_peak_state() {
        let snapshots = vec![
            resource(100, 10, 20, Some(1_000)),
            resource(140, 30, 25, Some(1_200)),
            resource(120, 20, 22, None),
        ];
        let summary = ResourceTraceSummary::from_snapshots(&snapshots).unwrap();
        assert_eq!(summary.final_persistent_state_bytes, 120);
        assert_eq!(summary.peak_persistent_state_bytes, 140);
        assert_eq!(summary.peak_replay_bytes, 30);
        assert_eq!(summary.peak_temporal_state_bytes, 25);
        assert_eq!(summary.peak_rss_bytes, Some(1_200));
    }

    #[test]
    fn trace_digest_is_order_sensitive_and_manifest_bound() {
        let first = trace(&[true, false, true]);
        let mut reordered = trace(&[false, true, true]);
        assert_ne!(first.digest().unwrap(), reordered.digest().unwrap());
        reordered = first.clone();
        reordered.manifest_digest = digest("c");
        assert_ne!(first.digest().unwrap(), reordered.digest().unwrap());
    }

    #[test]
    fn invalid_resource_component_fails_closed() {
        let mut invalid = trace(&[true]);
        invalid.steps[0].resource_after_update.replay_bytes = 2_000;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn summary_binds_learning_latency_resource_and_trace_identity() {
        let trace = trace(&[false, true, true, true]);
        let criterion = LearningCriterion {
            window_size: 2,
            accuracy_threshold: 1.0,
            consecutive_windows: 2,
        };
        let summary = trace.summarize(&criterion).unwrap();
        assert_eq!(summary.observations, 4);
        assert_eq!(summary.examples_to_criterion, Some(4));
        assert!(looks_like_digest(&summary.trace_digest));
        assert_eq!(summary.inference_latency.samples, 4);
        assert_eq!(summary.update_latency.samples, 4);
        assert_eq!(summary.resources.samples, 4);
    }
}
