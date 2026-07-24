// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reproducible experiment manifests and distribution-level evaluation summaries.

use serde::{Deserialize, Serialize};

use crate::training::EpisodeMetrics;
use crate::types::{ActuationMode, HumanoidConfig};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentManifest {
    pub protocol_version: u32,
    pub experiment_id: String,
    pub genesis_phrase: String,
    pub morphology_schema_id: String,
    pub observation_channels: usize,
    pub policy_actuation_mode: ActuationMode,
    pub backend_actuation_mode: ActuationMode,
    pub simulator: String,
    pub model_hash: Option<String>,
    pub seeds: Vec<u64>,
    pub config: HumanoidConfig,
    /// True only when observation, action, reward, termination, timestep, and
    /// model semantics are verified against the named external benchmark.
    pub externally_benchmark_compatible: bool,
}

impl ExperimentManifest {
    pub fn new(
        config: HumanoidConfig,
        simulator: impl Into<String>,
        backend_actuation_mode: ActuationMode,
        model_hash: Option<String>,
        seeds: Vec<u64>,
    ) -> Self {
        let mut manifest = Self {
            protocol_version: 1,
            experiment_id: String::new(),
            genesis_phrase: config.genesis_phrase.clone(),
            morphology_schema_id: config.morphology.schema_id().to_string(),
            observation_channels: config.morphology.num_observation_channels(),
            policy_actuation_mode: ActuationMode::NormalizedTorque,
            backend_actuation_mode,
            simulator: simulator.into(),
            model_hash,
            seeds,
            config,
            externally_benchmark_compatible: false,
        };
        manifest.experiment_id = manifest.compute_experiment_id();
        manifest
    }

    pub fn compute_experiment_id(&self) -> String {
        let canonical = serde_json::to_vec(&(
            self.protocol_version,
            &self.genesis_phrase,
            &self.morphology_schema_id,
            self.observation_channels,
            self.policy_actuation_mode,
            self.backend_actuation_mode,
            &self.simulator,
            &self.model_hash,
            &self.seeds,
            &self.config,
        ))
        .expect("experiment manifest fields are serializable");
        format!("humanoid-{:016x}", fnv1a64(&canonical))
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetricSummary {
    pub count: usize,
    pub mean: f64,
    pub median: f64,
    pub standard_deviation: f64,
    pub minimum: f64,
    pub maximum: f64,
    pub confidence_95_low: f64,
    pub confidence_95_high: f64,
}

impl MetricSummary {
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }

        let count = samples.len();
        let mean = samples.iter().sum::<f64>() / count as f64;
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        let median = if count % 2 == 0 {
            (sorted[count / 2 - 1] + sorted[count / 2]) * 0.5
        } else {
            sorted[count / 2]
        };
        let variance = if count > 1 {
            samples
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / (count - 1) as f64
        } else {
            0.0
        };
        let standard_deviation = variance.sqrt();
        let half_width = if count > 1 {
            1.96 * standard_deviation / (count as f64).sqrt()
        } else {
            0.0
        };

        Self {
            count,
            mean,
            median,
            standard_deviation,
            minimum: sorted[0],
            maximum: sorted[count - 1],
            confidence_95_low: mean - half_width,
            confidence_95_high: mean + half_width,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub manifest: ExperimentManifest,
    pub standing_reward: MetricSummary,
    pub task_reward: MetricSummary,
    pub uprightness: MetricSummary,
    pub horizontal_speed: MetricSummary,
    pub control_effort: MetricSummary,
    pub cost_of_transport_proxy: MetricSummary,
    pub gait_asymmetry: MetricSummary,
    pub safety_interventions_per_episode: MetricSummary,
    pub completed_episode_fraction: f64,
}

impl EvaluationReport {
    pub fn from_episodes(manifest: ExperimentManifest, episodes: &[EpisodeMetrics]) -> Self {
        let standing_reward = MetricSummary::from_samples(
            &episodes
                .iter()
                .map(|episode| episode.avg_standing_reward)
                .collect::<Vec<_>>(),
        );
        let task_reward = MetricSummary::from_samples(
            &episodes
                .iter()
                .map(|episode| episode.avg_episode_reward)
                .collect::<Vec<_>>(),
        );
        let uprightness = MetricSummary::from_samples(
            &episodes
                .iter()
                .map(|episode| episode.avg_uprightness)
                .collect::<Vec<_>>(),
        );
        let horizontal_speed = MetricSummary::from_samples(
            &episodes
                .iter()
                .map(|episode| episode.avg_horizontal_speed)
                .collect::<Vec<_>>(),
        );
        let control_effort = MetricSummary::from_samples(
            &episodes
                .iter()
                .map(|episode| episode.avg_control_effort)
                .collect::<Vec<_>>(),
        );
        let cost_of_transport_proxy = MetricSummary::from_samples(
            &episodes
                .iter()
                .map(|episode| episode.cost_of_transport)
                .collect::<Vec<_>>(),
        );
        let gait_asymmetry = MetricSummary::from_samples(
            &episodes
                .iter()
                .map(|episode| episode.gait_asymmetry)
                .collect::<Vec<_>>(),
        );
        let safety_interventions_per_episode = MetricSummary::from_samples(
            &episodes
                .iter()
                .map(|episode| episode.safety_interventions as f64)
                .collect::<Vec<_>>(),
        );
        let completed_episode_fraction = if episodes.is_empty() {
            0.0
        } else {
            let expected = manifest.config.steps_per_episode;
            episodes
                .iter()
                .filter(|episode| episode.total_steps == expected)
                .count() as f64
                / episodes.len() as f64
        };

        Self {
            manifest,
            standing_reward,
            task_reward,
            uprightness,
            horizontal_speed,
            control_effort,
            cost_of_transport_proxy,
            gait_asymmetry,
            safety_interventions_per_episode,
            completed_episode_fraction,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn experiment_id_is_deterministic_and_semantic() {
        let config = HumanoidConfig::default();
        let a = ExperimentManifest::new(
            config.clone(),
            "simple",
            ActuationMode::NormalizedTorque,
            None,
            vec![1, 2, 3],
        );
        let b = ExperimentManifest::new(
            config,
            "simple",
            ActuationMode::NormalizedTorque,
            None,
            vec![1, 2, 3],
        );
        assert_eq!(a.experiment_id, b.experiment_id);
    }

    #[test]
    fn metric_summary_reports_distribution_not_only_mean() {
        let summary = MetricSummary::from_samples(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(summary.count, 4);
        assert_eq!(summary.mean, 2.5);
        assert_eq!(summary.median, 2.5);
        assert!(summary.standard_deviation > 1.0);
        assert!(summary.confidence_95_low < summary.mean);
        assert!(summary.confidence_95_high > summary.mean);
    }
}
