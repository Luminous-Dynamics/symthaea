// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-scenario evaluation harness for sweeping task × road × perturbation combos.
//!
//! Runs each scenario as a short training session and collects structured results,
//! enabling ablation studies and regression checks across the full configuration space.

use crate::perturbations::PerturbationSchedule;
use crate::road::Road;
use crate::swarm::SwarmConfig;
use crate::training::{EpisodeMetrics, VehicleTrainer};
use crate::types::*;

/// Configuration for a single evaluation scenario.
#[derive(Debug, Clone)]
pub struct ScenarioConfig {
    /// Human-readable scenario name.
    pub name: String,
    /// Vehicle task.
    pub task: VehicleTask,
    /// Road model.
    pub road: Road,
    /// Optional swarm configuration.
    pub swarm: Option<SwarmConfig>,
    /// Physics perturbation schedule.
    pub perturbation: PerturbationSchedule,
    /// Training-level perturbations (swarm scenario events).
    pub training_perturbations: Vec<TrainingPerturbation>,
    /// Number of episodes to train.
    pub num_episodes: usize,
    /// Steps per episode.
    pub steps_per_episode: usize,
}

/// Result of running a single scenario.
#[derive(Debug, Clone)]
pub struct ScenarioResult {
    /// Scenario name.
    pub name: String,
    /// Task used.
    pub task: VehicleTask,
    /// Per-episode metrics.
    pub metrics: Vec<EpisodeMetrics>,
    /// Average episode reward across all episodes.
    pub avg_reward: f64,
    /// Average safety reward across all episodes.
    pub avg_safety: f64,
    /// Average speed across all episodes.
    pub avg_speed: f64,
    /// Average lateral offset across all episodes.
    pub avg_lateral_offset: f64,
    /// Whether all episodes completed without early termination.
    pub all_completed: bool,
    /// Whether all metrics are finite (no NaN/Inf).
    pub all_finite: bool,
}

/// Run a single scenario and return its result.
pub fn run_scenario(scenario: &ScenarioConfig) -> ScenarioResult {
    let config = VehicleConfig {
        num_episodes: scenario.num_episodes,
        steps_per_episode: scenario.steps_per_episode,
        early_termination: false,
        task: scenario.task,
        ..VehicleConfig::default()
    };

    let mut trainer = VehicleTrainer::new(config)
        .with_road(scenario.road.clone())
        .with_perturbations(scenario.perturbation.clone())
        .with_training_perturbations(scenario.training_perturbations.clone());

    if let Some(ref swarm) = scenario.swarm {
        trainer = trainer.with_swarm(swarm.clone());
    }

    let metrics = trainer.train();

    let n = metrics.len().max(1) as f64;
    let avg_reward = metrics.iter().map(|m| m.avg_episode_reward).sum::<f64>() / n;
    let avg_safety = metrics.iter().map(|m| m.avg_safety_reward).sum::<f64>() / n;
    let avg_speed = metrics.iter().map(|m| m.avg_speed).sum::<f64>() / n;
    let avg_lateral_offset = metrics.iter().map(|m| m.avg_lateral_offset).sum::<f64>() / n;

    let all_completed = metrics
        .iter()
        .all(|m| m.total_steps == scenario.steps_per_episode);
    let all_finite = metrics.iter().all(|m| {
        m.avg_episode_reward.is_finite()
            && m.avg_safety_reward.is_finite()
            && m.avg_speed.is_finite()
    });

    ScenarioResult {
        name: scenario.name.clone(),
        task: scenario.task,
        metrics,
        avg_reward,
        avg_safety,
        avg_speed,
        avg_lateral_offset,
        all_completed,
        all_finite,
    }
}

/// Default suite of 9 scenarios covering all tasks, roads, and perturbations.
pub fn default_scenarios() -> Vec<ScenarioConfig> {
    let ep = 5;
    let steps = 200;

    vec![
        // 1. LaneKeep on straight road (baseline)
        ScenarioConfig {
            name: "lanekeep_straight".into(),
            task: VehicleTask::LaneKeep,
            road: Road::straight(),
            swarm: None,
            perturbation: PerturbationSchedule::none(),
            training_perturbations: vec![],
            num_episodes: ep,
            steps_per_episode: steps,
        },
        // 2. LaneKeep on gentle highway
        ScenarioConfig {
            name: "lanekeep_highway".into(),
            task: VehicleTask::LaneKeep,
            road: Road::gentle_highway(),
            swarm: None,
            perturbation: PerturbationSchedule::none(),
            training_perturbations: vec![],
            num_episodes: ep,
            steps_per_episode: steps,
        },
        // 3. LaneKeep with crosswind perturbation
        ScenarioConfig {
            name: "lanekeep_crosswind".into(),
            task: VehicleTask::LaneKeep,
            road: Road::straight(),
            swarm: None,
            perturbation: PerturbationSchedule::crosswind(),
            training_perturbations: vec![],
            num_episodes: ep,
            steps_per_episode: steps,
        },
        // 4. LaneKeep with black ice
        ScenarioConfig {
            name: "lanekeep_black_ice".into(),
            task: VehicleTask::LaneKeep,
            road: Road::straight(),
            swarm: None,
            perturbation: PerturbationSchedule::black_ice(),
            training_perturbations: vec![],
            num_episodes: ep,
            steps_per_episode: steps,
        },
        // 5. Follow with swarm on straight road
        ScenarioConfig {
            name: "follow_straight_swarm".into(),
            task: VehicleTask::Follow { target_gap: 30.0 },
            road: Road::straight(),
            swarm: Some(SwarmConfig::convoy_straight(3)),
            perturbation: PerturbationSchedule::none(),
            training_perturbations: vec![],
            num_episodes: ep,
            steps_per_episode: steps,
        },
        // 6. Follow with lead-brake cascade
        ScenarioConfig {
            name: "follow_lead_brake".into(),
            task: VehicleTask::Follow { target_gap: 30.0 },
            road: Road::straight(),
            swarm: Some(SwarmConfig::convoy_straight(3)),
            perturbation: PerturbationSchedule::none(),
            training_perturbations: lead_brake_cascade(3),
            num_episodes: ep,
            steps_per_episode: steps,
        },
        // 7. LaneChange on straight road
        ScenarioConfig {
            name: "lanechange_straight".into(),
            task: VehicleTask::LaneChange,
            road: Road::straight(),
            swarm: None,
            perturbation: PerturbationSchedule::none(),
            training_perturbations: vec![],
            num_episodes: ep,
            steps_per_episode: steps,
        },
        // 8. EmergencyStop on straight road
        ScenarioConfig {
            name: "estop_straight".into(),
            task: VehicleTask::EmergencyStop,
            road: Road::straight(),
            swarm: None,
            perturbation: PerturbationSchedule::none(),
            training_perturbations: vec![],
            num_episodes: ep,
            steps_per_episode: steps,
        },
        // 9. LaneKeep with tire blowout
        ScenarioConfig {
            name: "lanekeep_tire_blowout".into(),
            task: VehicleTask::LaneKeep,
            road: Road::straight(),
            swarm: None,
            perturbation: PerturbationSchedule::tire_blowout_front(),
            training_perturbations: vec![],
            num_episodes: ep,
            steps_per_episode: steps,
        },
        // 10. LaneKeep with sensor blindness
        ScenarioConfig {
            name: "lanekeep_sensor_blind".into(),
            task: VehicleTask::LaneKeep,
            road: Road::straight(),
            swarm: None,
            perturbation: PerturbationSchedule::sensor_blindness(),
            training_perturbations: vec![],
            num_episodes: ep,
            steps_per_episode: steps,
        },
        // 11. Follow with mesh spoof attack
        ScenarioConfig {
            name: "follow_mesh_spoof".into(),
            task: VehicleTask::Follow { target_gap: 30.0 },
            road: Road::straight(),
            swarm: Some(SwarmConfig::convoy_straight(3)),
            perturbation: PerturbationSchedule::mesh_spoof(),
            training_perturbations: vec![],
            num_episodes: ep,
            steps_per_episode: steps,
        },
        // 12. LaneKeep with road impulse (pothole)
        ScenarioConfig {
            name: "lanekeep_road_impulse".into(),
            task: VehicleTask::LaneKeep,
            road: Road::straight(),
            swarm: None,
            perturbation: PerturbationSchedule::road_impulse(),
            training_perturbations: vec![],
            num_episodes: ep,
            steps_per_episode: steps,
        },
        // 13. EmergencyStop on black ice (nightmare scenario)
        ScenarioConfig {
            name: "estop_black_ice".into(),
            task: VehicleTask::EmergencyStop,
            road: Road::straight(),
            swarm: None,
            perturbation: PerturbationSchedule::black_ice(),
            training_perturbations: vec![],
            num_episodes: ep,
            steps_per_episode: steps,
        },
        // 14. LaneChange under crosswind
        ScenarioConfig {
            name: "lanechange_crosswind".into(),
            task: VehicleTask::LaneChange,
            road: Road::straight(),
            swarm: None,
            perturbation: PerturbationSchedule::crosswind(),
            training_perturbations: vec![],
            num_episodes: ep,
            steps_per_episode: steps,
        },
        // 15. Follow on highway (curved road with convoy)
        ScenarioConfig {
            name: "follow_highway".into(),
            task: VehicleTask::Follow { target_gap: 30.0 },
            road: Road::gentle_highway(),
            swarm: Some(SwarmConfig::convoy_highway(3)),
            perturbation: PerturbationSchedule::none(),
            training_perturbations: vec![],
            num_episodes: ep,
            steps_per_episode: steps,
        },
        // 16. Full gauntlet: ice + blindness + crosswind
        ScenarioConfig {
            name: "lanekeep_gauntlet".into(),
            task: VehicleTask::LaneKeep,
            road: Road::straight(),
            swarm: None,
            perturbation: PerturbationSchedule::gauntlet(),
            training_perturbations: vec![],
            num_episodes: ep,
            steps_per_episode: steps,
        },
    ]
}

/// Run all scenarios and return results.
pub fn run_all_scenarios(scenarios: &[ScenarioConfig]) -> Vec<ScenarioResult> {
    scenarios.iter().map(run_scenario).collect()
}

/// Export scenario results to CSV string.
pub fn results_to_csv(results: &[ScenarioResult]) -> String {
    let mut csv = String::from(
        "scenario,task,num_episodes,avg_reward,avg_safety,avg_speed,avg_lateral_offset,all_completed,all_finite\n",
    );
    for r in results {
        csv.push_str(&format!(
            "{},{:?},{},{:.6},{:.6},{:.6},{:.6},{},{}\n",
            r.name,
            r.task,
            r.metrics.len(),
            r.avg_reward,
            r.avg_safety,
            r.avg_speed,
            r.avg_lateral_offset,
            r.all_completed,
            r.all_finite,
        ));
    }
    csv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_scenarios_count() {
        let scenarios = default_scenarios();
        assert_eq!(scenarios.len(), 16);
    }

    #[test]
    fn test_single_scenario_baseline() {
        let scenario = ScenarioConfig {
            name: "test_baseline".into(),
            task: VehicleTask::LaneKeep,
            road: Road::straight(),
            swarm: None,
            perturbation: PerturbationSchedule::none(),
            training_perturbations: vec![],
            num_episodes: 3,
            steps_per_episode: 100,
        };

        let result = run_scenario(&scenario);
        assert_eq!(result.name, "test_baseline");
        assert_eq!(result.metrics.len(), 3);
        assert!(result.all_finite, "All metrics should be finite");
        assert!(result.all_completed, "All episodes should complete");
        assert!(result.avg_reward.is_finite());
        assert!(result.avg_safety.is_finite());
    }

    #[test]
    fn test_all_default_scenarios_complete() {
        let scenarios = default_scenarios();
        let results = run_all_scenarios(&scenarios);

        assert_eq!(results.len(), 16);
        for r in &results {
            assert!(r.all_finite, "Scenario '{}' has non-finite metrics", r.name);
            assert!(
                r.all_completed,
                "Scenario '{}' has incomplete episodes",
                r.name
            );
            assert!(
                r.avg_reward.is_finite(),
                "Scenario '{}' avg_reward not finite: {}",
                r.name,
                r.avg_reward
            );
        }
    }

    #[test]
    fn test_csv_export() {
        let scenario = ScenarioConfig {
            name: "csv_test".into(),
            task: VehicleTask::LaneKeep,
            road: Road::straight(),
            swarm: None,
            perturbation: PerturbationSchedule::none(),
            training_perturbations: vec![],
            num_episodes: 2,
            steps_per_episode: 50,
        };

        let result = run_scenario(&scenario);
        let csv = results_to_csv(&[result]);

        assert!(csv.starts_with("scenario,task,"));
        assert!(csv.contains("csv_test"));
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 2, "Header + 1 data row");
    }

    #[test]
    fn test_scenario_with_swarm() {
        let scenario = ScenarioConfig {
            name: "follow_test".into(),
            task: VehicleTask::Follow { target_gap: 30.0 },
            road: Road::straight(),
            swarm: Some(SwarmConfig::convoy_straight(2)),
            perturbation: PerturbationSchedule::none(),
            training_perturbations: vec![],
            num_episodes: 3,
            steps_per_episode: 100,
        };

        let result = run_scenario(&scenario);
        assert!(result.all_finite, "Follow scenario should be finite");
        assert!(result.all_completed, "Follow scenario should complete");
    }

    #[test]
    fn test_scenario_with_perturbation() {
        let scenario = ScenarioConfig {
            name: "perturb_test".into(),
            task: VehicleTask::LaneKeep,
            road: Road::straight(),
            swarm: None,
            perturbation: PerturbationSchedule::crosswind(),
            training_perturbations: vec![],
            num_episodes: 3,
            steps_per_episode: 100,
        };

        let result = run_scenario(&scenario);
        assert!(result.all_finite, "Perturbed scenario should be finite");
    }
}
