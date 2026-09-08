// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Live `ActiveInferenceAgent` bridge for the validated v1 snapshot contract.
//!
//! This module is a child of `agent`, so it can capture/restore the two private causal fields that
//! cannot be reconstructed from public state: the perception timestamp and stochastic action RNG.
//! Raw deserialized snapshots never become live agents directly; restoration requires the
//! non-serializable `ValidatedActiveInferenceAgentSnapshotV1` capability.

use super::ActiveInferenceAgent;
use crate::{
    ActiveInferenceAgentSnapshotV1, Observation, ValidatedActiveInferenceAgentSnapshotV1,
};

impl ActiveInferenceAgent {
    /// Capture the complete causal state needed for exact continuation.
    pub fn snapshot_v1(&self) -> ActiveInferenceAgentSnapshotV1 {
        ActiveInferenceAgentSnapshotV1 {
            config: self.config.clone(),
            belief: self.belief.clone(),
            previous_state: self.previous_state.clone(),
            last_action: self.last_action,
            model: self.model.clone(),
            free_energy_calc: self.free_energy_calc.clone(),
            precision: self.precision.clone(),
            efe_computer: self.efe_computer.clone(),
            td_learner: self.td_learner.clone(),
            last_fe_components: self.last_fe_components.clone(),
            stats: self.stats.clone(),
            timestamp: self.timestamp,
            rng_state: self.rng_state,
        }
    }

    /// Restore a live agent only from a snapshot that already passed the v1 validator.
    pub fn from_validated_snapshot_v1(snapshot: &ValidatedActiveInferenceAgentSnapshotV1) -> Self {
        let snapshot = snapshot.as_snapshot();
        Self {
            config: snapshot.config.clone(),
            belief: snapshot.belief.clone(),
            previous_state: snapshot.previous_state.clone(),
            last_action: snapshot.last_action,
            model: snapshot.model.clone(),
            free_energy_calc: snapshot.free_energy_calc.clone(),
            precision: snapshot.precision.clone(),
            efe_computer: snapshot.efe_computer.clone(),
            td_learner: snapshot.td_learner.clone(),
            last_fe_components: snapshot.last_fe_components.clone(),
            stats: snapshot.stats.clone(),
            timestamp: snapshot.timestamp,
            rng_state: snapshot.rng_state,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ActiveInferenceAgentConfig;

    fn config() -> ActiveInferenceAgentConfig {
        ActiveInferenceAgentConfig {
            state_dim: 2,
            obs_dim: 2,
            num_actions: 2,
            inference_iterations: 3,
            belief_learning_rate: 0.08,
            planning_horizon: 2,
            action_temperature: 0.7,
            enable_model_learning: true,
            enable_td_learning: true,
            ..Default::default()
        }
    }

    fn drive(agent: &mut ActiveInferenceAgent, start: usize, count: usize) {
        for i in start..start + count {
            let x = ((i * 37 % 101) as f64) / 100.0;
            let observation = Observation::new(vec![x, 1.0 - x], 0.8, "split-run");
            let _ = agent.perceive(&observation);
            let action = agent.select_action().action;
            let _ = agent.act(action);
            let outcome = Observation::new(
                vec![(x * 0.83 + 0.07).clamp(0.0, 1.0), (0.91 - x * 0.61).clamp(0.0, 1.0)],
                0.9,
                "split-run-outcome",
            );
            agent.learn_from_outcome(action, &outcome);
            if i % 7 == 0 {
                agent.end_episode();
            }
        }
    }

    fn snapshot_json(agent: &ActiveInferenceAgent) -> String {
        serde_json::to_string(&agent.snapshot_v1()).expect("serialize agent snapshot")
    }

    #[test]
    fn serialized_validated_restore_preserves_exact_future_agent_state() {
        let mut uninterrupted = ActiveInferenceAgent::new(config());
        uninterrupted.set_rng_seed(0xA11F_E001_D5EED);
        drive(&mut uninterrupted, 0, 13);

        let encoded = snapshot_json(&uninterrupted);
        let raw: ActiveInferenceAgentSnapshotV1 =
            serde_json::from_str(&encoded).expect("deserialize agent snapshot");
        let validated = raw.validate().expect("validate agent snapshot");
        let mut restored = ActiveInferenceAgent::from_validated_snapshot_v1(&validated);

        assert_eq!(snapshot_json(&uninterrupted), snapshot_json(&restored));
        for i in 13..45 {
            drive(&mut uninterrupted, i, 1);
            drive(&mut restored, i, 1);
            assert_eq!(
                snapshot_json(&uninterrupted),
                snapshot_json(&restored),
                "agent state diverged after restored continuation step {i}"
            );
        }
    }

    #[test]
    fn stochastic_action_stream_is_preserved_by_restore() {
        let mut uninterrupted = ActiveInferenceAgent::new(config());
        uninterrupted.set_rng_seed(0x51A7_E5EED);
        drive(&mut uninterrupted, 0, 5);

        let raw: ActiveInferenceAgentSnapshotV1 =
            serde_json::from_str(&snapshot_json(&uninterrupted)).unwrap();
        let validated = raw.validate().unwrap();
        let mut restored = ActiveInferenceAgent::from_validated_snapshot_v1(&validated);

        for _ in 0..128 {
            let a = uninterrupted.select_action();
            let b = restored.select_action();
            assert_eq!(a.action, b.action);
            assert_eq!(
                a.action_probabilities.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.action_probabilities.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        assert_eq!(snapshot_json(&uninterrupted), snapshot_json(&restored));
    }
}
