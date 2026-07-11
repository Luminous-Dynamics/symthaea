use crate::types::{NUM_STATE_CHANNELS, QuadrupedState};
use symthaea_fep::Observation;

pub struct QuadrupedFepResult {
    pub tau_factor: f32,
    pub free_energy: f64,
}

/// See SYMTHAEA_CLASSIC_PLATFORMS_FEP_HONESTY_2026-07-09.md -- reference fix
/// for the FEP-theater bug found in 6 of the 10 classic robotics platforms
/// (this crate had a real trainer already wired to `tau_factor`/`free_energy`,
/// but the values themselves were fabricated: `tick()` never called
/// `perceive()`/`act()`/`learn_from_outcome()` on the wrapped agent, and
/// `obs_dim` (12) didn't match the real 37-channel state). `tick()` now runs
/// a genuine perception step against the full state; `tau_factor`'s posture-
/// deviation gating is domain-specific logic kept as-is (not derived from the
/// old fabricated free energy), matching the precedent set for biota/clime in
/// the unaudited-platforms review.
pub struct ActiveInferenceQuadrupedAgent {
    agent: symthaea_fep::ActiveInferenceAgent,
    tick_count: u64,
}
impl ActiveInferenceQuadrupedAgent {
    pub fn new() -> Self {
        let c = symthaea_fep::ActiveInferenceAgentConfig {
            state_dim: 8,
            obs_dim: NUM_STATE_CHANNELS,
            num_actions: 1, // unused: no select_action/act in this wiring
            belief_learning_rate: 0.1,
            planning_horizon: 1,
            action_temperature: 0.5,
            ..symthaea_fep::ActiveInferenceAgentConfig::default()
        };
        Self {
            agent: symthaea_fep::ActiveInferenceAgent::new(c),
            tick_count: 0,
        }
    }
    pub fn tick(&mut self, s: &QuadrupedState) -> QuadrupedFepResult {
        self.tick_count += 1;
        let values: Vec<f64> = s.to_channels().iter().map(|v| *v as f64).collect();
        let observation = Observation {
            values,
            precision: 1.0,
            timestamp: self.tick_count,
            modality: "quadruped".to_string(),
        };
        self.agent.perceive(&observation);
        let he = (0.35 - s.height()).abs();
        QuadrupedFepResult {
            tau_factor: if he > 0.1 { 0.85 } else { 1.0 },
            free_energy: self.agent.current_free_energy(),
        }
    }
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}
impl Default for ActiveInferenceQuadrupedAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fep_home() {
        let mut a = ActiveInferenceQuadrupedAgent::new();
        let r = a.tick(&QuadrupedState::standing());
        assert!(r.free_energy.is_finite());
    }

    #[test]
    fn test_fep_free_energy_is_not_the_old_fabricated_heuristic() {
        // Regression: the old fake implementation was exactly
        // `(0.35 - height).abs() * 10.0 + sum(|angular_velocity|) * 5.0`.
        let mut a = ActiveInferenceQuadrupedAgent::new();
        let state = QuadrupedState::standing();
        let he = (0.35 - state.height()).abs();
        let ar: f64 = state.base_angular_velocity.iter().map(|v| v.abs()).sum();
        let fake_fe = he * 10.0 + ar * 5.0;
        let r = a.tick(&state);
        assert!(
            (r.free_energy - fake_fe).abs() > 1e-6,
            "free_energy ({}) must not equal the old fabricated heuristic ({})",
            r.free_energy,
            fake_fe
        );
    }

    #[test]
    fn test_fep_perceiving_changes_agent_belief() {
        let mut a = ActiveInferenceQuadrupedAgent::new();
        let belief_before = a.agent.belief.mean.clone();
        let mut fallen_state = QuadrupedState::standing();
        fallen_state.base_position[2] = 0.0;
        for v in &mut fallen_state.base_angular_velocity {
            *v = 5.0;
        }
        for _ in 0..10 {
            a.tick(&fallen_state);
        }
        assert_ne!(
            belief_before, a.agent.belief.mean,
            "belief must change after repeated perception of an extreme observation"
        );
    }
}
