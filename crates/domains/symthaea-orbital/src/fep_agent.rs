use crate::types::{NUM_STATE_CHANNELS, OrbitalState};
use symthaea_fep::Observation;

pub struct OrbitalFepResult {
    pub tau_factor: f32,
    pub free_energy: f64,
}

/// See SYMTHAEA_CLASSIC_PLATFORMS_FEP_HONESTY_2026-07-09.md -- same fix
/// pattern as the quadruped reference: `tick()` now runs a genuine
/// perception step instead of a hand-rolled heuristic, and `obs_dim` matches
/// `NUM_STATE_CHANNELS` (was mismatched at 7 against 33). `tau_factor`'s
/// angular-rate gating is domain-specific logic kept as-is.
pub struct ActiveInferenceOrbitalAgent {
    agent: symthaea_fep::ActiveInferenceAgent,
    tick_count: u64,
}
impl ActiveInferenceOrbitalAgent {
    pub fn new() -> Self {
        let c = symthaea_fep::ActiveInferenceAgentConfig {
            state_dim: 8,
            obs_dim: NUM_STATE_CHANNELS,
            num_actions: 1, // unused: no select_action/act in this wiring
            belief_learning_rate: 0.08,
            planning_horizon: 1,
            action_temperature: 0.5,
            ..symthaea_fep::ActiveInferenceAgentConfig::default()
        };
        Self {
            agent: symthaea_fep::ActiveInferenceAgent::new(c),
            tick_count: 0,
        }
    }
    pub fn tick(&mut self, s: &OrbitalState) -> OrbitalFepResult {
        self.tick_count += 1;
        let values: Vec<f64> = s.to_channels().iter().map(|v| *v as f64).collect();
        let observation = Observation {
            values,
            precision: 1.0,
            timestamp: self.tick_count,
            modality: "orbital".to_string(),
        };
        self.agent.perceive(&observation);
        let r: f64 = s.spacecraft_angular_velocity.iter().map(|v| v.abs()).sum();
        OrbitalFepResult {
            tau_factor: if r > 0.01 { 0.85 } else { 1.0 },
            free_energy: self.agent.current_free_energy(),
        }
    }
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}
impl Default for ActiveInferenceOrbitalAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fep_home() {
        let mut a = ActiveInferenceOrbitalAgent::new();
        let r = a.tick(&OrbitalState::stowed());
        assert!(r.free_energy.is_finite());
    }

    #[test]
    fn test_fep_free_energy_is_not_the_old_fabricated_heuristic() {
        let mut a = ActiveInferenceOrbitalAgent::new();
        let state = OrbitalState::stowed();
        let fake_fe: f64 = state
            .spacecraft_angular_velocity
            .iter()
            .map(|v| v.abs())
            .sum::<f64>()
            * 100.0;
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
        let mut a = ActiveInferenceOrbitalAgent::new();
        let belief_before = a.agent.belief.mean.clone();
        let mut tumbling_state = OrbitalState::stowed();
        tumbling_state.spacecraft_angular_velocity = [5.0, 5.0, 5.0];
        for _ in 0..10 {
            a.tick(&tumbling_state);
        }
        assert_ne!(
            belief_before, a.agent.belief.mean,
            "belief must change after repeated perception of an extreme observation"
        );
    }
}
