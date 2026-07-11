// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::types::{NUM_STATE_CHANNELS, SurgicalState};
use symthaea_fep::Observation;

pub struct SurgicalFepResult {
    pub tau_factor: f32,
    pub learning_rate_factor: f32,
    pub free_energy: f64,
    pub anomaly_detected: bool,
}

/// See SYMTHAEA_CLASSIC_PLATFORMS_FEP_HONESTY_2026-07-09.md -- same fix
/// pattern as the quadruped reference: `tick()` now runs a genuine
/// perception step instead of a hand-rolled heuristic (the `agent` field was
/// even explicitly marked `#[allow(dead_code)]` before this fix), and
/// `obs_dim` matches `NUM_STATE_CHANNELS` (was mismatched at 6 against 24).
/// `anomaly_detected` (force/proximity/compliance thresholds) is a
/// legitimate, already-tested domain signal kept as-is; only
/// `tau_factor`/`learning_rate_factor`'s secondary thresholds depended on
/// the old fabricated free energy, so they now use a plainly-named local
/// deviation estimate instead of masquerading as FEP output.
pub struct ActiveInferenceSurgicalAgent {
    agent: symthaea_fep::ActiveInferenceAgent,
    tick_count: u64,
}
impl ActiveInferenceSurgicalAgent {
    pub fn new() -> Self {
        let c = symthaea_fep::ActiveInferenceAgentConfig {
            state_dim: 8,
            obs_dim: NUM_STATE_CHANNELS,
            num_actions: 1, // unused: no select_action/act in this wiring
            belief_learning_rate: 0.05,
            planning_horizon: 1,
            action_temperature: 0.3,
            ..symthaea_fep::ActiveInferenceAgentConfig::default()
        };
        Self {
            agent: symthaea_fep::ActiveInferenceAgent::new(c),
            tick_count: 0,
        }
    }
    pub fn tick(&mut self, s: &SurgicalState) -> SurgicalFepResult {
        self.tick_count += 1;
        let values: Vec<f64> = s.to_channels().iter().map(|v| *v as f64).collect();
        let observation = Observation {
            values,
            precision: 1.0,
            timestamp: self.tick_count,
            modality: "surgical".to_string(),
        };
        self.agent.perceive(&observation);

        let f = s.force_magnitude();
        let cd = s.critical_structure_distance;
        let tc = s.trocar_compliance;
        // Local deviation estimate for the tau/lr gates below -- kept
        // separate from the agent's real free_energy field.
        let deviation_estimate = f * 2.0 + (20.0 - cd).max(0.0) + tc * 50.0;
        let anom = f > 3.0 || cd < 5.0 || tc > 0.3;
        let tau = if anom {
            0.7
        } else if deviation_estimate < 5.0 {
            1.2
        } else {
            1.0
        };
        let lr = if anom {
            2.0
        } else if deviation_estimate < 2.0 {
            0.5
        } else {
            1.0
        };
        SurgicalFepResult {
            tau_factor: tau,
            learning_rate_factor: lr,
            free_energy: self.agent.current_free_energy(),
            anomaly_detected: anom,
        }
    }
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}
impl Default for ActiveInferenceSurgicalAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::{SimpleSurgicalSimulator, SurgicalPhysicsSimulator};
    use crate::types::{SurgicalCommand, SurgicalConfig};

    #[test]
    fn test_anomaly_trips_on_real_proximity() {
        // The anomaly detector now keys off geometric state: place a
        // critical structure directly on the tip's sweep path and drive
        // toward it — the tip-to-structure distance must fall below the
        // 5 mm anomaly threshold and trip the detector.
        let mut cfg = SurgicalConfig::default();
        cfg.rcm_stiffness = 0.0; // let the sweep reach the structure
        // Tip position at q1 ≈ 0.12 rad along the +x sweep from home.
        cfg.critical_structure = [71.0, -14.4, -267.5];
        let mut sim = SimpleSurgicalSimulator::with_config(cfg);
        let mut agent = ActiveInferenceSurgicalAgent::new();
        let mut cmd = SurgicalCommand::zero();
        cmd.joint_torques[0] = 0.5;
        let start = {
            sim.step(&SurgicalCommand::zero(), 0.001);
            sim.state().critical_structure_distance
        };
        // Approach until the geometric distance itself crosses the 5 mm
        // proximity threshold (asserted directly, so this can't pass via
        // the trocar-breach channel instead), then confirm the detector
        // trips on that state.
        let mut reached = false;
        for _ in 0..2000 {
            sim.step(&cmd, 0.001);
            if sim.state().critical_structure_distance < 4.9 {
                reached = true;
                break;
            }
        }
        assert!(
            reached,
            "tip must approach within 5 mm of the structure (start {start:.1} mm, end {:.1} mm)",
            sim.state().critical_structure_distance
        );
        assert!(sim.state().critical_structure_distance < start);
        let r = agent.tick(sim.state());
        assert!(
            r.anomaly_detected,
            "anomaly must trip at {:.2} mm proximity",
            sim.state().critical_structure_distance
        );
    }

    #[test]
    fn test_no_anomaly_at_rest_far_from_structure() {
        // At home, far from the (default) structure and with the RCM spring
        // holding the port, the detector must stay quiet.
        let mut sim = SimpleSurgicalSimulator::new();
        let mut agent = ActiveInferenceSurgicalAgent::new();
        for _ in 0..200 {
            sim.step(&SurgicalCommand::zero(), 0.001);
        }
        let r = agent.tick(sim.state());
        assert!(
            !r.anomaly_detected,
            "resting far from the structure must not be anomalous (fe={})",
            r.free_energy
        );
    }

    #[test]
    fn test_fep_free_energy_is_not_the_old_fabricated_heuristic() {
        let mut agent = ActiveInferenceSurgicalAgent::new();
        let state = SurgicalState::home();
        let f = state.force_magnitude();
        let cd = state.critical_structure_distance;
        let tc = state.trocar_compliance;
        let fake_fe = f * 2.0 + (20.0 - cd).max(0.0) + tc * 50.0;
        let r = agent.tick(&state);
        assert!(
            (r.free_energy - fake_fe).abs() > 1e-6,
            "free_energy ({}) must not equal the old fabricated heuristic ({})",
            r.free_energy,
            fake_fe
        );
    }

    #[test]
    fn test_fep_perceiving_changes_agent_belief() {
        let mut agent = ActiveInferenceSurgicalAgent::new();
        let belief_before = agent.agent.belief.mean.clone();
        let mut strained_state = SurgicalState::home();
        strained_state.critical_structure_distance = 0.5;
        strained_state.trocar_compliance = 0.9;
        for _ in 0..10 {
            agent.tick(&strained_state);
        }
        assert_ne!(
            belief_before, agent.agent.belief.mean,
            "belief must change after repeated perception of an extreme observation"
        );
    }
}
