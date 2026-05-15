// Copyright (c) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Per-joint motor-command planning.
//!
//! The [`RoboticAgent`](crate::RoboticAgent) computes a scalar motor-authority
//! gain each tick (from its consciousness level and NRC safety tier). That
//! gain is a *cap*, not a command: for real embodied control, each actuator
//! needs its own signal.
//!
//! A [`MotorPlanner`] consumes the observation + the safety-gated gain and
//! produces per-actuator commands matching the platform's actuator count.
//! Platforms (humanoid, quadrotor, manipulator, …) provide their own
//! planners backed by their specific kinematics / dynamics; this crate
//! ships one trivial baseline: [`UniformGainPlanner`].
//!
//! # Example: humanoid with uniform-gain baseline
//!
//! ```no_run
//! use symtropy_robotics_bridge::{PlatformType, RoboticAgent};
//! use symtropy_physics::body::BodyHandle;
//!
//! let mut agent = RoboticAgent::new(BodyHandle(0), PlatformType::Humanoid, "Bot");
//! let commands = agent.tick_motor_commands(&[0.0; 4], 0.1);
//! assert_eq!(commands.len(), 21); // humanoid has 21 DOF
//! ```
//!
//! # Wiring a Symthaea platform
//!
//! Every `symthaea-<platform>` crate (humanoid, quadruped, manipulator, …)
//! ships a type that implements Symthaea's `EmbodimentBridge` trait. To
//! drive it from Symtropy, write a small [`MotorPlanner`] adapter that calls
//! `EmbodimentBridge::step(thought_hv, dt, phi)` and pulls per-joint
//! commands out of the resulting `EmbodimentResult`. Keeping the adapter in
//! user / platform code avoids pulling `symthaea-core` into this crate's
//! public dep graph.

/// Plan per-actuator motor commands for one tick.
///
/// `observation` is the agent's current sensor reading (as passed to
/// [`RoboticAgent::tick`](crate::RoboticAgent::tick)). `gain` is the
/// NRC-tier safety gain in `[0.0, 1.0]`. `num_actuators` is the target
/// platform's actuator count (see [`PlatformType::num_actuators`](crate::PlatformType::num_actuators)).
///
/// Returned vector **must** have length `num_actuators` — callers will
/// index it by joint id.
pub trait MotorPlanner {
    fn plan(&mut self, observation: &[f64], gain: f64, num_actuators: usize) -> Vec<f64>;
}

/// Trivial planner: emit the scalar gain uniformly to every actuator.
///
/// Useful as a baseline or for platforms where per-joint intelligence hasn't
/// been wired yet. NOT a realistic controller — a humanoid walking on this
/// planner will drive every joint with the same torque and fall over.
#[derive(Debug, Default, Clone, Copy)]
pub struct UniformGainPlanner;

impl MotorPlanner for UniformGainPlanner {
    fn plan(&mut self, _observation: &[f64], gain: f64, num_actuators: usize) -> Vec<f64> {
        vec![gain; num_actuators]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_gain_planner_produces_correct_length() {
        let mut p = UniformGainPlanner;
        let cmds = p.plan(&[], 0.5, 21);
        assert_eq!(cmds.len(), 21);
    }

    #[test]
    fn uniform_gain_planner_broadcasts_gain_to_every_joint() {
        let mut p = UniformGainPlanner;
        let cmds = p.plan(&[0.1, 0.2, 0.3], 0.75, 4);
        assert_eq!(cmds, vec![0.75; 4]);
    }

    #[test]
    fn uniform_gain_planner_handles_zero_gain() {
        let mut p = UniformGainPlanner;
        let cmds = p.plan(&[], 0.0, 8);
        assert_eq!(cmds, vec![0.0; 8]);
    }

    #[test]
    fn uniform_gain_planner_handles_zero_actuators() {
        let mut p = UniformGainPlanner;
        let cmds = p.plan(&[], 0.5, 0);
        assert!(cmds.is_empty());
    }
}
