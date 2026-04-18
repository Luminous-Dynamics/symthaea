// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-humanoid
//!
//! HDC-LTC + FEP Active Inference bipedal humanoid control on the DMC Humanoid benchmark.
//!
//! Uses the full 16,384D `HdcLtcUnifiedNetwork` from `symthaea-core` as the temporal
//! dynamics engine, with `symthaea-fep` providing precision-weighted Active Inference
//! modulation. Multi-rate architecture: 40Hz physics + 20Hz training + 10Hz cognitive tick.
//!
//! ## Architecture
//!
//! ```text
//! Sensors (72D) → HumanoidHdcEncoder → ContinuousHV(16384D)
//!                                            ↓
//!                               HdcLtcUnifiedNetwork (3×8 neurons, evolve_closed_form)
//!                                            ↓
//!                               HumanoidController (output projection 16384→21)
//!                                            ↓
//!                               HumanoidCommand [21 joint torques, all tanh]
//!                                            ↓
//!                               MuJoCo Physics Step (contact dynamics)
//!
//! Every 4th physics step (10Hz):
//!   ActiveInferenceHumanoidAgent modulates τ, learning rate, prior precision
//!   based on balance priors (head_height, uprightness, angular_momentum)
//! ```
//!
//! ## Tasks (DMC Standard)
//!
//! - **Stand**: Maintain upright posture (head_height > 1.4m, uprightness > 0.9)
//! - **Walk**: Stand + move forward at ~1 m/s
//! - **Run**: Stand + move forward at ~10 m/s
//!
//! ## Perturbation Crucibles
//!
//! - **Chest Shove**: 500N push at step 300
//! - **Ice Floor**: Friction drops to 0.2 at step 200
//! - **Backpack**: +30% torso mass at step 100
//! - **Phantom Limb**: Right knee actuator failure (zero-shot gait adaptation)
//!
//! ## Features
//!
//! - `viewer` — Interactive MuJoCo 3D viewer

#![deny(unsafe_code)]
#![allow(clippy::needless_range_loop, clippy::io_other_error)]

pub mod benchmarks;
pub mod controller;
pub mod embodiment;
pub mod encoder;
pub mod fep_agent;
pub mod gait;
pub mod morphology;
pub mod perturbations;
pub mod reward;
pub mod simulator;
pub mod training;
pub mod transfer;
pub mod types;

pub use controller::{ControllerCheckpoint, HumanoidController};
pub use embodiment::{HumanoidEmbodiment, HumanoidFallbackStage};
pub use encoder::HumanoidHdcEncoder;
pub use fep_agent::{ActiveInferenceHumanoidAgent, HumanoidFepConfig};
pub use gait::GaitAnalyzer;
pub use morphology::HumanoidMorphology;
pub use perturbations::{HumanoidPerturbation, PerturbationSchedule};
pub use reward::{clearance_reward, cot_efficiency_reward, episode_reward, standing_reward};
#[cfg(feature = "mujoco")]
pub use simulator::MuJoCoHumanoidSimulator;
pub use simulator::{HumanoidPhysicsSimulator, SimpleHumanoidSimulator};
pub use training::HumanoidTrainer;
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn humanoid_command_zero() {
        let cmd = HumanoidCommand::zero();
        assert_eq!(cmd.torques.len(), NUM_ACTUATORS);
        assert!(cmd.torques.iter().all(|&t| t == 0.0));
    }

    #[test]
    fn joint_names_count() {
        assert_eq!(JOINT_NAMES.len(), NUM_ACTUATORS);
    }
}
