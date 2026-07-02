// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bevy resources for the humanoid demo.

use bevy::prelude::*;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_humanoid::encoder::HumanoidHdcEncoder;
use symthaea_humanoid::simulator::SimpleHumanoidSimulator;
use symtropy_consciousness_physics::safety::SafetyTier;
use symtropy_physics::body::BodyHandle;
use symtropy_robotics_bridge::agent::RoboticAgent;
use symtropy_robotics_bridge::platform::PlatformType;

use crate::controller::BalanceController;
use crate::push::PushSource;

#[derive(Resource)]
pub struct HumanoidResources {
    pub simulator: SimpleHumanoidSimulator,
    pub encoder: HumanoidHdcEncoder,
    pub controller: BalanceController,
    pub robot_agent: RoboticAgent,

    pub current_phi: f64,
    pub current_safety: SafetyTier,
    pub current_motor_gain: f64,
    pub last_prediction_error: f32,
    pub last_perception: Option<ContinuousHV>,

    pub last_push_force: [f64; 3],
    pub last_push_intensity: f64,
    pub last_effort: f32,
}

impl HumanoidResources {
    pub fn new() -> Self {
        let genesis = GenesisSeed::from_phrase("symtropy-humanoid-demo");
        let robot_agent = RoboticAgent::new(BodyHandle(0), PlatformType::Humanoid, "DMC21-Phi");

        Self {
            simulator: SimpleHumanoidSimulator::new(),
            encoder: HumanoidHdcEncoder::new(&genesis, 32),
            controller: BalanceController::default(),
            robot_agent,
            current_phi: 0.8,
            current_safety: SafetyTier::Green,
            current_motor_gain: 1.0,
            last_prediction_error: 0.0,
            last_perception: None,
            last_push_force: [0.0; 3],
            last_push_intensity: 0.0,
            last_effort: 0.0,
        }
    }
}

#[derive(Resource, Default)]
pub struct Push {
    pub source: PushSource,
}

#[derive(Resource, Default)]
pub struct SimTime {
    pub elapsed: f64,
}
