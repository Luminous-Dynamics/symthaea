// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bevy resources for the helicopter demo.

use bevy::prelude::*;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_helicopter::encoder::HelicopterHdcEncoder;
use symthaea_helicopter::simulator::SimpleHelicopterSimulator;
use symthaea_helicopter::types::HelicopterCommand;
use symthaea_helicopter::wind_model::{WindConfig, WindModel};
use symtropy_consciousness_physics::safety::SafetyTier;
use symtropy_physics::body::BodyHandle;
use symtropy_robotics_bridge::agent::RoboticAgent;
use symtropy_robotics_bridge::platform::PlatformType;

use crate::controller::StationHoldController;

#[derive(Resource)]
pub struct HelicopterResources {
    pub simulator: SimpleHelicopterSimulator,
    pub encoder: HelicopterHdcEncoder,
    pub controller: StationHoldController,
    pub robot_agent: RoboticAgent,
    pub wind: WindModel,

    pub current_phi: f64,
    pub current_safety: SafetyTier,
    pub current_motor_gain: f64,
    pub last_prediction_error: f32,
    pub last_perception: Option<ContinuousHV>,

    pub last_wind_force: [f64; 3],
    pub last_wind_speed: f64,
    pub last_rotor_spin_angle: f32,
    pub last_collective: f32,
}

impl HelicopterResources {
    pub fn new() -> Self {
        let genesis = GenesisSeed::from_phrase("symtropy-helicopter-demo");
        let robot_agent = RoboticAgent::new(BodyHandle(0), PlatformType::Helicopter, "R44-Phi");

        Self {
            simulator: SimpleHelicopterSimulator::new(),
            encoder: HelicopterHdcEncoder::new(&genesis, 32),
            controller: StationHoldController::default(),
            robot_agent,
            wind: WindModel::new(WindConfig::moderate_wind()),
            current_phi: 0.8,
            current_safety: SafetyTier::Green,
            current_motor_gain: 1.0,
            last_prediction_error: 0.0,
            last_perception: None,
            last_wind_force: [0.0; 3],
            last_wind_speed: 0.0,
            last_rotor_spin_angle: 0.0,
            last_collective: HelicopterCommand::HOVER_COLLECTIVE,
        }
    }
}

#[derive(Resource, Default)]
pub struct SimTime {
    pub elapsed: f64,
}
