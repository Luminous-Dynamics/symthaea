// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bevy resources for the orbital demo.

use bevy::prelude::*;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_orbital::encoder::OrbitalHdcEncoder;
use symthaea_orbital::simulator::SimpleOrbitalSimulator;
use symtropy_consciousness_physics::safety::SafetyTier;
use symtropy_physics::body::BodyHandle;
use symtropy_robotics_bridge::agent::RoboticAgent;
use symtropy_robotics_bridge::platform::PlatformType;

use crate::controller::DeploymentController;

#[derive(Resource)]
pub struct OrbitalResources {
    pub simulator: SimpleOrbitalSimulator,
    pub encoder: OrbitalHdcEncoder,
    pub controller: DeploymentController,
    pub robot_agent: RoboticAgent,

    pub current_phi: f64,
    pub current_safety: SafetyTier,
    pub current_motor_gain: f64,
    pub last_prediction_error: f32,
    pub last_perception: Option<ContinuousHV>,

    pub last_effort: f32,
    /// Visually-integrated spacecraft attitude (the simulator only exposes
    /// angular velocity; attitude is accumulated here for the renderer).
    pub spacecraft_attitude: Quat,
}

impl OrbitalResources {
    pub fn new() -> Self {
        let genesis = GenesisSeed::from_phrase("symtropy-orbital-demo");
        let robot_agent = RoboticAgent::new(BodyHandle(0), PlatformType::Orbital, "Canadarm-Phi");

        Self {
            simulator: SimpleOrbitalSimulator::new(),
            encoder: OrbitalHdcEncoder::new(&genesis, 32),
            controller: DeploymentController::default(),
            robot_agent,
            current_phi: 0.8,
            current_safety: SafetyTier::Green,
            current_motor_gain: 1.0,
            last_prediction_error: 0.0,
            last_perception: None,
            last_effort: 0.0,
            spacecraft_attitude: Quat::IDENTITY,
        }
    }
}

#[derive(Resource, Default)]
pub struct SimTime {
    pub elapsed: f64,
}
