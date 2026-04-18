// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bevy resources for the surgical demo.

use bevy::prelude::*;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_surgical::encoder::SurgicalHdcEncoder;
use symthaea_surgical::simulator::SimpleSurgicalSimulator;
use symthaea_surgical::types::SurgicalSafetyLevel;
use symtropy_consciousness_physics::safety::SafetyTier;
use symtropy_physics::body::BodyHandle;
use symtropy_robotics_bridge::agent::RoboticAgent;
use symtropy_robotics_bridge::platform::PlatformType;

use crate::controller::CauteryProcedureController;

#[derive(Resource)]
pub struct SurgicalResources {
    pub simulator: SimpleSurgicalSimulator,
    pub encoder: SurgicalHdcEncoder,
    pub controller: CauteryProcedureController,
    pub robot_agent: RoboticAgent,

    pub current_phi: f64,
    pub current_safety: SafetyTier,
    pub current_level: SurgicalSafetyLevel,
    pub last_prediction_error: f32,
    pub last_perception: Option<ContinuousHV>,

    pub last_effort: f32,
    pub last_cautery: f32,
    pub last_jaw: f32,
}

impl SurgicalResources {
    pub fn new() -> Self {
        let genesis = GenesisSeed::from_phrase("symtropy-surgical-demo");
        let robot_agent = RoboticAgent::new(BodyHandle(0), PlatformType::Surgical, "DaVinci-Phi");

        Self {
            simulator: SimpleSurgicalSimulator::new(),
            encoder: SurgicalHdcEncoder::new(&genesis, 32),
            controller: CauteryProcedureController::default(),
            robot_agent,
            current_phi: 0.8,
            current_safety: SafetyTier::Green,
            current_level: SurgicalSafetyLevel::FullControl,
            last_prediction_error: 0.0,
            last_perception: None,
            last_effort: 0.0,
            last_cautery: 0.0,
            last_jaw: 0.0,
        }
    }
}

#[derive(Resource, Default)]
pub struct SimTime {
    pub elapsed: f64,
}
