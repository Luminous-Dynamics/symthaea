// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bevy resources for the exoskeleton demo.

use bevy::prelude::*;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_exoskeleton::encoder::ExoskeletonHdcEncoder;
use symthaea_exoskeleton::simulator::SimpleExoskeletonSimulator;
use symthaea_exoskeleton::types::AssistanceMode;
use symtropy_consciousness_physics::safety::SafetyTier;
use symtropy_physics::body::BodyHandle;
use symtropy_robotics_bridge::agent::RoboticAgent;
use symtropy_robotics_bridge::platform::PlatformType;

use crate::controller::AssistiveController;

#[derive(Resource)]
pub struct ExoskeletonResources {
    pub simulator: SimpleExoskeletonSimulator,
    pub encoder: ExoskeletonHdcEncoder,
    pub controller: AssistiveController,
    pub robot_agent: RoboticAgent,

    pub current_phi: f64,
    pub current_safety: SafetyTier,
    pub current_mode: AssistanceMode,
    pub last_prediction_error: f32,
    pub last_perception: Option<ContinuousHV>,

    pub last_exo_effort: f32,
}

impl ExoskeletonResources {
    pub fn new() -> Self {
        let genesis = GenesisSeed::from_phrase("symtropy-exoskeleton-demo");
        let robot_agent =
            RoboticAgent::new(BodyHandle(0), PlatformType::Exoskeleton, "FullFrame-Phi");

        Self {
            simulator: SimpleExoskeletonSimulator::new(),
            encoder: ExoskeletonHdcEncoder::new(&genesis, 32),
            controller: AssistiveController::default(),
            robot_agent,
            current_phi: 0.8,
            current_safety: SafetyTier::Green,
            current_mode: AssistanceMode::Predictive,
            last_prediction_error: 0.0,
            last_perception: None,
            last_exo_effort: 0.0,
        }
    }
}

#[derive(Resource, Default)]
pub struct SimTime {
    pub elapsed: f64,
}
