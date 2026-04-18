// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bevy resources for the quadruped demo.

use bevy::prelude::*;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_quadruped::encoder::QuadrupedHdcEncoder;
use symthaea_quadruped::simulator::SimpleQuadrupedSimulator;
use symthaea_quadruped::types::GaitType;
use symtropy_consciousness_physics::safety::SafetyTier;
use symtropy_physics::body::BodyHandle;
use symtropy_robotics_bridge::agent::RoboticAgent;
use symtropy_robotics_bridge::platform::PlatformType;

use crate::controller::StanceController;
use crate::terrain::TerrainField;

#[derive(Resource)]
pub struct QuadrupedResources {
    pub simulator: SimpleQuadrupedSimulator,
    pub encoder: QuadrupedHdcEncoder,
    pub controller: StanceController,
    pub robot_agent: RoboticAgent,

    pub current_phi: f64,
    pub current_safety: SafetyTier,
    pub current_gait: GaitType,
    pub last_prediction_error: f32,
    pub last_perception: Option<ContinuousHV>,

    pub last_terrain_roughness: f64,
    pub last_effort: f32,
}

impl QuadrupedResources {
    pub fn new() -> Self {
        let genesis = GenesisSeed::from_phrase("symtropy-quadruped-demo");
        let robot_agent = RoboticAgent::new(BodyHandle(0), PlatformType::Quadruped, "Spot-Phi");

        Self {
            simulator: SimpleQuadrupedSimulator::new(),
            encoder: QuadrupedHdcEncoder::new(&genesis, 32),
            controller: StanceController::default(),
            robot_agent,
            current_phi: 0.8,
            current_safety: SafetyTier::Green,
            current_gait: GaitType::Trot,
            last_prediction_error: 0.0,
            last_perception: None,
            last_terrain_roughness: 0.0,
            last_effort: 0.0,
        }
    }
}

#[derive(Resource, Default)]
pub struct Terrain {
    pub field: TerrainField,
}

#[derive(Resource, Default)]
pub struct SimTime {
    pub elapsed: f64,
}
