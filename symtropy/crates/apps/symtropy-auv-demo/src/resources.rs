// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bevy resources for the AUV demo.

use bevy::prelude::*;
use symthaea_auv::encoder::AuvHdcEncoder;
use symthaea_auv::simulator::SimpleAuvSimulator;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symtropy_consciousness_physics::safety::SafetyTier;
use symtropy_physics::body::BodyHandle;
use symtropy_robotics_bridge::agent::RoboticAgent;
use symtropy_robotics_bridge::platform::PlatformType;

use crate::controller::WaypointController;
use crate::current::UnderwaterCurrent;

#[derive(Resource)]
pub struct AuvResources {
    pub simulator: SimpleAuvSimulator,
    pub encoder: AuvHdcEncoder,
    pub controller: WaypointController,
    pub robot_agent: RoboticAgent,

    pub current_phi: f64,
    pub current_safety: SafetyTier,
    pub current_motor_gain: f64,
    pub last_prediction_error: f32,
    pub last_perception: Option<ContinuousHV>,

    pub last_current_force: [f64; 3],
    pub last_current_intensity: f64,
    pub last_thruster_effort: f32,
}

impl AuvResources {
    pub fn new() -> Self {
        let genesis = GenesisSeed::from_phrase("symtropy-auv-demo");
        let robot_agent = RoboticAgent::new(BodyHandle(0), PlatformType::Auv, "REMUS-Phi");

        Self {
            simulator: SimpleAuvSimulator::new(),
            encoder: AuvHdcEncoder::new(&genesis, 32),
            controller: WaypointController::default(),
            robot_agent,
            current_phi: 0.8,
            current_safety: SafetyTier::Green,
            current_motor_gain: 1.0,
            last_prediction_error: 0.0,
            last_perception: None,
            last_current_force: [0.0; 3],
            last_current_intensity: 0.0,
            last_thruster_effort: 0.0,
        }
    }
}

/// 3D waypoint path — rectangular patrol at varying depths (0 = surface,
/// positive = deeper in AUV depth convention).
#[derive(Resource)]
pub struct WaypointPath {
    pub waypoints: Vec<[f64; 3]>,
    pub current_index: usize,
    pub laps_completed: u32,
}

impl Default for WaypointPath {
    fn default() -> Self {
        // Patrol box at 4 depths: 6m, 12m (thermocline), 18m, 12m (back up).
        Self {
            waypoints: vec![
                [10.0, 10.0, 6.0],
                [10.0, -10.0, 12.0],
                [-10.0, -10.0, 18.0],
                [-10.0, 10.0, 12.0],
            ],
            current_index: 0,
            laps_completed: 0,
        }
    }
}

impl WaypointPath {
    pub fn current(&self) -> [f64; 3] {
        self.waypoints[self.current_index]
    }

    pub fn advance_if_reached(&mut self, pos: [f64; 3], depth: f64, tol: f64) -> bool {
        let tgt = self.current();
        let dx = tgt[0] - pos[0];
        let dy = tgt[1] - pos[1];
        let dz = tgt[2] - depth;
        if (dx * dx + dy * dy + dz * dz).sqrt() < tol {
            self.current_index = (self.current_index + 1) % self.waypoints.len();
            if self.current_index == 0 {
                self.laps_completed += 1;
            }
            true
        } else {
            false
        }
    }
}

#[derive(Resource, Default)]
pub struct Current {
    pub source: UnderwaterCurrent,
}

#[derive(Resource, Default)]
pub struct SimTime {
    pub elapsed: f64,
}
