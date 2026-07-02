// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bevy resources wrapping the quadrotor's physics, consciousness, and task state.

use bevy::prelude::*;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_multirotor::encoder::QuadrotorHdcEncoder;
use symthaea_multirotor::simulator::SimplePhysicsSimulator;
use symtropy_consciousness_physics::safety::SafetyTier;
use symtropy_physics::body::BodyHandle;
use symtropy_robotics_bridge::agent::RoboticAgent;
use symtropy_robotics_bridge::platform::PlatformType;

use crate::controller::WaypointController;
use crate::wind::WindGustSource;

/// All quadrotor-related state. One resource keeps the plugin system list short.
#[derive(Resource)]
pub struct QuadrotorState {
    pub simulator: SimplePhysicsSimulator,
    pub encoder: QuadrotorHdcEncoder,
    pub controller: WaypointController,
    pub robot_agent: RoboticAgent,

    pub current_phi: f64,
    pub current_safety: SafetyTier,
    pub current_motor_gain: f64,
    pub last_prediction_error: f32,
    pub last_perception: Option<ContinuousHV>,

    pub last_gust_force: [f64; 3],
    pub last_gust_intensity: f64,
    pub last_thrust: f32,
}

impl QuadrotorState {
    pub fn new() -> Self {
        let genesis = GenesisSeed::from_phrase("symtropy-flight-demo");
        let robot_agent =
            RoboticAgent::new(BodyHandle(0), PlatformType::Quadrotor, "Crazyflie-Phi");

        Self {
            simulator: SimplePhysicsSimulator::new(),
            encoder: QuadrotorHdcEncoder::new(&genesis, 32),
            controller: WaypointController::default(),
            robot_agent,
            current_phi: 0.8,
            current_safety: SafetyTier::Green,
            current_motor_gain: 1.0,
            last_prediction_error: 0.0,
            last_perception: None,
            last_gust_force: [0.0; 3],
            last_gust_intensity: 0.0,
            last_thrust: 0.0,
        }
    }
}

/// Figure-8 waypoint schedule.
#[derive(Resource)]
pub struct WaypointPath {
    pub waypoints: Vec<[f64; 3]>,
    pub current_index: usize,
    pub cycles_completed: u32,
}

impl Default for WaypointPath {
    fn default() -> Self {
        // A simple figure-8 at ~1.5 m altitude.
        Self {
            waypoints: vec![
                [0.0, 0.0, 1.5],
                [1.5, 1.0, 1.5],
                [0.0, 2.0, 1.8],
                [-1.5, 1.0, 1.5],
                [0.0, 0.0, 1.5],
                [1.5, -1.0, 1.5],
                [0.0, -2.0, 1.2],
                [-1.5, -1.0, 1.5],
            ],
            current_index: 0,
            cycles_completed: 0,
        }
    }
}

impl WaypointPath {
    pub fn current(&self) -> [f64; 3] {
        self.waypoints[self.current_index]
    }

    pub fn advance_if_reached(&mut self, pos: [f64; 3], tolerance: f64) -> bool {
        let tgt = self.current();
        let dx = tgt[0] - pos[0];
        let dy = tgt[1] - pos[1];
        let dz = tgt[2] - pos[2];
        if (dx * dx + dy * dy + dz * dz).sqrt() < tolerance {
            self.current_index = (self.current_index + 1) % self.waypoints.len();
            if self.current_index == 0 {
                self.cycles_completed += 1;
            }
            true
        } else {
            false
        }
    }
}

/// Shared wind-gust disturbance.
#[derive(Resource, Default)]
pub struct Wind {
    pub source: WindGustSource,
}

/// Elapsed simulation time.
#[derive(Resource, Default)]
pub struct SimTime {
    pub elapsed: f64,
}
