// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bevy resources for the vehicle demo.

use bevy::prelude::*;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_vehicle::encoder::VehicleHdcEncoder;
use symthaea_vehicle::simulator::BicycleModelSimulator;
use symtropy_consciousness_physics::safety::SafetyTier;
use symtropy_physics::body::BodyHandle;
use symtropy_robotics_bridge::agent::RoboticAgent;
use symtropy_robotics_bridge::platform::PlatformType;

use crate::controller::StanleyController;
use crate::ice_patch::IceField;

/// All vehicle-related state.
#[derive(Resource)]
pub struct VehicleResources {
    pub simulator: BicycleModelSimulator,
    pub encoder: VehicleHdcEncoder,
    pub controller: StanleyController,
    pub robot_agent: RoboticAgent,

    pub current_phi: f64,
    pub current_safety: SafetyTier,
    pub current_motor_gain: f64,
    pub last_prediction_error: f32,
    pub last_perception: Option<ContinuousHV>,

    pub current_friction: f64,
    pub current_ice_intensity: f64,
    pub current_gust: [f64; 2],
    pub last_throttle: f32,
    pub last_brake: f32,
    pub last_steering: f32,
}

impl VehicleResources {
    pub fn new() -> Self {
        let genesis = GenesisSeed::from_phrase("symtropy-vehicle-demo");
        let robot_agent = RoboticAgent::new(BodyHandle(0), PlatformType::Vehicle, "Bicycle-Phi");

        Self {
            simulator: BicycleModelSimulator::new(),
            encoder: VehicleHdcEncoder::new(&genesis, 32),
            controller: StanleyController::default(),
            robot_agent,
            current_phi: 0.8,
            current_safety: SafetyTier::Green,
            current_motor_gain: 1.0,
            last_prediction_error: 0.0,
            last_perception: None,
            current_friction: 1.0,
            current_ice_intensity: 0.0,
            current_gust: [0.0; 2],
            last_throttle: 0.0,
            last_brake: 0.0,
            last_steering: 0.0,
        }
    }
}

/// Figure-8 waypoint schedule in the XY plane (Z is always 0, this is a car).
#[derive(Resource)]
pub struct WaypointPath {
    pub waypoints: Vec<[f64; 2]>,
    pub current_index: usize,
    pub laps_completed: u32,
}

impl Default for WaypointPath {
    fn default() -> Self {
        // Figure-8 made of 16 points at ~20m radius.
        let mut waypoints = Vec::with_capacity(16);
        let r = 18.0_f64;
        for i in 0..8 {
            let theta = (i as f64) * std::f64::consts::FRAC_PI_4;
            // Upper lobe centered at (0, +r)
            waypoints.push([r * theta.cos(), r + r * theta.sin()]);
        }
        for i in 0..8 {
            let theta = (i as f64) * std::f64::consts::FRAC_PI_4;
            // Lower lobe centered at (0, -r), traversed in the opposite direction
            waypoints.push([-r * theta.cos(), -r - r * theta.sin()]);
        }
        Self {
            waypoints,
            current_index: 0,
            laps_completed: 0,
        }
    }
}

impl WaypointPath {
    pub fn current(&self) -> [f64; 2] {
        self.waypoints[self.current_index]
    }

    pub fn advance_if_reached(&mut self, pos: [f64; 2], tolerance: f64) -> bool {
        let tgt = self.current();
        let dx = tgt[0] - pos[0];
        let dy = tgt[1] - pos[1];
        if (dx * dx + dy * dy).sqrt() < tolerance {
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

/// Ice-patch disturbance field.
#[derive(Resource)]
pub struct Ice {
    pub field: IceField,
}

impl Default for Ice {
    fn default() -> Self {
        Self {
            field: IceField::default_layout(),
        }
    }
}

/// Elapsed simulation time.
#[derive(Resource, Default)]
pub struct SimTime {
    pub elapsed: f64,
}
