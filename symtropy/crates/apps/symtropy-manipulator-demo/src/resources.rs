// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bevy resources wrapping manipulator state for both arms.

use bevy::prelude::*;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_manipulator::{
    ManipulatorConfig, controller::ManipulatorController, encoder::ManipulatorHdcEncoder,
    kinematics::ManipulatorKinematics, simulator::SimpleManipulatorSimulator,
    workspace_safety::WorkspaceBoundary,
};
use symtropy_consciousness_physics::safety::SafetyTier;
use symtropy_physics::body::BodyHandle;
use symtropy_robotics_bridge::agent::RoboticAgent;
use symtropy_robotics_bridge::platform::PlatformType;

use crate::assembly_task::AssemblyTask;
use crate::grasp_object::GraspObject;
use crate::human_obstacle::HumanObstacle;
use crate::iso_comparison::IsoSsmController;

/// Phi-gradient arm state (left side of split-screen).
#[derive(Resource)]
pub struct PhiArmState {
    pub simulator: SimpleManipulatorSimulator,
    pub kinematics: ManipulatorKinematics,
    pub encoder: ManipulatorHdcEncoder,
    pub controller: ManipulatorController,
    pub workspace: WorkspaceBoundary,
    pub robot_agent: RoboticAgent,
    pub task: AssemblyTask,
    pub current_phi: f64,
    pub current_safety: SafetyTier,
    pub stiffness: f64,
    pub last_prediction_error: f32,
    /// Previous HDC perception for cosine dissimilarity PE computation.
    pub last_perception: Option<ContinuousHV>,
    pub cycles_completed: u32,
    pub total_cycle_time: f64,
    pub current_cycle_start: f64,
}

impl PhiArmState {
    pub fn new() -> Self {
        let genesis = GenesisSeed::from_phrase("symtropy-manipulator-demo-phi");
        let config = ManipulatorConfig::default();
        let robot_agent = RoboticAgent::new(BodyHandle(0), PlatformType::Manipulator, "Panda-Phi");

        Self {
            simulator: SimpleManipulatorSimulator::new(),
            kinematics: ManipulatorKinematics::default_7dof(),
            encoder: ManipulatorHdcEncoder::new(&genesis, 32),
            controller: ManipulatorController::new(&genesis, &config),
            workspace: WorkspaceBoundary::default(),
            robot_agent,
            task: AssemblyTask::new(),
            current_phi: 0.8,
            current_safety: SafetyTier::Green,
            stiffness: 1.0,
            last_prediction_error: 0.0,
            last_perception: None,
            cycles_completed: 0,
            total_cycle_time: 0.0,
            current_cycle_start: 0.0,
        }
    }

    pub fn mean_cycle_time(&self) -> f64 {
        if self.cycles_completed == 0 {
            0.0
        } else {
            self.total_cycle_time / self.cycles_completed as f64
        }
    }
}

/// ISO/TS 15066 SSM arm state (right side of split-screen).
#[derive(Resource)]
pub struct IsoArmState {
    pub simulator: SimpleManipulatorSimulator,
    pub kinematics: ManipulatorKinematics,
    pub task: AssemblyTask,
    pub ssm: IsoSsmController,
    pub is_stopped: bool,
    pub current_protective_distance: f64,
    pub cycles_completed: u32,
    pub total_cycle_time: f64,
    pub current_cycle_start: f64,
}

impl IsoArmState {
    pub fn new() -> Self {
        Self {
            simulator: SimpleManipulatorSimulator::new(),
            kinematics: ManipulatorKinematics::default_7dof(),
            task: AssemblyTask::new(),
            ssm: IsoSsmController::default(),
            is_stopped: false,
            current_protective_distance: 0.0,
            cycles_completed: 0,
            total_cycle_time: 0.0,
            current_cycle_start: 0.0,
        }
    }

    pub fn mean_cycle_time(&self) -> f64 {
        if self.cycles_completed == 0 {
            0.0
        } else {
            self.total_cycle_time / self.cycles_completed as f64
        }
    }
}

/// Shared human obstacle (same trajectory for both arms).
#[derive(Resource)]
pub struct SharedHuman {
    pub obstacle: HumanObstacle,
}

/// Shared grasp object for the adaptive arm.
#[derive(Resource)]
pub struct AdaptiveGraspObject {
    pub object: GraspObject,
}

/// Shared grasp object for the ISO arm.
#[derive(Resource)]
pub struct IsoGraspObject {
    pub object: GraspObject,
}

/// Elapsed simulation time.
#[derive(Resource, Default)]
pub struct SimTime {
    pub elapsed: f64,
}

/// Throughput comparison metrics.
#[derive(Resource, Default)]
pub struct ThroughputMetrics {
    /// Phi arm advantage as percentage (positive = Phi is faster).
    pub advantage_percent: f64,
}
