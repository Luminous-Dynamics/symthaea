// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! `PlatformPlugin` registration for the manipulator platform.

use crate::embodiment::ManipulatorEmbodiment;
use symthaea_core::embodiment::{EmbodimentBridge, EmbodimentPlatform, PlatformPlugin};
use symthaea_core::genesis::GenesisSeed;

pub struct ManipulatorPlugin;

impl PlatformPlugin for ManipulatorPlugin {
    fn platform(&self) -> EmbodimentPlatform {
        EmbodimentPlatform::Manipulator
    }
    fn feature_name(&self) -> &'static str {
        "manipulator"
    }
    fn num_actuators(&self) -> usize {
        8 // 7 joints + 1 gripper
    }
    fn create_bridge(&self, genesis: &GenesisSeed) -> Box<dyn EmbodimentBridge> {
        Box::new(ManipulatorEmbodiment::new(genesis))
    }
}
