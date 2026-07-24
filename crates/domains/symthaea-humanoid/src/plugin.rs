// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! `PlatformPlugin` registration for the humanoid platform.

use crate::embodiment::HumanoidEmbodiment;
use symthaea_core::embodiment::{EmbodimentBridge, EmbodimentPlatform, PlatformPlugin};
use symthaea_core::genesis::GenesisSeed;

pub struct HumanoidPlugin;

impl PlatformPlugin for HumanoidPlugin {
    fn platform(&self) -> EmbodimentPlatform {
        EmbodimentPlatform::Humanoid
    }
    fn feature_name(&self) -> &'static str {
        "humanoid"
    }
    fn num_actuators(&self) -> usize {
        21 // Default Dmc21 morphology (create_bridge constructs the default)
    }
    fn create_bridge(&self, genesis: &GenesisSeed) -> Box<dyn EmbodimentBridge> {
        Box::new(HumanoidEmbodiment::new(genesis))
    }
}
