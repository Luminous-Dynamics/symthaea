// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! `PlatformPlugin` registration for the helicopter platform.

use crate::embodiment::HelicopterEmbodiment;
use symthaea_core::embodiment::{EmbodimentBridge, EmbodimentPlatform, PlatformPlugin};
use symthaea_core::genesis::GenesisSeed;

pub struct HelicopterPlugin;

impl PlatformPlugin for HelicopterPlugin {
    fn platform(&self) -> EmbodimentPlatform {
        EmbodimentPlatform::Helicopter
    }
    fn feature_name(&self) -> &'static str {
        "helicopter"
    }
    fn num_actuators(&self) -> usize {
        6
    }
    fn create_bridge(&self, genesis: &GenesisSeed) -> Box<dyn EmbodimentBridge> {
        Box::new(HelicopterEmbodiment::new(genesis))
    }
}
