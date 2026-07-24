// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! `PlatformPlugin` registration for the vehicle platform.

use crate::embodiment::VehicleEmbodiment;
use symthaea_core::embodiment::{EmbodimentBridge, EmbodimentPlatform, PlatformPlugin};
use symthaea_core::genesis::GenesisSeed;

pub struct VehiclePlugin;

impl PlatformPlugin for VehiclePlugin {
    fn platform(&self) -> EmbodimentPlatform {
        EmbodimentPlatform::Vehicle
    }
    fn feature_name(&self) -> &'static str {
        "vehicle"
    }
    fn num_actuators(&self) -> usize {
        3 // steering, throttle, brake
    }
    fn create_bridge(&self, genesis: &GenesisSeed) -> Box<dyn EmbodimentBridge> {
        Box::new(VehicleEmbodiment::new(genesis))
    }
}
