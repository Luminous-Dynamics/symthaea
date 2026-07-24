// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! `PlatformPlugin` registration for the multirotor platform.

use crate::embodiment::FlightEmbodiment;
use symthaea_core::embodiment::{EmbodimentBridge, EmbodimentPlatform, PlatformPlugin};
use symthaea_core::genesis::GenesisSeed;

pub struct MultirotorPlugin;

impl PlatformPlugin for MultirotorPlugin {
    fn platform(&self) -> EmbodimentPlatform {
        // The platform variant is still named Quadrotor (pre-rename);
        // the crate/feature are canonical "multirotor".
        EmbodimentPlatform::Quadrotor
    }
    fn feature_name(&self) -> &'static str {
        "multirotor"
    }
    fn num_actuators(&self) -> usize {
        4
    }
    fn create_bridge(&self, genesis: &GenesisSeed) -> Box<dyn EmbodimentBridge> {
        Box::new(FlightEmbodiment::new(genesis))
    }
}
