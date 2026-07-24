// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! `PlatformPlugin` registration for the AUV platform.

use crate::embodiment::AuvEmbodiment;
use symthaea_core::embodiment::{EmbodimentBridge, EmbodimentPlatform, PlatformPlugin};
use symthaea_core::genesis::GenesisSeed;

pub struct AuvPlugin;

impl PlatformPlugin for AuvPlugin {
    fn platform(&self) -> EmbodimentPlatform {
        EmbodimentPlatform::Auv
    }
    fn feature_name(&self) -> &'static str {
        "auv"
    }
    fn num_actuators(&self) -> usize {
        8
    }
    fn create_bridge(&self, genesis: &GenesisSeed) -> Box<dyn EmbodimentBridge> {
        Box::new(AuvEmbodiment::new(genesis))
    }
}
