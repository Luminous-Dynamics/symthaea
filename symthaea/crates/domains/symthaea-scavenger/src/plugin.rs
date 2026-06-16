// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::embodiment::ScavengerEmbodiment;
use crate::types::NUM_ACTUATORS;
use symthaea_core::embodiment::{EmbodimentBridge, EmbodimentPlatform, PlatformPlugin};
use symthaea_core::genesis::GenesisSeed;

pub struct ScavengerPlugin;

impl PlatformPlugin for ScavengerPlugin {
    fn platform(&self) -> EmbodimentPlatform {
        EmbodimentPlatform::Scavenger
    }

    fn feature_name(&self) -> &'static str {
        "scavenger"
    }

    fn num_actuators(&self) -> usize {
        NUM_ACTUATORS
    }

    fn create_bridge(&self, genesis: &GenesisSeed) -> Box<dyn EmbodimentBridge> {
        Box::new(ScavengerEmbodiment::new(genesis))
    }
}
