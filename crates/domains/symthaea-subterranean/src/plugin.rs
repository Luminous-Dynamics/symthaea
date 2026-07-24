// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::embodiment::SubterraneanEmbodiment;
use crate::types::NUM_PHYSICAL_ACTUATORS;
use symthaea_core::embodiment::{EmbodimentBridge, EmbodimentPlatform, PlatformPlugin};
use symthaea_core::genesis::GenesisSeed;

pub struct SubterraneanPlugin;

impl PlatformPlugin for SubterraneanPlugin {
    fn platform(&self) -> EmbodimentPlatform {
        EmbodimentPlatform::Subterranean
    }

    fn feature_name(&self) -> &'static str {
        "subterranean"
    }

    fn num_actuators(&self) -> usize {
        NUM_PHYSICAL_ACTUATORS
    }

    fn create_bridge(&self, genesis: &GenesisSeed) -> Box<dyn EmbodimentBridge> {
        Box::new(SubterraneanEmbodiment::new(genesis))
    }
}
