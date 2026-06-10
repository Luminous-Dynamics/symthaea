// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::embodiment::AgribotEmbodiment;
use crate::types::NUM_ACTUATORS;
use symthaea_core::embodiment::{EmbodimentBridge, EmbodimentPlatform, PlatformPlugin};
use symthaea_core::genesis::GenesisSeed;

pub struct AgribotPlugin;

impl PlatformPlugin for AgribotPlugin {
    fn platform(&self) -> EmbodimentPlatform {
        EmbodimentPlatform::Agribot
    }

    fn feature_name(&self) -> &'static str {
        "agribot"
    }

    fn num_actuators(&self) -> usize {
        NUM_ACTUATORS
    }

    fn create_bridge(&self, genesis: &GenesisSeed) -> Box<dyn EmbodimentBridge> {
        Box::new(AgribotEmbodiment::new(genesis))
    }
}
