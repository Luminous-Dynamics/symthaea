// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::embodiment::ClimeEmbodiment;
use crate::types::NUM_ACTUATORS;
use symthaea_core::embodiment::{EmbodimentBridge, EmbodimentPlatform, PlatformPlugin};
use symthaea_core::genesis::GenesisSeed;

pub struct ClimePlugin;

impl PlatformPlugin for ClimePlugin {
    fn platform(&self) -> EmbodimentPlatform {
        EmbodimentPlatform::Clime
    }
    fn feature_name(&self) -> &'static str {
        "clime"
    }
    fn num_actuators(&self) -> usize {
        NUM_ACTUATORS
    }
    fn create_bridge(&self, genesis: &GenesisSeed) -> Box<dyn EmbodimentBridge> {
        Box::new(ClimeEmbodiment::new(genesis))
    }
}
