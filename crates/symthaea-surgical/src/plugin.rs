// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::embodiment::SurgicalEmbodiment;
use crate::types::NUM_ACTUATORS;
use symthaea_core::embodiment::{EmbodimentBridge, EmbodimentPlatform, PlatformPlugin};
use symthaea_core::genesis::GenesisSeed;
pub struct SurgicalPlugin;
impl PlatformPlugin for SurgicalPlugin {
    fn platform(&self) -> EmbodimentPlatform {
        EmbodimentPlatform::Surgical
    }
    fn feature_name(&self) -> &'static str {
        "surgical"
    }
    fn num_actuators(&self) -> usize {
        NUM_ACTUATORS
    }
    fn create_bridge(&self, g: &GenesisSeed) -> Box<dyn EmbodimentBridge> {
        Box::new(SurgicalEmbodiment::new(g))
    }
}
