// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use symthaea_core::embodiment::{EmbodimentBridge, EmbodimentPlatform, PlatformPlugin};
use symthaea_core::genesis::GenesisSeed;
use crate::embodiment::ExoskeletonEmbodiment;
use crate::types::NUM_ACTUATORS;

pub struct ExoskeletonPlugin;

impl PlatformPlugin for ExoskeletonPlugin {
    fn platform(&self) -> EmbodimentPlatform { EmbodimentPlatform::Exoskeleton }
    fn feature_name(&self) -> &'static str { "exoskeleton" }
    fn num_actuators(&self) -> usize { NUM_ACTUATORS }
    fn create_bridge(&self, genesis: &GenesisSeed) -> Box<dyn EmbodimentBridge> {
        Box::new(ExoskeletonEmbodiment::new(genesis))
    }
}
