use crate::embodiment::OrbitalEmbodiment;
use crate::types::NUM_ACTUATORS;
use symthaea_core::embodiment::{EmbodimentBridge, EmbodimentPlatform, PlatformPlugin};
use symthaea_core::genesis::GenesisSeed;
pub struct OrbitalPlugin;
impl PlatformPlugin for OrbitalPlugin {
    fn platform(&self) -> EmbodimentPlatform {
        EmbodimentPlatform::Orbital
    }
    fn feature_name(&self) -> &'static str {
        "orbital"
    }
    fn num_actuators(&self) -> usize {
        NUM_ACTUATORS
    }
    fn create_bridge(&self, g: &GenesisSeed) -> Box<dyn EmbodimentBridge> {
        Box::new(OrbitalEmbodiment::new(g))
    }
}
