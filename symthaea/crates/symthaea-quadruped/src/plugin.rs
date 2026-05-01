use crate::embodiment::QuadrupedEmbodiment;
use crate::types::NUM_ACTUATORS;
use symthaea_core::embodiment::{EmbodimentBridge, EmbodimentPlatform, PlatformPlugin};
use symthaea_core::genesis::GenesisSeed;
pub struct QuadrupedPlugin;
impl PlatformPlugin for QuadrupedPlugin {
    fn platform(&self) -> EmbodimentPlatform {
        EmbodimentPlatform::Quadruped
    }
    fn feature_name(&self) -> &'static str {
        "quadruped"
    }
    fn num_actuators(&self) -> usize {
        NUM_ACTUATORS
    }
    fn create_bridge(&self, g: &GenesisSeed) -> Box<dyn EmbodimentBridge> {
        Box::new(QuadrupedEmbodiment::new(g))
    }
}
