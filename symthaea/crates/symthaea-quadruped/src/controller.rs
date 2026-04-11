use symthaea_core::genesis::GenesisSeed; use symthaea_core::hdc::{ContinuousHV, HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig};
use crate::types::{QuadrupedCommand, QuadrupedConfig, NUM_ACTUATORS};
const HDC_DIM: usize = symthaea_core::hdc::HDC_DIMENSION;
pub struct QuadrupedController { network: HdcLtcUnifiedNetwork, weights: Vec<f32>, bias: [f32; NUM_ACTUATORS] }
impl QuadrupedController {
    pub fn new(g: &GenesisSeed, c: &QuadrupedConfig) -> Self {
        let nc = UnifiedConfig { tau_base: 1.0/c.physics_hz as f32, backbone_tau: 0.3, dimension: HDC_DIM, learning_rate: c.learning_rate, ..UnifiedConfig::default() };
        let net = UnifiedNetworkConfig { layer_sizes: vec![c.neurons_per_layer; c.network_layers], neuron_config: nc, use_layer_binding: true, skip_connections: false };
        let network = HdcLtcUnifiedNetwork::from_genesis(net, g);
        let wh = ContinuousHV::from_genesis(g, "quad::out_w", NUM_ACTUATORS * HDC_DIM);
        let mut w: Vec<f32> = wh.as_slice().to_vec(); for v in &mut w { *v *= 0.01; }
        Self { network, weights: w, bias: [0.0; NUM_ACTUATORS] }
    }
    pub fn forward(&mut self, hv: &ContinuousHV, dt: f32) -> QuadrupedCommand {
        self.network.evolve_closed_form(dt, hv); let out = self.network.output().normalize(); let d = out.as_slice();
        let mut t = [0.0f32; NUM_ACTUATORS]; for o in 0..NUM_ACTUATORS { let off = o*HDC_DIM; let mut s = self.bias[o]; for j in 0..HDC_DIM { s += self.weights[off+j]*d[j]; } t[o] = s.tanh(); }
        QuadrupedCommand { joint_torques: t }
    }
    pub fn reset(&mut self) { self.network.reset(); }
}
#[cfg(test)] mod tests { use super::*; #[test] fn test_fwd() { let mut c = QuadrupedController::new(&GenesisSeed::from_phrase("t"), &QuadrupedConfig::default()); let cmd = c.forward(&ContinuousHV::random(HDC_DIM, 42), 0.005); assert!(cmd.joint_torques.iter().all(|t| t.is_finite())); } }
