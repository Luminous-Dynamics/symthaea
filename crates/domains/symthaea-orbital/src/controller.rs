use crate::types::{NUM_ACTUATORS, OrbitalCommand, OrbitalConfig};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig};
const HDC_DIM: usize = symthaea_core::hdc::HDC_DIMENSION;
pub struct OrbitalController {
    network: HdcLtcUnifiedNetwork,
    weights: Vec<f32>,
    bias: [f32; NUM_ACTUATORS],
    learning_rate: f32,
    /// Cached final-layer HV from the last forward() (post-normalize) --
    /// needed by train_step's delta rule.
    last_features: Vec<f32>,
    /// Cached post-tanh joint-torque outputs from the last forward().
    last_outputs: [f32; NUM_ACTUATORS],
}
impl OrbitalController {
    pub fn new(g: &GenesisSeed, c: &OrbitalConfig) -> Self {
        let nc = UnifiedConfig {
            tau_base: 1.0 / c.physics_hz as f32,
            backbone_tau: 0.5,
            dimension: HDC_DIM,
            learning_rate: c.learning_rate,
            ..UnifiedConfig::default()
        };
        let net = UnifiedNetworkConfig {
            layer_sizes: vec![c.neurons_per_layer; c.network_layers],
            neuron_config: nc,
            use_layer_binding: true,
            skip_connections: false,
        };
        let network = HdcLtcUnifiedNetwork::from_genesis(net, g);
        let wh = ContinuousHV::from_genesis(g, "orb::out_w", NUM_ACTUATORS * HDC_DIM);
        let mut w: Vec<f32> = wh.as_slice().to_vec();
        for v in &mut w {
            *v *= 0.01;
        }
        Self {
            network,
            weights: w,
            bias: [0.0; NUM_ACTUATORS],
            learning_rate: c.learning_rate,
            last_features: Vec::new(),
            last_outputs: [0.0; NUM_ACTUATORS],
        }
    }
    pub fn forward(&mut self, hv: &ContinuousHV, dt: f32) -> OrbitalCommand {
        self.network.evolve_closed_form(dt, hv);
        let out = self.network.output().normalize();
        let d = out.as_slice();
        let mut t = [0.0f32; NUM_ACTUATORS];
        for o in 0..NUM_ACTUATORS {
            let off = o * HDC_DIM;
            let mut s = self.bias[o];
            for j in 0..HDC_DIM {
                s += self.weights[off + j] * d[j];
            }
            t[o] = s.tanh();
        }
        self.last_features = d.to_vec();
        self.last_outputs = t;
        // Network currently only drives the arm; translational burns and
        // desaturation commands are zero here (not yet wired to a learned
        // or planned policy — see crate README "not yet modeled").
        OrbitalCommand {
            joint_torques: t,
            translational_burn_mps: [0.0; 3],
            desaturation_torque_nm: [0.0; 3],
        }
    }

    /// One supervised update of the arm's output projection toward `target`
    /// (delta rule through tanh), using the features cached by the last
    /// `forward()`. Returns the pre-update mean-squared error. This is what
    /// makes `OrbitalTrainer` actually train (real-trainer follow-up to
    /// SYMTHAEA_CLASSIC_PLATFORMS_FEP_HONESTY_2026-07-09.md).
    pub fn train_step(&mut self, target: &[f32; NUM_ACTUATORS]) -> f32 {
        if self.last_features.is_empty() {
            return 0.0;
        }
        let mut mse = 0.0f32;
        for o in 0..NUM_ACTUATORS {
            let out = self.last_outputs[o];
            let err = target[o] - out;
            mse += err * err;
            let delta = self.learning_rate * err * (1.0 - out * out);
            let off = o * HDC_DIM;
            for (j, f) in self.last_features.iter().enumerate() {
                self.weights[off + j] += delta * f;
            }
            self.bias[o] += delta;
        }
        mse / NUM_ACTUATORS as f32
    }

    pub fn reset(&mut self) {
        self.network.reset();
        self.last_features.clear();
        self.last_outputs = [0.0; NUM_ACTUATORS];
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_fwd() {
        let mut c =
            OrbitalController::new(&GenesisSeed::from_phrase("t"), &OrbitalConfig::default());
        let cmd = c.forward(&ContinuousHV::random(HDC_DIM, 42), 0.01);
        assert!(cmd.joint_torques.iter().all(|t| t.is_finite()));
    }

    /// Direct unit-level proof that `train_step` actually reduces error
    /// toward a fixed target. An episode-level first-vs-last comparison
    /// (via `OrbitalTrainer`) turned out too noisy/slow for this crate: the
    /// simulator resets to the stowed (near-zero-error) pose every episode,
    /// so a whole-episode mean is dominated by how much that one episode's
    /// environmental perturbations happened to move the arm, not by how
    /// much the controller learned. This test isolates the actual claim
    /// (real-trainer follow-up to SYMTHAEA_CLASSIC_PLATFORMS_FEP_HONESTY_2026-07-09.md).
    #[test]
    fn test_train_step_reduces_error_toward_fixed_target() {
        let mut c = OrbitalController::new(
            &GenesisSeed::from_phrase("direct_train"),
            &OrbitalConfig::default(),
        );
        let hv = ContinuousHV::random(HDC_DIM, 7);
        let target = [0.5f32; NUM_ACTUATORS];
        c.forward(&hv, 0.01);
        let first = c.train_step(&target);
        for _ in 0..50 {
            c.forward(&hv, 0.01);
            c.train_step(&target);
        }
        c.forward(&hv, 0.01);
        let last = c.train_step(&target);
        assert!(
            last < first,
            "train_step must reduce imitation error toward a fixed target: first {first:.5} -> last {last:.5}"
        );
    }
}
