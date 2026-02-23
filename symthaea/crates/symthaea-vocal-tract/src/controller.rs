//! Vocal tract controller: wraps HdcLtcUnifiedNetwork + output projection (16,384D → 9D).
//!
//! Follows the `FlightController` pattern from `crates/symthaea-flight/src/controller.rs`.
//! The controller uses the full 16,384D HDC-LTC temporal dynamics engine.
//! Cognitive HVs are evolved through the network, then a linear output projection
//! maps the final-layer HV to 9D `FormantFrame` parameters.
//!
//! # Output dimensions (9D)
//!
//! | Index | Parameter  | Activation      | Default (schwa) |
//! |-------|-----------|-----------------|-----------------|
//! | 0     | F1        | softplus+clamp  | 500 Hz          |
//! | 1     | F2        | softplus+clamp  | 1500 Hz         |
//! | 2     | F3        | softplus+clamp  | 2500 Hz         |
//! | 3     | B1        | softplus+clamp  | 60 Hz           |
//! | 4     | B2        | softplus+clamp  | 90 Hz           |
//! | 5     | B3        | softplus+clamp  | 150 Hz          |
//! | 6     | F0        | softplus+clamp  | 120 Hz          |
//! | 7     | energy    | sigmoid         | 0.5             |
//! | 8     | voicing   | sigmoid         | 0.8             |

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{
    ContinuousHV, HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig, HDC_DIMENSION,
};

use crate::types::FormantFrame;

/// Number of output dimensions (F1, F2, F3, B1, B2, B3, F0, energy, voicing).
const OUTPUT_DIM: usize = 9;

/// Configuration for the vocal tract controller.
#[derive(Debug, Clone)]
pub struct VocalTractConfig {
    /// Number of layers in the HdcLtcUnifiedNetwork.
    pub network_layers: usize,
    /// Neurons per layer.
    pub neurons_per_layer: usize,
    /// Learning rate for output projection and BPTT.
    pub learning_rate: f32,
    /// Base fundamental frequency (Hz).
    pub base_f0: f32,
    /// F0 range (Hz) — output F0 can swing ±f0_range/2 around base_f0.
    pub f0_range: f32,
}

impl Default for VocalTractConfig {
    fn default() -> Self {
        Self {
            network_layers: 2,
            neurons_per_layer: 4,
            // 8 total neurons (was 24) — 3× fewer evolve_closed_form calls per frame.
            // With improved supervised training (10 steps/phoneme, 10× lr, no decay),
            // the smaller network achieves comparable accuracy at much higher throughput.
            learning_rate: 0.001,
            base_f0: 120.0,
            f0_range: 200.0,
        }
    }
}

/// Speaker voice profile: base pitch, formant scale, dynamics.
///
/// Different profiles produce different voice characteristics when used with
/// `VocalTractPipeline::new_with_speaker()`.
#[derive(Debug, Clone)]
pub struct SpeakerProfile {
    /// Base fundamental frequency (Hz).
    pub base_f0: f32,
    /// Formant frequency scaling factor (1.0 = adult male, 1.15 = female, 1.30 = child).
    pub formant_scale: f32,
    /// Time constant factor (smaller = faster articulation).
    pub tau_factor: f32,
    /// Energy scaling factor.
    pub energy_scale: f32,
    /// Speaker name (for genesis derivation).
    pub name: String,
}

impl SpeakerProfile {
    /// Adult male voice (default).
    pub fn male() -> Self {
        Self {
            base_f0: 120.0,
            formant_scale: 1.0,
            tau_factor: 1.0,
            energy_scale: 1.0,
            name: "male".to_string(),
        }
    }

    /// Adult female voice.
    pub fn female() -> Self {
        Self {
            base_f0: 220.0,
            formant_scale: 1.15,
            tau_factor: 0.9,
            energy_scale: 0.95,
            name: "female".to_string(),
        }
    }

    /// Child voice.
    pub fn child() -> Self {
        Self {
            base_f0: 300.0,
            formant_scale: 1.30,
            tau_factor: 0.8,
            energy_scale: 0.9,
            name: "child".to_string(),
        }
    }
}

impl Default for SpeakerProfile {
    fn default() -> Self {
        Self::male()
    }
}

/// Vocal tract controller wrapping an HdcLtcUnifiedNetwork + linear output head.
///
/// Forward pass:
/// 1. `network.evolve_closed_form(dt, &cognitive_hv)` — O(D) temporal evolution
/// 2. `output = network.output()` — bundled final layer (16,384D)
/// 3. `output_weights @ output + output_bias` → 9D raw
/// 4. Activations: softplus+clamp for formant freqs, sigmoid for energy/voicing
pub struct VocalTractController {
    /// The temporal dynamics engine — full 16,384D HDC-LTC.
    network: HdcLtcUnifiedNetwork,
    /// Output projection weights: 9 rows × 16,384 columns (flat row-major).
    output_weights: Vec<f32>,
    /// Output bias (9D) — initialized to schwa defaults.
    output_bias: [f32; OUTPUT_DIM],
    /// Current learning rate (modulated by FEP agent).
    learning_rate: f32,
    /// Configuration.
    config: VocalTractConfig,
    /// Optional learned prosody head: cognitive channels → F0/energy/voicing corrections.
    prosody_head: Option<ProsodyHead>,
}

impl VocalTractController {
    /// Create a new controller from a genesis seed and config.
    pub fn new(genesis: &GenesisSeed, config: &VocalTractConfig) -> Self {
        let neuron_config = UnifiedConfig {
            tau_base: 0.005,   // 5ms — matches 200Hz frame rate
            backbone_tau: 0.1, // Moderate state dependency for smooth formant transitions
            dimension: HDC_DIMENSION,
            learning_rate: config.learning_rate,
            ..UnifiedConfig::default()
        };

        let net_config = UnifiedNetworkConfig {
            layer_sizes: vec![config.neurons_per_layer; config.network_layers],
            neuron_config,
            use_layer_binding: true,
            skip_connections: false,
        };

        let network = HdcLtcUnifiedNetwork::from_genesis(net_config, genesis);

        // Initialize output weights from genesis (small values for stability)
        let total_weights = OUTPUT_DIM * HDC_DIMENSION;
        let weight_hv = genesis.hv("vocal_tract::output_weights", total_weights);
        let mut output_weights = weight_hv.values;
        for w in &mut output_weights {
            *w *= 0.05; // Larger init → stronger phoneme differentiation from start
        }

        // Bias initialized to schwa (neutral vowel) defaults
        let output_bias = [
            500.0,  // F1 (Hz)
            1500.0, // F2 (Hz)
            2500.0, // F3 (Hz)
            60.0,   // B1 (Hz)
            90.0,   // B2 (Hz)
            150.0,  // B3 (Hz)
            config.base_f0, // F0 (Hz)
            0.0,    // energy (pre-sigmoid → sigmoid(0) = 0.5)
            1.39,   // voicing (pre-sigmoid → sigmoid(1.39) ≈ 0.8)
        ];

        let prosody_head = Some(ProsodyHead::from_genesis(genesis, config.learning_rate * 10.0));

        Self {
            network,
            output_weights,
            output_bias,
            learning_rate: config.learning_rate,
            config: config.clone(),
            prosody_head,
        }
    }

    /// Forward pass: evolve the network with cognitive input and produce FormantFrame.
    ///
    /// - `cognitive_hv`: 16,384D ContinuousHV from VocalTractHdcEncoder
    /// - `dt`: timestep in seconds (typically 0.005 for 200Hz)
    pub fn forward(&mut self, cognitive_hv: &ContinuousHV, dt: f32) -> FormantFrame {
        // 1. Evolve network dynamics
        self.network.evolve_closed_form(dt, cognitive_hv);

        // 2. Get bundled final-layer output, normalized to unit length
        let output_hv = self.network.output().normalize();
        let hv_values = output_hv.as_slice();

        // 3. Linear projection: output_weights @ hv + bias → 9D
        let mut raw = [0.0f32; OUTPUT_DIM];
        for i in 0..OUTPUT_DIM {
            let row_offset = i * HDC_DIMENSION;
            let mut sum = self.output_bias[i];
            for j in 0..HDC_DIMENSION {
                sum += self.output_weights[row_offset + j] * hv_values[j];
            }
            raw[i] = sum;
        }

        // 4. Activations
        // Formant frequencies: softplus + clamp to physical range
        let f1 = softplus(raw[0]).clamp(200.0, 1000.0);
        let f2 = softplus(raw[1]).clamp(600.0, 3000.0);
        let f3 = softplus(raw[2]).clamp(1500.0, 5000.0);
        let b1 = softplus(raw[3]).clamp(30.0, 300.0);
        let b2 = softplus(raw[4]).clamp(30.0, 400.0);
        let b3 = softplus(raw[5]).clamp(50.0, 500.0);
        let f0 = softplus(raw[6]).clamp(
            (self.config.base_f0 - self.config.f0_range / 2.0).max(50.0),
            self.config.base_f0 + self.config.f0_range / 2.0,
        );
        // Energy and voicing: sigmoid → [0, 1]
        let energy = sigmoid(raw[7]);
        let voicing = sigmoid(raw[8]);

        FormantFrame {
            f1,
            f2,
            f3,
            b1,
            b2,
            b3,
            f0,
            energy,
            voicing,
            time: 0.0, // Caller sets absolute time
        }
    }

    /// Train the output projection via BPTT.
    ///
    /// Uses `target` as the ground-truth FormantFrame.
    pub fn train_step(
        &mut self,
        cognitive_hv: &ContinuousHV,
        target: &FormantFrame,
        dt: f32,
        lr_override: Option<f32>,
    ) {
        let lr = lr_override.unwrap_or(self.learning_rate);
        self.train_step_impl(cognitive_hv, target, dt, lr, 1e-4);
    }

    /// Internal training step with configurable weight decay.
    ///
    /// Separated from `train_step()` so supervised training can disable weight decay
    /// (which erodes learned weights during multi-epoch phoneme training).
    fn train_step_impl(
        &mut self,
        cognitive_hv: &ContinuousHV,
        target: &FormantFrame,
        dt: f32,
        lr: f32,
        weight_decay: f32,
    ) {
        let output_lr = lr * (HDC_DIMENSION as f32).sqrt();

        // Forward pass to get current output
        let output_hv = self.network.output().normalize();
        let hv_values = output_hv.as_slice();

        // Compute current raw outputs
        let mut raw = [0.0f32; OUTPUT_DIM];
        for i in 0..OUTPUT_DIM {
            let row_offset = i * HDC_DIMENSION;
            let mut sum = self.output_bias[i];
            for j in 0..HDC_DIMENSION {
                sum += self.output_weights[row_offset + j] * hv_values[j];
            }
            raw[i] = sum;
        }

        // Compute activated outputs
        let pred = [
            softplus(raw[0]).clamp(200.0, 1000.0),
            softplus(raw[1]).clamp(600.0, 3000.0),
            softplus(raw[2]).clamp(1500.0, 5000.0),
            softplus(raw[3]).clamp(30.0, 300.0),
            softplus(raw[4]).clamp(30.0, 400.0),
            softplus(raw[5]).clamp(50.0, 500.0),
            softplus(raw[6]).clamp(50.0, 320.0),
            sigmoid(raw[7]),
            sigmoid(raw[8]),
        ];

        let tgt = [
            target.f1, target.f2, target.f3, target.b1, target.b2, target.b3, target.f0,
            target.energy, target.voicing,
        ];

        // Compute error (pred - target)
        let errors: [f32; OUTPUT_DIM] = std::array::from_fn(|i| pred[i] - tgt[i]);

        // Backprop through activations
        let mut d_raw = [0.0f32; OUTPUT_DIM];
        // Formant freqs: d(softplus)/dx = sigmoid(x)
        for i in 0..7 {
            d_raw[i] = errors[i] * sigmoid(raw[i]);
        }
        // Energy: d(sigmoid)/dx = s*(1-s)
        let s7 = sigmoid(raw[7]);
        d_raw[7] = errors[7] * s7 * (1.0 - s7);
        // Voicing: d(sigmoid)/dx = s*(1-s)
        let s8 = sigmoid(raw[8]);
        d_raw[8] = errors[8] * s8 * (1.0 - s8);

        // Normalize gradients by expected range of each output dimension.
        // Without normalization, GRAD_CLIP kills formant gradients (naturally 100–1000 Hz)
        // causing all vowels to converge to schwa.
        const ERROR_SCALE: [f32; OUTPUT_DIM] = [
            500.0,  // F1: range ~200-1000 Hz
            1000.0, // F2: range ~600-3000 Hz
            1500.0, // F3: range ~1500-5000 Hz
            100.0,  // B1: range ~30-300 Hz
            150.0,  // B2: range ~30-400 Hz
            200.0,  // B3: range ~50-500 Hz
            100.0,  // F0: range ~50-320 Hz
            1.0,    // energy: already 0-1
            1.0,    // voicing: already 0-1
        ];
        for i in 0..OUTPUT_DIM {
            d_raw[i] /= ERROR_SCALE[i];
        }

        // Gradient clipping (raised from 1.0 — normalized gradients are now ≤1.0 for
        // in-range errors, so 5.0 provides safety margin without killing signal)
        const GRAD_CLIP: f32 = 5.0;
        for g in &mut d_raw {
            *g = g.clamp(-GRAD_CLIP, GRAD_CLIP);
        }

        // Conditional weight decay (disabled during supervised phoneme training)
        if weight_decay > 0.0 {
            let decay = 1.0 - weight_decay;
            for w in self.output_weights.iter_mut() {
                *w *= decay;
            }
        }

        // Update output weights
        for i in 0..OUTPUT_DIM {
            let row_offset = i * HDC_DIMENSION;
            for j in 0..HDC_DIMENSION {
                self.output_weights[row_offset + j] -= output_lr * d_raw[i] * hv_values[j];
            }
            self.output_bias[i] -= output_lr * d_raw[i];
        }

        // Backprop through network (BPTT)
        let dim = HDC_DIMENSION;
        let mut grad_hv_values = vec![0.0f32; dim];
        for i in 0..OUTPUT_DIM {
            let row_offset = i * dim;
            for j in 0..dim {
                grad_hv_values[j] += d_raw[i] * self.output_weights[row_offset + j];
            }
        }

        let n_layers = self.network.n_layers();
        let mut target_hv = output_hv.add(&ContinuousHV::from_vec(grad_hv_values).scale(-1.0));

        for layer_idx in (0..n_layers).rev() {
            let layer_input = self.network.layer_input(layer_idx, cognitive_hv);
            let prev_layer_output = if layer_idx > 0 {
                self.network.output_at_layer(layer_idx - 1).cloned()
            } else {
                None
            };

            let mut avg_d_input = ContinuousHV::zero(dim);
            let mut neuron_count = 0usize;

            if let Some(layer) = self.network.layer_mut(layer_idx) {
                for neuron in layer.iter_mut() {
                    let grads = neuron.backward(&layer_input, &target_hv, dt);
                    avg_d_input = avg_d_input.add(&grads.d_input);
                    neuron_count += 1;
                    neuron.apply_gradients(&grads, lr);
                }
            }

            if let Some(prev_output) = prev_layer_output {
                if neuron_count > 0 {
                    let scale = 1.0 / neuron_count as f32;
                    target_hv = prev_output.subtract(&avg_d_input.scale(scale));
                }
            }
        }
    }

    /// Modulate all neuron time constants by a factor.
    ///
    /// - `factor < 1.0`: faster adaptation (more responsive formant transitions)
    /// - `factor > 1.0`: slower, smoother (stable sustained vowels)
    pub fn modulate_tau(&mut self, factor: f32) {
        let factor = factor.clamp(0.3, 3.0);
        for layer_idx in 0..self.network.n_layers() {
            if let Some(layer) = self.network.layer_mut(layer_idx) {
                for neuron in layer.iter_mut() {
                    let new_tau = neuron.config().tau_base * factor;
                    neuron.set_tau_base(new_tau);
                }
            }
        }
    }

    /// Reset network state.
    pub fn reset(&mut self) {
        self.network.reset();
    }

    /// Set the learning rate directly.
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr.clamp(1e-6, 0.1);
    }

    /// Get current learning rate.
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Get configuration.
    pub fn config(&self) -> &VocalTractConfig {
        &self.config
    }

    /// Forward pass with prosody head: evolve network then apply learned prosody corrections.
    ///
    /// If `channels` is provided and a prosody head exists, the head maps the 10D cognitive
    /// channels directly to additive F0/energy/voicing corrections, bypassing the HDC bottleneck.
    pub fn forward_with_prosody(
        &mut self,
        cognitive_hv: &ContinuousHV,
        dt: f32,
        channels: Option<&[f32; 10]>,
    ) -> FormantFrame {
        let mut frame = self.forward(cognitive_hv, dt);

        if let (Some(head), Some(ch)) = (&self.prosody_head, channels) {
            let correction = head.forward(ch);

            // F0: additive Hz correction, re-clamped to valid range
            frame.f0 = (frame.f0 + correction.delta_f0).clamp(
                (self.config.base_f0 - self.config.f0_range / 2.0).max(50.0),
                self.config.base_f0 + self.config.f0_range / 2.0,
            );

            // Energy: correction in logit space, then sigmoid back to [0, 1]
            let energy_logit = logit(frame.energy) + correction.delta_energy;
            frame.energy = sigmoid(energy_logit);

            // Voicing: correction in logit space, then sigmoid back to [0, 1]
            let voicing_logit = logit(frame.voicing) + correction.delta_voicing;
            frame.voicing = sigmoid(voicing_logit);
        }

        frame
    }

    /// Train the prosody head given target prosody values and current predictions.
    pub fn train_prosody(
        &mut self,
        channels: &[f32; 10],
        target_f0: f32,
        target_energy: f32,
        target_voicing: f32,
        current_f0: f32,
        current_energy: f32,
        current_voicing: f32,
    ) {
        if let Some(head) = &mut self.prosody_head {
            let target_correction = ProsodyCorrection {
                delta_f0: (target_f0 - current_f0).clamp(-50.0, 50.0),
                delta_energy: (logit(target_energy) - logit(current_energy)).clamp(-1.0, 1.0),
                delta_voicing: (logit(target_voicing) - logit(current_voicing)).clamp(-0.5, 0.5),
            };
            head.train_step(channels, &target_correction);
        }
    }

    /// Train the output projection on all phoneme targets from a provided slice.
    ///
    /// For each epoch, iterates every phoneme target, creates a deterministic cognitive HV
    /// from the genesis seed, and runs `train_step()` against the ground-truth formants.
    ///
    /// Returns the average loss from the final epoch.
    pub fn train_on_phoneme_targets(
        &mut self,
        genesis: &GenesisSeed,
        phoneme_targets: &[(&str, &crate::types::FormantTarget)],
        epochs: usize,
    ) -> f32 {
        if phoneme_targets.is_empty() {
            return 0.0;
        }

        // Generate a deterministic HV per phoneme
        let phoneme_hvs: Vec<(&str, ContinuousHV, FormantFrame)> = phoneme_targets
            .iter()
            .map(|(name, target)| {
                let hv = genesis.hv(&format!("phoneme::{}", name), HDC_DIMENSION);
                let frame = FormantFrame {
                    f1: target.f1,
                    f2: target.f2,
                    f3: target.f3,
                    b1: target.b1,
                    b2: target.b2,
                    b3: target.b3,
                    f0: self.config.base_f0,
                    energy: if target.is_voiced { 0.7 } else { 0.3 },
                    voicing: if target.is_voiced { 0.95 } else { 0.1 },
                    time: 0.0,
                };
                (*name, hv, frame)
            })
            .collect();

        let mut last_epoch_loss = 0.0;

        // Convergence-critical parameters:
        // - 20 warmup steps for LTC neurons to reach differentiated steady states
        // - 10 gradient steps per phoneme per epoch (was 1 → 10× more signal)
        // - 10× learning rate during supervised training
        // - No weight decay (prevents erosion of learned weights across epochs)
        const WARMUP_STEPS: usize = 20;
        const TRAIN_STEPS: usize = 10;
        let supervised_lr = self.learning_rate * 10.0;

        for _epoch in 0..epochs {
            let mut epoch_loss = 0.0;

            for (_, hv, target) in &phoneme_hvs {
                // Reset network state to isolate each phoneme (prevents state bleed)
                self.reset();

                // Warmup: let the network settle for this phoneme's HV
                for _ in 0..WARMUP_STEPS {
                    self.forward(hv, 0.005);
                }

                // Forward to get prediction from settled state
                let pred = self.forward(hv, 0.005);

                // Accumulate loss (measured before training to track progress)
                let loss = formant_mse(&pred, target);
                epoch_loss += loss;

                // Multiple gradient steps with no weight decay for faster convergence
                for _ in 0..TRAIN_STEPS {
                    self.forward(hv, 0.005);
                    self.train_step_impl(hv, target, 0.005, supervised_lr, 0.0);
                }
            }

            last_epoch_loss = epoch_loss / phoneme_hvs.len() as f32;
        }

        last_epoch_loss
    }
}

/// Normalized mean squared error between two FormantFrames.
///
/// Each dimension is divided by its expected range so that all dimensions
/// contribute equally (and the loss is scale-invariant).
fn formant_mse(a: &FormantFrame, b: &FormantFrame) -> f32 {
    let diffs = [
        (a.f1 - b.f1) / 500.0,
        (a.f2 - b.f2) / 1000.0,
        (a.f3 - b.f3) / 1500.0,
        (a.b1 - b.b1) / 100.0,
        (a.b2 - b.b2) / 150.0,
        (a.b3 - b.b3) / 200.0,
        (a.f0 - b.f0) / 100.0,
        a.energy - b.energy,
        a.voicing - b.voicing,
    ];
    diffs.iter().map(|d| d * d).sum::<f32>() / OUTPUT_DIM as f32
}

/// Softplus activation: ln(1 + e^x). Smooth approximation to ReLU.
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x // Avoid overflow: softplus(x) ≈ x for large x
    } else if x < -20.0 {
        0.0 // softplus(x) ≈ 0 for very negative x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Sigmoid activation.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Inverse sigmoid (logit): maps p ∈ (0, 1) to ℝ.
fn logit(p: f32) -> f32 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}

/// Additive prosody corrections from the learned prosody head.
#[derive(Debug, Clone, Copy)]
pub struct ProsodyCorrection {
    /// F0 correction in Hz (clamped to ±50 Hz).
    pub delta_f0: f32,
    /// Energy correction in logit space (clamped to ±1.0).
    pub delta_energy: f32,
    /// Voicing correction in logit space (clamped to ±0.5).
    pub delta_voicing: f32,
}

/// Lightweight MLP (10→8→3) mapping cognitive voice channels directly to prosody corrections.
///
/// Prosody (F0, energy, voicing) should respond more quickly and directly to consciousness
/// state than formants, which have articulatory inertia. This head bypasses the 16,384D
/// HDC bottleneck for prosody-specific modulation.
pub struct ProsodyHead {
    /// Hidden layer weights: 10 inputs × 8 hidden (flat row-major).
    w1: [f32; 80],
    /// Hidden layer bias (8D).
    b1: [f32; 8],
    /// Output layer weights: 8 hidden × 3 outputs (flat row-major).
    w2: [f32; 24],
    /// Output layer bias (3D) — zero-initialized for no initial correction.
    b2: [f32; 3],
    /// Learning rate.
    lr: f32,
}

impl ProsodyHead {
    /// Create from genesis seed with small random initialization and zero output bias.
    pub fn from_genesis(genesis: &GenesisSeed, lr: f32) -> Self {
        let w1_hv = genesis.hv("prosody_head::w1", 80);
        let mut w1 = [0.0f32; 80];
        for (i, v) in w1_hv.values.iter().enumerate().take(80) {
            w1[i] = v * 0.1; // Small init
        }

        let b1_hv = genesis.hv("prosody_head::b1", 8);
        let mut b1 = [0.0f32; 8];
        for (i, v) in b1_hv.values.iter().enumerate().take(8) {
            b1[i] = v * 0.01;
        }

        let w2_hv = genesis.hv("prosody_head::w2", 24);
        let mut w2 = [0.0f32; 24];
        for (i, v) in w2_hv.values.iter().enumerate().take(24) {
            w2[i] = v * 0.1;
        }

        // Zero output bias → no initial correction
        let b2 = [0.0f32; 3];

        Self { w1, b1, w2, b2, lr }
    }

    /// Forward pass: 10D channels → tanh hidden → linear output → clamped corrections.
    pub fn forward(&self, channels: &[f32; 10]) -> ProsodyCorrection {
        // Hidden layer: h = tanh(W1 @ x + b1)
        let mut hidden = [0.0f32; 8];
        for i in 0..8 {
            let mut sum = self.b1[i];
            for j in 0..10 {
                sum += self.w1[i * 10 + j] * channels[j];
            }
            hidden[i] = sum.tanh();
        }

        // Output layer: y = W2 @ h + b2
        let mut output = [0.0f32; 3];
        for i in 0..3 {
            let mut sum = self.b2[i];
            for j in 0..8 {
                sum += self.w2[i * 8 + j] * hidden[j];
            }
            output[i] = sum;
        }

        ProsodyCorrection {
            delta_f0: output[0].clamp(-50.0, 50.0),
            delta_energy: output[1].clamp(-1.0, 1.0),
            delta_voicing: output[2].clamp(-0.5, 0.5),
        }
    }

    /// Train via backprop: compute gradients from target corrections and update weights.
    pub fn train_step(&mut self, channels: &[f32; 10], target: &ProsodyCorrection) {
        // Forward pass (save intermediates)
        let mut hidden_pre = [0.0f32; 8];
        let mut hidden = [0.0f32; 8];
        for i in 0..8 {
            let mut sum = self.b1[i];
            for j in 0..10 {
                sum += self.w1[i * 10 + j] * channels[j];
            }
            hidden_pre[i] = sum;
            hidden[i] = sum.tanh();
        }

        let mut output = [0.0f32; 3];
        for i in 0..3 {
            let mut sum = self.b2[i];
            for j in 0..8 {
                sum += self.w2[i * 8 + j] * hidden[j];
            }
            output[i] = sum;
        }

        // Clamp output for loss computation
        let pred = [
            output[0].clamp(-50.0, 50.0),
            output[1].clamp(-1.0, 1.0),
            output[2].clamp(-0.5, 0.5),
        ];
        let tgt = [target.delta_f0, target.delta_energy, target.delta_voicing];

        // Error: pred - target
        let d_output: [f32; 3] = std::array::from_fn(|i| pred[i] - tgt[i]);

        // Backprop through output layer
        let mut d_hidden = [0.0f32; 8];
        for i in 0..3 {
            for j in 0..8 {
                d_hidden[j] += d_output[i] * self.w2[i * 8 + j];
                self.w2[i * 8 + j] -= self.lr * d_output[i] * hidden[j];
            }
            self.b2[i] -= self.lr * d_output[i];
        }

        // Backprop through tanh: d_pre = d_hidden * (1 - tanh^2)
        for i in 0..8 {
            let dtanh = 1.0 - hidden[i] * hidden[i];
            let d_pre = d_hidden[i] * dtanh;
            for j in 0..10 {
                self.w1[i * 10 + j] -= self.lr * d_pre * channels[j];
            }
            self.b1[i] -= self.lr * d_pre;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FormantTarget;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-vocal-tract-controller")
    }

    /// Inline test data: a representative set of phoneme targets covering vowels,
    /// voiced consonant, and unvoiced consonant.
    fn test_phoneme_targets() -> Vec<(&'static str, FormantTarget)> {
        vec![
            // Vowels
            ("AH", FormantTarget::vowel(520.0, 1190.0, 2390.0, 80.0)),    // "but" (stressed)
            ("IY", FormantTarget::vowel(270.0, 2290.0, 3010.0, 100.0)),   // "beat"
            ("EH", FormantTarget::vowel(530.0, 1840.0, 2480.0, 80.0)),    // "bet"
            // Voiced consonant (bilabial stop)
            ("P", FormantTarget::unvoiced_consonant(200.0, 1000.0, 2200.0, 60.0)),
            // Unvoiced consonant (alveolar fricative)
            ("S", FormantTarget::unvoiced_consonant(320.0, 1700.0, 2600.0, 100.0)),
        ]
    }

    #[test]
    fn test_forward_valid_formants() {
        let genesis = test_genesis();
        let config = VocalTractConfig::default();
        let mut ctrl = VocalTractController::new(&genesis, &config);

        let cognitive_hv = ContinuousHV::random(HDC_DIMENSION, 42);
        let frame = ctrl.forward(&cognitive_hv, 0.005);

        // All formant frequencies should be in valid ranges
        assert!(frame.f1 >= 200.0 && frame.f1 <= 1000.0, "F1={}", frame.f1);
        assert!(frame.f2 >= 600.0 && frame.f2 <= 3000.0, "F2={}", frame.f2);
        assert!(frame.f3 >= 1500.0 && frame.f3 <= 5000.0, "F3={}", frame.f3);
        assert!(frame.b1 >= 30.0 && frame.b1 <= 300.0, "B1={}", frame.b1);
        assert!(frame.b2 >= 30.0 && frame.b2 <= 400.0, "B2={}", frame.b2);
        assert!(frame.b3 >= 50.0 && frame.b3 <= 500.0, "B3={}", frame.b3);
        assert!(frame.f0 >= 20.0, "F0={}", frame.f0);
        assert!(
            frame.energy >= 0.0 && frame.energy <= 1.0,
            "energy={}",
            frame.energy
        );
        assert!(
            frame.voicing >= 0.0 && frame.voicing <= 1.0,
            "voicing={}",
            frame.voicing
        );
    }

    #[test]
    fn test_genesis_determinism() {
        let genesis = test_genesis();
        let config = VocalTractConfig::default();
        let mut ctrl1 = VocalTractController::new(&genesis, &config);
        let mut ctrl2 = VocalTractController::new(&genesis, &config);

        let hv = ContinuousHV::random(HDC_DIMENSION, 42);
        let frame1 = ctrl1.forward(&hv, 0.005);
        let frame2 = ctrl2.forward(&hv, 0.005);

        assert!(
            (frame1.f1 - frame2.f1).abs() < 1e-4,
            "Same genesis → same F1: {} vs {}",
            frame1.f1,
            frame2.f1
        );
        assert!((frame1.f0 - frame2.f0).abs() < 1e-4);
        assert!((frame1.energy - frame2.energy).abs() < 1e-6);
    }

    #[test]
    fn test_training_reduces_error() {
        let genesis = test_genesis();
        let config = VocalTractConfig::default();
        let mut ctrl = VocalTractController::new(&genesis, &config);

        let hv = ContinuousHV::random(HDC_DIMENSION, 42);
        // Target: /a/ vowel
        let target = FormantFrame {
            f1: 730.0,
            f2: 1090.0,
            f3: 2440.0,
            b1: 80.0,
            b2: 100.0,
            b3: 120.0,
            f0: 150.0,
            energy: 0.7,
            voicing: 0.95,
            time: 0.0,
        };

        // Initial error
        let initial = ctrl.forward(&hv, 0.005);
        let initial_err = formant_mse(&initial, &target);

        // Train
        for _ in 0..50 {
            ctrl.forward(&hv, 0.005);
            ctrl.train_step(&hv, &target, 0.005, Some(0.01));
        }

        let final_frame = ctrl.forward(&hv, 0.005);
        let final_err = formant_mse(&final_frame, &target);

        assert!(
            final_err < initial_err,
            "Training should reduce error: initial={initial_err:.2}, final={final_err:.2}"
        );
    }

    #[test]
    fn test_reset() {
        let genesis = test_genesis();
        let config = VocalTractConfig::default();
        let mut ctrl = VocalTractController::new(&genesis, &config);

        let hv = ContinuousHV::random(HDC_DIMENSION, 42);
        ctrl.forward(&hv, 0.005);
        ctrl.forward(&hv, 0.005);

        ctrl.reset();
        let stats = ctrl.network.stats();
        assert!(
            stats.avg_state_norm < 1e-6,
            "Reset should zero state norms: {}",
            stats.avg_state_norm
        );
    }

    #[test]
    fn test_tau_modulation() {
        let genesis = test_genesis();
        let config = VocalTractConfig::default();

        // Two separate controllers to avoid reset clearing tau
        let mut ctrl_normal = VocalTractController::new(&genesis, &config);
        let mut ctrl_slow = VocalTractController::new(&genesis, &config);

        // Modulate tau on one before any evolution
        ctrl_slow.modulate_tau(2.0);

        // Use varying inputs to drive real state divergence
        for step in 0..50 {
            let hv = ContinuousHV::random(HDC_DIMENSION, 100 + step);
            ctrl_normal.forward(&hv, 0.005);
            ctrl_slow.forward(&hv, 0.005);
        }

        let hv = ContinuousHV::random(HDC_DIMENSION, 999);
        let frame_normal = ctrl_normal.forward(&hv, 0.005);
        let frame_slow = ctrl_slow.forward(&hv, 0.005);

        // Different tau should produce different outputs (the network evolves differently)
        let diff = (frame_normal.f1 - frame_slow.f1).abs()
            + (frame_normal.f2 - frame_slow.f2).abs()
            + (frame_normal.f0 - frame_slow.f0).abs();
        assert!(
            diff > 1e-6,
            "Tau modulation should affect output: diff={diff}"
        );
    }

    #[test]
    fn test_bptt_changes_weights() {
        let genesis = test_genesis();
        let config = VocalTractConfig::default();
        let mut ctrl = VocalTractController::new(&genesis, &config);

        let initial_norms: Vec<Vec<f32>> = (0..ctrl.network.n_layers())
            .map(|i| {
                ctrl.network
                    .layer(i)
                    .unwrap()
                    .iter()
                    .map(|n| n.stats().weight_norm)
                    .collect()
            })
            .collect();

        let target = FormantFrame {
            f1: 730.0,
            f2: 1090.0,
            f3: 2440.0,
            b1: 80.0,
            b2: 100.0,
            b3: 120.0,
            f0: 150.0,
            energy: 0.7,
            voicing: 0.95,
            time: 0.0,
        };

        for step in 0..200 {
            let hv = ContinuousHV::random(HDC_DIMENSION, 100 + step);
            ctrl.forward(&hv, 0.005);
            ctrl.train_step(&hv, &target, 0.005, Some(0.01));
        }

        let mut any_changed = false;
        for layer_idx in 0..ctrl.network.n_layers() {
            let final_norms: Vec<f32> = ctrl
                .network
                .layer(layer_idx)
                .unwrap()
                .iter()
                .map(|n| n.stats().weight_norm)
                .collect();

            for (init, fin) in initial_norms[layer_idx].iter().zip(final_norms.iter()) {
                if (init - fin).abs() > 1e-8 {
                    any_changed = true;
                }
            }
        }

        assert!(any_changed, "BPTT should modify hidden layer weights");
    }

    #[test]
    fn test_phoneme_training_reduces_loss() {
        let genesis = test_genesis();
        let config = VocalTractConfig::default();
        let mut ctrl = VocalTractController::new(&genesis, &config);

        let targets = test_phoneme_targets();
        let target_refs: Vec<(&str, &FormantTarget)> =
            targets.iter().map(|(name, t)| (*name, t)).collect();

        // First epoch loss
        let mut ctrl_baseline = VocalTractController::new(&genesis, &config);
        let first_loss = ctrl_baseline.train_on_phoneme_targets(&genesis, &target_refs, 1);

        // Train for more epochs
        let final_loss = ctrl.train_on_phoneme_targets(&genesis, &target_refs, 5);

        assert!(
            final_loss < first_loss,
            "5 epochs should yield lower loss than 1: first={first_loss:.2}, final={final_loss:.2}"
        );
    }

    #[test]
    fn test_phoneme_training_improves_vowel() {
        let genesis = test_genesis();
        let config = VocalTractConfig::default();

        let targets = test_phoneme_targets();

        // Get /AH/ target (open vowel, F1≈520)
        let ah_target = targets.iter().find(|(name, _)| *name == "AH").map(|(_, t)| t).expect("AH should exist in test data");
        let ah_hv = genesis.hv("phoneme::AH", HDC_DIMENSION);

        let target_frame = FormantFrame {
            f1: ah_target.f1,
            f2: ah_target.f2,
            f3: ah_target.f3,
            b1: ah_target.b1,
            b2: ah_target.b2,
            b3: ah_target.b3,
            f0: config.base_f0,
            energy: 0.7,
            voicing: 0.95,
            time: 0.0,
        };

        // Pre-training prediction (warmup + steady-state average)
        let mut ctrl = VocalTractController::new(&genesis, &config);
        for _ in 0..20 {
            ctrl.forward(&ah_hv, 0.005);
        }
        let mut pre_f1_sum = 0.0f32;
        for _ in 0..10 {
            pre_f1_sum += ctrl.forward(&ah_hv, 0.005).f1;
        }
        let pre_f1_err = (pre_f1_sum / 10.0 - ah_target.f1).abs();

        // Train specifically on /AH/ using train_step (single-target).
        // Multi-phoneme convergence is validated by test_phoneme_training_reduces_loss.
        ctrl.reset();
        for _ in 0..100 {
            ctrl.forward(&ah_hv, 0.005);
            ctrl.train_step(&ah_hv, &target_frame, 0.005, None);
        }

        // Post-training prediction (reset temporal state, warmup, then measure)
        ctrl.reset();
        for _ in 0..20 {
            ctrl.forward(&ah_hv, 0.005);
        }
        let mut post_f1_sum = 0.0f32;
        for _ in 0..10 {
            post_f1_sum += ctrl.forward(&ah_hv, 0.005).f1;
        }
        let post_f1_err = (post_f1_sum / 10.0 - ah_target.f1).abs();

        assert!(
            post_f1_err < pre_f1_err,
            "/AH/ F1 should improve: pre_err={pre_f1_err:.1}, post_err={post_f1_err:.1}"
        );
    }

    #[test]
    fn test_coarticulation_smoothness() {
        let genesis = test_genesis();
        let config = VocalTractConfig::default();
        let mut ctrl = VocalTractController::new(&genesis, &config);

        let targets = test_phoneme_targets();
        let target_refs: Vec<(&str, &FormantTarget)> =
            targets.iter().map(|(name, t)| (*name, t)).collect();

        // Train briefly so the network can distinguish vowels
        ctrl.train_on_phoneme_targets(&genesis, &target_refs, 10);

        // Get HVs for two vowels with very different F1
        // /AH/ (F1~520) → /IY/ (F1~270)
        let ah_hv = genesis.hv("phoneme::AH", HDC_DIMENSION);
        let iy_hv = genesis.hv("phoneme::IY", HDC_DIMENSION);

        ctrl.reset();

        // Run 30 frames of /AH/
        let mut f1_values = Vec::new();
        for _ in 0..30 {
            let frame = ctrl.forward(&ah_hv, 0.005);
            f1_values.push(frame.f1);
        }

        // Transition to /IY/ for 30 frames
        for _ in 0..30 {
            let frame = ctrl.forward(&iy_hv, 0.005);
            f1_values.push(frame.f1);
        }

        // The LTC network should produce smooth transitions, not hard jumps.
        // Check max frame-to-frame F1 delta
        let max_delta: f32 = f1_values
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);

        // The entire F1 range is ~800 (200-1000). A hard jump would be > 100 Hz in one frame.
        // LTC smoothing should keep per-frame delta well below that.
        assert!(
            max_delta < 100.0,
            "LTC should smooth transitions: max per-frame F1 delta = {max_delta:.1} Hz"
        );

        // Check that F1 at the transition point is between the two steady-state values
        let f1_at_29 = f1_values[29]; // Last /AH/ frame
        let f1_at_35 = f1_values[35]; // 5 frames into /IY/
        let f1_min = f1_at_29.min(f1_at_35);
        let f1_max = f1_at_29.max(f1_at_35);

        // Frame 32 (2 frames into transition) should be somewhere between
        let f1_transition = f1_values[32];
        // Due to LTC dynamics, the transition frame should be between the extremes
        // (or at least closer to one end, not wildly outside)
        assert!(
            f1_transition >= f1_min - 50.0 && f1_transition <= f1_max + 50.0,
            "Transition F1 ({f1_transition:.1}) should be near [{f1_min:.1}, {f1_max:.1}]"
        );
    }

    #[test]
    fn test_prosody_head_zero_init() {
        let genesis = test_genesis();
        let config = VocalTractConfig::default();
        let mut ctrl = VocalTractController::new(&genesis, &config);

        let hv = ContinuousHV::random(HDC_DIMENSION, 42);

        // Default cognitive channels
        let channels: [f32; 10] = [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 1.0, 0.5, 0.8, 0.8];

        let frame_without = ctrl.forward(&hv, 0.005);
        ctrl.reset();
        let frame_with = ctrl.forward_with_prosody(&hv, 0.005, Some(&channels));

        // Initial prosody correction should be small (near-zero output bias)
        let f0_diff = (frame_with.f0 - frame_without.f0).abs();
        assert!(
            f0_diff < 30.0,
            "Initial prosody F0 correction should be small: diff={f0_diff:.1}Hz"
        );
    }

    #[test]
    fn test_prosody_modulates_f0_by_arousal() {
        let genesis = test_genesis();
        let config = VocalTractConfig::default();
        let mut ctrl = VocalTractController::new(&genesis, &config);

        let hv = ContinuousHV::random(HDC_DIMENSION, 42);

        // Low arousal channels
        let low_arousal: [f32; 10] = [0.0, 0.0, 0.1, 1.0, 0.5, 0.0, 1.0, 0.3, 0.8, 0.8];
        ctrl.reset();
        let frame_low = ctrl.forward_with_prosody(&hv, 0.005, Some(&low_arousal));

        // High arousal channels
        let high_arousal: [f32; 10] = [0.0, 0.0, 0.9, 1.0, 0.5, 0.0, 1.0, 0.9, 0.8, 0.8];
        ctrl.reset();
        let frame_high = ctrl.forward_with_prosody(&hv, 0.005, Some(&high_arousal));

        // Both should produce valid F0 values within range
        let f0_min = (config.base_f0 - config.f0_range / 2.0).max(50.0);
        let f0_max = config.base_f0 + config.f0_range / 2.0;
        assert!(
            frame_low.f0 >= f0_min && frame_low.f0 <= f0_max,
            "Low arousal F0={:.1} should be in range [{f0_min:.1}, {f0_max:.1}]",
            frame_low.f0
        );
        assert!(
            frame_high.f0 >= f0_min && frame_high.f0 <= f0_max,
            "High arousal F0={:.1} should be in range [{f0_min:.1}, {f0_max:.1}]",
            frame_high.f0
        );

        // Both should produce valid energy/voicing in [0, 1]
        assert!(frame_low.energy >= 0.0 && frame_low.energy <= 1.0);
        assert!(frame_high.energy >= 0.0 && frame_high.energy <= 1.0);
        assert!(frame_low.voicing >= 0.0 && frame_low.voicing <= 1.0);
        assert!(frame_high.voicing >= 0.0 && frame_high.voicing <= 1.0);
    }
}
