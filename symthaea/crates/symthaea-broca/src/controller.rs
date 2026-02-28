//! Language controller: HdcLtcUnifiedNetwork + weight-tied output projection.
//!
//! Direct analog of `VocalTractController` (`crates/symthaea-vocal-tract/src/controller.rs`).
//! Instead of projecting to 9D formant parameters, this projects to vocab-sized logits
//! via weight-tied similarity with token embeddings.
//!
//! # Forward step (one token)
//!
//! ```text
//! 1. input_hv = thought_hv ⊗ token_emb[prev_token] ⊗ permute(position_base, pos)
//! 2. network.evolve_closed_form(dt, &input_hv)
//! 3. output_hv = network.output().normalize()
//! 4. logits[i] = cosine_similarity(output_hv, token_emb[i])  // weight-tied
//! ```

use rayon::prelude::*;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{
    ContinuousHV, HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig, HDC_DIMENSION,
};

/// Configuration for the language controller.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LanguageControllerConfig {
    /// Number of layers in the HdcLtcUnifiedNetwork.
    pub network_layers: usize,
    /// Neurons per layer.
    pub neurons_per_layer: usize,
    /// Learning rate for BPTT training.
    pub learning_rate: f32,
    /// Vocabulary size (number of token embeddings).
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Time step per token (seconds). Can be dynamically adjusted for
    /// thermodynamic subjective time.
    pub dt_per_token: f32,
    /// Temperature scaling for logits before softmax/sampling.
    pub logit_temperature: f32,
}

impl Default for LanguageControllerConfig {
    fn default() -> Self {
        Self {
            network_layers: 3,
            neurons_per_layer: 8,
            learning_rate: 0.001,
            vocab_size: 256, // Minimal default, override for real use
            max_seq_len: 512,
            dt_per_token: 0.02, // 20ms per token step
            logit_temperature: 1.0,
        }
    }
}

/// Language controller wrapping an HdcLtcUnifiedNetwork + weight-tied token output.
///
/// Forward pass:
/// 1. Compose input: thought_hv ⊗ token_emb ⊗ pos_emb
/// 2. `network.evolve_closed_form(dt, &input_hv)`
/// 3. `output_hv = network.output().normalize()`
/// 4. Weight-tied logits: `logits[i] = similarity(output_hv, token_emb[i])`
pub struct LanguageController {
    /// The temporal dynamics engine — full 16,384D HDC-LTC.
    network: HdcLtcUnifiedNetwork,
    /// Token embeddings (vocab_size × 16,384D). Weight-tied with output.
    /// Growable for swarm vocabulary extension.
    token_embeddings: Vec<ContinuousHV>,
    /// Position base vector — permute(base, pos) for positional encoding.
    position_base: ContinuousHV,
    /// Current learning rate.
    learning_rate: f32,
    /// Configuration.
    config: LanguageControllerConfig,
    /// Current sequence position (reset between generations).
    current_pos: usize,
}

impl LanguageController {
    /// Create a new controller from a genesis seed and config.
    pub fn new(genesis: &GenesisSeed, config: &LanguageControllerConfig) -> Self {
        let neuron_config = UnifiedConfig {
            tau_base: 0.02,    // 20ms — matches token generation rate
            backbone_tau: 0.3, // Moderate state dependency for coherent sequences
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

        // Genesis-seeded token embeddings
        let token_embeddings: Vec<ContinuousHV> = (0..config.vocab_size)
            .map(|i| {
                ContinuousHV::from_genesis(
                    genesis,
                    &format!("broca::token_emb::{i}"),
                    HDC_DIMENSION,
                )
            })
            .collect();

        let position_base =
            ContinuousHV::from_genesis(genesis, "broca::position_base", HDC_DIMENSION);

        Self {
            network,
            token_embeddings,
            position_base,
            learning_rate: config.learning_rate,
            config: config.clone(),
            current_pos: 0,
        }
    }

    /// Compute logits for all tokens via weight-tied similarity.
    ///
    /// `logits[i] = cosine_similarity(output_hv, token_embeddings[i])`
    /// Parallelized with rayon for large vocabularies.
    pub fn compute_logits(&self, output_hv: &ContinuousHV) -> Vec<f32> {
        if self.token_embeddings.len() > 128 {
            // Parallel for large vocabularies
            self.token_embeddings
                .par_iter()
                .map(|emb| output_hv.similarity(emb))
                .collect()
        } else {
            self.token_embeddings
                .iter()
                .map(|emb| output_hv.similarity(emb))
                .collect()
        }
    }

    /// Forward step: evolve network with composed input, return logits.
    ///
    /// - `thought_hv`: encoded thought (16,384D)
    /// - `prev_token_id`: previous token ID (for autoregressive input)
    /// - `pos`: current sequence position
    pub fn forward_step(
        &mut self,
        thought_hv: &ContinuousHV,
        prev_token_id: u32,
        pos: usize,
    ) -> Vec<f32> {
        self.forward_step_with_dt(thought_hv, prev_token_id, pos, self.config.dt_per_token)
    }

    /// Forward step with explicit dt (for thermodynamic subjective time).
    pub fn forward_step_with_dt(
        &mut self,
        thought_hv: &ContinuousHV,
        prev_token_id: u32,
        pos: usize,
        dt: f32,
    ) -> Vec<f32> {
        // 1. Compose input: thought_hv ⊗ token_emb[prev_token] ⊗ permute(pos_base, pos)
        let token_emb = self.get_token_embedding(prev_token_id);
        let pos_emb = self.position_base.permute(pos);
        let input_hv = thought_hv.bind(&token_emb).bind(&pos_emb);

        // 2. Evolve network dynamics
        self.network.evolve_closed_form(dt, &input_hv);

        // 3. Get normalized output
        let output_hv = self.network.output().normalize();

        // 4. Weight-tied logits
        let mut logits = self.compute_logits(&output_hv);

        // Apply temperature scaling
        if (self.config.logit_temperature - 1.0).abs() > 1e-6 {
            let inv_temp = 1.0 / self.config.logit_temperature;
            for l in &mut logits {
                *l *= inv_temp;
            }
        }

        self.current_pos = pos + 1;
        logits
    }

    /// Get the current network output HV (for coherence monitoring).
    pub fn output_hv(&self) -> ContinuousHV {
        self.network.output().normalize()
    }

    /// Get token embedding, returning a zero vector for out-of-range IDs.
    fn get_token_embedding(&self, token_id: u32) -> ContinuousHV {
        self.token_embeddings
            .get(token_id as usize)
            .cloned()
            .unwrap_or_else(|| ContinuousHV::zero(HDC_DIMENSION))
    }

    /// Reset the network state (between generations).
    pub fn reset(&mut self) {
        self.network.reset();
        self.current_pos = 0;
    }

    /// Append a new token embedding (for swarm vocabulary extension).
    /// The embedding is algebraically composed from context if `context_hvs` is provided.
    pub fn append_token(
        &mut self,
        genesis: &GenesisSeed,
        name: &str,
        context_hvs: Option<&[&ContinuousHV]>,
    ) {
        let emb = if let Some(ctx) = context_hvs {
            // Algebraically compose from surrounding semantics
            if ctx.is_empty() {
                ContinuousHV::from_genesis(
                    genesis,
                    &format!("broca::token_emb::{name}"),
                    HDC_DIMENSION,
                )
            } else {
                ContinuousHV::bundle(ctx)
            }
        } else {
            ContinuousHV::from_genesis(genesis, &format!("broca::token_emb::{name}"), HDC_DIMENSION)
        };
        self.token_embeddings.push(emb);
    }

    /// Current vocabulary size (may grow via append_token).
    pub fn vocab_size(&self) -> usize {
        self.token_embeddings.len()
    }

    /// Get current sequence position.
    pub fn current_pos(&self) -> usize {
        self.current_pos
    }

    /// Get learning rate.
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Set learning rate.
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    /// Set logit temperature.
    pub fn set_temperature(&mut self, temp: f32) {
        self.config.logit_temperature = temp;
    }

    /// Get dt per token.
    pub fn dt_per_token(&self) -> f32 {
        self.config.dt_per_token
    }

    /// Set dt per token (for thermodynamic subjective time).
    pub fn set_dt_per_token(&mut self, dt: f32) {
        self.config.dt_per_token = dt;
    }

    /// Get reference to the underlying network (for BPTT training).
    pub fn network(&self) -> &HdcLtcUnifiedNetwork {
        &self.network
    }

    /// Get mutable reference to the underlying network (for BPTT training).
    pub fn network_mut(&mut self) -> &mut HdcLtcUnifiedNetwork {
        &mut self.network
    }

    /// Get reference to token embeddings.
    pub fn token_embeddings(&self) -> &[ContinuousHV] {
        &self.token_embeddings
    }

    /// Get mutable reference to token embeddings (for training).
    pub fn token_embeddings_mut(&mut self) -> &mut Vec<ContinuousHV> {
        &mut self.token_embeddings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-broca-controller")
    }

    fn test_config() -> LanguageControllerConfig {
        LanguageControllerConfig {
            network_layers: 2,
            neurons_per_layer: 4,
            vocab_size: 32,
            max_seq_len: 16,
            ..Default::default()
        }
    }

    #[test]
    fn test_controller_creation() {
        let genesis = test_genesis();
        let config = test_config();
        let ctrl = LanguageController::new(&genesis, &config);
        assert_eq!(ctrl.vocab_size(), 32);
        assert_eq!(ctrl.current_pos(), 0);
    }

    #[test]
    fn test_forward_step_produces_logits() {
        let genesis = test_genesis();
        let config = test_config();
        let mut ctrl = LanguageController::new(&genesis, &config);

        let thought_hv = ContinuousHV::from_genesis(&genesis, "test_thought", HDC_DIMENSION);
        let logits = ctrl.forward_step(&thought_hv, 0, 0);

        assert_eq!(logits.len(), 32);
        // Logits should be finite
        assert!(
            logits.iter().all(|l| l.is_finite()),
            "all logits should be finite"
        );
        // At least some logits should be non-zero (network produces non-trivial output)
        assert!(
            logits.iter().any(|l| l.abs() > 1e-10),
            "should have non-zero logits"
        );
    }

    #[test]
    fn test_forward_determinism() {
        let genesis = test_genesis();
        let config = test_config();

        let thought_hv = ContinuousHV::from_genesis(&genesis, "test_thought", HDC_DIMENSION);

        let mut ctrl1 = LanguageController::new(&genesis, &config);
        let logits1 = ctrl1.forward_step(&thought_hv, 0, 0);

        let mut ctrl2 = LanguageController::new(&genesis, &config);
        let logits2 = ctrl2.forward_step(&thought_hv, 0, 0);

        for (a, b) in logits1.iter().zip(logits2.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "Same genesis should produce identical logits: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_different_tokens_different_logits() {
        let genesis = test_genesis();
        let config = test_config();
        let mut ctrl = LanguageController::new(&genesis, &config);

        let thought_hv = ContinuousHV::from_genesis(&genesis, "test_thought", HDC_DIMENSION);

        let logits_a = ctrl.forward_step(&thought_hv, 5, 0);
        ctrl.reset();
        let logits_b = ctrl.forward_step(&thought_hv, 10, 0);

        let diff: f32 = logits_a
            .iter()
            .zip(logits_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.01,
            "Different prev tokens should yield different logits: diff={diff}"
        );
    }

    #[test]
    fn test_different_positions_different_logits() {
        let genesis = test_genesis();
        let config = test_config();
        let mut ctrl = LanguageController::new(&genesis, &config);

        let thought_hv = ContinuousHV::from_genesis(&genesis, "test_thought", HDC_DIMENSION);

        let logits_a = ctrl.forward_step(&thought_hv, 5, 0);
        ctrl.reset();
        let logits_b = ctrl.forward_step(&thought_hv, 5, 3);

        let diff: f32 = logits_a
            .iter()
            .zip(logits_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.01,
            "Different positions should yield different logits: diff={diff}"
        );
    }

    #[test]
    fn test_reset_clears_state() {
        let genesis = test_genesis();
        let config = test_config();
        let mut ctrl = LanguageController::new(&genesis, &config);

        let thought_hv = ContinuousHV::from_genesis(&genesis, "test_thought", HDC_DIMENSION);
        let _ = ctrl.forward_step(&thought_hv, 5, 0);
        assert_eq!(ctrl.current_pos(), 1);

        ctrl.reset();
        assert_eq!(ctrl.current_pos(), 0);
    }

    #[test]
    fn test_append_token() {
        let genesis = test_genesis();
        let config = test_config();
        let mut ctrl = LanguageController::new(&genesis, &config);
        assert_eq!(ctrl.vocab_size(), 32);

        ctrl.append_token(&genesis, "new_concept", None);
        assert_eq!(ctrl.vocab_size(), 33);
    }

    #[test]
    fn test_temperature_scaling() {
        let genesis = test_genesis();
        let config = test_config();

        let thought_hv = ContinuousHV::from_genesis(&genesis, "test_thought", HDC_DIMENSION);

        let mut ctrl1 = LanguageController::new(&genesis, &config);
        ctrl1.set_temperature(1.0);
        let logits_t1 = ctrl1.forward_step(&thought_hv, 0, 0);

        let mut ctrl2 = LanguageController::new(&genesis, &config);
        ctrl2.set_temperature(0.5);
        let logits_t05 = ctrl2.forward_step(&thought_hv, 0, 0);

        // With temp=0.5, logits should be 2x the temp=1.0 values
        for (a, b) in logits_t1.iter().zip(logits_t05.iter()) {
            if a.abs() > 1e-6 {
                assert!(
                    (b / a - 2.0).abs() < 0.01,
                    "temp=0.5 should double logits: {a} -> {b}"
                );
            }
        }
    }
}
