// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC morphological transfer: flight → humanoid concept projection.
//!
//! Transfers learned temporal dynamics from a trained flight controller's
//! `HdcLtcUnifiedNetwork` to a humanoid controller by exploiting shared
//! semantic sensor channels (altitude, orientation, velocity).
//!
//! ## Transfer Mechanism
//!
//! 1. **Extract**: For each neuron in the trained flight network, get its
//!    `weight_hv` (learned temporal dynamics pattern).
//! 2. **Project**: Bind each weight HV with a morphology transform vector:
//!    `concept = weight_hv ⊗ morph_transform`, where `morph_transform` is
//!    the bundled sum of `source_base ⊗ target_base` for shared channels.
//! 3. **Inject**: Initialize humanoid network from genesis, then blend:
//!    `new_weight = lerp(genesis_weight, transferred_concept, blend_factor)`.
//! 4. **Output layer**: Only the network backbone transfers; output projection
//!    stays fresh (4D flight → 21D humanoid incompatible).
//!
//! ## Shared Concepts
//!
//! | Concept          | Flight Channel(s) | Humanoid Channel(s)        |
//! |------------------|-------------------|----------------------------|
//! | Altitude         | `pos_z` (idx 2)   | `root_height` (idx 0)      |
//! | Orientation      | `quat_w/x/y/z` (3-6) | `root_quaternion` (1-4) |
//! | Linear velocity  | `vel_x/y/z` (7-9) | `root_linear_velocity` (26-28) |
//! | Angular velocity | `angvel_x/y/z` (10-12) | `root_angular_velocity` (29-31) |

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{
    ContinuousHV, HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig, HDC_DIMENSION,
};

use crate::controller::HumanoidController;
use crate::training::{EpisodeMetrics, HumanoidTrainer};
use crate::types::HumanoidConfig;

/// A channel mapping entry: (source_channel_idx, target_channel_idx, weight).
///
/// Weight controls the contribution of this channel pair to the morphology
/// transform vector. Higher weight = stronger concept transfer for that channel.
#[derive(Debug, Clone)]
pub struct ChannelMapping {
    /// Index in the source (flight) sensor channel array.
    pub source_ch: usize,
    /// Index in the target (humanoid) sensor channel array.
    pub target_ch: usize,
    /// Contribution weight for this channel pair (typically 1.0).
    pub weight: f32,
}

/// HDC morphological transfer from a source domain to the humanoid domain.
///
/// Uses hyperdimensional binding operations to project learned temporal
/// dynamics concepts between morphologically distinct embodiments.
pub struct MorphologicalTransfer {
    /// Genesis seed for the source domain (flight).
    source_genesis: GenesisSeed,
    /// Genesis seed for the target domain (humanoid).
    target_genesis: GenesisSeed,
    /// Channel mapping: which source channels correspond to which target channels.
    channel_mapping: Vec<ChannelMapping>,
    /// Blend factor for lerp(genesis, transferred, blend). Default: 0.5.
    blend_factor: f32,
}

impl MorphologicalTransfer {
    /// Create a transfer mapping from flight → humanoid using shared concepts.
    ///
    /// Shared channels:
    /// - Altitude: flight pos_z (2) → humanoid root_height (0)
    /// - Orientation: flight quat_wxyz (3-6) → humanoid root_quaternion (1-4)
    /// - Linear velocity: flight vel_xyz (7-9) → humanoid root_linear_velocity (26-28)
    /// - Angular velocity: flight angvel_xyz (10-12) → humanoid root_angular_velocity (29-31)
    pub fn flight_to_humanoid() -> Self {
        // Channel weights reflect information content for bipedal control:
        // - Orientation is most critical (4D quaternion encodes full body attitude)
        // - Altitude/height is essential for standing reward
        // - Angular velocity carries learned stabilization dynamics
        // - Linear velocity is less informative (flight ≠ walking kinematics)
        let channel_mapping = vec![
            // Altitude: high weight — directly maps to standing reward
            ChannelMapping {
                source_ch: 2,
                target_ch: 0,
                weight: 1.8,
            },
            // Orientation (4 channels): highest weight — attitude control is the core
            // shared skill between hovering and standing
            ChannelMapping {
                source_ch: 3,
                target_ch: 1,
                weight: 2.0,
            },
            ChannelMapping {
                source_ch: 4,
                target_ch: 2,
                weight: 2.0,
            },
            ChannelMapping {
                source_ch: 5,
                target_ch: 3,
                weight: 2.0,
            },
            ChannelMapping {
                source_ch: 6,
                target_ch: 4,
                weight: 2.0,
            },
            // Linear velocity (3 channels): moderate weight
            ChannelMapping {
                source_ch: 7,
                target_ch: 26,
                weight: 0.8,
            },
            ChannelMapping {
                source_ch: 8,
                target_ch: 27,
                weight: 0.8,
            },
            ChannelMapping {
                source_ch: 9,
                target_ch: 28,
                weight: 0.8,
            },
            // Angular velocity (3 channels): high weight — stabilization dynamics
            // are the key transferable skill from hover control
            ChannelMapping {
                source_ch: 10,
                target_ch: 29,
                weight: 1.5,
            },
            ChannelMapping {
                source_ch: 11,
                target_ch: 30,
                weight: 1.5,
            },
            ChannelMapping {
                source_ch: 12,
                target_ch: 31,
                weight: 1.5,
            },
        ];

        Self {
            source_genesis: GenesisSeed::from_phrase("symthaea-multirotor-quadrotor"),
            target_genesis: GenesisSeed::from_phrase("symthaea-humanoid-dmc"),
            channel_mapping,
            blend_factor: 0.5,
        }
    }

    /// Create a custom transfer with explicit genesis seeds and channel mapping.
    pub fn new(
        source_genesis: GenesisSeed,
        target_genesis: GenesisSeed,
        channel_mapping: Vec<ChannelMapping>,
        blend_factor: f32,
    ) -> Self {
        Self {
            source_genesis,
            target_genesis,
            channel_mapping,
            blend_factor: blend_factor.clamp(0.0, 1.0),
        }
    }

    /// Build the morphology transform vector.
    ///
    /// For each shared channel pair, bind the source base vector with the
    /// target base vector (weighted), then bundle all pairs into a single
    /// transform HV.
    fn build_morph_transform(&self) -> ContinuousHV {
        let dim = HDC_DIMENSION;
        let mut pairs: Vec<ContinuousHV> = Vec::with_capacity(self.channel_mapping.len());

        for mapping in &self.channel_mapping {
            let source_label = format!("channel_base_{}", mapping.source_ch);
            let target_label = format!("channel_base_{}", mapping.target_ch);

            let source_base = self.source_genesis.hv(&source_label, dim);
            let target_base = self.target_genesis.hv(&target_label, dim);

            // Bind source ⊗ target, scaled by weight
            let pair = source_base.bind(&target_base).scale(mapping.weight);
            pairs.push(pair);
        }

        if pairs.is_empty() {
            return ContinuousHV::zero(dim);
        }

        // Bundle all pairs → single transform HV
        let refs: Vec<&ContinuousHV> = pairs.iter().collect();
        ContinuousHV::bundle(&refs).normalize()
    }

    /// Extract concept HVs from a trained source network.
    ///
    /// For each neuron in each layer, extracts `weight_hv` and binds it with
    /// the morphology transform to project it into the target domain.
    ///
    /// Returns a flat list of (layer_idx, neuron_idx, concept_hv) tuples.
    pub fn extract_concepts(
        &self,
        source_network: &HdcLtcUnifiedNetwork,
    ) -> Vec<(usize, usize, ContinuousHV)> {
        let morph_transform = self.build_morph_transform();
        let mut concepts = Vec::new();

        for layer_idx in 0..source_network.n_layers() {
            if let Some(layer) = source_network.layer(layer_idx) {
                for (neuron_idx, neuron) in layer.iter().enumerate() {
                    let concept = neuron.weight_hv_ref().bind(&morph_transform);
                    concepts.push((layer_idx, neuron_idx, concept));
                }
            }
        }

        concepts
    }

    /// Initialize a humanoid controller with transferred concepts from a source network.
    ///
    /// 1. Creates a fresh humanoid controller from target genesis
    /// 2. Extracts concepts from the source network
    /// 3. Blends transferred concepts into the target network's weight HVs
    /// 4. Output projection stays fresh (4D → 21D incompatible)
    pub fn initialize_humanoid(
        &self,
        source_network: &HdcLtcUnifiedNetwork,
        target_config: &HumanoidConfig,
    ) -> HumanoidController {
        let concepts = self.extract_concepts(source_network);

        // Create a fresh controller from target genesis
        let mut controller = HumanoidController::new(&self.target_genesis, target_config);

        // Access the target network and blend in transferred concepts
        self.inject_concepts(&mut controller, &concepts);

        controller
    }

    /// Inject extracted concepts into a humanoid controller's network.
    fn inject_concepts(
        &self,
        controller: &mut HumanoidController,
        concepts: &[(usize, usize, ContinuousHV)],
    ) {
        // We need mutable access to the network inside the controller.
        // The controller exposes network() as &HdcLtcUnifiedNetwork.
        // We'll build a new network with the blended weights.
        let target_config = UnifiedNetworkConfig {
            layer_sizes: vec![8; 3], // 3×8 default humanoid config
            neuron_config: UnifiedConfig {
                tau_base: 0.025,
                backbone_tau: 0.3,
                dimension: HDC_DIMENSION,
                learning_rate: 0.0005,
                ..UnifiedConfig::default()
            },
            use_layer_binding: true,
            skip_connections: false,
        };

        let mut target_network =
            HdcLtcUnifiedNetwork::from_genesis(target_config, &self.target_genesis);

        // For each extracted concept, find the corresponding target neuron
        // and blend the concept into its weight HV
        for (layer_idx, neuron_idx, concept) in concepts {
            if let Some(layer) = target_network.layer_mut(*layer_idx) {
                if *neuron_idx < layer.len() {
                    let weight = layer[*neuron_idx].weight_hv_mut();
                    // lerp: new_weight = (1-blend) * genesis_weight + blend * concept
                    weight.lerp_in_place(concept, 1.0 - self.blend_factor, self.blend_factor);
                }
            }
        }

        // Replace the controller's network with the blended one.
        // We reconstruct the controller since network is private.
        *controller = HumanoidController::from_network(target_network, &self.target_genesis);
    }

    /// Number of shared channel pairs in this transfer mapping.
    pub fn num_shared_channels(&self) -> usize {
        self.channel_mapping.len()
    }

    /// Get the blend factor.
    pub fn blend_factor(&self) -> f32 {
        self.blend_factor
    }
}

/// Compare transfer learning vs random initialization.
///
/// Trains two humanoid controllers (one transfer-initialized, one random genesis)
/// for `n_episodes` each and returns their metrics side-by-side.
///
/// Uses 300 steps/episode (7.5s) for more reliable convergence data.
///
/// Returns `(transfer_metrics, random_metrics)`.
pub fn transfer_learning_comparison(
    source_network: &HdcLtcUnifiedNetwork,
    n_episodes: usize,
) -> (Vec<EpisodeMetrics>, Vec<EpisodeMetrics>) {
    let config = HumanoidConfig {
        num_episodes: n_episodes,
        steps_per_episode: 300,
        ..HumanoidConfig::default()
    };

    // Transfer-initialized training
    let transfer = MorphologicalTransfer::flight_to_humanoid();
    let transfer_controller = transfer.initialize_humanoid(source_network, &config);
    let mut transfer_trainer =
        HumanoidTrainer::with_controller(config.clone(), transfer_controller);
    let transfer_metrics = transfer_trainer.train();

    // Random baseline training
    let mut random_trainer = HumanoidTrainer::new(config);
    let random_metrics = random_trainer.train();

    (transfer_metrics, random_metrics)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a small trained flight network for testing.
    fn make_trained_flight_network() -> HdcLtcUnifiedNetwork {
        let genesis = GenesisSeed::from_phrase("symthaea-multirotor-quadrotor");
        let config = UnifiedNetworkConfig {
            layer_sizes: vec![4; 2], // 2×4 flight config
            neuron_config: UnifiedConfig {
                tau_base: 0.01,
                backbone_tau: 0.5,
                dimension: HDC_DIMENSION,
                learning_rate: 0.001,
                ..UnifiedConfig::default()
            },
            use_layer_binding: true,
            skip_connections: false,
        };

        let mut network = HdcLtcUnifiedNetwork::from_genesis(config, &genesis);

        // Simulate training by evolving with several inputs
        for i in 0..20 {
            let input = ContinuousHV::random(HDC_DIMENSION, i as u64);
            network.evolve_closed_form(0.01, &input);
        }

        network
    }

    #[test]
    fn test_transfer_creates_valid_controller() {
        let source = make_trained_flight_network();
        let transfer = MorphologicalTransfer::flight_to_humanoid();
        let config = HumanoidConfig::default();

        let mut controller = transfer.initialize_humanoid(&source, &config);

        // Forward pass should produce finite 21D output
        let sensor = ContinuousHV::random(HDC_DIMENSION, 42);
        let cmd = controller.forward(&sensor, 0.025);

        assert_eq!(cmd.torques.len(), 21);
        for (i, &t) in cmd.torques.iter().enumerate() {
            assert!(t.is_finite(), "Torque {i} must be finite, got {t}");
            assert!(
                t >= -1.0 && t <= 1.0,
                "Torque {i} must be in [-1, 1], got {t}"
            );
        }
    }

    #[test]
    fn test_transfer_determinism() {
        let source = make_trained_flight_network();
        let transfer = MorphologicalTransfer::flight_to_humanoid();
        let config = HumanoidConfig::default();

        let mut ctrl1 = transfer.initialize_humanoid(&source, &config);
        let mut ctrl2 = transfer.initialize_humanoid(&source, &config);

        let sensor = ContinuousHV::random(HDC_DIMENSION, 42);
        let cmd1 = ctrl1.forward(&sensor, 0.025);
        let cmd2 = ctrl2.forward(&sensor, 0.025);

        for i in 0..21 {
            assert!(
                (cmd1.torques[i] - cmd2.torques[i]).abs() < 1e-5,
                "Same source → same output for joint {i}: {} vs {}",
                cmd1.torques[i],
                cmd2.torques[i]
            );
        }
    }

    #[test]
    fn test_transfer_concept_similarity() {
        let source = make_trained_flight_network();
        let transfer = MorphologicalTransfer::flight_to_humanoid();

        let concepts = transfer.extract_concepts(&source);
        assert!(!concepts.is_empty(), "Should extract at least one concept");

        // Verify all concepts are finite and non-zero
        for (layer, neuron, concept) in &concepts {
            let norm = concept.norm();
            assert!(
                norm.is_finite() && norm > 0.0,
                "Concept ({layer}, {neuron}) should have non-zero finite norm, got {norm}"
            );
        }

        // Shared channels should produce non-trivial morph transform
        let morph = transfer.build_morph_transform();
        assert!(
            morph.norm() > 0.1,
            "Morph transform should be non-trivial: norm = {}",
            morph.norm()
        );
    }

    #[test]
    fn test_transfer_differs_from_random() {
        let source = make_trained_flight_network();
        let transfer = MorphologicalTransfer::flight_to_humanoid();
        let config = HumanoidConfig::default();

        // Transfer-initialized controller
        let mut transfer_ctrl = transfer.initialize_humanoid(&source, &config);

        // Random-initialized controller (same genesis)
        let genesis = GenesisSeed::from_phrase("symthaea-humanoid-dmc");
        let mut random_ctrl = HumanoidController::new(&genesis, &config);

        // Run multiple forward steps to let weight differences accumulate
        // in the network state (weight_hv affects evolve_closed_form dynamics)
        let sensor = ContinuousHV::random(HDC_DIMENSION, 42);
        for _ in 0..10 {
            transfer_ctrl.forward(&sensor, 0.025);
            random_ctrl.forward(&sensor, 0.025);
        }

        transfer_ctrl.forward(&sensor, 0.025);
        random_ctrl.forward(&sensor, 0.025);

        // Check network state divergence directly
        let transfer_output = transfer_ctrl.network().output();
        let random_output = random_ctrl.network().output();
        let sim = transfer_output.similarity(&random_output);

        // Transferred network should have diverged from random genesis
        assert!(
            sim < 0.999,
            "Transfer-initialized network state should diverge from random: similarity = {sim}"
        );
    }

    #[test]
    fn test_morph_transform_properties() {
        let transfer = MorphologicalTransfer::flight_to_humanoid();
        assert_eq!(transfer.num_shared_channels(), 11);
        assert!((transfer.blend_factor() - 0.5).abs() < 1e-6);

        let morph = transfer.build_morph_transform();
        // Should be normalized (norm ≈ 1.0)
        assert!(
            (morph.norm() - 1.0).abs() < 0.01,
            "Morph transform should be normalized: {}",
            morph.norm()
        );
    }

    #[test]
    fn test_custom_transfer() {
        let source_genesis = GenesisSeed::from_phrase("custom-source");
        let target_genesis = GenesisSeed::from_phrase("custom-target");
        let mapping = vec![
            ChannelMapping {
                source_ch: 0,
                target_ch: 0,
                weight: 2.0,
            },
            ChannelMapping {
                source_ch: 1,
                target_ch: 1,
                weight: 1.0,
            },
        ];

        let transfer = MorphologicalTransfer::new(source_genesis, target_genesis, mapping, 0.7);
        assert_eq!(transfer.num_shared_channels(), 2);
        assert!((transfer.blend_factor() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_empty_mapping() {
        let transfer = MorphologicalTransfer::new(
            GenesisSeed::from_phrase("src"),
            GenesisSeed::from_phrase("tgt"),
            vec![],
            0.5,
        );

        let morph = transfer.build_morph_transform();
        assert!(
            morph.norm() < 1e-6,
            "Empty mapping should produce zero transform"
        );
    }
}
