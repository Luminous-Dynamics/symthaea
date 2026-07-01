// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC morphological transfer: flight → humanoid concept projection.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{
    ContinuousHV, HDC_DIMENSION, HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig,
};

use crate::controller::HumanoidController;
use crate::training::{EpisodeMetrics, HumanoidTrainer};
use crate::types::HumanoidConfig;

#[derive(Debug, Clone)]
pub struct ChannelMapping {
    pub source_ch: usize,
    pub target_ch: usize,
    pub weight: f32,
}

pub struct MorphologicalTransfer {
    source_genesis: GenesisSeed,
    target_genesis: GenesisSeed,
    channel_mapping: Vec<ChannelMapping>,
    blend_factor: f32,
}

impl MorphologicalTransfer {
    pub fn flight_to_humanoid() -> Self {
        let channel_mapping = vec![
            ChannelMapping {
                source_ch: 2,
                target_ch: 0,
                weight: 1.8,
            },
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

    fn build_morph_transform(&self) -> ContinuousHV {
        let dim = HDC_DIMENSION;
        let mut pairs: Vec<ContinuousHV> = Vec::with_capacity(self.channel_mapping.len());

        for mapping in &self.channel_mapping {
            let source_label = format!("channel_base_{}", mapping.source_ch);
            let target_label = format!("channel_base_{}", mapping.target_ch);

            let source_base = self.source_genesis.hv(&source_label, dim);
            let target_base = self.target_genesis.hv(&target_label, dim);

            let pair = source_base.bind(&target_base).scale(mapping.weight);
            pairs.push(pair);
        }

        if pairs.is_empty() {
            return ContinuousHV::zero(dim);
        }

        let refs: Vec<&ContinuousHV> = pairs.iter().collect();
        ContinuousHV::bundle(&refs).normalize()
    }

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

    pub fn initialize_humanoid(
        &self,
        source_network: &HdcLtcUnifiedNetwork,
        target_config: &HumanoidConfig,
    ) -> HumanoidController {
        let concepts = self.extract_concepts(source_network);
        let mut controller = HumanoidController::new(&self.target_genesis, target_config);
        self.inject_concepts(&mut controller, &concepts, target_config);
        controller
    }

    fn inject_concepts(
        &self,
        controller: &mut HumanoidController,
        concepts: &[(usize, usize, ContinuousHV)],
        config: &HumanoidConfig,
    ) {
        // Fix: Dynamically match layer specifications directly from target configuration profiles
        let target_network_config = UnifiedNetworkConfig {
            layer_sizes: vec![config.neurons_per_layer; config.network_layers],
            neuron_config: UnifiedConfig {
                tau_base: 0.025,
                backbone_tau: 0.3,
                dimension: HDC_DIMENSION,
                learning_rate: config.learning_rate,
                ..UnifiedConfig::default()
            },
            use_layer_binding: true,
            skip_connections: false,
        };

        let mut target_network =
            HdcLtcUnifiedNetwork::from_genesis(target_network_config, &self.target_genesis);

        for (layer_idx, neuron_idx, concept) in concepts {
            if let Some(layer) = target_network.layer_mut(*layer_idx) {
                if *neuron_idx < layer.len() {
                    let weight = layer[*neuron_idx].weight_hv_mut();
                    weight.lerp_in_place(concept, 1.0 - self.blend_factor, self.blend_factor);
                }
            }
        }

        *controller = HumanoidController::from_network(target_network, &self.target_genesis);
    }

    pub fn num_shared_channels(&self) -> usize {
        self.channel_mapping.len()
    }

    pub fn blend_factor(&self) -> f32 {
        self.blend_factor
    }
}

pub fn transfer_learning_comparison(
    source_network: &HdcLtcUnifiedNetwork,
    n_episodes: usize,
) -> (Vec<EpisodeMetrics>, Vec<EpisodeMetrics>) {
    let config = HumanoidConfig {
        num_episodes: n_episodes,
        steps_per_episode: 300,
        ..HumanoidConfig::default()
    };

    let transfer = MorphologicalTransfer::flight_to_humanoid();
    let transfer_controller = transfer.initialize_humanoid(source_network, &config);
    let mut transfer_trainer =
        HumanoidTrainer::with_controller(config.clone(), transfer_controller);
    let transfer_metrics = transfer_trainer.train();

    let mut random_trainer = HumanoidTrainer::new(config);
    let random_metrics = random_trainer.train();

    (transfer_metrics, random_metrics)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trained_flight_network() -> HdcLtcUnifiedNetwork {
        let genesis = GenesisSeed::from_phrase("symthaea-multirotor-quadrotor");
        let config = UnifiedNetworkConfig {
            layer_sizes: vec![4; 2],
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

        let sensor = ContinuousHV::random(HDC_DIMENSION, 42);
        let cmd = controller.forward(&sensor, 0.025);

        assert_eq!(cmd.torques.len(), 21);
    }
}
