// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Read-only topology adapters for active Symthaea engines.
//!
//! Adapters in this crate wrap active engine values rather than modifying their
//! runtime implementations. This keeps topology inspection observational and
//! avoids forcing the active foundation crate to depend on research tooling.

use symthaea_core::hdc::hdc_ltc_unified::HdcLtcUnifiedNetwork;
use symthaea_neuroarch_types::{
    BudgetClass, CircuitDescriptor, CircuitId, CircuitImplementation, EdgeDescriptor,
    EdgeDirection, EdgeId, EdgeTransform, InputMergePolicy, NeuroTopology, RecurrenceKind,
    SemanticChannel, TOPOLOGY_SCHEMA_VERSION, TimescaleClass, TopologyDescriptor, TopologyError,
};

const ACTIVE_HDC_LTC_LAYER: &str = "active_core:hdc_ltc_layer";

/// Read-only topology view of the HDC/LTC implementation actually exported by
/// `symthaea-core` and consumed across active Symthaea domains.
#[derive(Debug, Clone, Copy)]
pub struct ActiveCoreHdcLtcTopology<'a> {
    network: &'a HdcLtcUnifiedNetwork,
}

impl<'a> ActiveCoreHdcLtcTopology<'a> {
    pub fn new(network: &'a HdcLtcUnifiedNetwork) -> Self {
        Self { network }
    }

    pub fn network(&self) -> &'a HdcLtcUnifiedNetwork {
        self.network
    }
}

impl NeuroTopology for ActiveCoreHdcLtcTopology<'_> {
    fn topology_descriptor(&self) -> Result<TopologyDescriptor, TopologyError> {
        let config = self.network.config();
        if config.layer_sizes.is_empty() {
            return Err(TopologyError::NoComputationalCircuits);
        }

        let dim = u64::try_from(config.neuron_config.dimension)
            .map_err(|_| TopologyError::DimensionOverflow)?;
        let timescale = TimescaleClass::from_seconds(config.neuron_config.tau_base)?;

        let mut circuits = Vec::with_capacity(config.layer_sizes.len() + 1);
        circuits.push(CircuitDescriptor {
            id: CircuitId(0),
            role: "external_input".to_string(),
            timescale_class: TimescaleClass::Stateless,
            state_dimension: dim,
            unit_count: 1,
            implementation: CircuitImplementation::ExternalInput,
            input_merge_policy: InputMergePolicy::None,
            modulation_profile: None,
        });

        for (idx, &layer_size) in config.layer_sizes.iter().enumerate() {
            let unit_count =
                u64::try_from(layer_size).map_err(|_| TopologyError::DimensionOverflow)?;
            let id = u32::try_from(idx + 1).map_err(|_| TopologyError::DimensionOverflow)?;

            circuits.push(CircuitDescriptor {
                id: CircuitId(id),
                role: format!("layer:{idx}"),
                timescale_class: timescale.clone(),
                state_dimension: dim,
                unit_count,
                implementation: CircuitImplementation::Named(ACTIVE_HDC_LTC_LAYER.to_string()),
                input_merge_policy: if idx > 0 && config.skip_connections {
                    InputMergePolicy::BundleAll
                } else {
                    InputMergePolicy::Single
                },
                modulation_profile: None,
            });
        }

        let mut edges = Vec::new();
        let mut next_edge_id = 1u64;
        edges.push(EdgeDescriptor {
            id: EdgeId(next_edge_id),
            source: CircuitId(0),
            target: CircuitId(1),
            channel: SemanticChannel::ExternalInput,
            direction: EdgeDirection::Directed,
            recurrence: RecurrenceKind::FeedForward,
            budget_class: BudgetClass::LegacyUnbounded,
            transform: EdgeTransform::Direct,
        });
        next_edge_id += 1;

        for layer_idx in 1..config.layer_sizes.len() {
            let source =
                CircuitId(u32::try_from(layer_idx).map_err(|_| TopologyError::DimensionOverflow)?);
            let target = CircuitId(
                u32::try_from(layer_idx + 1).map_err(|_| TopologyError::DimensionOverflow)?,
            );

            edges.push(EdgeDescriptor {
                id: EdgeId(next_edge_id),
                source,
                target,
                channel: SemanticChannel::InterLayer,
                direction: EdgeDirection::Directed,
                recurrence: RecurrenceKind::FeedForward,
                budget_class: BudgetClass::LegacyUnbounded,
                transform: if config.use_layer_binding {
                    EdgeTransform::Bind
                } else {
                    EdgeTransform::Direct
                },
            });
            next_edge_id += 1;

            if config.skip_connections {
                edges.push(EdgeDescriptor {
                    id: EdgeId(next_edge_id),
                    source: CircuitId(0),
                    target,
                    channel: SemanticChannel::SkipInput,
                    direction: EdgeDirection::Directed,
                    recurrence: RecurrenceKind::FeedForward,
                    budget_class: BudgetClass::LegacyUnbounded,
                    transform: EdgeTransform::Direct,
                });
                next_edge_id += 1;
            }
        }

        let descriptor = TopologyDescriptor {
            schema_version: TOPOLOGY_SCHEMA_VERSION,
            circuits,
            edges,
            allow_parallel_channels: false,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::genesis::GenesisSeed;
    use symthaea_core::hdc::hdc_ltc_unified::{
        NetworkStateSnapshot, UnifiedActivation, UnifiedConfig, UnifiedNetworkConfig,
    };
    use symthaea_core::hdc::unified_hv::ContinuousHV;

    fn config() -> UnifiedNetworkConfig {
        UnifiedNetworkConfig {
            layer_sizes: vec![2, 3, 2],
            neuron_config: UnifiedConfig {
                dimension: 128,
                tau_base: 0.1,
                ..UnifiedConfig::default()
            },
            use_layer_binding: true,
            skip_connections: false,
        }
    }

    fn commitment(
        config: UnifiedNetworkConfig,
        seed: u64,
    ) -> symthaea_neuroarch_types::TopologyCommitment {
        let network = HdcLtcUnifiedNetwork::new(config, seed);
        ActiveCoreHdcLtcTopology::new(&network)
            .topology_commitment()
            .unwrap()
    }

    #[test]
    fn active_core_topology_is_seed_independent() {
        assert_eq!(commitment(config(), 1), commitment(config(), 999));
    }

    #[test]
    fn active_core_topology_is_initialization_lineage_independent() {
        let integer_seeded = HdcLtcUnifiedNetwork::new(config(), 42);
        let genesis = GenesisSeed::from_phrase("neuroarch-v1-initialization-lineage");
        let genesis_seeded = HdcLtcUnifiedNetwork::from_genesis(config(), &genesis);

        assert_eq!(
            ActiveCoreHdcLtcTopology::new(&integer_seeded)
                .topology_commitment()
                .unwrap(),
            ActiveCoreHdcLtcTopology::new(&genesis_seeded)
                .topology_commitment()
                .unwrap()
        );
    }

    #[test]
    fn active_core_state_dimension_is_per_unit() {
        let network = HdcLtcUnifiedNetwork::new(config(), 1);
        let topology = ActiveCoreHdcLtcTopology::new(&network)
            .topology_descriptor()
            .unwrap();

        let first_layer = &topology.circuits[1];
        assert_eq!(first_layer.state_dimension, 128);
        assert_eq!(first_layer.unit_count, 2);
        assert_eq!(first_layer.total_state_dimensions().unwrap(), 256);
        assert_eq!(
            first_layer.implementation,
            CircuitImplementation::Named(ACTIVE_HDC_LTC_LAYER.to_string())
        );
    }

    #[test]
    fn active_core_structural_changes_change_commitment() {
        let base_commit = commitment(config(), 1);

        let mut resized = config();
        resized.layer_sizes[1] += 1;
        assert_ne!(base_commit, commitment(resized, 1));

        let mut tau = config();
        tau.neuron_config.tau_base = 0.2;
        assert_ne!(base_commit, commitment(tau, 1));

        let mut unbound = config();
        unbound.use_layer_binding = false;
        assert_ne!(base_commit, commitment(unbound, 1));

        let mut skip = config();
        skip.skip_connections = true;
        assert_ne!(base_commit, commitment(skip, 1));
    }

    #[test]
    fn active_core_non_topological_model_changes_do_not_change_commitment() {
        let base_commit = commitment(config(), 1);

        let mut activation = config();
        activation.neuron_config.activation = UnifiedActivation::Sigmoid;
        assert_eq!(base_commit, commitment(activation, 1));

        let mut backbone = config();
        backbone.neuron_config.backbone_tau = 0.9;
        assert_eq!(base_commit, commitment(backbone, 1));

        let mut gating = config();
        gating.neuron_config.gating_steepness = 2.0;
        gating.neuron_config.interp_bias = 0.25;
        assert_eq!(base_commit, commitment(gating, 1));

        let mut learning = config();
        learning.neuron_config.learning_rate = 0.123;
        learning.neuron_config.momentum = 0.5;
        learning.neuron_config.weight_decay = 0.002;
        assert_eq!(base_commit, commitment(learning, 1));

        let mut fourier = config();
        fourier.neuron_config.fourier_frequencies = vec![1.0, 2.0, 5.0];
        fourier.neuron_config.fourier_amplitude = 0.42;
        assert_eq!(base_commit, commitment(fourier, 1));
    }

    #[test]
    fn active_core_inspection_is_snapshot_pure() {
        let mut network = HdcLtcUnifiedNetwork::new(config(), 42);
        let input = ContinuousHV::random(128, 7);
        network.evolve_closed_form(0.05, &input);

        let mut before = NetworkStateSnapshot::default();
        network.snapshot_state_into(&mut before);
        let _ = ActiveCoreHdcLtcTopology::new(&network)
            .topology_descriptor()
            .unwrap();
        let mut after = NetworkStateSnapshot::default();
        network.snapshot_state_into(&mut after);

        assert!(before.approx_eq(&after, 0.0));
    }

    #[test]
    fn active_core_closed_form_replay_parity_with_inspection() {
        let mut control = HdcLtcUnifiedNetwork::new(config(), 42);
        let mut observed = control.clone();

        for step in 0..8u64 {
            let input = ContinuousHV::random(128, 100 + step);
            let _ = ActiveCoreHdcLtcTopology::new(&observed)
                .topology_descriptor()
                .unwrap();
            control.evolve_closed_form(0.025, &input);
            observed.evolve_closed_form(0.025, &input);
            assert_eq!(control.output().values, observed.output().values);
        }
    }

    #[test]
    fn active_core_skip_merge_is_explicit_once() {
        let mut cfg = config();
        cfg.skip_connections = true;
        let network = HdcLtcUnifiedNetwork::new(cfg, 42);
        let topology = ActiveCoreHdcLtcTopology::new(&network)
            .topology_descriptor()
            .unwrap();

        assert_eq!(
            topology.circuits[2].input_merge_policy,
            InputMergePolicy::BundleAll
        );
        let inter_layer = topology
            .edges
            .iter()
            .find(|edge| edge.source == CircuitId(1) && edge.target == CircuitId(2))
            .unwrap();
        assert_eq!(inter_layer.transform, EdgeTransform::Bind);
        let skip = topology
            .edges
            .iter()
            .find(|edge| edge.source == CircuitId(0) && edge.target == CircuitId(2))
            .unwrap();
        assert_eq!(skip.channel, SemanticChannel::SkipInput);
        assert_eq!(skip.transform, EdgeTransform::Direct);
    }
}
