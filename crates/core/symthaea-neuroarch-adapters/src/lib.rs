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
    BudgetClass, CircuitDescriptor, CircuitId, CircuitImplementation, EdgeDescriptor, EdgeDirection,
    EdgeId, EdgeTransform, InputMergePolicy, NeuroTopology, RecurrenceKind, SemanticChannel,
    TimescaleClass, TOPOLOGY_SCHEMA_VERSION, TopologyDescriptor, TopologyError,
};

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
            let state_dimension = unit_count
                .checked_mul(dim)
                .ok_or(TopologyError::DimensionOverflow)?;
            let id = u32::try_from(idx + 1).map_err(|_| TopologyError::DimensionOverflow)?;

            circuits.push(CircuitDescriptor {
                id: CircuitId(id),
                role: format!("layer:{idx}"),
                timescale_class: timescale.clone(),
                state_dimension,
                unit_count,
                implementation: CircuitImplementation::IncumbentHdcLtcLayer,
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
            let source = CircuitId(
                u32::try_from(layer_idx).map_err(|_| TopologyError::DimensionOverflow)?,
            );
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
    use symthaea_core::hdc::hdc_ltc_unified::{
        NetworkStateSnapshot, UnifiedConfig, UnifiedNetworkConfig,
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

    #[test]
    fn active_core_topology_is_seed_independent() {
        let a = HdcLtcUnifiedNetwork::new(config(), 1);
        let b = HdcLtcUnifiedNetwork::new(config(), 999);
        assert_eq!(
            ActiveCoreHdcLtcTopology::new(&a)
                .topology_commitment()
                .unwrap(),
            ActiveCoreHdcLtcTopology::new(&b)
                .topology_commitment()
                .unwrap()
        );
    }

    #[test]
    fn active_core_structural_changes_change_commitment() {
        let base = HdcLtcUnifiedNetwork::new(config(), 1);
        let base_commit = ActiveCoreHdcLtcTopology::new(&base)
            .topology_commitment()
            .unwrap();

        let mut resized = config();
        resized.layer_sizes[1] += 1;
        let resized = HdcLtcUnifiedNetwork::new(resized, 1);
        assert_ne!(
            base_commit,
            ActiveCoreHdcLtcTopology::new(&resized)
                .topology_commitment()
                .unwrap()
        );

        let mut skip = config();
        skip.skip_connections = true;
        let skip = HdcLtcUnifiedNetwork::new(skip, 1);
        assert_ne!(
            base_commit,
            ActiveCoreHdcLtcTopology::new(&skip)
                .topology_commitment()
                .unwrap()
        );
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
