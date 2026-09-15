// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Compatibility/reference topology adapter for the standalone HDC/LTC extraction.
//!
//! The canonical topology contract lives in `symthaea-neuroarch-types`. This
//! module adapts the standalone `symthaea-hdc-ltc` network only; the active
//! Symthaea incumbent in `symthaea-core` has a separate adapter.

use crate::network::HdcLtcUnifiedNetwork;
pub use symthaea_neuroarch_types::*;

/// Pure descriptive adapter for the standalone HDC/LTC extraction.
///
/// Seed, realized weights/binding vectors, learned parameters, and runtime
/// state are excluded from topology identity and belong to benchmark subject
/// or runtime receipts instead.
impl NeuroTopology for HdcLtcUnifiedNetwork {
    fn topology_descriptor(&self) -> Result<TopologyDescriptor, TopologyError> {
        let config = self.config();
        if config.layer_sizes.is_empty() {
            return Err(TopologyError::NoComputationalCircuits);
        }

        let dim = u64::try_from(config.neuron_config.dim)
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
    use crate::config::{NetworkConfig, NeuronConfig};
    use crate::continuous_hv::ContinuousHV;

    fn config() -> NetworkConfig {
        NetworkConfig {
            layer_sizes: vec![2, 3, 2],
            neuron_config: NeuronConfig {
                dim: 128,
                tau_base: 0.1,
                ..NeuronConfig::default()
            },
            use_layer_binding: true,
            skip_connections: false,
        }
    }

    #[test]
    fn standalone_topology_is_seed_independent() {
        let a = HdcLtcUnifiedNetwork::new(config(), 1);
        let b = HdcLtcUnifiedNetwork::new(config(), 999);
        assert_eq!(
            a.topology_commitment().unwrap(),
            b.topology_commitment().unwrap()
        );
    }

    #[test]
    fn standalone_structural_changes_change_commitment() {
        let base = HdcLtcUnifiedNetwork::new(config(), 1);
        let base_commit = base.topology_commitment().unwrap();

        let mut resized = config();
        resized.layer_sizes[1] += 1;
        assert_ne!(
            base_commit,
            HdcLtcUnifiedNetwork::new(resized, 1)
                .topology_commitment()
                .unwrap()
        );

        let mut unbound = config();
        unbound.use_layer_binding = false;
        assert_ne!(
            base_commit,
            HdcLtcUnifiedNetwork::new(unbound, 1)
                .topology_commitment()
                .unwrap()
        );

        let mut skip = config();
        skip.skip_connections = true;
        assert_ne!(
            base_commit,
            HdcLtcUnifiedNetwork::new(skip, 1)
                .topology_commitment()
                .unwrap()
        );
    }

    #[test]
    fn standalone_skip_merge_is_explicit_once() {
        let mut cfg = config();
        cfg.skip_connections = true;
        let topology = HdcLtcUnifiedNetwork::new(cfg, 42)
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

    #[test]
    fn standalone_inspection_is_runtime_state_independent() {
        let mut network = HdcLtcUnifiedNetwork::new(config(), 42);
        let before = network.topology_commitment().unwrap();
        let input = ContinuousHV::new_random(128, 7);
        network.step(0.1, &input);
        network.step_with_timestamp(1.0, &input);
        assert_eq!(before, network.topology_commitment().unwrap());
    }

    #[test]
    fn standalone_fixed_step_replay_parity_with_inspection() {
        let mut control = HdcLtcUnifiedNetwork::new(config(), 42);
        let mut observed = control.clone();
        for step in 0..8u64 {
            let input = ContinuousHV::new_random(128, 100 + step);
            let _ = observed.topology_descriptor().unwrap();
            control.step(0.025, &input);
            observed.step(0.025, &input);
            assert_eq!(control.output().values, observed.output().values);
            assert_eq!(control.step_count(), observed.step_count());
        }
    }

    #[test]
    fn standalone_irregular_time_replay_parity_with_inspection() {
        let mut control = HdcLtcUnifiedNetwork::new(config(), 42);
        let mut observed = control.clone();
        for (idx, timestamp) in [0.0, 0.011, 0.039, 0.1, 0.101, 0.8]
            .into_iter()
            .enumerate()
        {
            let input = ContinuousHV::new_random(128, 200 + idx as u64);
            let _ = observed.topology_descriptor().unwrap();
            control.step_with_timestamp(timestamp, &input);
            observed.step_with_timestamp(timestamp, &input);
            assert_eq!(control.output().values, observed.output().values);
            assert_eq!(control.last_timestamp(), observed.last_timestamp());
            assert_eq!(control.step_count(), observed.step_count());
        }
    }
}
