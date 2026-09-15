// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Experimental, descriptive neuroarchitecture topology surface.
//!
//! This module describes static HDC/LTC structure without changing runtime
//! evolution. Structural topology is intentionally distinct from learned
//! parameters, runtime state, measured causal effect, and epistemic confidence.

use crate::network::HdcLtcUnifiedNetwork;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub const TOPOLOGY_SCHEMA_VERSION: u16 = 1;
pub const MAX_SYMBOLIC_TOKEN_BYTES: usize = 128;

const TOPOLOGY_DOMAIN: &[u8] = b"symthaea-neuroarch-topology-v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CircuitId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EdgeId(pub u64);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TimescaleClass {
    Stateless,
    Fast,
    Medium,
    Slow,
    CustomNanos(u64),
}

impl TimescaleClass {
    pub fn from_seconds(seconds: f32) -> Result<Self, TopologyError> {
        if !seconds.is_finite() || seconds <= 0.0 {
            return Err(TopologyError::InvalidTimescale);
        }
        let nanos = (f64::from(seconds) * 1_000_000_000.0).round();
        if nanos < 1.0 || nanos > u64::MAX as f64 {
            return Err(TopologyError::InvalidTimescale);
        }
        Ok(Self::CustomNanos(nanos as u64))
    }

    fn validate(&self) -> Result<(), TopologyError> {
        if matches!(self, Self::CustomNanos(0)) {
            Err(TopologyError::InvalidTimescale)
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CircuitImplementation {
    ExternalInput,
    IncumbentHdcLtcLayer,
    Named(String),
}

/// Target-level semantics for combining declared inbound routes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InputMergePolicy {
    None,
    Single,
    BundleAll,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CircuitDescriptor {
    pub id: CircuitId,
    /// Descriptive only; this label is not evidence that the circuit performs the named role.
    pub role: String,
    pub timescale_class: TimescaleClass,
    /// Total scalar state dimension represented by this circuit.
    pub state_dimension: u64,
    /// Number of implementation units represented by this circuit.
    pub unit_count: u64,
    pub implementation: CircuitImplementation,
    pub input_merge_policy: InputMergePolicy,
    pub modulation_profile: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SemanticChannel {
    ExternalInput,
    InterLayer,
    SkipInput,
    Named(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EdgeDirection {
    Directed,
    Bidirectional,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RecurrenceKind {
    FeedForward,
    Recurrent,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BudgetClass {
    /// The incumbent has no explicit per-route communication budget.
    LegacyUnbounded,
    Local,
    Global,
    Named(String),
}

/// Route-local execution transform. Multi-route merging belongs to the target circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EdgeTransform {
    Direct,
    Bind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeDescriptor {
    pub id: EdgeId,
    pub source: CircuitId,
    pub target: CircuitId,
    pub channel: SemanticChannel,
    pub direction: EdgeDirection,
    pub recurrence: RecurrenceKind,
    pub budget_class: BudgetClass,
    pub transform: EdgeTransform,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyDescriptor {
    pub schema_version: u16,
    pub circuits: Vec<CircuitDescriptor>,
    pub edges: Vec<EdgeDescriptor>,
    pub allow_parallel_channels: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyCommitment(pub [u8; 32]);

impl TopologyCommitment {
    pub const ALGORITHM: &'static str = "blake3";
}

impl fmt::Display for TopologyCommitment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopologyError {
    UnsupportedSchemaVersion(u16),
    NoComputationalCircuits,
    DuplicateCircuitId(CircuitId),
    DuplicateEdgeId(EdgeId),
    DanglingSource { edge: EdgeId, source: CircuitId },
    DanglingTarget { edge: EdgeId, target: CircuitId },
    DuplicateStructuralEdge { source: CircuitId, target: CircuitId },
    DuplicateChannelEdge { source: CircuitId, target: CircuitId },
    ForbiddenSelfEdge(EdgeId),
    InvalidStateDimension(CircuitId),
    InvalidUnitCount(CircuitId),
    InvalidTimescale,
    EmptyRole(CircuitId),
    EmptyNamedField,
    InvalidSymbolicToken,
    FieldTooLong,
    DimensionOverflow,
}

impl fmt::Display for TopologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid neuroarchitecture topology: {self:?}")
    }
}

impl Error for TopologyError {}

pub trait NeuroTopology {
    fn topology_descriptor(&self) -> Result<TopologyDescriptor, TopologyError>;

    fn topology_commitment(&self) -> Result<TopologyCommitment, TopologyError> {
        self.topology_descriptor()?.commitment()
    }
}

impl TopologyDescriptor {
    pub fn validate(&self) -> Result<(), TopologyError> {
        if self.schema_version != TOPOLOGY_SCHEMA_VERSION {
            return Err(TopologyError::UnsupportedSchemaVersion(self.schema_version));
        }

        let mut circuit_ids = BTreeSet::new();
        let mut computational_circuits = 0usize;
        for circuit in &self.circuits {
            if !circuit_ids.insert(circuit.id) {
                return Err(TopologyError::DuplicateCircuitId(circuit.id));
            }
            if circuit.role.is_empty() {
                return Err(TopologyError::EmptyRole(circuit.id));
            }
            validate_symbolic_token(&circuit.role)?;
            if circuit.state_dimension == 0 {
                return Err(TopologyError::InvalidStateDimension(circuit.id));
            }
            if circuit.unit_count == 0 {
                return Err(TopologyError::InvalidUnitCount(circuit.id));
            }
            circuit.timescale_class.validate()?;
            validate_impl(&circuit.implementation)?;
            if let Some(profile) = &circuit.modulation_profile {
                validate_named(profile)?;
            }
            if circuit.implementation != CircuitImplementation::ExternalInput {
                computational_circuits += 1;
            }
        }
        if computational_circuits == 0 {
            return Err(TopologyError::NoComputationalCircuits);
        }

        let mut edge_ids = BTreeSet::new();
        let mut endpoint_pairs = BTreeSet::new();
        let mut channel_edges = BTreeSet::new();
        for edge in &self.edges {
            if !edge_ids.insert(edge.id) {
                return Err(TopologyError::DuplicateEdgeId(edge.id));
            }
            if !circuit_ids.contains(&edge.source) {
                return Err(TopologyError::DanglingSource {
                    edge: edge.id,
                    source: edge.source,
                });
            }
            if !circuit_ids.contains(&edge.target) {
                return Err(TopologyError::DanglingTarget {
                    edge: edge.id,
                    target: edge.target,
                });
            }
            if edge.source == edge.target && edge.recurrence != RecurrenceKind::Recurrent {
                return Err(TopologyError::ForbiddenSelfEdge(edge.id));
            }
            validate_channel(&edge.channel)?;
            validate_budget(&edge.budget_class)?;

            let (source, target) = canonical_endpoints(edge);
            let direction = direction_tag(edge.direction);
            let pair = (direction, source, target);
            if !self.allow_parallel_channels && !endpoint_pairs.insert(pair) {
                return Err(TopologyError::DuplicateStructuralEdge { source, target });
            }
            endpoint_pairs.insert(pair);

            let channel_key = (direction, source, target, edge.channel.clone());
            if !channel_edges.insert(channel_key) {
                return Err(TopologyError::DuplicateChannelEdge { source, target });
            }
        }
        Ok(())
    }

    /// Return canonical bytes independent of circuit/edge insertion order and runtime state.
    ///
    /// V1 symbolic labels are case-sensitive bounded ASCII tokens. Bidirectional routes are
    /// endpoint-normalized so `A <-> B` and `B <-> A` canonicalize identically when every
    /// other static identity field, including the edge ID, is equal.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, TopologyError> {
        self.validate()?;
        let mut out = Vec::new();
        out.extend_from_slice(TOPOLOGY_DOMAIN);
        out.extend_from_slice(&self.schema_version.to_le_bytes());
        out.push(u8::from(self.allow_parallel_channels));

        let mut circuits: Vec<&CircuitDescriptor> = self.circuits.iter().collect();
        circuits.sort_by_key(|circuit| circuit.id);
        push_len(&mut out, circuits.len())?;
        for circuit in circuits {
            encode_circuit(&mut out, circuit)?;
        }

        let mut edges: Vec<&EdgeDescriptor> = self.edges.iter().collect();
        edges.sort_by_key(|edge| canonical_edge_sort_key(edge));
        push_len(&mut out, edges.len())?;
        for edge in edges {
            encode_edge(&mut out, edge)?;
        }
        Ok(out)
    }

    pub fn commitment(&self) -> Result<TopologyCommitment, TopologyError> {
        Ok(TopologyCommitment(*blake3::hash(&self.canonical_bytes()?).as_bytes()))
    }
}

/// Pure descriptive adapter for the current layered HDC/LTC incumbent.
///
/// Seed, realized weights/binding vectors, learned parameters, and runtime state are excluded
/// from topology identity. Benchmark subject/config receipts must bind those separately.
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
            let units = u64::try_from(layer_size).map_err(|_| TopologyError::DimensionOverflow)?;
            let state_dimension = units
                .checked_mul(dim)
                .ok_or(TopologyError::DimensionOverflow)?;
            let id = u32::try_from(idx + 1).map_err(|_| TopologyError::DimensionOverflow)?;
            circuits.push(CircuitDescriptor {
                id: CircuitId(id),
                role: format!("layer:{idx}"),
                timescale_class: timescale.clone(),
                state_dimension,
                unit_count: units,
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

fn validate_symbolic_token(value: &str) -> Result<(), TopologyError> {
    if value.is_empty() {
        return Err(TopologyError::EmptyNamedField);
    }
    if value.len() > MAX_SYMBOLIC_TOKEN_BYTES {
        return Err(TopologyError::FieldTooLong);
    }
    if !value.is_ascii()
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(byte, b'_' | b'-' | b'.' | b':' | b'/')
        })
    {
        return Err(TopologyError::InvalidSymbolicToken);
    }
    Ok(())
}

fn validate_named(value: &str) -> Result<(), TopologyError> {
    if value.is_empty() {
        return Err(TopologyError::EmptyNamedField);
    }
    validate_symbolic_token(value)
}

fn validate_impl(value: &CircuitImplementation) -> Result<(), TopologyError> {
    if let CircuitImplementation::Named(name) = value {
        validate_named(name)?;
    }
    Ok(())
}

fn validate_channel(value: &SemanticChannel) -> Result<(), TopologyError> {
    if let SemanticChannel::Named(name) = value {
        validate_named(name)?;
    }
    Ok(())
}

fn validate_budget(value: &BudgetClass) -> Result<(), TopologyError> {
    if let BudgetClass::Named(name) = value {
        validate_named(name)?;
    }
    Ok(())
}

fn direction_tag(direction: EdgeDirection) -> u8 {
    match direction {
        EdgeDirection::Directed => 0,
        EdgeDirection::Bidirectional => 1,
    }
}

fn canonical_endpoints(edge: &EdgeDescriptor) -> (CircuitId, CircuitId) {
    if edge.direction == EdgeDirection::Bidirectional && edge.target < edge.source {
        (edge.target, edge.source)
    } else {
        (edge.source, edge.target)
    }
}

fn canonical_edge_sort_key(
    edge: &EdgeDescriptor,
) -> (u8, CircuitId, CircuitId, SemanticChannel, EdgeId) {
    let (source, target) = canonical_endpoints(edge);
    (
        direction_tag(edge.direction),
        source,
        target,
        edge.channel.clone(),
        edge.id,
    )
}

fn push_len(out: &mut Vec<u8>, value: usize) -> Result<(), TopologyError> {
    let value = u32::try_from(value).map_err(|_| TopologyError::FieldTooLong)?;
    out.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn push_str(out: &mut Vec<u8>, value: &str) -> Result<(), TopologyError> {
    push_len(out, value.len())?;
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn push_opt_str(out: &mut Vec<u8>, value: &Option<String>) -> Result<(), TopologyError> {
    match value {
        Some(value) => {
            out.push(1);
            push_str(out, value)
        }
        None => {
            out.push(0);
            Ok(())
        }
    }
}

fn encode_timescale(out: &mut Vec<u8>, value: &TimescaleClass) {
    match value {
        TimescaleClass::Stateless => out.push(0),
        TimescaleClass::Fast => out.push(1),
        TimescaleClass::Medium => out.push(2),
        TimescaleClass::Slow => out.push(3),
        TimescaleClass::CustomNanos(nanos) => {
            out.push(4);
            out.extend_from_slice(&nanos.to_le_bytes());
        }
    }
}

fn encode_impl(out: &mut Vec<u8>, value: &CircuitImplementation) -> Result<(), TopologyError> {
    match value {
        CircuitImplementation::ExternalInput => out.push(0),
        CircuitImplementation::IncumbentHdcLtcLayer => out.push(1),
        CircuitImplementation::Named(name) => {
            out.push(2);
            push_str(out, name)?;
        }
    }
    Ok(())
}

fn encode_merge_policy(out: &mut Vec<u8>, value: InputMergePolicy) {
    out.push(match value {
        InputMergePolicy::None => 0,
        InputMergePolicy::Single => 1,
        InputMergePolicy::BundleAll => 2,
    });
}

fn encode_channel(out: &mut Vec<u8>, value: &SemanticChannel) -> Result<(), TopologyError> {
    match value {
        SemanticChannel::ExternalInput => out.push(0),
        SemanticChannel::InterLayer => out.push(1),
        SemanticChannel::SkipInput => out.push(2),
        SemanticChannel::Named(name) => {
            out.push(3);
            push_str(out, name)?;
        }
    }
    Ok(())
}

fn encode_budget(out: &mut Vec<u8>, value: &BudgetClass) -> Result<(), TopologyError> {
    match value {
        BudgetClass::LegacyUnbounded => out.push(0),
        BudgetClass::Local => out.push(1),
        BudgetClass::Global => out.push(2),
        BudgetClass::Named(name) => {
            out.push(3);
            push_str(out, name)?;
        }
    }
    Ok(())
}

fn encode_circuit(out: &mut Vec<u8>, circuit: &CircuitDescriptor) -> Result<(), TopologyError> {
    out.extend_from_slice(&circuit.id.0.to_le_bytes());
    push_str(out, &circuit.role)?;
    encode_timescale(out, &circuit.timescale_class);
    out.extend_from_slice(&circuit.state_dimension.to_le_bytes());
    out.extend_from_slice(&circuit.unit_count.to_le_bytes());
    encode_impl(out, &circuit.implementation)?;
    encode_merge_policy(out, circuit.input_merge_policy);
    push_opt_str(out, &circuit.modulation_profile)?;
    Ok(())
}

fn encode_edge(out: &mut Vec<u8>, edge: &EdgeDescriptor) -> Result<(), TopologyError> {
    let (source, target) = canonical_endpoints(edge);
    out.extend_from_slice(&edge.id.0.to_le_bytes());
    out.extend_from_slice(&source.0.to_le_bytes());
    out.extend_from_slice(&target.0.to_le_bytes());
    encode_channel(out, &edge.channel)?;
    out.push(direction_tag(edge.direction));
    out.push(match edge.recurrence {
        RecurrenceKind::FeedForward => 0,
        RecurrenceKind::Recurrent => 1,
    });
    encode_budget(out, &edge.budget_class)?;
    out.push(match edge.transform {
        EdgeTransform::Direct => 0,
        EdgeTransform::Bind => 1,
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{NetworkConfig, NeuronConfig};
    use crate::continuous_hv::ContinuousHV;

    fn test_circuit(id: u32, role: &str) -> CircuitDescriptor {
        CircuitDescriptor {
            id: CircuitId(id),
            role: role.to_string(),
            timescale_class: TimescaleClass::Medium,
            state_dimension: 64,
            unit_count: 1,
            implementation: CircuitImplementation::Named("test".to_string()),
            input_merge_policy: InputMergePolicy::Single,
            modulation_profile: None,
        }
    }

    fn test_edge(id: u64, source: u32, target: u32, channel: &str) -> EdgeDescriptor {
        EdgeDescriptor {
            id: EdgeId(id),
            source: CircuitId(source),
            target: CircuitId(target),
            channel: SemanticChannel::Named(channel.to_string()),
            direction: EdgeDirection::Directed,
            recurrence: RecurrenceKind::FeedForward,
            budget_class: BudgetClass::Local,
            transform: EdgeTransform::Direct,
        }
    }

    fn reverse_endpoints(edge: &mut EdgeDescriptor) {
        let source = edge.source;
        edge.source = edge.target;
        edge.target = source;
    }

    fn generic_topology() -> TopologyDescriptor {
        TopologyDescriptor {
            schema_version: TOPOLOGY_SCHEMA_VERSION,
            circuits: vec![test_circuit(1, "a"), test_circuit(2, "b")],
            edges: vec![test_edge(10, 1, 2, "x")],
            allow_parallel_channels: false,
        }
    }

    fn small_network_config() -> NetworkConfig {
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
    fn canonical_commitment_is_insertion_order_independent() {
        let a = TopologyDescriptor {
            schema_version: TOPOLOGY_SCHEMA_VERSION,
            circuits: vec![test_circuit(1, "a"), test_circuit(2, "b"), test_circuit(3, "c")],
            edges: vec![test_edge(10, 1, 2, "x"), test_edge(11, 2, 3, "y")],
            allow_parallel_channels: false,
        };
        let b = TopologyDescriptor {
            schema_version: TOPOLOGY_SCHEMA_VERSION,
            circuits: vec![test_circuit(3, "c"), test_circuit(1, "a"), test_circuit(2, "b")],
            edges: vec![test_edge(11, 2, 3, "y"), test_edge(10, 1, 2, "x")],
            allow_parallel_channels: false,
        };
        assert_eq!(a.commitment().unwrap(), b.commitment().unwrap());
        assert_eq!(a.canonical_bytes().unwrap(), b.canonical_bytes().unwrap());
    }

    #[test]
    fn bidirectional_orientation_is_canonical() {
        let mut a = generic_topology();
        a.edges[0].direction = EdgeDirection::Bidirectional;
        let mut b = a.clone();
        reverse_endpoints(&mut b.edges[0]);
        assert_eq!(a.canonical_bytes().unwrap(), b.canonical_bytes().unwrap());
        assert_eq!(a.commitment().unwrap(), b.commitment().unwrap());

        let mut duplicate = a.clone();
        let mut reversed = a.edges[0].clone();
        reversed.id = EdgeId(11);
        reverse_endpoints(&mut reversed);
        duplicate.edges.push(reversed);
        assert!(matches!(
            duplicate.validate(),
            Err(TopologyError::DuplicateStructuralEdge { .. })
        ));
    }

    #[test]
    fn symbolic_tokens_are_unambiguous_and_bounded() {
        let mut whitespace = generic_topology();
        whitespace.circuits[0].role = "visual cortex".to_string();
        assert_eq!(
            whitespace.validate(),
            Err(TopologyError::InvalidSymbolicToken)
        );

        let mut unicode = generic_topology();
        unicode.circuits[0].role = "visuál".to_string();
        assert_eq!(unicode.validate(), Err(TopologyError::InvalidSymbolicToken));

        let mut too_long = generic_topology();
        too_long.circuits[0].role = "a".repeat(MAX_SYMBOLIC_TOKEN_BYTES + 1);
        assert_eq!(too_long.validate(), Err(TopologyError::FieldTooLong));
    }

    #[test]
    fn static_mutation_changes_commitment() {
        let a = generic_topology();
        let mut b = a.clone();
        b.circuits[0].state_dimension += 1;
        assert_ne!(a.commitment().unwrap(), b.commitment().unwrap());

        let mut c = a.clone();
        c.circuits[1].input_merge_policy = InputMergePolicy::BundleAll;
        assert_ne!(a.commitment().unwrap(), c.commitment().unwrap());
    }

    #[test]
    fn rejects_dangling_duplicate_and_forbidden_self_edges() {
        let mut dangling = generic_topology();
        dangling.edges[0].target = CircuitId(99);
        assert!(matches!(
            dangling.validate(),
            Err(TopologyError::DanglingTarget { .. })
        ));

        let mut duplicate = generic_topology();
        duplicate.circuits.push(test_circuit(1, "duplicate"));
        assert!(matches!(
            duplicate.validate(),
            Err(TopologyError::DuplicateCircuitId(_))
        ));

        let mut self_edge = generic_topology();
        self_edge.edges[0].target = self_edge.edges[0].source;
        assert!(matches!(
            self_edge.validate(),
            Err(TopologyError::ForbiddenSelfEdge(_))
        ));
    }

    #[test]
    fn parallel_channels_are_explicit_policy() {
        let mut topology = generic_topology();
        topology.edges.push(test_edge(11, 1, 2, "y"));
        assert!(matches!(
            topology.validate(),
            Err(TopologyError::DuplicateStructuralEdge { .. })
        ));
        topology.allow_parallel_channels = true;
        assert!(topology.validate().is_ok());

        topology.edges.push(test_edge(12, 1, 2, "y"));
        assert!(matches!(
            topology.validate(),
            Err(TopologyError::DuplicateChannelEdge { .. })
        ));
    }

    #[test]
    fn incumbent_adapter_binds_structure_not_seed() {
        let config = small_network_config();
        let a = HdcLtcUnifiedNetwork::new(config.clone(), 1);
        let b = HdcLtcUnifiedNetwork::new(config, 999);
        assert_eq!(
            a.topology_commitment().unwrap(),
            b.topology_commitment().unwrap()
        );
    }

    #[test]
    fn incumbent_skip_semantics_are_not_double_encoded() {
        let mut config = small_network_config();
        config.skip_connections = true;
        let topology = HdcLtcUnifiedNetwork::new(config, 42)
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
    fn incumbent_structural_changes_change_commitment() {
        let base = small_network_config();
        let base_commit = HdcLtcUnifiedNetwork::new(base.clone(), 1)
            .topology_commitment()
            .unwrap();

        let mut resized = base.clone();
        resized.layer_sizes[1] += 1;
        assert_ne!(
            base_commit,
            HdcLtcUnifiedNetwork::new(resized, 1)
                .topology_commitment()
                .unwrap()
        );

        let mut unbound = base.clone();
        unbound.use_layer_binding = false;
        assert_ne!(
            base_commit,
            HdcLtcUnifiedNetwork::new(unbound, 1)
                .topology_commitment()
                .unwrap()
        );

        let mut skip = base;
        skip.skip_connections = true;
        assert_ne!(
            base_commit,
            HdcLtcUnifiedNetwork::new(skip, 1)
                .topology_commitment()
                .unwrap()
        );
    }

    #[test]
    fn inspection_is_runtime_state_independent() {
        let mut network = HdcLtcUnifiedNetwork::new(small_network_config(), 42);
        let before = network.topology_commitment().unwrap();
        let input = ContinuousHV::new_random(128, 7);
        network.step(0.1, &input);
        network.step_with_timestamp(1.0, &input);
        assert_eq!(before, network.topology_commitment().unwrap());
    }

    #[test]
    fn incumbent_fixed_step_replay_parity_with_inspection() {
        let mut control = HdcLtcUnifiedNetwork::new(small_network_config(), 42);
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
    fn incumbent_irregular_time_replay_parity_with_inspection() {
        let mut control = HdcLtcUnifiedNetwork::new(small_network_config(), 42);
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
