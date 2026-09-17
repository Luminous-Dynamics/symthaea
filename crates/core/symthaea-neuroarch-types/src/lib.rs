// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Neutral, engine-independent neuroarchitecture topology contracts.
//!
//! This crate owns static topology identity only. It deliberately contains no
//! HDC/LTC engine, runtime state, learned parameters, causal-effect claims, or
//! cognition policy. Engine-specific adapters implement [`NeuroTopology`].

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
    /// Convert an exact model time constant into the V1 structural declaration.
    ///
    /// V1 structural topology records time at nanosecond resolution. Exact
    /// floating-point model configuration belongs in subject/config identity.
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

/// Static implementation identity for a circuit.
///
/// `ExternalInput` is the only schema-level special case. Computational
/// implementations use a bounded canonical token so this neutral crate never
/// needs a variant for each concrete engine lineage.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CircuitImplementation {
    ExternalInput,
    Named(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InputMergePolicy {
    None,
    Single,
    BundleAll,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CircuitDescriptor {
    pub id: CircuitId,
    /// Descriptive metadata only; this is not evidence of actual function.
    pub role: String,
    pub timescale_class: TimescaleClass,
    /// Logical state dimensions per implementation unit.
    pub state_dimension: u64,
    /// Number of implementation units represented by this circuit.
    pub unit_count: u64,
    pub implementation: CircuitImplementation,
    pub input_merge_policy: InputMergePolicy,
    pub modulation_profile: Option<String>,
}

impl CircuitDescriptor {
    /// Checked logical-state total for resource accounting.
    ///
    /// Receipts should preserve both raw factors and derive the total exactly
    /// once through this operation.
    pub fn total_state_dimensions(&self) -> Result<u64, TopologyError> {
        self.state_dimension
            .checked_mul(self.unit_count)
            .ok_or(TopologyError::DimensionOverflow)
    }
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
    /// No bounded route budget is declared by the adapted legacy interface.
    /// This is not a claim of infinite physical capacity.
    LegacyUnbounded,
    Local,
    Global,
    Named(String),
}

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

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
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
    DanglingSource {
        edge: EdgeId,
        source: CircuitId,
    },
    DanglingTarget {
        edge: EdgeId,
        target: CircuitId,
    },
    DuplicateStructuralEdge {
        source: CircuitId,
        target: CircuitId,
    },
    DuplicateChannelEdge {
        source: CircuitId,
        target: CircuitId,
    },
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
            circuit.total_state_dimensions()?;
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
            let first_for_pair = endpoint_pairs.insert(pair);
            if !self.allow_parallel_channels && !first_for_pair {
                return Err(TopologyError::DuplicateStructuralEdge { source, target });
            }

            let channel_key = (direction, source, target, edge.channel.clone());
            if !channel_edges.insert(channel_key) {
                return Err(TopologyError::DuplicateChannelEdge { source, target });
            }
        }
        Ok(())
    }

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
        Ok(TopologyCommitment(
            *blake3::hash(&self.canonical_bytes()?).as_bytes(),
        ))
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
            byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b':' | b'/')
        })
    {
        return Err(TopologyError::InvalidSymbolicToken);
    }
    Ok(())
}

fn validate_named(value: &str) -> Result<(), TopologyError> {
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
        CircuitImplementation::Named(name) => {
            out.push(1);
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

    fn circuit(id: u32, role: &str) -> CircuitDescriptor {
        CircuitDescriptor {
            id: CircuitId(id),
            role: role.to_string(),
            timescale_class: TimescaleClass::Medium,
            state_dimension: 64,
            unit_count: 2,
            implementation: CircuitImplementation::Named("test".to_string()),
            input_merge_policy: InputMergePolicy::Single,
            modulation_profile: None,
        }
    }

    fn edge(id: u64, source: u32, target: u32, channel: &str) -> EdgeDescriptor {
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

    fn topology() -> TopologyDescriptor {
        TopologyDescriptor {
            schema_version: TOPOLOGY_SCHEMA_VERSION,
            circuits: vec![circuit(1, "a"), circuit(2, "b")],
            edges: vec![edge(10, 1, 2, "x")],
            allow_parallel_channels: false,
        }
    }

    #[test]
    fn insertion_order_is_canonical() {
        let mut a = topology();
        a.circuits.push(circuit(3, "c"));
        a.edges.push(edge(11, 2, 3, "y"));
        let mut b = a.clone();
        b.circuits.reverse();
        b.edges.reverse();
        assert_eq!(a.canonical_bytes().unwrap(), b.canonical_bytes().unwrap());
        assert_eq!(a.commitment().unwrap(), b.commitment().unwrap());
    }

    #[test]
    fn bidirectional_orientation_is_canonical() {
        let mut a = topology();
        a.edges[0].direction = EdgeDirection::Bidirectional;
        let mut b = a.clone();
        std::mem::swap(&mut b.edges[0].source, &mut b.edges[0].target);
        assert_eq!(a.canonical_bytes().unwrap(), b.canonical_bytes().unwrap());
    }

    #[test]
    fn symbolic_identity_is_fail_closed() {
        let mut bad = topology();
        bad.circuits[0].role = "visual cortex".to_string();
        assert_eq!(bad.validate(), Err(TopologyError::InvalidSymbolicToken));

        bad = topology();
        bad.circuits[0].role = "visuál".to_string();
        assert_eq!(bad.validate(), Err(TopologyError::InvalidSymbolicToken));
    }

    #[test]
    fn state_total_is_checked() {
        let circuit = circuit(1, "a");
        assert_eq!(circuit.total_state_dimensions().unwrap(), 128);

        let mut overflow = circuit;
        overflow.state_dimension = u64::MAX;
        overflow.unit_count = 2;
        assert_eq!(
            overflow.total_state_dimensions(),
            Err(TopologyError::DimensionOverflow)
        );
    }

    #[test]
    fn structural_mutation_changes_commitment() {
        let a = topology();
        let mut b = a.clone();
        b.circuits[0].state_dimension += 1;
        assert_ne!(a.commitment().unwrap(), b.commitment().unwrap());
    }
}
