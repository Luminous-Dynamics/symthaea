// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed systems-engineering semantic/lifecycle graph.
//!
//! This crate is descriptive only:
//!
//! ```text
//! valid semantic graph != verified design != qualified design
//! ```
//!
//! It contains no ETK evidence-admission, discharge, requirement-satisfaction,
//! qualification, deployment, manufacturing, or actuation authority.

#![deny(unsafe_code)]

use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use thiserror::Error;

const NODE_REVISION_DOMAIN: &str = "symthaea.se.node-revision.v1";
const GRAPH_SNAPSHOT_DOMAIN: &str = "symthaea.se.graph-snapshot.v1";

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ModelError {
    #[error("identifier is empty or contains a control character")]
    InvalidIdentifier,
    #[error("label/statement must not be empty")]
    EmptyText,
    #[error("node already exists: {0}")]
    DuplicateNode(String),
    #[error("node does not exist: {0}")]
    MissingNode(String),
    #[error("relation already exists")]
    DuplicateRelation,
    #[error("symmetric relation already exists in reverse order")]
    DuplicateSymmetricRelation,
    #[error("self relation is not allowed for {0:?}")]
    SelfRelation(RelationKind),
    #[error("invalid role pairing for {kind:?}: {source:?} -> {target:?}")]
    InvalidRelationRoles {
        kind: RelationKind,
        source: NodeKind,
        target: NodeKind,
    },
    #[error("relation {0:?} would introduce a forbidden cycle")]
    CyclicRelation(RelationKind),
    #[error("replacement identity mismatch: expected {expected}, got {actual}")]
    ReplacementIdentityMismatch { expected: String, actual: String },
}

fn validate_identifier(value: &str) -> Result<(), ModelError> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        return Err(ModelError::InvalidIdentifier);
    }
    Ok(())
}

fn validate_text(value: &str) -> Result<(), ModelError> {
    if value.trim().is_empty() {
        return Err(ModelError::EmptyText);
    }
    Ok(())
}

macro_rules! typed_id {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, ModelError> {
                let value = value.into();
                validate_identifier(&value)?;
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

typed_id!(StakeholderNeedId);
typed_id!(RequirementId);
typed_id!(FunctionId);
typed_id!(ComponentId);
typed_id!(InterfaceId);
typed_id!(ConstraintId);
typed_id!(AssumptionId);
typed_id!(HazardId);
typed_id!(ControlId);
typed_id!(VerificationActivityId);
typed_id!(ConfigurationId);
typed_id!(ChangeId);
typed_id!(DecisionId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NodeKind {
    StakeholderNeed,
    Requirement,
    Function,
    Component,
    Interface,
    Constraint,
    Assumption,
    Hazard,
    Control,
    VerificationActivity,
    Configuration,
    Change,
    Decision,
}

impl NodeKind {
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::StakeholderNeed => "stakeholder_need",
            Self::Requirement => "requirement",
            Self::Function => "function",
            Self::Component => "component",
            Self::Interface => "interface",
            Self::Constraint => "constraint",
            Self::Assumption => "assumption",
            Self::Hazard => "hazard",
            Self::Control => "control",
            Self::VerificationActivity => "verification_activity",
            Self::Configuration => "configuration",
            Self::Change => "change",
            Self::Decision => "decision",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NodeId {
    StakeholderNeed(StakeholderNeedId),
    Requirement(RequirementId),
    Function(FunctionId),
    Component(ComponentId),
    Interface(InterfaceId),
    Constraint(ConstraintId),
    Assumption(AssumptionId),
    Hazard(HazardId),
    Control(ControlId),
    VerificationActivity(VerificationActivityId),
    Configuration(ConfigurationId),
    Change(ChangeId),
    Decision(DecisionId),
}

impl NodeId {
    pub const fn kind(&self) -> NodeKind {
        match self {
            Self::StakeholderNeed(_) => NodeKind::StakeholderNeed,
            Self::Requirement(_) => NodeKind::Requirement,
            Self::Function(_) => NodeKind::Function,
            Self::Component(_) => NodeKind::Component,
            Self::Interface(_) => NodeKind::Interface,
            Self::Constraint(_) => NodeKind::Constraint,
            Self::Assumption(_) => NodeKind::Assumption,
            Self::Hazard(_) => NodeKind::Hazard,
            Self::Control(_) => NodeKind::Control,
            Self::VerificationActivity(_) => NodeKind::VerificationActivity,
            Self::Configuration(_) => NodeKind::Configuration,
            Self::Change(_) => NodeKind::Change,
            Self::Decision(_) => NodeKind::Decision,
        }
    }

    pub fn local_id(&self) -> &str {
        match self {
            Self::StakeholderNeed(id) => id.as_str(),
            Self::Requirement(id) => id.as_str(),
            Self::Function(id) => id.as_str(),
            Self::Component(id) => id.as_str(),
            Self::Interface(id) => id.as_str(),
            Self::Constraint(id) => id.as_str(),
            Self::Assumption(id) => id.as_str(),
            Self::Hazard(id) => id.as_str(),
            Self::Control(id) => id.as_str(),
            Self::VerificationActivity(id) => id.as_str(),
            Self::Configuration(id) => id.as_str(),
            Self::Change(id) => id.as_str(),
            Self::Decision(id) => id.as_str(),
        }
    }

    fn canonical_key(&self) -> String {
        format!("{}:{}", self.kind().canonical_name(), self.local_id())
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.canonical_key())
    }
}

macro_rules! node_id_from {
    ($id:ident, $variant:ident) => {
        impl From<$id> for NodeId {
            fn from(value: $id) -> Self {
                Self::$variant(value)
            }
        }
    };
}

node_id_from!(StakeholderNeedId, StakeholderNeed);
node_id_from!(RequirementId, Requirement);
node_id_from!(FunctionId, Function);
node_id_from!(ComponentId, Component);
node_id_from!(InterfaceId, Interface);
node_id_from!(ConstraintId, Constraint);
node_id_from!(AssumptionId, Assumption);
node_id_from!(HazardId, Hazard);
node_id_from!(ControlId, Control);
node_id_from!(VerificationActivityId, VerificationActivity);
node_id_from!(ConfigurationId, Configuration);
node_id_from!(ChangeId, Change);
node_id_from!(DecisionId, Decision);

/// Descriptive model-lifecycle state only.
///
/// These names deliberately avoid `Accepted`, `Verified`, `Qualified`, or
/// similar authority-bearing vocabulary. A modeled requirement is not an ETK
/// accepted requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LifecycleState {
    Draft,
    Modeled,
    Deprecated,
    Retired,
}

impl LifecycleState {
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Draft => "draft",
            Self::Modeled => "modeled",
            Self::Deprecated => "deprecated",
            Self::Retired => "retired",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SemanticRevision([u8; 32]);

impl SemanticRevision {
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for SemanticRevision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sha256:{}", hex_bytes(&self.0))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GraphSnapshotId([u8; 32]);

impl GraphSnapshotId {
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for GraphSnapshotId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sha256:{}", hex_bytes(&self.0))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticNode {
    id: NodeId,
    label: String,
    statement: String,
    lifecycle: LifecycleState,
}

impl SemanticNode {
    pub fn new(
        id: NodeId,
        label: impl Into<String>,
        statement: impl Into<String>,
        lifecycle: LifecycleState,
    ) -> Result<Self, ModelError> {
        let label = label.into();
        let statement = statement.into();
        validate_text(&label)?;
        validate_text(&statement)?;
        Ok(Self {
            id,
            label,
            statement,
            lifecycle,
        })
    }

    pub fn id(&self) -> &NodeId {
        &self.id
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn statement(&self) -> &str {
        &self.statement
    }

    pub const fn lifecycle(&self) -> LifecycleState {
        self.lifecycle
    }

    fn validate(&self) -> Result<(), ModelError> {
        validate_identifier(self.id.local_id())?;
        validate_text(&self.label)?;
        validate_text(&self.statement)
    }

    pub fn revision(&self) -> SemanticRevision {
        let mut hasher = Sha256::new();
        push_field(&mut hasher, NODE_REVISION_DOMAIN.as_bytes());
        push_field(&mut hasher, self.id.kind().canonical_name().as_bytes());
        push_field(&mut hasher, self.id.local_id().as_bytes());
        push_field(&mut hasher, self.label.as_bytes());
        push_field(&mut hasher, self.statement.as_bytes());
        push_field(&mut hasher, self.lifecycle.canonical_name().as_bytes());
        SemanticRevision(finalize(hasher))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RelationKind {
    DerivedFrom,
    Refines,
    DecomposesTo,
    AllocatedTo,
    InterfacesWith,
    ConstrainedBy,
    Assumes,
    ThreatenedBy,
    MitigatedBy,
    VerifiedByIntent,
    ValidatedByIntent,
    DependsOn,
    IncludedInConfiguration,
    DecidedBy,
    Supersedes,
    Contradicts,
    Changes,
    Impacts,
    InvalidatesCandidate,
}

impl RelationKind {
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::DerivedFrom => "derived_from",
            Self::Refines => "refines",
            Self::DecomposesTo => "decomposes_to",
            Self::AllocatedTo => "allocated_to",
            Self::InterfacesWith => "interfaces_with",
            Self::ConstrainedBy => "constrained_by",
            Self::Assumes => "assumes",
            Self::ThreatenedBy => "threatened_by",
            Self::MitigatedBy => "mitigated_by",
            Self::VerifiedByIntent => "verified_by_intent",
            Self::ValidatedByIntent => "validated_by_intent",
            Self::DependsOn => "depends_on",
            Self::IncludedInConfiguration => "included_in_configuration",
            Self::DecidedBy => "decided_by",
            Self::Supersedes => "supersedes",
            Self::Contradicts => "contradicts",
            Self::Changes => "changes",
            Self::Impacts => "impacts",
            Self::InvalidatesCandidate => "invalidates_candidate",
        }
    }

    const fn forbids_cycles(self) -> bool {
        matches!(
            self,
            Self::DerivedFrom | Self::Refines | Self::DecomposesTo | Self::Supersedes
        )
    }

    const fn is_symmetric(self) -> bool {
        matches!(self, Self::Contradicts)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Relation {
    pub source: NodeId,
    pub kind: RelationKind,
    pub target: NodeId,
}

impl Relation {
    pub fn new(source: NodeId, kind: RelationKind, target: NodeId) -> Self {
        Self {
            source,
            kind,
            target,
        }
    }

    fn reverse(&self) -> Self {
        Self::new(self.target.clone(), self.kind, self.source.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImpactStep {
    pub from: NodeId,
    pub via: Relation,
    pub to: NodeId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImpactPath {
    pub root: NodeId,
    pub affected: NodeId,
    pub steps: Vec<ImpactStep>,
}

#[derive(Debug, Clone, Default)]
pub struct EngineeringGraph {
    nodes: BTreeMap<NodeId, SemanticNode>,
    relations: BTreeSet<Relation>,
}

impl EngineeringGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn relation_count(&self) -> usize {
        self.relations.len()
    }

    pub fn node(&self, id: &NodeId) -> Option<&SemanticNode> {
        self.nodes.get(id)
    }

    pub fn nodes(&self) -> impl Iterator<Item = (&NodeId, &SemanticNode)> {
        self.nodes.iter()
    }

    pub fn relations(&self) -> impl Iterator<Item = &Relation> {
        self.relations.iter()
    }

    pub fn insert_node(&mut self, node: SemanticNode) -> Result<(), ModelError> {
        node.validate()?;
        if self.nodes.contains_key(node.id()) {
            return Err(ModelError::DuplicateNode(node.id().to_string()));
        }
        self.nodes.insert(node.id().clone(), node);
        Ok(())
    }

    pub fn replace_node(
        &mut self,
        expected_id: &NodeId,
        replacement: SemanticNode,
    ) -> Result<SemanticRevision, ModelError> {
        replacement.validate()?;
        if replacement.id() != expected_id {
            return Err(ModelError::ReplacementIdentityMismatch {
                expected: expected_id.to_string(),
                actual: replacement.id().to_string(),
            });
        }
        let old = self
            .nodes
            .get(expected_id)
            .ok_or_else(|| ModelError::MissingNode(expected_id.to_string()))?;
        let old_revision = old.revision();
        self.nodes.insert(replacement.id().clone(), replacement);
        Ok(old_revision)
    }

    pub fn add_relation(&mut self, relation: Relation) -> Result<(), ModelError> {
        self.validate_relation_endpoints(&relation)?;
        self.validate_relation_roles(&relation)?;

        if relation.source == relation.target {
            return Err(ModelError::SelfRelation(relation.kind));
        }
        if self.relations.contains(&relation) {
            return Err(ModelError::DuplicateRelation);
        }
        if relation.kind.is_symmetric() && self.relations.contains(&relation.reverse()) {
            return Err(ModelError::DuplicateSymmetricRelation);
        }

        self.relations.insert(relation.clone());
        if relation.kind.forbids_cycles() && self.has_cycle_for(relation.kind) {
            self.relations.remove(&relation);
            return Err(ModelError::CyclicRelation(relation.kind));
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        for (id, node) in &self.nodes {
            node.validate()?;
            if node.id() != id {
                return Err(ModelError::ReplacementIdentityMismatch {
                    expected: id.to_string(),
                    actual: node.id().to_string(),
                });
            }
        }

        for relation in &self.relations {
            self.validate_relation_endpoints(relation)?;
            self.validate_relation_roles(relation)?;
            if relation.source == relation.target {
                return Err(ModelError::SelfRelation(relation.kind));
            }
        }

        for kind in [
            RelationKind::DerivedFrom,
            RelationKind::Refines,
            RelationKind::DecomposesTo,
            RelationKind::Supersedes,
        ] {
            if self.has_cycle_for(kind) {
                return Err(ModelError::CyclicRelation(kind));
            }
        }

        for relation in self.relations.iter().filter(|r| r.kind.is_symmetric()) {
            if self.relations.contains(&relation.reverse()) {
                return Err(ModelError::DuplicateSymmetricRelation);
            }
        }

        Ok(())
    }

    pub fn snapshot_id(&self) -> GraphSnapshotId {
        let mut hasher = Sha256::new();
        push_field(&mut hasher, GRAPH_SNAPSHOT_DOMAIN.as_bytes());

        push_field(&mut hasher, b"nodes");
        push_u64_field(&mut hasher, self.nodes.len() as u64);
        for (id, node) in &self.nodes {
            push_field(&mut hasher, b"node");
            push_field(&mut hasher, id.canonical_key().as_bytes());
            push_field(&mut hasher, node.revision().as_bytes());
        }

        push_field(&mut hasher, b"relations");
        push_u64_field(&mut hasher, self.relations.len() as u64);
        for relation in &self.relations {
            push_field(&mut hasher, b"relation");
            push_field(&mut hasher, relation.source.canonical_key().as_bytes());
            push_field(&mut hasher, relation.kind.canonical_name().as_bytes());
            push_field(&mut hasher, relation.target.canonical_key().as_bytes());
        }

        GraphSnapshotId(finalize(hasher))
    }

    /// Returns one deterministic shortest review path to each transitively
    /// affected semantic node.
    ///
    /// `ImpactStep` records the actual traversal direction independently from
    /// the authored direction of the underlying relation. The result is still
    /// descriptive only: it does not invalidate ETK evidence or mint authority.
    pub fn impact_paths_from(&self, changed: &NodeId) -> Result<Vec<ImpactPath>, ModelError> {
        if !self.nodes.contains_key(changed) {
            return Err(ModelError::MissingNode(changed.to_string()));
        }

        let mut adjacency: BTreeMap<NodeId, Vec<(NodeId, Relation)>> = BTreeMap::new();
        for relation in &self.relations {
            for (from, to) in impact_arcs(relation) {
                adjacency
                    .entry(from)
                    .or_default()
                    .push((to, relation.clone()));
            }
        }
        for neighbors in adjacency.values_mut() {
            neighbors.sort_by(|left, right| {
                left.0
                    .cmp(&right.0)
                    .then_with(|| left.1.kind.cmp(&right.1.kind))
                    .then_with(|| left.1.source.cmp(&right.1.source))
                    .then_with(|| left.1.target.cmp(&right.1.target))
            });
        }

        let mut queue = VecDeque::from([changed.clone()]);
        let mut visited = BTreeSet::from([changed.clone()]);
        let mut paths: BTreeMap<NodeId, Vec<ImpactStep>> = BTreeMap::new();

        while let Some(current) = queue.pop_front() {
            let Some(neighbors) = adjacency.get(&current) else {
                continue;
            };
            for (neighbor, relation) in neighbors {
                if visited.contains(neighbor) {
                    continue;
                }

                let mut path = paths.get(&current).cloned().unwrap_or_default();
                path.push(ImpactStep {
                    from: current.clone(),
                    via: relation.clone(),
                    to: neighbor.clone(),
                });
                paths.insert(neighbor.clone(), path);

                visited.insert(neighbor.clone());
                queue.push_back(neighbor.clone());
            }
        }

        Ok(paths
            .into_iter()
            .map(|(affected, steps)| ImpactPath {
                root: changed.clone(),
                affected,
                steps,
            })
            .collect())
    }

    fn validate_relation_endpoints(&self, relation: &Relation) -> Result<(), ModelError> {
        for id in [&relation.source, &relation.target] {
            if !self.nodes.contains_key(id) {
                return Err(ModelError::MissingNode(id.to_string()));
            }
        }
        Ok(())
    }

    fn validate_relation_roles(&self, relation: &Relation) -> Result<(), ModelError> {
        let source = relation.source.kind();
        let target = relation.target.kind();

        let valid = match relation.kind {
            RelationKind::DerivedFrom => {
                source == NodeKind::Requirement
                    && matches!(target, NodeKind::Requirement | NodeKind::StakeholderNeed)
            }
            RelationKind::Refines => source == target,
            RelationKind::DecomposesTo => {
                (source == NodeKind::Function && target == NodeKind::Function)
                    || (source == NodeKind::Requirement && target == NodeKind::Requirement)
            }
            RelationKind::AllocatedTo => {
                source == NodeKind::Function && target == NodeKind::Component
            }
            RelationKind::InterfacesWith => {
                source == NodeKind::Interface && target == NodeKind::Component
            }
            RelationKind::ConstrainedBy => target == NodeKind::Constraint,
            RelationKind::Assumes => target == NodeKind::Assumption,
            RelationKind::ThreatenedBy => target == NodeKind::Hazard,
            RelationKind::MitigatedBy => {
                source == NodeKind::Hazard && target == NodeKind::Control
            }
            RelationKind::VerifiedByIntent => {
                source == NodeKind::Requirement && target == NodeKind::VerificationActivity
            }
            RelationKind::ValidatedByIntent => {
                matches!(source, NodeKind::Requirement | NodeKind::StakeholderNeed)
                    && target == NodeKind::VerificationActivity
            }
            RelationKind::DependsOn => true,
            RelationKind::IncludedInConfiguration => {
                source != NodeKind::Configuration && target == NodeKind::Configuration
            }
            RelationKind::DecidedBy => {
                source != NodeKind::Decision && target == NodeKind::Decision
            }
            RelationKind::Supersedes | RelationKind::Contradicts => source == target,
            RelationKind::Changes
            | RelationKind::Impacts
            | RelationKind::InvalidatesCandidate => source == NodeKind::Change,
        };

        if valid {
            Ok(())
        } else {
            Err(ModelError::InvalidRelationRoles {
                kind: relation.kind,
                source,
                target,
            })
        }
    }

    fn has_cycle_for(&self, kind: RelationKind) -> bool {
        let mut adjacency: BTreeMap<NodeId, Vec<NodeId>> = BTreeMap::new();
        for relation in self.relations.iter().filter(|relation| relation.kind == kind) {
            adjacency
                .entry(relation.source.clone())
                .or_default()
                .push(relation.target.clone());
        }

        let mut temporary = BTreeSet::new();
        let mut permanent = BTreeSet::new();

        self.nodes
            .keys()
            .any(|node| cycle_visit(node, &adjacency, &mut temporary, &mut permanent))
    }
}

fn cycle_visit(
    node: &NodeId,
    adjacency: &BTreeMap<NodeId, Vec<NodeId>>,
    temporary: &mut BTreeSet<NodeId>,
    permanent: &mut BTreeSet<NodeId>,
) -> bool {
    if permanent.contains(node) {
        return false;
    }
    if !temporary.insert(node.clone()) {
        return true;
    }
    if let Some(neighbors) = adjacency.get(node) {
        for neighbor in neighbors {
            if cycle_visit(neighbor, adjacency, temporary, permanent) {
                return true;
            }
        }
    }
    temporary.remove(node);
    permanent.insert(node.clone());
    false
}

fn impact_arcs(relation: &Relation) -> Vec<(NodeId, NodeId)> {
    use RelationKind::*;

    match relation.kind {
        // Source depends semantically on target.
        DerivedFrom | Refines | ConstrainedBy | Assumes | DependsOn | DecidedBy => {
            vec![(relation.target.clone(), relation.source.clone())]
        }

        // Configuration membership: member changes affect the configuration and
        // configuration changes may require review of included members.
        IncludedInConfiguration => vec![
            (relation.source.clone(), relation.target.clone()),
            (relation.target.clone(), relation.source.clone()),
        ],

        // Conservative bidirectional coupling.
        DecomposesTo
        | AllocatedTo
        | InterfacesWith
        | ThreatenedBy
        | MitigatedBy
        | VerifiedByIntent
        | ValidatedByIntent
        | Supersedes
        | Contradicts => vec![
            (relation.source.clone(), relation.target.clone()),
            (relation.target.clone(), relation.source.clone()),
        ],

        // Explicit change edges propagate as authored.
        Changes | Impacts | InvalidatesCandidate => {
            vec![(relation.source.clone(), relation.target.clone())]
        }
    }
}

fn push_field(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
}

fn push_u64_field(hasher: &mut Sha256, value: u64) {
    push_field(hasher, &value.to_be_bytes());
}

fn finalize(hasher: Sha256) -> [u8; 32] {
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 32];
    bytes.copy_from_slice(&digest);
    bytes
}

fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: NodeId, label: &str, statement: &str) -> SemanticNode {
        SemanticNode::new(id, label, statement, LifecycleState::Modeled).unwrap()
    }

    #[test]
    fn identifier_validation_fails_closed() {
        assert_eq!(
            RequirementId::new("  ").unwrap_err(),
            ModelError::InvalidIdentifier
        );
        assert_eq!(
            ComponentId::new("bad\nid").unwrap_err(),
            ModelError::InvalidIdentifier
        );
    }

    #[test]
    fn lifecycle_vocabulary_is_descriptive_not_authority_bearing() {
        assert_eq!(
            [
                LifecycleState::Draft.canonical_name(),
                LifecycleState::Modeled.canonical_name(),
                LifecycleState::Deprecated.canonical_name(),
                LifecycleState::Retired.canonical_name(),
            ],
            ["draft", "modeled", "deprecated", "retired"]
        );
    }

    #[test]
    fn dangling_relation_is_rejected() {
        let mut graph = EngineeringGraph::new();
        let req = NodeId::Requirement(RequirementId::new("REQ-1").unwrap());
        graph
            .insert_node(node(req.clone(), "shutdown", "Provide safe shutdown"))
            .unwrap();

        let missing = NodeId::StakeholderNeed(StakeholderNeedId::new("NEED-1").unwrap());
        assert!(matches!(
            graph
                .add_relation(Relation::new(req, RelationKind::DerivedFrom, missing))
                .unwrap_err(),
            ModelError::MissingNode(_)
        ));
    }

    #[test]
    fn need_cannot_be_derived_from_requirement() {
        let mut graph = EngineeringGraph::new();
        let need = NodeId::StakeholderNeed(StakeholderNeedId::new("NEED-1").unwrap());
        let req = NodeId::Requirement(RequirementId::new("REQ-1").unwrap());
        graph
            .insert_node(node(need.clone(), "need", "Protect occupants"))
            .unwrap();
        graph
            .insert_node(node(req.clone(), "shutdown", "Provide safe shutdown"))
            .unwrap();

        assert!(matches!(
            graph
                .add_relation(Relation::new(need, RelationKind::DerivedFrom, req))
                .unwrap_err(),
            ModelError::InvalidRelationRoles { .. }
        ));
    }

    #[test]
    fn role_mismatch_is_rejected() {
        let mut graph = EngineeringGraph::new();
        let req = NodeId::Requirement(RequirementId::new("REQ-1").unwrap());
        let component = NodeId::Component(ComponentId::new("COMP-1").unwrap());
        graph
            .insert_node(node(req.clone(), "shutdown", "Provide safe shutdown"))
            .unwrap();
        graph
            .insert_node(node(component.clone(), "controller", "Control computer"))
            .unwrap();

        assert!(matches!(
            graph
                .add_relation(Relation::new(
                    req,
                    RelationKind::AllocatedTo,
                    component
                ))
                .unwrap_err(),
            ModelError::InvalidRelationRoles { .. }
        ));
    }

    #[test]
    fn duplicate_and_reverse_duplicate_symmetric_relations_are_rejected() {
        let mut graph = EngineeringGraph::new();
        let a = NodeId::Requirement(RequirementId::new("REQ-A").unwrap());
        let b = NodeId::Requirement(RequirementId::new("REQ-B").unwrap());
        graph
            .insert_node(node(a.clone(), "A", "Requirement A"))
            .unwrap();
        graph
            .insert_node(node(b.clone(), "B", "Requirement B"))
            .unwrap();

        let relation = Relation::new(a.clone(), RelationKind::Contradicts, b.clone());
        graph.add_relation(relation.clone()).unwrap();
        assert_eq!(
            graph.add_relation(relation).unwrap_err(),
            ModelError::DuplicateRelation
        );
        assert_eq!(
            graph
                .add_relation(Relation::new(b, RelationKind::Contradicts, a))
                .unwrap_err(),
            ModelError::DuplicateSymmetricRelation
        );
    }

    #[test]
    fn forbidden_refinement_and_supersession_cycles_are_rejected() {
        for kind in [RelationKind::Refines, RelationKind::Supersedes] {
            let mut graph = EngineeringGraph::new();
            let a = NodeId::Requirement(RequirementId::new("REQ-A").unwrap());
            let b = NodeId::Requirement(RequirementId::new("REQ-B").unwrap());
            graph
                .insert_node(node(a.clone(), "A", "Requirement A"))
                .unwrap();
            graph
                .insert_node(node(b.clone(), "B", "Requirement B"))
                .unwrap();

            graph
                .add_relation(Relation::new(a.clone(), kind, b.clone()))
                .unwrap();

            assert_eq!(
                graph
                    .add_relation(Relation::new(b, kind, a))
                    .unwrap_err(),
                ModelError::CyclicRelation(kind)
            );
        }
    }

    #[test]
    fn semantic_revision_and_framed_snapshot_vectors_are_frozen() {
        let req = NodeId::Requirement(RequirementId::new("REQ-1").unwrap());
        let req_node = node(req, "shutdown", "Provide safe shutdown");

        assert_eq!(
            req_node.revision().to_string(),
            "sha256:30181cffc416564994ffb6e51340273d1dd7013c47f1596b48b3372ecea828dc"
        );

        let mut graph = EngineeringGraph::new();
        graph.insert_node(req_node).unwrap();

        assert_eq!(
            graph.snapshot_id().to_string(),
            "sha256:94c024e10ac11cc82a5308139def332211d3e5bb98391bda0eada0f54297568c"
        );
    }

    #[test]
    fn snapshot_is_insertion_order_invariant() {
        let need = NodeId::StakeholderNeed(StakeholderNeedId::new("NEED-1").unwrap());
        let req = NodeId::Requirement(RequirementId::new("REQ-1").unwrap());
        let need_node = node(need.clone(), "occupants", "Protect occupants");
        let req_node = node(req.clone(), "shutdown", "Provide safe shutdown");
        let relation = Relation::new(req.clone(), RelationKind::DerivedFrom, need.clone());

        let mut left = EngineeringGraph::new();
        left.insert_node(need_node.clone()).unwrap();
        left.insert_node(req_node.clone()).unwrap();
        left.add_relation(relation.clone()).unwrap();

        let mut right = EngineeringGraph::new();
        right.insert_node(req_node).unwrap();
        right.insert_node(need_node).unwrap();
        right.add_relation(relation).unwrap();

        assert_eq!(left.snapshot_id(), right.snapshot_id());
    }

    #[test]
    fn semantic_change_rekeys_revision_and_snapshot() {
        let req = NodeId::Requirement(RequirementId::new("REQ-1").unwrap());
        let original = node(req.clone(), "shutdown", "Provide safe shutdown");

        let mut graph = EngineeringGraph::new();
        graph.insert_node(original.clone()).unwrap();

        let before_revision = original.revision();
        let before_snapshot = graph.snapshot_id();

        let changed = node(
            req.clone(),
            "shutdown",
            "Provide safe shutdown within 100 ms",
        );
        let old_revision = graph.replace_node(&req, changed.clone()).unwrap();

        assert_eq!(old_revision, before_revision);
        assert_ne!(changed.revision(), before_revision);
        assert_ne!(graph.snapshot_id(), before_snapshot);
    }

    #[test]
    fn assumption_change_path_records_actual_traversal_direction() {
        let mut graph = EngineeringGraph::new();

        let assumption = NodeId::Assumption(AssumptionId::new("ASM-THERMAL").unwrap());
        let req = NodeId::Requirement(RequirementId::new("REQ-THERMAL").unwrap());
        let function = NodeId::Function(FunctionId::new("FUNC-CONTROL").unwrap());
        let component = NodeId::Component(ComponentId::new("COMP-CONTROLLER").unwrap());

        graph
            .insert_node(node(
                assumption.clone(),
                "ambient",
                "Ambient stays below 40 C",
            ))
            .unwrap();
        graph
            .insert_node(node(req.clone(), "thermal", "Stay within limits"))
            .unwrap();
        graph
            .insert_node(node(function.clone(), "regulate", "Regulate actuator"))
            .unwrap();
        graph
            .insert_node(node(component.clone(), "controller", "Control computer"))
            .unwrap();

        graph
            .add_relation(Relation::new(
                req.clone(),
                RelationKind::Assumes,
                assumption.clone(),
            ))
            .unwrap();
        graph
            .add_relation(Relation::new(
                function.clone(),
                RelationKind::DependsOn,
                req.clone(),
            ))
            .unwrap();
        graph
            .add_relation(Relation::new(
                function,
                RelationKind::AllocatedTo,
                component.clone(),
            ))
            .unwrap();

        let paths = graph.impact_paths_from(&assumption).unwrap();
        let component_path = paths
            .iter()
            .find(|path| path.affected == component)
            .unwrap();

        assert_eq!(component_path.steps.len(), 3);
        assert_eq!(component_path.steps[0].from, assumption);
        assert_eq!(component_path.steps[0].to, req);
        assert_eq!(component_path.steps[0].via.kind, RelationKind::Assumes);
    }

    #[test]
    fn configuration_and_decision_are_semantically_connected() {
        let mut graph = EngineeringGraph::new();
        let battery = NodeId::Component(ComponentId::new("COMP-BATTERY").unwrap());
        let config = NodeId::Configuration(ConfigurationId::new("CFG-A").unwrap());
        let decision = NodeId::Decision(DecisionId::new("DEC-BATTERY").unwrap());

        graph
            .insert_node(node(battery.clone(), "battery", "Energy storage"))
            .unwrap();
        graph
            .insert_node(node(config.clone(), "configuration", "Flight configuration A"))
            .unwrap();
        graph
            .insert_node(node(decision.clone(), "decision", "Select battery chemistry"))
            .unwrap();

        graph
            .add_relation(Relation::new(
                battery.clone(),
                RelationKind::IncludedInConfiguration,
                config.clone(),
            ))
            .unwrap();
        graph
            .add_relation(Relation::new(
                battery.clone(),
                RelationKind::DecidedBy,
                decision.clone(),
            ))
            .unwrap();

        let config_affected: BTreeSet<_> = graph
            .impact_paths_from(&battery)
            .unwrap()
            .into_iter()
            .map(|path| path.affected)
            .collect();
        assert!(config_affected.contains(&config));

        let decision_affected: BTreeSet<_> = graph
            .impact_paths_from(&decision)
            .unwrap()
            .into_iter()
            .map(|path| path.affected)
            .collect();
        assert!(decision_affected.contains(&battery));
    }

    #[test]
    fn interface_is_first_class_and_impacts_endpoints() {
        let mut graph = EngineeringGraph::new();
        let interface = NodeId::Interface(InterfaceId::new("IF-CONTROL").unwrap());
        let controller = NodeId::Component(ComponentId::new("COMP-CONTROLLER").unwrap());
        let actuator = NodeId::Component(ComponentId::new("COMP-ACTUATOR").unwrap());

        graph
            .insert_node(node(
                interface.clone(),
                "command",
                "Actuator command interface",
            ))
            .unwrap();
        graph
            .insert_node(node(controller.clone(), "controller", "Control computer"))
            .unwrap();
        graph
            .insert_node(node(actuator.clone(), "actuator", "Actuator"))
            .unwrap();

        graph
            .add_relation(Relation::new(
                interface.clone(),
                RelationKind::InterfacesWith,
                controller.clone(),
            ))
            .unwrap();
        graph
            .add_relation(Relation::new(
                interface.clone(),
                RelationKind::InterfacesWith,
                actuator.clone(),
            ))
            .unwrap();

        let affected: BTreeSet<_> = graph
            .impact_paths_from(&interface)
            .unwrap()
            .into_iter()
            .map(|path| path.affected)
            .collect();

        assert!(affected.contains(&controller));
        assert!(affected.contains(&actuator));
    }

    #[test]
    fn change_edges_are_review_paths_not_authority_state() {
        let mut graph = EngineeringGraph::new();
        let change = NodeId::Change(ChangeId::new("CHG-1").unwrap());
        let battery = NodeId::Component(ComponentId::new("COMP-BATTERY").unwrap());

        graph
            .insert_node(node(
                change.clone(),
                "battery change",
                "Replace battery chemistry",
            ))
            .unwrap();
        graph
            .insert_node(node(battery.clone(), "battery", "Energy storage"))
            .unwrap();
        graph
            .add_relation(Relation::new(
                change.clone(),
                RelationKind::Changes,
                battery.clone(),
            ))
            .unwrap();

        let paths = graph.impact_paths_from(&change).unwrap();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].affected, battery);
        assert_eq!(paths[0].steps[0].from, change);
    }
}
