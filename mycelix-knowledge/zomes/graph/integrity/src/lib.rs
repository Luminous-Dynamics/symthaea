// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Graph Integrity Zome
//! Defines entry types and validation for knowledge graph relationships
//!
//! Updated to use HDI 0.7 patterns

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// A relationship/edge between two claims
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Relationship {
    /// Relationship identifier
    pub id: String,
    /// Source claim ID
    pub source: String,
    /// Target claim ID
    pub target: String,
    /// Relationship type
    pub relationship_type: RelationshipType,
    /// Relationship strength/weight (0.0 to 1.0)
    pub weight: f64,
    /// Additional properties (JSON)
    pub properties: Option<String>,
    /// Creator's DID
    pub creator: String,
    /// Creation timestamp
    pub created: Timestamp,
}

/// Types of relationships between claims
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RelationshipType {
    /// Source supports/confirms target
    Supports,
    /// Source contradicts target
    Contradicts,
    /// Source is derived from target
    DerivedFrom,
    /// Source is an example of target
    ExampleOf,
    /// Source is a generalization of target
    Generalizes,
    /// Source is part of target
    PartOf,
    /// Source causes target
    Causes,
    /// Source is related to target (generic)
    RelatedTo,
    /// Source is equivalent to target
    Equivalent,
    /// Source is more specific than target
    SpecializedFrom,
    /// Custom relationship type
    Custom(String),
}

/// Ontology definition for knowledge organization
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Ontology {
    /// Ontology identifier
    pub id: String,
    /// Ontology name
    pub name: String,
    /// Description
    pub description: String,
    /// Namespace URI
    pub namespace: String,
    /// Schema (JSON-LD or similar)
    pub schema: String,
    /// Version
    pub version: String,
    /// Creator's DID
    pub creator: String,
    /// Creation timestamp
    pub created: Timestamp,
    /// Last update timestamp
    pub updated: Timestamp,
}

/// Concept within an ontology
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Concept {
    /// Concept identifier
    pub id: String,
    /// Ontology this concept belongs to
    pub ontology_id: String,
    /// Concept name
    pub name: String,
    /// Definition
    pub definition: String,
    /// Parent concept (if any)
    pub parent: Option<String>,
    /// Synonyms
    pub synonyms: Vec<String>,
    /// Creation timestamp
    pub created: Timestamp,
}

// ============================================================================
// BELIEF GRAPH TYPES
// ============================================================================

/// A node in the belief graph with propagated belief strength
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BeliefNode {
    /// Node identifier (matches claim ID)
    pub id: String,

    /// Claim this node represents
    pub claim_id: String,

    /// Current belief strength (0.0 to 1.0)
    pub belief_strength: f64,

    /// Prior belief (before propagation)
    pub prior_belief: f64,

    /// Confidence in this belief (0.0 to 1.0)
    pub confidence: f64,

    /// Number of supporting connections
    pub support_count: u32,

    /// Number of contradicting connections
    pub contradiction_count: u32,

    /// Last update timestamp
    pub last_updated: Timestamp,

    /// Number of propagation iterations
    pub propagation_iterations: u32,

    /// Whether this node has converged
    pub converged: bool,

    /// Sources of influence on this node
    pub influences: Vec<BeliefInfluence>,
}

/// An influence on a belief node
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BeliefInfluence {
    /// Source claim ID
    pub source_claim_id: String,

    /// Type of influence
    pub influence_type: InfluenceType,

    /// Weight of influence
    pub weight: f64,

    /// Source belief at time of influence
    pub source_belief: f64,
}

/// Type of influence between beliefs
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum InfluenceType {
    /// Direct support
    Support,
    /// Direct contradiction
    Contradiction,
    /// Evidential relevance
    Evidential,
    /// Logical entailment
    Entailment,
    /// Market verification
    MarketVerification,
}

/// Information value assessment for a claim
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct InformationValue {
    /// Unique identifier
    pub id: String,

    /// Claim this assessment is for
    pub claim_id: String,

    /// Expected value of resolving this claim
    pub expected_value: f64,

    /// Number of claims that depend on this one
    pub dependent_count: u32,

    /// Average weight of dependencies
    pub average_dependency_weight: f64,

    /// Current uncertainty (entropy-based)
    pub uncertainty: f64,

    /// Potential impact on knowledge graph
    pub impact_score: f64,

    /// Recommended for verification?
    pub recommended_for_verification: bool,

    /// Assessment timestamp
    pub assessed_at: Timestamp,

    /// Reasoning for assessment
    pub reasoning: String,
}

/// Result of belief propagation
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PropagationResult {
    /// Claim that was propagated from
    pub source_claim_id: String,

    /// Number of nodes affected
    pub nodes_affected: u32,

    /// Number of iterations to converge
    pub iterations: u32,

    /// Whether propagation converged
    pub converged: bool,

    /// Maximum change in any belief
    pub max_delta: f64,

    /// Processing time in milliseconds
    pub processing_time_ms: u64,

    /// Nodes that were updated
    pub updated_nodes: Vec<String>,
}

/// Dependency tree structure
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DependencyTree {
    /// Root claim
    pub root_claim_id: String,

    /// Tree nodes
    pub nodes: Vec<DependencyTreeNode>,

    /// Total depth of tree
    pub depth: u32,

    /// Total number of dependencies
    pub total_dependencies: u32,

    /// Aggregate weight
    pub aggregate_weight: f64,
}

/// Node in dependency tree
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DependencyTreeNode {
    /// Claim ID
    pub claim_id: String,

    /// Depth in tree (0 = root)
    pub depth: u32,

    /// Weight of connection to parent
    pub weight: f64,

    /// Children claim IDs
    pub children: Vec<String>,

    /// Is this a leaf node?
    pub is_leaf: bool,
}

/// Impact assessment for cascade
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CascadeImpact {
    /// Claim being assessed
    pub claim_id: String,

    /// Total number of claims affected
    pub total_affected: u32,

    /// Claims at each depth level
    pub affected_by_depth: Vec<u32>,

    /// Maximum cascade depth
    pub max_depth: u32,

    /// Aggregate impact score
    pub impact_score: f64,

    /// High-impact claims (> 0.8 weight)
    pub high_impact_claims: Vec<String>,

    /// Processing timestamp
    pub assessed_at: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Relationship(Relationship),
    Ontology(Ontology),
    Concept(Concept),
    BeliefNode(BeliefNode),
    InformationValue(InformationValue),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Source claim to relationship
    ClaimToOutgoingRelation,
    /// Target claim to relationship
    ClaimToIncomingRelation,
    /// Ontology to its concepts
    OntologyToConcept,
    /// Concept to child concepts
    ConceptToChild,
    /// Concept to claims using it
    ConceptToClaim,
    /// Claim to belief node
    ClaimToBeliefNode,
    /// Claim to information value assessment
    ClaimToInformationValue,
    /// Index by belief strength range
    BeliefStrengthIndex,
    /// Index by information value
    InformationValueIndex,
}

/// HDI 0.7 single validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Relationship(rel) => validate_create_relationship(action, rel),
                EntryTypes::Ontology(ont) => validate_create_ontology(action, ont),
                EntryTypes::Concept(concept) => validate_create_concept(action, concept),
                EntryTypes::BeliefNode(node) => validate_create_belief_node(action, node),
                EntryTypes::InformationValue(val) => validate_create_information_value(action, val),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Relationship(rel) => validate_update_relationship(action, rel),
                EntryTypes::Ontology(ont) => validate_update_ontology(action, ont),
                EntryTypes::Concept(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::BeliefNode(node) => validate_update_belief_node(action, node),
                EntryTypes::InformationValue(_) => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::ClaimToOutgoingRelation => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ClaimToIncomingRelation => Ok(ValidateCallbackResult::Valid),
            LinkTypes::OntologyToConcept => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ConceptToChild => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ConceptToClaim => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ClaimToBeliefNode => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ClaimToInformationValue => Ok(ValidateCallbackResult::Valid),
            LinkTypes::BeliefStrengthIndex => Ok(ValidateCallbackResult::Valid),
            LinkTypes::InformationValueIndex => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            original_action,
            action,
            ..
        } => {
            // Only the original link creator can delete a link
            if action.author != original_action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original link creator can delete a link".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(op_update) => {
            // Only the original author can update an entry
            let update_action = match op_update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(update_action.original_action_address.clone())?;
            if update_action.author != *original.hashed.author() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can update an entry".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDelete(op_delete) => {
            // Only the original author can delete an entry
            let original = must_get_action(op_delete.action.deletes_address.clone())?;
            if op_delete.action.author != *original.hashed.author() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete an entry".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
    }
}

/// Validate relationship creation
fn validate_create_relationship(_action: Create, relationship: Relationship) -> ExternResult<ValidateCallbackResult> {
    // Validate weight range
    if relationship.weight < 0.0 || relationship.weight > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Weight must be between 0.0 and 1.0".into(),
        ));
    }

    // Source and target must be different
    if relationship.source == relationship.target {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot create self-referential relationship".into(),
        ));
    }

    // Validate creator is a DID
    if !relationship.creator.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Creator must be a valid DID".into(),
        ));
    }

    // Validate properties is valid JSON if provided
    if let Some(ref props) = relationship.properties {
        if serde_json::from_str::<serde_json::Value>(props).is_err() {
            return Ok(ValidateCallbackResult::Invalid(
                "Properties must be valid JSON".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate ontology creation
fn validate_create_ontology(_action: Create, ontology: Ontology) -> ExternResult<ValidateCallbackResult> {
    // Validate namespace is a URI
    if !ontology.namespace.starts_with("http://") && !ontology.namespace.starts_with("https://") {
        return Ok(ValidateCallbackResult::Invalid(
            "Namespace must be a valid URI".into(),
        ));
    }

    // Validate schema is valid JSON
    if serde_json::from_str::<serde_json::Value>(&ontology.schema).is_err() {
        return Ok(ValidateCallbackResult::Invalid(
            "Schema must be valid JSON".into(),
        ));
    }

    // Validate creator is a DID
    if !ontology.creator.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Creator must be a valid DID".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate concept creation
fn validate_create_concept(_action: Create, concept: Concept) -> ExternResult<ValidateCallbackResult> {
    // Validate name not empty
    if concept.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Concept name cannot be empty".into(),
        ));
    }

    // Validate definition not empty
    if concept.definition.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Concept definition cannot be empty".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate belief node creation
fn validate_create_belief_node(_action: Create, node: BeliefNode) -> ExternResult<ValidateCallbackResult> {
    // Validate belief strength range
    if node.belief_strength < 0.0 || node.belief_strength > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Belief strength must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate prior belief range
    if node.prior_belief < 0.0 || node.prior_belief > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Prior belief must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate confidence range
    if node.confidence < 0.0 || node.confidence > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Confidence must be between 0.0 and 1.0".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate information value creation
fn validate_create_information_value(_action: Create, val: InformationValue) -> ExternResult<ValidateCallbackResult> {
    // Validate expected value is non-negative
    if val.expected_value < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Expected value must be non-negative".into(),
        ));
    }

    // Validate uncertainty range
    if val.uncertainty < 0.0 || val.uncertainty > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Uncertainty must be between 0.0 and 1.0".into(),
        ));
    }

    // Validate impact score range
    if val.impact_score < 0.0 || val.impact_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Impact score must be between 0.0 and 1.0".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate relationship update
fn validate_update_relationship(_action: Update, relationship: Relationship) -> ExternResult<ValidateCallbackResult> {
    // Validate weight range
    if relationship.weight < 0.0 || relationship.weight > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Weight must be between 0.0 and 1.0".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate ontology update
fn validate_update_ontology(_action: Update, ontology: Ontology) -> ExternResult<ValidateCallbackResult> {
    // Validate schema is valid JSON
    if serde_json::from_str::<serde_json::Value>(&ontology.schema).is_err() {
        return Ok(ValidateCallbackResult::Invalid(
            "Schema must be valid JSON".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate belief node update
fn validate_update_belief_node(_action: Update, node: BeliefNode) -> ExternResult<ValidateCallbackResult> {
    // Validate belief strength range
    if node.belief_strength < 0.0 || node.belief_strength > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Belief strength must be between 0.0 and 1.0".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
