// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Epistemic Provenance Chains
//!
//! Cryptographic proof of knowledge origin and derivation.
//!
//! ## Philosophy
//!
//! Every piece of knowledge has a genealogy. This module tracks:
//! - Where did this knowledge come from?
//! - How was it derived from its sources?
//! - How confident can we be, given its lineage?
//! - What is the highest epistemic level it can claim?
//!
//! ## Key Concepts
//!
//! - **Provenance Node**: A single piece of knowledge with hash and parents
//! - **Derivation Type**: How knowledge was transformed (synthesis, citation, etc.)
//! - **Confidence Decay**: Each derivation step reduces confidence
//! - **Epistemic Floor**: Cannot exceed the lowest parent's epistemic level
//! - **Chain Verification**: Cryptographic proof of unbroken lineage

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;

use crate::epistemic::{EmpiricalLevel, EpistemicClassificationExtended};

#[cfg(feature = "ts-export")]
use ts_rs::TS;

// ============================================================================
// Core Types
// ============================================================================

/// How knowledge was derived from its sources
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum DerivationType {
    /// Original creation - no parents
    Original,
    /// Direct citation of a single source
    Citation,
    /// Synthesis of multiple sources into new insight
    Synthesis,
    /// Transformation/reformulation of a source
    Transform,
    /// Inference from premises
    Inference,
    /// Aggregation of multiple similar sources
    Aggregation,
    /// Verification/confirmation of existing knowledge
    Verification,
    /// Translation between domains/formats
    Translation,
}

impl DerivationType {
    /// Confidence decay factor for this derivation type
    /// Lower = more confidence lost per derivation
    pub fn confidence_decay(&self) -> f64 {
        match self {
            Self::Original => 1.0,      // No decay
            Self::Citation => 0.98,     // Minimal decay
            Self::Verification => 0.99, // Almost no decay (confirms)
            Self::Aggregation => 0.95,  // Slight decay
            Self::Transform => 0.90,    // Moderate decay
            Self::Translation => 0.88,  // Moderate decay
            Self::Synthesis => 0.85,    // More decay (creative)
            Self::Inference => 0.80,    // Most decay (reasoning can fail)
        }
    }

    /// Whether this derivation can potentially raise epistemic level
    pub fn can_raise_epistemic(&self) -> bool {
        matches!(self, Self::Verification | Self::Aggregation)
    }
}

/// A node in the provenance chain
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct ProvenanceNode {
    /// Unique identifier (content hash)
    pub node_id: String,

    /// SHA3-256 hash of the content
    pub content_hash: [u8; 32],

    /// Hash of parent nodes (empty for Original)
    pub parent_hashes: Vec<[u8; 32]>,

    /// How this was derived from parents
    pub derivation_type: DerivationType,

    /// Agent that created this node
    pub creator_agent_id: String,

    /// Timestamp of creation
    pub timestamp: u64,

    /// Original confidence (before decay)
    pub base_confidence: f64,

    /// Confidence after applying derivation decay
    pub derived_confidence: f64,

    /// Epistemic classification of this node
    pub epistemic_level: EmpiricalLevel,

    /// The lowest epistemic level in the parent chain
    pub epistemic_floor: EmpiricalLevel,

    /// Depth in the provenance chain (0 for Original)
    pub chain_depth: u32,

    /// Optional human-readable description
    pub description: Option<String>,

    /// Commitment hash binding all fields
    pub commitment: [u8; 32],
}

impl ProvenanceNode {
    /// Verify the commitment hash
    pub fn verify_commitment(&self) -> bool {
        let computed = self.compute_commitment();
        constant_time_eq(&self.commitment, &computed)
    }

    /// Compute commitment hash from all fields
    fn compute_commitment(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();

        // Domain separation
        hasher.update(b"mycelix-provenance-commitment-v1");

        // Include all fields
        hasher.update(self.content_hash);
        for parent in &self.parent_hashes {
            hasher.update(parent);
        }
        hasher.update([self.derivation_type as u8]);
        hasher.update(self.creator_agent_id.as_bytes());
        hasher.update(self.timestamp.to_le_bytes());
        hasher.update(self.base_confidence.to_le_bytes());
        hasher.update(self.derived_confidence.to_le_bytes());
        hasher.update([self.epistemic_level as u8]);
        hasher.update([self.epistemic_floor as u8]);
        hasher.update(self.chain_depth.to_le_bytes());

        hasher.finalize().into()
    }
}

/// Constant-time comparison to prevent timing attacks
fn constant_time_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut result = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }
    result == 0
}

// ============================================================================
// Provenance Chain
// ============================================================================

/// A complete provenance chain for a piece of knowledge
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct ProvenanceChain {
    /// The final/current node
    pub head: ProvenanceNode,

    /// All nodes in the chain, indexed by content hash
    pub nodes: HashMap<String, ProvenanceNode>,

    /// Root nodes (Original derivation type)
    pub roots: Vec<String>,

    /// Total chain length (longest path to root)
    pub max_depth: u32,

    /// Chain-level confidence (product of all decay factors)
    pub chain_confidence: f64,

    /// Whether all commitments verify
    pub verified: bool,
}

impl ProvenanceChain {
    /// Verify the entire chain
    pub fn verify(&self) -> ChainVerificationResult {
        let mut errors = Vec::new();

        // Verify each node's commitment
        for (id, node) in &self.nodes {
            if !node.verify_commitment() {
                errors.push(ChainError::InvalidCommitment(id.clone()));
            }
        }

        // Verify parent references exist
        for (id, node) in &self.nodes {
            for parent_hash in &node.parent_hashes {
                let parent_id = hex::encode(parent_hash);
                if !self.nodes.contains_key(&parent_id) {
                    errors.push(ChainError::MissingParent {
                        node: id.clone(),
                        parent: parent_id,
                    });
                }
            }
        }

        // Verify epistemic floor is respected
        for (id, node) in &self.nodes {
            if node.epistemic_level as u8 > node.epistemic_floor as u8
                && !node.derivation_type.can_raise_epistemic()
            {
                errors.push(ChainError::EpistemicViolation {
                    node: id.clone(),
                    claimed: node.epistemic_level,
                    floor: node.epistemic_floor,
                });
            }
        }

        // Verify roots have Original derivation
        for root_id in &self.roots {
            if let Some(node) = self.nodes.get(root_id) {
                if node.derivation_type != DerivationType::Original {
                    errors.push(ChainError::InvalidRoot(root_id.clone()));
                }
            }
        }

        if errors.is_empty() {
            ChainVerificationResult::Valid
        } else {
            ChainVerificationResult::Invalid(errors)
        }
    }

    /// Get the lineage (path to roots) for the head node
    pub fn get_lineage(&self) -> Vec<&ProvenanceNode> {
        let mut lineage = Vec::new();
        let mut visited = std::collections::HashSet::new();
        self.collect_lineage(&self.head.node_id, &mut lineage, &mut visited);
        lineage
    }

    fn collect_lineage<'a>(
        &'a self,
        node_id: &str,
        lineage: &mut Vec<&'a ProvenanceNode>,
        visited: &mut std::collections::HashSet<String>,
    ) {
        if visited.contains(node_id) {
            return; // Prevent cycles
        }
        visited.insert(node_id.to_string());

        if let Some(node) = self.nodes.get(node_id) {
            lineage.push(node);
            for parent_hash in &node.parent_hashes {
                let parent_id = hex::encode(parent_hash);
                self.collect_lineage(&parent_id, lineage, visited);
            }
        }
    }

    /// Compute chain confidence from derivation path
    pub fn compute_chain_confidence(&self) -> f64 {
        let lineage = self.get_lineage();
        let mut confidence = 1.0;

        for node in lineage {
            confidence *= node.derivation_type.confidence_decay();
        }

        confidence * self.head.base_confidence
    }
}

/// Result of chain verification
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ChainVerificationResult {
    /// Chain is valid
    Valid,
    /// Chain has errors
    Invalid(Vec<ChainError>),
}

/// Errors in provenance chain
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ChainError {
    /// Node commitment doesn't verify
    InvalidCommitment(String),
    /// Parent reference doesn't exist
    MissingParent {
        /// Node ID with missing parent
        node: String,
        /// Expected parent ID
        parent: String,
    },
    /// Epistemic level exceeds floor without valid reason
    EpistemicViolation {
        /// Node ID violating epistemic constraint
        node: String,
        /// Claimed epistemic level
        claimed: EmpiricalLevel,
        /// Expected epistemic floor
        floor: EmpiricalLevel,
    },
    /// Root node doesn't have Original derivation
    InvalidRoot(String),
}

// ============================================================================
// Builder
// ============================================================================

/// Builder for creating provenance nodes
pub struct ProvenanceBuilder {
    content: Vec<u8>,
    parents: Vec<ProvenanceNode>,
    derivation_type: DerivationType,
    creator_agent_id: String,
    base_confidence: f64,
    epistemic_level: Option<EmpiricalLevel>,
    description: Option<String>,
}

impl ProvenanceBuilder {
    /// Create a new builder for original content
    pub fn original(content: &[u8], agent_id: &str) -> Self {
        Self {
            content: content.to_vec(),
            parents: Vec::new(),
            derivation_type: DerivationType::Original,
            creator_agent_id: agent_id.to_string(),
            base_confidence: 1.0,
            epistemic_level: None,
            description: None,
        }
    }

    /// Create a builder for derived content
    pub fn derived(content: &[u8], agent_id: &str, derivation: DerivationType) -> Self {
        Self {
            content: content.to_vec(),
            parents: Vec::new(),
            derivation_type: derivation,
            creator_agent_id: agent_id.to_string(),
            base_confidence: 1.0,
            epistemic_level: None,
            description: None,
        }
    }

    /// Add a parent node
    pub fn parent(mut self, parent: ProvenanceNode) -> Self {
        self.parents.push(parent);
        self
    }

    /// Add multiple parent nodes
    pub fn parents(mut self, parents: Vec<ProvenanceNode>) -> Self {
        self.parents.extend(parents);
        self
    }

    /// Set base confidence
    pub fn confidence(mut self, confidence: f64) -> Self {
        self.base_confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set epistemic level (will be clamped to floor)
    pub fn epistemic(mut self, level: EmpiricalLevel) -> Self {
        self.epistemic_level = Some(level);
        self
    }

    /// Set description
    pub fn description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Build the provenance node
    pub fn build(self, timestamp: u64) -> ProvenanceNode {
        // Compute content hash
        let content_hash = hash_content(&self.content);

        // Collect parent hashes
        let parent_hashes: Vec<[u8; 32]> = self.parents.iter().map(|p| p.content_hash).collect();

        // Compute epistemic floor
        // For derived nodes: minimum of parent epistemic levels (not floors!)
        // This propagates the actual epistemic constraints through the chain.
        // For originals: will be set to claimed level after level is determined.
        let parent_floor = if self.parents.is_empty() {
            None // Original - will use claimed level
        } else {
            // Take minimum of parent epistemic levels (what they actually claim)
            // This ensures derived knowledge can't exceed what parents actually proved
            Some(
                self.parents
                    .iter()
                    .map(|p| p.epistemic_level)
                    .min_by_key(|e| *e as u8)
                    .unwrap_or(EmpiricalLevel::E0Null),
            )
        };

        // Determine epistemic level (cannot exceed parent floor unless special derivation)
        let epistemic_level = match (self.epistemic_level, parent_floor) {
            // Original node - use claimed level (no parent constraint)
            (Some(level), None) => level,
            (None, None) => EmpiricalLevel::E0Null, // Original with no claim defaults to E0

            // Derived node with explicit claim
            (Some(level), Some(floor)) => {
                if self.derivation_type.can_raise_epistemic() {
                    level // Verification/Aggregation can potentially raise level
                } else {
                    // Clamp to floor
                    if level as u8 > floor as u8 {
                        floor
                    } else {
                        level
                    }
                }
            }

            // Derived node with no claim - default to floor
            (None, Some(floor)) => floor,
        };

        // Set epistemic floor:
        // - For originals: floor equals their own level (they set the constraint)
        // - For derived: floor is the minimum of parent levels
        let epistemic_floor = parent_floor.unwrap_or(epistemic_level);

        // Compute derived confidence
        let parent_confidence = if self.parents.is_empty() {
            1.0
        } else {
            self.parents
                .iter()
                .map(|p| p.derived_confidence)
                .fold(f64::MAX, f64::min) // Minimum parent confidence
        };

        let derived_confidence =
            parent_confidence * self.derivation_type.confidence_decay() * self.base_confidence;

        // Compute chain depth
        let chain_depth = if self.parents.is_empty() {
            0
        } else {
            self.parents
                .iter()
                .map(|p| p.chain_depth)
                .max()
                .unwrap_or(0)
                + 1
        };

        // Create node
        let node_id = hex::encode(content_hash);

        let mut node = ProvenanceNode {
            node_id,
            content_hash,
            parent_hashes,
            derivation_type: self.derivation_type,
            creator_agent_id: self.creator_agent_id,
            timestamp,
            base_confidence: self.base_confidence,
            derived_confidence,
            epistemic_level,
            epistemic_floor,
            chain_depth,
            description: self.description,
            commitment: [0u8; 32], // Will be set below
        };

        // Compute and set commitment
        node.commitment = node.compute_commitment();

        node
    }
}

/// Hash content using SHA3-256
fn hash_content(content: &[u8]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(b"mycelix-provenance-content-v1");
    hasher.update(content);
    hasher.finalize().into()
}

// ============================================================================
// Chain Builder
// ============================================================================

/// Builder for constructing provenance chains
pub struct ChainBuilder {
    nodes: HashMap<String, ProvenanceNode>,
    roots: Vec<String>,
}

impl ChainBuilder {
    /// Create a new chain builder
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            roots: Vec::new(),
        }
    }

    /// Add a node to the chain
    pub fn add_node(mut self, node: ProvenanceNode) -> Self {
        let id = node.node_id.clone();

        if node.derivation_type == DerivationType::Original {
            self.roots.push(id.clone());
        }

        self.nodes.insert(id, node);
        self
    }

    /// Build the chain with the given head node
    pub fn build(self, head: ProvenanceNode) -> ProvenanceChain {
        let mut nodes = self.nodes;
        let head_id = head.node_id.clone();

        // Add head if not already present
        if !nodes.contains_key(&head_id) {
            nodes.insert(head_id.clone(), head.clone());
        }

        // Compute max depth
        let max_depth = nodes.values().map(|n| n.chain_depth).max().unwrap_or(0);

        // Compute chain confidence
        let mut chain = ProvenanceChain {
            head,
            nodes,
            roots: self.roots,
            max_depth,
            chain_confidence: 0.0,
            verified: false,
        };

        chain.chain_confidence = chain.compute_chain_confidence();
        chain.verified = matches!(chain.verify(), ChainVerificationResult::Valid);

        chain
    }
}

impl Default for ChainBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Integration with Agent Outputs
// ============================================================================

/// Extend AgentOutput with provenance tracking
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProvenancedOutput {
    /// The output content (serialized)
    pub content: Vec<u8>,

    /// Provenance node for this output
    pub provenance: ProvenanceNode,

    /// Full provenance chain (if available)
    pub chain: Option<ProvenanceChain>,

    /// Agent output classification
    pub classification: EpistemicClassificationExtended,
}

impl ProvenancedOutput {
    /// Create a new provenanced output
    pub fn new(
        content: Vec<u8>,
        agent_id: &str,
        classification: EpistemicClassificationExtended,
        parents: Vec<ProvenanceNode>,
        derivation: DerivationType,
        confidence: f64,
        timestamp: u64,
    ) -> Self {
        let provenance = if parents.is_empty() {
            ProvenanceBuilder::original(&content, agent_id)
        } else {
            ProvenanceBuilder::derived(&content, agent_id, derivation).parents(parents)
        }
        .confidence(confidence)
        .epistemic(classification.empirical)
        .build(timestamp);

        Self {
            content,
            provenance,
            chain: None,
            classification,
        }
    }

    /// Get effective confidence (min of classification confidence and provenance confidence)
    pub fn effective_confidence(&self) -> f64 {
        self.provenance.derived_confidence
    }

    /// Check if provenance is valid
    pub fn verify_provenance(&self) -> bool {
        self.provenance.verify_commitment()
    }
}

// ============================================================================
// Provenance Registry
// ============================================================================

/// Registry for tracking all provenance nodes
pub struct ProvenanceRegistry {
    /// All nodes indexed by content hash
    nodes: HashMap<String, ProvenanceNode>,

    /// Index by creator agent
    by_agent: HashMap<String, Vec<String>>,

    /// Index by epistemic level
    by_epistemic: HashMap<u8, Vec<String>>,
}

impl ProvenanceRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            by_agent: HashMap::new(),
            by_epistemic: HashMap::new(),
        }
    }

    /// Register a new provenance node
    pub fn register(&mut self, node: ProvenanceNode) -> Result<(), RegistryError> {
        // Verify commitment
        if !node.verify_commitment() {
            return Err(RegistryError::InvalidCommitment);
        }

        // Check parents exist
        for parent_hash in &node.parent_hashes {
            let parent_id = hex::encode(parent_hash);
            if !self.nodes.contains_key(&parent_id) {
                return Err(RegistryError::MissingParent(parent_id));
            }
        }

        let id = node.node_id.clone();

        // Update indices
        self.by_agent
            .entry(node.creator_agent_id.clone())
            .or_default()
            .push(id.clone());

        self.by_epistemic
            .entry(node.epistemic_level as u8)
            .or_default()
            .push(id.clone());

        self.nodes.insert(id, node);

        Ok(())
    }

    /// Get a node by ID
    pub fn get(&self, node_id: &str) -> Option<&ProvenanceNode> {
        self.nodes.get(node_id)
    }

    /// Get all nodes by agent
    pub fn get_by_agent(&self, agent_id: &str) -> Vec<&ProvenanceNode> {
        self.by_agent
            .get(agent_id)
            .map(|ids| ids.iter().filter_map(|id| self.nodes.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get all nodes at epistemic level
    pub fn get_by_epistemic(&self, level: EmpiricalLevel) -> Vec<&ProvenanceNode> {
        self.by_epistemic
            .get(&(level as u8))
            .map(|ids| ids.iter().filter_map(|id| self.nodes.get(id)).collect())
            .unwrap_or_default()
    }

    /// Build a provenance chain for a given node
    pub fn build_chain(&self, head_id: &str) -> Option<ProvenanceChain> {
        let head = self.nodes.get(head_id)?.clone();

        let mut builder = ChainBuilder::new();
        let mut visited = std::collections::HashSet::new();

        self.collect_ancestors(head_id, &mut builder, &mut visited);

        Some(builder.build(head))
    }

    fn collect_ancestors(
        &self,
        node_id: &str,
        builder: &mut ChainBuilder,
        visited: &mut std::collections::HashSet<String>,
    ) {
        if visited.contains(node_id) {
            return;
        }
        visited.insert(node_id.to_string());

        if let Some(node) = self.nodes.get(node_id) {
            // Collect parents first (so roots are added first)
            for parent_hash in &node.parent_hashes {
                let parent_id = hex::encode(parent_hash);
                self.collect_ancestors(&parent_id, builder, visited);
            }

            // Add a clone via a temporary builder
            let temp_builder = std::mem::take(builder);
            *builder = temp_builder.add_node(node.clone());
        }
    }

    /// Get registry statistics
    pub fn stats(&self) -> RegistryStats {
        let mut by_derivation: HashMap<String, usize> = HashMap::new();
        let mut total_confidence = 0.0;
        let mut max_depth = 0u32;

        for node in self.nodes.values() {
            *by_derivation
                .entry(format!("{:?}", node.derivation_type))
                .or_insert(0) += 1;
            total_confidence += node.derived_confidence;
            max_depth = max_depth.max(node.chain_depth);
        }

        RegistryStats {
            total_nodes: self.nodes.len(),
            unique_agents: self.by_agent.len(),
            by_derivation,
            avg_confidence: if self.nodes.is_empty() {
                0.0
            } else {
                total_confidence / self.nodes.len() as f64
            },
            max_chain_depth: max_depth,
        }
    }
}

impl Default for ProvenanceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Registry errors
#[derive(Clone, Debug)]
pub enum RegistryError {
    /// Node commitment doesn't verify
    InvalidCommitment,
    /// Parent node not found
    MissingParent(String),
    /// Node already exists
    AlreadyExists,
}

/// Registry statistics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegistryStats {
    /// Total nodes registered
    pub total_nodes: usize,
    /// Unique agents that created nodes
    pub unique_agents: usize,
    /// Nodes by derivation type
    pub by_derivation: HashMap<String, usize>,
    /// Average derived confidence
    pub avg_confidence: f64,
    /// Maximum chain depth
    pub max_chain_depth: u32,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_original_provenance() {
        let content = b"Original knowledge claim";
        let node = ProvenanceBuilder::original(content, "agent-1")
            .confidence(0.9)
            .epistemic(EmpiricalLevel::E3Cryptographic)
            .description("Test original")
            .build(1000);

        assert_eq!(node.chain_depth, 0);
        assert_eq!(node.derivation_type, DerivationType::Original);
        assert!(node.parent_hashes.is_empty());
        assert!((node.derived_confidence - 0.9).abs() < 0.001);
        assert!(node.verify_commitment());
    }

    #[test]
    fn test_derived_provenance() {
        // Create original
        let original = ProvenanceBuilder::original(b"Source", "agent-1")
            .confidence(1.0)
            .epistemic(EmpiricalLevel::E3Cryptographic)
            .build(1000);

        // Derive from it
        let derived =
            ProvenanceBuilder::derived(b"Derived insight", "agent-2", DerivationType::Synthesis)
                .parent(original.clone())
                .confidence(0.9)
                .epistemic(EmpiricalLevel::E3Cryptographic)
                .build(2000);

        assert_eq!(derived.chain_depth, 1);
        assert_eq!(derived.parent_hashes.len(), 1);

        // Confidence should decay: 1.0 * 0.85 (synthesis) * 0.9 = 0.765
        assert!((derived.derived_confidence - 0.765).abs() < 0.001);

        // Epistemic floor should be parent's floor
        assert_eq!(derived.epistemic_floor, EmpiricalLevel::E3Cryptographic);

        assert!(derived.verify_commitment());
    }

    #[test]
    fn test_epistemic_floor_enforcement() {
        // Create low-epistemic original
        let original = ProvenanceBuilder::original(b"Hearsay", "agent-1")
            .epistemic(EmpiricalLevel::E1Testimonial)
            .build(1000);

        // Try to derive with higher epistemic level
        let derived =
            ProvenanceBuilder::derived(b"Still hearsay", "agent-2", DerivationType::Transform)
                .parent(original.clone())
                .epistemic(EmpiricalLevel::E4PublicRepro) // Try to claim E4
                .build(2000);

        // Should be clamped to parent's level (E1)
        assert_eq!(derived.epistemic_level, EmpiricalLevel::E1Testimonial);
    }

    #[test]
    fn test_verification_can_raise_epistemic() {
        // Create E2 original
        let original = ProvenanceBuilder::original(b"Private verification", "agent-1")
            .epistemic(EmpiricalLevel::E2PrivateVerify)
            .build(1000);

        // Verification can raise to E3
        let verified = ProvenanceBuilder::derived(
            b"Now cryptographically verified",
            "agent-2",
            DerivationType::Verification,
        )
        .parent(original)
        .epistemic(EmpiricalLevel::E3Cryptographic)
        .build(2000);

        // Verification CAN raise epistemic level
        assert_eq!(verified.epistemic_level, EmpiricalLevel::E3Cryptographic);
    }

    #[test]
    fn test_chain_building() {
        let root1 = ProvenanceBuilder::original(b"Root 1", "agent-1")
            .epistemic(EmpiricalLevel::E3Cryptographic)
            .build(1000);

        let root2 = ProvenanceBuilder::original(b"Root 2", "agent-1")
            .epistemic(EmpiricalLevel::E2PrivateVerify)
            .build(1001);

        let synthesis =
            ProvenanceBuilder::derived(b"Synthesis", "agent-2", DerivationType::Synthesis)
                .parents(vec![root1.clone(), root2.clone()])
                .build(2000);

        let chain = ChainBuilder::new()
            .add_node(root1)
            .add_node(root2)
            .build(synthesis);

        assert_eq!(chain.nodes.len(), 3);
        assert_eq!(chain.roots.len(), 2);
        assert_eq!(chain.max_depth, 1);
        assert!(chain.verified);

        // Floor should be minimum of parents (E2)
        assert_eq!(chain.head.epistemic_floor, EmpiricalLevel::E2PrivateVerify);
    }

    #[test]
    fn test_chain_verification() {
        let root = ProvenanceBuilder::original(b"Root", "agent-1")
            .epistemic(EmpiricalLevel::E3Cryptographic)
            .build(1000);

        let derived = ProvenanceBuilder::derived(b"Derived", "agent-2", DerivationType::Citation)
            .parent(root.clone())
            .build(2000);

        let chain = ChainBuilder::new().add_node(root).build(derived);

        assert!(matches!(chain.verify(), ChainVerificationResult::Valid));
    }

    #[test]
    fn test_registry() {
        let mut registry = ProvenanceRegistry::new();

        let root = ProvenanceBuilder::original(b"Root", "agent-1")
            .epistemic(EmpiricalLevel::E3Cryptographic)
            .build(1000);

        registry.register(root.clone()).unwrap();

        let derived = ProvenanceBuilder::derived(b"Derived", "agent-2", DerivationType::Citation)
            .parent(root.clone())
            .build(2000);

        registry.register(derived.clone()).unwrap();

        // Check retrieval
        assert!(registry.get(&root.node_id).is_some());
        assert_eq!(registry.get_by_agent("agent-1").len(), 1);
        assert_eq!(registry.get_by_agent("agent-2").len(), 1);

        // Build chain from registry
        let chain = registry.build_chain(&derived.node_id).unwrap();
        assert_eq!(chain.nodes.len(), 2);

        // Check stats
        let stats = registry.stats();
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.unique_agents, 2);
    }

    #[test]
    fn test_confidence_decay_chain() {
        // Build a 3-level chain with different derivations
        let root = ProvenanceBuilder::original(b"Root", "a1")
            .confidence(1.0)
            .build(1000);

        let level1 = ProvenanceBuilder::derived(b"L1", "a2", DerivationType::Citation)
            .parent(root.clone())
            .confidence(1.0)
            .build(2000);
        // Expected: 1.0 * 0.98 = 0.98

        let level2 = ProvenanceBuilder::derived(b"L2", "a3", DerivationType::Inference)
            .parent(level1.clone())
            .confidence(1.0)
            .build(3000);
        // Expected: 0.98 * 0.80 = 0.784

        assert!((level1.derived_confidence - 0.98).abs() < 0.001);
        assert!((level2.derived_confidence - 0.784).abs() < 0.001);
    }
}
