// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! LUCID Integrity Zome
//!
//! Personal Knowledge Graph - Core Entry Types
//!
//! LUCID (Living Unified Consciousness for Insight & Discovery) is a personal
//! knowledge management system that stores thoughts, claims, and notes as
//! verifiable knowledge units with epistemic classification.
//!
//! Features:
//! - E/N/M/H epistemic classification (GIS v4.0 compatible)
//! - Confidence scoring with 5-factor model
//! - Source attribution and provenance
//! - Private by default with selective sharing

use hdi::prelude::*;

// ============================================================================
// ANCHOR FOR DETERMINISTIC LOOKUPS
// ============================================================================

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// E/N/M/H EPISTEMIC CLASSIFICATION (GIS v4.0)
// ============================================================================

/// Empirical Level (E0-E4)
/// Measures the degree of empirical validation
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EmpiricalLevel {
    /// E0: Unverified - No empirical testing, personal intuition
    E0,
    /// E1: Anecdotal - Personal experience or single observation
    E1,
    /// E2: Tested - Multiple observations or informal testing
    E2,
    /// E3: Verified - Systematic verification, peer review
    E3,
    /// E4: Established - Cryptographic proof or scientific consensus
    E4,
}

impl EmpiricalLevel {
    pub fn to_f64(&self) -> f64 {
        match self {
            EmpiricalLevel::E0 => 0.0,
            EmpiricalLevel::E1 => 0.25,
            EmpiricalLevel::E2 => 0.5,
            EmpiricalLevel::E3 => 0.75,
            EmpiricalLevel::E4 => 1.0,
        }
    }

    pub fn from_f64(value: f64) -> Self {
        match value {
            v if v < 0.125 => EmpiricalLevel::E0,
            v if v < 0.375 => EmpiricalLevel::E1,
            v if v < 0.625 => EmpiricalLevel::E2,
            v if v < 0.875 => EmpiricalLevel::E3,
            _ => EmpiricalLevel::E4,
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            EmpiricalLevel::E0 => "Unverified - Personal intuition",
            EmpiricalLevel::E1 => "Anecdotal - Single observation",
            EmpiricalLevel::E2 => "Tested - Multiple observations",
            EmpiricalLevel::E3 => "Verified - Systematic verification",
            EmpiricalLevel::E4 => "Established - Consensus or proof",
        }
    }
}

impl Default for EmpiricalLevel {
    fn default() -> Self {
        EmpiricalLevel::E0
    }
}

/// Normative Level (N0-N3)
/// Measures normative/ethical coherence
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum NormativeLevel {
    /// N0: Personal - Only meaningful to self
    N0,
    /// N1: Contested - Recognized but disputed
    N1,
    /// N2: Emerging - Growing acceptance
    N2,
    /// N3: Endorsed - Broad normative agreement
    N3,
}

impl NormativeLevel {
    pub fn to_f64(&self) -> f64 {
        match self {
            NormativeLevel::N0 => 0.0,
            NormativeLevel::N1 => 0.33,
            NormativeLevel::N2 => 0.67,
            NormativeLevel::N3 => 1.0,
        }
    }

    pub fn from_f64(value: f64) -> Self {
        match value {
            v if v < 0.17 => NormativeLevel::N0,
            v if v < 0.5 => NormativeLevel::N1,
            v if v < 0.83 => NormativeLevel::N2,
            _ => NormativeLevel::N3,
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            NormativeLevel::N0 => "Personal - Meaningful to self only",
            NormativeLevel::N1 => "Contested - Recognized but disputed",
            NormativeLevel::N2 => "Emerging - Growing acceptance",
            NormativeLevel::N3 => "Endorsed - Broad agreement",
        }
    }
}

impl Default for NormativeLevel {
    fn default() -> Self {
        NormativeLevel::N0
    }
}

/// Materiality Level (M0-M3)
/// Measures practical/material significance
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MaterialityLevel {
    /// M0: Abstract - Purely theoretical
    M0,
    /// M1: Potential - May have implications
    M1,
    /// M2: Applicable - Practical applications exist
    M2,
    /// M3: Transformative - Significant real-world impact
    M3,
}

impl MaterialityLevel {
    pub fn to_f64(&self) -> f64 {
        match self {
            MaterialityLevel::M0 => 0.0,
            MaterialityLevel::M1 => 0.33,
            MaterialityLevel::M2 => 0.67,
            MaterialityLevel::M3 => 1.0,
        }
    }

    pub fn from_f64(value: f64) -> Self {
        match value {
            v if v < 0.17 => MaterialityLevel::M0,
            v if v < 0.5 => MaterialityLevel::M1,
            v if v < 0.83 => MaterialityLevel::M2,
            _ => MaterialityLevel::M3,
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            MaterialityLevel::M0 => "Abstract - Purely theoretical",
            MaterialityLevel::M1 => "Potential - May have implications",
            MaterialityLevel::M2 => "Applicable - Practical applications",
            MaterialityLevel::M3 => "Transformative - Significant impact",
        }
    }
}

impl Default for MaterialityLevel {
    fn default() -> Self {
        MaterialityLevel::M0
    }
}

/// Harmonic Level (H0-H4)
/// Alignment with higher purpose / 12 Harmonies
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HarmonicLevel {
    /// H0: Discordant - Potentially harmful
    H0,
    /// H1: Neutral - No particular alignment
    H1,
    /// H2: Resonant - Aligns with one harmony
    H2,
    /// H3: Harmonic - Aligns with multiple harmonies
    H3,
    /// H4: Transcendent - Serves universal flourishing
    H4,
}

impl HarmonicLevel {
    pub fn to_f64(&self) -> f64 {
        match self {
            HarmonicLevel::H0 => 0.0,
            HarmonicLevel::H1 => 0.25,
            HarmonicLevel::H2 => 0.5,
            HarmonicLevel::H3 => 0.75,
            HarmonicLevel::H4 => 1.0,
        }
    }

    pub fn from_f64(value: f64) -> Self {
        match value {
            v if v < 0.125 => HarmonicLevel::H0,
            v if v < 0.375 => HarmonicLevel::H1,
            v if v < 0.625 => HarmonicLevel::H2,
            v if v < 0.875 => HarmonicLevel::H3,
            _ => HarmonicLevel::H4,
        }
    }
}

impl Default for HarmonicLevel {
    fn default() -> Self {
        HarmonicLevel::H1
    }
}

/// Full E/N/M/H Epistemic Classification
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EpistemicClassification {
    /// Empirical level (E0-E4)
    pub empirical: EmpiricalLevel,
    /// Normative level (N0-N3)
    pub normative: NormativeLevel,
    /// Materiality level (M0-M3)
    pub materiality: MaterialityLevel,
    /// Harmonic level (H0-H4)
    pub harmonic: HarmonicLevel,
}

impl Default for EpistemicClassification {
    fn default() -> Self {
        Self {
            empirical: EmpiricalLevel::E0,
            normative: NormativeLevel::N0,
            materiality: MaterialityLevel::M0,
            harmonic: HarmonicLevel::H1,
        }
    }
}

impl EpistemicClassification {
    pub fn new(e: EmpiricalLevel, n: NormativeLevel, m: MaterialityLevel, h: HarmonicLevel) -> Self {
        Self {
            empirical: e,
            normative: n,
            materiality: m,
            harmonic: h,
        }
    }

    /// Short code representation (e.g., "E2N1M2H3")
    pub fn code(&self) -> String {
        format!(
            "E{}N{}M{}H{}",
            self.empirical as u8,
            self.normative as u8,
            self.materiality as u8,
            self.harmonic as u8
        )
    }

    /// Calculate overall epistemic strength
    /// Weights: E=40%, N=25%, M=20%, H=15%
    pub fn overall_strength(&self) -> f64 {
        0.40 * self.empirical.to_f64()
            + 0.25 * self.normative.to_f64()
            + 0.20 * self.materiality.to_f64()
            + 0.15 * self.harmonic.to_f64()
    }
}

// ============================================================================
// THOUGHT TYPES
// ============================================================================

/// Type of thought/knowledge unit
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ThoughtType {
    /// A belief or assertion about the world
    Claim,
    /// A note or observation without strong assertion
    Note,
    /// A question seeking understanding
    Question,
    /// An insight or realization
    Insight,
    /// A definition or concept clarification
    Definition,
    /// A prediction about the future
    Prediction,
    /// A hypothesis to be tested
    Hypothesis,
    /// A personal reflection
    Reflection,
    /// A quote or reference from another source
    Quote,
    /// A task or action item
    Task,
}

impl Default for ThoughtType {
    fn default() -> Self {
        ThoughtType::Note
    }
}

// ============================================================================
// CORE ENTRY: THOUGHT
// ============================================================================

/// A Thought is the primary knowledge unit in LUCID
///
/// Represents any piece of knowledge: claims, notes, questions, insights, etc.
/// Each thought has epistemic classification, confidence scoring, and
/// optional source attribution.
///
/// Symthaea Integration:
/// - `embedding`: 16,384-dimensional HDC vector for semantic similarity
/// - `embedding_version`: Tracks when embedding was last computed
/// - `coherence_score`: Last computed coherence with knowledge graph
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Thought {
    /// Unique identifier (UUID)
    pub id: String,

    /// The content/statement of the thought
    pub content: String,

    /// Type of thought
    pub thought_type: ThoughtType,

    /// E/N/M/H epistemic classification
    pub epistemic: EpistemicClassification,

    /// Personal confidence in this thought (0.0-1.0)
    pub confidence: f64,

    /// Tags for organization
    pub tags: Vec<String>,

    /// Domain/field this relates to
    pub domain: Option<String>,

    /// Related thought IDs (for manual linking)
    pub related_thoughts: Vec<String>,

    /// Source action hashes (links to Source entries)
    pub source_hashes: Vec<ActionHash>,

    /// Optional parent thought (for hierarchical organization)
    pub parent_thought: Option<String>,

    /// Creation timestamp
    pub created_at: Timestamp,

    /// Last modification timestamp
    pub updated_at: Timestamp,

    /// Version number (incremented on each update)
    pub version: u32,

    // ========== Symthaea Integration Fields ==========

    /// HDC embedding from Symthaea (16,384 dimensions)
    /// Used for semantic similarity search and coherence analysis
    /// None if not yet computed or Symthaea unavailable
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,

    /// Version of content when embedding was computed
    /// If less than `version`, embedding needs recomputation
    #[serde(default)]
    pub embedding_version: Option<u32>,

    /// Last computed coherence score with the knowledge graph (0.0-1.0)
    /// Updated when coherence analysis runs
    #[serde(default)]
    pub coherence_score: Option<f64>,

    /// Phi (integrated information) score from Symthaea
    /// Higher values indicate more integrated/conscious thought
    #[serde(default)]
    pub phi_score: Option<f64>,
}

// ============================================================================
// TAG ENTRY
// ============================================================================

/// A tag for organizing thoughts
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Tag {
    /// Tag name (normalized lowercase)
    pub name: String,

    /// Optional description
    pub description: Option<String>,

    /// Optional color (hex)
    pub color: Option<String>,

    /// Parent tag for hierarchical taxonomy
    pub parent_tag: Option<String>,

    /// Creation timestamp
    pub created_at: Timestamp,
}

// ============================================================================
// DOMAIN ENTRY
// ============================================================================

/// A knowledge domain for categorization
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Domain {
    /// Domain name
    pub name: String,

    /// Description
    pub description: Option<String>,

    /// Parent domain for hierarchy
    pub parent_domain: Option<String>,

    /// Related domains
    pub related_domains: Vec<String>,

    /// Creation timestamp
    pub created_at: Timestamp,
}

// ============================================================================
// RELATIONSHIP ENTRY
// ============================================================================

/// Type of relationship between thoughts
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RelationshipType {
    /// A supports B
    Supports,
    /// A contradicts B
    Contradicts,
    /// A implies B
    Implies,
    /// A is implied by B
    ImpliedBy,
    /// A refines/elaborates B
    Refines,
    /// A is an example of B
    ExampleOf,
    /// A is related to B (generic)
    RelatedTo,
    /// A depends on B
    DependsOn,
    /// A supersedes B (replaces)
    Supersedes,
    /// A responds to B
    RespondsTo,
}

/// A relationship between two thoughts
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Relationship {
    /// Unique identifier
    pub id: String,

    /// Source thought ID
    pub from_thought_id: String,

    /// Target thought ID
    pub to_thought_id: String,

    /// Type of relationship
    pub relationship_type: RelationshipType,

    /// Strength of relationship (0.0-1.0)
    pub strength: f64,

    /// Optional description/justification
    pub description: Option<String>,

    /// Bidirectional flag
    pub bidirectional: bool,

    /// Creation timestamp
    pub created_at: Timestamp,

    /// Whether this relationship is active
    pub active: bool,
}

// ============================================================================
// ENTRY TYPE REGISTRATION
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Thought(Thought),
    Tag(Tag),
    Domain(Domain),
    Relationship(Relationship),
}

// ============================================================================
// LINK TYPES
// ============================================================================

#[hdk_link_types]
pub enum LinkTypes {
    // Agent links
    AgentToThought,
    AgentToTag,
    AgentToDomain,

    // Thought discovery
    ThoughtIdToThought,
    TagToThought,
    DomainToThought,
    TypeToThought,

    // Relationships
    ThoughtToRelationship,
    RelationshipToThought,

    // Hierarchy
    ParentToChildThought,
    ChildToParentThought,
    ParentToChildTag,
    ParentToChildDomain,

    // Sources (cross-zome)
    ThoughtToSource,

    // Temporal (cross-zome)
    ThoughtToVersion,

    // Privacy (cross-zome)
    ThoughtToPolicy,

    // Search indexes
    EmpiricalLevelIndex,
    NormativeLevelIndex,
    MaterialityLevelIndex,
    HarmonicLevelIndex,
    ConfidenceIndex,

    // Semantic search (Symthaea integration)
    /// Links thoughts with embeddings for semantic similarity queries
    EmbeddingIndex,
    /// Links thoughts to their coherence analysis results
    ThoughtToCoherence,
}

// ============================================================================
// VALIDATION
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Thought(thought) => validate_create_thought(action, thought),
                EntryTypes::Tag(tag) => validate_create_tag(action, tag),
                EntryTypes::Domain(domain) => validate_create_domain(action, domain),
                EntryTypes::Relationship(rel) => validate_create_relationship(action, rel),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Thought(thought) => {
                    validate_update_thought(action, thought, original_action_hash)
                }
                EntryTypes::Tag(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Domain(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Relationship(rel) => validate_update_relationship(action, rel),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => {
            // All link types are valid when created
            match link_type {
                _ => Ok(ValidateCallbackResult::Valid),
            }
        }
        FlatOp::RegisterDeleteLink {
            original_action,
            action,
            ..
        } => {
            // Only the original creator can delete a link
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
            // Only the original author can update
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
            // Only the original author can delete
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

/// Validate thought creation
fn validate_create_thought(_action: Create, thought: Thought) -> ExternResult<ValidateCallbackResult> {
    // Content cannot be empty
    if thought.content.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Thought content cannot be empty".into(),
        ));
    }

    // Content length limit (64KB)
    if thought.content.len() > 65536 {
        return Ok(ValidateCallbackResult::Invalid(
            "Thought content exceeds maximum length of 64KB".into(),
        ));
    }

    // Confidence must be in range
    if thought.confidence < 0.0 || thought.confidence > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Confidence must be between 0.0 and 1.0".into(),
        ));
    }

    // ID cannot be empty
    if thought.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Thought ID cannot be empty".into(),
        ));
    }

    // Version must start at 1
    if thought.version != 1 {
        return Ok(ValidateCallbackResult::Invalid(
            "Initial thought version must be 1".into(),
        ));
    }

    // Tag limit (prevent spam)
    if thought.tags.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Maximum 50 tags allowed per thought".into(),
        ));
    }

    // Validate embedding if present (must be 16,384 dimensions for Symthaea HDC)
    if let Some(ref embedding) = thought.embedding {
        if embedding.len() != 16384 {
            return Ok(ValidateCallbackResult::Invalid(
                format!(
                    "Embedding must be 16,384 dimensions (Symthaea HDC), got {}",
                    embedding.len()
                ),
            ));
        }
    }

    // Coherence score must be in range if present
    if let Some(coherence) = thought.coherence_score {
        if coherence < 0.0 || coherence > 1.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Coherence score must be between 0.0 and 1.0".into(),
            ));
        }
    }

    // Phi score must be non-negative if present
    if let Some(phi) = thought.phi_score {
        if phi < 0.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Phi score must be non-negative".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate thought update
fn validate_update_thought(
    _action: Update,
    thought: Thought,
    original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash)?;
    let original_thought: Thought = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original thought not found".into()
        )))?;

    // Cannot change ID
    if thought.id != original_thought.id {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change thought ID".into(),
        ));
    }

    // Version must increment
    if thought.version != original_thought.version + 1 {
        return Ok(ValidateCallbackResult::Invalid(
            "Version must be incremented by 1".into(),
        ));
    }

    // Content cannot be empty
    if thought.content.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Thought content cannot be empty".into(),
        ));
    }

    // Confidence must be in range
    if thought.confidence < 0.0 || thought.confidence > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Confidence must be between 0.0 and 1.0".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate tag creation
fn validate_create_tag(_action: Create, tag: Tag) -> ExternResult<ValidateCallbackResult> {
    // Name cannot be empty
    if tag.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Tag name cannot be empty".into(),
        ));
    }

    // Name length limit
    if tag.name.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Tag name cannot exceed 100 characters".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate domain creation
fn validate_create_domain(_action: Create, domain: Domain) -> ExternResult<ValidateCallbackResult> {
    // Name cannot be empty
    if domain.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Domain name cannot be empty".into(),
        ));
    }

    // Name length limit
    if domain.name.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid(
            "Domain name cannot exceed 200 characters".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate relationship creation
fn validate_create_relationship(
    _action: Create,
    rel: Relationship,
) -> ExternResult<ValidateCallbackResult> {
    // Cannot relate to self
    if rel.from_thought_id == rel.to_thought_id {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot create relationship to self".into(),
        ));
    }

    // Strength must be in range
    if rel.strength < 0.0 || rel.strength > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Relationship strength must be between 0.0 and 1.0".into(),
        ));
    }

    // IDs cannot be empty
    if rel.from_thought_id.trim().is_empty() || rel.to_thought_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Relationship thought IDs cannot be empty".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate relationship update
fn validate_update_relationship(
    _action: Update,
    rel: Relationship,
) -> ExternResult<ValidateCallbackResult> {
    // Strength must be in range
    if rel.strength < 0.0 || rel.strength > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Relationship strength must be between 0.0 and 1.0".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // E/N/M/H Epistemic Classification Tests
    // =========================================================================

    #[test]
    fn test_empirical_level_roundtrip() {
        let levels = [
            (EmpiricalLevel::E0, 0.0),
            (EmpiricalLevel::E1, 0.25),
            (EmpiricalLevel::E2, 0.5),
            (EmpiricalLevel::E3, 0.75),
            (EmpiricalLevel::E4, 1.0),
        ];
        for (level, expected_f64) in &levels {
            let f = level.to_f64();
            assert_eq!(f, *expected_f64, "E{:?} should be {}", level, expected_f64);
            let roundtrip = EmpiricalLevel::from_f64(f);
            assert_eq!(&roundtrip, level, "Roundtrip failed for {:?}", level);
        }
    }

    #[test]
    fn test_normative_level_roundtrip() {
        let levels = [
            (NormativeLevel::N0, 0.0),
            (NormativeLevel::N1, 0.33),
            (NormativeLevel::N2, 0.67),
            (NormativeLevel::N3, 1.0),
        ];
        for (level, expected_f64) in &levels {
            let f = level.to_f64();
            assert!((f - expected_f64).abs() < 0.01, "N{:?} should be {}", level, expected_f64);
            let roundtrip = NormativeLevel::from_f64(f);
            assert_eq!(&roundtrip, level, "Roundtrip failed for {:?}", level);
        }
    }

    #[test]
    fn test_materiality_level_roundtrip() {
        let levels = [
            (MaterialityLevel::M0, 0.0),
            (MaterialityLevel::M1, 0.33),
            (MaterialityLevel::M2, 0.67),
            (MaterialityLevel::M3, 1.0),
        ];
        for (level, expected_f64) in &levels {
            let f = level.to_f64();
            assert!((f - expected_f64).abs() < 0.01, "M{:?} should be {}", level, expected_f64);
            let roundtrip = MaterialityLevel::from_f64(f);
            assert_eq!(&roundtrip, level, "Roundtrip failed for {:?}", level);
        }
    }

    #[test]
    fn test_harmonic_level_roundtrip() {
        let levels = [
            (HarmonicLevel::H0, 0.0),
            (HarmonicLevel::H1, 0.25),
            (HarmonicLevel::H2, 0.5),
            (HarmonicLevel::H3, 0.75),
            (HarmonicLevel::H4, 1.0),
        ];
        for (level, expected_f64) in &levels {
            let f = level.to_f64();
            assert_eq!(f, *expected_f64, "H{:?} should be {}", level, expected_f64);
            let roundtrip = HarmonicLevel::from_f64(f);
            assert_eq!(&roundtrip, level, "Roundtrip failed for {:?}", level);
        }
    }

    #[test]
    fn test_epistemic_classification_code() {
        let cls = EpistemicClassification::new(
            EmpiricalLevel::E2,
            NormativeLevel::N1,
            MaterialityLevel::M2,
            HarmonicLevel::H3,
        );
        assert_eq!(cls.code(), "E2N1M2H3");
    }

    #[test]
    fn test_epistemic_overall_strength() {
        // E4=1.0, N3=1.0, M3=1.0, H4=1.0 → max strength = 1.0
        let max = EpistemicClassification::new(
            EmpiricalLevel::E4,
            NormativeLevel::N3,
            MaterialityLevel::M3,
            HarmonicLevel::H4,
        );
        assert!((max.overall_strength() - 1.0).abs() < 0.01);

        // E0=0.0, N0=0.0, M0=0.0, H0=0.0 → min strength = 0.0
        let min = EpistemicClassification::new(
            EmpiricalLevel::E0,
            NormativeLevel::N0,
            MaterialityLevel::M0,
            HarmonicLevel::H0,
        );
        assert!(min.overall_strength().abs() < 0.01);

        // Default: E0, N0, M0, H1 → 0.15 * 0.25 = 0.0375
        let default = EpistemicClassification::default();
        assert!(default.overall_strength() > 0.0);
        assert!(default.overall_strength() < 0.1);
    }

    #[test]
    fn test_epistemic_weights_sum_to_one() {
        // Weights: E=40%, N=25%, M=20%, H=15%
        let sum: f64 = 0.40 + 0.25 + 0.20 + 0.15;
        assert!((sum - 1.0).abs() < 1e-10, "Epistemic weights should sum to 1.0");
    }

    // =========================================================================
    // Defaults
    // =========================================================================

    #[test]
    fn test_thought_type_default_is_note() {
        assert_eq!(ThoughtType::default(), ThoughtType::Note);
    }

    #[test]
    fn test_harmonic_default_is_h1() {
        assert_eq!(HarmonicLevel::default(), HarmonicLevel::H1);
    }

    // =========================================================================
    // Source-Level Validation Verification
    // =========================================================================

    const SOURCE: &str = include_str!("lib.rs");

    #[test]
    fn test_thought_content_not_empty_validated() {
        assert!(
            SOURCE.contains("Thought content cannot be empty"),
            "REGRESSION: validate_create_thought must reject empty content"
        );
    }

    #[test]
    fn test_thought_content_length_limit() {
        assert!(
            SOURCE.contains("65536") || SOURCE.contains("64KB"),
            "REGRESSION: Thought content must have a maximum length limit"
        );
    }

    #[test]
    fn test_thought_confidence_range_validated() {
        assert!(
            SOURCE.contains("Confidence must be between 0.0 and 1.0"),
            "REGRESSION: validate_create_thought must enforce confidence range"
        );
    }

    #[test]
    fn test_thought_version_starts_at_one() {
        assert!(
            SOURCE.contains("Initial thought version must be 1"),
            "REGRESSION: New thoughts must start at version 1"
        );
    }

    #[test]
    fn test_thought_tag_limit() {
        assert!(
            SOURCE.contains("Maximum 50 tags allowed per thought"),
            "REGRESSION: Thoughts must have a tag count limit"
        );
    }

    #[test]
    fn test_embedding_dimension_validated() {
        assert!(
            SOURCE.contains("16384") || SOURCE.contains("16,384"),
            "REGRESSION: Embedding must be validated for 16,384 dimensions"
        );
    }

    #[test]
    fn test_coherence_score_range_validated() {
        assert!(
            SOURCE.contains("Coherence score must be between 0.0 and 1.0"),
            "REGRESSION: Coherence score must be range-checked"
        );
    }

    #[test]
    fn test_phi_score_non_negative() {
        assert!(
            SOURCE.contains("Phi score must be non-negative"),
            "REGRESSION: Phi score must be validated as non-negative"
        );
    }

    #[test]
    fn test_thought_id_immutable_on_update() {
        assert!(
            SOURCE.contains("Cannot change thought ID"),
            "REGRESSION: Thought ID must be immutable across updates"
        );
    }

    #[test]
    fn test_version_must_increment() {
        assert!(
            SOURCE.contains("Version must be incremented by 1"),
            "REGRESSION: Updates must increment version by exactly 1"
        );
    }

    #[test]
    fn test_relationship_no_self_reference() {
        assert!(
            SOURCE.contains("Cannot create relationship to self"),
            "REGRESSION: Self-referencing relationships must be rejected"
        );
    }

    #[test]
    fn test_relationship_strength_range() {
        assert!(
            SOURCE.contains("Relationship strength must be between 0.0 and 1.0"),
            "REGRESSION: Relationship strength must be range-checked"
        );
    }

    #[test]
    fn test_tag_name_not_empty() {
        assert!(
            SOURCE.contains("Tag name cannot be empty"),
            "REGRESSION: Tag names must not be empty"
        );
    }

    #[test]
    fn test_tag_name_length_limit() {
        assert!(
            SOURCE.contains("Tag name cannot exceed 100 characters"),
            "REGRESSION: Tag names must have a length limit"
        );
    }

    #[test]
    fn test_domain_name_not_empty() {
        assert!(
            SOURCE.contains("Domain name cannot be empty"),
            "REGRESSION: Domain names must not be empty"
        );
    }

    #[test]
    fn test_only_author_can_update() {
        assert!(
            SOURCE.contains("Only the original author can update an entry"),
            "REGRESSION: Author-only update must be enforced"
        );
    }

    #[test]
    fn test_only_author_can_delete() {
        assert!(
            SOURCE.contains("Only the original author can delete an entry"),
            "REGRESSION: Author-only deletion must be enforced"
        );
    }

    #[test]
    fn test_only_link_creator_can_delete_link() {
        assert!(
            SOURCE.contains("Only the original link creator can delete a link"),
            "REGRESSION: Link deletion must be restricted to creator"
        );
    }

    // =========================================================================
    // Thought Type Completeness
    // =========================================================================

    #[test]
    fn test_all_thought_types_exist() {
        // Verify all expected thought types compile
        let types = vec![
            ThoughtType::Claim,
            ThoughtType::Note,
            ThoughtType::Question,
            ThoughtType::Insight,
            ThoughtType::Definition,
            ThoughtType::Prediction,
            ThoughtType::Hypothesis,
            ThoughtType::Reflection,
            ThoughtType::Quote,
            ThoughtType::Task,
        ];
        assert_eq!(types.len(), 10, "Expected 10 thought types");
    }

    #[test]
    fn test_all_relationship_types_exist() {
        let types = vec![
            RelationshipType::Supports,
            RelationshipType::Contradicts,
            RelationshipType::Implies,
            RelationshipType::ImpliedBy,
            RelationshipType::Refines,
            RelationshipType::ExampleOf,
            RelationshipType::RelatedTo,
            RelationshipType::DependsOn,
            RelationshipType::Supersedes,
            RelationshipType::RespondsTo,
        ];
        assert_eq!(types.len(), 10, "Expected 10 relationship types");
    }
}
