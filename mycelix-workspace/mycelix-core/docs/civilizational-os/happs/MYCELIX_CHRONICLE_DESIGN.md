# Mycelix-Chronicle: Epistemic Archive

## Vision & Design Document

**Version**: 1.0.0
**Created**: December 30, 2025
**Status**: Design Phase
**Priority**: Tier 1 - Research/Experimental

---

## Executive Summary

Mycelix-Chronicle is the long-term memory of the Mycelix civilization. It provides permanent, epistemically-classified archival of important claims, decisions, precedents, and knowledge across the ecosystem. Chronicle ensures that foundational truths (M3-Foundational) are preserved forever, while ephemeral data (M0-M1) is gracefully pruned.

### Why Chronicle?

Civilizations need memory:
- **Legal precedents**: Arbiter decisions guide future cases
- **Governance history**: Agora votes inform future decisions
- **Scientific knowledge**: Research findings accumulate over time
- **Cultural artifacts**: Stories and traditions define identity
- **Trust history**: Reputation has context and history

Without Chronicle, the ecosystem has no institutional memory.

---

## Core Principles

### 1. Epistemic Classification
Every archived item is classified on the Empirical-Normative-Materiality axes.

### 2. Durability Guarantees
M3-Foundational content is preserved permanently across multiple storage layers.

### 3. Provenance Tracking
Every claim's origin, transformations, and citations are tracked.

### 4. Graceful Decay
Lower-materiality content is pruned according to clear policies.

### 5. Search & Discovery
Archives are useless if they can't be found.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Content Sources                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Arbiter  │ │  Agora   │ │ Praxis   │ │   All    │ │ External │  │
│  │(Rulings) │ │ (Votes)  │ │(Research)│ │  hApps   │ │ (IPFS)   │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                       Chronicle hApp                                 │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    Coordinator Zomes                             ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           ││
│  │  │ Archive  │ │ Classify │ │ Citation │ │  Search  │           ││
│  │  │ Manager  │ │ Engine   │ │ Tracker  │ │  Index   │           ││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           ││
│  │  │ Prune    │ │ Replicate│ │Provenance│ │  Access  │           ││
│  │  │ Manager  │ │ Coordinator│ │ Chain  │ │ Control  │           ││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           ││
│  └─────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│                      Storage Layers                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │ Holochain    │  │    IPFS      │  │      Arweave             │   │
│  │ DHT          │  │ (Redundant)  │  │ (Permanent)              │   │
│  │ (Primary)    │  │              │  │                          │   │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Model

### Core Entry Types

```rust
/// An archived item
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ArchiveEntry {
    /// Unique archive ID
    pub archive_id: String,
    /// Content type
    pub content_type: ArchiveContentType,
    /// The actual content (or hash if large)
    pub content: ArchiveContent,
    /// Full epistemic classification
    pub epistemic: EpistemicClassification,
    /// Source information
    pub provenance: Provenance,
    /// Archiver
    pub archived_by: AgentPubKey,
    /// Archived at
    pub archived_at: Timestamp,
    /// Storage locations
    pub storage: Vec<StorageLocation>,
    /// Access control
    pub access: AccessPolicy,
    /// Related archives
    pub relations: Vec<ArchiveRelation>,
    /// Tags for discovery
    pub tags: Vec<String>,
    /// Citations to this archive
    pub citation_count: u64,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum ArchiveContentType {
    /// Legal precedent/ruling
    LegalPrecedent {
        case_type: String,
        jurisdiction: String,
    },
    /// Governance decision
    GovernanceDecision {
        space: String,
        proposal_type: String,
    },
    /// Research finding
    ResearchFinding {
        field: String,
        methodology: String,
    },
    /// Credential definition
    CredentialDefinition {
        credential_type: String,
    },
    /// Protocol specification
    ProtocolSpec {
        version: String,
    },
    /// Constitutional document
    Constitution {
        scope: String,
    },
    /// Historical event
    HistoricalEvent {
        event_type: String,
        date: String,
    },
    /// Cultural artifact
    CulturalArtifact {
        culture: String,
        medium: String,
    },
    /// Threat intelligence
    ThreatIntel {
        threat_type: String,
    },
    /// Aggregate statistics
    Statistics {
        domain: String,
        period: String,
    },
    /// Other
    Other(String),
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum ArchiveContent {
    /// Inline text content
    Text(String),
    /// Inline structured data
    Structured(String), // JSON
    /// Reference to Holochain entry
    HolochainRef(ActionHash),
    /// Reference to external storage
    ExternalRef {
        storage_type: String,
        identifier: String,
        hash: String,
    },
    /// Multi-part content
    MultiPart(Vec<ArchiveContent>),
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct EpistemicClassification {
    /// Empirical level (how verified)
    pub empirical: EmpiricalLevel,
    /// Normative level (who agrees)
    pub normative: NormativeLevel,
    /// Materiality level (how long matters)
    pub materiality: MaterialityLevel,
    /// Confidence in classification
    pub confidence: f64,
    /// Who classified
    pub classified_by: ClassificationAuthority,
    /// Classified at
    pub classified_at: Timestamp,
    /// Classification evidence
    pub evidence: Vec<ClassificationEvidence>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum ClassificationAuthority {
    /// System/algorithm classified
    System { algorithm: String },
    /// Expert classified
    Expert { credential: ActionHash },
    /// Community classified
    Community { votes: u64 },
    /// Original source classified
    Source,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassificationEvidence {
    pub evidence_type: String,
    pub description: String,
    pub reference: Option<String>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// Original source
    pub original_source: SourceInfo,
    /// Transformation chain
    pub transformations: Vec<Transformation>,
    /// Verification status
    pub verification: VerificationStatus,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceInfo {
    /// Source type
    pub source_type: SourceType,
    /// Source identifier
    pub identifier: String,
    /// Original timestamp
    pub original_timestamp: Timestamp,
    /// Original authors/creators
    pub authors: Vec<AgentPubKey>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum SourceType {
    /// Another hApp
    HApp { happ: String, entry_type: String },
    /// External system
    External { system: String },
    /// User submission
    UserSubmission,
    /// Aggregated from multiple sources
    Aggregated,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct Transformation {
    /// What transformation occurred
    pub transformation_type: TransformationType,
    /// Who performed it
    pub performed_by: AgentPubKey,
    /// When
    pub performed_at: Timestamp,
    /// Description
    pub description: String,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum TransformationType {
    Created,
    Updated,
    Summarized,
    Translated { from: String, to: String },
    Verified,
    Corrected,
    Annotated,
    Linked,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct StorageLocation {
    pub storage_type: StorageType,
    pub identifier: String,
    pub verified: bool,
    pub last_verified: Timestamp,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum StorageType {
    HolochainDHT,
    IPFS,
    Arweave,
    Filecoin,
    Custom(String),
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct AccessPolicy {
    /// Default access level
    pub default: AccessLevel,
    /// Specific overrides
    pub overrides: Vec<AccessOverride>,
    /// Encryption key reference (if encrypted)
    pub encryption: Option<EncryptionInfo>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum AccessLevel {
    Public,
    Authenticated,
    CommunityMembers { community: String },
    CredentialHolders { credential_type: String },
    SpecificAgents { agents: Vec<AgentPubKey> },
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchiveRelation {
    pub relation_type: RelationType,
    pub target: ActionHash,
    pub description: Option<String>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum RelationType {
    /// This supersedes target
    Supersedes,
    /// This is superseded by target
    SupersededBy,
    /// This cites target
    Cites,
    /// This is cited by target
    CitedBy,
    /// This is related to target
    RelatedTo,
    /// This is part of target
    PartOf,
    /// This contains target
    Contains,
    /// This contradicts target
    Contradicts,
    /// This confirms target
    Confirms,
}

/// A citation to an archive entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Citation {
    /// What is citing
    pub source: CitationSource,
    /// What is being cited
    pub target: ActionHash,
    /// Citation type
    pub citation_type: CitationType,
    /// Context of citation
    pub context: String,
    /// Created by
    pub created_by: AgentPubKey,
    /// Created at
    pub created_at: Timestamp,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum CitationSource {
    /// Another archive entry
    Archive(ActionHash),
    /// hApp entry
    HAppEntry { happ: String, entry: ActionHash },
    /// External
    External { system: String, identifier: String },
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum CitationType {
    /// Uses as authority
    Authority,
    /// Uses as evidence
    Evidence,
    /// References for context
    Reference,
    /// Builds upon
    Extension,
    /// Critiques
    Critique,
}

/// Prune policy for a content type
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PrunePolicy {
    /// Policy ID
    pub policy_id: String,
    /// What content type this applies to
    pub content_type: Option<ArchiveContentType>,
    /// What materiality level
    pub materiality: MaterialityLevel,
    /// Retention period (if not permanent)
    pub retention_days: Option<u64>,
    /// Conditions that prevent pruning
    pub retain_if: Vec<RetentionCondition>,
    /// Created by (governance)
    pub created_by: AgentPubKey,
    /// Active
    pub active: bool,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum RetentionCondition {
    /// Has citations
    HasCitations { min_count: u64 },
    /// Referenced by active content
    ActiveReferences,
    /// High access frequency
    HighAccess { min_per_month: u64 },
    /// Important source
    ImportantSource { min_trust: f64 },
    /// Community flag
    CommunityFlagged,
}
```

---

## Materiality-Based Retention

### M0: Ephemeral
- **Retention**: Immediate (not archived)
- **Examples**: Chat messages, temporary queries
- **Policy**: Never enters Chronicle

### M1: Temporal
- **Retention**: Until state change or 30-90 days
- **Examples**: Transaction details, session data
- **Policy**: Archive if explicitly requested

### M2: Persistent
- **Retention**: 1-7 years (configurable)
- **Examples**: Contract records, decisions, credentials
- **Policy**: Auto-archive, prune after retention period

### M3: Foundational
- **Retention**: Permanent
- **Examples**: Constitutional documents, precedents, scientific findings
- **Policy**: Multi-layer storage, never prune

```rust
fn get_retention_policy(materiality: &MaterialityLevel) -> RetentionConfig {
    match materiality {
        MaterialityLevel::M0Ephemeral => RetentionConfig {
            archive: false,
            retention_days: None,
            storage_layers: vec![],
        },
        MaterialityLevel::M1Temporal => RetentionConfig {
            archive: true, // On request
            retention_days: Some(90),
            storage_layers: vec![StorageType::HolochainDHT],
        },
        MaterialityLevel::M2Persistent => RetentionConfig {
            archive: true,
            retention_days: Some(365 * 5), // 5 years
            storage_layers: vec![
                StorageType::HolochainDHT,
                StorageType::IPFS,
            ],
        },
        MaterialityLevel::M3Foundational => RetentionConfig {
            archive: true,
            retention_days: None, // Permanent
            storage_layers: vec![
                StorageType::HolochainDHT,
                StorageType::IPFS,
                StorageType::Arweave,
            ],
        },
    }
}
```

---

## Zome Specifications

### 1. Archive Manager Zome

```rust
// archive_manager/src/lib.rs

/// Archive content from any source
#[hdk_extern]
pub fn archive_content(input: ArchiveInput) -> ExternResult<ActionHash> {
    let archiver = agent_info()?.agent_latest_pubkey;

    // Verify archiver has authority (for high-materiality content)
    if input.epistemic.materiality >= MaterialityLevel::M3Foundational {
        verify_archive_authority(&archiver)?;
    }

    // Validate epistemic classification
    validate_epistemic_classification(&input.epistemic)?;

    // Determine storage layers
    let storage_layers = get_retention_policy(&input.epistemic.materiality).storage_layers;

    // Store content
    let content = prepare_content(&input.content)?;
    let storage = store_to_layers(&content, &storage_layers)?;

    // Build provenance
    let provenance = build_provenance(&input)?;

    let archive = ArchiveEntry {
        archive_id: generate_archive_id(),
        content_type: input.content_type,
        content: content.clone(),
        epistemic: EpistemicClassification {
            empirical: input.epistemic.empirical,
            normative: input.epistemic.normative,
            materiality: input.epistemic.materiality,
            confidence: 1.0,
            classified_by: ClassificationAuthority::Source,
            classified_at: sys_time()?,
            evidence: vec![],
        },
        provenance,
        archived_by: archiver.clone(),
        archived_at: sys_time()?,
        storage,
        access: input.access.unwrap_or(AccessPolicy {
            default: AccessLevel::Public,
            overrides: vec![],
            encryption: None,
        }),
        relations: input.relations,
        tags: input.tags,
        citation_count: 0,
    };

    let hash = create_entry(&EntryTypes::ArchiveEntry(archive.clone()))?;

    // Index for search
    index_archive(&hash, &archive)?;

    // Create links for relations
    for relation in &archive.relations {
        create_link(
            relation.target.clone(),
            hash.clone(),
            LinkTypes::ArchiveRelation,
            relation.relation_type.to_bytes(),
        )?;
    }

    Ok(hash)
}

/// Auto-archive from hApp event
#[hdk_extern]
pub fn on_happ_event(event: HAppEvent) -> ExternResult<Option<ActionHash>> {
    // Check if this event should be archived
    let should_archive = should_archive_event(&event)?;

    if !should_archive {
        return Ok(None);
    }

    // Determine content type and classification
    let (content_type, epistemic) = classify_happ_event(&event)?;

    // Only archive M2+ by default
    if epistemic.materiality < MaterialityLevel::M2Persistent {
        return Ok(None);
    }

    // Archive
    let hash = archive_content(ArchiveInput {
        content_type,
        content: ArchiveContent::HolochainRef(event.entry_hash.clone()),
        epistemic,
        access: None,
        relations: vec![],
        tags: extract_tags_from_event(&event),
    })?;

    Ok(Some(hash))
}

/// Retrieve archive entry
#[hdk_extern]
pub fn get_archive(archive_hash: ActionHash) -> ExternResult<ArchiveEntry> {
    let requestor = agent_info()?.agent_latest_pubkey;

    let archive = get_archive_entry(&archive_hash)?;

    // Check access
    verify_archive_access(&requestor, &archive)?;

    // Log access for analytics
    log_archive_access(&archive_hash, &requestor)?;

    Ok(archive)
}

/// Store content to multiple storage layers
fn store_to_layers(
    content: &ArchiveContent,
    layers: &[StorageType],
) -> ExternResult<Vec<StorageLocation>> {
    let mut locations = vec![];

    for layer in layers {
        let location = match layer {
            StorageType::HolochainDHT => {
                // Already stored via create_entry
                StorageLocation {
                    storage_type: StorageType::HolochainDHT,
                    identifier: "local".to_string(),
                    verified: true,
                    last_verified: sys_time()?,
                }
            }
            StorageType::IPFS => {
                // Store to IPFS via oracle/bridge
                let cid = bridge_call::<String>(
                    "oracle",
                    "store_to_ipfs",
                    content.clone(),
                )?;
                StorageLocation {
                    storage_type: StorageType::IPFS,
                    identifier: cid,
                    verified: true,
                    last_verified: sys_time()?,
                }
            }
            StorageType::Arweave => {
                // Store to Arweave for permanence
                let tx_id = bridge_call::<String>(
                    "oracle",
                    "store_to_arweave",
                    content.clone(),
                )?;
                StorageLocation {
                    storage_type: StorageType::Arweave,
                    identifier: tx_id,
                    verified: true,
                    last_verified: sys_time()?,
                }
            }
            _ => continue,
        };
        locations.push(location);
    }

    Ok(locations)
}
```

### 2. Classify Engine Zome

```rust
// classify_engine/src/lib.rs

/// Classify content epistemically
#[hdk_extern]
pub fn classify_content(input: ClassifyInput) -> ExternResult<EpistemicClassification> {
    let classifier = agent_info()?.agent_latest_pubkey;

    // Determine empirical level
    let empirical = determine_empirical_level(&input)?;

    // Determine normative level
    let normative = determine_normative_level(&input)?;

    // Determine materiality level
    let materiality = determine_materiality_level(&input)?;

    // Calculate confidence
    let confidence = calculate_classification_confidence(&input)?;

    // Determine authority
    let authority = if has_expert_credential(&classifier, &input.domain)? {
        ClassificationAuthority::Expert {
            credential: get_expert_credential(&classifier, &input.domain)?,
        }
    } else {
        ClassificationAuthority::System {
            algorithm: "hybrid_classification_v1".to_string(),
        }
    };

    Ok(EpistemicClassification {
        empirical,
        normative,
        materiality,
        confidence,
        classified_by: authority,
        classified_at: sys_time()?,
        evidence: input.evidence,
    })
}

/// Determine empirical level based on content characteristics
fn determine_empirical_level(input: &ClassifyInput) -> ExternResult<EmpiricalLevel> {
    // Check for cryptographic proofs
    if has_cryptographic_proof(&input.content) {
        return Ok(EmpiricalLevel::E3Cryptographic);
    }

    // Check for reproducible methodology
    if has_reproducible_methodology(&input.content) {
        return Ok(EmpiricalLevel::E4PublicRepro);
    }

    // Check for audit trail
    if has_audit_trail(&input.content) {
        return Ok(EmpiricalLevel::E2PrivateVerify);
    }

    // Check for multiple witnesses
    if has_multiple_witnesses(&input.content) {
        return Ok(EmpiricalLevel::E2PrivateVerify);
    }

    // Single source testimonial
    if has_identified_source(&input.content) {
        return Ok(EmpiricalLevel::E1Testimonial);
    }

    // No verification possible
    Ok(EmpiricalLevel::E0Null)
}

/// Determine normative level based on consensus
fn determine_normative_level(input: &ClassifyInput) -> ExternResult<NormativeLevel> {
    // Check if axiomatic (mathematical/logical necessity)
    if is_axiomatic(&input.content) {
        return Ok(NormativeLevel::N3Axiomatic);
    }

    // Check network consensus
    if has_network_consensus(&input.content)? {
        return Ok(NormativeLevel::N2Network);
    }

    // Check community agreement
    if has_community_agreement(&input.content)? {
        return Ok(NormativeLevel::N1Communal);
    }

    // Personal claim
    Ok(NormativeLevel::N0Personal)
}

/// Determine materiality level
fn determine_materiality_level(input: &ClassifyInput) -> ExternResult<MaterialityLevel> {
    // Constitutional/foundational content
    if is_foundational_content(&input.content_type) {
        return Ok(MaterialityLevel::M3Foundational);
    }

    // Persistent records (contracts, decisions)
    if is_persistent_record(&input.content_type) {
        return Ok(MaterialityLevel::M2Persistent);
    }

    // Temporal (state-dependent)
    if is_state_dependent(&input.content) {
        return Ok(MaterialityLevel::M1Temporal);
    }

    // Ephemeral
    Ok(MaterialityLevel::M0Ephemeral)
}

fn is_foundational_content(content_type: &ArchiveContentType) -> bool {
    matches!(
        content_type,
        ArchiveContentType::Constitution { .. } |
        ArchiveContentType::ProtocolSpec { .. } |
        ArchiveContentType::CredentialDefinition { .. }
    )
}

fn is_persistent_record(content_type: &ArchiveContentType) -> bool {
    matches!(
        content_type,
        ArchiveContentType::LegalPrecedent { .. } |
        ArchiveContentType::GovernanceDecision { .. } |
        ArchiveContentType::ResearchFinding { .. } |
        ArchiveContentType::HistoricalEvent { .. }
    )
}
```

### 3. Citation Tracker Zome

```rust
// citation_tracker/src/lib.rs

/// Create a citation
#[hdk_extern]
pub fn cite(input: CiteInput) -> ExternResult<ActionHash> {
    let citer = agent_info()?.agent_latest_pubkey;

    // Verify target exists and is accessible
    let target = get_archive(&input.target)?;
    verify_archive_access(&citer, &target)?;

    let citation = Citation {
        source: input.source,
        target: input.target.clone(),
        citation_type: input.citation_type,
        context: input.context,
        created_by: citer,
        created_at: sys_time()?,
    };

    let hash = create_entry(&EntryTypes::Citation(citation))?;

    // Update citation count on target
    increment_citation_count(&input.target)?;

    // Create link
    create_link(
        input.target,
        hash.clone(),
        LinkTypes::ArchiveToCitation,
        (),
    )?;

    Ok(hash)
}

/// Get all citations of an archive entry
#[hdk_extern]
pub fn get_citations(archive_hash: ActionHash) -> ExternResult<Vec<Citation>> {
    let links = get_links(
        GetLinksInputBuilder::try_new(archive_hash, LinkTypes::ArchiveToCitation)?
            .build()
    )?;

    let citations: Vec<Citation> = links
        .iter()
        .filter_map(|link| {
            let hash = link.target.clone().into_action_hash()?;
            get_citation(&hash).ok()
        })
        .collect();

    Ok(citations)
}

/// Get citation network (for analysis)
#[hdk_extern]
pub fn get_citation_network(
    root: ActionHash,
    depth: u32,
) -> ExternResult<CitationNetwork> {
    let mut nodes = vec![];
    let mut edges = vec![];
    let mut visited = HashSet::new();

    build_citation_network(&root, depth, &mut nodes, &mut edges, &mut visited)?;

    Ok(CitationNetwork { nodes, edges })
}

fn build_citation_network(
    node: &ActionHash,
    remaining_depth: u32,
    nodes: &mut Vec<NetworkNode>,
    edges: &mut Vec<NetworkEdge>,
    visited: &mut HashSet<ActionHash>,
) -> ExternResult<()> {
    if remaining_depth == 0 || visited.contains(node) {
        return Ok(());
    }

    visited.insert(node.clone());

    let archive = get_archive(node.clone())?;
    nodes.push(NetworkNode {
        hash: node.clone(),
        title: get_archive_title(&archive),
        content_type: archive.content_type.clone(),
        citation_count: archive.citation_count,
    });

    // Get citations (what cites this)
    let citations = get_citations(node.clone())?;
    for citation in citations {
        if let CitationSource::Archive(source) = citation.source {
            edges.push(NetworkEdge {
                from: source.clone(),
                to: node.clone(),
                citation_type: citation.citation_type,
            });
            build_citation_network(
                &source,
                remaining_depth - 1,
                nodes,
                edges,
                visited,
            )?;
        }
    }

    // Get what this cites
    for relation in &archive.relations {
        if matches!(relation.relation_type, RelationType::Cites) {
            edges.push(NetworkEdge {
                from: node.clone(),
                to: relation.target.clone(),
                citation_type: CitationType::Reference,
            });
            build_citation_network(
                &relation.target,
                remaining_depth - 1,
                nodes,
                edges,
                visited,
            )?;
        }
    }

    Ok(())
}
```

### 4. Prune Manager Zome

```rust
// prune_manager/src/lib.rs

/// Run pruning cycle
#[hdk_extern]
pub fn run_prune_cycle() -> ExternResult<PruneResult> {
    // Get all active prune policies
    let policies = get_active_policies()?;

    let mut pruned_count = 0;
    let mut retained_count = 0;
    let mut errors = vec![];

    for policy in policies {
        // Get archives matching this policy
        let candidates = get_prune_candidates(&policy)?;

        for candidate in candidates {
            // Check retention conditions
            let should_retain = check_retention_conditions(&candidate, &policy)?;

            if should_retain {
                retained_count += 1;
                continue;
            }

            // Check if past retention period
            let age_days = (sys_time()? - candidate.archived_at).as_secs() / 86400;
            if let Some(retention_days) = policy.retention_days {
                if age_days < retention_days {
                    retained_count += 1;
                    continue;
                }
            }

            // Prune (soft delete or move to cold storage)
            match prune_archive(&candidate) {
                Ok(_) => pruned_count += 1,
                Err(e) => errors.push(format!(
                    "Failed to prune {}: {}",
                    candidate.archive_id, e
                )),
            }
        }
    }

    Ok(PruneResult {
        pruned_count,
        retained_count,
        errors,
        cycle_timestamp: sys_time()?,
    })
}

/// Check if archive should be retained despite age
fn check_retention_conditions(
    archive: &ArchiveEntry,
    policy: &PrunePolicy,
) -> ExternResult<bool> {
    for condition in &policy.retain_if {
        match condition {
            RetentionCondition::HasCitations { min_count } => {
                if archive.citation_count >= *min_count {
                    return Ok(true);
                }
            }
            RetentionCondition::ActiveReferences => {
                let refs = get_active_references(&archive)?;
                if !refs.is_empty() {
                    return Ok(true);
                }
            }
            RetentionCondition::HighAccess { min_per_month } => {
                let access_rate = get_access_rate(&archive)?;
                if access_rate >= *min_per_month as f64 {
                    return Ok(true);
                }
            }
            RetentionCondition::ImportantSource { min_trust } => {
                let source_trust = get_source_trust(&archive.provenance)?;
                if source_trust >= *min_trust {
                    return Ok(true);
                }
            }
            RetentionCondition::CommunityFlagged => {
                if is_community_flagged(&archive)? {
                    return Ok(true);
                }
            }
        }
    }

    Ok(false)
}

/// Prune an archive (soft delete, keep metadata)
fn prune_archive(archive: &ArchiveEntry) -> ExternResult<()> {
    // Move content to cold storage if needed
    if archive.epistemic.materiality >= MaterialityLevel::M2Persistent {
        // Keep in at least one external storage
        ensure_cold_storage(archive)?;
    }

    // Mark as pruned in DHT
    mark_as_pruned(&archive)?;

    // Keep metadata for reference
    create_prune_record(&archive)?;

    Ok(())
}

/// Ensure content exists in cold storage before pruning
fn ensure_cold_storage(archive: &ArchiveEntry) -> ExternResult<()> {
    let has_external = archive.storage
        .iter()
        .any(|s| matches!(s.storage_type, StorageType::IPFS | StorageType::Arweave));

    if !has_external {
        // Store to IPFS before pruning
        let cid = bridge_call::<String>(
            "oracle",
            "store_to_ipfs",
            archive.content.clone(),
        )?;

        // Update storage locations
        add_storage_location(archive, StorageLocation {
            storage_type: StorageType::IPFS,
            identifier: cid,
            verified: true,
            last_verified: sys_time()?,
        })?;
    }

    Ok(())
}
```

### 5. Search Index Zome

```rust
// search_index/src/lib.rs

/// Search archives
#[hdk_extern]
pub fn search(query: SearchQuery) -> ExternResult<SearchResults> {
    let searcher = agent_info()?.agent_latest_pubkey;

    let mut results = vec![];

    // Full text search
    if let Some(text) = &query.text {
        let text_results = full_text_search(text)?;
        results.extend(text_results);
    }

    // Filter by content type
    if let Some(content_type) = &query.content_type {
        let type_results = search_by_content_type(content_type)?;
        results = intersect_results(results, type_results);
    }

    // Filter by epistemic level
    if let Some(min_empirical) = &query.min_empirical {
        results.retain(|r| r.epistemic.empirical >= *min_empirical);
    }

    if let Some(min_materiality) = &query.min_materiality {
        results.retain(|r| r.epistemic.materiality >= *min_materiality);
    }

    // Filter by tags
    if !query.tags.is_empty() {
        results.retain(|r| query.tags.iter().any(|t| r.tags.contains(t)));
    }

    // Filter by date range
    if let Some(date_range) = &query.date_range {
        results.retain(|r| {
            r.archived_at >= date_range.start &&
            r.archived_at <= date_range.end
        });
    }

    // Filter by access
    results.retain(|r| can_access(&searcher, &r.access));

    // Sort by relevance
    sort_by_relevance(&mut results, &query);

    // Paginate
    let total = results.len();
    let page_results: Vec<_> = results
        .into_iter()
        .skip(query.offset)
        .take(query.limit)
        .collect();

    Ok(SearchResults {
        results: page_results,
        total,
        offset: query.offset,
        limit: query.limit,
    })
}

/// Get archive statistics
#[hdk_extern]
pub fn get_statistics() -> ExternResult<ArchiveStats> {
    let total_archives = count_all_archives()?;
    let by_type = count_by_content_type()?;
    let by_materiality = count_by_materiality()?;
    let total_citations = count_all_citations()?;
    let storage_usage = calculate_storage_usage()?;

    Ok(ArchiveStats {
        total_archives,
        by_content_type: by_type,
        by_materiality,
        total_citations,
        storage_usage_bytes: storage_usage,
        last_prune: get_last_prune_time()?,
        oldest_archive: get_oldest_archive_time()?,
    })
}
```

---

## Cross-hApp Integration

### Auto-Archive Events

```rust
// Register for events from other hApps
pub fn register_auto_archive_sources() -> ExternResult<()> {
    // Arbiter rulings
    bridge_call::<()>(
        "arbiter",
        "register_event_listener",
        EventListener {
            event_type: "ruling_issued".to_string(),
            callback_happ: "chronicle".to_string(),
            callback_fn: "on_happ_event".to_string(),
        },
    )?;

    // Agora decisions
    bridge_call::<()>(
        "agora",
        "register_event_listener",
        EventListener {
            event_type: "proposal_finalized".to_string(),
            callback_happ: "chronicle".to_string(),
            callback_fn: "on_happ_event".to_string(),
        },
    )?;

    // Praxis credentials
    bridge_call::<()>(
        "praxis",
        "register_event_listener",
        EventListener {
            event_type: "credential_issued".to_string(),
            callback_happ: "chronicle".to_string(),
            callback_fn: "on_happ_event".to_string(),
        },
    )?;

    Ok(())
}
```

---

## Success Metrics

| Metric | 1 Year | 3 Years |
|--------|--------|---------|
| Archives stored | 10,000 | 500,000 |
| M3 (permanent) entries | 500 | 10,000 |
| Citations | 5,000 | 100,000 |
| Search queries/month | 10,000 | 500,000 |
| Storage redundancy | 3x | 5x |
| Data integrity | 100% | 100% |

---

*"What is not remembered cannot guide the future. Chronicle is civilization's memory."*
