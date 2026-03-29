// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! LUCID Coordinator Zome
//!
//! Personal Knowledge Graph - Core Operations
//!
//! Provides CRUD operations for thoughts, tags, domains, and relationships.
//! All entries are indexed for efficient retrieval via anchor-based links.

use hdk::prelude::*;
use lucid_integrity::*;

// ============================================================================
// HELPERS
// ============================================================================

/// Get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Create an anchor and return its hash
fn create_anchor(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    create_entry(&EntryTypes::Anchor(anchor))?;
    anchor_hash(anchor_str)
}

/// Generate a UUID v4
fn generate_uuid() -> String {
    let mut bytes = [0u8; 16];
    getrandom_03::fill(&mut bytes).expect("Failed to generate random bytes");

    // Set version (4) and variant (RFC 4122)
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;

    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5],
        bytes[6], bytes[7],
        bytes[8], bytes[9],
        bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
    )
}

// ============================================================================
// THOUGHT OPERATIONS
// ============================================================================

/// Input for creating a new thought
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateThoughtInput {
    pub content: String,
    pub thought_type: Option<ThoughtType>,
    pub epistemic: Option<EpistemicClassification>,
    pub confidence: Option<f64>,
    pub tags: Option<Vec<String>>,
    pub domain: Option<String>,
    pub related_thoughts: Option<Vec<String>>,
    pub parent_thought: Option<String>,
    /// Optional pre-computed embedding from Symthaea (16,384 dimensions)
    pub embedding: Option<Vec<f32>>,
}

/// Create a new thought
#[hdk_extern]
pub fn create_thought(input: CreateThoughtInput) -> ExternResult<Record> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;
    let id = generate_uuid();

    // Set embedding version to 1 if embedding is provided
    let embedding_version = if input.embedding.is_some() { Some(1) } else { None };

    let thought = Thought {
        id: id.clone(),
        content: input.content,
        thought_type: input.thought_type.unwrap_or_default(),
        epistemic: input.epistemic.unwrap_or_default(),
        confidence: input.confidence.unwrap_or(0.5),
        tags: input.tags.unwrap_or_default(),
        domain: input.domain,
        related_thoughts: input.related_thoughts.unwrap_or_default(),
        source_hashes: vec![],
        parent_thought: input.parent_thought.clone(),
        created_at: now,
        updated_at: now,
        version: 1,
        // Symthaea integration fields
        embedding: input.embedding,
        embedding_version,
        coherence_score: None,
        phi_score: None,
    };

    let action_hash = create_entry(&EntryTypes::Thought(thought.clone()))?;

    // Index by agent
    let agent_anchor = format!("agent:{}", agent);
    create_anchor(&agent_anchor)?;
    create_link(
        anchor_hash(&agent_anchor)?,
        action_hash.clone(),
        LinkTypes::AgentToThought,
        (),
    )?;

    // Index by thought ID (for O(1) lookup)
    let id_anchor = format!("thought:{}", id);
    create_anchor(&id_anchor)?;
    create_link(
        anchor_hash(&id_anchor)?,
        action_hash.clone(),
        LinkTypes::ThoughtIdToThought,
        (),
    )?;

    // Index by tags
    for tag in &thought.tags {
        let tag_anchor = format!("tag:{}", tag.to_lowercase());
        create_anchor(&tag_anchor)?;
        create_link(
            anchor_hash(&tag_anchor)?,
            action_hash.clone(),
            LinkTypes::TagToThought,
            (),
        )?;
    }

    // Index by domain
    if let Some(ref domain) = thought.domain {
        let domain_anchor = format!("domain:{}", domain.to_lowercase());
        create_anchor(&domain_anchor)?;
        create_link(
            anchor_hash(&domain_anchor)?,
            action_hash.clone(),
            LinkTypes::DomainToThought,
            (),
        )?;
    }

    // Index by type
    let type_anchor = format!("type:{:?}", thought.thought_type);
    create_anchor(&type_anchor)?;
    create_link(
        anchor_hash(&type_anchor)?,
        action_hash.clone(),
        LinkTypes::TypeToThought,
        (),
    )?;

    // Index by epistemic levels (for filtering)
    let e_anchor = format!("epistemic:E{}", thought.epistemic.empirical as u8);
    create_anchor(&e_anchor)?;
    create_link(
        anchor_hash(&e_anchor)?,
        action_hash.clone(),
        LinkTypes::EmpiricalLevelIndex,
        (),
    )?;

    // Link to parent if hierarchical
    if let Some(ref parent_id) = input.parent_thought {
        let parent_anchor = format!("thought:{}", parent_id);
        if let Ok(parent_hash) = anchor_hash(&parent_anchor) {
            create_link(
                parent_hash,
                action_hash.clone(),
                LinkTypes::ParentToChildThought,
                (),
            )?;
            create_link(
                action_hash.clone(),
                anchor_hash(&parent_anchor)?,
                LinkTypes::ChildToParentThought,
                (),
            )?;
        }
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find created thought".into()
        )))
}

/// Get a thought by ID
#[hdk_extern]
pub fn get_thought(thought_id: String) -> ExternResult<Option<Record>> {
    let id_anchor = format!("thought:{}", thought_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&id_anchor)?, LinkTypes::ThoughtIdToThought)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.last() {
        let action_hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        return get(action_hash, GetOptions::default());
    }

    Ok(None)
}

/// Get a thought by action hash
#[hdk_extern]
pub fn get_thought_by_hash(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

/// Input for updating a thought
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateThoughtInput {
    pub thought_id: String,
    pub content: Option<String>,
    pub thought_type: Option<ThoughtType>,
    pub epistemic: Option<EpistemicClassification>,
    pub confidence: Option<f64>,
    pub tags: Option<Vec<String>>,
    pub domain: Option<String>,
    pub related_thoughts: Option<Vec<String>>,
}

/// Update an existing thought
#[hdk_extern]
pub fn update_thought(input: UpdateThoughtInput) -> ExternResult<Record> {
    let current_record = get_thought(input.thought_id.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Thought not found".into())))?;

    let current_thought: Thought = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid thought entry".into())))?;

    let now = sys_time()?;

    // Check if content changed - if so, embedding needs recomputation
    let content_changed = input.content.is_some() && input.content.as_ref() != Some(&current_thought.content);

    let updated_thought = Thought {
        id: current_thought.id,
        content: input.content.unwrap_or(current_thought.content),
        thought_type: input.thought_type.unwrap_or(current_thought.thought_type),
        epistemic: input.epistemic.unwrap_or(current_thought.epistemic),
        confidence: input.confidence.unwrap_or(current_thought.confidence),
        tags: input.tags.unwrap_or(current_thought.tags),
        domain: input.domain.or(current_thought.domain),
        related_thoughts: input.related_thoughts.unwrap_or(current_thought.related_thoughts),
        source_hashes: current_thought.source_hashes,
        parent_thought: current_thought.parent_thought,
        created_at: current_thought.created_at,
        updated_at: now,
        version: current_thought.version + 1,
        // Preserve embedding but mark as stale if content changed
        embedding: if content_changed { None } else { current_thought.embedding },
        embedding_version: if content_changed { None } else { current_thought.embedding_version },
        coherence_score: if content_changed { None } else { current_thought.coherence_score },
        phi_score: current_thought.phi_score,
    };

    let action_hash = update_entry(
        current_record.action_address().clone(),
        &EntryTypes::Thought(updated_thought),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find updated thought".into()
        )))
}

/// Delete a thought
#[hdk_extern]
pub fn delete_thought(thought_id: String) -> ExternResult<ActionHash> {
    let record = get_thought(thought_id)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Thought not found".into())))?;

    delete_entry(record.action_address().clone())
}

/// Get all thoughts for the current agent
#[hdk_extern]
pub fn get_my_thoughts(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_anchor = format!("agent:{}", agent);

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&agent_anchor)?, LinkTypes::AgentToThought)?,
        GetStrategy::default(),
    )?;

    let mut thoughts = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            thoughts.push(record);
        }
    }

    Ok(thoughts)
}

/// Get thoughts by tag
#[hdk_extern]
pub fn get_thoughts_by_tag(tag: String) -> ExternResult<Vec<Record>> {
    let tag_anchor = format!("tag:{}", tag.to_lowercase());

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&tag_anchor)?, LinkTypes::TagToThought)?,
        GetStrategy::default(),
    )?;

    let mut thoughts = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            thoughts.push(record);
        }
    }

    Ok(thoughts)
}

/// Get thoughts by domain
#[hdk_extern]
pub fn get_thoughts_by_domain(domain: String) -> ExternResult<Vec<Record>> {
    let domain_anchor = format!("domain:{}", domain.to_lowercase());

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&domain_anchor)?, LinkTypes::DomainToThought)?,
        GetStrategy::default(),
    )?;

    let mut thoughts = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            thoughts.push(record);
        }
    }

    Ok(thoughts)
}

/// Get thoughts by type
#[hdk_extern]
pub fn get_thoughts_by_type(thought_type: ThoughtType) -> ExternResult<Vec<Record>> {
    let type_anchor = format!("type:{:?}", thought_type);

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&type_anchor)?, LinkTypes::TypeToThought)?,
        GetStrategy::default(),
    )?;

    let mut thoughts = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            thoughts.push(record);
        }
    }

    Ok(thoughts)
}

/// Get child thoughts of a parent
#[hdk_extern]
pub fn get_child_thoughts(parent_thought_id: String) -> ExternResult<Vec<Record>> {
    let parent_anchor = format!("thought:{}", parent_thought_id);

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&parent_anchor)?, LinkTypes::ParentToChildThought)?,
        GetStrategy::default(),
    )?;

    let mut thoughts = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            thoughts.push(record);
        }
    }

    Ok(thoughts)
}

// ============================================================================
// SEARCH & FILTER
// ============================================================================

/// Search input for filtering thoughts
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SearchThoughtsInput {
    /// Filter by tags (AND logic)
    pub tags: Option<Vec<String>>,
    /// Filter by domain
    pub domain: Option<String>,
    /// Filter by thought type
    pub thought_type: Option<ThoughtType>,
    /// Minimum empirical level
    pub min_empirical: Option<EmpiricalLevel>,
    /// Minimum confidence
    pub min_confidence: Option<f64>,
    /// Text search in content
    pub content_contains: Option<String>,
    /// Limit results
    pub limit: Option<u32>,
    /// Offset for pagination
    pub offset: Option<u32>,
}

/// Search thoughts with filters
#[hdk_extern]
pub fn search_thoughts(input: SearchThoughtsInput) -> ExternResult<Vec<Record>> {
    // Start with all agent's thoughts
    let all_thoughts = get_my_thoughts(())?;

    let limit = input.limit.unwrap_or(100) as usize;
    let offset = input.offset.unwrap_or(0) as usize;

    let filtered: Vec<Record> = all_thoughts
        .into_iter()
        .filter(|record| {
            if let Some(thought) = record
                .entry()
                .to_app_option::<Thought>()
                .ok()
                .flatten()
            {
                // Filter by tags
                if let Some(ref filter_tags) = input.tags {
                    for tag in filter_tags {
                        if !thought.tags.iter().any(|t| t.to_lowercase() == tag.to_lowercase()) {
                            return false;
                        }
                    }
                }

                // Filter by domain
                if let Some(ref filter_domain) = input.domain {
                    if thought.domain.as_ref().map(|d| d.to_lowercase())
                        != Some(filter_domain.to_lowercase())
                    {
                        return false;
                    }
                }

                // Filter by type
                if let Some(ref filter_type) = input.thought_type {
                    if thought.thought_type != *filter_type {
                        return false;
                    }
                }

                // Filter by minimum empirical level
                if let Some(min_e) = input.min_empirical {
                    if thought.epistemic.empirical < min_e {
                        return false;
                    }
                }

                // Filter by minimum confidence
                if let Some(min_conf) = input.min_confidence {
                    if thought.confidence < min_conf {
                        return false;
                    }
                }

                // Text search in content
                if let Some(ref search_text) = input.content_contains {
                    if !thought
                        .content
                        .to_lowercase()
                        .contains(&search_text.to_lowercase())
                    {
                        return false;
                    }
                }

                true
            } else {
                false
            }
        })
        .skip(offset)
        .take(limit)
        .collect();

    Ok(filtered)
}

// ============================================================================
// TAG OPERATIONS
// ============================================================================

/// Create a new tag
#[hdk_extern]
pub fn create_tag(input: CreateTagInput) -> ExternResult<Record> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let tag = Tag {
        name: input.name.to_lowercase(),
        description: input.description,
        color: input.color,
        parent_tag: input.parent_tag,
        created_at: now,
    };

    let action_hash = create_entry(&EntryTypes::Tag(tag.clone()))?;

    // Index by agent
    let agent_anchor = format!("agent_tags:{}", agent);
    create_anchor(&agent_anchor)?;
    create_link(
        anchor_hash(&agent_anchor)?,
        action_hash.clone(),
        LinkTypes::AgentToTag,
        (),
    )?;

    // Index by tag name
    let name_anchor = format!("tagdef:{}", tag.name);
    create_anchor(&name_anchor)?;
    create_link(anchor_hash(&name_anchor)?, action_hash.clone(), LinkTypes::AgentToTag, ())?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find created tag".into()
        )))
}

/// Input for creating a tag
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateTagInput {
    pub name: String,
    pub description: Option<String>,
    pub color: Option<String>,
    pub parent_tag: Option<String>,
}

/// Get all tags for the current agent
#[hdk_extern]
pub fn get_my_tags(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_anchor = format!("agent_tags:{}", agent);

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&agent_anchor)?, LinkTypes::AgentToTag)?,
        GetStrategy::default(),
    )?;

    let mut tags = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            tags.push(record);
        }
    }

    Ok(tags)
}

// ============================================================================
// DOMAIN OPERATIONS
// ============================================================================

/// Create a new domain
#[hdk_extern]
pub fn create_domain(input: CreateDomainInput) -> ExternResult<Record> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let domain = Domain {
        name: input.name,
        description: input.description,
        parent_domain: input.parent_domain,
        related_domains: input.related_domains.unwrap_or_default(),
        created_at: now,
    };

    let action_hash = create_entry(&EntryTypes::Domain(domain.clone()))?;

    // Index by agent
    let agent_anchor = format!("agent_domains:{}", agent);
    create_anchor(&agent_anchor)?;
    create_link(
        anchor_hash(&agent_anchor)?,
        action_hash.clone(),
        LinkTypes::AgentToDomain,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find created domain".into()
        )))
}

/// Input for creating a domain
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateDomainInput {
    pub name: String,
    pub description: Option<String>,
    pub parent_domain: Option<String>,
    pub related_domains: Option<Vec<String>>,
}

/// Get all domains for the current agent
#[hdk_extern]
pub fn get_my_domains(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_anchor = format!("agent_domains:{}", agent);

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&agent_anchor)?, LinkTypes::AgentToDomain)?,
        GetStrategy::default(),
    )?;

    let mut domains = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            domains.push(record);
        }
    }

    Ok(domains)
}

// ============================================================================
// RELATIONSHIP OPERATIONS
// ============================================================================

/// Input for creating a relationship
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateRelationshipInput {
    pub from_thought_id: String,
    pub to_thought_id: String,
    pub relationship_type: RelationshipType,
    pub strength: Option<f64>,
    pub description: Option<String>,
    pub bidirectional: Option<bool>,
}

/// Create a relationship between two thoughts
#[hdk_extern]
pub fn create_relationship(input: CreateRelationshipInput) -> ExternResult<Record> {
    // Verify both thoughts exist
    let _ = get_thought(input.from_thought_id.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("From thought not found".into())))?;
    let _ = get_thought(input.to_thought_id.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("To thought not found".into())))?;

    let now = sys_time()?;
    let id = generate_uuid();

    let relationship = Relationship {
        id,
        from_thought_id: input.from_thought_id.clone(),
        to_thought_id: input.to_thought_id.clone(),
        relationship_type: input.relationship_type,
        strength: input.strength.unwrap_or(1.0),
        description: input.description,
        bidirectional: input.bidirectional.unwrap_or(false),
        created_at: now,
        active: true,
    };

    let action_hash = create_entry(&EntryTypes::Relationship(relationship.clone()))?;

    // Link from source thought to relationship
    let from_anchor = format!("thought:{}", input.from_thought_id);
    create_link(
        anchor_hash(&from_anchor)?,
        action_hash.clone(),
        LinkTypes::ThoughtToRelationship,
        (),
    )?;

    // Link from relationship to target thought
    let to_anchor = format!("thought:{}", input.to_thought_id);
    create_link(
        action_hash.clone(),
        anchor_hash(&to_anchor)?,
        LinkTypes::RelationshipToThought,
        (),
    )?;

    // If bidirectional, create reverse links too
    if relationship.bidirectional {
        create_link(
            anchor_hash(&to_anchor)?,
            action_hash.clone(),
            LinkTypes::ThoughtToRelationship,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find created relationship".into()
        )))
}

/// Get relationships from a thought
#[hdk_extern]
pub fn get_thought_relationships(thought_id: String) -> ExternResult<Vec<Record>> {
    let thought_anchor = format!("thought:{}", thought_id);

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&thought_anchor)?, LinkTypes::ThoughtToRelationship)?,
        GetStrategy::default(),
    )?;

    let mut relationships = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            relationships.push(record);
        }
    }

    Ok(relationships)
}

/// Get thoughts related to a given thought
#[hdk_extern]
pub fn get_related_thoughts(thought_id: String) -> ExternResult<Vec<Record>> {
    let relationships = get_thought_relationships(thought_id.clone())?;

    let mut related = Vec::new();
    for rel_record in relationships {
        if let Some(rel) = rel_record
            .entry()
            .to_app_option::<Relationship>()
            .ok()
            .flatten()
        {
            // Get the "other" thought in the relationship
            let other_id = if rel.from_thought_id == thought_id {
                rel.to_thought_id
            } else {
                rel.from_thought_id
            };

            if let Some(thought_record) = get_thought(other_id)? {
                related.push(thought_record);
            }
        }
    }

    Ok(related)
}

// ============================================================================
// STATISTICS
// ============================================================================

/// Statistics about the user's knowledge graph
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KnowledgeGraphStats {
    pub total_thoughts: u32,
    pub by_type: Vec<(String, u32)>,
    pub by_domain: Vec<(String, u32)>,
    pub average_confidence: f64,
    pub average_epistemic_strength: f64,
    pub top_tags: Vec<(String, u32)>,
}

/// Get statistics about the knowledge graph
#[hdk_extern]
pub fn get_stats(_: ()) -> ExternResult<KnowledgeGraphStats> {
    let thoughts = get_my_thoughts(())?;
    let total = thoughts.len() as u32;

    let mut type_counts: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut domain_counts: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut tag_counts: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut total_confidence = 0.0;
    let mut total_strength = 0.0;

    for record in &thoughts {
        if let Some(thought) = record.entry().to_app_option::<Thought>().ok().flatten() {
            // Count by type
            let type_str = format!("{:?}", thought.thought_type);
            *type_counts.entry(type_str).or_insert(0) += 1;

            // Count by domain
            if let Some(domain) = thought.domain {
                *domain_counts.entry(domain).or_insert(0) += 1;
            }

            // Count tags
            for tag in thought.tags {
                *tag_counts.entry(tag).or_insert(0) += 1;
            }

            // Sum confidence and strength
            total_confidence += thought.confidence;
            total_strength += thought.epistemic.overall_strength();
        }
    }

    let mut by_type: Vec<(String, u32)> = type_counts.into_iter().collect();
    by_type.sort_by(|a, b| b.1.cmp(&a.1));

    let mut by_domain: Vec<(String, u32)> = domain_counts.into_iter().collect();
    by_domain.sort_by(|a, b| b.1.cmp(&a.1));

    let mut top_tags: Vec<(String, u32)> = tag_counts.into_iter().collect();
    top_tags.sort_by(|a, b| b.1.cmp(&a.1));
    top_tags.truncate(10);

    Ok(KnowledgeGraphStats {
        total_thoughts: total,
        by_type,
        by_domain,
        average_confidence: if total > 0 {
            total_confidence / total as f64
        } else {
            0.0
        },
        average_epistemic_strength: if total > 0 {
            total_strength / total as f64
        } else {
            0.0
        },
        top_tags,
    })
}

// ============================================================================
// SYMTHAEA INTEGRATION - SEMANTIC SEARCH & EMBEDDINGS
// ============================================================================

/// Input for updating a thought's embedding
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateEmbeddingInput {
    pub thought_id: String,
    /// 16,384-dimensional HDC embedding from Symthaea
    pub embedding: Vec<f32>,
}

/// Update a thought with its Symthaea embedding
/// Called from Tauri after computing embedding via Symthaea
#[hdk_extern]
pub fn update_thought_embedding(input: UpdateEmbeddingInput) -> ExternResult<Record> {
    // Validate embedding dimension
    if input.embedding.len() != 16384 {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Embedding must be 16,384 dimensions, got {}",
            input.embedding.len()
        ))));
    }

    let current_record = get_thought(input.thought_id.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Thought not found".into())))?;

    let current_thought: Thought = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid thought entry".into())))?;

    let now = sys_time()?;

    let updated_thought = Thought {
        embedding: Some(input.embedding),
        embedding_version: Some(current_thought.version),
        updated_at: now,
        version: current_thought.version + 1,
        ..current_thought
    };

    let action_hash = update_entry(
        current_record.action_address().clone(),
        &EntryTypes::Thought(updated_thought),
    )?;

    // Index in the embedding index for quick semantic queries
    let embedding_anchor = "embeddings:all";
    create_anchor(embedding_anchor)?;
    create_link(
        anchor_hash(embedding_anchor)?,
        action_hash.clone(),
        LinkTypes::EmbeddingIndex,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find updated thought".into()
        )))
}

/// Input for updating coherence/phi scores
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateCoherenceInput {
    pub thought_id: String,
    pub coherence_score: f64,
    pub phi_score: Option<f64>,
}

/// Update a thought with coherence analysis results
#[hdk_extern]
pub fn update_thought_coherence(input: UpdateCoherenceInput) -> ExternResult<Record> {
    // Validate scores
    if input.coherence_score < 0.0 || input.coherence_score > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Coherence score must be between 0.0 and 1.0".into()
        )));
    }

    let current_record = get_thought(input.thought_id.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Thought not found".into())))?;

    let current_thought: Thought = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid thought entry".into())))?;

    let now = sys_time()?;

    let updated_thought = Thought {
        coherence_score: Some(input.coherence_score),
        phi_score: input.phi_score.or(current_thought.phi_score),
        updated_at: now,
        version: current_thought.version + 1,
        ..current_thought
    };

    let action_hash = update_entry(
        current_record.action_address().clone(),
        &EntryTypes::Thought(updated_thought),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find updated thought".into()
        )))
}

/// Input for semantic search
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SemanticSearchInput {
    /// Query embedding (16,384 dimensions)
    pub query_embedding: Vec<f32>,
    /// Minimum similarity threshold (0.0-1.0)
    pub threshold: f64,
    /// Maximum number of results
    pub limit: Option<u32>,
}

/// Result of semantic search
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SemanticSearchResult {
    pub thought_id: String,
    pub action_hash: ActionHash,
    pub similarity: f64,
    pub content_preview: String,
}

/// Perform semantic search using cosine similarity on embeddings
#[hdk_extern]
pub fn semantic_search(input: SemanticSearchInput) -> ExternResult<Vec<SemanticSearchResult>> {
    // Validate query embedding dimension
    if input.query_embedding.len() != 16384 {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Query embedding must be 16,384 dimensions, got {}",
            input.query_embedding.len()
        ))));
    }

    let limit = input.limit.unwrap_or(20) as usize;

    // Get all thoughts with embeddings
    let embedding_anchor = "embeddings:all";
    let links = get_links(
        LinkQuery::try_new(anchor_hash(embedding_anchor)?, LinkTypes::EmbeddingIndex)?,
        GetStrategy::default(),
    )?;

    let mut results: Vec<SemanticSearchResult> = Vec::new();

    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
            if let Some(thought) = record
                .entry()
                .to_app_option::<Thought>()
                .ok()
                .flatten()
            {
                if let Some(ref embedding) = thought.embedding {
                    let similarity = cosine_similarity(&input.query_embedding, embedding);

                    if similarity >= input.threshold {
                        results.push(SemanticSearchResult {
                            thought_id: thought.id,
                            action_hash,
                            similarity,
                            content_preview: thought.content.chars().take(200).collect(),
                        });
                    }
                }
            }
        }
    }

    // Sort by similarity (descending) and limit results
    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);

    Ok(results)
}

/// Calculate cosine similarity between two embeddings
/// NOTE: Duplicated in collective/coordinator (WASM boundary prevents sharing)
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        norm_a += (*x as f64) * (*x as f64);
        norm_b += (*y as f64) * (*y as f64);
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        dot / denom
    }
}

/// Get thoughts that need embeddings computed
/// Returns thoughts where embedding is None or embedding_version < version
#[hdk_extern]
pub fn get_thoughts_needing_embeddings(_: ()) -> ExternResult<Vec<Record>> {
    let all_thoughts = get_my_thoughts(())?;

    let needing_embeddings: Vec<Record> = all_thoughts
        .into_iter()
        .filter(|record| {
            if let Some(thought) = record
                .entry()
                .to_app_option::<Thought>()
                .ok()
                .flatten()
            {
                // Needs embedding if none, or if content was updated after embedding
                thought.embedding.is_none()
                    || thought.embedding_version.map_or(true, |ev| ev < thought.version)
            } else {
                false
            }
        })
        .collect();

    Ok(needing_embeddings)
}

/// Find thoughts with low coherence that may have contradictions
#[hdk_extern]
pub fn get_low_coherence_thoughts(threshold: f64) -> ExternResult<Vec<Record>> {
    let all_thoughts = get_my_thoughts(())?;

    let low_coherence: Vec<Record> = all_thoughts
        .into_iter()
        .filter(|record| {
            if let Some(thought) = record
                .entry()
                .to_app_option::<Thought>()
                .ok()
                .flatten()
            {
                thought.coherence_score.map_or(false, |cs| cs < threshold)
            } else {
                false
            }
        })
        .collect();

    Ok(low_coherence)
}

// ============================================================================
// EPISTEMIC GARDEN
// ============================================================================

/// Output for a thought cluster in the epistemic garden
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ThoughtCluster {
    pub cluster_id: u32,
    pub centroid_thought_id: String,
    pub thought_ids: Vec<String>,
    pub avg_confidence: f64,
    pub avg_coherence: Option<f64>,
    pub dominant_domain: Option<String>,
    pub dominant_tags: Vec<String>,
}

/// Input for exploring the epistemic garden
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ExploreGardenInput {
    pub max_clusters: u32,
    pub min_cluster_size: u32,
    pub domain_filter: Option<String>,
}

/// Explore the epistemic garden — cluster thoughts by semantic similarity
#[hdk_extern]
pub fn explore_garden(input: ExploreGardenInput) -> ExternResult<Vec<ThoughtCluster>> {
    let all = get_my_thoughts(())?;

    // Collect thoughts with embeddings
    let mut embedded_thoughts: Vec<(String, Vec<f32>, Thought)> = Vec::new();
    for record in &all {
        if let Some(thought) = record.entry().to_app_option::<Thought>().ok().flatten() {
            if let Some(ref domain_filter) = input.domain_filter {
                if thought.domain.as_deref() != Some(domain_filter.as_str()) {
                    continue;
                }
            }
            if let Some(ref emb) = thought.embedding {
                embedded_thoughts.push((thought.id.clone(), emb.clone(), thought));
            }
        }
    }

    if embedded_thoughts.is_empty() {
        return Ok(Vec::new());
    }

    // Simple greedy clustering: assign each thought to the nearest seed
    let max_clusters = (input.max_clusters as usize).min(embedded_thoughts.len());
    // Pick seeds evenly spaced through the list
    let step = embedded_thoughts.len().max(1) / max_clusters.max(1);
    let seeds: Vec<usize> = (0..max_clusters).map(|i| (i * step).min(embedded_thoughts.len() - 1)).collect();

    let mut assignments: Vec<usize> = vec![0; embedded_thoughts.len()];
    for (i, (_, emb, _)) in embedded_thoughts.iter().enumerate() {
        let mut best_cluster = 0;
        let mut best_sim = f64::NEG_INFINITY;
        for (ci, &seed_idx) in seeds.iter().enumerate() {
            let sim = cosine_similarity_f32(emb, &embedded_thoughts[seed_idx].1);
            if sim > best_sim {
                best_sim = sim;
                best_cluster = ci;
            }
        }
        assignments[i] = best_cluster;
    }

    // Build clusters
    let mut clusters: Vec<ThoughtCluster> = Vec::new();
    for ci in 0..max_clusters {
        let members: Vec<usize> = assignments.iter().enumerate()
            .filter(|(_, &c)| c == ci)
            .map(|(i, _)| i)
            .collect();

        if (members.len() as u32) < input.min_cluster_size {
            continue;
        }

        let thought_ids: Vec<String> = members.iter().map(|&i| embedded_thoughts[i].0.clone()).collect();
        let avg_confidence = members.iter()
            .map(|&i| embedded_thoughts[i].2.confidence)
            .sum::<f64>() / members.len() as f64;
        let avg_coherence = {
            let scores: Vec<f64> = members.iter()
                .filter_map(|&i| embedded_thoughts[i].2.coherence_score)
                .collect();
            if scores.is_empty() { None } else { Some(scores.iter().sum::<f64>() / scores.len() as f64) }
        };

        // Most common domain
        let mut domain_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for &i in &members {
            if let Some(ref d) = embedded_thoughts[i].2.domain {
                *domain_counts.entry(d.clone()).or_default() += 1;
            }
        }
        let dominant_domain = domain_counts.into_iter().max_by_key(|(_, c)| *c).map(|(d, _)| d);

        // Most common tags (top 3)
        let mut tag_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for &i in &members {
            for tag in &embedded_thoughts[i].2.tags {
                *tag_counts.entry(tag.clone()).or_default() += 1;
            }
        }
        let mut tag_vec: Vec<(String, usize)> = tag_counts.into_iter().collect();
        tag_vec.sort_by(|a, b| b.1.cmp(&a.1));
        let dominant_tags: Vec<String> = tag_vec.into_iter().take(3).map(|(t, _)| t).collect();

        clusters.push(ThoughtCluster {
            cluster_id: ci as u32,
            centroid_thought_id: embedded_thoughts[seeds[ci]].0.clone(),
            thought_ids,
            avg_confidence,
            avg_coherence,
            dominant_domain,
            dominant_tags,
        });
    }

    // Sort by size descending
    clusters.sort_by(|a, b| b.thought_ids.len().cmp(&a.thought_ids.len()));
    Ok(clusters)
}

/// Suggestion for connecting thoughts
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConnectionSuggestion {
    pub thought_a_id: String,
    pub thought_b_id: String,
    pub similarity: f64,
    pub shared_tags: Vec<String>,
}

/// Input for suggesting connections
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SuggestConnectionsInput {
    pub thought_id: String,
    pub max_suggestions: u32,
    pub min_similarity: f64,
}

/// Suggest unlinked thoughts that are semantically similar
#[hdk_extern]
pub fn suggest_connections(input: SuggestConnectionsInput) -> ExternResult<Vec<ConnectionSuggestion>> {
    let all = get_my_thoughts(())?;

    // Find the target thought
    let mut target: Option<Thought> = None;
    let mut others: Vec<Thought> = Vec::new();
    for record in &all {
        if let Some(thought) = record.entry().to_app_option::<Thought>().ok().flatten() {
            if thought.id == input.thought_id {
                target = Some(thought);
            } else {
                others.push(thought);
            }
        }
    }

    let target = match target {
        Some(t) => t,
        None => return Err(wasm_error!(WasmErrorInner::Guest("Thought not found".into()))),
    };

    let target_emb = match target.embedding {
        Some(ref e) => e,
        None => return Ok(Vec::new()), // No embedding, can't suggest
    };

    // Already-linked thought IDs
    let linked: std::collections::HashSet<&str> = target.related_thoughts.iter().map(|s| s.as_str()).collect();

    let mut suggestions: Vec<ConnectionSuggestion> = Vec::new();
    for other in &others {
        // Skip already-linked thoughts
        if linked.contains(other.id.as_str()) {
            continue;
        }
        if let Some(ref other_emb) = other.embedding {
            let sim = cosine_similarity_f32(target_emb, other_emb);
            if sim >= input.min_similarity {
                let shared_tags: Vec<String> = target.tags.iter()
                    .filter(|t| other.tags.contains(t))
                    .cloned()
                    .collect();
                suggestions.push(ConnectionSuggestion {
                    thought_a_id: target.id.clone(),
                    thought_b_id: other.id.clone(),
                    similarity: sim,
                    shared_tags,
                });
            }
        }
    }

    // Sort by similarity descending
    suggestions.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
    suggestions.truncate(input.max_suggestions as usize);
    Ok(suggestions)
}

/// Knowledge gap in a domain
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KnowledgeGap {
    pub domain: String,
    pub thought_count: u32,
    pub avg_confidence: f64,
    pub low_confidence_count: u32,
    pub missing_embeddings: u32,
    pub sparse: bool,
}

/// Find knowledge gaps — domains with sparse or low-confidence coverage
#[hdk_extern]
pub fn find_knowledge_gaps(_: ()) -> ExternResult<Vec<KnowledgeGap>> {
    let all = get_my_thoughts(())?;

    let mut domain_stats: std::collections::HashMap<String, (u32, f64, u32, u32)> = std::collections::HashMap::new();
    for record in &all {
        if let Some(thought) = record.entry().to_app_option::<Thought>().ok().flatten() {
            if let Some(ref domain) = thought.domain {
                let entry = domain_stats.entry(domain.clone()).or_insert((0, 0.0, 0, 0));
                entry.0 += 1; // count
                entry.1 += thought.confidence; // sum confidence
                if thought.confidence < 0.5 {
                    entry.2 += 1; // low confidence count
                }
                if thought.embedding.is_none() {
                    entry.3 += 1; // missing embeddings
                }
            }
        }
    }

    let total_thoughts: u32 = domain_stats.values().map(|(c, _, _, _)| c).sum();
    let avg_per_domain = if domain_stats.is_empty() { 0.0 } else {
        total_thoughts as f64 / domain_stats.len() as f64
    };

    let mut gaps: Vec<KnowledgeGap> = domain_stats.into_iter().map(|(domain, (count, conf_sum, low, missing))| {
        let avg_confidence = if count > 0 { conf_sum / count as f64 } else { 0.0 };
        let sparse = (count as f64) < avg_per_domain * 0.5;
        KnowledgeGap {
            domain,
            thought_count: count,
            avg_confidence,
            low_confidence_count: low,
            missing_embeddings: missing,
            sparse,
        }
    }).collect();

    // Sort: sparse first, then by low avg confidence
    gaps.sort_by(|a, b| {
        b.sparse.cmp(&a.sparse)
            .then(a.avg_confidence.partial_cmp(&b.avg_confidence).unwrap_or(std::cmp::Ordering::Equal))
    });
    Ok(gaps)
}

/// Discovered pattern in the knowledge graph
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DiscoveredPattern {
    pub pattern_type: String,
    pub description: String,
    pub involved_thought_ids: Vec<String>,
    pub confidence: f64,
}

/// Discover patterns — find recurring themes, contradictions, and evolution
#[hdk_extern]
pub fn discover_patterns(_: ()) -> ExternResult<Vec<DiscoveredPattern>> {
    let all = get_my_thoughts(())?;

    let mut thoughts: Vec<Thought> = Vec::new();
    for record in &all {
        if let Some(thought) = record.entry().to_app_option::<Thought>().ok().flatten() {
            thoughts.push(thought);
        }
    }

    let mut patterns: Vec<DiscoveredPattern> = Vec::new();

    // Pattern 1: Tag clusters (tags shared by 3+ thoughts)
    let mut tag_thoughts: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    for t in &thoughts {
        for tag in &t.tags {
            tag_thoughts.entry(tag.clone()).or_default().push(t.id.clone());
        }
    }
    for (tag, ids) in &tag_thoughts {
        if ids.len() >= 3 {
            patterns.push(DiscoveredPattern {
                pattern_type: "recurring_theme".to_string(),
                description: format!("Tag '{}' appears in {} thoughts", tag, ids.len()),
                involved_thought_ids: ids.clone(),
                confidence: (ids.len() as f64 / thoughts.len().max(1) as f64).min(1.0),
            });
        }
    }

    // Pattern 2: Confidence evolution (thoughts in same domain with increasing/decreasing confidence)
    let mut domain_thoughts: std::collections::HashMap<String, Vec<&Thought>> = std::collections::HashMap::new();
    for t in &thoughts {
        if let Some(ref domain) = t.domain {
            domain_thoughts.entry(domain.clone()).or_default().push(t);
        }
    }
    for (domain, mut dts) in domain_thoughts {
        if dts.len() < 3 {
            continue;
        }
        dts.sort_by_key(|t| t.created_at.as_micros());
        let confidences: Vec<f64> = dts.iter().map(|t| t.confidence).collect();
        let increasing = confidences.windows(2).all(|w| w[1] >= w[0]);
        let decreasing = confidences.windows(2).all(|w| w[1] <= w[0]);
        if increasing || decreasing {
            patterns.push(DiscoveredPattern {
                pattern_type: if increasing { "confidence_growth" } else { "confidence_decline" }.to_string(),
                description: format!(
                    "Confidence {} in domain '{}' across {} thoughts",
                    if increasing { "increasing" } else { "decreasing" },
                    domain,
                    dts.len()
                ),
                involved_thought_ids: dts.iter().map(|t| t.id.clone()).collect(),
                confidence: 0.7,
            });
        }
    }

    // Pattern 3: Low coherence clusters (multiple thoughts with coherence < 0.4)
    let low_coherence: Vec<&Thought> = thoughts.iter()
        .filter(|t| t.coherence_score.map_or(false, |c| c < 0.4))
        .collect();
    if low_coherence.len() >= 2 {
        patterns.push(DiscoveredPattern {
            pattern_type: "coherence_tension".to_string(),
            description: format!("{} thoughts have low coherence — potential contradictions", low_coherence.len()),
            involved_thought_ids: low_coherence.iter().map(|t| t.id.clone()).collect(),
            confidence: 0.6,
        });
    }

    // Sort by confidence descending
    patterns.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
    Ok(patterns)
}

/// Compute cosine similarity between two f32 vectors
fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for i in 0..a.len() {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 { 0.0 } else { dot / denom }
}

// ============================================================================
// COHERENCE QUERIES
// ============================================================================

/// Get thoughts by coherence score range
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CoherenceRangeInput {
    pub min_coherence: f64,
    pub max_coherence: f64,
}

#[hdk_extern]
pub fn get_thoughts_by_coherence(input: CoherenceRangeInput) -> ExternResult<Vec<Record>> {
    let all_thoughts = get_my_thoughts(())?;

    let in_range: Vec<Record> = all_thoughts
        .into_iter()
        .filter(|record| {
            if let Some(thought) = record
                .entry()
                .to_app_option::<Thought>()
                .ok()
                .flatten()
            {
                thought.coherence_score.map_or(false, |cs| {
                    cs >= input.min_coherence && cs <= input.max_coherence
                })
            } else {
                false
            }
        })
        .collect();

    Ok(in_range)
}
