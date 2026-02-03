//! Civic Knowledge Coordinator Zome
//!
//! Provides CRUD operations and search functionality for civic knowledge.
//! This is the interface that Symthaea AI agents use to query civic information.
//!
//! ## Key Functions
//!
//! - `create_knowledge`: Add new civic knowledge to the DHT
//! - `get_knowledge`: Retrieve a specific knowledge entry
//! - `search_by_domain`: Find knowledge by civic domain
//! - `search_by_keywords`: Search across all knowledge by keywords
//! - `validate_knowledge`: Submit a validation for knowledge accuracy

use hdk::prelude::*;
use civic_knowledge_integrity::*;

/// Helper to create or get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Ensure an anchor exists and return its hash
fn ensure_anchor(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    create_entry(&EntryTypes::Anchor(anchor.clone()))?;
    anchor_hash(anchor_str)
}

/// Input for creating civic knowledge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateKnowledgeInput {
    pub domain: CivicDomain,
    pub knowledge_type: KnowledgeType,
    pub title: String,
    pub content: String,
    pub geographic_scope: Option<String>,
    pub keywords: Vec<String>,
    pub source: Option<String>,
    pub expires_at: Option<u64>,
    pub links: Vec<String>,
    pub contact_phone: Option<String>,
    pub address: Option<String>,
}

/// Create a new civic knowledge entry
#[hdk_extern]
pub fn create_knowledge(input: CreateKnowledgeInput) -> ExternResult<ActionHash> {
    let knowledge = CivicKnowledge {
        domain: input.domain,
        knowledge_type: input.knowledge_type,
        title: input.title,
        content: input.content,
        geographic_scope: input.geographic_scope.clone(),
        keywords: input.keywords.clone(),
        source: input.source,
        last_verified: Some(sys_time()?.as_micros() as u64),
        expires_at: input.expires_at,
        links: input.links,
        contact_phone: input.contact_phone,
        address: input.address,
    };

    let action_hash = create_entry(&EntryTypes::CivicKnowledge(knowledge.clone()))?;

    // Link from domain anchor
    let domain_anchor = format!("domain:{:?}", input.domain).to_lowercase();
    let domain_hash = ensure_anchor(&domain_anchor)?;
    create_link(
        domain_hash,
        action_hash.clone(),
        CivicKnowledgeLinkTypes::DomainToKnowledge,
        (),
    )?;

    // Link from geographic anchor if provided
    if let Some(ref geo_scope) = input.geographic_scope {
        let geo_anchor = format!("geo:{}", geo_scope.to_lowercase());
        let geo_hash = ensure_anchor(&geo_anchor)?;
        create_link(
            geo_hash,
            action_hash.clone(),
            CivicKnowledgeLinkTypes::GeoToKnowledge,
            (),
        )?;
    }

    // Link from keyword anchors
    for keyword in &input.keywords {
        let keyword_anchor = format!("keyword:{}", keyword.to_lowercase());
        let keyword_hash = ensure_anchor(&keyword_anchor)?;
        create_link(
            keyword_hash,
            action_hash.clone(),
            CivicKnowledgeLinkTypes::KeywordToKnowledge,
            (),
        )?;
    }

    // Link from agent
    let my_pubkey = agent_info()?.agent_initial_pubkey;
    create_link(
        my_pubkey,
        action_hash.clone(),
        CivicKnowledgeLinkTypes::AgentToKnowledge,
        (),
    )?;

    Ok(action_hash)
}

/// Get a specific knowledge entry by hash
#[hdk_extern]
pub fn get_knowledge(action_hash: ActionHash) -> ExternResult<Option<CivicKnowledge>> {
    let Some(record) = get(action_hash, GetOptions::default())? else {
        return Ok(None);
    };

    let knowledge: CivicKnowledge = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Not a CivicKnowledge entry".into())))?;

    Ok(Some(knowledge))
}

/// Search result containing knowledge and its hash
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeSearchResult {
    pub action_hash: ActionHash,
    pub knowledge: CivicKnowledge,
}

/// Search by civic domain
#[hdk_extern]
pub fn search_by_domain(domain: CivicDomain) -> ExternResult<Vec<KnowledgeSearchResult>> {
    let domain_anchor = format!("domain:{:?}", domain).to_lowercase();
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&domain_anchor)?,
            CivicKnowledgeLinkTypes::DomainToKnowledge,
        )?,
        GetStrategy::default(),
    )?;

    let mut results = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(knowledge) = get_knowledge(action_hash.clone())? {
            results.push(KnowledgeSearchResult {
                action_hash,
                knowledge,
            });
        }
    }

    Ok(results)
}

/// Search by geographic scope
#[hdk_extern]
pub fn search_by_location(geo_scope: String) -> ExternResult<Vec<KnowledgeSearchResult>> {
    let geo_anchor = format!("geo:{}", geo_scope.to_lowercase());
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&geo_anchor)?,
            CivicKnowledgeLinkTypes::GeoToKnowledge,
        )?,
        GetStrategy::default(),
    )?;

    let mut results = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(knowledge) = get_knowledge(action_hash.clone())? {
            results.push(KnowledgeSearchResult {
                action_hash,
                knowledge,
            });
        }
    }

    Ok(results)
}

/// Search by keyword
#[hdk_extern]
pub fn search_by_keyword(keyword: String) -> ExternResult<Vec<KnowledgeSearchResult>> {
    let keyword_anchor = format!("keyword:{}", keyword.to_lowercase());
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&keyword_anchor)?,
            CivicKnowledgeLinkTypes::KeywordToKnowledge,
        )?,
        GetStrategy::default(),
    )?;

    let mut results = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(knowledge) = get_knowledge(action_hash.clone())? {
            results.push(KnowledgeSearchResult {
                action_hash,
                knowledge,
            });
        }
    }

    Ok(results)
}

/// Combined search input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchInput {
    pub domain: Option<CivicDomain>,
    pub geo_scope: Option<String>,
    pub keywords: Vec<String>,
    pub limit: Option<usize>,
}

/// Combined search across multiple criteria
#[hdk_extern]
pub fn search_knowledge(input: SearchInput) -> ExternResult<Vec<KnowledgeSearchResult>> {
    let mut all_results: Vec<KnowledgeSearchResult> = Vec::new();
    let mut seen_hashes: std::collections::HashSet<ActionHash> = std::collections::HashSet::new();

    // Search by domain if provided
    if let Some(domain) = input.domain {
        for result in search_by_domain(domain)? {
            if seen_hashes.insert(result.action_hash.clone()) {
                all_results.push(result);
            }
        }
    }

    // Search by location if provided
    if let Some(ref geo_scope) = input.geo_scope {
        for result in search_by_location(geo_scope.clone())? {
            if seen_hashes.insert(result.action_hash.clone()) {
                all_results.push(result);
            }
        }
    }

    // Search by keywords
    for keyword in &input.keywords {
        for result in search_by_keyword(keyword.clone())? {
            if seen_hashes.insert(result.action_hash.clone()) {
                all_results.push(result);
            }
        }
    }

    // Apply limit
    if let Some(limit) = input.limit {
        all_results.truncate(limit);
    }

    Ok(all_results)
}

/// Input for validating knowledge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidateKnowledgeInput {
    pub knowledge_hash: ActionHash,
    pub is_valid: bool,
    pub authority_type: String,
    pub notes: Option<String>,
}

/// Submit a validation for a knowledge entry
#[hdk_extern]
pub fn validate_knowledge(input: ValidateKnowledgeInput) -> ExternResult<ActionHash> {
    // Verify the knowledge entry exists
    let _ = get_knowledge(input.knowledge_hash.clone())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Knowledge entry not found".into())))?;

    let validation = KnowledgeValidation {
        knowledge_hash: input.knowledge_hash.clone(),
        is_valid: input.is_valid,
        authority_type: input.authority_type,
        notes: input.notes,
        validated_at: sys_time()?.as_micros() as u64,
    };

    let action_hash = create_entry(&EntryTypes::KnowledgeValidation(validation))?;

    // Link validation to knowledge
    create_link(
        input.knowledge_hash,
        action_hash.clone(),
        CivicKnowledgeLinkTypes::KnowledgeToValidation,
        (),
    )?;

    Ok(action_hash)
}

/// Get all validations for a knowledge entry
#[hdk_extern]
pub fn get_knowledge_validations(knowledge_hash: ActionHash) -> ExternResult<Vec<KnowledgeValidation>> {
    let links = get_links(
        LinkQuery::try_new(
            knowledge_hash,
            CivicKnowledgeLinkTypes::KnowledgeToValidation,
        )?,
        GetStrategy::default(),
    )?;

    let mut validations = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(validation) = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                validations.push(validation);
            }
        }
    }

    Ok(validations)
}

/// Input for updating knowledge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateKnowledgeInput {
    pub original_hash: ActionHash,
    pub updated_content: String,
    pub reason: String,
    pub submitter_note: Option<String>,
}

/// Submit an update to existing knowledge
#[hdk_extern]
pub fn update_knowledge(input: UpdateKnowledgeInput) -> ExternResult<ActionHash> {
    // Verify the original knowledge entry exists
    let _ = get_knowledge(input.original_hash.clone())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Original knowledge entry not found".into())))?;

    let update = KnowledgeUpdate {
        original_hash: input.original_hash.clone(),
        updated_content: input.updated_content,
        reason: input.reason,
        submitter_note: input.submitter_note,
    };

    let action_hash = create_entry(&EntryTypes::KnowledgeUpdate(update))?;

    // Link update to original knowledge
    create_link(
        input.original_hash,
        action_hash.clone(),
        CivicKnowledgeLinkTypes::KnowledgeToUpdate,
        (),
    )?;

    Ok(action_hash)
}

/// Get all updates for a knowledge entry
#[hdk_extern]
pub fn get_knowledge_updates(knowledge_hash: ActionHash) -> ExternResult<Vec<KnowledgeUpdate>> {
    let links = get_links(
        LinkQuery::try_new(
            knowledge_hash,
            CivicKnowledgeLinkTypes::KnowledgeToUpdate,
        )?,
        GetStrategy::default(),
    )?;

    let mut updates = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(update) = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                updates.push(update);
            }
        }
    }

    Ok(updates)
}

/// Get all knowledge authored by an agent
#[hdk_extern]
pub fn get_my_knowledge(_: ()) -> ExternResult<Vec<KnowledgeSearchResult>> {
    let my_pubkey = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(
            my_pubkey,
            CivicKnowledgeLinkTypes::AgentToKnowledge,
        )?,
        GetStrategy::default(),
    )?;

    let mut results = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(knowledge) = get_knowledge(action_hash.clone())? {
            results.push(KnowledgeSearchResult {
                action_hash,
                knowledge,
            });
        }
    }

    Ok(results)
}
