//! LUCID Sources Coordinator Zome
//!
//! Source attribution and provenance operations.

use hdk::prelude::*;
use sources_integrity::*;

// ============================================================================
// HELPERS
// ============================================================================

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn create_anchor(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    create_entry(&EntryTypes::Anchor(anchor))?;
    anchor_hash(anchor_str)
}

fn generate_uuid() -> String {
    let mut bytes = [0u8; 16];
    getrandom_03::fill(&mut bytes).expect("Failed to generate random bytes");
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5], bytes[6], bytes[7],
        bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
    )
}

// ============================================================================
// SOURCE OPERATIONS
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateSourceInput {
    pub source_type: SourceType,
    pub title: String,
    pub url: Option<String>,
    pub authors: Option<Vec<String>>,
    pub publication_date: Option<String>,
    pub publisher: Option<String>,
    pub doi: Option<String>,
    pub isbn: Option<String>,
    pub description: Option<String>,
    pub credibility: Option<f64>,
    pub notes: Option<String>,
    pub tags: Option<Vec<String>>,
}

#[hdk_extern]
pub fn create_source(input: CreateSourceInput) -> ExternResult<Record> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;
    let id = generate_uuid();

    let source = Source {
        id: id.clone(),
        source_type: input.source_type.clone(),
        title: input.title,
        url: input.url.clone(),
        authors: input.authors.unwrap_or_default(),
        publication_date: input.publication_date,
        publisher: input.publisher,
        doi: input.doi.clone(),
        isbn: input.isbn,
        description: input.description,
        content_hash: None,
        credibility: input.credibility.unwrap_or(input.source_type.quality_weight()),
        notes: input.notes,
        tags: input.tags.unwrap_or_default(),
        added_at: now,
        updated_at: now,
    };

    let action_hash = create_entry(&EntryTypes::Source(source.clone()))?;

    // Index by agent
    let agent_anchor = format!("agent_sources:{}", agent);
    create_anchor(&agent_anchor)?;
    create_link(
        anchor_hash(&agent_anchor)?,
        action_hash.clone(),
        LinkTypes::AgentToSource,
        (),
    )?;

    // Index by source ID
    let id_anchor = format!("source:{}", id);
    create_anchor(&id_anchor)?;
    create_link(
        anchor_hash(&id_anchor)?,
        action_hash.clone(),
        LinkTypes::SourceIdToSource,
        (),
    )?;

    // Index by type
    let type_anchor = format!("source_type:{:?}", input.source_type);
    create_anchor(&type_anchor)?;
    create_link(
        anchor_hash(&type_anchor)?,
        action_hash.clone(),
        LinkTypes::SourceTypeToSource,
        (),
    )?;

    // Index by URL domain if available
    if let Some(ref url) = input.url {
        if let Some(domain) = extract_domain(url) {
            let domain_anchor = format!("source_domain:{}", domain);
            create_anchor(&domain_anchor)?;
            create_link(
                anchor_hash(&domain_anchor)?,
                action_hash.clone(),
                LinkTypes::DomainToSource,
                (),
            )?;
        }

        // Index by full URL for deduplication
        let url_anchor = format!("url:{}", url);
        create_anchor(&url_anchor)?;
        create_link(
            anchor_hash(&url_anchor)?,
            action_hash.clone(),
            LinkTypes::UrlToSource,
            (),
        )?;
    }

    // Index by DOI if available
    if let Some(ref doi) = input.doi {
        let doi_anchor = format!("doi:{}", doi);
        create_anchor(&doi_anchor)?;
        create_link(
            anchor_hash(&doi_anchor)?,
            action_hash.clone(),
            LinkTypes::DoiToSource,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find created source".into()
        )))
}

fn extract_domain(url: &str) -> Option<String> {
    let url = url.strip_prefix("https://").or_else(|| url.strip_prefix("http://"))?;
    let domain = url.split('/').next()?;
    Some(domain.to_lowercase())
}

#[hdk_extern]
pub fn get_source(source_id: String) -> ExternResult<Option<Record>> {
    let id_anchor = format!("source:{}", source_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&id_anchor)?, LinkTypes::SourceIdToSource)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.last() {
        let action_hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        return get(action_hash, GetOptions::default());
    }

    Ok(None)
}

#[hdk_extern]
pub fn get_source_by_url(url: String) -> ExternResult<Option<Record>> {
    let url_anchor = format!("url:{}", url);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&url_anchor)?, LinkTypes::UrlToSource)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.last() {
        let action_hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        return get(action_hash, GetOptions::default());
    }

    Ok(None)
}

#[hdk_extern]
pub fn get_my_sources(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_anchor = format!("agent_sources:{}", agent);

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&agent_anchor)?, LinkTypes::AgentToSource)?,
        GetStrategy::default(),
    )?;

    let mut sources = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            sources.push(record);
        }
    }

    Ok(sources)
}

#[hdk_extern]
pub fn get_sources_by_type(source_type: SourceType) -> ExternResult<Vec<Record>> {
    let type_anchor = format!("source_type:{:?}", source_type);

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&type_anchor)?, LinkTypes::SourceTypeToSource)?,
        GetStrategy::default(),
    )?;

    let mut sources = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            sources.push(record);
        }
    }

    Ok(sources)
}

// ============================================================================
// CITATION OPERATIONS
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateCitationInput {
    pub thought_id: String,
    pub source_id: String,
    pub location: Option<String>,
    pub quote: Option<String>,
    pub relationship: Option<CitationRelationship>,
    pub strength: Option<f64>,
    pub notes: Option<String>,
}

#[hdk_extern]
pub fn create_citation(input: CreateCitationInput) -> ExternResult<Record> {
    // Verify source exists
    let _ = get_source(input.source_id.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Source not found".into())))?;

    let now = sys_time()?;
    let id = generate_uuid();

    let citation = Citation {
        id,
        thought_id: input.thought_id.clone(),
        source_id: input.source_id.clone(),
        location: input.location,
        quote: input.quote,
        relationship: input.relationship.unwrap_or_default(),
        strength: input.strength.unwrap_or(1.0),
        notes: input.notes,
        created_at: now,
    };

    let action_hash = create_entry(&EntryTypes::Citation(citation))?;

    // Link thought to citation
    let thought_anchor = format!("thought:{}", input.thought_id);
    create_anchor(&thought_anchor)?;
    create_link(
        anchor_hash(&thought_anchor)?,
        action_hash.clone(),
        LinkTypes::ThoughtToCitation,
        (),
    )?;

    // Link citation to source
    let source_anchor = format!("source:{}", input.source_id);
    create_link(
        action_hash.clone(),
        anchor_hash(&source_anchor)?,
        LinkTypes::CitationToSource,
        (),
    )?;

    // Link source to citation (for reverse lookup)
    create_link(
        anchor_hash(&source_anchor)?,
        action_hash.clone(),
        LinkTypes::SourceToCitation,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find created citation".into()
        )))
}

#[hdk_extern]
pub fn get_thought_citations(thought_id: String) -> ExternResult<Vec<Record>> {
    let thought_anchor = format!("thought:{}", thought_id);

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&thought_anchor)?, LinkTypes::ThoughtToCitation)?,
        GetStrategy::default(),
    )?;

    let mut citations = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            citations.push(record);
        }
    }

    Ok(citations)
}

#[hdk_extern]
pub fn get_source_citations(source_id: String) -> ExternResult<Vec<Record>> {
    let source_anchor = format!("source:{}", source_id);

    let links = get_links(
        LinkQuery::try_new(anchor_hash(&source_anchor)?, LinkTypes::SourceToCitation)?,
        GetStrategy::default(),
    )?;

    let mut citations = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            citations.push(record);
        }
    }

    Ok(citations)
}

/// Get all sources for a thought (via citations)
#[hdk_extern]
pub fn get_thought_sources(thought_id: String) -> ExternResult<Vec<Record>> {
    let citations = get_thought_citations(thought_id)?;

    let mut sources = Vec::new();
    for citation_record in citations {
        if let Some(citation) = citation_record
            .entry()
            .to_app_option::<Citation>()
            .ok()
            .flatten()
        {
            if let Some(source_record) = get_source(citation.source_id)? {
                sources.push(source_record);
            }
        }
    }

    Ok(sources)
}
