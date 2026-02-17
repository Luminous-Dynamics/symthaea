//! Search Coordinator Zome - Business logic for full-text search
use hdk::prelude::*;
use search_integrity::*;
use std::collections::HashMap;

// ===== Input/Output Types =====

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateIndexInput {
    pub entity_type: EntityType,
    pub entity_hash: ActionHash,
    pub title: String,
    pub description: String,
    pub tags: Vec<String>,
    pub category: String,
    pub creator_matl_score: f64,
    pub price_cents: Option<u64>,
    pub status: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SearchQuery {
    pub query: String,
    pub entity_types: Option<Vec<EntityType>>,
    pub limit: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SearchResult {
    pub entity_hash: ActionHash,
    pub entity_type: EntityType,
    pub title: String,
    pub description_snippet: String,
    pub score: f64,
    pub creator: AgentPubKey,
    pub created_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub total_count: u32,
    pub query_time_ms: u64,
}

// ===== Helper Functions =====

/// Simple text processor - normalizes and tokenizes text
fn process_text(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .filter(|word| word.len() > 2) // Filter short words
        .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|word| !word.is_empty())
        .map(String::from)
        .collect()
}

/// Calculate term frequencies
fn calculate_term_frequencies(terms: &[String]) -> HashMap<String, u32> {
    let mut frequencies = HashMap::new();
    for term in terms {
        *frequencies.entry(term.clone()).or_insert(0) += 1;
    }
    frequencies
}

/// Simple relevance scoring based on term matches
fn calculate_relevance_score(index_entry: &SearchIndexEntry, query_terms: &[String]) -> f64 {
    let mut score = 0.0;

    for query_term in query_terms {
        // Check title (weighted higher)
        if index_entry.title.to_lowercase().contains(query_term) {
            score += 3.0;
        }

        // Check description
        if index_entry.description.to_lowercase().contains(query_term) {
            score += 1.0;
        }

        // Check tags
        for tag in &index_entry.tags {
            if tag.to_lowercase().contains(query_term) {
                score += 2.0;
            }
        }

        // Check processed terms (exact match)
        if index_entry.processed_terms.contains(query_term) {
            score += 0.5;
        }
    }

    // Apply MATL trust weighting
    score * (0.5 + 0.5 * index_entry.creator_matl_score)
}

fn get_all_indices_links() -> ExternResult<Vec<Link>> {
    let path = Path::from("all_indices");
    let filter = LinkTypeFilter::try_from(LinkTypes::AllIndices)?;
    get_links(LinkQuery::new(path.path_entry_hash()?, filter), GetStrategy::default())
}

// ===== Public API =====

/// Create or update a search index entry
///
/// This is called by other zomes when creating/updating searchable entries
#[hdk_extern]
pub fn create_search_index(input: CreateIndexInput) -> ExternResult<ActionHash> {
    // Input validation
    if input.title.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Title cannot be empty".into()
        )));
    }

    if input.description.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Description cannot be empty".into()
        )));
    }

    let agent_info = agent_info()?;
    let now = sys_time()?;

    // Process all searchable text
    let all_text = format!(
        "{} {} {}",
        input.title,
        input.description,
        input.tags.join(" ")
    );
    let processed_terms = process_text(&all_text);
    let term_frequencies = calculate_term_frequencies(&processed_terms);

    // Create index entry
    let index_entry = SearchIndexEntry {
        index_id: format!("{:?}_{}", input.entity_type, input.entity_hash),
        entity_type: input.entity_type.clone(),
        entity_hash: input.entity_hash.clone(),
        title: input.title,
        description: input.description,
        tags: input.tags,
        category: input.category,
        creator: agent_info.agent_initial_pubkey,
        creator_matl_score: input.creator_matl_score,
        created_at: now,
        updated_at: now,
        price_cents: input.price_cents,
        status: input.status,
        processed_terms,
        term_frequencies,
    };

    // Create entry
    let action_hash = create_entry(&EntryTypes::SearchIndexEntry(index_entry.clone()))?;
    let entry_hash = hash_entry(&index_entry)?;

    // Create discovery links

    // 1. Entity -> Index
    create_link(
        input.entity_hash,
        entry_hash.clone(),
        LinkTypes::EntityToIndex,
        (),
    )?;

    // 2. All indices anchor
    let all_path = Path::from("all_indices");
    create_link(
        all_path.path_entry_hash()?,
        entry_hash.clone(),
        LinkTypes::AllIndices,
        (),
    )?;

    // 3. EntityType -> Index (for filtering by type)
    let type_path = Path::from(format!("indices.type.{:?}", index_entry.entity_type));
    create_link(
        type_path.path_entry_hash()?,
        entry_hash,
        LinkTypes::EntityTypeToIndex,
        (),
    )?;

    Ok(action_hash)
}

/// Search across indexed entities
///
/// Performs full-text search using pre-built search indices
#[hdk_extern]
pub fn search(query: SearchQuery) -> ExternResult<SearchResponse> {
    let start_time = sys_time()?;

    // Validate input
    if query.query.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Query cannot be empty".into()
        )));
    }

    // Process query into search terms
    let query_terms = process_text(&query.query);

    if query_terms.is_empty() {
        return Ok(SearchResponse {
            results: Vec::new(),
            total_count: 0,
            query_time_ms: 0,
        });
    }

    // Get all search indices
    let links = get_all_indices_links()?;

    let mut scored_results: Vec<(SearchIndexEntry, f64)> = Vec::new();

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(index_entry) = record
                    .entry()
                    .to_app_option::<SearchIndexEntry>()
                    .map_err(|e| wasm_error!(e))?
                {
                    // Apply entity type filter if specified
                    if let Some(ref types) = query.entity_types {
                        if !types.contains(&index_entry.entity_type) {
                            continue;
                        }
                    }

                    // Calculate relevance score
                    let score = calculate_relevance_score(&index_entry, &query_terms);

                    // Only include if score > 0
                    if score > 0.0 {
                        scored_results.push((index_entry, score));
                    }
                }
            }
        }
    }

    // Sort by score (descending)
    scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Apply limit
    scored_results.truncate(query.limit as usize);

    // Convert to search results
    let results: Vec<SearchResult> = scored_results
        .into_iter()
        .map(|(entry, score)| {
            // Create description snippet (first 200 chars)
            let description_snippet = if entry.description.len() > 200 {
                format!("{}...", &entry.description[..200])
            } else {
                entry.description.clone()
            };

            SearchResult {
                entity_hash: entry.entity_hash,
                entity_type: entry.entity_type,
                title: entry.title,
                description_snippet,
                score,
                creator: entry.creator,
                created_at: entry.created_at,
            }
        })
        .collect();

    let total_count = results.len() as u32;
    let end_time = sys_time()?;
    let query_time_ms = ((end_time.as_micros() - start_time.as_micros()) / 1000) as u64;

    Ok(SearchResponse {
        results,
        total_count,
        query_time_ms,
    })
}

/// Get search index for a specific entity
#[hdk_extern]
pub fn get_entity_index(entity_hash: ActionHash) -> ExternResult<Option<SearchIndexEntry>> {
    let filter = LinkTypeFilter::try_from(LinkTypes::EntityToIndex)?;
    let links = get_links(LinkQuery::new(entity_hash, filter), GetStrategy::default())?;

    if links.is_empty() {
        return Ok(None);
    }

    // Get the first (should only be one)
    if let Some(action_hash) = links[0].target.clone().into_action_hash() {
        if let Some(record) = get(action_hash, GetOptions::default())? {
            let index_entry = record
                .entry()
                .to_app_option::<SearchIndexEntry>()
                .map_err(|e| wasm_error!(e))?;
            return Ok(index_entry);
        }
    }

    Ok(None)
}

/// Delete search index for an entity
#[hdk_extern]
pub fn delete_search_index(entity_hash: ActionHash) -> ExternResult<()> {
    let filter = LinkTypeFilter::try_from(LinkTypes::EntityToIndex)?;
    let links = get_links(LinkQuery::new(entity_hash, filter), GetStrategy::default())?;

    // Delete all links to this entity's index
    for link in links {
        delete_link(link.create_link_hash, GetOptions::default())?;
    }

    Ok(())
}

#[cfg(test)]
mod tests;
