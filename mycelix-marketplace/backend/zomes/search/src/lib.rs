use hdk::prelude::*;
use std::collections::HashMap;

mod text_processor;
mod tf_idf;
mod query_parser;
mod ranker;

use text_processor::TextProcessor;
use tf_idf::TfIdfCalculator;
use query_parser::QueryParser;
use ranker::Ranker;

// ===== Data Structures =====

/// Entity types that can be searched
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "PascalCase")]
pub enum EntityType {
    Listing,
    Transaction,
    Conversation,
    Review,
    Agent,
}

/// Search index entry for an entity
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SearchIndexEntry {
    /// Unique index ID
    pub index_id: String,

    /// Entity type being indexed
    pub entity_type: EntityType,

    /// Original entry hash
    pub entity_hash: ActionHash,

    /// Searchable fields
    pub title: String,
    pub description: String,
    pub tags: Vec<String>,
    pub category: String,

    /// Metadata for ranking
    pub creator: AgentPubKey,
    pub creator_matl_score: f64,
    pub created_at: u64,
    pub updated_at: u64,
    pub view_count: u32,
    pub engagement_score: f64,

    /// Filters
    pub price_cents: Option<u64>,
    pub location: Option<String>,
    pub status: String,

    /// Processed terms for search
    pub processed_terms: Vec<String>,
    pub term_frequencies: HashMap<String, u32>,
}

/// Search query with filters
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SearchQuery {
    /// Raw query string
    pub query: String,

    /// Filters
    pub filters: SearchFilters,

    /// Pagination
    pub offset: u32,
    pub limit: u32,

    /// Sorting
    pub sort_by: SortOption,
    pub sort_order: SortOrder,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SearchFilters {
    pub entity_types: Option<Vec<EntityType>>,
    pub price_min: Option<u64>,
    pub price_max: Option<u64>,
    pub categories: Option<Vec<String>>,
    pub min_matl_score: Option<f64>,
    pub created_after: Option<u64>,
    pub created_before: Option<u64>,
    pub location: Option<String>,
    pub status: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "PascalCase")]
pub enum SortOption {
    Relevance,
    Price,
    Date,
    Trust,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "PascalCase")]
pub enum SortOrder {
    Asc,
    Desc,
}

/// Search result for a single entity
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SearchResult {
    /// Matched entity
    pub entity_hash: ActionHash,
    pub entity_type: EntityType,

    /// Score breakdown
    pub total_score: f64,
    pub relevance_score: f64,
    pub trust_score: f64,
    pub recency_score: f64,
    pub engagement_score: f64,

    /// Matched terms (for highlighting)
    pub matched_terms: Vec<String>,
    pub matched_fields: Vec<String>,

    /// Preview data
    pub preview_title: String,
    pub preview_snippet: String,

    /// Metadata
    pub creator: AgentPubKey,
    pub created_at: u64,
}

/// Search response with results and metadata
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub total_count: u32,
    pub query_time_ms: u32,
    pub suggested_queries: Vec<String>,
}

// ===== API Endpoints =====

/// Perform an advanced search across marketplace entities
///
/// Uses TF-IDF ranking combined with MATL trust scores for intelligent results.
///
/// # Example
/// ```ignore
/// let query = SearchQuery {
///     query: "macbook laptop".to_string(),
///     filters: SearchFilters {
///         price_max: Some(100000),
///         min_matl_score: Some(0.5),
///         ..Default::default()
///     },
///     offset: 0,
///     limit: 20,
///     sort_by: SortOption::Relevance,
///     sort_order: SortOrder::Desc,
/// };
///
/// let results = search(query)?;
/// ```
#[hdk_extern]
pub fn search(query: SearchQuery) -> ExternResult<SearchResponse> {
    let start_time = sys_time()?;

    // 1. Parse query
    let parser = QueryParser::new();
    let parsed_query = parser.parse(&query.query);

    // 2. Get all indexed entries (in production, this would query an index)
    let all_entries = get_all_search_entries()?;

    // 3. Filter entries based on query filters
    let filtered_entries: Vec<SearchIndexEntry> = all_entries
        .into_iter()
        .filter(|entry| apply_filters(entry, &query.filters))
        .collect();

    // 4. Calculate TF-IDF scores
    let calculator = TfIdfCalculator::new(&filtered_entries);
    let mut scored_results: Vec<(SearchIndexEntry, f64)> = filtered_entries
        .into_iter()
        .filter_map(|entry| {
            let tfidf_score = calculator.calculate_score(&entry, &parsed_query.terms);
            if tfidf_score > 0.0 {
                Some((entry, tfidf_score))
            } else {
                None
            }
        })
        .collect();

    // 5. Apply MATL-weighted ranking
    let ranker = Ranker::new();
    for (entry, tfidf_score) in scored_results.iter_mut() {
        *tfidf_score = ranker.calculate_final_score(
            *tfidf_score,
            entry.creator_matl_score,
            entry.created_at,
        );
    }

    // 6. Sort results
    scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // 7. Apply pagination
    let total_count = scored_results.len() as u32;
    let offset = query.offset as usize;
    let limit = query.limit as usize;
    let paginated_results = scored_results
        .into_iter()
        .skip(offset)
        .take(limit)
        .map(|(entry, score)| create_search_result(entry, score, &parsed_query.terms))
        .collect();

    let end_time = sys_time()?;
    let query_time_ms = ((end_time - start_time) / 1_000_000) as u32;

    Ok(SearchResponse {
        results: paginated_results,
        total_count,
        query_time_ms,
        suggested_queries: generate_suggestions(&query.query),
    })
}

/// Build search index for an entity
///
/// This creates or updates the search index for a marketplace entity.
/// Called by other zomes when creating/updating entries.
#[hdk_extern]
pub fn build_search_index(entity_data: IndexableEntity) -> ExternResult<ActionHash> {
    // Create search index entry
    let processor = TextProcessor::new();

    let all_text = format!(
        "{} {} {}",
        entity_data.title,
        entity_data.description,
        entity_data.tags.join(" ")
    );

    let processed_terms = processor.process_text(&all_text);
    let term_frequencies = calculate_term_frequencies(&processed_terms);

    let index_entry = SearchIndexEntry {
        index_id: format!("{}_{}", entity_data.entity_type as u8, entity_data.entity_hash),
        entity_type: entity_data.entity_type,
        entity_hash: entity_data.entity_hash.clone(),
        title: entity_data.title,
        description: entity_data.description,
        tags: entity_data.tags,
        category: entity_data.category,
        creator: entity_data.creator,
        creator_matl_score: entity_data.creator_matl_score,
        created_at: entity_data.created_at,
        updated_at: entity_data.updated_at,
        view_count: entity_data.view_count,
        engagement_score: entity_data.engagement_score,
        price_cents: entity_data.price_cents,
        location: entity_data.location,
        status: entity_data.status,
        processed_terms,
        term_frequencies,
    };

    // Store in DHT (create entry)
    create_entry(&EntryTypes::SearchIndex(index_entry.clone()))?;

    let index_hash = hash_entry(&index_entry)?;

    // Create links for fast retrieval
    create_link(
        entity_data.entity_hash.clone(),
        index_hash.clone(),
        LinkTypes::EntityToIndex,
        (),
    )?;

    Ok(index_hash)
}

/// Autocomplete suggestions for search
#[hdk_extern]
pub fn autocomplete(prefix: String, limit: u32) -> ExternResult<Vec<String>> {
    let processor = TextProcessor::new();
    let normalized_prefix = processor.normalize(&prefix);

    // Get all search entries
    let all_entries = get_all_search_entries()?;

    // Extract all terms that start with prefix
    let mut matching_terms: HashMap<String, u32> = HashMap::new();

    for entry in all_entries {
        for term in &entry.processed_terms {
            if term.starts_with(&normalized_prefix) {
                *matching_terms.entry(term.clone()).or_insert(0) += 1;
            }
        }
    }

    // Sort by frequency and take top N
    let mut suggestions: Vec<(String, u32)> = matching_terms.into_iter().collect();
    suggestions.sort_by(|a, b| b.1.cmp(&a.1));

    Ok(suggestions
        .into_iter()
        .take(limit as usize)
        .map(|(term, _)| term)
        .collect())
}

// ===== Helper Types =====

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IndexableEntity {
    pub entity_type: EntityType,
    pub entity_hash: ActionHash,
    pub title: String,
    pub description: String,
    pub tags: Vec<String>,
    pub category: String,
    pub creator: AgentPubKey,
    pub creator_matl_score: f64,
    pub created_at: u64,
    pub updated_at: u64,
    pub view_count: u32,
    pub engagement_score: f64,
    pub price_cents: Option<u64>,
    pub location: Option<String>,
    pub status: String,
}

// These would be actual entry/link types in production
#[derive(Serialize, Deserialize, Debug, Clone)]
enum EntryTypes {
    SearchIndex(SearchIndexEntry),
}

#[derive(Clone)]
enum LinkTypes {
    EntityToIndex,
}

// ===== Helper Functions =====

fn get_all_search_entries() -> ExternResult<Vec<SearchIndexEntry>> {
    // In production, this would query the DHT for all SearchIndex entries
    // For now, return empty vec (would be populated by build_search_index calls)
    Ok(Vec::new())
}

fn apply_filters(entry: &SearchIndexEntry, filters: &SearchFilters) -> bool {
    // Entity type filter
    if let Some(ref types) = filters.entity_types {
        if !types.contains(&entry.entity_type) {
            return false;
        }
    }

    // Price filters
    if let Some(price) = entry.price_cents {
        if let Some(min) = filters.price_min {
            if price < min {
                return false;
            }
        }
        if let Some(max) = filters.price_max {
            if price > max {
                return false;
            }
        }
    }

    // Category filter
    if let Some(ref categories) = filters.categories {
        if !categories.contains(&entry.category) {
            return false;
        }
    }

    // MATL score filter
    if let Some(min_matl) = filters.min_matl_score {
        if entry.creator_matl_score < min_matl {
            return false;
        }
    }

    // Time filters
    if let Some(after) = filters.created_after {
        if entry.created_at < after {
            return false;
        }
    }
    if let Some(before) = filters.created_before {
        if entry.created_at > before {
            return false;
        }
    }

    // Location filter
    if let Some(ref location) = filters.location {
        if let Some(ref entry_location) = entry.location {
            if !entry_location.to_lowercase().contains(&location.to_lowercase()) {
                return false;
            }
        } else {
            return false;
        }
    }

    // Status filter
    if let Some(ref status) = filters.status {
        if &entry.status != status {
            return false;
        }
    }

    true
}

fn calculate_term_frequencies(terms: &[String]) -> HashMap<String, u32> {
    let mut frequencies = HashMap::new();
    for term in terms {
        *frequencies.entry(term.clone()).or_insert(0) += 1;
    }
    frequencies
}

fn create_search_result(
    entry: SearchIndexEntry,
    score: f64,
    query_terms: &[String],
) -> SearchResult {
    // Find matched terms
    let matched_terms: Vec<String> = query_terms
        .iter()
        .filter(|term| entry.processed_terms.contains(term))
        .cloned()
        .collect();

    // Create preview snippet with highlighting
    let preview_snippet = create_highlighted_snippet(&entry.description, &matched_terms);

    // Determine matched fields
    let mut matched_fields = Vec::new();
    if query_terms.iter().any(|term| entry.title.to_lowercase().contains(term)) {
        matched_fields.push("title".to_string());
    }
    if query_terms.iter().any(|term| entry.description.to_lowercase().contains(term)) {
        matched_fields.push("description".to_string());
    }
    if query_terms.iter().any(|term| entry.tags.iter().any(|tag| tag.to_lowercase().contains(term))) {
        matched_fields.push("tags".to_string());
    }

    // Calculate score components (simplified - real ranker does this)
    let relevance_score = score * 0.6;
    let trust_score = entry.creator_matl_score * 0.3;
    let recency_score = calculate_recency_score(entry.created_at) * 0.1;
    let engagement_score = entry.engagement_score;

    SearchResult {
        entity_hash: entry.entity_hash,
        entity_type: entry.entity_type,
        total_score: score,
        relevance_score,
        trust_score,
        recency_score,
        engagement_score,
        matched_terms,
        matched_fields,
        preview_title: entry.title.clone(),
        preview_snippet,
        creator: entry.creator,
        created_at: entry.created_at,
    }
}

fn create_highlighted_snippet(text: &str, terms: &[String]) -> String {
    // Simple highlighting - wrap matched terms in <b> tags
    let mut snippet = text.to_string();
    for term in terms {
        let pattern = regex::escape(term);
        let re = regex::Regex::new(&format!(r"(?i)\b{}\b", pattern)).unwrap();
        snippet = re.replace_all(&snippet, |caps: &regex::Captures| {
            format!("<b>{}</b>", &caps[0])
        }).to_string();
    }

    // Truncate to 200 characters
    if snippet.len() > 200 {
        snippet = format!("{}...", &snippet[..200]);
    }

    snippet
}

fn calculate_recency_score(created_at: u64) -> f64 {
    let now = sys_time().unwrap_or(0);
    let age_days = ((now - created_at) / (86400 * 1_000_000_000)) as f64;

    // Exponential decay: e^(-age/30)
    // Recent listings (0-7 days) get ~0.8-1.0
    // Month-old listings get ~0.37
    // 3-month-old listings get ~0.05
    (-age_days / 30.0).exp()
}

fn generate_suggestions(query: &str) -> Vec<String> {
    // Simple suggestions - in production, this would use query logs
    let variations = vec![
        format!("{} deals", query),
        format!("{} reviews", query),
        format!("best {}", query),
    ];

    variations
}

// Placeholder for hash_entry
fn hash_entry<T: Serialize>(_entry: &T) -> ExternResult<ActionHash> {
    // In production, this would properly hash the entry
    Ok(ActionHash::from_raw_39(vec![0; 39]).unwrap())
}
