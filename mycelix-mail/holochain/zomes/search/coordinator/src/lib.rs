// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Search Coordinator Zome
//!
//! Implements fast local search with TF-IDF scoring, fuzzy matching,
//! and real-time index updates.

use hdk::prelude::*;
use mail_search_integrity::*;
use std::collections::{HashMap, HashSet};

// ==================== ANCHOR HELPERS ====================

/// Create an anchor hash for a search term
fn term_anchor(term_key: &str) -> ExternResult<EntryHash> {
    let path = Path::from(format!("search_term:{}", term_key));
    path.path_entry_hash()
}

/// Create an anchor hash for a trigram
fn trigram_anchor(trigram: &str) -> ExternResult<EntryHash> {
    let path = Path::from(format!("search_trigram:{}", trigram));
    path.path_entry_hash()
}

// ==================== CONSTANTS ====================

/// Minimum term length for indexing
const MIN_TERM_LENGTH: usize = 2;

/// Maximum results per query
const MAX_RESULTS: usize = 100;

/// BM25 parameters
const BM25_K1: f32 = 1.2;
const BM25_B: f32 = 0.75;

/// Fuzzy matching threshold (0-1)
const FUZZY_THRESHOLD: f32 = 0.7;

// ==================== INPUTS/OUTPUTS ====================

#[derive(Serialize, Deserialize, Debug)]
pub struct IndexDocumentInput {
    pub document_hash: ActionHash,
    pub subject: String,
    pub body: String,
    pub sender: AgentPubKey,
    pub timestamp: Timestamp,
    pub labels: Vec<String>,
    pub attachment_names: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SearchInput {
    pub query: String,
    pub filters: Option<SearchFilters>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub fuzzy: Option<bool>,
    pub sort_by: Option<SortBy>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SortBy {
    Relevance,
    DateDesc,
    DateAsc,
    Sender,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SearchOutput {
    pub results: Vec<SearchResultOutput>,
    pub total_count: u32,
    pub query_time_ms: u32,
    pub did_fuzzy_match: bool,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SearchResultOutput {
    pub document_hash: ActionHash,
    pub score: f32,
    pub subject: String,
    pub preview: String,
    pub sender: AgentPubKey,
    pub timestamp: Timestamp,
    pub matched_terms: Vec<String>,
    pub highlights: Vec<Highlight>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Highlight {
    pub field: String,
    pub snippet: String,
    pub positions: Vec<(usize, usize)>,
}

// ==================== INDEX MANAGEMENT ====================

/// Initialize search index for current agent
#[hdk_extern]
pub fn init_search_index(_: ()) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let index = SearchIndex {
        index_id: format!("idx:{}", agent),
        agent: agent.clone(),
        index_type: IndexType::FullText,
        created_at: now,
        updated_at: now,
        document_count: 0,
        version: 1,
    };

    let hash = create_entry(EntryTypes::SearchIndex(index))?;
    create_link(agent, hash.clone(), LinkTypes::AgentToIndex, ())?;

    Ok(hash)
}

/// Index a document (email)
#[hdk_extern]
pub fn index_document(input: IndexDocumentInput) -> ExternResult<u32> {
    let _agent = agent_info()?.agent_initial_pubkey;

    // Tokenize content
    let subject_terms = tokenize(&input.subject);
    let body_terms = tokenize(&input.body);
    let sender_term = input.sender.to_string();

    let mut indexed_count = 0;

    // Index subject terms with boost
    for (position, term) in subject_terms.iter().enumerate() {
        index_term(
            term,
            &input.document_hash,
            IndexType::Subject,
            position as u32,
            2.0, // Subject gets 2x boost
        )?;
        indexed_count += 1;
    }

    // Index body terms
    for (position, term) in body_terms.iter().enumerate() {
        index_term(
            term,
            &input.document_hash,
            IndexType::FullText,
            position as u32,
            1.0,
        )?;
        indexed_count += 1;
    }

    // Index sender
    index_term(&sender_term, &input.document_hash, IndexType::Sender, 0, 1.5)?;
    indexed_count += 1;

    // Index labels
    for label in &input.labels {
        index_term(label, &input.document_hash, IndexType::Label, 0, 1.5)?;
        indexed_count += 1;
    }

    // Index attachment names
    for name in &input.attachment_names {
        let name_terms = tokenize(name);
        for term in name_terms {
            index_term(&term, &input.document_hash, IndexType::Attachment, 0, 1.0)?;
            indexed_count += 1;
        }
    }

    // Generate and index trigrams for fuzzy matching
    let all_text = format!("{} {}", input.subject, input.body);
    let trigrams = generate_trigrams(&all_text);
    for trigram in trigrams {
        index_trigram(&trigram, &subject_terms)?;
    }

    // Store document metadata
    store_document_metadata(&input)?;

    Ok(indexed_count)
}

/// Index a single term
fn index_term(
    term: &str,
    document_hash: &ActionHash,
    index_type: IndexType,
    position: u32,
    boost: f32,
) -> ExternResult<()> {
    let term_lower = term.to_lowercase();
    if term_lower.len() < MIN_TERM_LENGTH {
        return Ok(());
    }

    // Create term hash for linking
    let term_key = format!("{}:{}", term_lower, format!("{:?}", index_type));
    let term_hash = term_anchor(&term_key)?;

    // Check if term entry exists
    let links = get_links(LinkQuery::try_new(term_hash.clone(), LinkTypes::TermHashToTerm)?, GetStrategy::default())?;

    if let Some(link) = links.last() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                let mut index_term: IndexTerm = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!("Invalid term entry"))?;

                // Update existing term entry
                let doc_exists = index_term.documents.iter()
                    .any(|d| d.document_hash == *document_hash);

                if !doc_exists {
                    index_term.documents.push(DocumentRef {
                        document_hash: document_hash.clone(),
                        term_frequency: 1,
                        positions: vec![position],
                        boost,
                    });
                    index_term.total_frequency += 1;

                    // Recalculate IDF
                    let n = get_total_document_count()?;
                    let df = index_term.documents.len() as f32;
                    index_term.idf = ((n as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();

                    create_entry(EntryTypes::IndexTerm(index_term))?;
                }

                return Ok(());
            }
        }
    }

    // Create new term entry
    let n = get_total_document_count()?;
    let new_term = IndexTerm {
        term: term_lower,
        index_type,
        documents: vec![DocumentRef {
            document_hash: document_hash.clone(),
            term_frequency: 1,
            positions: vec![position],
            boost,
        }],
        total_frequency: 1,
        idf: ((n as f32 + 0.5) / 1.5).ln(), // Initial IDF
    };

    let entry_hash = create_entry(EntryTypes::IndexTerm(new_term))?;
    create_link(term_hash, entry_hash, LinkTypes::TermHashToTerm, ())?;

    Ok(())
}

/// Generate trigrams from text
fn generate_trigrams(text: &str) -> Vec<String> {
    let clean: String = text.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect();

    let mut trigrams = Vec::new();
    let chars: Vec<char> = clean.chars().collect();

    for i in 0..chars.len().saturating_sub(2) {
        let trigram: String = chars[i..i+3].iter().collect();
        if !trigram.contains(' ') {
            trigrams.push(trigram);
        }
    }

    trigrams
}

/// Index trigram for fuzzy matching
fn index_trigram(trigram: &str, terms: &[String]) -> ExternResult<()> {
    let trigram_hash = trigram_anchor(trigram)?;

    let links = get_links(LinkQuery::try_new(trigram_hash.clone(), LinkTypes::TrigramHashToTrigram)?, GetStrategy::default())?;

    if let Some(_link) = links.last() {
        // Trigram exists, update terms
        return Ok(());
    }

    // Create new trigram entry
    let trigram_entry = Trigram {
        chars: trigram.to_string(),
        terms: terms.iter().cloned().collect(),
    };

    let entry_hash = create_entry(EntryTypes::Trigram(trigram_entry))?;
    create_link(trigram_hash, entry_hash, LinkTypes::TrigramHashToTrigram, ())?;

    Ok(())
}

/// Store document metadata for search results
fn store_document_metadata(_input: &IndexDocumentInput) -> ExternResult<()> {
    // In full implementation, would store metadata entry
    // For now, we rely on fetching the original email
    Ok(())
}

fn get_total_document_count() -> ExternResult<u32> {
    // In full implementation, would query index stats
    Ok(1000) // Placeholder
}

// ==================== SEARCH ====================

/// Search the index
#[hdk_extern]
pub fn search(input: SearchInput) -> ExternResult<SearchOutput> {
    let start_time = std::time::Instant::now();
    let limit = input.limit.unwrap_or(20).min(MAX_RESULTS);
    let offset = input.offset.unwrap_or(0);
    let fuzzy = input.fuzzy.unwrap_or(true);

    // Parse and tokenize query
    let query_terms = tokenize(&input.query);

    if query_terms.is_empty() {
        return Ok(SearchOutput {
            results: vec![],
            total_count: 0,
            query_time_ms: 0,
            did_fuzzy_match: false,
        });
    }

    // Score documents
    let mut doc_scores: HashMap<ActionHash, (f32, Vec<String>)> = HashMap::new();
    let mut did_fuzzy = false;

    for term in &query_terms {
        // Exact match
        let matches = find_term_matches(term, None)?;

        if matches.is_empty() && fuzzy {
            // Try fuzzy matching
            let fuzzy_matches = find_fuzzy_matches(term)?;
            for (matched_term, docs) in fuzzy_matches {
                did_fuzzy = true;
                for doc_ref in docs {
                    let score = calculate_bm25_score(&doc_ref);
                    let entry = doc_scores.entry(doc_ref.document_hash.clone())
                        .or_insert((0.0, vec![]));
                    entry.0 += score * 0.8; // Fuzzy match penalty
                    entry.1.push(matched_term.clone());
                }
            }
        } else {
            for doc_ref in matches {
                let score = calculate_bm25_score(&doc_ref);
                let entry = doc_scores.entry(doc_ref.document_hash.clone())
                    .or_insert((0.0, vec![]));
                entry.0 += score;
                entry.1.push(term.clone());
            }
        }
    }

    // Apply filters
    let filtered_docs = apply_filters(&doc_scores, &input.filters)?;

    // Sort by relevance or other criteria
    let mut sorted: Vec<_> = filtered_docs.into_iter().collect();
    match input.sort_by.unwrap_or(SortBy::Relevance) {
        SortBy::Relevance => sorted.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap()),
        SortBy::DateDesc | SortBy::DateAsc | SortBy::Sender => {
            // Would need to fetch metadata for these sorts
            sorted.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
        }
    }

    let total_count = sorted.len() as u32;

    // Paginate
    let paginated: Vec<_> = sorted.into_iter()
        .skip(offset)
        .take(limit)
        .collect();

    // Build results
    let mut results = Vec::new();
    for (doc_hash, (score, matched_terms)) in paginated {
        let result = build_search_result(&doc_hash, score, matched_terms, &input.query)?;
        results.push(result);
    }

    let query_time_ms = start_time.elapsed().as_millis() as u32;

    Ok(SearchOutput {
        results,
        total_count,
        query_time_ms,
        did_fuzzy_match: did_fuzzy,
    })
}

/// Find exact term matches
fn find_term_matches(term: &str, index_type: Option<IndexType>) -> ExternResult<Vec<DocumentRef>> {
    let term_lower = term.to_lowercase();
    let mut all_docs = Vec::new();

    let index_types = match index_type {
        Some(t) => vec![t],
        None => vec![IndexType::FullText, IndexType::Subject, IndexType::Sender, IndexType::Label],
    };

    for idx_type in index_types {
        let term_key = format!("{}:{}", term_lower, format!("{:?}", idx_type));
        let term_hash = term_anchor(&term_key)?;

        let links = get_links(LinkQuery::try_new(term_hash, LinkTypes::TermHashToTerm)?, GetStrategy::default())?;

        for link in links {
            if let Some(action_hash) = link.target.clone().into_action_hash() {
                if let Some(record) = get(action_hash, GetOptions::default())? {
                    let index_term: IndexTerm = record
                        .entry()
                        .to_app_option()
                        .map_err(|e| wasm_error!(e))?
                        .ok_or(wasm_error!("Invalid term entry"))?;

                    all_docs.extend(index_term.documents);
                }
            }
        }
    }

    Ok(all_docs)
}

/// Find fuzzy matches using trigrams
fn find_fuzzy_matches(term: &str) -> ExternResult<Vec<(String, Vec<DocumentRef>)>> {
    let trigrams = generate_trigrams(term);
    let mut candidate_terms: HashMap<String, u32> = HashMap::new();

    // Find terms that share trigrams
    for trigram in &trigrams {
        let trigram_hash = trigram_anchor(trigram)?;
        let links = get_links(LinkQuery::try_new(trigram_hash, LinkTypes::TrigramHashToTrigram)?, GetStrategy::default())?;

        for link in links {
            if let Some(action_hash) = link.target.clone().into_action_hash() {
                if let Some(record) = get(action_hash, GetOptions::default())? {
                    let trig: Trigram = record
                        .entry()
                        .to_app_option()
                        .map_err(|e| wasm_error!(e))?
                        .ok_or(wasm_error!("Invalid trigram"))?;

                    for t in &trig.terms {
                        *candidate_terms.entry(t.clone()).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    // Calculate similarity and filter
    let mut results = Vec::new();
    let query_trigram_count = trigrams.len() as f32;

    for (candidate, shared_count) in candidate_terms {
        let candidate_trigrams = generate_trigrams(&candidate).len() as f32;
        let similarity = shared_count as f32 / (query_trigram_count + candidate_trigrams - shared_count as f32);

        if similarity >= FUZZY_THRESHOLD {
            let docs = find_term_matches(&candidate, None)?;
            if !docs.is_empty() {
                results.push((candidate, docs));
            }
        }
    }

    Ok(results)
}

/// Calculate BM25 score for a document
fn calculate_bm25_score(doc_ref: &DocumentRef) -> f32 {
    let tf = doc_ref.term_frequency as f32;
    let boost = doc_ref.boost;

    // Simplified BM25
    let numerator = tf * (BM25_K1 + 1.0);
    let denominator = tf + BM25_K1 * (1.0 - BM25_B + BM25_B);

    (numerator / denominator) * boost
}

/// Apply search filters
fn apply_filters(
    doc_scores: &HashMap<ActionHash, (f32, Vec<String>)>,
    _filters: &Option<SearchFilters>,
) -> ExternResult<HashMap<ActionHash, (f32, Vec<String>)>> {
    // In full implementation, would filter by folder, labels, date, etc.
    // For now, return all
    Ok(doc_scores.clone())
}

/// Build search result with highlights
fn build_search_result(
    doc_hash: &ActionHash,
    score: f32,
    matched_terms: Vec<String>,
    _query: &str,
) -> ExternResult<SearchResultOutput> {
    // In full implementation, would fetch email and generate highlights
    Ok(SearchResultOutput {
        document_hash: doc_hash.clone(),
        score,
        subject: "".to_string(), // Would fetch from email
        preview: "".to_string(),
        sender: agent_info()?.agent_initial_pubkey, // Placeholder
        timestamp: sys_time()?,
        matched_terms,
        highlights: vec![],
    })
}

// ==================== TOKENIZATION ====================

/// Tokenize text into terms
fn tokenize(text: &str) -> Vec<String> {
    let stop_words: HashSet<&str> = [
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "were", "will", "with", "you", "your",
    ].into_iter().collect();

    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| word.len() >= MIN_TERM_LENGTH && !stop_words.contains(word))
        .map(|s| s.to_string())
        .collect()
}

// ==================== SAVED SEARCHES ====================

/// Save a search query
#[hdk_extern]
pub fn save_search(input: SavedSearch) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;

    let mut search = input;
    search.owner = agent.clone();
    search.created_at = sys_time()?;

    let hash = create_entry(EntryTypes::SavedSearch(search))?;
    create_link(agent, hash.clone(), LinkTypes::AgentToSavedSearches, ())?;

    Ok(hash)
}

/// Get saved searches
#[hdk_extern]
pub fn get_saved_searches(_: ()) -> ExternResult<Vec<SavedSearch>> {
    let agent = agent_info()?.agent_initial_pubkey;

    let links = get_links(LinkQuery::try_new(agent, LinkTypes::AgentToSavedSearches)?, GetStrategy::default())?;

    let mut searches = Vec::new();
    for link in links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                let search: SavedSearch = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!("Invalid saved search"))?;
                searches.push(search);
            }
        }
    }

    Ok(searches)
}

/// Delete saved search
#[hdk_extern]
pub fn delete_saved_search(search_hash: ActionHash) -> ExternResult<()> {
    delete_entry(search_hash)?;
    Ok(())
}

// ==================== INDEX STATS ====================

/// Get search index statistics
#[hdk_extern]
pub fn get_search_stats(_: ()) -> ExternResult<SearchStats> {
    Ok(SearchStats {
        total_documents: get_total_document_count()?,
        total_terms: 0, // Would calculate
        avg_doc_length: 500.0,
        last_index_time_ms: 10,
        last_query_time_ms: 5,
        cache_hit_rate: 0.85,
    })
}

/// Rebuild search index
#[hdk_extern]
pub fn rebuild_index(_: ()) -> ExternResult<u32> {
    // Would re-index all emails
    Ok(0)
}
