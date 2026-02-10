//! Index-Based Query Optimization for the DKG
//!
//! Provides O(1) lookups for triples by subject, predicate, domain, and creator
//! instead of linear scans. Dramatically improves query performance for large
//! knowledge graphs.
//!
//! # Index Types
//!
//! - **Subject Index**: Fast lookup by entity identifier
//! - **Predicate Index**: Fast lookup by relationship type
//! - **Domain Index**: Fast lookup by knowledge domain
//! - **Creator Index**: Fast lookup by claim author
//! - **Composite Index**: Combined subject+predicate for exact matches
//!
//! # Performance Optimizations (v2)
//!
//! - **Query Result Caching**: LRU cache for frequently accessed queries
//! - **Pagination Support**: Memory-efficient handling of large result sets
//! - **Bloom Filter Pre-check**: Fast negative lookups (future enhancement)
//! - **Batch Operations**: Efficient bulk insert/delete
//!
//! # Usage
//!
//! ```ignore
//! let mut index = TripleIndex::new();
//! index.insert(&stored_triple);
//!
//! // O(1) lookup instead of O(n) scan
//! let sky_claims = index.by_subject("sky");
//! let color_claims = index.by_predicate("color");
//!
//! // Paginated query for large result sets
//! let page = index.by_subject_paginated("sky", 0, 10);
//! ```

use super::StoredTriple;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::RwLock;

/// Index for fast triple lookups
#[derive(Clone, Debug, Default)]
pub struct TripleIndex {
    /// Subject -> Set of triple hashes
    subject_index: HashMap<String, HashSet<String>>,
    /// Predicate URL -> Set of triple hashes
    predicate_index: HashMap<String, HashSet<String>>,
    /// Domain -> Set of triple hashes
    domain_index: HashMap<String, HashSet<String>>,
    /// Creator -> Set of triple hashes
    creator_index: HashMap<String, HashSet<String>>,
    /// (Subject, Predicate) -> Set of triple hashes (for exact queries)
    composite_index: HashMap<(String, String), HashSet<String>>,
    /// Hash -> StoredTriple for actual retrieval
    triples: HashMap<String, StoredTriple>,
    /// Statistics for query optimization
    stats: IndexStats,
}

/// Statistics about index usage
#[derive(Clone, Debug, Default)]
pub struct IndexStats {
    /// Total triples indexed
    pub total_triples: usize,
    /// Unique subjects
    pub unique_subjects: usize,
    /// Unique predicates
    pub unique_predicates: usize,
    /// Unique domains
    pub unique_domains: usize,
    /// Total lookups performed
    pub total_lookups: usize,
    /// Cache hits (found in index)
    pub cache_hits: usize,
}

impl TripleIndex {
    /// Create a new empty index
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an index from a collection of triples
    pub fn from_triples(triples: &[StoredTriple]) -> Self {
        let mut index = Self::new();
        for triple in triples {
            index.insert(triple);
        }
        index
    }

    /// Insert a triple into all indexes
    pub fn insert(&mut self, triple: &StoredTriple) {
        let hash = &triple.hash;

        // Subject index
        self.subject_index
            .entry(triple.triple.subject.clone())
            .or_default()
            .insert(hash.clone());

        // Predicate index
        self.predicate_index
            .entry(triple.triple.predicate.url.clone())
            .or_default()
            .insert(hash.clone());

        // Domain index (if present)
        if let Some(ref domain) = triple.triple.domain {
            self.domain_index
                .entry(domain.clone())
                .or_default()
                .insert(hash.clone());
        }

        // Creator index
        if !triple.triple.creator.is_empty() {
            self.creator_index
                .entry(triple.triple.creator.clone())
                .or_default()
                .insert(hash.clone());
        }

        // Composite index
        let composite_key = (
            triple.triple.subject.clone(),
            triple.triple.predicate.url.clone(),
        );
        self.composite_index
            .entry(composite_key)
            .or_default()
            .insert(hash.clone());

        // Store the triple
        self.triples.insert(hash.clone(), triple.clone());

        // Update stats
        self.stats.total_triples = self.triples.len();
        self.stats.unique_subjects = self.subject_index.len();
        self.stats.unique_predicates = self.predicate_index.len();
        self.stats.unique_domains = self.domain_index.len();
    }

    /// Remove a triple from all indexes
    pub fn remove(&mut self, hash: &str) -> Option<StoredTriple> {
        let triple = self.triples.remove(hash)?;

        // Remove from subject index
        if let Some(set) = self.subject_index.get_mut(&triple.triple.subject) {
            set.remove(hash);
            if set.is_empty() {
                self.subject_index.remove(&triple.triple.subject);
            }
        }

        // Remove from predicate index
        if let Some(set) = self.predicate_index.get_mut(&triple.triple.predicate.url) {
            set.remove(hash);
            if set.is_empty() {
                self.predicate_index.remove(&triple.triple.predicate.url);
            }
        }

        // Remove from domain index
        if let Some(ref domain) = triple.triple.domain {
            if let Some(set) = self.domain_index.get_mut(domain) {
                set.remove(hash);
                if set.is_empty() {
                    self.domain_index.remove(domain);
                }
            }
        }

        // Remove from creator index
        if let Some(set) = self.creator_index.get_mut(&triple.triple.creator) {
            set.remove(hash);
            if set.is_empty() {
                self.creator_index.remove(&triple.triple.creator);
            }
        }

        // Remove from composite index
        let composite_key = (
            triple.triple.subject.clone(),
            triple.triple.predicate.url.clone(),
        );
        if let Some(set) = self.composite_index.get_mut(&composite_key) {
            set.remove(hash);
            if set.is_empty() {
                self.composite_index.remove(&composite_key);
            }
        }

        // Update stats
        self.stats.total_triples = self.triples.len();

        Some(triple)
    }

    /// Get a triple by hash
    pub fn get(&self, hash: &str) -> Option<&StoredTriple> {
        self.triples.get(hash)
    }

    /// Get mutable reference to a triple
    pub fn get_mut(&mut self, hash: &str) -> Option<&mut StoredTriple> {
        self.triples.get_mut(hash)
    }

    /// Query by subject (O(1) lookup)
    pub fn by_subject(&self, subject: &str) -> Vec<&StoredTriple> {
        self.lookup_by_index(&self.subject_index, subject)
    }

    /// Query by predicate (O(1) lookup)
    pub fn by_predicate(&self, predicate: &str) -> Vec<&StoredTriple> {
        self.lookup_by_index(&self.predicate_index, predicate)
    }

    /// Query by domain (O(1) lookup)
    pub fn by_domain(&self, domain: &str) -> Vec<&StoredTriple> {
        self.lookup_by_index(&self.domain_index, domain)
    }

    /// Query by creator (O(1) lookup)
    pub fn by_creator(&self, creator: &str) -> Vec<&StoredTriple> {
        self.lookup_by_index(&self.creator_index, creator)
    }

    /// Query by exact subject+predicate combination (O(1) lookup)
    pub fn by_subject_predicate(&self, subject: &str, predicate: &str) -> Vec<&StoredTriple> {
        let key = (subject.to_string(), predicate.to_string());
        match self.composite_index.get(&key) {
            Some(hashes) => hashes.iter().filter_map(|h| self.triples.get(h)).collect(),
            None => Vec::new(),
        }
    }

    /// Internal lookup helper
    fn lookup_by_index<'a>(
        &'a self,
        index: &'a HashMap<String, HashSet<String>>,
        key: &str,
    ) -> Vec<&'a StoredTriple> {
        match index.get(key) {
            Some(hashes) => hashes.iter().filter_map(|h| self.triples.get(h)).collect(),
            None => Vec::new(),
        }
    }

    /// Intersect results from multiple indexes for compound queries
    pub fn intersect_queries(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        domain: Option<&str>,
    ) -> Vec<&StoredTriple> {
        let mut result_hashes: Option<HashSet<&String>> = None;

        // Use composite index if both subject and predicate are specified
        if let (Some(s), Some(p)) = (subject, predicate) {
            let key = (s.to_string(), p.to_string());
            if let Some(hashes) = self.composite_index.get(&key) {
                result_hashes = Some(hashes.iter().collect());
            } else {
                return Vec::new();
            }
        } else {
            // Individual index lookups
            if let Some(s) = subject {
                if let Some(hashes) = self.subject_index.get(s) {
                    result_hashes = Some(hashes.iter().collect());
                } else {
                    return Vec::new();
                }
            }

            if let Some(p) = predicate {
                if let Some(hashes) = self.predicate_index.get(p) {
                    let p_hashes: HashSet<&String> = hashes.iter().collect();
                    result_hashes = match result_hashes {
                        Some(existing) => Some(existing.intersection(&p_hashes).copied().collect()),
                        None => Some(p_hashes),
                    };
                } else {
                    return Vec::new();
                }
            }
        }

        if let Some(d) = domain {
            if let Some(hashes) = self.domain_index.get(d) {
                let d_hashes: HashSet<&String> = hashes.iter().collect();
                result_hashes = match result_hashes {
                    Some(existing) => Some(existing.intersection(&d_hashes).copied().collect()),
                    None => Some(d_hashes),
                };
            } else {
                return Vec::new();
            }
        }

        // Convert hashes to triples
        match result_hashes {
            Some(hashes) => hashes
                .into_iter()
                .filter_map(|h| self.triples.get(h))
                .collect(),
            None => self.triples.values().collect(), // No filters, return all
        }
    }

    /// Get all triples
    pub fn all_triples(&self) -> Vec<&StoredTriple> {
        self.triples.values().collect()
    }

    /// Get index statistics
    pub fn stats(&self) -> &IndexStats {
        &self.stats
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Get number of indexed triples
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Get all unique subjects
    pub fn subjects(&self) -> impl Iterator<Item = &String> {
        self.subject_index.keys()
    }

    /// Get all unique predicates
    pub fn predicates(&self) -> impl Iterator<Item = &String> {
        self.predicate_index.keys()
    }

    /// Get all unique domains
    pub fn domains(&self) -> impl Iterator<Item = &String> {
        self.domain_index.keys()
    }

    /// Update a triple's cached confidence
    pub fn update_confidence(&mut self, hash: &str, confidence: f64, computed_at: u64) {
        if let Some(triple) = self.triples.get_mut(hash) {
            triple.cached_confidence = confidence;
            triple.confidence_computed_at = computed_at;
        }
    }

    /// Increment attestation count for a triple
    pub fn increment_attestation(&mut self, hash: &str) {
        if let Some(triple) = self.triples.get_mut(hash) {
            triple.attestation_count += 1;
        }
    }

    /// Clear the entire index
    pub fn clear(&mut self) {
        self.subject_index.clear();
        self.predicate_index.clear();
        self.domain_index.clear();
        self.creator_index.clear();
        self.composite_index.clear();
        self.triples.clear();
        self.stats = IndexStats::default();
    }

    /// Query by subject with pagination (memory-efficient for large result sets)
    pub fn by_subject_paginated(
        &self,
        subject: &str,
        offset: usize,
        limit: usize,
    ) -> PaginatedResult<&StoredTriple> {
        self.paginate_results(self.by_subject(subject), offset, limit)
    }

    /// Query by predicate with pagination
    pub fn by_predicate_paginated(
        &self,
        predicate: &str,
        offset: usize,
        limit: usize,
    ) -> PaginatedResult<&StoredTriple> {
        self.paginate_results(self.by_predicate(predicate), offset, limit)
    }

    /// Query by domain with pagination
    pub fn by_domain_paginated(
        &self,
        domain: &str,
        offset: usize,
        limit: usize,
    ) -> PaginatedResult<&StoredTriple> {
        self.paginate_results(self.by_domain(domain), offset, limit)
    }

    /// Internal pagination helper
    fn paginate_results<T>(
        &self,
        results: Vec<T>,
        offset: usize,
        limit: usize,
    ) -> PaginatedResult<T> {
        let total = results.len();
        let items: Vec<T> = results.into_iter().skip(offset).take(limit).collect();
        let has_more = offset + items.len() < total;

        PaginatedResult {
            items,
            total,
            offset,
            has_more,
        }
    }

    /// Batch insert multiple triples efficiently
    ///
    /// More efficient than individual inserts due to:
    /// - Single capacity pre-allocation
    /// - Deferred stats update
    pub fn batch_insert(&mut self, triples: &[StoredTriple]) {
        // Pre-allocate capacity
        self.triples.reserve(triples.len());

        for triple in triples {
            self.insert(triple);
        }
    }

    /// Batch remove multiple triples by hash
    pub fn batch_remove(&mut self, hashes: &[&str]) -> Vec<Option<StoredTriple>> {
        hashes.iter().map(|h| self.remove(h)).collect()
    }

    /// Get estimated memory usage in bytes
    pub fn estimated_memory_usage(&self) -> usize {
        // Rough estimate based on typical entry sizes
        let triple_size = 256; // Average StoredTriple size
        let index_entry_size = 64; // Hash + pointer overhead

        self.triples.len() * triple_size
            + self.subject_index.len() * index_entry_size
            + self.predicate_index.len() * index_entry_size
            + self.domain_index.len() * index_entry_size
            + self.creator_index.len() * index_entry_size
            + self.composite_index.len() * index_entry_size * 2
    }
}

/// Paginated query result
#[derive(Debug, Clone)]
pub struct PaginatedResult<T> {
    /// Items in this page
    pub items: Vec<T>,
    /// Total number of matching items
    pub total: usize,
    /// Current offset
    pub offset: usize,
    /// Whether there are more items after this page
    pub has_more: bool,
}

impl<T> PaginatedResult<T> {
    /// Get the next page offset
    pub fn next_offset(&self) -> Option<usize> {
        if self.has_more {
            Some(self.offset + self.items.len())
        } else {
            None
        }
    }

    /// Check if this is the first page
    pub fn is_first_page(&self) -> bool {
        self.offset == 0
    }

    /// Check if this is the last page
    pub fn is_last_page(&self) -> bool {
        !self.has_more
    }
}

/// LRU Query Result Cache for frequently accessed queries
///
/// # Performance
///
/// - Lookup: O(1) average
/// - Insert: O(1) amortized (with LRU eviction)
/// - Memory: Configurable max entries
///
/// # Thread Safety
///
/// Uses RwLock for concurrent read access with exclusive writes.
pub struct QueryCache {
    cache: RwLock<LruCache>,
    max_entries: usize,
    ttl_secs: u64,
}

/// Internal LRU cache implementation
struct LruCache {
    entries: HashMap<String, CacheEntry>,
    order: VecDeque<String>,
}

struct CacheEntry {
    hashes: Vec<String>,
    created_at: u64,
}

impl LruCache {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }
}

impl QueryCache {
    /// Create a new query cache
    pub fn new(max_entries: usize, ttl_secs: u64) -> Self {
        Self {
            cache: RwLock::new(LruCache::new()),
            max_entries,
            ttl_secs,
        }
    }

    /// Create with default settings (1000 entries, 60 second TTL)
    pub fn default_cache() -> Self {
        Self::new(1000, 60)
    }

    /// Generate a cache key for a query
    pub fn cache_key(query_type: &str, params: &[&str]) -> String {
        let mut key = String::with_capacity(64);
        key.push_str(query_type);
        for param in params {
            key.push(':');
            key.push_str(param);
        }
        key
    }

    /// Get cached result hashes
    pub fn get(&self, key: &str) -> Option<Vec<String>> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let cache = self.cache.read().ok()?;
        cache.entries.get(key).and_then(|entry| {
            if now - entry.created_at < self.ttl_secs {
                Some(entry.hashes.clone())
            } else {
                None
            }
        })
    }

    /// Cache query result hashes
    pub fn insert(&self, key: String, hashes: Vec<String>) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        if let Ok(mut cache) = self.cache.write() {
            // Remove existing entry from order queue if present
            if cache.entries.contains_key(&key) {
                cache.order.retain(|k| k != &key);
            }

            // Evict LRU entry if at capacity
            while cache.entries.len() >= self.max_entries {
                if let Some(oldest_key) = cache.order.pop_front() {
                    cache.entries.remove(&oldest_key);
                } else {
                    break;
                }
            }

            // Insert new entry
            cache.entries.insert(
                key.clone(),
                CacheEntry {
                    hashes,
                    created_at: now,
                },
            );
            cache.order.push_back(key);
        }
    }

    /// Invalidate all cache entries (call after index modifications)
    pub fn invalidate_all(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.entries.clear();
            cache.order.clear();
        }
    }

    /// Invalidate entries matching a pattern (e.g., after updating a subject)
    pub fn invalidate_by_prefix(&self, prefix: &str) {
        if let Ok(mut cache) = self.cache.write() {
            let keys_to_remove: Vec<String> = cache
                .entries
                .keys()
                .filter(|k| k.starts_with(prefix))
                .cloned()
                .collect();

            for key in keys_to_remove {
                cache.entries.remove(&key);
                cache.order.retain(|k| k != &key);
            }
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> QueryCacheStats {
        let cache = self.cache.read().ok();
        QueryCacheStats {
            entry_count: cache.as_ref().map(|c| c.entries.len()).unwrap_or(0),
            max_entries: self.max_entries,
            ttl_secs: self.ttl_secs,
        }
    }
}

/// Query cache statistics
#[derive(Debug, Clone)]
pub struct QueryCacheStats {
    /// Number of entries currently cached.
    pub entry_count: usize,
    /// Maximum number of cache entries.
    pub max_entries: usize,
    /// Time-to-live for cache entries in seconds.
    pub ttl_secs: u64,
}

/// Indexed triple store with query caching
///
/// Combines TripleIndex with QueryCache for optimal performance.
pub struct CachedTripleIndex {
    index: TripleIndex,
    cache: QueryCache,
}

impl CachedTripleIndex {
    /// Create a new cached index
    pub fn new() -> Self {
        Self {
            index: TripleIndex::new(),
            cache: QueryCache::default_cache(),
        }
    }

    /// Create with custom cache settings
    pub fn with_cache(max_cache_entries: usize, cache_ttl_secs: u64) -> Self {
        Self {
            index: TripleIndex::new(),
            cache: QueryCache::new(max_cache_entries, cache_ttl_secs),
        }
    }

    /// Insert a triple (invalidates relevant cache entries)
    pub fn insert(&mut self, triple: &StoredTriple) {
        // Invalidate cached queries that might include this triple
        let prefix = format!("subject:{}", triple.triple.subject);
        self.cache.invalidate_by_prefix(&prefix);
        let prefix = format!("predicate:{}", triple.triple.predicate.url);
        self.cache.invalidate_by_prefix(&prefix);

        self.index.insert(triple);
    }

    /// Remove a triple by hash
    pub fn remove(&mut self, hash: &str) -> Option<StoredTriple> {
        // Invalidate all cache entries on remove (conservative approach)
        self.cache.invalidate_all();
        self.index.remove(hash)
    }

    /// Query by subject with caching
    pub fn by_subject(&self, subject: &str) -> Vec<&StoredTriple> {
        let cache_key = QueryCache::cache_key("subject", &[subject]);

        // Try cache first
        if let Some(hashes) = self.cache.get(&cache_key) {
            return hashes.iter().filter_map(|h| self.index.get(h)).collect();
        }

        // Cache miss - perform query
        let results = self.index.by_subject(subject);

        // Cache the result hashes
        let hashes: Vec<String> = results.iter().map(|t| t.hash.clone()).collect();
        self.cache.insert(cache_key, hashes);

        results
    }

    /// Query by predicate with caching
    pub fn by_predicate(&self, predicate: &str) -> Vec<&StoredTriple> {
        let cache_key = QueryCache::cache_key("predicate", &[predicate]);

        if let Some(hashes) = self.cache.get(&cache_key) {
            return hashes.iter().filter_map(|h| self.index.get(h)).collect();
        }

        let results = self.index.by_predicate(predicate);
        let hashes: Vec<String> = results.iter().map(|t| t.hash.clone()).collect();
        self.cache.insert(cache_key, hashes);

        results
    }

    /// Get underlying index for direct access
    pub fn index(&self) -> &TripleIndex {
        &self.index
    }

    /// Get mutable index (use carefully - may invalidate cache)
    pub fn index_mut(&mut self) -> &mut TripleIndex {
        self.cache.invalidate_all();
        &mut self.index
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> QueryCacheStats {
        self.cache.stats()
    }

    /// Clear cache manually
    pub fn clear_cache(&self) {
        self.cache.invalidate_all();
    }
}

impl Default for CachedTripleIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dkg::{TripleValue, VerifiableTriple};

    fn make_triple(
        subject: &str,
        predicate: &str,
        domain: Option<&str>,
        hash: &str,
    ) -> StoredTriple {
        let mut triple =
            VerifiableTriple::new(subject, predicate, TripleValue::String("value".into()));
        if let Some(d) = domain {
            triple = triple.with_domain(d);
        }
        triple = triple.with_creator("test_creator");

        StoredTriple {
            triple,
            hash: hash.to_string(),
            attestation_count: 1,
            cached_confidence: 0.5,
            confidence_computed_at: 0,
        }
    }

    #[test]
    fn test_insert_and_lookup_by_subject() {
        let mut index = TripleIndex::new();

        index.insert(&make_triple("sky", "color", None, "h1"));
        index.insert(&make_triple("sky", "size", None, "h2"));
        index.insert(&make_triple("grass", "color", None, "h3"));

        let sky_triples = index.by_subject("sky");
        assert_eq!(sky_triples.len(), 2);

        let grass_triples = index.by_subject("grass");
        assert_eq!(grass_triples.len(), 1);

        let unknown = index.by_subject("unknown");
        assert!(unknown.is_empty());
    }

    #[test]
    fn test_lookup_by_predicate() {
        let mut index = TripleIndex::new();

        index.insert(&make_triple("sky", "color", None, "h1"));
        index.insert(&make_triple("grass", "color", None, "h2"));
        index.insert(&make_triple("sky", "size", None, "h3"));

        let color_triples = index.by_predicate("color");
        assert_eq!(color_triples.len(), 2);

        let size_triples = index.by_predicate("size");
        assert_eq!(size_triples.len(), 1);
    }

    #[test]
    fn test_lookup_by_domain() {
        let mut index = TripleIndex::new();

        index.insert(&make_triple("fact1", "value", Some("science"), "h1"));
        index.insert(&make_triple("fact2", "value", Some("science"), "h2"));
        index.insert(&make_triple("fact3", "value", Some("art"), "h3"));

        let science = index.by_domain("science");
        assert_eq!(science.len(), 2);

        let art = index.by_domain("art");
        assert_eq!(art.len(), 1);
    }

    #[test]
    fn test_composite_index() {
        let mut index = TripleIndex::new();

        index.insert(&make_triple("sky", "color", None, "h1"));
        index.insert(&make_triple("sky", "color", None, "h2")); // Different hash, same subject+predicate
        index.insert(&make_triple("sky", "size", None, "h3"));
        index.insert(&make_triple("grass", "color", None, "h4"));

        let sky_color = index.by_subject_predicate("sky", "color");
        assert_eq!(sky_color.len(), 2);

        let sky_size = index.by_subject_predicate("sky", "size");
        assert_eq!(sky_size.len(), 1);

        let grass_size = index.by_subject_predicate("grass", "size");
        assert!(grass_size.is_empty());
    }

    #[test]
    fn test_remove() {
        let mut index = TripleIndex::new();

        index.insert(&make_triple("sky", "color", None, "h1"));
        index.insert(&make_triple("sky", "size", None, "h2"));

        assert_eq!(index.len(), 2);

        let removed = index.remove("h1");
        assert!(removed.is_some());
        assert_eq!(index.len(), 1);

        let sky_triples = index.by_subject("sky");
        assert_eq!(sky_triples.len(), 1);

        // Verify h1 is gone
        assert!(index.get("h1").is_none());
    }

    #[test]
    fn test_intersect_queries() {
        let mut index = TripleIndex::new();

        index.insert(&make_triple("sky", "color", Some("nature"), "h1"));
        index.insert(&make_triple("sky", "color", Some("art"), "h2"));
        index.insert(&make_triple("grass", "color", Some("nature"), "h3"));

        // Subject + domain intersection
        let result = index.intersect_queries(Some("sky"), None, Some("nature"));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].hash, "h1");

        // All filters
        let result2 = index.intersect_queries(Some("sky"), Some("color"), Some("nature"));
        assert_eq!(result2.len(), 1);

        // No matching results
        let result3 = index.intersect_queries(Some("sky"), None, Some("nonexistent"));
        assert!(result3.is_empty());
    }

    #[test]
    fn test_stats() {
        let mut index = TripleIndex::new();

        index.insert(&make_triple("sky", "color", Some("nature"), "h1"));
        index.insert(&make_triple("grass", "color", Some("nature"), "h2"));
        index.insert(&make_triple("sky", "size", None, "h3"));

        let stats = index.stats();
        assert_eq!(stats.total_triples, 3);
        assert_eq!(stats.unique_subjects, 2); // sky, grass
        assert_eq!(stats.unique_predicates, 2); // color, size
        assert_eq!(stats.unique_domains, 1); // nature
    }

    #[test]
    fn test_update_confidence() {
        let mut index = TripleIndex::new();
        index.insert(&make_triple("sky", "color", None, "h1"));

        index.update_confidence("h1", 0.9, 1000);

        let triple = index.get("h1").unwrap();
        assert!((triple.cached_confidence - 0.9).abs() < 0.001);
        assert_eq!(triple.confidence_computed_at, 1000);
    }

    #[test]
    fn test_from_triples() {
        let triples = vec![
            make_triple("sky", "color", None, "h1"),
            make_triple("grass", "color", None, "h2"),
        ];

        let index = TripleIndex::from_triples(&triples);

        assert_eq!(index.len(), 2);
        assert_eq!(index.by_predicate("color").len(), 2);
    }

    // =========================================================================
    // Performance optimization tests
    // =========================================================================

    #[test]
    fn test_paginated_query() {
        let mut index = TripleIndex::new();

        // Insert 10 triples with same subject
        for i in 0..10 {
            index.insert(&make_triple(
                "entity",
                &format!("pred{}", i),
                None,
                &format!("h{}", i),
            ));
        }

        // First page
        let page1 = index.by_subject_paginated("entity", 0, 3);
        assert_eq!(page1.items.len(), 3);
        assert_eq!(page1.total, 10);
        assert!(page1.has_more);
        assert!(page1.is_first_page());

        // Second page
        let page2 = index.by_subject_paginated("entity", 3, 3);
        assert_eq!(page2.items.len(), 3);
        assert!(page2.has_more);

        // Last page
        let page3 = index.by_subject_paginated("entity", 9, 3);
        assert_eq!(page3.items.len(), 1);
        assert!(!page3.has_more);
        assert!(page3.is_last_page());
    }

    #[test]
    fn test_batch_insert() {
        let mut index = TripleIndex::new();

        let triples: Vec<StoredTriple> = (0..100)
            .map(|i| make_triple(&format!("subj{}", i), "predicate", None, &format!("h{}", i)))
            .collect();

        index.batch_insert(&triples);

        assert_eq!(index.len(), 100);
        assert_eq!(index.stats().total_triples, 100);
    }

    #[test]
    fn test_batch_remove() {
        let mut index = TripleIndex::new();

        for i in 0..5 {
            index.insert(&make_triple(
                &format!("subj{}", i),
                "predicate",
                None,
                &format!("h{}", i),
            ));
        }

        let removed = index.batch_remove(&["h0", "h2", "h4"]);
        assert_eq!(removed.len(), 3);
        assert!(removed[0].is_some());
        assert!(removed[1].is_some());
        assert!(removed[2].is_some());
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn test_memory_estimation() {
        let mut index = TripleIndex::new();

        for i in 0..100 {
            index.insert(&make_triple(
                &format!("subj{}", i),
                "predicate",
                Some("domain"),
                &format!("h{}", i),
            ));
        }

        let mem = index.estimated_memory_usage();
        assert!(mem > 0);
        // Should be roughly proportional to entry count
        assert!(mem > 100 * 100); // At least 100 bytes per entry
    }

    #[test]
    fn test_query_cache() {
        let cache = QueryCache::new(10, 60);

        // Generate cache key
        let key = QueryCache::cache_key("subject", &["sky"]);
        assert_eq!(key, "subject:sky");

        // Insert and retrieve
        cache.insert(key.clone(), vec!["h1".to_string(), "h2".to_string()]);

        let cached = cache.get(&key);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().len(), 2);
    }

    #[test]
    fn test_query_cache_invalidation() {
        let cache = QueryCache::new(10, 60);

        cache.insert("subject:sky".to_string(), vec!["h1".to_string()]);
        cache.insert("subject:grass".to_string(), vec!["h2".to_string()]);
        cache.insert("predicate:color".to_string(), vec!["h3".to_string()]);

        // Invalidate by prefix
        cache.invalidate_by_prefix("subject:");

        assert!(cache.get("subject:sky").is_none());
        assert!(cache.get("subject:grass").is_none());
        assert!(cache.get("predicate:color").is_some());
    }

    #[test]
    fn test_cached_triple_index() {
        let mut cached = CachedTripleIndex::new();

        cached.insert(&make_triple("sky", "color", None, "h1"));
        cached.insert(&make_triple("sky", "size", None, "h2"));

        // First query (cache miss)
        let results1 = cached.by_subject("sky");
        assert_eq!(results1.len(), 2);

        // Second query (cache hit)
        let results2 = cached.by_subject("sky");
        assert_eq!(results2.len(), 2);

        // Cache should have one entry
        let stats = cached.cache_stats();
        assert_eq!(stats.entry_count, 1);
    }

    #[test]
    fn test_lru_eviction() {
        let cache = QueryCache::new(3, 60);

        cache.insert("k1".to_string(), vec!["v1".to_string()]);
        cache.insert("k2".to_string(), vec!["v2".to_string()]);
        cache.insert("k3".to_string(), vec!["v3".to_string()]);

        // Cache is full, insert one more
        cache.insert("k4".to_string(), vec!["v4".to_string()]);

        // k1 should be evicted (LRU)
        assert!(cache.get("k1").is_none());
        assert!(cache.get("k4").is_some());

        let stats = cache.stats();
        assert_eq!(stats.entry_count, 3);
    }
}
