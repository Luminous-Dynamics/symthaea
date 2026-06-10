// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Proof caching system for reusing ZK proofs within the same tax year.
//!
//! This module provides caching capabilities to avoid regenerating expensive
//! ZK proofs when the same bracket proof has already been generated.
//!
//! # Features
//!
//! - **LRU Eviction**: Automatic eviction when capacity is reached
//! - **TTL Expiration**: Time-based proof expiration
//! - **Thread-Safe**: Safe for concurrent access
//! - **Global Cache**: Convenient global cache instance
//! - **Configurable**: Flexible configuration options
//!
//! # Example
//!
//! ```rust
//! use mycelix_zk_tax::{Jurisdiction, FilingStatus};
//! use mycelix_zk_tax::cache::{ProofCache, ProofCacheKey, CacheConfig};
//!
//! // Create a cache with custom configuration
//! let config = CacheConfig::default()
//!     .with_capacity(10_000)
//!     .with_ttl_secs(3600);
//! let cache = ProofCache::with_config(config);
//!
//! // Use the cache
//! let key = ProofCacheKey::new(Jurisdiction::US, FilingStatus::Single, 2024, 2);
//! if let Some(proof) = cache.get(&key) {
//!     println!("Cache hit!");
//! }
//! ```

use crate::jurisdiction::{FilingStatus, Jurisdiction};
use crate::proof::TaxBracketProof;
use crate::types::TaxYear;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Cache key for proof lookups.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ProofCacheKey {
    /// Tax jurisdiction
    pub jurisdiction: Jurisdiction,
    /// Filing status
    pub filing_status: FilingStatus,
    /// Tax year
    pub tax_year: TaxYear,
    /// Bracket index (the proof is valid for the entire bracket)
    pub bracket_index: u8,
}

impl ProofCacheKey {
    /// Create a new cache key.
    pub fn new(
        jurisdiction: Jurisdiction,
        filing_status: FilingStatus,
        tax_year: TaxYear,
        bracket_index: u8,
    ) -> Self {
        Self {
            jurisdiction,
            filing_status,
            tax_year,
            bracket_index,
        }
    }

    /// Create a cache key from a proof.
    pub fn from_proof(proof: &TaxBracketProof) -> Self {
        Self {
            jurisdiction: proof.jurisdiction,
            filing_status: proof.filing_status,
            tax_year: proof.tax_year,
            bracket_index: proof.bracket_index,
        }
    }

    /// Create a key from income and parameters (finds bracket internally).
    pub fn from_income(
        income: u64,
        jurisdiction: Jurisdiction,
        filing_status: FilingStatus,
        tax_year: TaxYear,
    ) -> Option<Self> {
        use crate::brackets::find_bracket;
        find_bracket(income, jurisdiction, tax_year, filing_status)
            .ok()
            .map(|bracket| Self::new(jurisdiction, filing_status, tax_year, bracket.index))
    }
}

// =============================================================================
// Cache Configuration
// =============================================================================

/// Configuration for the proof cache.
#[derive(Clone, Debug)]
pub struct CacheConfig {
    /// Maximum number of entries (0 = unlimited)
    pub capacity: usize,
    /// Time-to-live in seconds (0 = no expiration)
    pub ttl_secs: u64,
    /// Whether to use LRU eviction when at capacity
    pub lru_enabled: bool,
    /// Whether to verify proofs on retrieval
    pub verify_on_get: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            capacity: 10_000,
            ttl_secs: 3600, // 1 hour default
            lru_enabled: true,
            verify_on_get: false,
        }
    }
}

impl CacheConfig {
    /// Set the cache capacity.
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    /// Set the TTL in seconds.
    pub fn with_ttl_secs(mut self, secs: u64) -> Self {
        self.ttl_secs = secs;
        self
    }

    /// Enable or disable LRU eviction.
    pub fn with_lru(mut self, enabled: bool) -> Self {
        self.lru_enabled = enabled;
        self
    }

    /// Enable verification on get.
    pub fn with_verify(mut self, enabled: bool) -> Self {
        self.verify_on_get = enabled;
        self
    }

    /// Create a high-performance configuration.
    pub fn high_performance() -> Self {
        Self {
            capacity: 100_000,
            ttl_secs: 86400, // 24 hours
            lru_enabled: true,
            verify_on_get: false,
        }
    }

    /// Create a strict verification configuration.
    pub fn strict() -> Self {
        Self {
            capacity: 1_000,
            ttl_secs: 300, // 5 minutes
            lru_enabled: true,
            verify_on_get: true,
        }
    }

    /// Create a persistent (no expiry) configuration.
    pub fn persistent() -> Self {
        Self {
            capacity: 0, // Unlimited
            ttl_secs: 0, // No expiry
            lru_enabled: false,
            verify_on_get: false,
        }
    }
}

/// Cache entry with metadata.
#[derive(Clone, Debug)]
pub struct CacheEntry {
    /// The cached proof
    pub proof: TaxBracketProof,
    /// Timestamp when cached (Unix seconds)
    pub cached_at: u64,
    /// Last access timestamp (Unix seconds) for LRU
    pub last_accessed: u64,
    /// Number of times this proof has been retrieved
    pub hit_count: u32,
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl CacheEntry {
    /// Create a new cache entry.
    pub fn new(proof: TaxBracketProof) -> Self {
        let now = current_timestamp();
        Self {
            proof,
            cached_at: now,
            last_accessed: now,
            hit_count: 0,
        }
    }

    /// Check if this entry is still valid for the current tax year.
    pub fn is_valid_for_year(&self, year: TaxYear) -> bool {
        self.proof.tax_year == year
    }

    /// Check if this entry has expired based on TTL.
    pub fn is_expired(&self, ttl_secs: u64) -> bool {
        if ttl_secs == 0 {
            return false; // No expiration
        }
        let now = current_timestamp();
        now > self.cached_at + ttl_secs
    }

    /// Update the last accessed time and hit count.
    pub fn touch(&mut self) {
        self.last_accessed = current_timestamp();
        self.hit_count += 1;
    }
}

/// Thread-safe proof cache with LRU eviction and TTL support.
#[derive(Clone)]
pub struct ProofCache {
    entries: Arc<RwLock<HashMap<ProofCacheKey, CacheEntry>>>,
    stats: Arc<RwLock<CacheStats>>,
    config: CacheConfig,
}

impl Default for ProofCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache statistics.
#[derive(Clone, Debug, Default)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total proofs cached
    pub cached: u64,
    /// Total proofs evicted
    pub evicted: u64,
}

impl CacheStats {
    /// Calculate hit rate.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

impl ProofCache {
    /// Create a new empty cache with default configuration.
    pub fn new() -> Self {
        Self::with_config(CacheConfig::default())
    }

    /// Create a cache with custom configuration.
    pub fn with_config(config: CacheConfig) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(CacheStats::default())),
            config,
        }
    }

    /// Get the cache configuration.
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// Get a cached proof if available.
    pub fn get(&self, key: &ProofCacheKey) -> Option<TaxBracketProof> {
        let mut entries = self.entries.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        if let Some(entry) = entries.get_mut(key) {
            // Check TTL expiration
            if entry.is_expired(self.config.ttl_secs) {
                entries.remove(key);
                stats.misses += 1;
                stats.evicted += 1;
                return None;
            }

            // Optional verification
            if self.config.verify_on_get {
                if entry.proof.verify().is_err() {
                    entries.remove(key);
                    stats.misses += 1;
                    stats.evicted += 1;
                    return None;
                }
            }

            entry.touch();
            stats.hits += 1;
            Some(entry.proof.clone())
        } else {
            stats.misses += 1;
            None
        }
    }

    /// Store a proof in the cache.
    pub fn put(&self, proof: TaxBracketProof) {
        let key = ProofCacheKey::from_proof(&proof);
        let entry = CacheEntry::new(proof);

        let mut entries = self.entries.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        // Check if we need to evict
        if self.config.capacity > 0 && entries.len() >= self.config.capacity {
            if self.config.lru_enabled {
                self.evict_lru_internal(&mut entries, &mut stats);
            }
        }

        entries.insert(key, entry);
        stats.cached += 1;
    }

    /// Internal LRU eviction (must be called with lock held).
    fn evict_lru_internal(
        &self,
        entries: &mut HashMap<ProofCacheKey, CacheEntry>,
        stats: &mut CacheStats,
    ) {
        // Find the least recently used entry
        let lru_key = entries
            .iter()
            .min_by_key(|(_, e)| e.last_accessed)
            .map(|(k, _)| k.clone());

        if let Some(key) = lru_key {
            entries.remove(&key);
            stats.evicted += 1;
        }
    }

    /// Evict expired entries.
    pub fn evict_expired(&self) -> usize {
        if self.config.ttl_secs == 0 {
            return 0; // No TTL configured
        }

        let mut entries = self.entries.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        let before = entries.len();
        entries.retain(|_, e| !e.is_expired(self.config.ttl_secs));
        let removed = before - entries.len();

        stats.evicted += removed as u64;
        removed
    }

    /// Get a proof or generate one using the provided closure.
    pub fn get_or_generate<F, E>(&self, key: &ProofCacheKey, generator: F) -> Result<TaxBracketProof, E>
    where
        F: FnOnce() -> Result<TaxBracketProof, E>,
    {
        // Try cache first
        if let Some(proof) = self.get(key) {
            return Ok(proof);
        }

        // Generate new proof
        let proof = generator()?;

        // Cache it
        self.put(proof.clone());

        Ok(proof)
    }

    /// Convenience method to get or generate from income amount.
    ///
    /// Returns None if the income doesn't match a valid bracket.
    pub fn get_or_generate_for_income<F, E>(
        &self,
        income: u64,
        jurisdiction: Jurisdiction,
        filing_status: FilingStatus,
        tax_year: TaxYear,
        generator: F,
    ) -> Option<Result<TaxBracketProof, E>>
    where
        F: FnOnce() -> Result<TaxBracketProof, E>,
    {
        let key = ProofCacheKey::from_income(income, jurisdiction, filing_status, tax_year)?;
        Some(self.get_or_generate(&key, generator))
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        self.stats.read().unwrap().clone()
    }

    /// Clear all entries for a specific tax year.
    pub fn clear_year(&self, year: TaxYear) -> usize {
        let mut entries = self.entries.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        let before = entries.len();
        entries.retain(|k, _| k.tax_year != year);
        let removed = before - entries.len();

        stats.evicted += removed as u64;
        removed
    }

    /// Clear all cached proofs.
    pub fn clear_all(&self) -> usize {
        let mut entries = self.entries.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        let count = entries.len();
        entries.clear();
        stats.evicted += count as u64;
        count
    }

    /// Get the number of cached proofs.
    pub fn len(&self) -> usize {
        self.entries.read().unwrap().len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.read().unwrap().is_empty()
    }

    /// List all cached proof keys.
    pub fn keys(&self) -> Vec<ProofCacheKey> {
        self.entries.read().unwrap().keys().cloned().collect()
    }

    /// Get cache capacity.
    pub fn capacity(&self) -> usize {
        self.config.capacity
    }
}

// =============================================================================
// Global Cache
// =============================================================================

lazy_static::lazy_static! {
    /// Global proof cache instance with default configuration.
    static ref GLOBAL_CACHE: ProofCache = ProofCache::new();
}

/// Get the global cache instance.
pub fn global_cache() -> &'static ProofCache {
    &GLOBAL_CACHE
}

/// Get a cached proof from the global cache.
pub fn get_cached(
    jurisdiction: Jurisdiction,
    filing_status: FilingStatus,
    tax_year: TaxYear,
    bracket_index: u8,
) -> Option<TaxBracketProof> {
    let key = ProofCacheKey::new(jurisdiction, filing_status, tax_year, bracket_index);
    GLOBAL_CACHE.get(&key)
}

/// Cache a proof in the global cache.
pub fn cache_proof(proof: TaxBracketProof) {
    GLOBAL_CACHE.put(proof);
}

/// Get or generate a proof using the global cache.
pub fn get_or_generate_cached<F, E>(key: &ProofCacheKey, generator: F) -> Result<TaxBracketProof, E>
where
    F: FnOnce() -> Result<TaxBracketProof, E>,
{
    GLOBAL_CACHE.get_or_generate(key, generator)
}

/// Get global cache statistics.
pub fn global_stats() -> CacheStats {
    GLOBAL_CACHE.stats()
}

/// Clear the global cache.
pub fn clear_global_cache() -> usize {
    GLOBAL_CACHE.clear_all()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brackets::{compute_commitment, get_brackets};

    fn make_test_proof() -> TaxBracketProof {
        let brackets = get_brackets(Jurisdiction::US, 2024, FilingStatus::Single).unwrap();
        let bracket = &brackets[2]; // 22% bracket

        TaxBracketProof {
            jurisdiction: Jurisdiction::US,
            filing_status: FilingStatus::Single,
            tax_year: 2024,
            bracket_index: bracket.index,
            rate_bps: bracket.rate_bps,
            bracket_lower: bracket.lower,
            bracket_upper: bracket.upper,
            commitment: compute_commitment(bracket.lower, bracket.upper, 2024),
            receipt_bytes: vec![0xDE, 0xAD, 0xBE, 0xEF], // Dev mode marker
            image_id: vec![0u8; 32],
        }
    }

    fn make_test_proof_for_bracket(bracket_index: u8) -> TaxBracketProof {
        let brackets = get_brackets(Jurisdiction::US, 2024, FilingStatus::Single).unwrap();
        let bracket = &brackets[bracket_index as usize];

        TaxBracketProof {
            jurisdiction: Jurisdiction::US,
            filing_status: FilingStatus::Single,
            tax_year: 2024,
            bracket_index: bracket.index,
            rate_bps: bracket.rate_bps,
            bracket_lower: bracket.lower,
            bracket_upper: bracket.upper,
            commitment: compute_commitment(bracket.lower, bracket.upper, 2024),
            receipt_bytes: vec![0xDE, 0xAD, 0xBE, 0xEF], // Dev mode marker
            image_id: vec![0u8; 32],
        }
    }

    #[test]
    fn test_cache_hit() {
        let cache = ProofCache::new();
        let proof = make_test_proof();
        let key = ProofCacheKey::from_proof(&proof);

        cache.put(proof.clone());

        let cached = cache.get(&key).unwrap();
        assert_eq!(cached.bracket_index, 2);

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_cache_miss() {
        let cache = ProofCache::new();
        let key = ProofCacheKey::new(Jurisdiction::US, FilingStatus::Single, 2024, 2);

        assert!(cache.get(&key).is_none());

        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_get_or_generate() {
        let cache = ProofCache::new();
        let key = ProofCacheKey::new(Jurisdiction::US, FilingStatus::Single, 2024, 2);

        let mut generated = false;
        let proof: Result<TaxBracketProof, ()> = cache.get_or_generate(&key, || {
            generated = true;
            Ok(make_test_proof())
        });

        assert!(proof.is_ok());
        assert!(generated);

        // Second call should hit cache
        generated = false;
        let proof2: Result<TaxBracketProof, ()> = cache.get_or_generate(&key, || {
            generated = true;
            Ok(make_test_proof())
        });

        assert!(proof2.is_ok());
        assert!(!generated); // Generator not called
    }

    #[test]
    fn test_clear_year() {
        let cache = ProofCache::new();

        let proof_2024 = make_test_proof();

        let brackets_2025 = get_brackets(Jurisdiction::US, 2025, FilingStatus::Single).unwrap();
        let bracket = &brackets_2025[2];
        let proof_2025 = TaxBracketProof {
            jurisdiction: Jurisdiction::US,
            filing_status: FilingStatus::Single,
            tax_year: 2025,
            bracket_index: bracket.index,
            rate_bps: bracket.rate_bps,
            bracket_lower: bracket.lower,
            bracket_upper: bracket.upper,
            commitment: compute_commitment(bracket.lower, bracket.upper, 2025),
            receipt_bytes: vec![0xDE, 0xAD, 0xBE, 0xEF],
            image_id: vec![0u8; 32],
        };

        cache.put(proof_2024);
        cache.put(proof_2025);

        assert_eq!(cache.len(), 2);

        let removed = cache.clear_year(2024);
        assert_eq!(removed, 1);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_lru_eviction() {
        // Create cache with capacity of 2
        let config = CacheConfig::default()
            .with_capacity(2)
            .with_lru(true)
            .with_ttl_secs(0); // No TTL expiration
        let cache = ProofCache::with_config(config);

        // Add first proof (bracket 0)
        let proof0 = make_test_proof_for_bracket(0);
        cache.put(proof0);

        // Add second proof (bracket 1)
        let proof1 = make_test_proof_for_bracket(1);
        cache.put(proof1);

        assert_eq!(cache.len(), 2);

        // Add third proof (bracket 2) - should evict one of the previous entries
        let proof2 = make_test_proof_for_bracket(2);
        cache.put(proof2);

        // Cache should still be at capacity
        assert_eq!(cache.len(), 2);

        // Bracket 2 should definitely be cached (just added)
        let key2 = ProofCacheKey::new(Jurisdiction::US, FilingStatus::Single, 2024, 2);
        assert!(cache.get(&key2).is_some());

        // Verify eviction stats
        let stats = cache.stats();
        assert!(stats.evicted >= 1, "At least one entry should have been evicted");
    }

    #[test]
    fn test_cache_config() {
        let config = CacheConfig::high_performance();
        assert_eq!(config.capacity, 100_000);
        assert_eq!(config.ttl_secs, 86400);

        let config = CacheConfig::strict();
        assert!(config.verify_on_get);
        assert_eq!(config.ttl_secs, 300);

        let config = CacheConfig::persistent();
        assert_eq!(config.ttl_secs, 0);
        assert_eq!(config.capacity, 0);
    }

    #[test]
    fn test_cache_stats() {
        let cache = ProofCache::new();
        let proof = make_test_proof();
        let key = ProofCacheKey::from_proof(&proof);

        // Miss
        cache.get(&key);

        // Put
        cache.put(proof);

        // Hit
        cache.get(&key);
        cache.get(&key);

        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.cached, 1);
        assert!(stats.hit_rate() > 0.5);
    }

    #[test]
    fn test_global_cache() {
        clear_global_cache();

        let proof = make_test_proof();
        let key = ProofCacheKey::from_proof(&proof);

        // Should miss initially
        assert!(get_cached(Jurisdiction::US, FilingStatus::Single, 2024, 2).is_none());

        // Cache it
        cache_proof(proof.clone());

        // Should hit now
        let cached = get_cached(Jurisdiction::US, FilingStatus::Single, 2024, 2);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().bracket_index, proof.bracket_index);
    }

    #[test]
    fn test_key_from_income() {
        let key = ProofCacheKey::from_income(85000, Jurisdiction::US, FilingStatus::Single, 2024);
        assert!(key.is_some());
        let key = key.unwrap();
        assert_eq!(key.jurisdiction, Jurisdiction::US);
        assert_eq!(key.tax_year, 2024);
        // 85000 should be in the 22% bracket (bracket index 2)
        assert_eq!(key.bracket_index, 2);
    }
}
