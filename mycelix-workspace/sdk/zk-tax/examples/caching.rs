//! Proof caching example - improve performance with cached proofs.
//!
//! Run with: `cargo run --example caching --features prover`

use mycelix_zk_tax::{
    ProofCache, CacheConfig, CacheStats,
    global_cache, cache_proof, get_cached, clear_global_cache, global_stats,
    FilingStatus, Jurisdiction, TaxBracketProver,
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Proof Caching Example ===\n");

    let prover = TaxBracketProver::dev_mode();

    // ==========================================================================
    // Example 1: Basic cache usage
    // ==========================================================================
    println!("1. Basic cache operations");

    let cache = ProofCache::new();

    // Generate a proof
    let proof = prover.prove(85_000, Jurisdiction::US, FilingStatus::Single, 2024)?;

    // Cache it
    cache.put(proof.clone());
    println!("   Proof cached: ✓");

    // Retrieve from cache
    let cached = cache.get_for_income(85_000, Jurisdiction::US, FilingStatus::Single, 2024);
    println!("   Retrieved from cache: {}\n", if cached.is_some() { "✓" } else { "✗" });

    // ==========================================================================
    // Example 2: Cache with configuration
    // ==========================================================================
    println!("2. Configured cache (capacity: 100, TTL: 60s, LRU enabled)");

    let config = CacheConfig::default()
        .with_capacity(100)
        .with_ttl_secs(60)
        .with_lru(true);

    let cache = ProofCache::with_config(config);

    // Add several proofs
    for income in [50_000u64, 75_000, 100_000, 150_000, 200_000] {
        let proof = prover.prove(income, Jurisdiction::US, FilingStatus::Single, 2024)?;
        cache.put(proof);
    }

    let stats = cache.stats();
    println!("   Proofs cached: {}", stats.total_proofs);
    println!("   Cache hits: {}", stats.cache_hits);
    println!("   Cache misses: {}\n", stats.cache_misses);

    // ==========================================================================
    // Example 3: High-performance preset
    // ==========================================================================
    println!("3. High-performance cache preset");

    let config = CacheConfig::high_performance();
    let cache = ProofCache::with_config(config);

    println!("   Capacity: unlimited");
    println!("   TTL: disabled (infinite)");
    println!("   LRU: disabled\n");

    // ==========================================================================
    // Example 4: Strict verification preset
    // ==========================================================================
    println!("4. Strict verification cache preset");

    let config = CacheConfig::strict();
    let _cache = ProofCache::with_config(config);

    println!("   Verifies proofs on retrieval");
    println!("   Useful for high-security applications\n");

    // ==========================================================================
    // Example 5: Global cache usage
    // ==========================================================================
    println!("5. Global cache (singleton pattern)");

    // Clear any existing entries
    clear_global_cache();

    // Generate and cache via global cache
    let proof = prover.prove(120_000, Jurisdiction::US, FilingStatus::Single, 2024)?;
    cache_proof(proof);
    println!("   Proof added to global cache");

    // Try to get from global cache
    let result = get_cached(120_000, Jurisdiction::US, FilingStatus::Single, 2024);
    println!("   Retrieved: {}", if result.is_some() { "✓" } else { "✗" });

    let stats = global_stats();
    println!("   Global cache stats: {} proofs\n", stats.total_proofs);

    // ==========================================================================
    // Example 6: Performance comparison
    // ==========================================================================
    println!("6. Performance comparison: Cache vs. Generate");

    let cache = ProofCache::new();

    // First call (cache miss - generate)
    let start = Instant::now();
    let proof = prover.prove(95_000, Jurisdiction::US, FilingStatus::Single, 2024)?;
    let generate_time = start.elapsed();
    cache.put(proof);

    // Second call (cache hit)
    let start = Instant::now();
    let _cached = cache.get_for_income(95_000, Jurisdiction::US, FilingStatus::Single, 2024);
    let cache_time = start.elapsed();

    println!("   Generate time: {:?}", generate_time);
    println!("   Cache lookup: {:?}", cache_time);
    println!("   Speedup: {:.0}x", generate_time.as_nanos() as f64 / cache_time.as_nanos() as f64);

    println!("\n=== Caching Example Complete ===");
    Ok(())
}
