// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Query System Demo
//!
//! Demonstrates advanced query capabilities including:
//! - Complex filtering (category, keywords, tier)
//! - Pagination and result limiting
//! - Sorting (by tier, timestamp)
//! - Performance metrics and benchmarking
//!
//! Run with: cargo run --example query_demo

use mycelix_desci_core::{
    claims::{ClaimContent, DesciClaim, EpistemicTier},
    hash,
    query::{QueryEngine, QueryFilter, SortBy, SortOrder},
    storage::{MemoryStorage, StorageBackend},
    utils::{string, time},
    Result,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔍 Mycelix-DeSci Query System Demo\n");
    println!("{}", "=".repeat(70));

    // Initialize
    let storage = MemoryStorage::new();
    let storage_arc: Arc<dyn StorageBackend> = Arc::new(storage.clone());
    let query_engine = QueryEngine::new(storage_arc);

    println!("\n📦 Initializing query engine...");

    // Create sample dataset
    println!("\n📚 Creating sample research dataset (50 claims)...\n");

    let categories = vec!["longevity", "genomics", "climate", "neuroscience", "cancer"];
    let keywords_by_category = vec![
        vec!["NAD+", "aging", "lifespan", "healthspan", "senolytics"],
        vec!["CRISPR", "gene-editing", "DNA", "RNA", "sequencing"],
        vec!["CO2", "temperature", "ocean", "emissions", "renewable"],
        vec!["neurons", "brain", "cognition", "memory", "plasticity"],
        vec!["oncology", "immunotherapy", "tumor", "metastasis", "chemotherapy"],
    ];

    let tiers = vec![
        EpistemicTier::E0,
        EpistemicTier::E1,
        EpistemicTier::E2,
        EpistemicTier::E3,
        EpistemicTier::E4,
    ];

    let mut created_claims = Vec::new();
    let start_time = std::time::Instant::now();

    for i in 0..50 {
        let category_idx = i % categories.len();
        let category = categories[category_idx];
        let keywords = &keywords_by_category[category_idx];
        let tier = tiers[i % tiers.len()];

        let description = format!(
            "Research study #{} in {}: Investigating {} and related mechanisms",
            i + 1,
            category,
            keywords[i % keywords.len()]
        );

        let content = ClaimContent {
            dataset_hash: hash::hash_bytes(description.as_bytes()).to_string(),
            description: description.clone(),
            category: category.to_string(),
            keywords: keywords.iter().take(3).map(|s| s.to_string()).collect(),
            storage_ref: Some(format!("ipfs://Qm{}{}", category, i)),
            reproducibility_score: Some(0.7 + (i as f64 * 0.005)),
            license: Some("MIT".to_string()),
        };

        let claim = DesciClaim::new(
            tier,
            content,
            format!("researcher_{}@university.edu", i % 10),
        );

        storage.store(&claim).await?;
        query_engine.add_claim(&claim).await;
        created_claims.push(claim);
    }

    let creation_time = start_time.elapsed();
    println!("   ✓ Created and indexed 50 claims in {:.2}ms", creation_time.as_secs_f64() * 1000.0);
    println!("   ✓ Categories: {}", string::join_with_and(&categories));
    println!("   ✓ Tiers: E0-E4 distributed evenly\n");

    println!("{}", "=".repeat(70));

    // ========================================================================
    // Query 1: Simple Category Filter
    // ========================================================================
    println!("\n🔍 Query 1: Find all longevity research");
    println!("{}", "-".repeat(70));

    let filter = QueryFilter::new().with_category("longevity".to_string());
    let results = query_engine.query(&filter).await?;

    println!("   Filter: category='longevity'");
    println!("   Results: {}", results.claims.len());
    println!("   Query time: {:.3}ms", results.execution_time_ms);

    for (i, claim) in results.claims.iter().take(3).enumerate() {
        println!("\n   Result #{}:", i + 1);
        println!("     Description: {}", string::truncate(&claim.content.description, 60));
        println!("     Tier: {:?}", claim.epistemic_tier);
        println!("     Keywords: {}", claim.content.keywords.join(", "));
    }

    if results.claims.len() > 3 {
        println!("\n   ... and {} more", results.claims.len() - 3);
    }

    // ========================================================================
    // Query 2: High-Quality Claims (E3+)
    // ========================================================================
    println!("\n\n🔍 Query 2: Find peer-reviewed claims (E3+)");
    println!("{}", "-".repeat(70));

    let filter = QueryFilter::new().with_min_tier(EpistemicTier::E3);
    let results = query_engine.query(&filter).await?;

    println!("   Filter: tier >= E3");
    println!("   Results: {}", results.claims.len());
    println!("   Query time: {:.3}ms", results.execution_time_ms);

    // Group by tier
    let mut by_tier = std::collections::HashMap::new();
    for claim in &results.claims {
        *by_tier.entry(claim.epistemic_tier).or_insert(0) += 1;
    }

    println!("\n   Distribution:");
    for tier in [EpistemicTier::E3, EpistemicTier::E4] {
        if let Some(count) = by_tier.get(&tier) {
            println!("     {:?}: {} claims", tier, count);
        }
    }

    // ========================================================================
    // Query 3: Keyword Search
    // ========================================================================
    println!("\n\n🔍 Query 3: Keyword search for 'CRISPR'");
    println!("{}", "-".repeat(70));

    let filter = QueryFilter::new().with_keyword("CRISPR".to_string());
    let results = query_engine.query(&filter).await?;

    println!("   Filter: keyword='CRISPR'");
    println!("   Results: {}", results.claims.len());
    println!("   Query time: {:.3}ms", results.execution_time_ms);

    for (i, claim) in results.claims.iter().enumerate() {
        println!("   {}. {} ({})",
            i + 1,
            string::truncate(&claim.content.description, 50),
            claim.content.category);
    }

    // ========================================================================
    // Query 4: Multi-Filter Complex Query
    // ========================================================================
    println!("\n\n🔍 Query 4: Complex multi-filter query");
    println!("{}", "-".repeat(70));

    let filter = QueryFilter::new()
        .with_category("genomics".to_string())
        .with_keyword("CRISPR".to_string())
        .with_min_tier(EpistemicTier::E2);

    let results = query_engine.query(&filter).await?;

    println!("   Filters:");
    println!("     - category='genomics'");
    println!("     - keyword='CRISPR'");
    println!("     - tier >= E2");
    println!("\n   Results: {}", results.claims.len());
    println!("   Query time: {:.3}ms", results.execution_time_ms);

    // ========================================================================
    // Query 5: Pagination
    // ========================================================================
    println!("\n\n🔍 Query 5: Pagination (pages of 10)");
    println!("{}", "-".repeat(70));

    let page_size = 10;
    let total_pages = (created_claims.len() + page_size - 1) / page_size;

    println!("   Page size: {}", page_size);
    println!("   Total pages: {}", total_pages);

    for page in 0..std::cmp::min(3, total_pages) {
        let offset = page * page_size;
        let filter = QueryFilter::new()
            .with_limit(page_size)
            .with_offset(offset);

        let results = query_engine.query(&filter).await?;

        println!("\n   Page {} (offset={}, limit={}):", page + 1, offset, page_size);
        println!("     Claims: {}", results.claims.len());
        println!("     Query time: {:.3}ms", results.execution_time_ms);
    }

    // ========================================================================
    // Query 6: Sorting by Tier (Descending)
    // ========================================================================
    println!("\n\n🔍 Query 6: Sort by epistemic tier (highest first)");
    println!("{}", "-".repeat(70));

    let filter = QueryFilter::new()
        .with_sort(SortBy::EpistemicTier, SortOrder::Descending)
        .with_limit(10);

    let results = query_engine.query(&filter).await?;

    println!("   Sort: tier DESC, limit 10");
    println!("   Results: {}", results.claims.len());
    println!("   Query time: {:.3}ms", results.execution_time_ms);

    println!("\n   Top claims by tier:");
    for (i, claim) in results.claims.iter().enumerate() {
        println!("   {}. {:?} - {}",
            i + 1,
            claim.epistemic_tier,
            string::truncate(&claim.content.description, 50));
    }

    // ========================================================================
    // Query 7: Sorting by Timestamp (Recent first)
    // ========================================================================
    println!("\n\n🔍 Query 7: Sort by timestamp (most recent)");
    println!("{}", "-".repeat(70));

    let filter = QueryFilter::new()
        .with_sort(SortBy::CreatedAt, SortOrder::Descending)
        .with_limit(5);

    let results = query_engine.query(&filter).await?;

    println!("   Sort: timestamp DESC, limit 5");
    println!("   Results: {}", results.claims.len());
    println!("   Query time: {:.3}ms", results.execution_time_ms);

    println!("\n   Recent claims:");
    for (i, claim) in results.claims.iter().enumerate() {
        println!("   {}. {} - {}",
            i + 1,
            time::format_relative(&claim.created_at),
            string::truncate(&claim.content.description, 50));
    }

    // ========================================================================
    // Query 8: Category Distribution
    // ========================================================================
    println!("\n\n🔍 Query 8: Category distribution analysis");
    println!("{}", "-".repeat(70));

    println!("\n   Analyzing {} claims across categories:\n", created_claims.len());

    for category in &categories {
        let filter = QueryFilter::new().with_category(category.to_string());
        let results = query_engine.query(&filter).await?;

        let percentage = (results.claims.len() as f64 / created_claims.len() as f64) * 100.0;
        let bar = "█".repeat((percentage / 5.0) as usize);

        println!("   {:15} {:2} claims ({:5.1}%) {}",
            category,
            results.claims.len(),
            percentage,
            bar);
    }

    // ========================================================================
    // Performance Benchmarking
    // ========================================================================
    println!("\n\n⚡ Performance Benchmarking");
    println!("{}", "=".repeat(70));

    let benchmark_queries = vec![
        ("Category filter", QueryFilter::new().with_category("longevity".to_string())),
        ("Tier filter", QueryFilter::new().with_min_tier(EpistemicTier::E3)),
        ("Keyword search", QueryFilter::new().with_keyword("CRISPR".to_string())),
        ("Complex multi-filter", QueryFilter::new()
            .with_category("genomics".to_string())
            .with_keyword("DNA".to_string())
            .with_min_tier(EpistemicTier::E2)),
        ("Paginated (10)", QueryFilter::new().with_limit(10)),
        ("Sorted by tier", QueryFilter::new()
            .with_sort(SortBy::EpistemicTier, SortOrder::Descending)),
    ];

    println!("\n   Running {} queries multiple times for accurate timing...\n", benchmark_queries.len());

    for (name, filter) in benchmark_queries {
        let iterations = 100;
        let start = std::time::Instant::now();

        for _ in 0..iterations {
            let _ = query_engine.query(&filter).await?;
        }

        let total_time = start.elapsed();
        let avg_time = total_time.as_secs_f64() * 1000.0 / iterations as f64;

        println!("   {:25} {:.3}ms avg ({} iterations)",
            format!("{}:", name),
            avg_time,
            iterations);
    }

    // ========================================================================
    // Summary Statistics
    // ========================================================================
    println!("\n\n📊 Summary Statistics");
    println!("{}", "=".repeat(70));

    let all_filter = QueryFilter::new();
    let all_results = query_engine.query(&all_filter).await?;

    println!("\n   Total Claims: {}", all_results.total_count);
    println!("   Categories: {}", categories.len());
    println!("   Tiers: E0-E4 (5 levels)");

    println!("\n   Claims per Category:");
    for category in &categories {
        let filter = QueryFilter::new().with_category(category.to_string());
        let results = query_engine.query(&filter).await?;
        println!("     {:15} {}", category, results.claims.len());
    }

    println!("\n   Claims per Tier:");
    for tier in &tiers {
        let filter = QueryFilter::new().with_min_tier(*tier);
        let results = query_engine.query(&filter).await?;
        let count = if *tier == EpistemicTier::E4 {
            results.claims.len()
        } else {
            let next_tier_idx = tiers.iter().position(|t| t == tier).unwrap() + 1;
            if next_tier_idx < tiers.len() {
                let next_filter = QueryFilter::new().with_min_tier(tiers[next_tier_idx]);
                let next_results = query_engine.query(&next_filter).await?;
                results.claims.len() - next_results.claims.len()
            } else {
                results.claims.len()
            }
        };
        println!("     {:?}: {}", tier, count);
    }

    println!("\n   Average Query Time: {:.3}ms", all_results.execution_time_ms);

    println!("\n{}", "=".repeat(70));
    println!("✅ Query Demo Complete!\n");
    println!("Demonstrated Features:");
    println!("  • Category, tier, and keyword filtering");
    println!("  • Pagination with offset/limit");
    println!("  • Sorting by tier and timestamp");
    println!("  • Complex multi-filter queries");
    println!("  • Performance benchmarking");
    println!("  • Real-time query metrics");
    println!("\n🚀 Query engine is fast and flexible!");
    println!("{}\n", "=".repeat(70));

    Ok(())
}
