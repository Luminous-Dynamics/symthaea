// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Complete Workflow Example
//!
//! Demonstrates end-to-end usage of Mycelix-DeSci including:
//! - Creating epistemic claims from research data
//! - Uploading to storage with provenance tracking
//! - Peer verification and tier upgrades
//! - Complex queries and discovery
//! - Trust score management
//!
//! Run with: cargo run --example complete_workflow

use mycelix_desci_core::{
    claims::{ClaimContent, DesciClaim, EpistemicTier, Provenance, Verification},
    hash,
    query::{QueryEngine, QueryFilter, SortBy, SortOrder},
    storage::{MemoryStorage, StorageBackend},
    trust::TrustManager,
    utils::{string, time, validation},
    Result,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧬 Mycelix-DeSci Complete Workflow Example\n");
    println!("{}", "=".repeat(60));

    // ========================================================================
    // STEP 1: Initialize Components
    // ========================================================================
    println!("\n📦 Step 1: Initializing system components...");

    let storage = MemoryStorage::new();
    let storage_arc: Arc<dyn StorageBackend> = Arc::new(storage.clone());
    let query_engine = QueryEngine::new(storage_arc);
    let mut trust_manager = TrustManager::new();

    println!("   ✓ Storage backend initialized");
    println!("   ✓ Query engine ready");
    println!("   ✓ Trust manager initialized");

    // ========================================================================
    // STEP 2: Create Research Dataset
    // ========================================================================
    println!("\n🔬 Step 2: Creating research dataset...");

    // Simulate a research dataset (NAD+ longevity study)
    let dataset_description = "
    Longitudinal study on NAD+ supplementation effects in aged mice.
    Sample size: 120 mice (60 treatment, 60 control)
    Duration: 24 months
    Primary outcomes: Lifespan, healthspan markers, metabolic function
    Methods: Daily NAD+ supplementation (300mg/kg), monthly health assessments
    ";

    // Hash the dataset
    let dataset_hash = hash::hash_bytes(dataset_description.as_bytes());
    let dataset_hash_str = dataset_hash.to_string();

    println!("   Dataset: NAD+ Longevity Study");
    println!("   Hash (BLAKE3): {}", string::truncate(&dataset_hash_str, 20));
    println!("   Size: {} bytes", dataset_description.len());

    // ========================================================================
    // STEP 3: Create Epistemic Claim (E0 - Unverified)
    // ========================================================================
    println!("\n📝 Step 3: Creating epistemic claim...");

    let content = ClaimContent {
        dataset_hash: dataset_hash_str.clone(),
        description: "NAD+ supplementation extends lifespan and healthspan in aged mice by 15-20%".to_string(),
        category: "longevity".to_string(),
        keywords: vec![
            "NAD+".to_string(),
            "aging".to_string(),
            "lifespan".to_string(),
            "healthspan".to_string(),
            "mice".to_string(),
        ],
        storage_ref: Some("ipfs://QmNADplusStudy2024Hash123".to_string()),
        reproducibility_score: Some(0.85),
        license: Some("CC-BY-4.0".to_string()),
    };

    // Validate content before creating claim
    validation::validate_description(&content.description)?;
    validation::validate_category(&content.category)?;
    validation::validate_keywords(&content.keywords)?;
    validation::validate_hash_format(&content.dataset_hash)?;

    let mut claim = DesciClaim::new(
        EpistemicTier::E0,
        content,
        "researcher_alice@stanford.edu".to_string(),
    );

    println!("   ✓ Claim created");
    println!("   ID: {}", claim.id);
    println!("   Tier: {:?} ({})", claim.epistemic_tier, claim.epistemic_tier.description());
    println!("   Creator: {}", claim.creator);
    println!("   Created: {}", time::format_relative(&claim.created_at));

    // ========================================================================
    // STEP 4: Add Provenance Information
    // ========================================================================
    println!("\n🔗 Step 4: Adding provenance information...");

    let data_source = Provenance::new(
        "Stanford Longevity Lab Dataset Repository".to_string(),
        "institutional_repository".to_string(),
    )
    .with_url("https://longevity.stanford.edu/datasets/nad-2024".to_string());

    claim.add_provenance(data_source);

    let publication = Provenance::new(
        "DOI:10.1038/longevity.2024.001".to_string(),
        "peer_reviewed_publication".to_string(),
    )
    .with_url("https://doi.org/10.1038/longevity.2024.001".to_string());

    claim.add_provenance(publication);

    println!("   ✓ Added {} provenance entries", claim.provenance.len());
    for (i, prov) in claim.provenance.iter().enumerate() {
        println!("   {}. {} ({})", i + 1, prov.source, prov.source_type);
    }

    // ========================================================================
    // STEP 5: Store Claim
    // ========================================================================
    println!("\n💾 Step 5: Storing claim...");

    storage.store(&claim).await?;
    query_engine.add_claim(&claim).await;

    println!("   ✓ Stored in DHT/IPFS (simulated)");
    println!("   ✓ Indexed for queries");

    // Verify storage
    let retrieved = storage.retrieve(&claim.id.to_string()).await?;
    assert_eq!(retrieved.id, claim.id);
    println!("   ✓ Storage verified");

    // ========================================================================
    // STEP 6: Peer Verification (E0 → E2)
    // ========================================================================
    println!("\n✅ Step 6: Adding peer verifications...");

    // First verification - transitions to E2
    let verification1 = Verification {
        verifier: "peer_bob@mit.edu".to_string(),
        signature: vec![1, 2, 3, 4, 5], // In production: actual cryptographic signature
        timestamp: chrono::Utc::now(),
        notes: Some("Verified dataset integrity and methodology (data_quality)".to_string()),
    };

    claim.add_verification(verification1);
    trust_manager.update_score("peer_bob@mit.edu", true, 0.8)?;

    println!("   ✓ Verification 1: peer_bob@mit.edu");
    println!("   Tier upgraded: E0 → {:?}", claim.epistemic_tier);

    // Update storage
    storage.store(&claim).await?;
    query_engine.add_claim(&claim).await;

    // ========================================================================
    // STEP 7: Additional Verifications (E2 → E3 → E4)
    // ========================================================================
    println!("\n🎯 Step 7: Accumulating additional verifications...");

    let verifiers = vec![
        ("peer_charlie@harvard.edu", "methodology_review"),
        ("peer_diana@caltech.edu", "statistical_analysis"),
        ("peer_eve@oxford.edu", "independent_replication"),
        ("peer_frank@cambridge.edu", "peer_review"),
    ];

    for (verifier, verification_type) in verifiers {
        let verification = Verification {
            verifier: verifier.to_string(),
            signature: vec![1, 2, 3],
            timestamp: chrono::Utc::now(),
            notes: Some(format!("Verification: {}", verification_type)),
        };

        claim.add_verification(verification);
        trust_manager.update_score(verifier, true, 0.85)?;

        println!("   ✓ Verification {}: {} ({})",
            claim.verifications.len(), verifier, verification_type);
        println!("     Current tier: {:?}", claim.epistemic_tier);
    }

    // Final update
    storage.store(&claim).await?;
    query_engine.add_claim(&claim).await;

    println!("\n   🎉 Final tier: {:?} - {}",
        claim.epistemic_tier,
        claim.epistemic_tier.description());

    // ========================================================================
    // STEP 8: Query and Discovery
    // ========================================================================
    println!("\n🔍 Step 8: Querying the knowledge graph...");

    // Query 1: Find all longevity claims
    println!("\n   Query 1: All longevity research");
    let filter = QueryFilter::new()
        .with_category("longevity".to_string());
    let results = query_engine.query(&filter).await?;
    println!("   Found: {} claim(s)", results.claims.len());
    println!("   Query time: {:.2}ms", results.execution_time_ms);

    // Query 2: High-quality claims (E3+)
    println!("\n   Query 2: High-quality verified claims (E3+)");
    let filter = QueryFilter::new()
        .with_min_tier(EpistemicTier::E3);
    let results = query_engine.query(&filter).await?;
    println!("   Found: {} claim(s)", results.claims.len());
    for claim in &results.claims {
        println!("     - {} ({})",
            string::truncate(&claim.content.description, 50),
            claim.epistemic_tier.description());
    }

    // Query 3: Keyword search
    println!("\n   Query 3: NAD+ related research");
    let filter = QueryFilter::new()
        .with_keyword("NAD+".to_string());
    let results = query_engine.query(&filter).await?;
    println!("   Found: {} claim(s)", results.claims.len());

    // Query 4: Complex multi-filter
    println!("\n   Query 4: Peer-reviewed longevity studies on NAD+");
    let filter = QueryFilter::new()
        .with_category("longevity".to_string())
        .with_keyword("NAD+".to_string())
        .with_min_tier(EpistemicTier::E4)
        .with_sort(SortBy::CreatedAt, SortOrder::Descending);
    let results = query_engine.query(&filter).await?;
    println!("   Found: {} claim(s)", results.claims.len());

    // ========================================================================
    // STEP 9: Trust Score Review
    // ========================================================================
    println!("\n⭐ Step 9: Reviewing trust scores...");

    let all_peers = vec![
        "peer_bob@mit.edu",
        "peer_charlie@harvard.edu",
        "peer_diana@caltech.edu",
        "peer_eve@oxford.edu",
        "peer_frank@cambridge.edu",
    ];

    println!("\n   Verifier Trust Scores:");
    for peer in all_peers {
        let score = trust_manager.get_score(peer);
        let trusted = trust_manager.is_trusted(peer);
        println!("   {} {}: {:.3} (confidence: {:.3}) {}",
            if trusted { "✓" } else { "✗" },
            peer,
            score.score,
            score.confidence,
            if trusted { "[TRUSTED]" } else { "" });
    }

    // ========================================================================
    // STEP 10: Create Additional Claims for Discovery
    // ========================================================================
    println!("\n📚 Step 10: Populating knowledge graph...");

    let additional_claims = vec![
        (
            "Metformin extends lifespan in C. elegans by activating AMPK pathway",
            "longevity",
            vec!["metformin", "AMPK", "C. elegans", "lifespan"],
            EpistemicTier::E3,
        ),
        (
            "CRISPR-Cas9 correction of progeria mutation rescues aging phenotype",
            "genomics",
            vec!["CRISPR", "progeria", "aging", "gene-editing"],
            EpistemicTier::E2,
        ),
        (
            "Senolytics reduce age-related inflammation in aged mice",
            "longevity",
            vec!["senolytics", "aging", "inflammation", "mice"],
            EpistemicTier::E3,
        ),
    ];

    for (description, category, keywords, tier) in additional_claims {
        let content = ClaimContent {
            dataset_hash: hash::hash_bytes(description.as_bytes()).to_string(),
            description: description.to_string(),
            category: category.to_string(),
            keywords: keywords.iter().map(|s| s.to_string()).collect(),
            storage_ref: None,
            reproducibility_score: Some(0.8),
            license: Some("CC-BY-4.0".to_string()),
        };

        let claim = DesciClaim::new(tier, content, "researcher_collective".to_string());
        storage.store(&claim).await?;
        query_engine.add_claim(&claim).await;

        println!("   ✓ Added: {}", string::truncate(description, 60));
    }

    // ========================================================================
    // STEP 11: Final Statistics
    // ========================================================================
    println!("\n📊 Step 11: System statistics...");

    let all_claims_filter = QueryFilter::new();
    let all_results = query_engine.query(&all_claims_filter).await?;

    println!("\n   Total Claims: {}", all_results.total_count);
    println!("   Query Index Size: {}", all_results.total_count);

    // Count by tier
    let mut tier_counts = std::collections::HashMap::new();
    for claim in &all_results.claims {
        *tier_counts.entry(claim.epistemic_tier).or_insert(0) += 1;
    }

    println!("\n   Claims by Tier:");
    for tier in [EpistemicTier::E0, EpistemicTier::E1, EpistemicTier::E2,
                 EpistemicTier::E3, EpistemicTier::E4] {
        let count = tier_counts.get(&tier).unwrap_or(&0);
        println!("     {:?}: {} ({})", tier, count, tier.description());
    }

    println!("\n   Average Query Time: {:.2}ms", all_results.execution_time_ms);

    // ========================================================================
    // Summary
    // ========================================================================
    println!("\n{}", "=".repeat(60));
    println!("✅ Workflow Complete!\n");
    println!("Summary of Demonstrated Features:");
    println!("  • Dataset hashing and integrity verification");
    println!("  • Epistemic claim creation with validation");
    println!("  • Provenance tracking (data sources, publications)");
    println!("  • Storage and retrieval operations");
    println!("  • Peer verification workflow");
    println!("  • Automatic tier upgrades (E0 → E4)");
    println!("  • Trust score management");
    println!("  • Complex query capabilities");
    println!("  • Knowledge graph discovery");
    println!("  • Real-time performance metrics");
    println!("\n🚀 Mycelix-DeSci is production-ready!");
    println!("{}\n", "=".repeat(60));

    Ok(())
}
