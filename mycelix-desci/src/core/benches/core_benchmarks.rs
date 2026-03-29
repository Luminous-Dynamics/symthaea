// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core Performance Benchmarks
//!
//! Comprehensive benchmark suite for Mycelix-DeSci core components using Criterion.rs
//! Run with: cargo bench --bench core_benchmarks
//!
//! Benchmarks cover:
//! - Claims: creation, serialization, validation, tier upgrades, provenance
//! - Storage: read, write, concurrent access, bulk operations
//! - Query: indexing, filtering, keyword search, complex queries, pagination
//! - Hash: BLAKE3, SHA-256, Merkle tree construction
//! - Trust: score updates, queries, decay calculations

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mycelix_desci_core::{
    claims::{ClaimContent, DesciClaim, EpistemicTier, Provenance, Verification},
    hash,
    query::{QueryEngine, QueryFilter, SortBy, SortOrder},
    storage::{MemoryStorage, StorageBackend},
    trust::TrustManager,
};
use std::sync::Arc;
use tokio::runtime::Runtime;

// =============================================================================
// CLAIMS BENCHMARKS (5)
// =============================================================================

fn bench_claim_creation(c: &mut Criterion) {
    c.bench_function("claim_creation", |b| {
        b.iter(|| {
            let content = ClaimContent {
                dataset_hash: hash::hash_bytes(b"test data").to_string(),
                description: "NAD+ extends lifespan in mice".to_string(),
                category: "longevity".to_string(),
                keywords: vec!["NAD+".to_string(), "aging".to_string()],
                storage_ref: Some("ipfs://Qm123".to_string()),
                reproducibility_score: Some(0.85),
                license: Some("MIT".to_string()),
            };

            black_box(DesciClaim::new(
                EpistemicTier::E0,
                content,
                "researcher@stanford.edu".to_string(),
            ))
        });
    });
}

fn bench_claim_serialization(c: &mut Criterion) {
    let content = ClaimContent {
        dataset_hash: hash::hash_bytes(b"test data").to_string(),
        description: "NAD+ extends lifespan in mice".to_string(),
        category: "longevity".to_string(),
        keywords: vec!["NAD+".to_string(), "aging".to_string()],
        storage_ref: Some("ipfs://Qm123".to_string()),
        reproducibility_score: Some(0.85),
        license: Some("MIT".to_string()),
    };
    let claim = DesciClaim::new(
        EpistemicTier::E0,
        content,
        "researcher@stanford.edu".to_string(),
    );

    c.bench_function("claim_serialization_json", |b| {
        b.iter(|| {
            let json = serde_json::to_string(&claim).unwrap();
            let _deserialized: DesciClaim = serde_json::from_str(&json).unwrap();
            black_box(_deserialized)
        });
    });
}

fn bench_claim_validation(c: &mut Criterion) {
    let content = ClaimContent {
        dataset_hash: hash::hash_bytes(b"test data").to_string(),
        description: "NAD+ extends lifespan in mice".to_string(),
        category: "longevity".to_string(),
        keywords: vec!["NAD+".to_string(), "aging".to_string()],
        storage_ref: Some("ipfs://Qm123".to_string()),
        reproducibility_score: Some(0.85),
        license: Some("MIT".to_string()),
    };
    let claim = DesciClaim::new(
        EpistemicTier::E0,
        content,
        "researcher@stanford.edu".to_string(),
    );

    c.bench_function("claim_validation", |b| {
        b.iter(|| {
            use mycelix_desci_core::utils::validation;
            black_box(validation::validate_claim(&claim))
        });
    });
}

fn bench_tier_upgrade(c: &mut Criterion) {
    let content = ClaimContent {
        dataset_hash: hash::hash_bytes(b"test data").to_string(),
        description: "NAD+ extends lifespan in mice".to_string(),
        category: "longevity".to_string(),
        keywords: vec!["NAD+".to_string(), "aging".to_string()],
        storage_ref: Some("ipfs://Qm123".to_string()),
        reproducibility_score: Some(0.85),
        license: Some("MIT".to_string()),
    };

    c.bench_function("tier_upgrade_e0_to_e4", |b| {
        b.iter(|| {
            let mut claim = DesciClaim::new(
                EpistemicTier::E0,
                content.clone(),
                "researcher@stanford.edu".to_string(),
            );

            // Add 5 verifications to reach E4
            for i in 0..5 {
                let verification = Verification {
                    verifier: format!("peer_{}@university.edu", i),
                    signature: vec![1, 2, 3, 4, 5],
                    timestamp: chrono::Utc::now(),
                    notes: Some("Verified".to_string()),
                };
                claim.add_verification(verification);
            }

            black_box(claim)
        });
    });
}

fn bench_provenance_add(c: &mut Criterion) {
    let content = ClaimContent {
        dataset_hash: hash::hash_bytes(b"test data").to_string(),
        description: "NAD+ extends lifespan in mice".to_string(),
        category: "longevity".to_string(),
        keywords: vec!["NAD+".to_string(), "aging".to_string()],
        storage_ref: Some("ipfs://Qm123".to_string()),
        reproducibility_score: Some(0.85),
        license: Some("MIT".to_string()),
    };

    c.bench_function("provenance_add", |b| {
        b.iter(|| {
            let mut claim = DesciClaim::new(
                EpistemicTier::E0,
                content.clone(),
                "researcher@stanford.edu".to_string(),
            );

            for i in 0..5 {
                let prov = Provenance::new(
                    format!("DOI:10.1038/journal.{}", i),
                    "peer_reviewed_publication".to_string(),
                )
                .with_url(format!("https://doi.org/10.1038/journal.{}", i));
                claim.add_provenance(prov);
            }

            black_box(claim)
        });
    });
}

// =============================================================================
// STORAGE BENCHMARKS (4)
// =============================================================================

fn bench_storage_write(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let storage = MemoryStorage::new();

    c.bench_function("storage_write_1000_claims", |b| {
        b.iter(|| {
            rt.block_on(async {
                for i in 0..1000 {
                    let content = ClaimContent {
                        dataset_hash: hash::hash_bytes(format!("data_{}", i).as_bytes()).to_string(),
                        description: format!("Research finding #{}", i),
                        category: "test".to_string(),
                        keywords: vec!["test".to_string()],
                        storage_ref: None,
                        reproducibility_score: Some(0.8),
                        license: Some("MIT".to_string()),
                    };
                    let claim = DesciClaim::new(
                        EpistemicTier::E0,
                        content,
                        "researcher@test.edu".to_string(),
                    );
                    storage.store(&claim).await.unwrap();
                }
                black_box(())
            })
        });
    });
}

fn bench_storage_read(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let storage = MemoryStorage::new();

    // Pre-populate storage and collect IDs
    let ids: Vec<_> = rt.block_on(async {
        let mut claim_ids = Vec::new();
        for i in 0..100 {
            let content = ClaimContent {
                dataset_hash: hash::hash_bytes(format!("data_{}", i).as_bytes()).to_string(),
                description: format!("Research finding #{}", i),
                category: "test".to_string(),
                keywords: vec!["test".to_string()],
                storage_ref: None,
                reproducibility_score: Some(0.8),
                license: Some("MIT".to_string()),
            };
            let claim = DesciClaim::new(
                EpistemicTier::E0,
                content,
                "researcher@test.edu".to_string(),
            );
            claim_ids.push(claim.id.to_string());
            storage.store(&claim).await.unwrap();
        }
        claim_ids
    });

    c.bench_function("storage_read_100_claims", |b| {
        b.iter(|| {
            rt.block_on(async {
                for id in &ids {
                    let _ = storage.retrieve(id).await.unwrap();
                }
                black_box(())
            })
        });
    });
}

fn bench_storage_concurrent(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("storage_concurrent_10_threads", |b| {
        b.iter(|| {
            rt.block_on(async {
                let storage = MemoryStorage::new();
                let storage = Arc::new(storage);

                let mut handles = vec![];
                for thread_id in 0..10 {
                    let storage = storage.clone();
                    let handle = tokio::spawn(async move {
                        for i in 0..10 {
                            let content = ClaimContent {
                                dataset_hash: hash::hash_bytes(
                                    format!("data_{}_{}", thread_id, i).as_bytes()
                                ).to_string(),
                                description: format!("Finding {} from thread {}", i, thread_id),
                                category: "test".to_string(),
                                keywords: vec!["test".to_string()],
                                storage_ref: None,
                                reproducibility_score: Some(0.8),
                                license: Some("MIT".to_string()),
                            };
                            let claim = DesciClaim::new(
                                EpistemicTier::E0,
                                content,
                                format!("researcher_{}@test.edu", thread_id),
                            );
                            storage.store(&claim).await.unwrap();
                        }
                    });
                    handles.push(handle);
                }

                for handle in handles {
                    handle.await.unwrap();
                }
                black_box(())
            })
        });
    });
}

fn bench_storage_bulk(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("storage_bulk_retrieve_100", |b| {
        b.iter(|| {
            rt.block_on(async {
                let storage = MemoryStorage::new();
                let mut claim_ids = Vec::new();

                // Store 100 claims
                for i in 0..100 {
                    let content = ClaimContent {
                        dataset_hash: hash::hash_bytes(format!("data_{}", i).as_bytes()).to_string(),
                        description: format!("Research finding #{}", i),
                        category: "test".to_string(),
                        keywords: vec!["test".to_string()],
                        storage_ref: None,
                        reproducibility_score: Some(0.8),
                        license: Some("MIT".to_string()),
                    };
                    let claim = DesciClaim::new(
                        EpistemicTier::E0,
                        content,
                        "researcher@test.edu".to_string(),
                    );
                    claim_ids.push(claim.id.to_string());
                    storage.store(&claim).await.unwrap();
                }

                // Retrieve all claims
                for id in claim_ids {
                    let _ = storage.retrieve(&id).await.unwrap();
                }
                black_box(())
            })
        });
    });
}

// =============================================================================
// QUERY BENCHMARKS (5)
// =============================================================================

fn bench_index_build(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("query_index_build_1000_claims", |b| {
        b.iter(|| {
            rt.block_on(async {
                let storage = MemoryStorage::new();
                let storage_arc: Arc<dyn StorageBackend> = Arc::new(storage.clone());
                let query_engine = QueryEngine::new(storage_arc);

                for i in 0..1000 {
                    let content = ClaimContent {
                        dataset_hash: hash::hash_bytes(format!("data_{}", i).as_bytes()).to_string(),
                        description: format!("Research finding #{}", i),
                        category: if i % 3 == 0 { "longevity" } else { "genomics" }.to_string(),
                        keywords: vec!["test".to_string(), format!("keyword_{}", i % 10)],
                        storage_ref: None,
                        reproducibility_score: Some(0.8),
                        license: Some("MIT".to_string()),
                    };
                    let claim = DesciClaim::new(
                        if i % 5 == 0 { EpistemicTier::E3 } else { EpistemicTier::E0 },
                        content,
                        "researcher@test.edu".to_string(),
                    );
                    query_engine.add_claim(&claim).await;
                }
                black_box(())
            })
        });
    });
}

fn bench_category_query(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (storage, query_engine) = rt.block_on(async {
        let storage = MemoryStorage::new();
        let storage_arc: Arc<dyn StorageBackend> = Arc::new(storage.clone());
        let query_engine = QueryEngine::new(storage_arc);

        // Pre-populate
        for i in 0..1000 {
            let content = ClaimContent {
                dataset_hash: hash::hash_bytes(format!("data_{}", i).as_bytes()).to_string(),
                description: format!("Research finding #{}", i),
                category: if i % 3 == 0 { "longevity" } else { "genomics" }.to_string(),
                keywords: vec!["test".to_string()],
                storage_ref: None,
                reproducibility_score: Some(0.8),
                license: Some("MIT".to_string()),
            };
            let claim = DesciClaim::new(
                EpistemicTier::E0,
                content,
                "researcher@test.edu".to_string(),
            );
            storage.store(&claim).await.unwrap();
            query_engine.add_claim(&claim).await;
        }
        (storage, query_engine)
    });

    c.bench_function("query_category_filter", |b| {
        b.iter(|| {
            rt.block_on(async {
                let filter = QueryFilter::new().with_category("longevity".to_string());
                let results = query_engine.query(&filter).await.unwrap();
                black_box(results)
            })
        });
    });
}

fn bench_keyword_search(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (storage, query_engine) = rt.block_on(async {
        let storage = MemoryStorage::new();
        let storage_arc: Arc<dyn StorageBackend> = Arc::new(storage.clone());
        let query_engine = QueryEngine::new(storage_arc);

        // Pre-populate
        for i in 0..1000 {
            let content = ClaimContent {
                dataset_hash: hash::hash_bytes(format!("data_{}", i).as_bytes()).to_string(),
                description: format!("Research finding #{}", i),
                category: "test".to_string(),
                keywords: vec!["NAD+".to_string(), format!("keyword_{}", i % 50)],
                storage_ref: None,
                reproducibility_score: Some(0.8),
                license: Some("MIT".to_string()),
            };
            let claim = DesciClaim::new(
                EpistemicTier::E0,
                content,
                "researcher@test.edu".to_string(),
            );
            storage.store(&claim).await.unwrap();
            query_engine.add_claim(&claim).await;
        }
        (storage, query_engine)
    });

    c.bench_function("query_keyword_search", |b| {
        b.iter(|| {
            rt.block_on(async {
                let filter = QueryFilter::new().with_keyword("NAD+".to_string());
                let results = query_engine.query(&filter).await.unwrap();
                black_box(results)
            })
        });
    });
}

fn bench_complex_filter(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (storage, query_engine) = rt.block_on(async {
        let storage = MemoryStorage::new();
        let storage_arc: Arc<dyn StorageBackend> = Arc::new(storage.clone());
        let query_engine = QueryEngine::new(storage_arc);

        // Pre-populate
        for i in 0..1000 {
            let content = ClaimContent {
                dataset_hash: hash::hash_bytes(format!("data_{}", i).as_bytes()).to_string(),
                description: format!("Research finding #{}", i),
                category: if i % 3 == 0 { "longevity" } else { "genomics" }.to_string(),
                keywords: vec!["NAD+".to_string(), format!("keyword_{}", i % 10)],
                storage_ref: None,
                reproducibility_score: Some(0.8),
                license: Some("MIT".to_string()),
            };
            let mut claim = DesciClaim::new(
                EpistemicTier::E0,
                content,
                "researcher@test.edu".to_string(),
            );

            // Add verifications to some claims
            if i % 5 == 0 {
                for j in 0..3 {
                    let verification = Verification {
                        verifier: format!("peer_{}@test.edu", j),
                        signature: vec![1, 2, 3],
                        timestamp: chrono::Utc::now(),
                        notes: Some("Verified".to_string()),
                    };
                    claim.add_verification(verification);
                }
            }

            storage.store(&claim).await.unwrap();
            query_engine.add_claim(&claim).await;
        }
        (storage, query_engine)
    });

    c.bench_function("query_complex_multi_filter", |b| {
        b.iter(|| {
            rt.block_on(async {
                let filter = QueryFilter::new()
                    .with_category("longevity".to_string())
                    .with_keyword("NAD+".to_string())
                    .with_min_tier(EpistemicTier::E2)
                    .with_sort(SortBy::EpistemicTier, SortOrder::Descending);
                let results = query_engine.query(&filter).await.unwrap();
                black_box(results)
            })
        });
    });
}

fn bench_pagination(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (storage, query_engine) = rt.block_on(async {
        let storage = MemoryStorage::new();
        let storage_arc: Arc<dyn StorageBackend> = Arc::new(storage.clone());
        let query_engine = QueryEngine::new(storage_arc);

        // Pre-populate
        for i in 0..1000 {
            let content = ClaimContent {
                dataset_hash: hash::hash_bytes(format!("data_{}", i).as_bytes()).to_string(),
                description: format!("Research finding #{}", i),
                category: "test".to_string(),
                keywords: vec!["test".to_string()],
                storage_ref: None,
                reproducibility_score: Some(0.8),
                license: Some("MIT".to_string()),
            };
            let claim = DesciClaim::new(
                EpistemicTier::E0,
                content,
                "researcher@test.edu".to_string(),
            );
            storage.store(&claim).await.unwrap();
            query_engine.add_claim(&claim).await;
        }
        (storage, query_engine)
    });

    c.bench_function("query_pagination_10_per_page", |b| {
        b.iter(|| {
            rt.block_on(async {
                for page in 0..10 {
                    let filter = QueryFilter::new()
                        .with_offset(page * 10)
                        .with_limit(10);
                    let results = query_engine.query(&filter).await.unwrap();
                    black_box(results);
                }
            })
        });
    });
}

// =============================================================================
// HASH BENCHMARKS (3)
// =============================================================================

fn bench_blake3(c: &mut Criterion) {
    let data = vec![0u8; 1024 * 1024]; // 1MB of data

    c.bench_function("hash_blake3_1mb", |b| {
        b.iter(|| {
            black_box(hash::hash_bytes(&data))
        });
    });
}

fn bench_sha256(c: &mut Criterion) {
    let data = vec![0u8; 1024 * 1024]; // 1MB of data

    c.bench_function("hash_sha256_1mb", |b| {
        b.iter(|| {
            use sha2::{Sha256, Digest};
            let mut hasher = Sha256::new();
            hasher.update(&data);
            black_box(hasher.finalize())
        });
    });
}

fn bench_merkle_tree(c: &mut Criterion) {
    let leaves: Vec<_> = (0..1000)
        .map(|i| hash::hash_bytes(format!("data_{}", i).as_bytes()))
        .collect();

    c.bench_function("hash_merkle_tree_1000_leaves", |b| {
        b.iter(|| {
            let tree = hash::build_merkle_tree(leaves.clone()).unwrap();
            black_box(tree)
        });
    });
}

// =============================================================================
// TRUST BENCHMARKS (3)
// =============================================================================

fn bench_trust_update(c: &mut Criterion) {
    c.bench_function("trust_update_1000_scores", |b| {
        b.iter(|| {
            let mut trust_manager = TrustManager::new();
            for i in 0..1000 {
                trust_manager
                    .update_score(&format!("peer_{}@university.edu", i), true, 0.8)
                    .unwrap();
            }
            black_box(trust_manager)
        });
    });
}

fn bench_trust_query(c: &mut Criterion) {
    let mut trust_manager = TrustManager::new();

    // Pre-populate
    for i in 0..1000 {
        trust_manager
            .update_score(&format!("peer_{}@university.edu", i), true, 0.8)
            .unwrap();
    }

    c.bench_function("trust_query_1000_participants", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let score = trust_manager.get_score(&format!("peer_{}@university.edu", i));
                black_box(score);
            }
        });
    });
}

fn bench_trust_decay(c: &mut Criterion) {
    c.bench_function("trust_decay_100_participants", |b| {
        b.iter(|| {
            let mut trust_manager = TrustManager::new();

            // Pre-populate with scores
            for i in 0..100 {
                trust_manager
                    .update_score(&format!("peer_{}@university.edu", i), true, 0.9)
                    .unwrap();
            }

            trust_manager.apply_decay(); // Apply decay
            black_box(trust_manager)
        });
    });
}

// =============================================================================
// BENCHMARK GROUPS
// =============================================================================

criterion_group!(
    claims_benches,
    bench_claim_creation,
    bench_claim_serialization,
    bench_claim_validation,
    bench_tier_upgrade,
    bench_provenance_add
);

criterion_group!(
    storage_benches,
    bench_storage_write,
    bench_storage_read,
    bench_storage_concurrent,
    bench_storage_bulk
);

criterion_group!(
    query_benches,
    bench_index_build,
    bench_category_query,
    bench_keyword_search,
    bench_complex_filter,
    bench_pagination
);

criterion_group!(
    hash_benches,
    bench_blake3,
    bench_sha256,
    bench_merkle_tree
);

criterion_group!(
    trust_benches,
    bench_trust_update,
    bench_trust_query,
    bench_trust_decay
);

criterion_main!(
    claims_benches,
    storage_benches,
    query_benches,
    hash_benches,
    trust_benches
);
