//! Core Operations Performance Benchmarks
//!
//! This benchmark suite establishes baseline performance metrics for critical
//! Mycelix operations. These benchmarks are used for:
//! - Detecting performance regressions in CI
//! - Establishing performance targets for optimization
//! - Comparing different implementation strategies
//!
//! # Performance Targets (P50/P95/P99)
//!
//! | Operation          | P50 Target | P95 Target | P99 Target |
//! |--------------------|------------|------------|------------|
//! | DID Resolution     | 20ms       | 50ms       | 100ms      |
//! | Market Trade (FL)  | 100ms      | 300ms      | 500ms      |
//! | Reputation Query   | 30ms       | 80ms       | 200ms      |
//! | Credential Verify  | 50ms       | 150ms      | 300ms      |
//!
//! Run with: `cargo bench --bench core_operations`
//! Compare against baseline: `cargo bench --bench core_operations -- --baseline main`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// Import Mycelix SDK types
use mycelix_sdk::bridge::{CrossHappReputation, HappReputationScore};
use mycelix_sdk::credentials::VerifiableCredential;
use mycelix_sdk::dkg::{query_triples, QueryFilter, TripleValue, VerifiableTriple, URI};
use mycelix_sdk::epistemic::{
    EmpiricalLevel, EpistemicClaim, EpistemicClassification, MaterialityLevel, NormativeLevel,
};
use mycelix_sdk::fl::{
    coordinate_median, fedavg, fedavg_optimized, krum, trimmed_mean, EarlyByzantineDetector,
    GradientAccumulator, GradientStats, GradientUpdate,
};
use mycelix_sdk::identity::WebAuthnService;
use mycelix_sdk::matl::{CachedKVector, KVectorBatch, KVECTOR_WEIGHTS};
use mycelix_sdk::matl::{
    CartelDetector, GovernanceTier, KVector, ProofOfGradientQuality, RbBftConfig, RbBftConsensus,
};
use mycelix_sdk::storage::StorageRouter;

// =============================================================================
// DID RESOLUTION BENCHMARKS
// =============================================================================

/// Benchmark DID resolution simulation via K-Vector trust lookup
///
/// In the Mycelix system, DID resolution involves:
/// 1. Looking up the agent's K-Vector trust state
/// 2. Computing the trust score from 9 dimensions
/// 3. Determining governance tier
///
/// Target: P50 < 20ms, P95 < 50ms, P99 < 100ms
fn benchmark_did_resolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("did_resolution");

    // Single DID resolution
    group.bench_function("single", |b| {
        let kvector = create_test_kvector();
        b.iter(|| {
            // Simulate DID resolution: compute trust score and governance tier
            let trust_score = black_box(&kvector).trust_score();
            let tier = GovernanceTier::from_trust_score(trust_score);
            (trust_score, tier)
        })
    });

    // Batch DID resolution (common in permission checks)
    for count in [10, 50, 100, 500] {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("batch", count), &count, |b, &count| {
            let kvectors: Vec<KVector> = (0..count).map(|i| create_kvector_with_seed(i)).collect();

            b.iter(|| {
                kvectors
                    .iter()
                    .map(|kv| {
                        let score = kv.trust_score();
                        (score, GovernanceTier::from_trust_score(score))
                    })
                    .collect::<Vec<_>>()
            })
        });
    }

    group.finish();
}

/// Benchmark identity verification via WebAuthn service
fn benchmark_identity_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("identity_verification");

    group.bench_function("create_registration_challenge", |b| {
        let mut service = WebAuthnService::new(
            "mycelix.example.com".to_string(),
            "https://mycelix.example.com".to_string(),
        );
        let user_id = b"did:mycelix:agent123".to_vec();

        b.iter(|| service.create_registration_challenge(black_box(&user_id), black_box("testuser")))
    });

    group.bench_function("create_auth_challenge", |b| {
        let mut service = WebAuthnService::new(
            "mycelix.example.com".to_string(),
            "https://mycelix.example.com".to_string(),
        );
        let credential_ids = vec![b"credential123".to_vec()];

        b.iter(|| service.create_authentication_challenge(black_box(&credential_ids)))
    });

    group.finish();
}

// =============================================================================
// MARKET TRADE / FL AGGREGATION BENCHMARKS
// =============================================================================

/// Benchmark FL aggregation operations (analogous to market trades)
///
/// Federated Learning aggregation is the core "trading" mechanism in Mycelix:
/// - FedAvg: Standard weighted averaging
/// - Krum: Byzantine-resistant selection
/// - TrimmedMean: Outlier-resistant averaging
/// - CoordinateMedian: Robust median aggregation
///
/// Target: P50 < 100ms, P95 < 300ms, P99 < 500ms
fn benchmark_market_trade(c: &mut Criterion) {
    let mut group = c.benchmark_group("market_trade");

    // Benchmark FedAvg aggregation at different participant counts
    for participant_count in [10, 50, 100, 500] {
        let gradient_size = 1000; // Typical model gradient size
        group.throughput(Throughput::Elements(participant_count as u64));

        group.bench_with_input(
            BenchmarkId::new("fedavg", participant_count),
            &participant_count,
            |b, &count| {
                let updates = create_gradient_updates(count, gradient_size);

                b.iter(|| fedavg(black_box(&updates)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("krum", participant_count),
            &participant_count,
            |b, &count| {
                let updates = create_gradient_updates(count.max(3), gradient_size);

                b.iter(|| krum(black_box(&updates), 1))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("trimmed_mean", participant_count),
            &participant_count,
            |b, &count| {
                let updates = create_gradient_updates(count, gradient_size);

                b.iter(|| trimmed_mean(black_box(&updates), 0.1))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("coordinate_median", participant_count),
            &participant_count,
            |b, &count| {
                let updates = create_gradient_updates(count, gradient_size);

                b.iter(|| coordinate_median(black_box(&updates)))
            },
        );
    }

    group.finish();
}

/// Benchmark PoGQ (Proof of Gradient Quality) computation
fn benchmark_pogq_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("pogq");

    group.bench_function("create", |b| {
        b.iter(|| {
            ProofOfGradientQuality::new(
                black_box(0.95), // quality
                black_box(0.88), // consistency
                black_box(0.12), // entropy
            )
        })
    });

    group.bench_function("composite_score", |b| {
        let pogq = ProofOfGradientQuality::new(0.95, 0.88, 0.12);

        b.iter(|| pogq.composite_score(black_box(0.75)))
    });

    group.finish();
}

// =============================================================================
// REPUTATION QUERY BENCHMARKS
// =============================================================================

/// Benchmark reputation queries and calculations
///
/// Reputation operations include:
/// - K-Vector trust score computation
/// - Cross-hApp reputation aggregation
/// - Cartel detection
///
/// Target: P50 < 30ms, P95 < 80ms, P99 < 200ms
fn benchmark_reputation_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("reputation_query");

    // K-Vector trust score
    group.bench_function("kvector_trust_score", |b| {
        let kvector = create_test_kvector();

        b.iter(|| black_box(&kvector).trust_score())
    });

    // K-Vector governance tier determination
    group.bench_function("kvector_governance_tier", |b| {
        let kvector = create_test_kvector();

        b.iter(|| {
            let score = black_box(&kvector).trust_score();
            GovernanceTier::from_trust_score(score)
        })
    });

    // Cross-hApp reputation aggregation
    group.bench_function("cross_happ_reputation", |b| {
        let scores = create_happ_reputation_scores(5);

        b.iter(|| {
            CrossHappReputation::from_scores(
                black_box("did:mycelix:agent123"),
                black_box(scores.clone()),
            )
        })
    });

    // Cartel detection
    for node_count in [10, 50, 100, 200] {
        group.bench_with_input(
            BenchmarkId::new("cartel_detection", node_count),
            &node_count,
            |b, &count| {
                let mut detector = CartelDetector::new(0.8, 3);

                // Build similarity matrix
                for i in 0..count {
                    for j in (i + 1)..count {
                        let similarity = if i < 5 && j < 5 {
                            0.9 + (rand_float(i * count + j) * 0.1) as f64
                        } else {
                            rand_float(i * count + j) as f64 * 0.5
                        };
                        detector.record_similarity(
                            &format!("node_{}", i),
                            &format!("node_{}", j),
                            similarity,
                        );
                    }
                }

                b.iter(|| {
                    let mut d = detector.clone();
                    d.detect()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark DKG query operations
fn benchmark_dkg_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("dkg_query");

    // Query performance with varying triple counts
    for triple_count in [100, 1000, 10000] {
        group.bench_with_input(
            BenchmarkId::new("query_triples", triple_count),
            &triple_count,
            |b, &count| {
                let triples = create_test_triples(count);
                let filter = QueryFilter::new()
                    .with_subject("did:mycelix:subject")
                    .with_limit(100);
                let current_time = 1234567890u64;

                b.iter(|| {
                    query_triples(
                        black_box(&triples),
                        black_box(&filter),
                        black_box(current_time),
                    )
                })
            },
        );
    }

    // Filter by predicate
    group.bench_function("query_by_predicate", |b| {
        let triples = create_test_triples(1000);
        let filter = QueryFilter::new()
            .with_predicate("has_property_5")
            .with_limit(50);
        let current_time = 1234567890u64;

        b.iter(|| {
            query_triples(
                black_box(&triples),
                black_box(&filter),
                black_box(current_time),
            )
        })
    });

    group.finish();
}

// =============================================================================
// CREDENTIAL VERIFICATION BENCHMARKS
// =============================================================================

/// Benchmark credential operations
///
/// Verifiable credential operations include:
/// - Credential creation
/// - Expiration checking
/// - Proof verification (placeholder - full crypto verification)
/// - Epistemic classification
///
/// Target: P50 < 50ms, P95 < 150ms, P99 < 300ms
fn benchmark_credential_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("credential_verify");

    // Credential building
    group.bench_function("build_credential", |b| {
        b.iter(|| {
            VerifiableCredential::builder()
                .issuer(black_box("did:mycelix:issuer"))
                .subject(black_box("did:mycelix:subject"))
                .credential_type(black_box("ReputationCredential"))
                .claims(serde_json::json!({
                    "reputation_score": 0.85,
                    "trust_level": "high",
                    "verified_at": "2024-01-15T00:00:00Z"
                }))
                .build()
        })
    });

    // Credential with epistemic classification
    group.bench_function("build_with_epistemic", |b| {
        b.iter(|| {
            let epistemic = EpistemicClaim::new(
                "Reputation credential",
                EmpiricalLevel::E3Cryptographic,
                NormativeLevel::N2Network,
                MaterialityLevel::M2Persistent,
            );

            VerifiableCredential::builder()
                .issuer(black_box("did:mycelix:issuer"))
                .subject(black_box("did:mycelix:subject"))
                .credential_type(black_box("ReputationCredential"))
                .epistemic(epistemic)
                .build()
        })
    });

    // Expiration check
    group.bench_function("check_expiration", |b| {
        let vc = VerifiableCredential::builder()
            .issuer("did:mycelix:issuer")
            .subject("did:mycelix:subject")
            .expires("2025-12-31T23:59:59Z")
            .build();

        b.iter(|| black_box(&vc).is_expired())
    });

    // Proof presence check
    group.bench_function("check_proof", |b| {
        let vc = VerifiableCredential::builder()
            .issuer("did:mycelix:issuer")
            .subject("did:mycelix:subject")
            .build();

        b.iter(|| black_box(&vc).has_proof())
    });

    // Epistemic code extraction
    group.bench_function("epistemic_code", |b| {
        let epistemic = EpistemicClaim::new(
            "Test claim",
            EmpiricalLevel::E3Cryptographic,
            NormativeLevel::N2Network,
            MaterialityLevel::M2Persistent,
        );
        let vc = VerifiableCredential::builder()
            .issuer("did:mycelix:issuer")
            .subject("did:mycelix:subject")
            .epistemic(epistemic)
            .build();

        b.iter(|| black_box(&vc).epistemic_code())
    });

    // Batch credential verification
    for count in [10, 50, 100] {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_verify", count),
            &count,
            |b, &count| {
                let credentials: Vec<VerifiableCredential> = (0..count)
                    .map(|i| {
                        VerifiableCredential::builder()
                            .issuer(&format!("did:mycelix:issuer{}", i))
                            .subject(&format!("did:mycelix:subject{}", i))
                            .build()
                    })
                    .collect();

                b.iter(|| {
                    credentials
                        .iter()
                        .map(|vc| (!vc.is_expired(), vc.has_proof()))
                        .collect::<Vec<_>>()
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// STORAGE ROUTING BENCHMARKS
// =============================================================================

/// Benchmark storage routing decisions
fn benchmark_storage_routing(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_routing");

    group.bench_function("route_decision", |b| {
        let router = StorageRouter::default_router();
        let classification = EpistemicClassification {
            empirical: EmpiricalLevel::E2PrivateVerify,
            normative: NormativeLevel::N1Communal,
            materiality: MaterialityLevel::M2Persistent,
        };

        b.iter(|| router.route(black_box(&classification)))
    });

    // Batch routing
    for count in [10, 100, 1000] {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_route", count),
            &count,
            |b, &count| {
                let router = StorageRouter::default_router();
                let classifications: Vec<EpistemicClassification> = (0..count)
                    .map(|i| EpistemicClassification {
                        empirical: match i % 5 {
                            0 => EmpiricalLevel::E0Null,
                            1 => EmpiricalLevel::E1Testimonial,
                            2 => EmpiricalLevel::E2PrivateVerify,
                            3 => EmpiricalLevel::E3Cryptographic,
                            _ => EmpiricalLevel::E4PublicRepro,
                        },
                        normative: match i % 4 {
                            0 => NormativeLevel::N0Personal,
                            1 => NormativeLevel::N1Communal,
                            2 => NormativeLevel::N2Network,
                            _ => NormativeLevel::N3Axiomatic,
                        },
                        materiality: match i % 4 {
                            0 => MaterialityLevel::M0Ephemeral,
                            1 => MaterialityLevel::M1Temporal,
                            2 => MaterialityLevel::M2Persistent,
                            _ => MaterialityLevel::M3Foundational,
                        },
                    })
                    .collect();

                b.iter(|| {
                    classifications
                        .iter()
                        .map(|c| router.route(c))
                        .collect::<Vec<_>>()
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// RB-BFT CONSENSUS BENCHMARKS
// =============================================================================

/// Benchmark RB-BFT consensus operations
fn benchmark_rbbft_consensus(c: &mut Criterion) {
    let mut group = c.benchmark_group("rbbft_consensus");

    group.bench_function("create_consensus", |b| {
        let config = RbBftConfig::default();

        b.iter(|| RbBftConsensus::new(black_box(config.clone())))
    });

    group.finish();
}

// =============================================================================
// OPTIMIZATION COMPARISON BENCHMARKS
// =============================================================================

/// Benchmark original vs optimized K-Vector trust computation
fn benchmark_kvector_optimizations(c: &mut Criterion) {
    let mut group = c.benchmark_group("kvector_optimizations");

    let kv = create_test_kvector();

    // Original method
    group.bench_function("trust_score_original", |b| {
        b.iter(|| black_box(&kv).trust_score_with_weights(&KVECTOR_WEIGHTS))
    });

    // Optimized fast method
    group.bench_function("trust_score_fast", |b| {
        b.iter(|| black_box(&kv).trust_score_fast())
    });

    // Cached K-Vector (repeated access)
    group.bench_function("cached_kvector_5x_access", |b| {
        let cached = CachedKVector::new(kv);
        b.iter(|| {
            let c = black_box(&cached);
            c.trust_score() + c.trust_score() + c.trust_score() + c.trust_score() + c.trust_score()
        })
    });

    // Uncached repeated access
    group.bench_function("uncached_kvector_5x_access", |b| {
        b.iter(|| {
            let k = black_box(&kv);
            k.trust_score() + k.trust_score() + k.trust_score() + k.trust_score() + k.trust_score()
        })
    });

    group.finish();
}

/// Benchmark batch K-Vector operations
fn benchmark_kvector_batch_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("kvector_batch_ops");

    for size in [10, 100, 1000] {
        let vectors: Vec<KVector> = (0..size).map(|i| create_kvector_with_seed(i)).collect();

        group.throughput(Throughput::Elements(size as u64));

        // Individual computation loop
        group.bench_with_input(
            BenchmarkId::new("individual_loop", size),
            &vectors,
            |b, v| b.iter(|| v.iter().map(|kv| kv.trust_score()).collect::<Vec<_>>()),
        );

        // Batch computation
        group.bench_with_input(BenchmarkId::new("batch_compute", size), &vectors, |b, v| {
            b.iter(|| KVectorBatch::compute_trust_scores(black_box(v)))
        });

        // Filter by threshold
        group.bench_with_input(
            BenchmarkId::new("filter_threshold", size),
            &vectors,
            |b, v| b.iter(|| KVectorBatch::filter_by_threshold(black_box(v), 0.5)),
        );
    }

    group.finish();
}

/// Benchmark FL aggregation optimizations
fn benchmark_fl_optimizations(c: &mut Criterion) {
    let mut group = c.benchmark_group("fl_optimizations");

    for (participant_count, gradient_size) in [(20, 500), (50, 1000), (100, 2000)] {
        let updates = create_gradient_updates(participant_count, gradient_size);
        let label = format!("{}x{}", participant_count, gradient_size);

        group.throughput(Throughput::Elements(
            (participant_count * gradient_size) as u64,
        ));

        // Original FedAvg
        group.bench_with_input(
            BenchmarkId::new("fedavg_original", &label),
            &updates,
            |b, u| b.iter(|| fedavg(black_box(u))),
        );

        // Optimized FedAvg
        group.bench_with_input(
            BenchmarkId::new("fedavg_optimized", &label),
            &updates,
            |b, u| b.iter(|| fedavg_optimized(black_box(u))),
        );

        // Gradient accumulator
        group.bench_with_input(BenchmarkId::new("accumulator", &label), &updates, |b, u| {
            b.iter(|| {
                let total_samples: usize = u.iter().map(|upd| upd.metadata.batch_size).sum();
                let mut acc = GradientAccumulator::new(u[0].gradients.len());
                for upd in u {
                    let weight = upd.metadata.batch_size as f64 / total_samples as f64;
                    acc.accumulate(upd, weight);
                }
                acc.finalize()
            })
        });
    }

    group.finish();
}

/// Benchmark Byzantine detection optimizations
fn benchmark_byzantine_optimizations(c: &mut Criterion) {
    let mut group = c.benchmark_group("byzantine_optimizations");

    for participant_count in [20, 50, 100, 200] {
        // Create updates with some Byzantine participants
        let updates: Vec<GradientUpdate> = (0..participant_count)
            .map(|i| {
                let grad_val = if i % 10 == 0 {
                    1000.0 // Byzantine
                } else {
                    0.1 + (i as f64 * 0.001)
                };
                GradientUpdate::new(
                    format!("p{}", i),
                    1,
                    vec![grad_val, grad_val * 1.1, grad_val * 0.9],
                    100,
                    0.5,
                )
            })
            .collect();

        // Early Byzantine detection
        group.bench_with_input(
            BenchmarkId::new("early_detection", participant_count),
            &updates,
            |b, u| {
                let detector = EarlyByzantineDetector::new();
                b.iter(|| detector.detect(black_box(u)))
            },
        );

        // Filter and detect
        group.bench_with_input(
            BenchmarkId::new("filter_updates", participant_count),
            &updates,
            |b, u| {
                let detector = EarlyByzantineDetector::new();
                b.iter(|| detector.filter_updates(black_box(u)))
            },
        );

        // Gradient statistics
        group.bench_with_input(
            BenchmarkId::new("gradient_stats", participant_count),
            &updates,
            |b, u| b.iter(|| GradientStats::compute(black_box(u))),
        );
    }

    group.finish();
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Create a test K-Vector with realistic values
fn create_test_kvector() -> KVector {
    KVector::new(
        0.8, // k_r: Reliability
        0.6, // k_a: Activity
        0.9, // k_i: Integrity
        0.7, // k_p: Performance
        0.5, // k_m: Membership
        0.4, // k_s: Stake
        0.6, // k_h: Historical
        0.3, // k_topo: Topology
        0.7, // k_v: Verification
    )
}

/// Create a K-Vector with seed for variation
fn create_kvector_with_seed(seed: usize) -> KVector {
    let base = rand_float(seed);
    KVector::new(
        (base + 0.1).min(1.0),
        (base + 0.05).min(1.0),
        (base + 0.15).min(1.0),
        (base + 0.02).min(1.0),
        (base - 0.1).max(0.0),
        (base - 0.05).max(0.0),
        (base + 0.08).min(1.0),
        (base - 0.2).max(0.0),
        (base + 0.12).min(1.0),
    )
}

/// Create gradient updates for FL benchmarks
fn create_gradient_updates(count: usize, gradient_size: usize) -> Vec<GradientUpdate> {
    (0..count)
        .map(|i| {
            let gradients: Vec<f64> = (0..gradient_size)
                .map(|j| (rand_float(i * gradient_size + j) as f64) * 0.1 - 0.05)
                .collect();

            GradientUpdate::new(
                format!("participant_{}", i),
                1, // model version
                gradients,
                100 + (i % 50),                     // batch size variation
                0.5 - (rand_float(i) as f64) * 0.1, // loss
            )
        })
        .collect()
}

/// Create test triples for DKG benchmarks
fn create_test_triples(count: usize) -> Vec<VerifiableTriple> {
    (0..count)
        .map(|i| {
            let subject = if i % 10 == 0 {
                "did:mycelix:subject".to_string()
            } else {
                format!("did:mycelix:subject_{}", i)
            };

            VerifiableTriple::new(
                subject,
                URI::new(format!("has_property_{}", i % 20)),
                TripleValue::String(format!("value_{}", i)),
            )
            .with_timestamp(1234567890 + i as u64)
            .with_creator(format!("creator_{}", i % 5))
        })
        .collect()
}

/// Create HappReputationScore instances for cross-hApp reputation benchmarks
fn create_happ_reputation_scores(count: usize) -> Vec<HappReputationScore> {
    (0..count)
        .map(|i| HappReputationScore {
            happ_id: format!("happ_{}", i),
            happ_name: format!("Mycelix App {}", i),
            score: 0.5 + (rand_float(i) as f64) * 0.5,
            interactions: 10 + (i as u64 * 5),
            last_updated: 1234567890,
        })
        .collect()
}

/// Simple deterministic pseudo-random for benchmarks (returns f32)
fn rand_float(seed: usize) -> f32 {
    let x = seed.wrapping_mul(1103515245).wrapping_add(12345);
    (x % 1000) as f32 / 1000.0
}

// =============================================================================
// CRITERION GROUPS
// =============================================================================

criterion_group!(
    name = did_benchmarks;
    config = Criterion::default().sample_size(100);
    targets = benchmark_did_resolution, benchmark_identity_verification
);

criterion_group!(
    name = trade_benchmarks;
    config = Criterion::default().sample_size(50);
    targets = benchmark_market_trade, benchmark_pogq_computation
);

criterion_group!(
    name = reputation_benchmarks;
    config = Criterion::default().sample_size(100);
    targets = benchmark_reputation_query, benchmark_dkg_query
);

criterion_group!(
    name = credential_benchmarks;
    config = Criterion::default().sample_size(100);
    targets = benchmark_credential_verify
);

criterion_group!(
    name = infrastructure_benchmarks;
    config = Criterion::default().sample_size(100);
    targets = benchmark_storage_routing, benchmark_rbbft_consensus
);

criterion_group!(
    name = optimization_benchmarks;
    config = Criterion::default().sample_size(100);
    targets = benchmark_kvector_optimizations, benchmark_kvector_batch_ops,
              benchmark_fl_optimizations, benchmark_byzantine_optimizations
);

criterion_main!(
    did_benchmarks,
    trade_benchmarks,
    reputation_benchmarks,
    credential_benchmarks,
    infrastructure_benchmarks,
    optimization_benchmarks
);
