//! Benchmarks for zkSTARK proof generation and verification
//!
//! Run with: cargo bench --features proofs --bench proof_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

#[cfg(feature = "proofs")]
mod proofs_benchmarks {
    use super::*;
    use fl_aggregator::proofs::{
        // Circuits
        RangeProof, MembershipProof, GradientIntegrityProof,
        IdentityAssuranceProof, VoteEligibilityProof,
        // Types
        ProofConfig, SecurityLevel, ProofAssuranceLevel,
        ProofIdentityFactor, ProofProposalType, ProofVoterProfile,
        // Merkle helpers
        build_merkle_tree,
    };

    fn config_96() -> ProofConfig {
        ProofConfig {
            security_level: SecurityLevel::Standard96,
            parallel: false,
            max_proof_size: 0,
        }
    }

    fn config_128() -> ProofConfig {
        ProofConfig {
            security_level: SecurityLevel::Standard128,
            parallel: false,
            max_proof_size: 0,
        }
    }

    pub fn bench_range_proof(c: &mut Criterion) {
        let mut group = c.benchmark_group("RangeProof");

        // Generation
        group.bench_function("generate_96", |b| {
            b.iter(|| {
                RangeProof::generate(black_box(500), black_box(0), black_box(1000), config_96())
                    .unwrap()
            })
        });

        group.bench_function("generate_128", |b| {
            b.iter(|| {
                RangeProof::generate(black_box(500), black_box(0), black_box(1000), config_128())
                    .unwrap()
            })
        });

        // Verification
        let proof = RangeProof::generate(500, 0, 1000, config_128()).unwrap();
        group.bench_function("verify", |b| {
            b.iter(|| proof.verify().unwrap())
        });

        // Proof size
        let size = proof.size();
        println!("RangeProof size: {} bytes", size);

        group.finish();
    }

    pub fn bench_membership_proof(c: &mut Criterion) {
        let mut group = c.benchmark_group("MembershipProof");

        // Build a merkle tree
        let leaves: Vec<[u8; 32]> = (0..16)
            .map(|i| {
                let mut leaf = [0u8; 32];
                leaf[0] = i as u8;
                leaf
            })
            .collect();
        let (root, all_paths) = build_merkle_tree(&leaves);

        // Use the actual path for leaf 5 from the tree
        let leaf_idx = 5;
        let path = all_paths[leaf_idx].clone();

        group.bench_function("generate_depth4", |b| {
            b.iter(|| {
                MembershipProof::generate(
                    black_box(leaves[leaf_idx]),
                    black_box(path.clone()),
                    black_box(root),
                    config_96(),
                ).unwrap()
            })
        });

        let proof = MembershipProof::generate(leaves[leaf_idx], path.clone(), root, config_128()).unwrap();
        group.bench_function("verify", |b| {
            b.iter(|| proof.verify().unwrap())
        });

        println!("MembershipProof size: {} bytes", proof.size());

        group.finish();
    }

    pub fn bench_gradient_proof(c: &mut Criterion) {
        let mut group = c.benchmark_group("GradientIntegrityProof");
        group.sample_size(20); // Fewer samples for expensive operations

        // Different gradient sizes - use small values to keep norm within bounds
        // For N elements with value v, L2 norm = v * sqrt(N)
        // To keep norm <= max_norm, use v = max_norm / (2 * sqrt(N))
        let max_norm = 10.0;
        for size in [10, 100, 1000].iter() {
            let scale = max_norm / (2.0 * (*size as f32).sqrt());
            let gradients: Vec<f32> = (0..*size)
                .map(|i| if i % 2 == 0 { scale } else { -scale })
                .collect();

            group.throughput(Throughput::Elements(*size as u64));
            group.bench_with_input(
                BenchmarkId::new("generate", size),
                &gradients,
                |b, g| {
                    b.iter(|| {
                        GradientIntegrityProof::generate(
                            black_box(g),
                            black_box(max_norm),
                            config_96(),
                        ).unwrap()
                    })
                },
            );
        }

        // Verification (use small gradient for speed)
        let small_gradient: Vec<f32> = (0..100).map(|i| if i % 2 == 0 { 0.5 } else { -0.5 }).collect();
        let proof = GradientIntegrityProof::generate(&small_gradient, max_norm, config_128()).unwrap();

        group.bench_function("verify_100", |b| {
            b.iter(|| proof.verify().unwrap())
        });

        println!("GradientIntegrityProof (100 elements) size: {} bytes", proof.size());

        group.finish();
    }

    pub fn bench_identity_proof(c: &mut Criterion) {
        let mut group = c.benchmark_group("IdentityAssuranceProof");

        let factors = vec![
            ProofIdentityFactor::new(0.4, 0, true),
            ProofIdentityFactor::new(0.3, 1, true),
            ProofIdentityFactor::new(0.2, 2, true),
        ];

        group.bench_function("generate_3factors", |b| {
            b.iter(|| {
                IdentityAssuranceProof::generate(
                    black_box("did:mycelix:bench"),
                    black_box(&factors),
                    black_box(ProofAssuranceLevel::E2),
                    config_96(),
                ).unwrap()
            })
        });

        let proof = IdentityAssuranceProof::generate(
            "did:mycelix:bench",
            &factors,
            ProofAssuranceLevel::E2,
            config_128(),
        ).unwrap();

        group.bench_function("verify", |b| {
            b.iter(|| proof.verify().unwrap())
        });

        println!("IdentityAssuranceProof size: {} bytes", proof.size());

        group.finish();
    }

    pub fn bench_vote_proof(c: &mut Criterion) {
        let mut group = c.benchmark_group("VoteEligibilityProof");

        let voter = ProofVoterProfile {
            did: "did:mycelix:voter".to_string(),
            assurance_level: 3,
            matl_score: 0.8,
            stake: 600.0,
            account_age_days: 100,
            participation_rate: 0.5,
            has_humanity_proof: true,
            fl_contributions: 25,
        };

        // Different proposal types have different active requirements
        for proposal_type in [
            ProofProposalType::Standard,
            ProofProposalType::Constitutional,
            ProofProposalType::Emergency,
        ].iter() {
            group.bench_with_input(
                BenchmarkId::new("generate", format!("{:?}", proposal_type)),
                proposal_type,
                |b, pt| {
                    b.iter(|| {
                        VoteEligibilityProof::generate(
                            black_box(&voter),
                            black_box(*pt),
                            config_96(),
                        ).unwrap()
                    })
                },
            );
        }

        let proof = VoteEligibilityProof::generate(
            &voter,
            ProofProposalType::Constitutional,
            config_128(),
        ).unwrap();

        group.bench_function("verify", |b| {
            b.iter(|| proof.verify().unwrap())
        });

        println!("VoteEligibilityProof size: {} bytes", proof.size());

        group.finish();
    }

    pub fn bench_security_levels(c: &mut Criterion) {
        let mut group = c.benchmark_group("SecurityLevels");
        group.sample_size(20);

        for level in [SecurityLevel::Standard96, SecurityLevel::Standard128].iter() {
            let config = ProofConfig {
                security_level: *level,
                parallel: false,
                max_proof_size: 0,
            };

            group.bench_with_input(
                BenchmarkId::new("range_generate", format!("{:?}", level)),
                &config,
                |b, cfg| {
                    b.iter(|| {
                        RangeProof::generate(black_box(500), black_box(0), black_box(1000), cfg.clone())
                            .unwrap()
                    })
                },
            );
        }

        group.finish();
    }

    pub fn bench_serialization(c: &mut Criterion) {
        use fl_aggregator::proofs::integration::ProofEnvelope;

        let mut group = c.benchmark_group("Serialization");

        // Create a proof and envelope
        let proof = RangeProof::generate(500, 0, 1000, config_128()).unwrap();
        let envelope = ProofEnvelope::from_range_proof(&proof).unwrap();
        let bytes = envelope.to_bytes().unwrap();

        group.throughput(Throughput::Bytes(bytes.len() as u64));

        group.bench_function("envelope_serialize", |b| {
            b.iter(|| envelope.to_bytes().unwrap())
        });

        group.bench_function("envelope_deserialize", |b| {
            b.iter(|| ProofEnvelope::from_bytes(black_box(&bytes)).unwrap())
        });

        group.bench_function("proof_serialize", |b| {
            b.iter(|| proof.to_bytes())
        });

        let proof_bytes = proof.to_bytes();
        group.bench_function("proof_deserialize", |b| {
            b.iter(|| RangeProof::from_bytes(black_box(&proof_bytes)).unwrap())
        });

        // Roundtrip
        group.bench_function("roundtrip", |b| {
            b.iter(|| {
                let bytes = envelope.to_bytes().unwrap();
                let restored = ProofEnvelope::from_bytes(&bytes).unwrap();
                RangeProof::from_bytes(&restored.proof_bytes).unwrap()
            })
        });

        println!("Envelope size: {} bytes", bytes.len());
        println!("Proof size: {} bytes", proof_bytes.len());

        group.finish();
    }

    #[cfg(feature = "proofs-compressed")]
    pub fn bench_compression(c: &mut Criterion) {
        use fl_aggregator::proofs::integration::ProofEnvelope;

        let mut group = c.benchmark_group("Compression");
        group.sample_size(50);

        // Create proofs of different sizes
        let proof = RangeProof::generate(500, 0, 1000, config_128()).unwrap();
        let envelope = ProofEnvelope::from_range_proof(&proof).unwrap();

        let original_size = envelope.to_bytes().unwrap().len();

        group.bench_function("compress", |b| {
            b.iter(|| envelope.compress().unwrap())
        });

        let compressed = envelope.compress().unwrap();
        let compressed_size = compressed.to_bytes().unwrap().len();

        group.bench_function("decompress", |b| {
            b.iter(|| compressed.decompress().unwrap())
        });

        // Full roundtrip with compression
        group.bench_function("roundtrip_compressed", |b| {
            b.iter(|| {
                let compressed = envelope.compress().unwrap();
                let bytes = compressed.to_bytes().unwrap();
                let restored = ProofEnvelope::from_bytes(&bytes).unwrap();
                restored.decompress().unwrap()
            })
        });

        println!("Original size: {} bytes", original_size);
        println!("Compressed size: {} bytes", compressed_size);
        println!("Compression ratio: {:.2}%", 100.0 * compressed_size as f64 / original_size as f64);

        group.finish();
    }

    pub fn bench_batch_verification(c: &mut Criterion) {
        use fl_aggregator::proofs::integration::{ProofEnvelope, verify_batch_sync};

        let mut group = c.benchmark_group("BatchVerification");
        group.sample_size(20);

        // Create batch of envelopes
        for batch_size in [5, 10, 20].iter() {
            let envelopes: Vec<_> = (0..*batch_size)
                .map(|i| {
                    // Keep values within [0, 100] range
                    let value = (i * 4) % 100;
                    let proof = RangeProof::generate(value, 0, 100, config_96()).unwrap();
                    ProofEnvelope::from_range_proof(&proof).unwrap()
                })
                .collect();

            group.throughput(Throughput::Elements(*batch_size as u64));
            group.bench_with_input(
                BenchmarkId::new("sync_verify", batch_size),
                &envelopes,
                |b, envs| {
                    b.iter(|| verify_batch_sync(black_box(envs)))
                },
            );
        }

        group.finish();
    }

    criterion_group!(
        proofs,
        bench_range_proof,
        bench_membership_proof,
        bench_gradient_proof,
        bench_identity_proof,
        bench_vote_proof,
        bench_security_levels,
        bench_serialization,
        bench_batch_verification,
    );

    #[cfg(feature = "proofs-compressed")]
    criterion_group!(
        proofs_compressed,
        bench_compression,
    );
}

#[cfg(feature = "proofs")]
criterion_main!(proofs_benchmarks::proofs);

#[cfg(not(feature = "proofs"))]
fn main() {
    println!("Proof benchmarks require the 'proofs' feature");
}
