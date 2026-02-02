//! Benchmarks for ZK Tax proof generation.
//!
//! Run with: `cargo bench`
//! Results are saved to `target/criterion/`

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use mycelix_zk_tax::{
    FilingStatus, Jurisdiction, TaxBracketProver,
    RangeProofBuilder, BatchProofBuilder, EffectiveTaxRateProof,
    CrossJurisdictionProofBuilder, DeductionProofBuilder, DeductionCategory,
    CompositeProofBuilder, ProofChain,
    // Caching
    ProofCache, CacheConfig, global_cache, clear_global_cache,
    // Compression
    proof::CompressedProof, proof::CompressedProofType,
};
use mycelix_zk_tax::subnational::{USState, SwissCanton, get_state_brackets, get_cantonal_brackets};

/// Benchmark basic tax bracket proof generation
fn bench_bracket_proof(c: &mut Criterion) {
    let prover = TaxBracketProver::dev_mode();

    c.bench_function("bracket_proof_single", |b| {
        b.iter(|| {
            prover.prove(
                black_box(85_000),
                black_box(Jurisdiction::US),
                black_box(FilingStatus::Single),
                black_box(2024),
            ).unwrap()
        })
    });
}

/// Benchmark bracket proofs across income levels
fn bench_bracket_by_income(c: &mut Criterion) {
    let prover = TaxBracketProver::dev_mode();
    let incomes = [25_000u64, 50_000, 100_000, 250_000, 500_000, 1_000_000];

    let mut group = c.benchmark_group("bracket_by_income");
    for income in incomes {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("${}", income)),
            &income,
            |b, &inc| {
                b.iter(|| {
                    prover.prove(
                        black_box(inc),
                        Jurisdiction::US,
                        FilingStatus::Single,
                        2024,
                    ).unwrap()
                })
            },
        );
    }
    group.finish();
}

/// Benchmark bracket proofs across jurisdictions
fn bench_bracket_by_jurisdiction(c: &mut Criterion) {
    let prover = TaxBracketProver::dev_mode();
    let jurisdictions = [
        Jurisdiction::US, Jurisdiction::UK, Jurisdiction::DE,
        Jurisdiction::FR, Jurisdiction::JP, Jurisdiction::AU,
    ];

    let mut group = c.benchmark_group("bracket_by_jurisdiction");
    for jurisdiction in jurisdictions {
        group.bench_with_input(
            BenchmarkId::from_parameter(jurisdiction.name()),
            &jurisdiction,
            |b, &j| {
                b.iter(|| {
                    prover.prove(
                        100_000,
                        black_box(j),
                        FilingStatus::Single,
                        2024,
                    ).unwrap()
                })
            },
        );
    }
    group.finish();
}

/// Benchmark income range proofs
fn bench_range_proof(c: &mut Criterion) {
    let mut group = c.benchmark_group("range_proof");

    group.bench_function("prove_above", |b| {
        b.iter(|| {
            RangeProofBuilder::new(black_box(95_000), 2024)
                .prove_above(black_box(90_000))
                .unwrap()
        })
    });

    group.bench_function("prove_below", |b| {
        b.iter(|| {
            RangeProofBuilder::new(black_box(45_000), 2024)
                .prove_below(black_box(50_000))
                .unwrap()
        })
    });

    group.bench_function("prove_between", |b| {
        b.iter(|| {
            RangeProofBuilder::new(black_box(75_000), 2024)
                .prove_between(black_box(50_000), black_box(100_000))
                .unwrap()
        })
    });

    group.finish();
}

/// Benchmark effective tax rate proofs
fn bench_effective_rate(c: &mut Criterion) {
    c.bench_function("effective_rate_proof", |b| {
        b.iter(|| {
            EffectiveTaxRateProof::prove_dev(
                black_box(150_000),
                black_box(Jurisdiction::US),
                black_box(FilingStatus::Single),
                black_box(2024),
            ).unwrap()
        })
    });
}

/// Benchmark multi-year batch proofs
fn bench_batch_proof(c: &mut Criterion) {
    let years_and_incomes = [
        (2022, 75_000u64),
        (2023, 80_000),
        (2024, 85_000),
    ];

    let mut group = c.benchmark_group("batch_proof");

    group.bench_function("3_year_history", |b| {
        b.iter(|| {
            let mut builder = BatchProofBuilder::new(
                black_box(Jurisdiction::US),
                black_box(FilingStatus::Single),
            );
            for (year, income) in &years_and_incomes {
                builder = builder.add_year(*year, *income);
            }
            builder.build_dev().unwrap()
        })
    });

    // 5-year batch
    let five_years = [
        (2020, 65_000u64), (2021, 70_000), (2022, 75_000),
        (2023, 80_000), (2024, 85_000),
    ];

    group.bench_function("5_year_history", |b| {
        b.iter(|| {
            let mut builder = BatchProofBuilder::new(
                Jurisdiction::US,
                FilingStatus::Single,
            );
            for (year, income) in &five_years {
                builder = builder.add_year(*year, *income);
            }
            builder.build_dev().unwrap()
        })
    });

    group.finish();
}

/// Benchmark cross-jurisdiction proofs
fn bench_cross_jurisdiction(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_jurisdiction");

    group.bench_function("2_countries", |b| {
        b.iter(|| {
            CrossJurisdictionProofBuilder::new(black_box(100_000), 2024)
                .add_jurisdiction(Jurisdiction::US, FilingStatus::Single)
                .add_jurisdiction(Jurisdiction::UK, FilingStatus::Single)
                .build_dev()
                .unwrap()
        })
    });

    group.bench_function("oecd_7_countries", |b| {
        b.iter(|| {
            CrossJurisdictionProofBuilder::new(black_box(100_000), 2024)
                .add_oecd_common(FilingStatus::Single)
                .build_dev()
                .unwrap()
        })
    });

    group.bench_function("g20_19_countries", |b| {
        b.iter(|| {
            CrossJurisdictionProofBuilder::new(black_box(100_000), 2024)
                .add_jurisdictions(&[
                    Jurisdiction::US, Jurisdiction::CA, Jurisdiction::MX,
                    Jurisdiction::BR, Jurisdiction::AR, Jurisdiction::UK,
                    Jurisdiction::DE, Jurisdiction::FR, Jurisdiction::IT,
                    Jurisdiction::RU, Jurisdiction::TR, Jurisdiction::JP,
                    Jurisdiction::CN, Jurisdiction::IN, Jurisdiction::KR,
                    Jurisdiction::ID, Jurisdiction::AU, Jurisdiction::SA,
                    Jurisdiction::ZA,
                ], FilingStatus::Single)
                .build_dev()
                .unwrap()
        })
    });

    group.finish();
}

/// Benchmark deduction proofs
fn bench_deduction_proof(c: &mut Criterion) {
    let mut group = c.benchmark_group("deduction_proof");

    group.bench_function("3_categories", |b| {
        b.iter(|| {
            DeductionProofBuilder::new(FilingStatus::Single, 2024)
                .add(DeductionCategory::Charitable, black_box(5_000))
                .add(DeductionCategory::MortgageInterest, black_box(12_000))
                .add(DeductionCategory::StateLocalTaxes, black_box(10_000))
                .build()
                .unwrap()
        })
    });

    group.bench_function("all_categories", |b| {
        b.iter(|| {
            DeductionProofBuilder::new(FilingStatus::Single, 2024)
                .add(DeductionCategory::Charitable, 5_000)
                .add(DeductionCategory::MortgageInterest, 12_000)
                .add(DeductionCategory::StateLocalTaxes, 10_000)
                .add(DeductionCategory::Medical, 8_000)
                .add(DeductionCategory::Retirement, 22_500)
                .build()
                .unwrap()
        })
    });

    group.finish();
}

/// Benchmark composite proofs (all-in-one)
fn bench_composite_proof(c: &mut Criterion) {
    let mut group = c.benchmark_group("composite_proof");

    group.bench_function("bracket_only", |b| {
        b.iter(|| {
            CompositeProofBuilder::new(
                black_box(85_000),
                Jurisdiction::US,
                FilingStatus::Single,
                2024,
            )
            .with_bracket()
            .build_dev()
            .unwrap()
        })
    });

    group.bench_function("full_composite", |b| {
        b.iter(|| {
            CompositeProofBuilder::new(85_000, Jurisdiction::US, FilingStatus::Single, 2024)
                .with_bracket()
                .with_effective_rate()
                .with_range(0, 200_000)
                .build_dev()
                .unwrap()
        })
    });

    group.finish();
}

/// Benchmark proof chains
fn bench_proof_chain(c: &mut Criterion) {
    let prover = TaxBracketProver::dev_mode();
    let years_incomes = [(2022u32, 75_000u64), (2023, 80_000), (2024, 85_000)];

    let mut group = c.benchmark_group("proof_chain");

    group.bench_function("3_link_chain", |b| {
        b.iter(|| {
            let mut chain = ProofChain::new("bench-user");
            for (year, income) in &years_incomes {
                let proof = prover.prove(
                    *income,
                    Jurisdiction::US,
                    FilingStatus::Single,
                    *year,
                ).unwrap();
                chain.add_bracket_proof(&proof);
            }
            chain
        })
    });

    group.bench_function("chain_verify", |b| {
        // Pre-build the chain
        let mut chain = ProofChain::new("bench-user");
        for (year, income) in &years_incomes {
            let proof = prover.prove(
                *income,
                Jurisdiction::US,
                FilingStatus::Single,
                *year,
            ).unwrap();
            chain.add_bracket_proof(&proof);
        }

        b.iter(|| {
            black_box(&chain).verify().unwrap()
        })
    });

    group.finish();
}

/// Benchmark US state tax bracket lookups
fn bench_state_brackets(c: &mut Criterion) {
    let states = [
        USState::CA, USState::NY, USState::TX,
        USState::FL, USState::WA,
    ];

    let mut group = c.benchmark_group("state_brackets");
    for state in states {
        group.bench_with_input(
            BenchmarkId::from_parameter(state.name()),
            &state,
            |b, &s| {
                b.iter(|| {
                    get_state_brackets(
                        black_box(s),
                        black_box(2024),
                        black_box(FilingStatus::Single),
                    ).unwrap()
                })
            },
        );
    }
    group.finish();
}

/// Benchmark Swiss cantonal tax bracket lookups
fn bench_canton_brackets(c: &mut Criterion) {
    let cantons = [
        SwissCanton::ZH, SwissCanton::ZG, SwissCanton::GE,
        SwissCanton::BS, SwissCanton::BE,
    ];

    let mut group = c.benchmark_group("canton_brackets");
    for canton in cantons {
        group.bench_with_input(
            BenchmarkId::from_parameter(canton.name()),
            &canton,
            |b, &c| {
                b.iter(|| {
                    get_cantonal_brackets(
                        black_box(c),
                        black_box(2024),
                        black_box(FilingStatus::Single),
                    ).unwrap()
                })
            },
        );
    }
    group.finish();
}

/// Benchmark proof verification
fn bench_verification(c: &mut Criterion) {
    let prover = TaxBracketProver::dev_mode();
    let proof = prover.prove(85_000, Jurisdiction::US, FilingStatus::Single, 2024).unwrap();

    let mut group = c.benchmark_group("verification");

    group.bench_function("bracket_verify", |b| {
        b.iter(|| {
            black_box(&proof).verify().unwrap()
        })
    });

    // Range proof verification
    let range_proof = RangeProofBuilder::new(95_000, 2024)
        .prove_above(90_000)
        .unwrap();

    group.bench_function("range_verify", |b| {
        b.iter(|| {
            black_box(&range_proof).verify().unwrap()
        })
    });

    // Effective rate verification
    let eff_proof = EffectiveTaxRateProof::prove_dev(
        150_000, Jurisdiction::US, FilingStatus::Single, 2024
    ).unwrap();

    group.bench_function("effective_rate_verify", |b| {
        b.iter(|| {
            black_box(&eff_proof).verify().unwrap()
        })
    });

    group.finish();
}

/// Benchmark proof caching
fn bench_caching(c: &mut Criterion) {
    let prover = TaxBracketProver::dev_mode();
    let proof = prover.prove(85_000, Jurisdiction::US, FilingStatus::Single, 2024).unwrap();

    let mut group = c.benchmark_group("caching");

    // Cache operations
    group.bench_function("cache_insert", |b| {
        let cache = ProofCache::new();
        b.iter(|| {
            cache.put(black_box(proof.clone()));
        })
    });

    group.bench_function("cache_lookup_hit", |b| {
        let cache = ProofCache::new();
        cache.put(proof.clone());
        let key = cache.stats().total_proofs; // Just to get a valid key reference
        b.iter(|| {
            cache.get_for_income(
                black_box(85_000),
                black_box(Jurisdiction::US),
                black_box(FilingStatus::Single),
                black_box(2024),
            )
        })
    });

    group.bench_function("cache_lookup_miss", |b| {
        let cache = ProofCache::new();
        b.iter(|| {
            cache.get_for_income(
                black_box(999_999),
                black_box(Jurisdiction::US),
                black_box(FilingStatus::Single),
                black_box(2024),
            )
        })
    });

    // Cache with configuration
    group.bench_function("cache_high_perf_insert", |b| {
        let config = CacheConfig::high_performance();
        let cache = ProofCache::with_config(config);
        b.iter(|| {
            cache.put(black_box(proof.clone()));
        })
    });

    group.finish();
}

/// Benchmark proof compression
fn bench_compression(c: &mut Criterion) {
    let prover = TaxBracketProver::dev_mode();
    let proof = prover.prove(85_000, Jurisdiction::US, FilingStatus::Single, 2024).unwrap();

    let mut group = c.benchmark_group("compression");
    group.throughput(Throughput::Elements(1));

    // Compress
    group.bench_function("compress_rle_base64", |b| {
        b.iter(|| {
            CompressedProof::compress(
                black_box(&proof),
                black_box(CompressedProofType::RleBase64),
            ).unwrap()
        })
    });

    group.bench_function("compress_zstd", |b| {
        b.iter(|| {
            CompressedProof::compress(
                black_box(&proof),
                black_box(CompressedProofType::Zstd),
            ).unwrap()
        })
    });

    // Decompress
    let compressed_rle = CompressedProof::compress(&proof, CompressedProofType::RleBase64).unwrap();
    let compressed_zstd = CompressedProof::compress(&proof, CompressedProofType::Zstd).unwrap();

    group.bench_function("decompress_rle_base64", |b| {
        b.iter(|| {
            black_box(&compressed_rle).decompress().unwrap()
        })
    });

    group.bench_function("decompress_zstd", |b| {
        b.iter(|| {
            black_box(&compressed_zstd).decompress().unwrap()
        })
    });

    // Round-trip
    group.bench_function("round_trip_rle", |b| {
        b.iter(|| {
            let c = CompressedProof::compress(&proof, CompressedProofType::RleBase64).unwrap();
            c.decompress().unwrap()
        })
    });

    group.finish();
}

/// Benchmark global cache operations
fn bench_global_cache(c: &mut Criterion) {
    let prover = TaxBracketProver::dev_mode();

    let mut group = c.benchmark_group("global_cache");

    group.bench_function("global_cache_access", |b| {
        b.iter(|| {
            let _cache = global_cache();
        })
    });

    group.bench_function("global_cache_stats", |b| {
        b.iter(|| {
            global_cache().stats()
        })
    });

    // Proof via global cache (simulates real usage)
    group.bench_function("prove_with_cache", |b| {
        clear_global_cache();
        b.iter(|| {
            let proof = prover.prove(
                black_box(85_000),
                Jurisdiction::US,
                FilingStatus::Single,
                2024,
            ).unwrap();
            global_cache().put(proof);
        })
    });

    group.finish();
}

/// Benchmark all 58 jurisdictions
fn bench_all_jurisdictions(c: &mut Criterion) {
    let prover = TaxBracketProver::dev_mode();
    let jurisdictions = [
        // Americas
        Jurisdiction::US, Jurisdiction::CA, Jurisdiction::MX, Jurisdiction::BR,
        Jurisdiction::AR, Jurisdiction::CL, Jurisdiction::CO, Jurisdiction::PE,
        // Europe
        Jurisdiction::UK, Jurisdiction::DE, Jurisdiction::FR, Jurisdiction::IT,
        Jurisdiction::ES, Jurisdiction::NL, Jurisdiction::BE, Jurisdiction::CH,
        // Asia-Pacific
        Jurisdiction::JP, Jurisdiction::CN, Jurisdiction::IN, Jurisdiction::AU,
        Jurisdiction::SG, Jurisdiction::HK, Jurisdiction::KR,
        // Middle East & Africa
        Jurisdiction::AE, Jurisdiction::SA, Jurisdiction::ZA,
    ];

    let mut group = c.benchmark_group("all_jurisdictions");
    group.sample_size(50); // Reduce sample size for faster benchmarking

    for jurisdiction in jurisdictions {
        group.bench_with_input(
            BenchmarkId::from_parameter(jurisdiction.code()),
            &jurisdiction,
            |b, &j| {
                b.iter(|| {
                    prover.prove(100_000, black_box(j), FilingStatus::Single, 2024).unwrap()
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_bracket_proof,
    bench_bracket_by_income,
    bench_bracket_by_jurisdiction,
    bench_range_proof,
    bench_effective_rate,
    bench_batch_proof,
    bench_cross_jurisdiction,
    bench_deduction_proof,
    bench_composite_proof,
    bench_proof_chain,
    bench_state_brackets,
    bench_canton_brackets,
    bench_verification,
    // New benchmarks
    bench_caching,
    bench_compression,
    bench_global_cache,
    bench_all_jurisdictions,
);

criterion_main!(benches);
