use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use symthaea_algorithm_lab::{HammingCandidate, candidate_distance, verify_candidate};
use symthaea_core::hdc::binary_hv::BinaryHV;

fn hdc_hamming_candidates(c: &mut Criterion) {
    // Correctness is checked before any candidate enters the performance group. Benchmark timing
    // is measurement evidence only; it cannot rescue a semantically incorrect implementation.
    let correctness_seeds: Vec<u64> = (0..128).collect();
    let left = BinaryHV::random(0x4844_432d_4c41_422d);
    let right = BinaryHV::random(0x4841_4d4d_494e_4721);

    let mut group = c.benchmark_group("hdc_hamming_16384");
    for candidate in HammingCandidate::ALL {
        let evidence = verify_candidate(candidate, &correctness_seeds);
        assert!(
            evidence.passed(),
            "candidate {} failed exact oracle comparison: {:?}",
            candidate.name(),
            evidence.first_mismatch()
        );
        evidence
            .validate()
            .expect("correctness evidence must self-validate before benchmarking");

        group.bench_with_input(
            BenchmarkId::from_parameter(candidate.name()),
            &candidate,
            |b, &candidate| {
                b.iter(|| {
                    black_box(candidate_distance(
                        candidate,
                        black_box(&left),
                        black_box(&right),
                    ))
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, hdc_hamming_candidates);
criterion_main!(benches);
