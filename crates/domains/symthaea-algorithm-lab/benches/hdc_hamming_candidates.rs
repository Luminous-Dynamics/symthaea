use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use symthaea_algorithm_lab::{HammingCandidate, candidate_distance, verify_candidate};
use symthaea_core::hdc::binary_hv::BinaryHV;

fn hdc_hamming_candidates(c: &mut Criterion) {
    // Correctness is checked before any candidate enters the performance group.
    // The benchmark uses multiple deterministic distance distributions so a candidate
    // cannot win solely on one convenient random pair.
    let correctness_seeds: Vec<u64> = (0..128).collect();
    let random_left = BinaryHV::random(0x4844_432d_4c41_422d);
    let random_right = BinaryHV::random(0x4841_4d4d_494e_4721);

    let mut sparse_difference = BinaryHV::zero();
    sparse_difference.0[0] = 0x01;
    sparse_difference.0[BinaryHV::BYTES - 1] = 0x80;

    let mut alternating = BinaryHV::zero();
    let mut inverse_alternating = BinaryHV::zero();
    for (index, (left, right)) in alternating
        .0
        .iter_mut()
        .zip(inverse_alternating.0.iter_mut())
        .enumerate()
    {
        *left = if index % 2 == 0 { 0x55 } else { 0xAA };
        *right = !*left;
    }

    let input_profiles = [
        ("random-balanced", random_left, random_right),
        ("sparse-two-bits", BinaryHV::zero(), sparse_difference),
        ("alternating-complement", alternating, inverse_alternating),
        ("maximum-distance", BinaryHV::zero(), BinaryHV::ones()),
    ];

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

        for (profile, left, right) in input_profiles {
            group.bench_with_input(
                BenchmarkId::new(candidate.name(), profile),
                &(left, right),
                |b, (left, right)| {
                    b.iter(|| {
                        black_box(candidate_distance(
                            candidate,
                            black_box(left),
                            black_box(right),
                        ))
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, hdc_hamming_candidates);
criterion_main!(benches);
