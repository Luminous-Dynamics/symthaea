// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root.

use symthaea_core::hdc::linear_code::{
    BinaryCodeword, RandomLinearCode, factorization_affine_fiber, factorization_algebra,
};
use symthaea_core::hdc::syndrome_decoder::{
    BoundedDistanceDecode, BoundedDistanceSyndromeDecoder, ParityCheckMatrix,
};

fn boundary_code() -> RandomLinearCode {
    RandomLinearCode::from_basis(vec![
        BinaryCodeword::from_words(8, vec![0b1111_0000]),
        BinaryCodeword::from_words(8, vec![0b0000_1111]),
    ])
    .expect("boundary fixture")
}

/// Published fixture specification for the canonical [8,2,4] code.
///
/// These six parity checks are intentionally hand-specified from the fixture's
/// defining invariant: bits 0..=3 are equal and bits 4..=7 are equal. This is
/// not derived from the production parity-check constructor and therefore acts
/// as a mathematical specification oracle for the finite fixture.
const CANONICAL_BOUNDARY_CHECKS: [u8; 6] = [
    0b0000_0011, // x0 + x1
    0b0000_0110, // x1 + x2
    0b0000_1100, // x2 + x3
    0b0011_0000, // x4 + x5
    0b0110_0000, // x5 + x6
    0b1100_0000, // x6 + x7
];

fn hamming_distance(left: &BinaryCodeword, right: &BinaryCodeword) -> usize {
    left.words()
        .iter()
        .zip(right.words())
        .map(|(a, b)| (a ^ b).count_ones() as usize)
        .sum()
}

fn error_from_mask(mask: usize, dimension: usize) -> BinaryCodeword {
    let mut error = BinaryCodeword::zero(dimension);
    for index in 0..dimension {
        if (mask >> index) & 1 == 1 {
            error.set_bit(index, true);
        }
    }
    error
}

fn nearest_codewords(
    observation: &BinaryCodeword,
    codewords: &[BinaryCodeword],
) -> (usize, Vec<usize>) {
    let distances = codewords
        .iter()
        .map(|candidate| hamming_distance(observation, candidate))
        .collect::<Vec<_>>();
    let minimum = *distances.iter().min().expect("non-empty codebook");
    let nearest = distances
        .iter()
        .enumerate()
        .filter_map(|(index, &distance)| (distance == minimum).then_some(index))
        .collect::<Vec<_>>();
    (minimum, nearest)
}

#[test]
fn parity_check_is_full_rank_and_annihilates_code_space() {
    let code = RandomLinearCode::generate(31, 5, 0x5EED);
    let parity_check = ParityCheckMatrix::from_code(&code).expect("parity-check matrix");

    let independent_generator_rank = independent_generator_rank(&code);
    assert_eq!(independent_generator_rank, 5);
    assert_eq!(independent_generator_rank, code.rank());

    assert_eq!(parity_check.dimension(), 31);
    assert_eq!(
        parity_check.syndrome_dimension(),
        31 - independent_generator_rank
    );
    assert_eq!(parity_check.rows().len(), 31 - independent_generator_rank);
    assert_eq!(parity_check.columns().len(), 31);
    let independent_check_rank = independent_binary_rank(
        &parity_check
            .rows()
            .iter()
            .map(|row| row.words()[0])
            .collect::<Vec<_>>(),
        code.dimension(),
    );
    assert_eq!(independent_check_rank, parity_check.rows().len());

    for codeword in code.enumerate() {
        let syndrome = parity_check.syndrome(&codeword).expect("same dimension");
        assert_eq!(syndrome.weight(), 0);
    }

    println!(
        "PARITY_CHECK_LEDGER=dimension={};code_rank={};check_rank={};syndrome_dimension={};fingerprint={}",
        code.dimension(),
        code.rank(),
        independent_binary_rank(
            &parity_check
                .rows()
                .iter()
                .map(|row| row.words()[0])
                .collect::<Vec<_>>(),
            code.dimension(),
        ),
        parity_check.syndrome_dimension(),
        parity_check
            .fingerprint()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>(),
    );
}

#[test]
fn parity_check_annihilation_and_row_independence_hold_by_exhaustive_small_fixture() {
    let code = boundary_code();
    let parity_check = ParityCheckMatrix::from_code(&code).expect("parity-check matrix");
    let rows = parity_check.rows();

    for codeword in code.enumerate() {
        for row in rows {
            let parity = row
                .words()
                .iter()
                .zip(codeword.words())
                .map(|(left, right)| (left & right).count_ones() as usize)
                .sum::<usize>()
                % 2;
            assert_eq!(parity, 0, "generator/codeword was not annihilated by H");
        }
    }

    assert!(rows.len() < usize::BITS as usize);
    for mask in 1usize..(1usize << rows.len()) {
        let mut combination = BinaryCodeword::zero(parity_check.dimension());
        for (index, row) in rows.iter().enumerate() {
            if (mask >> index) & 1 == 1 {
                combination.xor_assign(row);
            }
        }
        assert_ne!(
            combination.weight(),
            0,
            "non-empty parity-check row combination collapsed to zero: mask={mask:#x}"
        );
    }

    println!(
        "PARITY_CHECK_EXHAUSTIVE=rows={};nonzero_row_combinations={}",
        rows.len(),
        (1usize << rows.len()) - 1,
    );
}

#[test]
fn parity_check_derivation_is_deterministic_for_same_code() {
    let a = RandomLinearCode::generate(73, 8, 0xC0DE);
    let b = RandomLinearCode::generate(73, 8, 0xC0DE);
    let pa = ParityCheckMatrix::from_code(&a).expect("parity-check A");
    let pb = ParityCheckMatrix::from_code(&b).expect("parity-check B");

    assert_eq!(a.fingerprint(), b.fingerprint());
    assert_eq!(pa.rows(), pb.rows());
    assert_eq!(pa.columns(), pb.columns());
    assert_eq!(pa.fingerprint(), pb.fingerprint());
}

#[test]
fn small_fixture_kernel_equals_code_and_syndrome_cosets_are_exact() {
    let code = boundary_code();
    let parity_check = ParityCheckMatrix::from_code(&code).expect("parity-check matrix");
    let codewords = code.enumerate();
    let dimension = parity_check.dimension();
    let syndrome_dimension = parity_check.syndrome_dimension();
    let syndrome_space_size = 1usize << syndrome_dimension;

    assert_eq!(dimension, 8);
    assert_eq!(code.rank(), 2);
    assert_eq!(syndrome_dimension, 6);
    assert_eq!(syndrome_space_size, 64);
    assert_eq!(codewords.len(), 4);

    // Exhaustively partition all 2^8 ambient observations by syndrome.
    let mut buckets = vec![Vec::<usize>::new(); syndrome_space_size];
    for mask in 0..(1usize << dimension) {
        let word = error_from_mask(mask, dimension);
        let syndrome = parity_check.syndrome(&word).expect("same dimension");
        assert_eq!(syndrome.words().len(), 1);
        let syndrome_index = syndrome.words()[0] as usize;
        assert!(syndrome_index < buckets.len());
        buckets[syndrome_index].push(mask);
    }

    // Every syndrome occurs and every fiber has exactly |C| = 2^k elements.
    assert_eq!(
        buckets.iter().filter(|bucket| !bucket.is_empty()).count(),
        64
    );
    assert!(buckets.iter().all(|bucket| bucket.len() == codewords.len()));

    // The zero-syndrome kernel is exactly the code, element-for-element.
    let kernel = &buckets[0];
    assert_eq!(kernel.len(), codewords.len());
    for &mask in kernel {
        let word = error_from_mask(mask, dimension);
        assert!(
            code.contains(&word),
            "zero-syndrome word is outside the code: mask={mask:#x}"
        );
    }
    for codeword in &codewords {
        let mask = codeword.words()[0] as usize;
        assert!(
            kernel.contains(&mask),
            "codeword is missing from zero-syndrome kernel: mask={mask:#x}"
        );
    }

    // Same syndrome iff the difference is in C. Each bucket is checked in
    // both directions against translation by every codeword.
    for bucket in &buckets {
        let representative = error_from_mask(bucket[0], dimension);
        let representative_syndrome = parity_check
            .syndrome(&representative)
            .expect("same dimension");

        for &mask in bucket {
            let member = error_from_mask(mask, dimension);
            let mut difference = member.clone();
            difference.xor_assign(&representative);
            assert!(
                code.contains(&difference),
                "same-syndrome words differed by a non-codeword: representative={:#x}, member={mask:#x}",
                bucket[0]
            );
        }

        for codeword in &codewords {
            let mut translated = representative.clone();
            translated.xor_assign(codeword);
            assert_eq!(
                parity_check.syndrome(&translated),
                Some(representative_syndrome.clone())
            );
        }
    }

    println!(
        "SYNDROME_COSET_LEDGER=dimension={};code_rank={};syndrome_dimension={};kernel_size={};code_size={};syndrome_count={};min_coset_size={};max_coset_size={};kernel_equals_code=true;coset_partition_exact=true",
        dimension,
        code.rank(),
        syndrome_dimension,
        kernel.len(),
        codewords.len(),
        buckets.iter().filter(|bucket| !bucket.is_empty()).count(),
        buckets.iter().map(Vec::len).min().unwrap_or(0),
        buckets.iter().map(Vec::len).max().unwrap_or(0),
    );
}

#[test]
fn small_fixture_coset_leader_profile_matches_decoder_without_codeword_oracle() {
    let code = boundary_code();
    let decoder = BoundedDistanceSyndromeDecoder::from_code(&code).expect("decoder");
    let checks = independent_parity_check_rows(&code);
    let profile = independent_coset_leader_profile(&checks, code.dimension());

    assert_eq!(profile.len(), 64);
    assert!(profile.iter().all(|(distance, matches)| *distance <= 4 && *matches > 0));

    let mut minimum_distance_histogram = [0usize; 5];
    let mut total_minimum_matches = 0usize;
    let mut maximum_minimum_multiplicity = 0usize;

    for mask in 0..(1usize << code.dimension()) {
        let observation = error_from_mask(mask, code.dimension());
        let syndrome = independent_syndrome(mask as u64, &checks) as usize;
        let (expected_distance, expected_matches) = profile[syndrome];
        assert!(expected_distance <= 4);
        assert!(expected_matches > 0);

        let (outcome, work) = decoder.decode_with_work(&observation, 4);
        assert_eq!(work.matching_error_patterns, expected_matches);

        match outcome {
            BoundedDistanceDecode::Unique { distance, .. } => {
                assert_eq!(expected_matches, 1);
                assert_eq!(distance, expected_distance);
            }
            BoundedDistanceDecode::Ambiguous {
                distance,
                matching_error_patterns,
            } => {
                assert!(expected_matches > 1);
                assert_eq!(distance, expected_distance);
                assert_eq!(matching_error_patterns, expected_matches);
            }
            other => panic!(
                "covering-radius bound must produce a minimum syndrome representative: {other:?}"
            ),
        }

        minimum_distance_histogram[expected_distance] += 1;
        total_minimum_matches += expected_matches;
        maximum_minimum_multiplicity =
            maximum_minimum_multiplicity.max(expected_matches);
    }

    assert_eq!(minimum_distance_histogram[0], 4);
    assert_eq!(
        minimum_distance_histogram.iter().sum::<usize>(),
        1usize << code.dimension()
    );
    assert_eq!(total_minimum_matches, 484);
    assert_eq!(maximum_minimum_multiplicity, 4);

    let histogram = minimum_distance_histogram
        .iter()
        .enumerate()
        .map(|(weight, count)| format!("{weight}:{count}"))
        .collect::<Vec<_>>()
        .join(",");

    println!(
        "SYNDROME_COSET_LEADER_ORACLE=dimension={};syndromes={};observations={};minimum_weight_histogram={histogram};total_minimum_matches={total_minimum_matches};maximum_minimum_multiplicity={maximum_minimum_multiplicity};independent_coset_oracle=true;codeword_enumeration=false",
        code.dimension(),
        profile.len(),
        1usize << code.dimension(),
    );
}

#[test]
fn small_fixture_fixed_parity_check_spec_agrees_with_all_syndrome_oracles() {
    let code = boundary_code();
    let parity_check = ParityCheckMatrix::from_code(&code).expect("parity-check matrix");
    let independent_checks = independent_parity_check_rows(&code);

    assert_eq!(CANONICAL_BOUNDARY_CHECKS.len(), 6);
    assert_eq!(parity_check.syndrome_dimension(), CANONICAL_BOUNDARY_CHECKS.len());
    assert_eq!(independent_checks.len(), CANONICAL_BOUNDARY_CHECKS.len());

    let mut buckets = [0usize; 64];
    let mut zero_syndrome_members = 0usize;

    for mask in 0u16..256 {
        let fixed = canonical_boundary_syndrome(mask as u8);
        let production = parity_check
            .syndrome(&error_from_mask(mask as usize, code.dimension()))
            .expect("same dimension")
            .words()[0] as u8;
        let independent = independent_syndrome(mask as u64, &independent_checks) as u8;

        assert_eq!(
            production, fixed,
            "production syndrome diverged from fixed fixture specification: mask={mask:#x}"
        );
        assert_eq!(
            independent, fixed,
            "independent nullspace oracle diverged from fixed fixture specification: mask={mask:#x}"
        );

        buckets[fixed as usize] += 1;
        if fixed == 0 {
            zero_syndrome_members += 1;
            assert!(
                code.contains(&error_from_mask(mask as usize, code.dimension())),
                "fixed zero-syndrome specification admitted a non-codeword: mask={mask:#x}"
            );
        }
    }

    assert_eq!(
        buckets.iter().filter(|&&count| count != 0).count(),
        64,
        "fixed fixture specification did not expose the full syndrome space"
    );
    assert!(buckets.iter().all(|&count| count == 4));
    assert_eq!(zero_syndrome_members, code.enumerate().len());
    assert_eq!(zero_syndrome_members, 4);

    println!(
        "FIXED_SYNDROME_SPEC=dimension={};checks={};syndromes=64;fiber_size=4;kernel_size=4;production_matches=true;independent_matches=true;spec_oracle=true",
        code.dimension(),
        CANONICAL_BOUNDARY_CHECKS.len()
    );
}

fn canonical_boundary_syndrome(mask: u8) -> u8 {
    CANONICAL_BOUNDARY_CHECKS
        .iter()
        .enumerate()
        .fold(0u8, |syndrome, (index, &check)| {
            syndrome | ((((mask & check).count_ones() & 1) as u8) << index)
        })
}

#[test]
fn small_fixture_quotient_metric_matches_ambient_coset_geometry() {
    let mut buckets = [Vec::<u8>::new(); 64];
    let mut leaders = [usize::MAX; 64];

    for mask in 0u16..256 {
        let byte = mask as u8;
        let syndrome = canonical_boundary_syndrome(byte) as usize;
        buckets[syndrome].push(byte);
        leaders[syndrome] = leaders[syndrome].min(byte.count_ones() as usize);
    }

    assert!(leaders.iter().all(|&distance| distance <= 4));
    assert_eq!(*leaders.iter().max().unwrap(), 4);
    assert!(buckets.iter().all(|bucket| bucket.len() == 4));

    let mut pair_checks = 0usize;
    for left in 0..64usize {
        for right in 0..64usize {
            let mut ambient_distance = usize::MAX;
            for &left_word in &buckets[left] {
                for &right_word in &buckets[right] {
                    ambient_distance = ambient_distance.min(
                        (left_word ^ right_word).count_ones() as usize
                    );
                    pair_checks += 1;
                }
            }

            assert_eq!(
                ambient_distance,
                leaders[left ^ right],
                "quotient distance disagreed with ambient coset distance: left={left} right={right}"
            );
        }
    }

    let mut triangle_checks = 0usize;
    for left in 0..64usize {
        for middle in 0..64usize {
            for right in 0..64usize {
                assert!(
                    leaders[left ^ right] <= leaders[left ^ middle] + leaders[middle ^ right],
                    "quotient metric violated triangle inequality: left={left} middle={middle} right={right}"
                );
                triangle_checks += 1;
            }
        }
    }

    println!(
        "SYNDROME_QUOTIENT_METRIC=syndromes=64;fiber_size=4;max_distance=4;ambient_pair_checks={pair_checks};triangle_checks={triangle_checks};translation_invariant=true;ambient_metric_exact=true"
    );
}

#[test]
fn small_fixture_production_syndrome_matches_independent_oracle_and_is_linear() {
    let code = boundary_code();
    let parity_check = ParityCheckMatrix::from_code(&code).expect("parity-check matrix");
    let checks = independent_parity_check_rows(&code);

    assert_eq!(checks.len(), parity_check.syndrome_dimension());
    assert_eq!(checks.len(), 6);

    let mut syndromes = Vec::with_capacity(1usize << code.dimension());
    for mask in 0..(1usize << code.dimension()) {
        let word = error_from_mask(mask, code.dimension());
        let production = parity_check.syndrome(&word).expect("same dimension");
        let independent = independent_syndrome(mask as u64, &checks);
        assert_eq!(
            production.words()[0],
            independent,
            "production and independent syndrome oracles diverged: mask={mask:#x}"
        );
        syndromes.push(independent);
    }

    for left in 0..(1usize << code.dimension()) {
        for right in 0..(1usize << code.dimension()) {
            let xor = (left ^ right) as u64;
            let expected = syndromes[left] ^ syndromes[right];
            assert_eq!(
                independent_syndrome(xor, &checks),
                expected,
                "independent syndrome map was not linear: left={left:#x} right={right:#x}"
            );

            let production_left = parity_check
                .syndrome(&error_from_mask(left, code.dimension()))
                .expect("same dimension");
            let production_right = parity_check
                .syndrome(&error_from_mask(right, code.dimension()))
                .expect("same dimension");
            let production_xor = parity_check
                .syndrome(&error_from_mask(left ^ right, code.dimension()))
                .expect("same dimension");
            let expected_production = production_left.bound(&production_right);
            assert_eq!(
                production_xor, expected_production,
                "production syndrome map was not linear: left={left:#x} right={right:#x}"
            );
        }
    }

    println!(
        "SYNDROME_ORACLE_LINEAREXHAUSTIVE=dimension={};observations={};pairs={};independent_agreement=true;production_linearity=true;independent_linearity=true",
        code.dimension(),
        1usize << code.dimension(),
        (1usize << code.dimension()) * (1usize << code.dimension()),
    );
}

#[test]
fn bounded_distance_decoder_matches_exhaustive_oracle_below_half_distance() {
    let code = boundary_code();
    let codewords = code.enumerate();
    let decoder = BoundedDistanceSyndromeDecoder::from_code(&code).expect("decoder");
    let mut observations = 0usize;

    for clean in &codewords {
        for mask in 0..(1usize << 8) {
            let error = error_from_mask(mask, 8);
            if error.weight() > 1 {
                continue;
            }

            let observation = clean.bound(&error);
            let (nearest_distance, nearest) = nearest_codewords(&observation, &codewords);
            assert_eq!(nearest_distance, error.weight());
            assert_eq!(nearest.len(), 1);
            assert_eq!(codewords[nearest[0]], *clean);

            match decoder.decode(&observation, 1) {
                BoundedDistanceDecode::Unique {
                    codeword,
                    error: decoded_error,
                    distance,
                } => {
                    assert_eq!(codeword, *clean);
                    assert_eq!(decoded_error, error);
                    assert_eq!(distance, error.weight());
                }
                other => panic!("expected unique bounded-distance decode, got {other:?}"),
            }
            observations += 1;
        }
    }

    assert_eq!(observations, 36);
}

#[test]
fn boundary_decoder_matches_exhaustive_nearest_codeword_geometry() {
    let code = boundary_code();
    let codewords = code.enumerate();
    let decoder = BoundedDistanceSyndromeDecoder::from_code(&code).expect("decoder");

    let mut observations = 0usize;
    let mut unique = 0usize;
    let mut ambiguous = 0usize;
    let mut ambiguous_matches = 0usize;

    for clean in &codewords {
        for mask in 0..(1usize << 8) {
            let error = error_from_mask(mask, 8);
            if error.weight() != 2 {
                continue;
            }

            let observation = clean.bound(&error);
            let (nearest_distance, nearest) = nearest_codewords(&observation, &codewords);
            assert_eq!(nearest_distance, 2);
            assert!(!nearest.is_empty());
            assert!(nearest.len() <= 2);

            match decoder.decode(&observation, 2) {
                BoundedDistanceDecode::Unique {
                    codeword,
                    error: decoded_error,
                    distance,
                } => {
                    assert_eq!(nearest.len(), 1);
                    assert_eq!(codeword, *clean);
                    assert_eq!(decoded_error, error);
                    assert_eq!(distance, 2);
                    unique += 1;
                }
                BoundedDistanceDecode::Ambiguous {
                    distance,
                    matching_error_patterns,
                } => {
                    assert_eq!(nearest.len(), 2);
                    assert_eq!(distance, 2);
                    assert_eq!(matching_error_patterns, nearest.len());
                    ambiguous += 1;
                    ambiguous_matches += matching_error_patterns;
                }
                other => panic!("expected unique/ambiguous boundary decode, got {other:?}"),
            }

            observations += 1;
        }
    }

    assert_eq!(observations, 112);
    assert_eq!(unique, 64);
    assert_eq!(ambiguous, 48);
    assert_eq!(ambiguous_matches, 96);
}

#[test]
fn boundary_decoder_preserves_the_same_geometry_around_every_codeword() {
    let code = boundary_code();
    let codewords = code.enumerate();
    let decoder = BoundedDistanceSyndromeDecoder::from_code(&code).expect("decoder");

    for (clean_index, clean) in codewords.iter().enumerate() {
        let mut unique = 0usize;
        let mut ambiguous = 0usize;

        for mask in 0..(1usize << 8) {
            let error = error_from_mask(mask, 8);
            if error.weight() != 2 {
                continue;
            }

            let observation = clean.bound(&error);
            let (nearest_distance, nearest) = nearest_codewords(&observation, &codewords);
            assert_eq!(nearest_distance, 2);

            match decoder.decode(&observation, 2) {
                BoundedDistanceDecode::Unique { .. } => {
                    assert_eq!(nearest.len(), 1);
                    unique += 1;
                }
                BoundedDistanceDecode::Ambiguous {
                    distance,
                    matching_error_patterns,
                } => {
                    assert_eq!(nearest.len(), 2);
                    assert_eq!(distance, 2);
                    assert_eq!(matching_error_patterns, 2);
                    ambiguous += 1;
                }
                other => panic!("clean index {clean_index}: unexpected result {other:?}"),
            }
        }

        assert_eq!(unique, 16, "clean codeword index {clean_index}");
        assert_eq!(ambiguous, 12, "clean codeword index {clean_index}");
    }
}

#[test]
fn minimum_weight_syndrome_multiplicity_matches_nearest_codeword_multiplicity_exhaustively() {
    let code = boundary_code();
    let codewords = code.enumerate();
    let decoder = BoundedDistanceSyndromeDecoder::from_code(&code).expect("decoder");

    let mut covering_radius = 0usize;
    for mask in 0..(1usize << 8) {
        let observation = error_from_mask(mask, 8);
        let (nearest_distance, _) = nearest_codewords(&observation, &codewords);
        covering_radius = covering_radius.max(nearest_distance);
    }
    assert_eq!(covering_radius, 4);

    let mut observations = 0usize;
    let mut unique = 0usize;
    let mut ambiguous = 0usize;
    let mut total_nearest_codewords = 0usize;
    let mut maximum_nearest_multiplicity = 0usize;

    // The [8,2,4] fixture has covering radius 4, so bound=4 reaches the
    // minimum possible distance for every ambient observation.
    for mask in 0..(1usize << 8) {
        let observation = error_from_mask(mask, 8);
        let (nearest_distance, nearest) = nearest_codewords(&observation, &codewords);
        let (result, work) = decoder.decode_with_work(&observation, 4);
        let listed = decoder.decode_with_minimum_list(&observation, 4, 8);

        assert!(listed.list_complete);
        assert_eq!(listed.minimum_errors.len(), nearest.len());
        assert_eq!(listed.nearest_codewords.len(), nearest.len());
        for codeword in &listed.nearest_codewords {
            assert!(
                nearest.iter().any(|&index| codewords[index] == *codeword),
                "listed codeword was not nearest for observation={mask:#x}"
            );
        }
        assert_eq!(
            listed.outcome, result,
            "list-valued and scalar decoder outcomes diverged for observation={mask:#x}"
        );
        assert_eq!(
            work.matching_error_patterns,
            nearest.len(),
            "minimum syndrome multiplicity must equal nearest-codeword multiplicity for observation={mask:#x}"
        );
        maximum_nearest_multiplicity = maximum_nearest_multiplicity.max(nearest.len());

        match result {
            BoundedDistanceDecode::Unique {
                codeword, distance, ..
            } => {
                assert_eq!(nearest.len(), 1);
                assert_eq!(distance, nearest_distance);
                assert_eq!(codeword, codewords[nearest[0]]);
                unique += 1;
            }
            BoundedDistanceDecode::Ambiguous {
                distance,
                matching_error_patterns,
            } => {
                assert!(nearest.len() > 1);
                assert_eq!(distance, nearest_distance);
                assert_eq!(matching_error_patterns, nearest.len());
                ambiguous += 1;
            }
            other => {
                panic!("covering-radius bound must decode every ambient observation, got {other:?}")
            }
        }

        total_nearest_codewords += nearest.len();
        observations += 1;
    }

    assert_eq!(observations, 256);
    assert_eq!(unique + ambiguous, 256);
    assert_eq!(ambiguous, 156);
    assert_eq!(unique, 100);
    assert_eq!(total_nearest_codewords, 484);

    println!(
        "MIN_SYNDROME_MULTIPLICITY_LEDGER=observations={observations};bound=4;covering_radius={covering_radius};unique={unique};ambiguous={ambiguous};total_nearest_codewords={total_nearest_codewords};maximum_nearest_multiplicity={maximum_nearest_multiplicity};list_cap=8;list_complete=true;multiplicity_identity=true"
    );
}

#[test]
fn factorization_ambiguity_is_distinct_from_syndrome_ambiguity() {
    let code = boundary_code();
    let g1 = code.basis()[0].clone();
    let g2 = code.basis()[1].clone();
    let g12 = g1.bound(&g2);

    // Four factor presentations span the same [8,2,4] code but add two
    // independent presentation-level kernel dimensions:
    // [g1], [g2], [g1+g2], [g1+g2].
    let f1 = RandomLinearCode::from_basis(vec![g1]).expect("factor 1");
    let f2 = RandomLinearCode::from_basis(vec![g2]).expect("factor 2");
    let f3 = RandomLinearCode::from_basis(vec![g12.clone()]).expect("factor 3");
    let f4 = RandomLinearCode::from_basis(vec![g12]).expect("factor 4");
    let factors = [&f1, &f2, &f3, &f4];

    let algebra = factorization_algebra(&factors).expect("factorization algebra");
    assert_eq!(algebra.factor_dimension_sum, 4);
    assert_eq!(algebra.union_generator_rank, code.rank());
    assert_eq!(algebra.kernel_dimension, 2);
    assert_eq!(algebra.factorization_count_per_target.exponent(), 2);
    assert!(!algebra.unique_factorization);

    let target = BinaryCodeword::zero(code.dimension());
    let fiber = factorization_affine_fiber(&target, &factors).expect("target fiber");
    assert!(fiber.verifies_against(&target, &factors));
    let representations = fiber
        .iter_bounded(4)
        .expect("bounded factor fiber")
        .collect::<Vec<_>>();
    assert_eq!(representations.len(), 4);

    let combined_basis = factors
        .iter()
        .flat_map(|factor| factor.basis())
        .collect::<Vec<_>>();
    for coefficients in &representations {
        let mut reconstructed = BinaryCodeword::zero(code.dimension());
        for (&coefficient, generator) in coefficients.iter().zip(&combined_basis) {
            if coefficient {
                reconstructed.xor_assign(generator);
            }
        }
        assert_eq!(reconstructed, target);
    }

    // Ambient syndrome/coset ambiguity is a separate geometry. This observation
    // has exactly two nearest codewords in the same [8,2,4] code.
    let observation = error_from_mask(0b0000_0011, code.dimension());
    let codewords = code.enumerate();
    let (nearest_distance, nearest) = nearest_codewords(&observation, &codewords);
    assert_eq!(nearest_distance, 2);
    assert_eq!(nearest.len(), 2);

    let decoder = BoundedDistanceSyndromeDecoder::from_code(&code).expect("decoder");
    let (result, work) = decoder.decode_with_work(&observation, 4);
    assert_eq!(work.matching_error_patterns, nearest.len());
    assert_eq!(
        result,
        BoundedDistanceDecode::Ambiguous {
            distance: nearest_distance,
            matching_error_patterns: nearest.len(),
        }
    );

    assert_ne!(representations.len(), nearest.len());
    println!(
        "FACTOR_SYNDROME_SEPARATION=factorization_count={};kernel_dimension={};nearest_codeword_multiplicity={};syndrome_min_multiplicity={};counts_distinct=true",
        representations.len(),
        algebra.kernel_dimension,
        nearest.len(),
        work.matching_error_patterns,
    );
}

#[test]
fn minimum_syndrome_list_truncation_is_explicit_and_fail_closed() {
    let code = boundary_code();
    let decoder = BoundedDistanceSyndromeDecoder::from_code(&code).expect("decoder");
    let observation = error_from_mask(0b0000_0011, code.dimension());

    let complete = decoder.decode_with_minimum_list(&observation, 4, 8);
    assert!(complete.list_complete);
    assert_eq!(complete.minimum_errors.len(), 2);
    assert_eq!(complete.nearest_codewords.len(), 2);

    let truncated = decoder.decode_with_minimum_list(&observation, 4, 1);
    assert!(!truncated.list_complete);
    assert_eq!(truncated.minimum_errors.len(), 1);
    assert_eq!(truncated.nearest_codewords.len(), 1);
    assert_eq!(truncated.outcome, complete.outcome);

    let empty = decoder.decode_with_minimum_list(&observation, 4, 0);
    assert!(!empty.list_complete);
    assert!(empty.minimum_errors.is_empty());
    assert!(empty.nearest_codewords.is_empty());
    assert_eq!(empty.outcome, complete.outcome);

    println!(
        "MIN_SYNDROME_LIST_CAPTURE=full_count=2;cap_8_complete=true;cap_1_complete=false;cap_0_complete=false;truncation_explicit=true"
    );
}

#[test]
fn decoder_refuses_observations_outside_the_declared_bound() {
    let code = boundary_code();
    let decoder = BoundedDistanceSyndromeDecoder::from_code(&code).expect("decoder");
    let observation = BinaryCodeword::from_words(8, vec![0x33]);

    let oracle_distances = code
        .enumerate()
        .iter()
        .map(|candidate| hamming_distance(&observation, candidate))
        .collect::<Vec<_>>();
    assert_eq!(oracle_distances, vec![4, 4, 4, 4]);

    assert_eq!(
        decoder.decode(&observation, 2),
        BoundedDistanceDecode::NoMatchWithinBound {
            max_error_weight: 2,
        }
    );
}

#[test]
fn decoder_work_ledger_matches_exact_weight_search_space() {
    let code = boundary_code();
    let decoder = BoundedDistanceSyndromeDecoder::from_code(&code).expect("decoder");
    let clean = code.encode(&[true, false]);
    let error = BinaryCodeword::from_words(8, vec![0b0000_0011]);
    let observation = clean.bound(&error);

    let (result, work) = decoder.decode_with_work(&observation, 2);
    assert!(matches!(
        result,
        BoundedDistanceDecode::Ambiguous {
            distance: 2,
            matching_error_patterns: 2
        }
    ));
    assert_eq!(work.weights_examined, 3);
    assert_eq!(work.error_patterns_examined, 37);
    assert_eq!(work.matching_error_patterns, 2);

    println!(
        "SYNDROME_DECODER_LEDGER=dimension={};code_rank={};max_error_weight=2;weights_examined={};error_patterns_examined={};syndrome_column_xors={};matching_error_patterns={}",
        code.dimension(),
        code.rank(),
        work.weights_examined,
        work.error_patterns_examined,
        work.syndrome_column_xors,
        work.matching_error_patterns,
    );
}

#[test]
fn invalid_observation_dimension_is_rejected_without_search() {
    let code = boundary_code();
    let decoder = BoundedDistanceSyndromeDecoder::from_code(&code).expect("decoder");
    let observation = BinaryCodeword::zero(7);

    let (result, work) = decoder.decode_with_work(&observation, 2);
    assert_eq!(
        result,
        BoundedDistanceDecode::InvalidObservationDimension {
            observation_dimension: 7,
            dimension: 8,
        }
    );
    assert_eq!(work, Default::default());
}

#[test]
fn invalid_decoder_bound_is_rejected_without_search() {
    let code = boundary_code();
    let decoder = BoundedDistanceSyndromeDecoder::from_code(&code).expect("decoder");
    let observation = BinaryCodeword::zero(8);

    let (result, work) = decoder.decode_with_work(&observation, 9);
    assert_eq!(
        result,
        BoundedDistanceDecode::InvalidBound {
            max_error_weight: 9,
            dimension: 8,
        }
    );
    assert_eq!(work, Default::default());
}

fn error_words_up_to_weight(dimension: usize, max_weight: usize) -> Vec<BinaryCodeword> {
    fn visit(
        dimension: usize,
        remaining: usize,
        start: usize,
        word: &mut BinaryCodeword,
        output: &mut Vec<BinaryCodeword>,
    ) {
        if remaining == 0 {
            output.push(word.clone());
            return;
        }
        let last_start = dimension - remaining;
        for index in start..=last_start {
            word.set_bit(index, true);
            visit(dimension, remaining - 1, index + 1, word, output);
            word.set_bit(index, false);
        }
    }

    let mut output = Vec::new();
    let max_weight = max_weight.min(dimension);
    for weight in 0..=max_weight {
        let mut word = BinaryCodeword::zero(dimension);
        visit(dimension, weight, 0, &mut word, &mut output);
    }
    output
}

// Independent test oracle: derive H as a nullspace basis using u64 row operations.
// This intentionally does not call ParityCheckMatrix::from_code.
fn independently_enumerated_codewords(code: &RandomLinearCode) -> Vec<BinaryCodeword> {
    let basis = code.basis();
    let combinations = 1usize << basis.len();
    let mut words = Vec::with_capacity(combinations);

    for mask in 0..combinations {
        let mut word = BinaryCodeword::zero(code.dimension());
        for (index, basis_word) in basis.iter().enumerate() {
            if ((mask >> index) & 1) == 1 {
                word.xor_assign(basis_word);
            }
        }
        words.push(word);
    }

    words
}

fn independent_binary_rank(rows: &[u64], dimension: usize) -> usize {
    assert!(dimension <= 64);

    let mut reduced = rows.to_vec();
    let mut pivot_row = 0usize;

    for column in 0..dimension {
        let Some(found) =
            (pivot_row..reduced.len()).find(|&row| ((reduced[row] >> column) & 1) == 1)
        else {
            continue;
        };

        reduced.swap(pivot_row, found);
        for row in 0..reduced.len() {
            if row != pivot_row && ((reduced[row] >> column) & 1) == 1 {
                reduced[row] ^= reduced[pivot_row];
            }
        }
        pivot_row += 1;
        if pivot_row == reduced.len() {
            break;
        }
    }

    pivot_row
}

fn independent_generator_rank(code: &RandomLinearCode) -> usize {
    let rows = code
        .basis()
        .iter()
        .map(|word| word.words()[0])
        .collect::<Vec<_>>();
    independent_binary_rank(&rows, code.dimension())
}

fn independent_parity_check_rows(code: &RandomLinearCode) -> Vec<u64> {
    let dimension = code.dimension();
    assert!(dimension <= 64);

    let independent_rank = independent_generator_rank(code);
    assert_eq!(independent_rank, code.rank());

    let mut reduced = code
        .basis()
        .iter()
        .map(|word| word.words()[0])
        .collect::<Vec<_>>();
    let mut pivots = Vec::with_capacity(independent_rank);
    let mut pivot_row = 0usize;

    for column in 0..dimension {
        let found = (pivot_row..reduced.len()).find(|&row| ((reduced[row] >> column) & 1) == 1);
        let Some(found) = found else {
            continue;
        };

        reduced.swap(pivot_row, found);
        for row in 0..reduced.len() {
            if row != pivot_row && ((reduced[row] >> column) & 1) == 1 {
                reduced[row] ^= reduced[pivot_row];
            }
        }
        pivots.push(column);
        pivot_row += 1;
        if pivot_row == reduced.len() {
            break;
        }
    }

    assert_eq!(pivots.len(), independent_rank);

    let mut is_pivot = vec![false; dimension];
    for &pivot in &pivots {
        is_pivot[pivot] = true;
    }

    let mut checks = Vec::with_capacity(dimension - independent_rank);
    for (free_column, &pivot) in is_pivot.iter().enumerate() {
        if pivot {
            continue;
        }

        let mut check = 1u64 << free_column;
        for (row, &pivot_column) in pivots.iter().enumerate() {
            if ((reduced[row] >> free_column) & 1) == 1 {
                check |= 1u64 << pivot_column;
            }
        }
        checks.push(check);
    }
    assert_eq!(checks.len(), dimension - independent_rank);
    checks
}

fn independent_coset_leader_profile(checks: &[u64], dimension: usize) -> Vec<(usize, usize)> {
    let syndrome_count = 1usize << checks.len();
    let mut profile = vec![(usize::MAX, 0usize); syndrome_count];

    for mask in 0..(1usize << dimension) {
        let syndrome = independent_syndrome(mask as u64, checks) as usize;
        let weight = mask.count_ones() as usize;
        let entry = &mut profile[syndrome];
        match weight.cmp(&entry.0) {
            std::cmp::Ordering::Less => *entry = (weight, 1),
            std::cmp::Ordering::Equal => entry.1 += 1,
            std::cmp::Ordering::Greater => {}
        }
    }

    profile
}

fn independent_syndrome(mask: u64, checks: &[u64]) -> u64 {
    let mut syndrome = 0u64;
    for (index, &check) in checks.iter().enumerate() {
        if (mask & check).count_ones() % 2 == 1 {
            syndrome |= 1u64 << index;
        }
    }
    syndrome
}

fn deterministic_probe_masks(mut state: u64, count: usize, dimension: usize) -> Vec<u64> {
    let mask = if dimension == 64 {
        u64::MAX
    } else {
        (1u64 << dimension) - 1
    };
    let mut output = Vec::with_capacity(count);
    for _ in 0..count {
        state = state
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0xD1B5_4A32_D192_ED03);
        output.push(state & mask);
    }
    output
}

#[test]
fn random_code_list_surface_matches_independent_oracles_and_is_deterministic() {
    let regimes = [
        (12usize, 4usize, 64u64, 0xE100_0000u64, 24usize),
        (20usize, 3usize, 32u64, 0xE200_0000u64, 24usize),
    ];

    fn probe(
        regimes: &[(usize, usize, u64, u64, usize)],
    ) -> Vec<(usize, usize, usize, usize, usize, [usize; 17])> {
        let mut ledgers = Vec::with_capacity(regimes.len());

        for &(dimension, rank, trials, seed_base, probes_per_code) in regimes {
            let mut qualifying_codes = 0usize;
            let mut checked_observations = 0usize;
            let mut no_match = 0usize;
            let mut max_multiplicity = 0usize;
            let mut histogram = [0usize; 17];

            for seed_offset in 0..trials {
                let seed = seed_base + seed_offset;
                let code = RandomLinearCode::generate(dimension, rank, seed);
                let production_codewords = code.enumerate();
                let codewords = independently_enumerated_codewords(&code);

                let mut production_masks = production_codewords
                    .iter()
                    .map(|word| word.words()[0])
                    .collect::<Vec<_>>();
                let mut independent_masks = codewords
                    .iter()
                    .map(|word| word.words()[0])
                    .collect::<Vec<_>>();
                production_masks.sort_unstable();
                independent_masks.sort_unstable();
                assert_eq!(
                    production_masks, independent_masks,
                    "production codeword enumeration diverged from independent basis reconstruction: regime={dimension}x{rank} seed=0x{seed:X}"
                );

                let independent_rank = independent_generator_rank(&code);
                assert_eq!(independent_rank, rank);
                assert_eq!(independent_rank, code.rank());
                assert_eq!(codewords.len(), 1usize << independent_rank);
                let min_distance = codewords
                    .iter()
                    .filter(|word| word.weight() > 0)
                    .map(|word| word.weight())
                    .min()
                    .expect("non-zero codeword");
                let unique_radius = (min_distance - 1) / 2;
                let bound = unique_radius + 1;
                let decoder = BoundedDistanceSyndromeDecoder::from_code(&code).expect("decoder");
                let checks = independent_parity_check_rows(&code);

                assert_eq!(checks.len(), dimension - rank);
                for codeword in &codewords {
                    let mask = codeword.words()[0];
                    assert_eq!(independent_syndrome(mask, &checks), 0);
                }

                for mask in
                    deterministic_probe_masks(seed ^ 0x5A17_0A1E, probes_per_code, dimension)
                {
                    let observation = error_from_mask(mask as usize, dimension);
                    let (nearest_distance, nearest) = nearest_codewords(&observation, &codewords);
                    let observed_syndrome = independent_syndrome(mask, &checks);
                    let parity_check_syndrome = decoder
                        .parity_check()
                        .syndrome(&observation)
                        .expect("same dimension");
                    assert_eq!(
                        parity_check_syndrome.words()[0],
                        observed_syndrome,
                        "production and independent parity-check oracles diverged: regime={dimension}x{rank} seed=0x{seed:X} mask={mask:#x}"
                    );

                    let listed_a = decoder.decode_with_minimum_list(&observation, bound, 32);
                    let listed_b = decoder.decode_with_minimum_list(&observation, bound, 32);
                    assert_eq!(
                        listed_a, listed_b,
                        "list surface was not deterministic: regime={dimension}x{rank} seed=0x{seed:X} mask={mask:#x}"
                    );

                    let shift = &codewords[(seed as usize) % codewords.len()];
                    let mut shifted_observation = observation.clone();
                    shifted_observation.xor_assign(shift);
                    let shifted = decoder.decode_with_minimum_list(&shifted_observation, bound, 32);
                    assert_eq!(
                        shifted.minimum_errors, listed_a.minimum_errors,
                        "codeword translation changed minimum error representatives: regime={dimension}x{rank} seed=0x{seed:X} mask={mask:#x}"
                    );
                    assert_eq!(
                        shifted.list_complete, listed_a.list_complete,
                        "codeword translation changed list completeness: regime={dimension}x{rank} seed=0x{seed:X} mask={mask:#x}"
                    );
                    assert_eq!(
                        shifted.nearest_codewords.len(),
                        listed_a.nearest_codewords.len(),
                        "codeword translation changed nearest-list cardinality: regime={dimension}x{rank} seed=0x{seed:X} mask={mask:#x}"
                    );
                    for (original, translated) in listed_a
                        .nearest_codewords
                        .iter()
                        .zip(&shifted.nearest_codewords)
                    {
                        let mut expected = original.clone();
                        expected.xor_assign(shift);
                        assert_eq!(
                            *translated, expected,
                            "codeword translation did not translate nearest codeword: regime={dimension}x{rank} seed=0x{seed:X} mask={mask:#x}"
                        );
                    }

                    match (&listed_a.outcome, &shifted.outcome) {
                        (
                            BoundedDistanceDecode::Unique {
                                error: original_error,
                                distance: original_distance,
                                ..
                            },
                            BoundedDistanceDecode::Unique {
                                error: shifted_error,
                                distance: shifted_distance,
                                ..
                            },
                        ) => {
                            assert_eq!(shifted_error, original_error);
                            assert_eq!(shifted_distance, original_distance);
                        }
                        (
                            BoundedDistanceDecode::Ambiguous {
                                distance: original_distance,
                                matching_error_patterns: original_matches,
                            },
                            BoundedDistanceDecode::Ambiguous {
                                distance: shifted_distance,
                                matching_error_patterns: shifted_matches,
                            },
                        ) => {
                            assert_eq!(shifted_distance, original_distance);
                            assert_eq!(shifted_matches, original_matches);
                        }
                        (
                            BoundedDistanceDecode::NoMatchWithinBound {
                                max_error_weight: original_bound,
                            },
                            BoundedDistanceDecode::NoMatchWithinBound {
                                max_error_weight: shifted_bound,
                            },
                        ) => assert_eq!(shifted_bound, original_bound),
                        (original, shifted) => panic!(
                            "codeword translation changed scalar outcome: original={original:?} shifted={shifted:?}"
                        ),
                    }

                    if nearest_distance > bound {
                        assert!(matches!(
                            listed_a.outcome,
                            BoundedDistanceDecode::NoMatchWithinBound { .. }
                        ));
                        assert!(listed_a.minimum_errors.is_empty());
                        assert!(listed_a.nearest_codewords.is_empty());
                        assert!(listed_a.list_complete);
                        no_match += 1;
                    } else {
                        assert!(listed_a.list_complete);
                        assert_eq!(listed_a.minimum_errors.len(), nearest.len());
                        assert_eq!(listed_a.nearest_codewords.len(), nearest.len());
                        assert_eq!(listed_a.outcome, decoder.decode(&observation, bound));

                        let mut seen = vec![false; nearest.len()];
                        for (error, codeword) in listed_a
                            .minimum_errors
                            .iter()
                            .zip(&listed_a.nearest_codewords)
                        {
                            assert_eq!(error.weight(), nearest_distance);
                            let error_syndrome = independent_syndrome(error.words()[0], &checks);
                            assert_eq!(error_syndrome, observed_syndrome);
                            assert_eq!(hamming_distance(&observation, codeword), nearest_distance);

                            let index = nearest
                                .iter()
                                .position(|&candidate| codewords[candidate] == *codeword)
                                .expect("listed codeword must be nearest");
                            assert!(!seen[index], "duplicate nearest codeword in list");
                            seen[index] = true;
                        }
                        assert!(seen.into_iter().all(|present| present));

                        max_multiplicity = max_multiplicity.max(nearest.len());
                        histogram[nearest.len()] += 1;
                    }

                    checked_observations += 1;
                }

                qualifying_codes += 1;
            }

            assert_eq!(qualifying_codes as u64, trials);
            assert_eq!(
                histogram[0], 0,
                "zero multiplicity must never be recorded as a decoded list"
            );
            assert!(max_multiplicity <= (1usize << rank));
            ledgers.push((
                dimension,
                rank,
                checked_observations,
                no_match,
                max_multiplicity,
                histogram,
            ));
        }

        ledgers
    }

    let first = probe(&regimes);
    let second = probe(&regimes);
    assert_eq!(
        first, second,
        "deterministic random-code list-size distribution changed between identical probes"
    );

    for &(dimension, rank, observations, no_match, max_multiplicity, histogram) in &first {
        let total_decoded = histogram.iter().skip(1).sum::<usize>();
        assert_eq!(total_decoded + no_match, observations);
        assert!(max_multiplicity <= 16);
        let histogram_serialized = histogram[1..]
            .iter()
            .enumerate()
            .map(|(multiplicity, count)| format!("{}:{}", multiplicity + 1, count))
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "RANDOM_LIST_ORACLE=dimension={dimension};rank={rank};observations={observations};no_match={no_match};max_multiplicity={max_multiplicity};histogram={histogram_serialized};deterministic=true;independent_syndrome_oracle=true;independent_codeword_oracle=true;translation_equivariant=true",
        );
    }
}

#[test]
fn random_small_and_low_rate_code_sweep_matches_exhaustive_oracle_within_guaranteed_radius() {
    let regimes = [
        (12usize, 4usize, 64u64, 0xD300_0000u64),
        (20, 3, 32, 0xD400_0000),
    ];
    let mut qualifying_by_regime = [0usize; 2];
    let mut checked_observations_by_regime = [0usize; 2];

    for (regime_index, &(dimension, rank, trials, seed_base)) in regimes.iter().enumerate() {
        for seed_offset in 0..trials {
            let seed = seed_base + seed_offset;
            let code = RandomLinearCode::generate(dimension, rank, seed);
            let codewords = code.enumerate();
            let min_distance = codewords
                .iter()
                .filter(|word| word.weight() > 0)
                .map(|word| hamming_distance(word, &BinaryCodeword::zero(dimension)))
                .min()
                .expect("non-zero codeword");
            let unique_radius = (min_distance - 1) / 2;
            if unique_radius == 0 || unique_radius > 4 {
                continue;
            }

            qualifying_by_regime[regime_index] += 1;
            let errors = error_words_up_to_weight(dimension, unique_radius);
            let decoder = BoundedDistanceSyndromeDecoder::from_code(&code).expect("decoder");

            for clean in &codewords {
                for error in &errors {
                    let observation = clean.bound(error);
                    let (nearest_distance, nearest) = nearest_codewords(&observation, &codewords);
                    assert_eq!(nearest_distance, error.weight());
                    assert_eq!(nearest.len(), 1);
                    assert_eq!(codewords[nearest[0]], *clean);

                    match decoder.decode(&observation, unique_radius) {
                        BoundedDistanceDecode::Unique {
                            codeword,
                            error: decoded_error,
                            distance,
                        } => {
                            assert_eq!(codeword, *clean);
                            assert_eq!(decoded_error, *error);
                            assert_eq!(distance, error.weight());
                        }
                        other => {
                            panic!(
                                "regime={dimension}x{rank} seed=0x{seed:X} radius={unique_radius}: expected unique decode, got {other:?}"
                            );
                        }
                    }
                    checked_observations_by_regime[regime_index] += 1;
                }
            }
        }
    }

    assert!(
        qualifying_by_regime[0] >= 16,
        "moderate-rate sweep found too few distance>=3 random codes: {}",
        qualifying_by_regime[0]
    );
    assert!(
        qualifying_by_regime[1] >= 8,
        "low-rate sweep found too few usable random codes: {}",
        qualifying_by_regime[1]
    );
    println!(
        "RANDOM_SYNDROME_SWEEP=regimes=12x4,20x3;moderate_rate_qualifying={};low_rate_qualifying={};moderate_rate_observations={};low_rate_observations={}",
        qualifying_by_regime[0],
        qualifying_by_regime[1],
        checked_observations_by_regime[0],
        checked_observations_by_regime[1],
    );
}

#[test]
fn beyond_unique_radius_decoder_matches_nearest_or_no_match_without_claiming_target_recovery() {
    let code = boundary_code();
    let codewords = code.enumerate();
    let decoder = BoundedDistanceSyndromeDecoder::from_code(&code).expect("decoder");
    let clean = code.encode(&[true, false]);

    let mut nearest_within_bound = 0usize;
    let mut nearest_outside_bound = 0usize;
    let mut unique_matches = 0usize;
    let mut ambiguous_matches = 0usize;

    for mask in 0..(1usize << 8) {
        let error = error_from_mask(mask, 8);
        if error.weight() != 3 {
            continue;
        }

        let observation = clean.bound(&error);
        let (nearest_distance, nearest) = nearest_codewords(&observation, &codewords);

        match decoder.decode(&observation, 2) {
            BoundedDistanceDecode::Unique {
                codeword,
                error: decoded_error,
                distance,
            } => {
                assert!(nearest_distance <= 2);
                assert_eq!(nearest.len(), 1);
                assert_eq!(distance, nearest_distance);
                assert_eq!(codeword, codewords[nearest[0]]);
                assert_eq!(hamming_distance(&observation, &codeword), distance);
                assert_ne!(
                    codeword, clean,
                    "a unique correction beyond the code's unique radius must not be credited as intended-target recovery"
                );
                assert_eq!(decoded_error.weight(), distance);
                unique_matches += 1;
                nearest_within_bound += 1;
            }
            BoundedDistanceDecode::Ambiguous {
                distance,
                matching_error_patterns,
            } => {
                assert!(nearest_distance <= 2);
                assert_eq!(nearest.len(), matching_error_patterns);
                assert_eq!(distance, nearest_distance);
                ambiguous_matches += 1;
                nearest_within_bound += 1;
            }
            BoundedDistanceDecode::NoMatchWithinBound { max_error_weight } => {
                assert_eq!(max_error_weight, 2);
                assert!(nearest_distance > 2);
                nearest_outside_bound += 1;
            }
            other => panic!("unexpected result for weight-3 corruption: {other:?}"),
        }
    }

    assert_eq!(nearest_within_bound + nearest_outside_bound, 56);
    assert_eq!(unique_matches + ambiguous_matches, nearest_within_bound);
    assert_eq!(ambiguous_matches, 0);
    assert_eq!(nearest_within_bound, 8);
    assert_eq!(nearest_outside_bound, 48);
    assert_eq!(unique_matches, 8);

    println!(
        "BEYOND_RADIUS_LEDGER=observations=56;bound=2;nearest_within_bound={nearest_within_bound};nearest_outside_bound={nearest_outside_bound};unique_matches={unique_matches};ambiguous_matches={ambiguous_matches};intended_target_recovery_claim=false"
    );
}
