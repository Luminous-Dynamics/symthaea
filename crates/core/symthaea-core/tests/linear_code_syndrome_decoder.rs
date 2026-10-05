// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root.

use symthaea_core::hdc::linear_code::{BinaryCodeword, RandomLinearCode, basis_rank};
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

    assert_eq!(parity_check.dimension(), 31);
    assert_eq!(parity_check.syndrome_dimension(), 26);
    assert_eq!(parity_check.rows().len(), 26);
    assert_eq!(parity_check.columns().len(), 31);
    assert_eq!(basis_rank(parity_check.rows(), 31), 26);

    for codeword in code.enumerate() {
        let syndrome = parity_check.syndrome(&codeword).expect("same dimension");
        assert_eq!(syndrome.weight(), 0);
    }

    println!(
        "PARITY_CHECK_LEDGER=dimension={};code_rank={};check_rank={};syndrome_dimension={};fingerprint={}",
        code.dimension(),
        code.rank(),
        basis_rank(parity_check.rows(), code.dimension()),
        parity_check.syndrome_dimension(),
        parity_check
            .fingerprint()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>(),
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

#[test]
fn random_small_and_low_rate_code_sweep_matches_exhaustive_oracle_within_guaranteed_radius() {
    let regimes = [(12usize, 4usize, 64u64, 0xD300_0000u64), (20, 3, 32, 0xD400_0000)];
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
                    let (nearest_distance, nearest) =
                        nearest_codewords(&observation, &codewords);
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
