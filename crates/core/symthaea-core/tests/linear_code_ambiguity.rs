use symthaea_core::hdc::linear_code::{
    BinaryCodeword, RandomLinearCode, basis_rank, recover_direct_sum_bound,
    recover_independent_bound, recover_linear_bound, recover_linear_bound_with_work,
    solve_linear_combination,
};

const CANONICAL_FIXTURE_DIMENSION: usize = 96;
const CANONICAL_FIXTURE_RANK: usize = 8;
const CANONICAL_FIXTURE_SEED: u64 = 0xC0DE;

#[test]
fn canonical_fixture_fingerprint_is_emitted() {
    let code = RandomLinearCode::generate(
        CANONICAL_FIXTURE_DIMENSION,
        CANONICAL_FIXTURE_RANK,
        CANONICAL_FIXTURE_SEED,
    );
    println!(
        "FIXTURE_SPEC=dimension={CANONICAL_FIXTURE_DIMENSION};rank={CANONICAL_FIXTURE_RANK};seed=0x{CANONICAL_FIXTURE_SEED:X}"
    );
    let fingerprint = code
        .fingerprint()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    println!("FIXTURE_FINGERPRINT={fingerprint}");
    assert_eq!(fingerprint.len(), 64);

    // The fingerprint is intentionally sensitive to basis ordering: the
    // ordered packed basis is part of fixture identity, not merely the span.
    let mut reordered_basis = code.basis().to_vec();
    reordered_basis.reverse();
    let reordered = RandomLinearCode::from_basis(reordered_basis).expect("reordered basis");
    assert_ne!(code.fingerprint(), reordered.fingerprint());

    // Dimension is also part of the domain-separated identity.
    let different_dimension = RandomLinearCode::generate(95, 8, 0xC0DE);
    assert_ne!(code.fingerprint(), different_dimension.fingerprint());
}

#[test]
fn small_code_geometry_ledger_is_exhaustively_self_consistent() {
    let dimension = 16;
    let code = RandomLinearCode::generate(dimension, 5, 0x6E01);
    let codewords = code.enumerate();

    let min_distance = codewords
        .iter()
        .filter(|word| word.weight() > 0)
        .map(BinaryCodeword::weight)
        .min()
        .expect("non-zero codeword must exist");

    let mut pairwise_min_distance = usize::MAX;
    let mut max_inner_product = isize::MIN;
    for left in &codewords {
        for right in &codewords {
            if left == right {
                continue;
            }
            let distance = left
                .words()
                .iter()
                .zip(right.words())
                .map(|(a, b)| (a ^ b).count_ones() as usize)
                .sum::<usize>();
            pairwise_min_distance = pairwise_min_distance.min(distance);

            let inner_product = dimension as isize - 2 * distance as isize;
            max_inner_product = max_inner_product.max(inner_product);
        }
    }

    assert_eq!(pairwise_min_distance, min_distance);
    assert_eq!(
        max_inner_product,
        dimension as isize - 2 * min_distance as isize
    );

    println!(
        "GEOMETRY_LEDGER=dimension={dimension};rank={};seed=0x6E01;codewords={};min_distance={min_distance};max_bipolar_inner_product={max_inner_product}",
        code.rank(),
        codewords.len(),
    );
}

#[test]
fn small_code_geometry_seed_sweep_is_exhaustively_self_consistent() {
    let dimension = 16;
    let rank = 5;
    let seeds = [0x6E01u64, 0x6E02, 0x6E03, 0x6E04];

    for seed in seeds {
        let code = RandomLinearCode::generate(dimension, rank, seed);
        let codewords = code.enumerate();
        let min_distance = codewords
            .iter()
            .filter(|word| word.weight() > 0)
            .map(BinaryCodeword::weight)
            .min()
            .expect("non-zero codeword must exist");

        let mut pairwise_min_distance = usize::MAX;
        let mut max_inner_product = isize::MIN;
        for left in &codewords {
            for right in &codewords {
                if left == right {
                    continue;
                }
                let distance = left
                    .words()
                    .iter()
                    .zip(right.words())
                    .map(|(a, b)| (a ^ b).count_ones() as usize)
                    .sum::<usize>();
                pairwise_min_distance = pairwise_min_distance.min(distance);
                max_inner_product =
                    max_inner_product.max(dimension as isize - 2 * distance as isize);
            }
        }

        assert_eq!(pairwise_min_distance, min_distance);
        assert_eq!(
            max_inner_product,
            dimension as isize - 2 * min_distance as isize
        );

        println!(
            "GEOMETRY_SWEEP=dimension={dimension};rank={rank};seed=0x{seed:X};codewords={};min_distance={min_distance};max_bipolar_inner_product={max_inner_product}",
            codewords.len(),
        );
    }
}

#[test]
fn recovery_work_ledger_is_deterministic_and_semantically_linked() {
    let (parent, left, right) =
        RandomLinearCode::generate_direct_sum(96, 6, 6, 0x1111).expect("valid direct sum");
    let target = left.basis()[0].bound(&right.basis()[0]);

    let (first, first_work) = recover_linear_bound_with_work(&target, &[&left, &right]);
    let (second, second_work) = recover_linear_bound_with_work(&target, &[&left, &right]);

    assert_eq!(first, second);
    assert_eq!(first_work, second_work);
    assert_eq!(first_work.retained_generators, parent.rank());
    assert_eq!(first_work.span_membership_checks, parent.rank());
    assert!(first_work.basis_rank_pivots > 0);
    assert!(first_work.solve_pivots > 0);
    assert_eq!(
        first_work.solve_basis_bit_probes,
        parent.dimension() * parent.rank()
    );
    assert_eq!(
        first_work.solve_matrix_word_cells,
        parent.dimension() * (parent.rank().div_ceil(64) + 1)
    );
    assert!(first_work.basis_rank_input_word_copies > 0);
    assert!(first_work.basis_rank_row_xor_words > 0);
    assert!(first_work.projection_word_xor_ops > 0);
    assert!(first_work.solve_row_xor_words > 0);

    println!("WORK_FIXTURE_SPEC=dimension=96;left_rank=6;right_rank=6;seed=0x1111");
    println!(
        "WORK_LEDGER=span_membership_checks={};basis_rank_pivots={};basis_rank_row_xor_words={};basis_rank_input_word_copies={};solve_basis_bit_probes={};solve_matrix_word_cells={};solve_pivots={};solve_row_xor_words={};retained_generators={};projection_word_xor_ops={}",
        first_work.span_membership_checks,
        first_work.basis_rank_pivots,
        first_work.basis_rank_row_xor_words,
        first_work.basis_rank_input_word_copies,
        first_work.solve_basis_bit_probes,
        first_work.solve_matrix_word_cells,
        first_work.solve_pivots,
        first_work.solve_row_xor_words,
        first_work.retained_generators,
        first_work.projection_word_xor_ops,
    );
}

#[test]
fn recovery_work_scales_over_small_structural_fixtures() {
    let fixtures = [
        (32usize, 2usize, 2usize, 0x3202u64),
        (64, 4, 4, 0x6404),
        (96, 6, 6, 0x9606),
    ];

    for &(dimension, left_rank, right_rank, seed) in &fixtures {
        let (parent, left, right) =
            RandomLinearCode::generate_direct_sum(dimension, left_rank, right_rank, seed)
                .expect("valid direct sum");
        let target = left.basis()[0].bound(&right.basis()[0]);
        let (recovered, work) = recover_linear_bound_with_work(&target, &[&left, &right]);

        assert!(recovered.is_some());
        assert_eq!(work.retained_generators, parent.rank());
        assert_eq!(work.span_membership_checks, parent.rank());
        // Every new independent generator increases the rank by exactly one.
        // The implementation performs one rank test before and after adding
        // each generator, giving 2 * sum(0..Delta) + Delta = Delta^2 pivots.
        assert_eq!(
            work.basis_rank_pivots,
            parent.rank() * parent.rank(),
            "rank-ledger contract at dimension {dimension}, rank {}",
            parent.rank()
        );

        println!(
            "SCALING_LEDGER=dimension={dimension};left_rank={left_rank};right_rank={right_rank};rank={};seed=0x{seed:X};span_checks={};rank_pivots={};rank_row_xor_words={};solve_pivots={};solve_row_xor_words={};retained_generators={}",
            parent.rank(),
            work.span_membership_checks,
            work.basis_rank_pivots,
            work.basis_rank_row_xor_words,
            work.solve_pivots,
            work.solve_row_xor_words,
            work.retained_generators,
        );
    }
}

#[test]
fn recovery_crosses_u64_packing_boundaries() {
    for &(dimension, seed) in &[(63usize, 0x6301u64), (64, 0x6401), (65, 0x6501)] {
        let (parent, left, right) =
            RandomLinearCode::generate_direct_sum(dimension, 2, 2, seed).expect("valid direct sum");
        let left_message = [true, false];
        let right_message = [false, true];
        let left_word = left.encode(&left_message);
        let right_word = right.encode(&right_message);
        let target = left_word.bound(&right_word);

        let (recovered, work) = recover_linear_bound_with_work(&target, &[&left, &right]);
        let recovered = recovered.expect("packing-boundary fixture must recover");
        assert_eq!(recovered, vec![left_word, right_word]);
        assert_eq!(work.retained_generators, parent.rank());
        assert_eq!(work.basis_rank_pivots, parent.rank() * parent.rank());

        let bipolar = target.to_bipolar();
        assert_eq!(BinaryCodeword::from_bipolar(&bipolar), Some(target.clone()));
        println!(
            "PACKING_BOUNDARY=dimension={dimension};rank={};seed=0x{seed:X};retained_generators={};basis_rank_pivots={}",
            parent.rank(),
            work.retained_generators,
            work.basis_rank_pivots,
        );
    }
}

#[test]
fn exhaustive_direct_sum_recovery_matches_all_factor_pairs() {
    let (parent, left, right) =
        RandomLinearCode::generate_direct_sum(10, 2, 2, 0x5A11).expect("valid direct sum");
    assert_eq!(basis_rank(parent.basis(), 10), parent.rank());

    for left_word in left.enumerate() {
        for right_word in right.enumerate() {
            let target = left_word.bound(&right_word);
            let recovered =
                recover_direct_sum_bound(&target, &left, &right).expect("pair must recover");
            assert_eq!(recovered, (left_word.clone(), right_word.clone()));
        }
    }
}

#[test]
fn arbitrary_same_subspace_bound_is_not_uniquely_identifiable() {
    let code = RandomLinearCode::generate(96, 8, 0xD00D);
    let a = code.encode(&[true, false, true, false, false, true, false, true]);
    let b = code.encode(&[false, true, true, false, true, false, false, true]);
    let composite = a.bound(&b);
    let zero = BinaryCodeword::zero(96);

    // (a, b) is valid, but so is (0, a XOR b). The algebra alone therefore
    // cannot establish a unique factor pair for arbitrary same-code factors.
    assert_eq!(a.bound(&b), composite);
    assert_eq!(zero.bound(&composite), composite);
    assert!(code.contains(&a));
    assert!(code.contains(&b));
    assert!(code.contains(&zero));
    assert!(code.contains(&composite));
    assert_ne!(a, zero);
    assert_ne!(b, composite);
}

#[test]
fn disjoint_factor_subspaces_have_unique_exhaustive_decomposition() {
    // Reproduce the paper's C = K × V construction: K and V are subcodes
    // obtained by partitioning one parent-code basis, so the direct-sum
    // relationship is structural rather than an accidental property of two
    // independently sampled subspaces.
    let (parent, left, right) =
        RandomLinearCode::generate_direct_sum(96, 6, 6, 0x1111).expect("valid direct sum");

    let mut combined_basis = left.basis().to_vec();
    combined_basis.extend(right.basis().iter().cloned());
    assert_eq!(combined_basis, parent.basis());
    assert_eq!(basis_rank(&combined_basis, 96), parent.rank());

    let left_message = [true, false, true, false, true, false];
    let right_message = [false, true, true, false, false, true];
    let a = left.encode(&left_message);
    let b = right.encode(&right_message);
    let composite = a.bound(&b);

    // Stage-C algebraic recovery: solve the clean bound directly in the
    // concatenated factor basis, then split the coefficients by domain.
    let mut basis = left.basis().to_vec();
    basis.extend(right.basis().iter().cloned());
    assert_eq!(basis_rank(&basis, 96), basis.len());
    let recovered =
        solve_linear_combination(&composite, &basis).expect("clean bound must be in span");
    let expected: Vec<bool> = left_message.into_iter().chain(right_message).collect();
    assert_eq!(recovered, expected);

    let (recovered_left, recovered_right) =
        recover_direct_sum_bound(&composite, &left, &right).expect("direct-sum recovery");
    assert_eq!(recovered_left, a);
    assert_eq!(recovered_right, b);

    let matches: Vec<_> = left
        .enumerate()
        .into_iter()
        .flat_map(|candidate_a| {
            let composite = composite.clone();
            right
                .enumerate()
                .into_iter()
                .filter_map(move |candidate_b| {
                    (candidate_a.bound(&candidate_b) == composite)
                        .then_some((candidate_a.clone(), candidate_b))
                })
        })
        .collect();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0], (a, b));

    // A corrupted query is classified explicitly. Choose a one-bit flip that
    // leaves the jointly generated factor span; this must not be silently
    // treated as a valid clean decomposition.
    let mut corrupted = composite.clone();
    let corrupted_index = (0..96)
        .find(|&index| {
            let mut candidate = composite.clone();
            candidate.set_bit(index, !candidate.bit(index));
            solve_linear_combination(&candidate, &basis).is_none()
        })
        .expect("at least one single-bit corruption should leave this low-rate span");
    corrupted.set_bit(corrupted_index, !corrupted.bit(corrupted_index));
    assert!(solve_linear_combination(&corrupted, &basis).is_none());

    let corrupted_matches: Vec<_> = left
        .enumerate()
        .into_iter()
        .flat_map(|candidate_a| {
            let corrupted = corrupted.clone();
            right
                .enumerate()
                .into_iter()
                .filter_map(move |candidate_b| {
                    (candidate_a.bound(&candidate_b) == corrupted)
                        .then_some((candidate_a.clone(), candidate_b))
                })
        })
        .collect();
    assert!(corrupted_matches.is_empty());
}

#[test]
fn packed_solver_handles_coefficient_word_boundary() {
    // 65 basis vectors forces the packed coefficient matrix across a u64
    // boundary; the solver must preserve both the 64th and 65th coefficients.
    let dimension = 160;
    let code = RandomLinearCode::generate(dimension, 65, 0x65AA);
    let message: Vec<bool> = (0..65).map(|index| index % 3 == 1).collect();
    let target = code.encode(&message);

    let recovered =
        solve_linear_combination(&target, code.basis()).expect("target must be in span");
    assert_eq!(recovered, message);
}

#[test]
fn dependent_basis_returns_a_solution_but_not_a_uniqueness_claim() {
    let code = RandomLinearCode::generate(96, 8, 0xDADA);
    let basis = code.basis();
    let mut dependent = basis.to_vec();
    dependent.push(basis[0].clone());

    let target = basis[2].bound(&basis[5]);
    let recovered =
        solve_linear_combination(&target, &dependent).expect("target must be in dependent span");

    let mut reconstructed = BinaryCodeword::zero(96);
    for (coefficient, vector) in recovered.iter().zip(&dependent) {
        if *coefficient {
            reconstructed.xor_assign(vector);
        }
    }
    assert_eq!(reconstructed, target);
    assert_eq!(basis_rank(&dependent, 96), 8);
}

#[test]
fn solver_rejects_dimension_mismatch_without_panicking() {
    let code = RandomLinearCode::generate(96, 4, 0x5151);
    let target = BinaryCodeword::zero(95);
    assert!(solve_linear_combination(&target, code.basis()).is_none());
}

#[test]
fn three_partitioned_factor_subcodes_recover_exactly() {
    // Extend the paper's parent-code partition construction to F=3 while
    // keeping the combined basis independent. This validates the generalized
    // implementation without introducing overlap ambiguity.
    let parent = RandomLinearCode::generate(96, 12, 0x3333);
    let left = RandomLinearCode::from_basis(parent.basis()[..4].to_vec()).expect("left subcode");
    let middle =
        RandomLinearCode::from_basis(parent.basis()[4..8].to_vec()).expect("middle subcode");
    let right =
        RandomLinearCode::from_basis(parent.basis()[8..12].to_vec()).expect("right subcode");

    let left_message = [true, false, true, true];
    let middle_message = [false, true, true, false];
    let right_message = [true, true, false, true];
    let left_word = left.encode(&left_message);
    let middle_word = middle.encode(&middle_message);
    let right_word = right.encode(&right_message);

    let target = left_word.bound(&middle_word).bound(&right_word);
    let recovered = recover_independent_bound(&target, &[&left, &middle, &right])
        .expect("independent factor subcodes must recover");

    assert_eq!(
        recovered,
        vec![left_word.clone(), middle_word.clone(), right_word.clone()]
    );

    let exhaustive_matches: Vec<_> = left
        .enumerate()
        .into_iter()
        .flat_map(|a| {
            let target = target.clone();
            let middle_words = middle.enumerate();
            let right_words = right.enumerate();
            middle_words.into_iter().flat_map(move |b| {
                let target = target.clone();
                let a = a.clone();
                right_words.clone().into_iter().filter_map(move |c| {
                    (a.bound(&b).bound(&c) == target).then_some((a.clone(), b.clone(), c))
                })
            })
        })
        .collect();

    assert_eq!(
        exhaustive_matches,
        vec![(left_word, middle_word, right_word)]
    );
}

#[test]
fn exhaustive_noise_profile_separates_in_span_from_out_of_span_errors() {
    // Exact GF(2) recovery is not itself error correction. Enumerate the full
    // Boolean error space on a small fixture and classify each error by
    // whether it lies in the code subspace.
    let code = RandomLinearCode::generate(16, 5, 0xE770);
    let message = [true, false, true, false, true];
    let clean = code.encode(&message);

    let mut in_span_by_weight = [0usize; 17];
    let mut out_of_span_by_weight = [0usize; 17];
    let total_patterns = 1usize << 16;

    for mask in 0..total_patterns {
        let mut error = BinaryCodeword::zero(16);
        let mut weight = 0usize;
        for index in 0..16 {
            if (mask >> index) & 1 == 1 {
                error.set_bit(index, true);
                weight += 1;
            }
        }

        let corrupted = clean.bound(&error);
        if code.contains(&error) {
            in_span_by_weight[weight] += 1;

            // A non-zero in-span error remains exactly solvable, but the
            // recovered codeword is different from the original message.
            if weight > 0 {
                let recovered =
                    solve_linear_combination(&corrupted, code.basis()).expect("in-span target");
                assert_ne!(recovered, message);
            }
        } else {
            out_of_span_by_weight[weight] += 1;
            assert!(solve_linear_combination(&corrupted, code.basis()).is_none());
        }
    }

    // A rank-r binary subspace contains exactly 2^r of the 2^n possible
    // error vectors.
    assert_eq!(in_span_by_weight.iter().sum::<usize>(), 1 << code.rank());
    assert_eq!(
        out_of_span_by_weight.iter().sum::<usize>(),
        total_patterns - (1 << code.rank())
    );
    assert_eq!(in_span_by_weight[0], 1);
    assert_eq!(out_of_span_by_weight[0], 0);
}

#[test]
fn exhaustive_bounded_distance_oracle_separates_detection_from_correction() {
    // The exact GF(2) solver only answers whether an observation is in the
    // code span. A bounded-distance decoder has a different contract: for
    // errors below half the minimum distance, the clean codeword is the
    // unique nearest codeword even though the corrupted observation itself
    // is outside the code.
    let code = RandomLinearCode::generate(16, 5, 0xE770);
    let codewords = code.enumerate();
    let hamming_distance = |left: &BinaryCodeword, right: &BinaryCodeword| {
        left.words()
            .iter()
            .zip(right.words())
            .map(|(a, b)| (a ^ b).count_ones() as usize)
            .sum::<usize>()
    };

    let min_distance = codewords
        .iter()
        .filter(|word| word.weight() > 0)
        .map(|word| hamming_distance(word, &BinaryCodeword::zero(16)))
        .min()
        .expect("non-zero codeword must exist");
    assert!(
        min_distance >= 3,
        "deterministic noise fixture must admit a non-trivial correction radius"
    );
    let correction_radius = (min_distance - 1) / 2;

    let clean = code.encode(&[true, false, true, false, true]);
    let mut checked_patterns = 0usize;
    let mut observed_out_of_span = 0usize;

    for mask in 0..(1usize << 16) {
        let error_weight = mask.count_ones() as usize;
        if error_weight > correction_radius {
            continue;
        }

        let mut error = BinaryCodeword::zero(16);
        for index in 0..16 {
            if (mask >> index) & 1 == 1 {
                error.set_bit(index, true);
            }
        }

        let corrupted = clean.bound(&error);
        let distances: Vec<usize> = codewords
            .iter()
            .map(|candidate| hamming_distance(&corrupted, candidate))
            .collect();
        let nearest_distance = *distances.iter().min().expect("codebook is non-empty");
        let nearest_indices: Vec<usize> = distances
            .iter()
            .enumerate()
            .filter_map(|(index, &distance)| (distance == nearest_distance).then_some(index))
            .collect();

        // This is an exhaustive oracle, not a production decoder. Coding
        // theory predicts unique nearest-codeword recovery below d_min / 2.
        assert_eq!(nearest_indices.len(), 1);
        assert_eq!(codewords[nearest_indices[0]], clean);
        assert_eq!(nearest_distance, error_weight);

        if error_weight > 0 {
            // No non-zero codeword can have weight below d_min, so these
            // non-zero low-weight errors are necessarily outside the code.
            assert!(!code.contains(&error));
            assert!(solve_linear_combination(&corrupted, code.basis()).is_none());
            observed_out_of_span += 1;
        }
        checked_patterns += 1;
    }

    assert!(checked_patterns > 1);
    assert!(observed_out_of_span > 0);
}

#[test]
fn minimum_distance_boundary_can_map_one_valid_codeword_to_another() {
    // At d_min, a corruption can itself be a non-zero codeword. The observed
    // vector is then another valid codeword, so an exact span solver has no
    // information with which to recover the originally transmitted word.
    let code = RandomLinearCode::generate(16, 5, 0xE770);
    let clean_message = [true, false, true, false, true];
    let clean = code.encode(&clean_message);

    let mut min_distance = usize::MAX;
    let mut min_error = BinaryCodeword::zero(16);
    let mut min_error_mask = 0usize;

    for mask in 1..(1usize << code.rank()) {
        let error: BinaryCodeword = code.encode(
            &(0..code.rank())
                .map(|bit| (mask >> bit) & 1 == 1)
                .collect::<Vec<_>>(),
        );
        if error.weight() < min_distance {
            min_distance = error.weight();
            min_error = error;
            min_error_mask = mask;
        }
    }

    assert!(min_distance >= 3);
    let corrupted = clean.bound(&min_error);
    assert!(code.contains(&min_error));
    assert!(code.contains(&corrupted));

    let recovered = solve_linear_combination(&corrupted, code.basis())
        .expect("a codeword corruption remains in the exact span");

    let expected_message: Vec<bool> = (0..code.rank())
        .map(|bit| {
            let clean_bit = clean_message[bit];
            let error_bit = (min_error_mask >> bit) & 1 == 1;
            clean_bit ^ error_bit
        })
        .collect();

    assert_eq!(recovered, expected_message);
    assert_ne!(recovered, clean_message);
    assert_eq!(min_error.weight(), min_distance);
}

#[test]
fn exhaustive_overlapping_recovery_preserves_valid_factorization() {
    let parent = RandomLinearCode::generate(96, 9, 0x4949);
    let left = RandomLinearCode::from_basis(parent.basis()[..6].to_vec()).expect("left subcode");
    let right =
        RandomLinearCode::from_basis(parent.basis()[3..9].to_vec()).expect("overlapping subcode");

    // The two factors overlap in three generators. Every parent-codeword is
    // therefore a clean target in the union span, but some targets admit more
    // than one factorization. Exhaust all 2^9 parent messages to verify that
    // the general recovery path always returns a valid factorization.
    for mask in 0..(1usize << parent.rank()) {
        let message: Vec<bool> = (0..parent.rank())
            .map(|bit| (mask >> bit) & 1 == 1)
            .collect();
        let target = parent.encode(&message);

        let recovered =
            recover_linear_bound(&target, &[&left, &right]).expect("target is in union span");
        assert_eq!(recovered.len(), 2);
        assert!(left.contains(&recovered[0]));
        assert!(right.contains(&recovered[1]));
        assert_eq!(recovered[0].bound(&recovered[1]), target);
    }
}

#[test]
fn general_recovery_rejects_target_outside_union_span() {
    let parent = RandomLinearCode::generate(16, 5, 0xBADA);
    let left = RandomLinearCode::from_basis(parent.basis()[..3].to_vec()).expect("left subcode");
    let right = RandomLinearCode::from_basis(parent.basis()[3..].to_vec()).expect("right subcode");

    let mut outsider = BinaryCodeword::zero(16);
    let index = (0..16)
        .find(|&index| {
            let mut candidate = BinaryCodeword::zero(16);
            candidate.set_bit(index, true);
            !parent.contains(&candidate)
        })
        .expect("low-rate parent code must have an outsider");
    outsider.set_bit(index, true);

    assert!(!parent.contains(&outsider));
    assert!(recover_linear_bound(&outsider, &[&left, &right]).is_none());
}

#[test]
fn overlapping_subspaces_without_shared_generators_still_recover_representatively() {
    let parent = RandomLinearCode::generate(96, 4, 0x7A7A);
    let left = RandomLinearCode::from_basis(parent.basis()[..2].to_vec()).expect("left subcode");

    // The right subcode intersects the left span, but shares no generator
    // vector literally: its first basis vector is g0 XOR g1 from the left
    // basis. This exercises overlap at the subspace level rather than only
    // through duplicate generator identities.
    let shared = parent.basis()[0].bound(&parent.basis()[1]);
    let right_basis = vec![shared.clone(), parent.basis()[2].clone()];
    let right = RandomLinearCode::from_basis(right_basis).expect("right subcode");

    assert!(left.contains(&shared));
    assert!(right.contains(&shared));
    assert_ne!(left.basis()[0], right.basis()[0]);
    assert_ne!(left.basis()[1], right.basis()[0]);

    let target = shared.clone();
    let recovered =
        recover_linear_bound(&target, &[&left, &right]).expect("target is in the union span");

    assert_eq!(recovered.len(), 2);
    assert!(left.contains(&recovered[0]));
    assert!(right.contains(&recovered[1]));
    assert_eq!(recovered[0].bound(&recovered[1]), target);

    // The factorization is not uniquely identifiable because the shared
    // vector can be assigned to either factor through different valid pairs.
    let matches: Vec<_> = left
        .enumerate()
        .into_iter()
        .flat_map(|a| {
            let target = target.clone();
            right
                .enumerate()
                .into_iter()
                .filter_map(move |b| (a.bound(&b) == target).then_some((a.clone(), b)))
        })
        .collect();
    assert!(matches.len() > 1);

    assert!(recover_independent_bound(&target, &[&left, &right]).is_none());
}

#[test]
fn overlapping_recovery_is_order_deterministic_but_not_permutation_invariant() {
    let parent = RandomLinearCode::generate(96, 4, 0x7B7B);
    let left = RandomLinearCode::from_basis(parent.basis()[..2].to_vec()).expect("left subcode");
    let shared = parent.basis()[0].bound(&parent.basis()[1]);
    let right = RandomLinearCode::from_basis(vec![shared, parent.basis()[2].clone()])
        .expect("right subcode");

    let target = parent.basis()[0].clone();
    let first = recover_linear_bound(&target, &[&left, &right]).expect("forward recovery");
    let repeated = recover_linear_bound(&target, &[&left, &right]).expect("repeat recovery");
    assert_eq!(first, repeated);
    assert_eq!(first.len(), 2);
    assert!(left.contains(&first[0]));
    assert!(right.contains(&first[1]));
    assert_eq!(first[0].bound(&first[1]), target);

    let reversed = recover_linear_bound(&target, &[&right, &left]).expect("reversed recovery");
    assert_eq!(reversed.len(), 2);
    assert!(right.contains(&reversed[0]));
    assert!(left.contains(&reversed[1]));
    assert_eq!(reversed[0].bound(&reversed[1]), target);

    // The theorem's representative construction is deterministic for a fixed
    // factor ordering, but overlap means the representative is allowed to
    // change when the factor ordering changes.
    assert_ne!(first, reversed);
}

#[test]
fn overlapping_recovery_remains_valid_under_basis_reordering() {
    let parent = RandomLinearCode::generate(12, 6, 0xB515);
    let left = RandomLinearCode::from_basis(parent.basis()[..4].to_vec()).expect("left subcode");
    let right = RandomLinearCode::from_basis(parent.basis()[2..6].to_vec()).expect("right subcode");

    let mut left_reordered_basis = left.basis().to_vec();
    left_reordered_basis.reverse();
    let left_reordered =
        RandomLinearCode::from_basis(left_reordered_basis).expect("reordered left basis");

    let mut right_reordered_basis = right.basis().to_vec();
    right_reordered_basis.reverse();
    let right_reordered =
        RandomLinearCode::from_basis(right_reordered_basis).expect("reordered right basis");

    for target in parent.enumerate() {
        let recovered =
            recover_linear_bound(&target, &[&left, &right]).expect("original ordering recovers");
        assert_eq!(recovered.len(), 2);
        assert!(left.contains(&recovered[0]));
        assert!(right.contains(&recovered[1]));
        assert_eq!(recovered[0].bound(&recovered[1]), target);

        let reordered = recover_linear_bound(&target, &[&left_reordered, &right_reordered])
            .expect("reordered bases recover");
        assert_eq!(reordered.len(), 2);
        assert!(left_reordered.contains(&reordered[0]));
        assert!(right_reordered.contains(&reordered[1]));
        assert_eq!(reordered[0].bound(&reordered[1]), target);
    }
}

#[test]
fn overlapping_factor_bases_allow_valid_recovery_without_uniqueness() {
    let parent = RandomLinearCode::generate(96, 8, 0x4444);
    let left = RandomLinearCode::from_basis(parent.basis()[..4].to_vec()).expect("left subcode");
    let overlapping =
        RandomLinearCode::from_basis(parent.basis()[2..6].to_vec()).expect("overlapping subcode");

    // The shared generator is intentionally present in both factor bases.
    // Its observation therefore has at least two valid decompositions:
    // (shared, 0) and (0, shared). The general theorem permits recovery of
    // a representative factorization, but does not imply uniqueness.
    let target = parent.basis()[2].clone();

    let recovered =
        recover_linear_bound(&target, &[&left, &overlapping]).expect("target is in the union span");
    assert_eq!(recovered.len(), 2);
    assert!(left.contains(&recovered[0]));
    assert!(overlapping.contains(&recovered[1]));
    assert_eq!(recovered[0].bound(&recovered[1]), target);

    let matches: Vec<_> = left
        .enumerate()
        .into_iter()
        .flat_map(|a| {
            let target = target.clone();
            overlapping
                .enumerate()
                .into_iter()
                .filter_map(move |b| (a.bound(&b) == target).then_some((a.clone(), b)))
        })
        .collect();
    assert!(matches.len() > 1);

    // The stricter API intentionally refuses to label this overlapping basis
    // pair as an independent/direct-sum recovery problem.
    assert!(recover_independent_bound(&target, &[&left, &overlapping]).is_none());
}
