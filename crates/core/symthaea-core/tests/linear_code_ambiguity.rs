use blake3::Hasher;
use symthaea_core::hdc::linear_code::{
    BinaryCodeword, ExactPowerOfTwo, LinearCodeWork, RandomLinearCode, basis_rank,
    factorization_affine_fiber, factorization_algebra, factorization_count_for_target,
    factorization_dependency_witness, factorization_kernel_basis, recover_direct_sum_bound,
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

    let codewords = code.enumerate();
    let nonzero_weights: Vec<_> = codewords
        .iter()
        .filter(|word| word.weight() > 0)
        .map(BinaryCodeword::weight)
        .collect();
    let min_weight = *nonzero_weights
        .iter()
        .min()
        .expect("non-zero codeword must exist");
    let max_weight = *nonzero_weights
        .iter()
        .max()
        .expect("non-zero codeword must exist");
    let mut pairwise_min_distance = usize::MAX;
    let mut pairwise_max_distance = 0usize;
    let mut max_inner_product = isize::MIN;
    let mut min_inner_product = isize::MAX;
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
            pairwise_max_distance = pairwise_max_distance.max(distance);
            let inner_product = CANONICAL_FIXTURE_DIMENSION as isize - 2 * distance as isize;
            max_inner_product = max_inner_product.max(inner_product);
            min_inner_product = min_inner_product.min(inner_product);
        }
    }
    let max_abs_inner_product = max_inner_product
        .unsigned_abs()
        .max(min_inner_product.unsigned_abs());
    assert_eq!(pairwise_min_distance, min_weight);
    assert_eq!(pairwise_max_distance, max_weight);
    assert_eq!(
        max_abs_inner_product,
        (CANONICAL_FIXTURE_DIMENSION as isize - 2 * min_weight as isize)
            .unsigned_abs()
            .max((CANONICAL_FIXTURE_DIMENSION as isize - 2 * max_weight as isize).unsigned_abs())
    );
    println!(
        "FIXTURE_GEOMETRY=dimension={CANONICAL_FIXTURE_DIMENSION};rank={CANONICAL_FIXTURE_RANK};seed=0x{CANONICAL_FIXTURE_SEED:X};codewords={};min_distance={pairwise_min_distance};max_distance={pairwise_max_distance};max_bipolar_inner_product={max_inner_product};min_bipolar_inner_product={min_inner_product};max_abs_bipolar_inner_product={max_abs_inner_product};balanced_epsilon_num={max_abs_inner_product};balanced_epsilon_den={}",
        codewords.len(),
        2 * CANONICAL_FIXTURE_DIMENSION,
    );

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
        let nonzero_weights: Vec<_> = codewords
            .iter()
            .filter(|word| word.weight() > 0)
            .map(BinaryCodeword::weight)
            .collect();
        let min_weight = *nonzero_weights
            .iter()
            .min()
            .expect("non-zero codeword must exist");
        let max_weight = *nonzero_weights
            .iter()
            .max()
            .expect("non-zero codeword must exist");

        let mut pairwise_min_distance = usize::MAX;
        let mut pairwise_max_distance = 0usize;
        let mut max_inner_product = isize::MIN;
        let mut min_inner_product = isize::MAX;
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
                pairwise_max_distance = pairwise_max_distance.max(distance);
                let inner_product = dimension as isize - 2 * distance as isize;
                max_inner_product = max_inner_product.max(inner_product);
                min_inner_product = min_inner_product.min(inner_product);
            }
        }

        let max_abs_inner_product = max_inner_product
            .unsigned_abs()
            .max(min_inner_product.unsigned_abs());
        assert_eq!(pairwise_min_distance, min_weight);
        assert_eq!(pairwise_max_distance, max_weight);
        assert_eq!(
            max_abs_inner_product,
            (dimension as isize - 2 * min_weight as isize)
                .unsigned_abs()
                .max((dimension as isize - 2 * max_weight as isize).unsigned_abs())
        );

        println!(
            "GEOMETRY_SWEEP=dimension={dimension};rank={rank};seed=0x{seed:X};codewords={};min_distance={pairwise_min_distance};max_distance={pairwise_max_distance};max_bipolar_inner_product={max_inner_product};min_bipolar_inner_product={min_inner_product};max_abs_bipolar_inner_product={max_abs_inner_product};balanced_epsilon_num={max_abs_inner_product};balanced_epsilon_den={}",
            codewords.len(),
            2 * dimension,
        );
    }
}

#[test]
fn published_parameter_storage_ledger_is_exact_and_packed_consistent() {
    let dimensions = [500usize, 1000, 2000];
    let ranks = [3usize, 5, 7];
    let factor_counts = [3usize, 4, 5];

    let mut cases = 0usize;
    let mut total_theoretical_codebook_bits = 0u128;
    let mut total_theoretical_generator_bits = 0u128;
    let mut total_packed_generator_bytes = 0u128;
    let mut total_packed_target_bytes = 0u128;

    for &dimension in &dimensions {
        let packed_words = BinaryCodeword::zero(dimension).words().len() as u128;
        let packed_target_bytes = packed_words * 8;

        for &rank in &ranks {
            let codeword_count = 1u128 << rank;

            for &factor_count in &factor_counts {
                let arbitrary_codebook_bits =
                    (factor_count as u128 * codeword_count + 1) * dimension as u128;
                let generator_matrix_bits = (factor_count * rank + 1) as u128 * dimension as u128;
                let packed_generator_bytes = (factor_count * rank) as u128 * packed_words * 8;

                total_theoretical_codebook_bits += arbitrary_codebook_bits;
                total_theoretical_generator_bits += generator_matrix_bits;
                total_packed_generator_bytes += packed_generator_bytes;
                total_packed_target_bytes += packed_target_bytes;
                cases += 1;

                assert_eq!(
                    packed_generator_bytes * 8,
                    (factor_count * rank) as u128 * packed_words * 64
                );
                assert!(generator_matrix_bits <= arbitrary_codebook_bits);
            }
        }
    }

    let expected_target_bytes = dimensions
        .iter()
        .map(|&dimension| BinaryCodeword::zero(dimension).words().len() as u128 * 8)
        .sum::<u128>()
        * ranks.len() as u128
        * factor_counts.len() as u128;
    assert_eq!(cases, 27);
    assert!(total_theoretical_generator_bits < total_theoretical_codebook_bits);
    assert!(total_packed_generator_bytes > 0);
    assert_eq!(total_packed_target_bytes, expected_target_bytes);

    println!(
        "STORAGE_LEDGER=dimensions=500,1000,2000;ranks=3,5,7;factors=3,4,5;cases={cases};theoretical_codebook_bits={total_theoretical_codebook_bits};theoretical_generator_matrix_bits={total_theoretical_generator_bits};packed_generator_bytes={total_packed_generator_bytes};packed_target_bytes={total_packed_target_bytes}"
    );
}

#[test]
fn published_parameter_search_space_ledger_is_exact() {
    let dimensions = [500usize, 1000, 2000];
    let ranks = [3usize, 5, 7];
    let factor_counts = [3usize, 4, 5];

    let mut cases = 0usize;
    let mut total_exhaustive_candidates = 0u128;
    let mut max_exhaustive_candidates = 0u128;
    let mut total_factor_slots = 0usize;
    let mut total_linear_variables = 0usize;
    let mut total_solver_matrix_word_cells = 0u128;

    for &dimension in &dimensions {
        for &rank in &ranks {
            for &factor_count in &factor_counts {
                let factor_slots = factor_count;
                let linear_variables = factor_count * rank;
                let exhaustive_candidates = 1u128 << linear_variables;
                let coefficient_words = linear_variables.div_ceil(64) as u128;
                let solver_matrix_word_cells = dimension as u128 * (coefficient_words + 1);

                assert_eq!(
                    exhaustive_candidates,
                    (0..factor_count).map(|_| 1u128 << rank).product::<u128>()
                );
                assert_eq!(
                    solver_matrix_word_cells,
                    dimension as u128 * (linear_variables.div_ceil(64) as u128 + 1)
                );

                total_exhaustive_candidates += exhaustive_candidates;
                max_exhaustive_candidates = max_exhaustive_candidates.max(exhaustive_candidates);
                total_factor_slots += factor_slots;
                total_linear_variables += linear_variables;
                total_solver_matrix_word_cells += solver_matrix_word_cells;
                cases += 1;
            }
        }
    }

    assert_eq!(cases, 27);
    assert_eq!(total_factor_slots, 108);
    assert!(max_exhaustive_candidates >= (1u128 << 35));
    assert!(total_exhaustive_candidates > total_solver_matrix_word_cells);

    println!(
        "SEARCH_LEDGER=dimensions=500,1000,2000;ranks=3,5,7;factors=3,4,5;cases={cases};total_exhaustive_candidates={total_exhaustive_candidates};max_exhaustive_candidates={max_exhaustive_candidates};total_factor_slots={total_factor_slots};total_linear_variables={total_linear_variables};total_solver_matrix_word_cells={total_solver_matrix_word_cells}"
    );
}

#[test]
fn dependency_witness_ledger_is_canonical_and_verifiable() {
    let c1 =
        RandomLinearCode::from_basis(vec![BinaryCodeword::from_words(3, vec![0b001])]).expect("c1");
    let c2 =
        RandomLinearCode::from_basis(vec![BinaryCodeword::from_words(3, vec![0b010])]).expect("c2");
    let c3 =
        RandomLinearCode::from_basis(vec![BinaryCodeword::from_words(3, vec![0b011])]).expect("c3");
    let factors = [&c1, &c2, &c3];

    let algebra = factorization_algebra(&factors).expect("algebra");
    let kernel = factorization_kernel_basis(&factors).expect("kernel basis");
    assert_eq!(kernel.len(), algebra.kernel_dimension);
    assert_eq!(kernel.len(), 1);

    let first = factorization_dependency_witness(&factors).expect("dependency exists");
    let second = factorization_dependency_witness(&factors).expect("dependency exists");
    assert_eq!(first, second);
    assert_eq!(first, kernel[0]);
    assert_eq!(first.generator_coefficients, vec![true, true, true]);
    assert_eq!(first.factor_support, vec![0, 1, 2]);
    assert_eq!(first.dependent_generator_index, 2);
    assert_eq!(first.generator_support_size(), 3);
    assert_eq!(first.factor_support_size(), 3);
    assert!(first.verifies_against(&factors));

    let mut tampered = first.clone();
    tampered.factor_support = vec![0, 1];
    assert!(!tampered.verifies_against(&factors));

    let repeated = RandomLinearCode::from_basis(vec![
        BinaryCodeword::from_words(4, vec![0b0001]),
        BinaryCodeword::from_words(4, vec![0b0010]),
    ])
    .expect("repeated 2D code");
    let repeated_factors = [&repeated, &repeated, &repeated];
    let repeated_algebra = factorization_algebra(&repeated_factors).expect("repeated algebra");
    assert_eq!(repeated_algebra.factor_dimension_sum, 6);
    assert_eq!(repeated_algebra.union_generator_rank, 2);
    assert_eq!(repeated_algebra.kernel_dimension, 4);
    let repeated_kernel =
        factorization_kernel_basis(&repeated_factors).expect("repeated kernel basis");
    assert_eq!(repeated_kernel.len(), 4);
    assert_eq!(
        repeated_kernel
            .iter()
            .filter(|witness| witness.verifies_against(&repeated_factors))
            .count(),
        4
    );
    let repeated_coefficient_basis = repeated_kernel
        .iter()
        .map(|witness| {
            let mut vector = BinaryCodeword::zero(witness.generator_coefficients.len());
            for (index, coefficient) in witness.generator_coefficients.iter().enumerate() {
                vector.set_bit(index, *coefficient);
            }
            vector
        })
        .collect::<Vec<_>>();
    assert_eq!(
        basis_rank(
            &repeated_coefficient_basis,
            repeated_algebra.factor_dimension_sum,
        ),
        repeated_algebra.kernel_dimension
    );
    let repeated_basis = repeated_factors
        .iter()
        .flat_map(|factor| factor.basis().iter().cloned())
        .collect::<Vec<_>>();
    let repeated_target = repeated.encode(&[true, false]);
    let base_coefficients =
        solve_linear_combination(&repeated_target, &repeated_basis).expect("base coefficients");
    let expected_fiber_size = 1usize << repeated_algebra.kernel_dimension;
    let mut fiber_coefficients = Vec::with_capacity(expected_fiber_size);
    for mask in 0..expected_fiber_size {
        let mut coefficients = base_coefficients.clone();
        for (kernel_index, witness) in repeated_kernel.iter().enumerate() {
            if (mask >> kernel_index) & 1 == 1 {
                for (index, coefficient) in witness.generator_coefficients.iter().enumerate() {
                    coefficients[index] ^= *coefficient;
                }
            }
        }
        assert!(!fiber_coefficients.contains(&coefficients));
        let mut reconstructed = BinaryCodeword::zero(4);
        for (coefficient, generator) in coefficients.iter().zip(&repeated_basis) {
            if *coefficient {
                reconstructed.xor_assign(generator);
            }
        }
        assert_eq!(reconstructed, repeated_target);
        fiber_coefficients.push(coefficients);
    }
    let mut target_counts = repeated
        .enumerate()
        .iter()
        .map(|target| (target.clone(), 0usize))
        .collect::<Vec<_>>();
    for first in repeated.enumerate() {
        for second in repeated.enumerate() {
            for third in repeated.enumerate() {
                let target = first.bound(&second).bound(&third);
                let (_, count) = target_counts
                    .iter_mut()
                    .find(|(candidate, _)| *candidate == target)
                    .expect("target is reachable");
                *count += 1;
            }
        }
    }
    assert!(
        target_counts
            .iter()
            .all(|(_, count)| *count == expected_fiber_size)
    );
    assert_eq!(target_counts.len(), 4);
    assert_eq!(fiber_coefficients.len(), expected_fiber_size);
    assert!(!repeated_algebra.factorization_count_per_target.is_one());
    assert_eq!(
        factorization_count_for_target(&repeated.encode(&[true, false]), &repeated_factors)
            .expect("repeated target"),
        repeated_algebra.factorization_count_per_target
    );

    println!(
        "DEPENDENCY_WITNESS=fixture=three-way;kernel_dimension={};generator_coefficients=111;factor_support=0,1,2;dependent_generator_index={};generator_support_size={};factor_support_size={};verifies=true",
        algebra.kernel_dimension,
        first.dependent_generator_index,
        first.generator_support_size(),
        first.factor_support_size(),
    );
}
#[test]
fn independent_dependency_witness_is_absent() {
    let (_, left, right) =
        RandomLinearCode::generate_direct_sum(32, 3, 4, 0x51A7).expect("valid direct sum");
    assert!(factorization_dependency_witness(&[&left, &right]).is_none());
}
#[test]
fn algebraic_multiplicity_ledger_is_exhaustively_self_consistent() {
    let shared = RandomLinearCode::from_basis(vec![BinaryCodeword::from_words(3, vec![0b001])])
        .expect("shared code");
    let orthogonal = RandomLinearCode::from_basis(vec![BinaryCodeword::from_words(3, vec![0b010])])
        .expect("orthogonal code");

    let pair = vec![&shared, &shared];
    let pair_algebra = factorization_algebra(&pair).expect("pair algebra");
    assert_eq!(pair_algebra.factor_dimension_sum, 2);
    assert_eq!(pair_algebra.union_generator_rank, 1);
    assert_eq!(pair_algebra.kernel_dimension, 1);
    assert_eq!(pair_algebra.raw_factor_tuple_count.exponent(), 2);
    assert_eq!(pair_algebra.reachable_target_count.exponent(), 1);
    assert_eq!(pair_algebra.factorization_count_per_target.exponent(), 1);
    assert!(!pair_algebra.unique_factorization);
    assert_eq!(pair_algebra.dependency_order, Some(2));

    let pair_targets = [shared.encode(&[false]), shared.encode(&[true])];
    for target in pair_targets {
        assert_eq!(
            factorization_count_for_target(&target, &pair)
                .expect("pair target is representable")
                .exponent(),
            1,
        );
    }
    let mut pair_fibers: Vec<(BinaryCodeword, usize)> = Vec::new();
    for left in shared.enumerate() {
        for right in shared.enumerate() {
            let target = left.bound(&right);
            if let Some((_, count)) = pair_fibers
                .iter_mut()
                .find(|(candidate, _)| *candidate == target)
            {
                *count += 1;
            } else {
                pair_fibers.push((target, 1));
            }
        }
    }
    assert_eq!(pair_fibers.len(), 2);
    assert!(pair_fibers.iter().all(|(_, count)| *count == 2));

    // Three one-dimensional subcodes: every pair intersects trivially, but
    // e1 + e2 + (e1 + e2) = 0 creates a genuine 3-way dependency.
    let c1 = shared.clone();
    let c2 = orthogonal.clone();
    let c3 = RandomLinearCode::from_basis(vec![BinaryCodeword::from_words(3, vec![0b011])])
        .expect("third code");

    for factors in [[&c1, &c2], [&c1, &c3], [&c2, &c3]] {
        let algebra = factorization_algebra(&factors).expect("pairwise algebra");
        assert_eq!(algebra.kernel_dimension, 0);
        assert!(algebra.unique_factorization);
        assert_eq!(algebra.dependency_order, None);
    }

    let triple = vec![&c1, &c2, &c3];
    let triple_algebra = factorization_algebra(&triple).expect("triple algebra");
    assert_eq!(triple_algebra.factor_dimension_sum, 3);
    assert_eq!(triple_algebra.union_generator_rank, 2);
    assert_eq!(triple_algebra.kernel_dimension, 1);
    assert_eq!(triple_algebra.raw_factor_tuple_count.exponent(), 3);
    assert_eq!(triple_algebra.reachable_target_count.exponent(), 2);
    assert_eq!(triple_algebra.factorization_count_per_target.exponent(), 1);
    assert!(!triple_algebra.unique_factorization);
    assert_eq!(triple_algebra.dependency_order, Some(3));

    let mut triple_fibers: Vec<(BinaryCodeword, usize)> = Vec::new();
    for a in c1.enumerate() {
        for b in c2.enumerate() {
            for d in c3.enumerate() {
                let target = a.bound(&b).bound(&d);
                if let Some((_, count)) = triple_fibers
                    .iter_mut()
                    .find(|(candidate, _)| *candidate == target)
                {
                    *count += 1;
                } else {
                    triple_fibers.push((target, 1));
                }
            }
        }
    }
    assert_eq!(triple_fibers.len(), 4);
    assert!(triple_fibers.iter().all(|(_, count)| *count == 2));
    for (target, count) in &triple_fibers {
        assert_eq!(
            factorization_count_for_target(target, &triple)
                .expect("triple target is representable")
                .exponent(),
            1,
        );
        assert_eq!(*count, 1usize << triple_algebra.kernel_dimension);
    }

    let outsider = BinaryCodeword::from_words(3, vec![0b100]);
    assert!(factorization_count_for_target(&outsider, &triple).is_none());

    println!(
        "ALGEBRAIC_LEDGER=pair_delta={};pair_rank={};pair_kernel={};pair_raw=2^{};pair_reachable=2^{};pair_fiber=2^{};pair_unique={};pair_dependency_order={:?};triple_delta={};triple_rank={};triple_kernel={};triple_raw=2^{};triple_reachable=2^{};triple_fiber=2^{};triple_unique={};triple_dependency_order={:?};pair_distinct_targets={};triple_distinct_targets={};pair_fibers_exact=true;triple_fibers_exact=true",
        pair_algebra.factor_dimension_sum,
        pair_algebra.union_generator_rank,
        pair_algebra.kernel_dimension,
        pair_algebra.raw_factor_tuple_count.exponent(),
        pair_algebra.reachable_target_count.exponent(),
        pair_algebra.factorization_count_per_target.exponent(),
        pair_algebra.unique_factorization,
        pair_algebra.dependency_order,
        triple_algebra.factor_dimension_sum,
        triple_algebra.union_generator_rank,
        triple_algebra.kernel_dimension,
        triple_algebra.raw_factor_tuple_count.exponent(),
        triple_algebra.reachable_target_count.exponent(),
        triple_algebra.factorization_count_per_target.exponent(),
        triple_algebra.unique_factorization,
        triple_algebra.dependency_order,
        pair_fibers.len(),
        triple_fibers.len(),
    );
}

#[test]
fn higher_order_dependency_stress_has_pairwise_trivial_cases() {
    let dimension = 8usize;
    let rank = 3usize;
    let realizations = 20_000usize;
    let mut pairwise_trivial = 0usize;
    let mut three_way_dependencies = 0usize;

    for realization in 0..realizations {
        let seed = 0x8A00_0000u64 + realization as u64;
        let c1 = RandomLinearCode::generate(dimension, rank, seed);
        let c2 = RandomLinearCode::generate(dimension, rank, seed.wrapping_add(1));
        let c3 = RandomLinearCode::generate(dimension, rank, seed.wrapping_add(2));
        let pairwise = [[&c1, &c2], [&c1, &c3], [&c2, &c3]];
        let pairwise_trivial_here = pairwise.iter().all(|pair| {
            factorization_algebra(pair)
                .expect("pair algebra")
                .kernel_dimension
                == 0
        });

        if pairwise_trivial_here {
            pairwise_trivial += 1;
            let triple = factorization_algebra(&[&c1, &c2, &c3]).expect("triple algebra");
            assert!(triple.kernel_dimension > 0);
            assert_eq!(triple.dependency_order, Some(3));
            three_way_dependencies += 1;
        }
    }

    assert!(pairwise_trivial > 0);
    assert_eq!(three_way_dependencies, pairwise_trivial);
    println!(
        "HIGHER_ORDER_SWEEP=dimension={dimension};rank={rank};realizations={realizations};pairwise_trivial={pairwise_trivial};three_way_dependencies={three_way_dependencies}",
    );
}
#[test]
fn paper_scale_binding_recovery_smoke_matrix_is_valid() {
    let dimensions = [500usize, 1000, 2000];
    let ranks = [3usize, 5, 7];
    let factor_counts = [3usize, 4, 5];
    let repeats = 10usize;

    let mut cases = 0usize;
    let mut exact_original = 0usize;
    let mut valid_representative = 0usize;
    let mut jointly_dependent = 0usize;
    let mut failures = 0usize;
    let mut total_work = LinearCodeWork::default();
    let mut result_digest = Hasher::new();
    result_digest.update(b"symthaea-hdc-paper-matrix-v3\\0");
    let mut unique_cases = 0usize;
    let mut non_unique_valid = 0usize;
    let mut nonexistent_targets = 0usize;
    let mut max_kernel_dimension = 0usize;
    let mut max_dependency_order = 0usize;
    let mut witnessed_dependency_cases = 0usize;

    for &dimension in &dimensions {
        for &rank in &ranks {
            for &factor_count in &factor_counts {
                for repeat in 0..repeats {
                    let seed_base = 0xB100_0000u64
                        ^ ((dimension as u64) << 16)
                        ^ ((rank as u64) << 8)
                        ^ factor_count as u64
                        ^ ((repeat as u64) << 32);

                    let codes: Vec<_> = (0..factor_count)
                        .map(|factor_index| {
                            RandomLinearCode::generate(
                                dimension,
                                rank,
                                seed_base + factor_index as u64,
                            )
                        })
                        .collect();
                    let factors: Vec<&RandomLinearCode> = codes.iter().collect();

                    let words: Vec<_> = factors
                        .iter()
                        .enumerate()
                        .map(|(factor_index, factor)| {
                            let message: Vec<bool> = (0..rank)
                                .map(|bit| (bit + factor_index + repeat) % 3 == 0)
                                .collect();
                            factor.encode(&message)
                        })
                        .collect();

                    let target = words
                        .iter()
                        .cloned()
                        .reduce(|left, right| left.bound(&right))
                        .expect("at least one factor");

                    result_digest.update(&(dimension as u64).to_le_bytes());
                    result_digest.update(&(rank as u64).to_le_bytes());
                    result_digest.update(&(factor_count as u64).to_le_bytes());
                    result_digest.update(&(repeat as u64).to_le_bytes());
                    result_digest.update(&seed_base.to_le_bytes());
                    result_digest.update(&(target.words().len() as u64).to_le_bytes());
                    for word in target.words() {
                        result_digest.update(&word.to_le_bytes());
                    }
                    for word in &words {
                        result_digest.update(&(word.words().len() as u64).to_le_bytes());
                        for packed_word in word.words() {
                            result_digest.update(&packed_word.to_le_bytes());
                        }
                    }

                    let mut combined_basis = Vec::with_capacity(rank * factor_count);
                    for factor in &factors {
                        combined_basis.extend(factor.basis().iter().cloned());
                    }

                    let algebra = factorization_algebra(&factors).expect("valid paper fixture");
                    assert_eq!(algebra.factor_dimension_sum, rank * factor_count);
                    assert_eq!(
                        algebra.union_generator_rank,
                        basis_rank(&combined_basis, dimension)
                    );
                    assert_eq!(
                        algebra.factor_dimension_sum,
                        algebra.union_generator_rank + algebra.kernel_dimension
                    );
                    assert_eq!(
                        algebra.factorization_count_per_target.exponent(),
                        algebra.kernel_dimension
                    );
                    assert_eq!(algebra.unique_factorization, algebra.kernel_dimension == 0);
                    max_kernel_dimension = max_kernel_dimension.max(algebra.kernel_dimension);
                    if let Some(order) = algebra.dependency_order {
                        max_dependency_order = max_dependency_order.max(order);
                    }

                    result_digest.update(&(algebra.factor_dimension_sum as u64).to_le_bytes());
                    result_digest.update(&(algebra.union_generator_rank as u64).to_le_bytes());
                    result_digest.update(&(algebra.kernel_dimension as u64).to_le_bytes());
                    result_digest
                        .update(&(algebra.raw_factor_tuple_count.exponent() as u64).to_le_bytes());
                    result_digest
                        .update(&(algebra.reachable_target_count.exponent() as u64).to_le_bytes());
                    result_digest.update(
                        &(algebra.factorization_count_per_target.exponent() as u64).to_le_bytes(),
                    );
                    result_digest.update(&[algebra.unique_factorization as u8]);
                    result_digest
                        .update(&(algebra.dependency_order.unwrap_or(0) as u64).to_le_bytes());

                    let target_multiplicity = factorization_count_for_target(&target, &factors);
                    let Some(target_multiplicity) = target_multiplicity else {
                        nonexistent_targets += 1;
                        continue;
                    };
                    assert_eq!(
                        target_multiplicity.exponent(),
                        algebra.kernel_dimension,
                        "every representable target must have the algebraic fiber cardinality"
                    );

                    let affine_fiber = factorization_affine_fiber(&target, &factors)
                        .expect("representable target must expose an affine fiber");
                    assert_eq!(
                        affine_fiber.cardinality.exponent(),
                        algebra.kernel_dimension
                    );
                    assert_eq!(affine_fiber.kernel_basis.len(), algebra.kernel_dimension);
                    assert_eq!(
                        affine_fiber.representative_coefficients.len(),
                        algebra.factor_dimension_sum
                    );
                    assert!(affine_fiber.verifies_against(&target, &factors));
                    result_digest.update(b"affine-fiber-certificate");
                    result_digest.update(
                        &(affine_fiber.representative_coefficients.len() as u64).to_le_bytes(),
                    );
                    for coefficient in &affine_fiber.representative_coefficients {
                        result_digest.update(&[*coefficient as u8]);
                    }
                    result_digest.update(&affine_fiber.fingerprint());

                    let jointly_independent = algebra.unique_factorization;
                    let kernel_basis = if jointly_independent {
                        Vec::new()
                    } else {
                        let kernel_basis = factorization_kernel_basis(&factors)
                            .expect("dependent factors must expose a complete kernel basis");
                        assert_eq!(kernel_basis.len(), algebra.kernel_dimension);
                        let mut dependent_indices = kernel_basis
                            .iter()
                            .map(|witness| witness.dependent_generator_index)
                            .collect::<Vec<_>>();
                        dependent_indices.sort_unstable();
                        dependent_indices.dedup();
                        assert_eq!(dependent_indices.len(), kernel_basis.len());
                        for witness in &kernel_basis {
                            assert!(witness.verifies_against(&factors));
                            assert!(
                                witness.factor_support_size()
                                    >= algebra.dependency_order.unwrap_or(2)
                            );
                        }
                        kernel_basis
                    };

                    if kernel_basis.is_empty() {
                        result_digest.update(b"no-dependency-kernel");
                    } else {
                        witnessed_dependency_cases += 1;
                        result_digest.update(b"dependency-kernel-basis");
                        result_digest.update(&(kernel_basis.len() as u64).to_le_bytes());
                        for witness in &kernel_basis {
                            result_digest.update(
                                &(witness.generator_coefficients.len() as u64).to_le_bytes(),
                            );
                            for coefficient in &witness.generator_coefficients {
                                result_digest.update(&[*coefficient as u8]);
                            }
                            result_digest
                                .update(&(witness.factor_support.len() as u64).to_le_bytes());
                            for factor_index in &witness.factor_support {
                                result_digest.update(&(*factor_index as u64).to_le_bytes());
                            }
                            result_digest
                                .update(&(witness.dependent_generator_index as u64).to_le_bytes());
                        }
                    }

                    if !jointly_independent {
                        jointly_dependent += 1;
                    }
                    let (recovered, work) = recover_linear_bound_with_work(&target, &factors);
                    total_work.span_membership_checks += work.span_membership_checks;
                    total_work.basis_rank_pivots += work.basis_rank_pivots;
                    total_work.basis_rank_row_xor_words += work.basis_rank_row_xor_words;
                    total_work.basis_rank_input_word_copies += work.basis_rank_input_word_copies;
                    total_work.solve_basis_bit_probes += work.solve_basis_bit_probes;
                    total_work.solve_matrix_word_cells += work.solve_matrix_word_cells;
                    total_work.solve_pivots += work.solve_pivots;
                    total_work.solve_row_xor_words += work.solve_row_xor_words;
                    total_work.retained_generators += work.retained_generators;
                    total_work.projection_word_xor_ops += work.projection_word_xor_ops;

                    cases += 1;
                    let Some(recovered) = recovered else {
                        failures += 1;
                        continue;
                    };

                    result_digest.update(b"recovered");
                    result_digest.update(&(recovered.len() as u64).to_le_bytes());
                    for word in &recovered {
                        result_digest.update(&(word.words().len() as u64).to_le_bytes());
                        for packed_word in word.words() {
                            result_digest.update(&packed_word.to_le_bytes());
                        }
                    }

                    let valid = recovered.len() == factor_count
                        && recovered
                            .iter()
                            .zip(&factors)
                            .all(|(word, factor)| factor.contains(word))
                        && recovered
                            .iter()
                            .skip(1)
                            .fold(recovered[0].clone(), |bound, word| bound.bound(word))
                            == target;
                    assert!(
                        valid,
                        "recovered factors must belong to their factor codes and rebind to target"
                    );
                    valid_representative += 1;

                    if jointly_independent {
                        assert_eq!(
                            recovered, words,
                            "independent paper-style factors have unique exact recovery"
                        );
                        exact_original += 1;
                        unique_cases += 1;
                    } else {
                        non_unique_valid += 1;
                    }
                }
            }
        }
    }

    assert_eq!(
        cases,
        dimensions.len() * ranks.len() * factor_counts.len() * repeats
    );
    assert_eq!(cases, 270);
    assert_eq!(failures, 0);
    assert_eq!(valid_representative, cases);
    assert_eq!(unique_cases + non_unique_valid, cases);
    assert_eq!(nonexistent_targets, 0);
    assert_eq!(witnessed_dependency_cases, jointly_dependent);

    let result_digest = result_digest.finalize();
    let result_digest = result_digest
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    println!(
        "PAPER_MATRIX=dimensions=500,1000,2000;ranks=3,5,7;factors=3,4,5;repeats={repeats};cases={cases};exact_original={exact_original};unique_cases={unique_cases};non_unique_valid={non_unique_valid};valid_representative={valid_representative};jointly_dependent={jointly_dependent};nonexistent_targets={nonexistent_targets};max_kernel_dimension={max_kernel_dimension};max_dependency_order={max_dependency_order};failures={failures};witnessed_dependency_cases={witnessed_dependency_cases};result_digest={result_digest};total_span_membership_checks={};total_basis_rank_pivots={};total_solve_pivots={};total_solve_row_xor_words={}",
        total_work.span_membership_checks,
        total_work.basis_rank_pivots,
        total_work.solve_pivots,
        total_work.solve_row_xor_words,
    );
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
fn exhaustive_general_recovery_matches_product_code_truth() {
    let (parent, left, right) =
        RandomLinearCode::generate_direct_sum(10, 2, 2, 0x5A12).expect("valid direct sum");

    for left_word in left.enumerate() {
        for right_word in right.enumerate() {
            let target = left_word.bound(&right_word);
            let recovered =
                recover_linear_bound(&target, &[&left, &right]).expect("product recovery");
            assert_eq!(recovered, vec![left_word.clone(), right_word.clone()]);
            assert_eq!(recovered[0].bound(&recovered[1]), target);
        }
    }

    assert_eq!(parent.rank(), left.rank() + right.rank());
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
fn exhaustive_overlapping_recovery_existence_matches_truth_oracle() {
    let parent = RandomLinearCode::generate(10, 5, 0x5A13);
    let left = RandomLinearCode::from_basis(parent.basis()[..3].to_vec()).expect("left subcode");
    let right = RandomLinearCode::from_basis(parent.basis()[2..5].to_vec()).expect("right subcode");

    let left_words = left.enumerate();
    let right_words = right.enumerate();

    for mask in 0..(1usize << 10) {
        let target = BinaryCodeword::from_words(10, vec![mask as u64]);
        let oracle_exists = left_words
            .iter()
            .any(|a| right_words.iter().any(|b| a.bound(b) == target));
        let recovered = recover_linear_bound(&target, &[&left, &right]);

        assert_eq!(
            recovered.is_some(),
            oracle_exists,
            "recovery existence mismatch for target mask={mask:#x}"
        );

        if let Some(factors) = recovered {
            assert_eq!(factors.len(), 2);
            assert!(left.contains(&factors[0]));
            assert!(right.contains(&factors[1]));
            assert_eq!(factors[0].bound(&factors[1]), target);
        }
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

#[test]
fn exact_factorization_neighborhood_identity_holds_under_hamming_noise() {
    // For a linear factor-to-bound map, every reachable target has exactly
    // 2^d preimages. Therefore a Hamming-ball oracle over factor tuples must
    // equal the number of reachable targets in that ball multiplied by 2^d.
    // This cleanly separates noise-induced target ambiguity from intrinsic
    // affine-fiber multiplicity.
    let repeated = RandomLinearCode::from_basis(vec![
        BinaryCodeword::from_words(4, vec![0b0001]),
        BinaryCodeword::from_words(4, vec![0b0010]),
    ])
    .expect("repeated 2D code");
    let factors = [&repeated, &repeated, &repeated];
    let algebra = factorization_algebra(&factors).expect("algebra");
    assert_eq!(algebra.factor_dimension_sum, 6);
    assert_eq!(algebra.union_generator_rank, 2);
    assert_eq!(algebra.kernel_dimension, 4);

    let combined_basis = factors
        .iter()
        .flat_map(|factor| factor.basis().iter())
        .collect::<Vec<_>>();
    let all_factorization_targets = (0..(1usize << algebra.factor_dimension_sum))
        .map(|mask| {
            let mut target = BinaryCodeword::zero(4);
            for (index, generator) in combined_basis.iter().enumerate() {
                if (mask >> index) & 1 == 1 {
                    target.xor_assign(generator);
                }
            }
            target
        })
        .collect::<Vec<_>>();

    let reachable_targets = repeated.enumerate();
    assert_eq!(
        all_factorization_targets.len(),
        1usize << algebra.factor_dimension_sum
    );
    assert_eq!(
        reachable_targets.len(),
        1usize << algebra.union_generator_rank
    );

    let hamming_distance = |left: &BinaryCodeword, right: &BinaryCodeword| {
        left.words()
            .iter()
            .zip(right.words())
            .map(|(a, b)| (a ^ b).count_ones() as usize)
            .sum::<usize>()
    };

    for observed_mask in 0..(1usize << 4) {
        let observed = BinaryCodeword::from_words(4, vec![observed_mask as u64]);

        for radius in 0..=2 {
            let factor_tuple_neighbors = all_factorization_targets
                .iter()
                .filter(|target| hamming_distance(target, &observed) <= radius)
                .count();
            let reachable_target_neighbors = reachable_targets
                .iter()
                .filter(|target| hamming_distance(target, &observed) <= radius)
                .count();

            assert_eq!(
                factor_tuple_neighbors,
                reachable_target_neighbors * (1usize << algebra.kernel_dimension),
                "factorization neighborhood mismatch for observed={observed_mask:#x}, radius={radius}"
            );
        }
    }
}

#[test]
fn bounded_affine_fiber_iterator_exhausts_declared_multiplicity() {
    let repeated = RandomLinearCode::from_basis(vec![
        BinaryCodeword::from_words(4, vec![0b0001]),
        BinaryCodeword::from_words(4, vec![0b0010]),
    ])
    .expect("repeated 2D code");
    let factors = [&repeated, &repeated, &repeated];
    let target = repeated.encode(&[true, false]);

    let algebra = factorization_algebra(&factors).expect("algebra");
    assert_eq!(algebra.kernel_dimension, 4);
    let fiber = factorization_affine_fiber(&target, &factors).expect("affine fiber");
    let expected = 1usize << algebra.kernel_dimension;

    let mut iterator = fiber.iter_bounded(expected).expect("bounded enumeration");
    assert_eq!(iterator.len(), expected);
    let coefficients = iterator.by_ref().collect::<Vec<_>>();
    assert_eq!(iterator.len(), 0);
    assert_eq!(coefficients.len(), expected);

    for coefficients in &coefficients {
        let mut reconstructed = BinaryCodeword::zero(4);
        for (coefficient, generator) in coefficients
            .iter()
            .zip(factors.iter().flat_map(|factor| factor.basis().iter()))
        {
            if *coefficient {
                reconstructed.xor_assign(generator);
            }
        }
        assert_eq!(reconstructed, target);
    }

    for left in 0..coefficients.len() {
        for right in (left + 1)..coefficients.len() {
            assert_ne!(coefficients[left], coefficients[right]);
        }
    }

    assert!(fiber.iter_bounded(expected - 1).is_none());
    assert_eq!(fiber.iter_bounded(0), None);

    let mut tampered = fiber.clone();
    tampered.cardinality = ExactPowerOfTwo::new(algebra.kernel_dimension - 1);
    assert!(tampered.iter_bounded(expected).is_none());
    assert!(
        tampered
            .coefficients_for_mask(&[false, false, false, false])
            .is_none()
    );

    let mut dependent_kernel = fiber.clone();
    dependent_kernel.kernel_basis[1].generator_coefficients = dependent_kernel.kernel_basis[0]
        .generator_coefficients
        .clone();
    assert!(dependent_kernel.iter_bounded(expected).is_none());
    assert!(
        dependent_kernel
            .coefficients_for_mask(&[false, false, false, false])
            .is_none()
    );

    let mut semantically_invalid = fiber.clone();
    semantically_invalid.kernel_basis[0].generator_coefficients[1] ^= true;
    assert!(
        semantically_invalid.kernel_basis[0].generator_coefficients
            [semantically_invalid.kernel_basis[0].dependent_generator_index]
    );
    assert!(!semantically_invalid.kernel_basis[0].verifies_against(&factors));

    let invalid_coefficient_basis = semantically_invalid
        .kernel_basis
        .iter()
        .map(|witness| {
            let mut vector =
                BinaryCodeword::zero(semantically_invalid.representative_coefficients.len());
            for (index, coefficient) in witness.generator_coefficients.iter().enumerate() {
                vector.set_bit(index, *coefficient);
            }
            vector
        })
        .collect::<Vec<_>>();
    assert_eq!(
        basis_rank(
            &invalid_coefficient_basis,
            semantically_invalid.representative_coefficients.len()
        ),
        algebra.kernel_dimension
    );
    assert_ne!(
        semantically_invalid.fingerprint(),
        fiber.fingerprint(),
        "tampering must alter the recomputed certificate fingerprint"
    );
    assert!(semantically_invalid.iter_bounded(expected).is_none());
    assert!(
        semantically_invalid
            .coefficients_for_mask(&[true, false, false, false])
            .is_none()
    );
    assert!(!semantically_invalid.verifies_against(&target, &factors));

    assert!(
        fiber
            .coefficients_for_mask(&[false, false, false])
            .is_none()
    );

    println!(
        "AFFINE_FIBER_ENUMERATION=kernel_dimension={};expected_fibers={};enumerated_fibers={};all_targets_match=true;all_coefficients_unique=true;bounded=true",
        algebra.kernel_dimension,
        expected,
        coefficients.len(),
    );
}
