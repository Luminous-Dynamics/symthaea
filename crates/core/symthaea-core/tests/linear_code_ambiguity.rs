use symthaea_core::hdc::linear_code::{basis_rank, recover_direct_sum_bound, solve_linear_combination, BinaryCodeword, RandomLinearCode};

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
    assert_eq!(
        symthaea_core::hdc::linear_code::basis_rank(&combined_basis, 96),
        parent.rank()
    );

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
    let recovered = solve_linear_combination(&composite, &basis).expect("clean bound must be in span");
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
            right.enumerate().into_iter().filter_map(move |candidate_b| {
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
            right.enumerate().into_iter().filter_map(move |candidate_b| {
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
