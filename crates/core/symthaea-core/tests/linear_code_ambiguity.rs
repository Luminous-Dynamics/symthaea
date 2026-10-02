use symthaea_core::hdc::linear_code::{BinaryCodeword, RandomLinearCode};

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
    let left = RandomLinearCode::generate(96, 6, 0x1111);
    let right = RandomLinearCode::generate(96, 6, 0x2222);

    // The direct-sum condition is rank(C1 + C2) = rank(C1) + rank(C2).
    // We deliberately validate this algebraically instead of assuming that
    // independently generated random subspaces are disjoint.
    let mut combined_basis = left.basis().to_vec();
    combined_basis.extend(right.basis().iter().cloned());
    assert_eq!(
        symthaea_core::hdc::linear_code::basis_rank(&combined_basis, 96),
        left.rank() + right.rank()
    );

    let a = left.encode(&[true, false, true, false, true, false]);
    let b = right.encode(&[false, true, true, false, false, true]);
    let composite = a.bound(&b);

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
}
