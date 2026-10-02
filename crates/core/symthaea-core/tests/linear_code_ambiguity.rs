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
