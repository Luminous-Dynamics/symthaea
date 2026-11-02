use symthaea_complexity_lab::profile::{ProfileError, WeightProfile};

fn next_u64(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

fn sample_profile(seed: u64, bounds: &[u32]) -> WeightProfile {
    let mut state = seed.max(1);
    let mut profile = WeightProfile::zero(bounds.to_vec()).unwrap();
    for _ in 0..12 {
        let mut exponent = Vec::with_capacity(bounds.len());
        for &bound in bounds {
            exponent.push((next_u64(&mut state) % u64::from(bound)) as u32);
        }
        let coefficient = (next_u64(&mut state) % 7) as i128 - 3;
        let term = WeightProfile::monomial(bounds.to_vec(), exponent, coefficient).unwrap();
        profile = profile.checked_add(&term).unwrap();
    }
    profile
}

#[test]
fn truncated_integer_profile_ring_laws_hold_on_deterministic_cases() {
    let bounds = [4, 3];
    let zero = WeightProfile::zero(bounds.to_vec()).unwrap();
    let one = WeightProfile::one(bounds.to_vec()).unwrap();

    for seed in 1..=128 {
        let a = sample_profile(seed, &bounds);
        let b = sample_profile(seed * 17 + 3, &bounds);
        let c = sample_profile(seed * 101 + 11, &bounds);

        assert_eq!(a.checked_add(&zero).unwrap(), a);
        assert_eq!(a.checked_mul(&one).unwrap(), a);
        assert!(a.checked_mul(&zero).unwrap().is_zero());

        assert_eq!(a.checked_add(&b).unwrap(), b.checked_add(&a).unwrap());
        assert_eq!(a.checked_mul(&b).unwrap(), b.checked_mul(&a).unwrap());

        let left_assoc = a.checked_mul(&b).unwrap().checked_mul(&c).unwrap();
        let right_assoc = a.checked_mul(&b.checked_mul(&c).unwrap()).unwrap();
        assert_eq!(left_assoc, right_assoc);

        let left_distrib = a.checked_mul(&b.checked_add(&c).unwrap()).unwrap();
        let right_distrib = a.checked_mul(&b).unwrap().checked_add(&a.checked_mul(&c).unwrap()).unwrap();
        assert_eq!(left_distrib, right_distrib);

        let cancelled = a.checked_add(&a.checked_scale(-1).unwrap()).unwrap();
        assert!(cancelled.is_zero());
    }
}

#[test]
fn signed_leading_term_cancellation_defeats_naive_max_plus() {
    // max-degree(W) = max-degree(1-W) = 1, but W + (1-W) = 1 has degree 0.
    let w = WeightProfile::monomial(vec![3], vec![1], 1).unwrap();
    let one_minus_w = WeightProfile::one(vec![3]).unwrap()
        .checked_add(&w.checked_scale(-1).unwrap()).unwrap();
    let sum = w.checked_add(&one_minus_w).unwrap();

    assert_eq!(w.maximum_exponent(0), Some(1));
    assert_eq!(one_minus_w.maximum_exponent(0), Some(1));
    assert_eq!(sum.maximum_exponent(0), Some(0));
    assert_eq!(sum.coefficient(&[0]).unwrap(), 1);
    assert_eq!(sum.coefficient(&[1]).unwrap(), 0);
}

#[test]
fn exponent_and_coefficient_overflow_remain_distinct_failures() {
    let high = WeightProfile::monomial(vec![u32::MAX], vec![u32::MAX - 1], 2).unwrap();
    let next = WeightProfile::monomial(vec![u32::MAX], vec![1], 3).unwrap();
    assert!(matches!(high.checked_mul(&next), Err(ProfileError::ExponentOverflow { .. })));

    let max = WeightProfile::monomial(vec![1], vec![0], i128::MAX).unwrap();
    let two = WeightProfile::monomial(vec![1], vec![0], 2).unwrap();
    assert!(matches!(max.checked_add(&two), Err(ProfileError::CoefficientOverflow { .. })));
}
