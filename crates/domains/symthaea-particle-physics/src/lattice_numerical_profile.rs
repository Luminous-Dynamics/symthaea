// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic keyed binary64 reductions for lattice-QCD evidence.
//!
//! This is the production counterpart to the independently executed LQCD-021L
//! oracle. Scientific leaves carry unique canonical keys. The reducer sorts by
//! key and then applies one fixed adjacent-pair tree, carrying an odd leaf
//! unchanged to the next level.
//!
//! Shards may be joined in arbitrary order only if they preserve the original
//! keyed leaves. Arbitrary shard-local floating partial sums are not equivalent
//! leaves and are intentionally outside this contract.

pub const LQCD_CANONICAL_KEYED_PAIRWISE_F64_ID: &str =
    "lqcd_canonical_keyed_pairwise_f64_v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LqcdReductionProfile {
    pub stable_id: &'static str,
    pub scalar: &'static str,
    pub key_order: &'static str,
    pub tree: &'static str,
    pub shard_contract: &'static str,
}

pub const LQCD_CANONICAL_REDUCTION_PROFILE: LqcdReductionProfile = LqcdReductionProfile {
    stable_id: LQCD_CANONICAL_KEYED_PAIRWISE_F64_ID,
    scalar: "ieee754_binary64",
    key_order: "ascending_unique_key",
    tree: "adjacent_pairwise_left_to_right_carry_odd_leaf",
    shard_contract: "preserve_atomic_keyed_leaves_then_union_before_reduce",
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LqcdReductionError {
    Empty,
    DuplicateKey { sorted_index: usize },
    NonFiniteInput { sorted_index: usize },
    NonFiniteIntermediate { level: usize, pair_start: usize },
    NonFiniteMean,
}

/// Deterministically reduce finite binary64 values after canonical key ordering.
///
/// The key is scientific provenance, not merely an implementation index: callers
/// must choose a key whose ordering is frozen by their own measurement/analysis
/// contract.
pub fn canonical_keyed_sum<K: Ord + Clone>(
    records: &[(K, f64)],
) -> Result<f64, LqcdReductionError> {
    if records.is_empty() {
        return Err(LqcdReductionError::Empty);
    }

    let mut ordered = records.to_vec();
    ordered.sort_by(|left, right| left.0.cmp(&right.0));

    for (index, pair) in ordered.windows(2).enumerate() {
        if pair[0].0 == pair[1].0 {
            return Err(LqcdReductionError::DuplicateKey {
                sorted_index: index,
            });
        }
    }

    let mut level = Vec::with_capacity(ordered.len());
    for (sorted_index, (_, value)) in ordered.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(LqcdReductionError::NonFiniteInput { sorted_index });
        }
        level.push(value);
    }

    let mut level_index = 0usize;
    while level.len() > 1 {
        let mut next = Vec::with_capacity((level.len() + 1) / 2);
        let mut pair_start = 0usize;
        while pair_start < level.len() {
            if pair_start + 1 == level.len() {
                next.push(level[pair_start]);
            } else {
                let value = level[pair_start] + level[pair_start + 1];
                if !value.is_finite() {
                    return Err(LqcdReductionError::NonFiniteIntermediate {
                        level: level_index,
                        pair_start,
                    });
                }
                next.push(value);
            }
            pair_start += 2;
        }
        level = next;
        level_index += 1;
    }

    Ok(level[0])
}

/// Deterministic mean using the same canonical sum followed by one binary64
/// division by the exact record count represented as `f64`.
pub fn canonical_keyed_mean<K: Ord + Clone>(
    records: &[(K, f64)],
) -> Result<f64, LqcdReductionError> {
    let sum = canonical_keyed_sum(records)?;
    let mean = sum / records.len() as f64;
    if !mean.is_finite() {
        return Err(LqcdReductionError::NonFiniteMean);
    }
    Ok(mean)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn baseline() -> Vec<(u64, f64)> {
        vec![
            (40, 3.0),
            (10, 1.0e16),
            (30, -1.0e16),
            (20, 1.0),
            (50, -0.25),
            (60, 0.5),
            (70, 0.125),
        ]
    }

    #[test]
    fn reproduces_independent_oracle_bits() {
        let values = baseline();
        let sum = canonical_keyed_sum(&values).unwrap();
        let mean = canonical_keyed_mean(&values).unwrap();

        assert_eq!(sum.to_bits(), 0x4011_8000_0000_0000);
        assert_eq!(mean.to_bits(), 0x3fe4_0000_0000_0000);

        let odd = vec![
            (4u64, 0.5),
            (1, 0.25),
            (3, 1.0),
            (2, -0.125),
            (5, 0.0625),
        ];
        assert_eq!(
            canonical_keyed_sum(&odd).unwrap().to_bits(),
            0x3ffb_0000_0000_0000
        );
    }

    #[test]
    fn insertion_order_does_not_change_committed_bits() {
        let expected = canonical_keyed_sum(&baseline()).unwrap().to_bits();

        let mut reversed = baseline();
        reversed.reverse();
        assert_eq!(canonical_keyed_sum(&reversed).unwrap().to_bits(), expected);

        let mut rotated = baseline();
        rotated.rotate_left(3);
        assert_eq!(canonical_keyed_sum(&rotated).unwrap().to_bits(), expected);

        let values = baseline();
        let permuted = vec![
            values[3], values[6], values[1], values[4], values[0], values[5], values[2],
        ];
        assert_eq!(canonical_keyed_sum(&permuted).unwrap().to_bits(), expected);
    }

    #[test]
    fn shard_union_order_does_not_change_committed_bits() {
        let values = baseline();
        let expected = canonical_keyed_sum(&values).unwrap().to_bits();
        let shards = [
            vec![values[0], values[4]],
            vec![values[2], values[6], values[1]],
            vec![values[5]],
            vec![values[3]],
        ];

        for order in [
            [0usize, 1, 2, 3],
            [3, 2, 1, 0],
            [1, 3, 0, 2],
            [2, 0, 3, 1],
        ] {
            let mut union = Vec::new();
            for shard_index in order {
                union.extend_from_slice(&shards[shard_index]);
            }
            assert_eq!(canonical_keyed_sum(&union).unwrap().to_bits(), expected);
        }
    }

    #[test]
    fn naive_insertion_order_negative_control_is_detectable() {
        let a = [1.0e16_f64, 1.0, -1.0e16]
            .into_iter()
            .sum::<f64>();
        let b = [1.0e16_f64, -1.0e16, 1.0]
            .into_iter()
            .sum::<f64>();

        assert_eq!(a.to_bits(), 0x0000_0000_0000_0000);
        assert_eq!(b.to_bits(), 0x3ff0_0000_0000_0000);
        assert_ne!(a.to_bits(), b.to_bits());
    }

    #[test]
    fn malformed_reductions_fail_closed() {
        assert_eq!(
            canonical_keyed_sum::<u64>(&[]),
            Err(LqcdReductionError::Empty)
        );
        assert_eq!(
            canonical_keyed_sum(&[(1u64, 1.0), (1, 2.0)]),
            Err(LqcdReductionError::DuplicateKey { sorted_index: 0 })
        );

        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                canonical_keyed_sum(&[(1u64, bad)]),
                Err(LqcdReductionError::NonFiniteInput { sorted_index: 0 })
            );
        }
    }

    #[test]
    fn generic_tuple_keys_have_canonical_lexicographic_order() {
        let records = [
            ((1i32, 0i32, 0i32), 1.0),
            ((-1, 0, 0), 2.0),
            ((0, 1, 0), 4.0),
        ];
        assert_eq!(canonical_keyed_sum(&records).unwrap(), 7.0);
    }

    #[test]
    fn stable_profile_matches_oracle_contract() {
        assert_eq!(
            LQCD_CANONICAL_REDUCTION_PROFILE.stable_id,
            LQCD_CANONICAL_KEYED_PAIRWISE_F64_ID
        );
        assert_eq!(
            LQCD_CANONICAL_REDUCTION_PROFILE.tree,
            "adjacent_pairwise_left_to_right_carry_odd_leaf"
        );
        assert_eq!(
            LQCD_CANONICAL_REDUCTION_PROFILE.shard_contract,
            "preserve_atomic_keyed_leaves_then_union_before_reduce"
        );
    }
}
