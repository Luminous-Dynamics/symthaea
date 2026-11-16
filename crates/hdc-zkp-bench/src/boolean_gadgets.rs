// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared, semantically explicit bit-level gadgets for public Binius demonstrations.
//!
//! Majority uses bit-sliced counters so each of the 64 bit lanes is counted
//! independently. Ties resolve to zero: an output bit is one only when strictly
//! more than half the inputs have that bit set.

use binius_core::{verify::verify_constraints, word::Word};
use binius_frontend::{CircuitBuilder, Wire};

fn add64(builder: &CircuitBuilder, a: Wire, b: Wire, zero: Wire) -> Wire {
    builder.iadd_cin_cout(a, b, zero).0
}

/// Return the constrained population count (0..=64) of a 64-bit word.
///
/// This uses a standard SWAR network. The returned count is encoded in the low
/// seven bits; the function does not confuse the numeric word value with a count.
pub fn popcount_u64(builder: &CircuitBuilder, value: Wire) -> Wire {
    let zero = builder.add_constant_64(0);
    let m1 = builder.add_constant_64(0x5555_5555_5555_5555);
    let m2 = builder.add_constant_64(0x3333_3333_3333_3333);
    let m4 = builder.add_constant_64(0x0f0f_0f0f_0f0f_0f0f);
    let m7 = builder.add_constant_64(0x7f);

    let shifted_1 = builder.shr(value, 1);
    let subtractor = builder.band(shifted_1, m1);
    let x1 = builder.isub_bin_bout(value, subtractor, zero).0;

    let low_pairs = builder.band(x1, m2);
    let high_pairs = builder.band(builder.shr(x1, 2), m2);
    let x2 = add64(builder, low_pairs, high_pairs, zero);

    let x3_sum = add64(builder, x2, builder.shr(x2, 4), zero);
    let x3 = builder.band(x3_sum, m4);
    let x4 = add64(builder, x3, builder.shr(x3, 8), zero);
    let x5 = add64(builder, x4, builder.shr(x4, 16), zero);
    let x6 = add64(builder, x5, builder.shr(x5, 32), zero);

    builder.band(x6, m7)
}

/// Return the strict bitwise majority of packed words for any input count >= 2.
///
/// The counter is represented as bit planes: bit k in a plane gives bit k of the
/// count for each of the 64 independent lanes. A comparator then checks whether
/// each lane's count is at least floor(n/2)+1.
pub fn bitwise_majority(builder: &CircuitBuilder, inputs: &[Wire]) -> Wire {
    assert!(inputs.len() >= 2, "majority requires at least two inputs");

    let zero = builder.add_constant_64(0);
    let ones = builder.add_constant_64(u64::MAX);
    let count_width = (usize::BITS - inputs.len().leading_zeros()) as usize;
    let mut count_bits = vec![zero; count_width];

    for &input in inputs {
        let mut carry = input;
        for count_bit in &mut count_bits {
            let previous = *count_bit;
            *count_bit = builder.bxor(previous, carry);
            carry = builder.band(previous, carry);
        }
    }

    let threshold = inputs.len() / 2 + 1;
    let mut greater = zero;
    let mut equal = ones;
    for bit in (0..count_width).rev() {
        let count_bit = count_bits[bit];
        if ((threshold >> bit) & 1) == 0 {
            let greater_here = builder.band(equal, count_bit);
            greater = builder.bor(greater, greater_here);
            equal = builder.band(equal, builder.bnot(count_bit));
        } else {
            equal = builder.band(equal, count_bit);
        }
    }

    builder.bor(greater, equal)
}

/// Host-side reference function used to construct witnesses and test semantics.
pub fn bitwise_majority_reference(inputs: &[u64]) -> u64 {
    assert!(inputs.len() >= 2, "majority requires at least two inputs");
    let threshold = inputs.len() / 2 + 1;
    let mut output = 0u64;
    for bit in 0..64 {
        let set_count = inputs
            .iter()
            .filter(|word| ((**word >> bit) & 1) != 0)
            .count();
        if set_count >= threshold {
            output |= 1u64 << bit;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn relation_holds_popcount(value: u64, claimed_count: u64) -> bool {
        let builder = CircuitBuilder::new();
        let input = builder.add_witness();
        let output = builder.add_inout();
        let actual = popcount_u64(&builder, input);
        builder.assert_eq("popcount", actual, output);

        let circuit = builder.build();
        let mut witness = circuit.new_witness_filler();
        witness[input] = Word(value);
        witness[output] = Word(claimed_count);
        if circuit.populate_wire_witness(&mut witness).is_err() {
            return false;
        }
        let cs = circuit.constraint_system();
        verify_constraints(cs, &witness.into_value_vec()).is_ok()
    }

    fn relation_holds_majority(inputs: &[u64], claimed_output: u64) -> bool {
        let builder = CircuitBuilder::new();
        let input_wires: Vec<_> = inputs.iter().map(|_| builder.add_witness()).collect();
        let output = builder.add_inout();
        let actual = bitwise_majority(&builder, &input_wires);
        builder.assert_eq("bitwise_majority", actual, output);

        let circuit = builder.build();
        let mut witness = circuit.new_witness_filler();
        for (wire, value) in input_wires.iter().zip(inputs) {
            witness[*wire] = Word(*value);
        }
        witness[output] = Word(claimed_output);
        if circuit.populate_wire_witness(&mut witness).is_err() {
            return false;
        }
        let cs = circuit.constraint_system();
        verify_constraints(cs, &witness.into_value_vec()).is_ok()
    }

    #[test]
    fn popcount_matches_reference_on_boundary_and_pattern_values() {
        let values = [
            0,
            1,
            2,
            3,
            u64::MAX,
            1u64 << 63,
            0xaaaa_aaaa_aaaa_aaaa,
            0x5555_5555_5555_5555,
            0x3333_3333_3333_3333,
            0x0123_4567_89ab_cdef,
        ];
        for value in values {
            assert!(
                relation_holds_popcount(value, value.count_ones() as u64),
                "popcount relation failed for {value:#018x}"
            );
        }
    }

    #[test]
    fn popcount_handles_every_single_bit_position() {
        for bit in 0..64 {
            let value = 1u64 << bit;
            assert!(
                relation_holds_popcount(value, 1),
                "single-bit population count failed at bit {bit}"
            );
        }
    }
    #[test]
    fn popcount_relation_rejects_a_wrong_claim() {
        assert!(!relation_holds_popcount(u64::MAX, 0));
    }

    #[test]
    fn bitwise_majority_matches_reference_for_multiple_party_counts() {
        for n in [2usize, 3, 4, 5, 8, 16] {
            for trial in 0..6usize {
                let inputs: Vec<u64> = (0..n)
                    .map(|i| {
                        let shift = ((trial * 9 + i * 7) % 64) as u32;
                        0x9e37_79b9_7f4a_7c15u64
                            .wrapping_mul((i + 1) as u64)
                            .rotate_left(shift)
                    })
                    .collect();
                let expected = bitwise_majority_reference(&inputs);
                assert!(
                    relation_holds_majority(&inputs, expected),
                    "majority relation failed for {n} inputs (trial {trial})"
                );
            }
        }
    }

    #[test]
    fn strict_majority_threshold_boundaries_and_even_ties() {
        for n in [2usize, 3, 4, 5, 8, 16] {
            let threshold = n / 2 + 1;

            let mut below_threshold = vec![u64::MAX; threshold - 1];
            below_threshold.resize(n, 0);
            assert_eq!(bitwise_majority_reference(&below_threshold), 0);
            assert!(relation_holds_majority(&below_threshold, 0));

            let mut at_threshold = vec![u64::MAX; threshold];
            at_threshold.resize(n, 0);
            assert_eq!(bitwise_majority_reference(&at_threshold), u64::MAX);
            assert!(relation_holds_majority(&at_threshold, u64::MAX));

            if n % 2 == 0 {
                let mut tied = vec![u64::MAX; n / 2];
                tied.resize(n, 0);
                assert_eq!(bitwise_majority_reference(&tied), 0);
                assert!(relation_holds_majority(&tied, 0));
            }
        }
    }
    #[test]
    fn four_input_majority_is_not_all_input_and() {
        let inputs = [u64::MAX, u64::MAX, u64::MAX, 0];
        assert_eq!(bitwise_majority_reference(&inputs), u64::MAX);
        assert!(relation_holds_majority(
            &inputs,
            bitwise_majority_reference(&inputs)
        ));
        assert!(!relation_holds_majority(&inputs, 0));
    }
}
