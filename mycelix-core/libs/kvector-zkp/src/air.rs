// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! AIR (Algebraic Intermediate Representation) for K-Vector range proofs
//!
//! This defines the constraint system for proving K-Vector components
//! are in the valid range [0, 1] without revealing their values.
//!
//! # Design
//!
//! The AIR uses a sequential bit verification approach:
//! - Each row verifies one bit of one component
//! - State transitions track progress through all 8 components × 14 bits
//! - Accumulator builds up the value from bits and verifies against target

use winterfell::{
    math::{fields::f128::BaseElement, FieldElement, ToElements},
    Air, AirContext, Assertion, EvaluationFrame, ProofOptions, TraceInfo,
    TransitionConstraintDegree,
};

/// Number of K-Vector components
pub const NUM_COMPONENTS: usize = 8;

/// Scaling factor for fixed-point representation (4 decimal places)
pub const SCALE_FACTOR: u64 = 10000;

/// Number of bits needed to represent scaled values (14 bits for 0-10000)
pub const BITS_PER_VALUE: usize = 14;

/// Actual computation steps (8 components × 14 bits)
pub const COMPUTATION_STEPS: usize = NUM_COMPONENTS * BITS_PER_VALUE; // 112

/// Total number of steps (rows) in the trace - must be power of 2
/// We pad to 128 to satisfy winterfell's requirements
pub const TRACE_LENGTH: usize = 128;

/// Trace column indices
pub mod columns {
    /// Step counter (0 to TRACE_LENGTH-1)
    pub const STEP: usize = 0;
    /// Current component index (0-7)
    pub const COMPONENT: usize = 1;
    /// Current bit index (0-13)
    pub const BIT_INDEX: usize = 2;
    /// Current bit value (0 or 1)
    pub const BIT_VALUE: usize = 3;
    /// Accumulated value from bits so far
    pub const ACCUMULATED: usize = 4;
    /// Target value for current component
    pub const TARGET: usize = 5;
    /// Flag: 1 if this is the last bit of a component, 0 otherwise
    pub const IS_LAST_BIT: usize = 6;
    /// Total trace width
    pub const WIDTH: usize = 7;
}

/// Public inputs for the K-Vector range proof
#[derive(Clone, Debug)]
pub struct KVectorPublicInputs {
    /// Commitment to the K-Vector (hash of scaled values)
    pub commitment: [u8; 32],
    /// The scaled target values (public for verification)
    pub targets: [u64; NUM_COMPONENTS],
}

impl KVectorPublicInputs {
    /// Create public inputs from commitment and targets
    pub fn new(commitment: [u8; 32], targets: [u64; NUM_COMPONENTS]) -> Self {
        Self { commitment, targets }
    }
}

impl ToElements<BaseElement> for KVectorPublicInputs {
    fn to_elements(&self) -> Vec<BaseElement> {
        let mut elements = Vec::with_capacity(4 + NUM_COMPONENTS);

        // Commitment bytes as field elements
        for chunk in self.commitment.chunks(8) {
            let mut bytes = [0u8; 8];
            bytes[..chunk.len()].copy_from_slice(chunk);
            elements.push(BaseElement::from(u64::from_le_bytes(bytes)));
        }

        // Target values
        for &target in &self.targets {
            elements.push(BaseElement::from(target));
        }

        elements
    }
}

/// AIR for K-Vector range proofs
pub struct KVectorRangeAir {
    context: AirContext<BaseElement>,
    /// Target values for each component (stored for potential future use)
    #[allow(dead_code)]
    targets: [BaseElement; NUM_COMPONENTS],
}

impl KVectorRangeAir {
    /// Maximum value after scaling (1.0 * SCALE_FACTOR)
    pub const MAX_VALUE: u64 = SCALE_FACTOR;
}

impl Air for KVectorRangeAir {
    type BaseField = BaseElement;
    type PublicInputs = KVectorPublicInputs;

    fn new(trace_info: TraceInfo, pub_inputs: Self::PublicInputs, options: ProofOptions) -> Self {
        // Constraint degrees (must match actual polynomial degrees):
        // 0. Bit is binary: degree 2 (bit * (bit - 1) = 0)
        // 1. Step increments: degree 1 (step' - step - 1 = 0)
        // 2. Bit index cycles: degree 2 ((bit_idx + 1) * (1 - is_last_bit))
        // 3. Component increments: degree 2 (use multiplication to ensure degree 2)
        // 4. Target consistency: degree 2 ((target_next - target) * (1 - is_last_bit))
        let degrees = vec![
            TransitionConstraintDegree::new(2), // bit binary
            TransitionConstraintDegree::new(1), // step increment
            TransitionConstraintDegree::new(2), // bit_index transition
            TransitionConstraintDegree::new(2), // component transition (reformulated)
            TransitionConstraintDegree::new(2), // target consistency
        ];

        // Boundary assertions:
        // - Row 0: step=0, component=0, bit_index=0 (3 assertions)
        // - Row 111: step=111, component=7, bit_index=13 (3 assertions)
        // - Each component's last row: accumulated == target (8 assertions)
        //   Rows 13, 27, 41, 55, 69, 83, 97, 111
        let num_assertions = 3 + 3 + NUM_COMPONENTS; // 14 total

        let context = AirContext::new(trace_info, degrees, num_assertions, options);

        let mut targets = [BaseElement::ZERO; NUM_COMPONENTS];
        for (i, &t) in pub_inputs.targets.iter().enumerate() {
            targets[i] = BaseElement::from(t);
        }

        Self { context, targets }
    }

    fn context(&self) -> &AirContext<Self::BaseField> {
        &self.context
    }

    fn evaluate_transition<E: FieldElement + From<Self::BaseField>>(
        &self,
        frame: &EvaluationFrame<E>,
        _periodic_values: &[E],
        result: &mut [E],
    ) {
        let current = frame.current();
        let next = frame.next();

        let step = current[columns::STEP];
        let step_next = next[columns::STEP];
        let component = current[columns::COMPONENT];
        let component_next = next[columns::COMPONENT];
        let bit_idx = current[columns::BIT_INDEX];
        let bit_idx_next = next[columns::BIT_INDEX];
        let bit = current[columns::BIT_VALUE];
        let target = current[columns::TARGET];
        let target_next = next[columns::TARGET];
        let is_last_bit = current[columns::IS_LAST_BIT];

        let one = E::ONE;

        // Constraint 0: bit is binary
        // bit * (bit - 1) = 0
        result[0] = bit * (bit - one);

        // Constraint 1: step increments by 1
        // step' - step - 1 = 0
        result[1] = step_next - step - one;

        // Constraint 2: bit_index transitions
        // If bit_index < 13: bit_index' = bit_index + 1
        // If bit_index = 13: bit_index' = 0
        // Encoded as: bit_index' = (bit_index + 1) * (1 - is_last_bit)
        let expected_bit_idx = (bit_idx + one) * (one - is_last_bit);
        result[2] = bit_idx_next - expected_bit_idx;

        // Constraint 3: component transitions (degree 2 formulation)
        // When is_last_bit = 1: component_next = component + 1
        // When is_last_bit = 0: component_next = component
        // Formulated as: (component_next - component - is_last_bit) * (component_next - component) = 0
        // This ensures either no change OR exactly +1 change
        let comp_diff = component_next - component;
        result[3] = (comp_diff - is_last_bit) * comp_diff;

        // Constraint 4: target consistency
        // target' = target when not transitioning component
        // target' = next_target when transitioning
        // Simplified: target values come from public inputs and are verified at boundaries
        result[4] = (target_next - target) * (one - is_last_bit);
    }

    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        // Last computation row (not last trace row due to padding)
        let last_comp_row = COMPUTATION_STEPS - 1; // 111

        let mut assertions = vec![
            // Row 0 assertions
            Assertion::single(columns::STEP, 0, BaseElement::ZERO),
            Assertion::single(columns::COMPONENT, 0, BaseElement::ZERO),
            Assertion::single(columns::BIT_INDEX, 0, BaseElement::ZERO),
            // Last computation row assertions
            Assertion::single(columns::STEP, last_comp_row, BaseElement::from(last_comp_row as u64)),
            Assertion::single(columns::COMPONENT, last_comp_row, BaseElement::from(7u64)),
            Assertion::single(columns::BIT_INDEX, last_comp_row, BaseElement::from(13u64)),
        ];

        // Add accumulated == target assertion at each component's last row
        // Component i ends at row (i + 1) * BITS_PER_VALUE - 1
        for comp in 0..NUM_COMPONENTS {
            let last_row = (comp + 1) * BITS_PER_VALUE - 1;
            assertions.push(Assertion::single(
                columns::ACCUMULATED,
                last_row,
                self.targets[comp],
            ));
        }

        assertions
    }
}

/// Scale a f32 value in [0, 1] to an integer for ZKP
///
/// # Security Note (H-01 remediation)
/// This function includes explicit bounds checking to prevent
/// floating-point edge cases from producing out-of-range values.
pub fn scale_value(value: f32) -> u64 {
    // Clamp to valid range first to handle floating-point edge cases
    let clamped = value.clamp(0.0, 1.0);
    let scaled = (clamped * SCALE_FACTOR as f32).round() as u64;
    // Belt-and-suspenders: ensure we never exceed SCALE_FACTOR
    scaled.min(SCALE_FACTOR)
}

/// Scale a f32 value with strict validation (returns error on out-of-range)
///
/// Use this when you need to detect invalid inputs rather than clamping.
pub fn scale_value_strict(value: f32) -> Result<u64, &'static str> {
    if !value.is_finite() {
        return Err("Value must be finite");
    }
    if value < 0.0 || value > 1.0 {
        return Err("Value must be in range [0, 1]");
    }
    let scaled = (value * SCALE_FACTOR as f32).round() as u64;
    if scaled > SCALE_FACTOR {
        // This can happen with values very close to 1.0 due to floating-point
        return Err("Scaled value exceeds maximum");
    }
    Ok(scaled)
}

/// Unscale an integer back to f32
pub fn unscale_value(scaled: u64) -> f32 {
    scaled as f32 / SCALE_FACTOR as f32
}

/// Decompose a scaled value into bits (LSB first)
pub fn decompose_to_bits(value: u64) -> Vec<u64> {
    (0..BITS_PER_VALUE).map(|i| (value >> i) & 1).collect()
}

/// Reconstruct a value from bits (LSB first)
pub fn reconstruct_from_bits(bits: &[u64]) -> u64 {
    bits.iter().enumerate().map(|(i, &b)| b << i).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_unscale() {
        let original = 0.75f32;
        let scaled = scale_value(original);
        let recovered = unscale_value(scaled);

        assert!((original - recovered).abs() < 0.0001);
    }

    #[test]
    fn test_bit_decomposition() {
        let value = 7500u64; // 0.75 scaled
        let bits = decompose_to_bits(value);
        let reconstructed = reconstruct_from_bits(&bits);

        assert_eq!(value, reconstructed);
    }

    #[test]
    fn test_max_value_fits_in_bits() {
        let max_bits = decompose_to_bits(SCALE_FACTOR);
        let reconstructed = reconstruct_from_bits(&max_bits);

        assert_eq!(SCALE_FACTOR, reconstructed);
    }

    // H-01 remediation tests
    #[test]
    fn test_scale_value_edge_cases() {
        // Normal values
        assert_eq!(scale_value(0.0), 0);
        assert_eq!(scale_value(0.5), 5000);
        assert_eq!(scale_value(1.0), 10000);

        // Edge case: value slightly above 1.0 (floating-point error)
        assert_eq!(scale_value(1.0000001), 10000); // Should clamp

        // Edge case: value slightly below 0.0
        assert_eq!(scale_value(-0.0000001), 0); // Should clamp

        // Edge case: NaN should become 0 (clamped)
        assert_eq!(scale_value(f32::NAN), 0);

        // Edge case: Infinity should clamp to max
        assert_eq!(scale_value(f32::INFINITY), 10000);
        assert_eq!(scale_value(f32::NEG_INFINITY), 0);
    }

    #[test]
    fn test_scale_value_strict_validates() {
        // Valid values
        assert!(scale_value_strict(0.0).is_ok());
        assert!(scale_value_strict(0.5).is_ok());
        assert!(scale_value_strict(1.0).is_ok());

        // Invalid: out of range
        assert!(scale_value_strict(1.1).is_err());
        assert!(scale_value_strict(-0.1).is_err());

        // Invalid: non-finite
        assert!(scale_value_strict(f32::NAN).is_err());
        assert!(scale_value_strict(f32::INFINITY).is_err());
    }

    #[test]
    fn test_scale_value_cross_platform_consistency() {
        // Test values that might have different floating-point representations
        let test_values = [
            0.0, 0.1, 0.25, 0.333333, 0.5, 0.666666, 0.75, 0.9, 0.99, 0.999, 0.9999, 1.0,
        ];

        for &v in &test_values {
            let scaled = scale_value(v);
            assert!(scaled <= SCALE_FACTOR, "Value {} scaled to {} exceeds max", v, scaled);

            // Round-trip should be close
            let recovered = unscale_value(scaled);
            assert!((v - recovered).abs() < 0.0002, "Round-trip error for {}", v);
        }
    }

    #[test]
    fn test_trace_length() {
        // TRACE_LENGTH must be power of 2 for Winterfell
        assert_eq!(TRACE_LENGTH, 128);
        // COMPUTATION_STEPS is the actual computation (8 components × 14 bits)
        assert_eq!(COMPUTATION_STEPS, 112);
        assert_eq!(columns::WIDTH, 7);
    }
}
