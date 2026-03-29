// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Prover for K-Vector range proofs
//!
//! Generates STARK proofs that K-Vector components are in [0, 1].

use sha3::{Digest, Sha3_256};
use winterfell::{
    math::{fields::f128::BaseElement, FieldElement, StarkField},
    matrix::ColMatrix,
    BatchingMethod, CompositionPoly, CompositionPolyTrace, DefaultConstraintCommitment,
    DefaultTraceLde, PartitionOptions, ProofOptions, Prover, StarkDomain, Trace, TraceInfo,
    TracePolyTable,
};

use crate::air::{
    columns, decompose_to_bits, scale_value, KVectorPublicInputs, KVectorRangeAir,
    BITS_PER_VALUE, NUM_COMPONENTS, TRACE_LENGTH,
};
use crate::error::{ZkpError, ZkpResult};

/// K-Vector values for proving
#[derive(Clone, Debug)]
pub struct KVectorWitness {
    /// k_r: Reputation score
    pub k_r: f32,
    /// k_a: Activity score
    pub k_a: f32,
    /// k_i: Integrity score
    pub k_i: f32,
    /// k_p: Performance score
    pub k_p: f32,
    /// k_m: Membership duration score
    pub k_m: f32,
    /// k_s: Stake weight score
    pub k_s: f32,
    /// k_h: Historical consistency score
    pub k_h: f32,
    /// k_topo: Network topology contribution score
    pub k_topo: f32,
}

impl KVectorWitness {
    /// Create from an array of values
    pub fn from_array(values: [f32; 8]) -> Self {
        Self {
            k_r: values[0],
            k_a: values[1],
            k_i: values[2],
            k_p: values[3],
            k_m: values[4],
            k_s: values[5],
            k_h: values[6],
            k_topo: values[7],
        }
    }

    /// Convert to array
    pub fn to_array(&self) -> [f32; 8] {
        [
            self.k_r, self.k_a, self.k_i, self.k_p, self.k_m, self.k_s, self.k_h, self.k_topo,
        ]
    }

    /// Get scaled values as u64 array
    pub fn scaled_values(&self) -> [u64; NUM_COMPONENTS] {
        let arr = self.to_array();
        [
            scale_value(arr[0]),
            scale_value(arr[1]),
            scale_value(arr[2]),
            scale_value(arr[3]),
            scale_value(arr[4]),
            scale_value(arr[5]),
            scale_value(arr[6]),
            scale_value(arr[7]),
        ]
    }

    /// Validate all components are in range and not degenerate
    ///
    /// M-05 remediation: Expanded degenerate input detection to catch
    /// more edge cases that could affect proof soundness.
    pub fn validate(&self) -> ZkpResult<()> {
        let components = [
            ("k_r", self.k_r),
            ("k_a", self.k_a),
            ("k_i", self.k_i),
            ("k_p", self.k_p),
            ("k_m", self.k_m),
            ("k_s", self.k_s),
            ("k_h", self.k_h),
            ("k_topo", self.k_topo),
        ];

        // H-02: Check for non-finite values (NaN, Infinity) - critical for ZK soundness
        // NaN/Inf values can bypass constraint validation and create invalid proofs
        for (name, value) in components {
            if value.is_nan() {
                return Err(ZkpError::DegenerateInput(
                    format!("component {} is NaN - invalid for ZK proofs", name),
                ));
            }
            if value.is_infinite() {
                return Err(ZkpError::DegenerateInput(
                    format!("component {} is infinite - invalid for ZK proofs", name),
                ));
            }
            if !(0.0..=1.0).contains(&value) {
                return Err(ZkpError::ValueOutOfRange {
                    component: name,
                    value,
                });
            }
        }

        let values: Vec<f32> = components.iter().map(|(_, v)| *v).collect();

        // Check for degenerate all-zeros case which causes trivial constraint polynomials
        // In the ZKP circuit, when all values are 0, the bit and target columns become
        // constant zero, making constraints 0 and 4 identically zero (degree 0 polynomial).
        // This is rejected by winterfell's degree validation.
        let all_zero = values.iter().all(|&v| v == 0.0);
        if all_zero {
            return Err(ZkpError::DegenerateInput(
                "all components are zero - at least one must be non-zero".into(),
            ));
        }

        // M-05: Check for all-identical values (except all-zeros already handled)
        // This can cause linear dependencies in constraints
        let first = values[0];
        let all_identical = values.iter().all(|&v| (v - first).abs() < 1e-7);
        if all_identical && first != 0.0 {
            return Err(ZkpError::DegenerateInput(
                "all components have identical values - need diversity for sound proofs".into(),
            ));
        }

        // M-05: Check for very low variance (near-degenerate case)
        // Low variance means values are clustered, which can reduce constraint effectiveness
        let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
        let variance: f32 = values.iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f32>() / values.len() as f32;

        // Minimum variance threshold - if all values are within 0.01 of each other
        const MIN_VARIANCE: f32 = 0.0001; // (0.01)^2
        if variance < MIN_VARIANCE && mean > 0.01 {
            return Err(ZkpError::DegenerateInput(
                format!(
                    "K-Vector variance too low ({:.6}) - values are too similar for sound proofs",
                    variance
                ),
            ));
        }

        // M-05: Check for extreme skew (all values near 0 or all near 1)
        // This can reduce the effectiveness of bit decomposition constraints
        let near_zero_count = values.iter().filter(|&&v| v < 0.01).count();
        let near_one_count = values.iter().filter(|&&v| v > 0.99).count();

        if near_zero_count >= 7 {
            return Err(ZkpError::DegenerateInput(
                "too many near-zero values - need more diversity for sound proofs".into(),
            ));
        }

        if near_one_count >= 7 {
            return Err(ZkpError::DegenerateInput(
                "too many near-one values - need more diversity for sound proofs".into(),
            ));
        }

        Ok(())
    }

    /// Check if the witness is likely to produce a valid proof
    ///
    /// This is a quick check that can be used before expensive proof generation.
    pub fn is_likely_valid(&self) -> bool {
        self.validate().is_ok()
    }

    /// Compute commitment hash
    pub fn commitment(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();

        for value in self.to_array() {
            let scaled = scale_value(value);
            hasher.update(scaled.to_le_bytes());
        }

        hasher.finalize().into()
    }
}

/// Execution trace for the K-Vector range proof
pub struct KVectorTrace {
    info: TraceInfo,
    trace: ColMatrix<BaseElement>,
}

impl KVectorTrace {
    /// Build the trace from a K-Vector witness
    pub fn new(witness: &KVectorWitness) -> ZkpResult<Self> {
        witness.validate()?;

        // Get scaled values and their bit decompositions
        let scaled = witness.scaled_values();
        let bits: Vec<Vec<u64>> = scaled.iter().map(|&v| decompose_to_bits(v)).collect();

        // Initialize trace columns
        let mut step_col = vec![BaseElement::ZERO; TRACE_LENGTH];
        let mut component_col = vec![BaseElement::ZERO; TRACE_LENGTH];
        let mut bit_idx_col = vec![BaseElement::ZERO; TRACE_LENGTH];
        let mut bit_val_col = vec![BaseElement::ZERO; TRACE_LENGTH];
        let mut accumulated_col = vec![BaseElement::ZERO; TRACE_LENGTH];
        let mut target_col = vec![BaseElement::ZERO; TRACE_LENGTH];
        let mut is_last_bit_col = vec![BaseElement::ZERO; TRACE_LENGTH];

        // Fill in the trace row by row
        let mut row = 0;
        for comp in 0..NUM_COMPONENTS {
            let target = BaseElement::from(scaled[comp]);
            let mut accumulated = 0u64;

            for bit_idx in 0..BITS_PER_VALUE {
                let bit = bits[comp][bit_idx];
                accumulated += bit << bit_idx;

                step_col[row] = BaseElement::from(row as u64);
                component_col[row] = BaseElement::from(comp as u64);
                bit_idx_col[row] = BaseElement::from(bit_idx as u64);
                bit_val_col[row] = BaseElement::from(bit);
                accumulated_col[row] = BaseElement::from(accumulated);
                target_col[row] = target;
                is_last_bit_col[row] = if bit_idx == BITS_PER_VALUE - 1 {
                    BaseElement::ONE
                } else {
                    BaseElement::ZERO
                };

                row += 1;
            }

            // Verify accumulated equals target
            debug_assert_eq!(accumulated, scaled[comp], "Bit reconstruction failed for component {}", comp);
        }

        // Pad remaining rows to reach TRACE_LENGTH (power of 2)
        // Padding rows must satisfy transition constraints, so we continue the pattern
        // as if processing phantom components (8, 9, ...) with zero values
        let mut pad_component = NUM_COMPONENTS as u64; // Start at component 8
        let mut pad_bit_idx = 0usize;
        let mut pad_accumulated = 0u64;

        for pad_row in row..TRACE_LENGTH {
            let is_last_bit = pad_bit_idx == BITS_PER_VALUE - 1;

            step_col[pad_row] = BaseElement::from(pad_row as u64);
            component_col[pad_row] = BaseElement::from(pad_component);
            bit_idx_col[pad_row] = BaseElement::from(pad_bit_idx as u64);
            bit_val_col[pad_row] = BaseElement::ZERO; // Padding bits are 0
            accumulated_col[pad_row] = BaseElement::from(pad_accumulated);
            target_col[pad_row] = BaseElement::ZERO; // Phantom components have zero target
            is_last_bit_col[pad_row] = if is_last_bit {
                BaseElement::ONE
            } else {
                BaseElement::ZERO
            };

            // Advance to next step (mirroring the computation logic)
            if is_last_bit {
                pad_component += 1;
                pad_bit_idx = 0;
                pad_accumulated = 0;
            } else {
                pad_bit_idx += 1;
                // accumulated stays 0 since all padding bits are 0
            }
        }

        let trace = ColMatrix::new(vec![
            step_col,
            component_col,
            bit_idx_col,
            bit_val_col,
            accumulated_col,
            target_col,
            is_last_bit_col,
        ]);

        let info = TraceInfo::new(columns::WIDTH, TRACE_LENGTH);

        Ok(Self { info, trace })
    }

    /// Get the public inputs for this trace
    pub fn public_inputs(&self, witness: &KVectorWitness) -> KVectorPublicInputs {
        KVectorPublicInputs::new(witness.commitment(), witness.scaled_values())
    }
}

impl Trace for KVectorTrace {
    type BaseField = BaseElement;

    fn info(&self) -> &TraceInfo {
        &self.info
    }

    fn main_segment(&self) -> &ColMatrix<Self::BaseField> {
        &self.trace
    }

    fn read_main_frame(
        &self,
        row_idx: usize,
        frame: &mut winterfell::EvaluationFrame<Self::BaseField>,
    ) {
        let next_row_idx = (row_idx + 1) % self.trace.num_rows();
        self.trace.read_row_into(row_idx, frame.current_mut());
        self.trace.read_row_into(next_row_idx, frame.next_mut());
    }
}

/// Prover for K-Vector range proofs
pub struct KVectorProver {
    options: ProofOptions,
}

impl KVectorProver {
    /// Create a new prover with default options
    pub fn new() -> Self {
        Self {
            options: ProofOptions::new(
                32,                              // number of queries
                8,                               // blowup factor
                0,                               // grinding factor
                winterfell::FieldExtension::None,
                8,                               // FRI folding factor
                31,                              // FRI max remainder polynomial degree
                BatchingMethod::Linear,          // trace batching
                BatchingMethod::Linear,          // constraint batching
            ),
        }
    }

    /// Create a prover with custom options
    pub fn with_options(options: ProofOptions) -> Self {
        Self { options }
    }
}

impl Default for KVectorProver {
    fn default() -> Self {
        Self::new()
    }
}

impl Prover for KVectorProver {
    type BaseField = BaseElement;
    type Air = KVectorRangeAir;
    type Trace = KVectorTrace;
    type HashFn = winterfell::crypto::hashers::Blake3_256<BaseElement>;
    type VC = winterfell::crypto::MerkleTree<Self::HashFn>;
    type RandomCoin = winterfell::crypto::DefaultRandomCoin<Self::HashFn>;
    type TraceLde<E: FieldElement<BaseField = Self::BaseField>> =
        DefaultTraceLde<E, Self::HashFn, Self::VC>;
    type ConstraintCommitment<E: FieldElement<BaseField = Self::BaseField>> =
        DefaultConstraintCommitment<E, Self::HashFn, Self::VC>;
    type ConstraintEvaluator<'a, E: FieldElement<BaseField = Self::BaseField>> =
        winterfell::DefaultConstraintEvaluator<'a, Self::Air, E>;

    fn options(&self) -> &ProofOptions {
        &self.options
    }

    fn get_pub_inputs(&self, trace: &Self::Trace) -> KVectorPublicInputs {
        // Extract target values from the trace
        let mut targets = [0u64; NUM_COMPONENTS];
        for comp in 0..NUM_COMPONENTS {
            // Target is in the last row of each component
            let row = (comp + 1) * BITS_PER_VALUE - 1;
            let target_elem = trace.main_segment().get(columns::TARGET, row);
            targets[comp] = target_elem.as_int() as u64;
        }

        // Compute commitment from the target values
        let mut hasher = Sha3_256::new();
        for &target in &targets {
            hasher.update(target.to_le_bytes());
        }
        let commitment: [u8; 32] = hasher.finalize().into();

        KVectorPublicInputs::new(commitment, targets)
    }

    fn new_trace_lde<E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        trace_info: &TraceInfo,
        main_trace: &ColMatrix<Self::BaseField>,
        domain: &StarkDomain<Self::BaseField>,
        partition_options: PartitionOptions,
    ) -> (Self::TraceLde<E>, TracePolyTable<E>) {
        DefaultTraceLde::new(trace_info, main_trace, domain, partition_options)
    }

    fn build_constraint_commitment<E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        composition_poly_trace: CompositionPolyTrace<E>,
        num_constraint_composition_columns: usize,
        domain: &StarkDomain<Self::BaseField>,
        partition_options: PartitionOptions,
    ) -> (Self::ConstraintCommitment<E>, CompositionPoly<E>) {
        DefaultConstraintCommitment::new(
            composition_poly_trace,
            num_constraint_composition_columns,
            domain,
            partition_options,
        )
    }

    fn new_evaluator<'a, E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        air: &'a Self::Air,
        aux_rand_elements: Option<winterfell::AuxRandElements<E>>,
        composition_coefficients: winterfell::ConstraintCompositionCoefficients<E>,
    ) -> Self::ConstraintEvaluator<'a, E> {
        winterfell::DefaultConstraintEvaluator::new(air, aux_rand_elements, composition_coefficients)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::COMPUTATION_STEPS;

    #[test]
    fn test_witness_validation() {
        let valid = KVectorWitness {
            k_r: 0.8,
            k_a: 0.7,
            k_i: 0.9,
            k_p: 0.6,
            k_m: 0.5,
            k_s: 0.4,
            k_h: 0.85,
            k_topo: 0.75,
        };

        assert!(valid.validate().is_ok());

        let invalid = KVectorWitness {
            k_r: 1.5, // Out of range
            k_a: 0.7,
            k_i: 0.9,
            k_p: 0.6,
            k_m: 0.5,
            k_s: 0.4,
            k_h: 0.85,
            k_topo: 0.75,
        };

        assert!(invalid.validate().is_err());
    }

    // M-05 remediation tests
    #[test]
    fn test_degenerate_all_zeros() {
        let all_zeros = KVectorWitness::from_array([0.0; 8]);
        assert!(all_zeros.validate().is_err());
    }

    #[test]
    fn test_degenerate_all_identical() {
        let all_same = KVectorWitness::from_array([0.5; 8]);
        assert!(all_same.validate().is_err());
    }

    #[test]
    fn test_degenerate_low_variance() {
        // All values between 0.49 and 0.51
        let low_variance = KVectorWitness {
            k_r: 0.500,
            k_a: 0.501,
            k_i: 0.502,
            k_p: 0.499,
            k_m: 0.500,
            k_s: 0.501,
            k_h: 0.500,
            k_topo: 0.499,
        };
        assert!(low_variance.validate().is_err());
    }

    #[test]
    fn test_degenerate_too_many_near_zero() {
        let near_zeros = KVectorWitness {
            k_r: 0.001,
            k_a: 0.002,
            k_i: 0.003,
            k_p: 0.001,
            k_m: 0.002,
            k_s: 0.003,
            k_h: 0.001,
            k_topo: 0.5, // Only one non-near-zero
        };
        assert!(near_zeros.validate().is_err());
    }

    #[test]
    fn test_degenerate_too_many_near_one() {
        let near_ones = KVectorWitness {
            k_r: 0.999,
            k_a: 0.998,
            k_i: 0.997,
            k_p: 0.999,
            k_m: 0.998,
            k_s: 0.997,
            k_h: 0.999,
            k_topo: 0.5, // Only one non-near-one
        };
        assert!(near_ones.validate().is_err());
    }

    #[test]
    fn test_valid_diverse_witness() {
        // Good distribution of values
        let diverse = KVectorWitness {
            k_r: 0.2,
            k_a: 0.4,
            k_i: 0.6,
            k_p: 0.8,
            k_m: 0.3,
            k_s: 0.5,
            k_h: 0.7,
            k_topo: 0.9,
        };
        assert!(diverse.validate().is_ok());
    }

    #[test]
    fn test_non_finite_values() {
        let nan_value = KVectorWitness {
            k_r: f32::NAN,
            k_a: 0.5,
            k_i: 0.5,
            k_p: 0.5,
            k_m: 0.5,
            k_s: 0.5,
            k_h: 0.5,
            k_topo: 0.5,
        };
        assert!(nan_value.validate().is_err());

        let inf_value = KVectorWitness {
            k_r: f32::INFINITY,
            k_a: 0.5,
            k_i: 0.5,
            k_p: 0.5,
            k_m: 0.5,
            k_s: 0.5,
            k_h: 0.5,
            k_topo: 0.5,
        };
        assert!(inf_value.validate().is_err());
    }

    #[test]
    fn test_is_likely_valid() {
        let valid = KVectorWitness::from_array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        assert!(valid.is_likely_valid());

        let invalid = KVectorWitness::from_array([0.5; 8]);
        assert!(!invalid.is_likely_valid());
    }

    #[test]
    fn test_commitment_deterministic() {
        let witness = KVectorWitness {
            k_r: 0.8,
            k_a: 0.7,
            k_i: 0.9,
            k_p: 0.6,
            k_m: 0.5,
            k_s: 0.4,
            k_h: 0.85,
            k_topo: 0.75,
        };

        let commit1 = witness.commitment();
        let commit2 = witness.commitment();

        assert_eq!(commit1, commit2);
    }

    #[test]
    fn test_trace_creation() {
        let witness = KVectorWitness {
            k_r: 0.8,
            k_a: 0.7,
            k_i: 0.9,
            k_p: 0.6,
            k_m: 0.5,
            k_s: 0.4,
            k_h: 0.85,
            k_topo: 0.75,
        };

        let trace = KVectorTrace::new(&witness);
        assert!(trace.is_ok());

        let trace = trace.unwrap();
        assert_eq!(trace.info().length(), TRACE_LENGTH);
        assert_eq!(trace.info().width(), columns::WIDTH);
    }

    #[test]
    fn test_trace_state_transitions() {
        // Diverse values satisfying M-05 variance requirements
        let witness = KVectorWitness {
            k_r: 0.40,
            k_a: 0.45,
            k_i: 0.50,
            k_p: 0.55,
            k_m: 0.60,
            k_s: 0.65,
            k_h: 0.70,
            k_topo: 0.75,
        };

        let trace = KVectorTrace::new(&witness).unwrap();
        let main = trace.main_segment();

        // Check first row
        assert_eq!(main.get(columns::STEP, 0), BaseElement::ZERO);
        assert_eq!(main.get(columns::COMPONENT, 0), BaseElement::ZERO);
        assert_eq!(main.get(columns::BIT_INDEX, 0), BaseElement::ZERO);

        // Check step increments
        for i in 0..TRACE_LENGTH {
            assert_eq!(main.get(columns::STEP, i), BaseElement::from(i as u64));
        }

        // Check last computation row (before padding)
        let last_comp = COMPUTATION_STEPS - 1; // Row 111
        assert_eq!(main.get(columns::STEP, last_comp), BaseElement::from(last_comp as u64));
        assert_eq!(main.get(columns::COMPONENT, last_comp), BaseElement::from(7u64));
        assert_eq!(main.get(columns::BIT_INDEX, last_comp), BaseElement::from(13u64));

        // Check that padding rows exist and have valid values
        let last_trace = TRACE_LENGTH - 1;
        assert_eq!(main.get(columns::STEP, last_trace), BaseElement::from(last_trace as u64));
    }

    #[test]
    fn test_scaled_values() {
        let witness = KVectorWitness {
            k_r: 0.8,
            k_a: 0.7,
            k_i: 0.9,
            k_p: 0.6,
            k_m: 0.5,
            k_s: 0.4,
            k_h: 0.85,
            k_topo: 0.75,
        };

        let scaled = witness.scaled_values();
        assert_eq!(scaled[0], 8000); // 0.8 * 10000
        assert_eq!(scaled[1], 7000); // 0.7 * 10000
        assert_eq!(scaled[4], 5000); // 0.5 * 10000
    }
}
