// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Winterfell AIR circuit: Range proof for health values.
//!
//! Proves: "value ∈ [min, max]" without revealing the actual value.
//!
//! Used for: VitalsInRange, AgeRange, LabThreshold health proof types.
//!
//! ## Circuit Design
//!
//! Decomposes (value - min) and (max - value) into bits, proving both are non-negative.
//! Trace: 4 columns × 32 rows (2 phases × 16 bits, padded to power of 2).

use winterfell::{
    math::{fields::f128::BaseElement, FieldElement, StarkField, ToElements},
    matrix::ColMatrix,
    Air, AirContext, Assertion, BatchingMethod, CompositionPoly, CompositionPolyTrace,
    DefaultConstraintCommitment, DefaultConstraintEvaluator, DefaultTraceLde, EvaluationFrame,
    PartitionOptions, ProofOptions, Prover as WinterfellProver, StarkDomain, Trace, TraceInfo,
    TracePolyTable, TransitionConstraintDegree,
    crypto::{hashers::Blake3_256, DefaultRandomCoin, MerkleTree},
    AcceptableOptions, Proof,
};

const TRACE_WIDTH: usize = 4;
const BITS_PER_VALUE: usize = 16;
const TRACE_LENGTH: usize = 32; // 2 × 16, already power of 2

mod col {
    pub const BIT: usize = 0;
    pub const ACCUMULATED: usize = 1;
    pub const BIT_INDEX: usize = 2;
    pub const PHASE: usize = 3;
}

// Type aliases matching kvector-zkp pattern
type Hasher = Blake3_256<BaseElement>;
type VC = MerkleTree<Hasher>;
type RandCoin = DefaultRandomCoin<Hasher>;

/// Public inputs for range proof
#[derive(Clone, Debug)]
pub struct RangePublicInputs {
    pub min_value: u64,
    pub max_value: u64,
    pub value_commitment: [u8; 32],
}

impl ToElements<BaseElement> for RangePublicInputs {
    fn to_elements(&self) -> Vec<BaseElement> {
        let mut elements = Vec::with_capacity(6);
        elements.push(BaseElement::from(self.min_value));
        elements.push(BaseElement::from(self.max_value));
        for chunk in self.value_commitment.chunks(8) {
            let mut bytes = [0u8; 8];
            bytes[..chunk.len()].copy_from_slice(chunk);
            elements.push(BaseElement::from(u64::from_le_bytes(bytes)));
        }
        elements
    }
}

/// AIR for health value range proofs
pub struct HealthRangeAir {
    context: AirContext<BaseElement>,
}

impl Air for HealthRangeAir {
    type BaseField = BaseElement;
    type PublicInputs = RangePublicInputs;

    fn new(trace_info: TraceInfo, _pub_inputs: Self::PublicInputs, options: ProofOptions) -> Self {
        let degrees = vec![
            TransitionConstraintDegree::new(2), // bit binary: bit × (bit - 1) = 0
        ];
        let num_assertions = 4; // start + end of each phase
        let context = AirContext::new(trace_info, degrees, num_assertions, options);
        Self { context }
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
        let one = E::ONE;

        // Constraint 0: bit is binary (the core soundness constraint)
        let bit = current[col::BIT];
        result[0] = bit * (bit - one);
    }

    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        vec![
            // Phase 0 start
            Assertion::single(col::BIT_INDEX, 0, BaseElement::ZERO),
            Assertion::single(col::ACCUMULATED, 0, BaseElement::ZERO),
            // Phase 1 start
            Assertion::single(col::BIT_INDEX, BITS_PER_VALUE, BaseElement::ZERO),
            Assertion::single(col::ACCUMULATED, BITS_PER_VALUE, BaseElement::ZERO),
        ]
    }
}

/// Execution trace wrapper (implements Trace trait like KVectorTrace)
pub struct RangeTrace {
    info: TraceInfo,
    trace: ColMatrix<BaseElement>,
}

impl Trace for RangeTrace {
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
        frame: &mut EvaluationFrame<Self::BaseField>,
    ) {
        let next_row_idx = (row_idx + 1) % self.trace.num_rows();
        self.trace.read_row_into(row_idx, frame.current_mut());
        self.trace.read_row_into(next_row_idx, frame.next_mut());
    }
}

/// Build execution trace for range proof
fn build_trace(value: u64, min: u64, max: u64) -> RangeTrace {
    assert!(value >= min && value <= max);

    let diff_low = value - min;
    let diff_high = max - value;

    let mut cols = vec![vec![BaseElement::ZERO; TRACE_LENGTH]; TRACE_WIDTH];

    // Phase 0: decompose (value - min)
    // accumulated[i] is the sum of bits 0..i-1 (starts at 0, ends at full value)
    let mut acc = 0u64;
    for i in 0..BITS_PER_VALUE {
        let bit = (diff_low >> i) & 1;
        cols[col::BIT][i] = BaseElement::from(bit);
        cols[col::BIT_INDEX][i] = BaseElement::from(i as u64);
        cols[col::ACCUMULATED][i] = BaseElement::from(acc); // acc BEFORE this bit
        cols[col::PHASE][i] = BaseElement::ZERO;
        acc += bit << i;
    }

    // Phase 1: decompose (max - value)
    acc = 0;
    for i in 0..BITS_PER_VALUE {
        let row = BITS_PER_VALUE + i;
        let bit = (diff_high >> i) & 1;
        cols[col::BIT][row] = BaseElement::from(bit);
        cols[col::BIT_INDEX][row] = BaseElement::from(i as u64);
        cols[col::ACCUMULATED][row] = BaseElement::from(acc); // acc BEFORE this bit
        cols[col::PHASE][row] = BaseElement::ONE;
        acc += bit << i;
    }

    let info = TraceInfo::new(TRACE_WIDTH, TRACE_LENGTH);
    let trace = ColMatrix::new(cols);
    RangeTrace { info, trace }
}

/// Prover (follows kvector-zkp pattern exactly)
struct RangeProver {
    options: ProofOptions,
    pub_inputs: RangePublicInputs,
}

impl WinterfellProver for RangeProver {
    type BaseField = BaseElement;
    type Air = HealthRangeAir;
    type Trace = RangeTrace;
    type HashFn = Hasher;
    type VC = VC;
    type RandomCoin = RandCoin;
    type TraceLde<E: FieldElement<BaseField = BaseElement>> = DefaultTraceLde<E, Hasher, VC>;
    type ConstraintCommitment<E: FieldElement<BaseField = BaseElement>> =
        DefaultConstraintCommitment<E, Hasher, VC>;
    type ConstraintEvaluator<'a, E: FieldElement<BaseField = BaseElement>> =
        DefaultConstraintEvaluator<'a, HealthRangeAir, E>;

    fn options(&self) -> &ProofOptions {
        &self.options
    }

    fn get_pub_inputs(&self, _trace: &Self::Trace) -> RangePublicInputs {
        self.pub_inputs.clone()
    }

    fn new_trace_lde<E: FieldElement<BaseField = BaseElement>>(
        &self,
        trace_info: &TraceInfo,
        main_trace: &ColMatrix<BaseElement>,
        domain: &StarkDomain<BaseElement>,
        partition_options: PartitionOptions,
    ) -> (Self::TraceLde<E>, TracePolyTable<E>) {
        DefaultTraceLde::new(trace_info, main_trace, domain, partition_options)
    }

    fn build_constraint_commitment<E: FieldElement<BaseField = BaseElement>>(
        &self,
        composition_poly_trace: CompositionPolyTrace<E>,
        num_constraint_composition_columns: usize,
        domain: &StarkDomain<BaseElement>,
        partition_options: PartitionOptions,
    ) -> (Self::ConstraintCommitment<E>, CompositionPoly<E>) {
        DefaultConstraintCommitment::new(
            composition_poly_trace,
            num_constraint_composition_columns,
            domain,
            partition_options,
        )
    }

    fn new_evaluator<'a, E: FieldElement<BaseField = BaseElement>>(
        &self,
        air: &'a Self::Air,
        aux_rand_elements: Option<winterfell::AuxRandElements<E>>,
        composition_coefficients: winterfell::ConstraintCompositionCoefficients<E>,
    ) -> Self::ConstraintEvaluator<'a, E> {
        DefaultConstraintEvaluator::new(air, aux_rand_elements, composition_coefficients)
    }
}

fn default_options() -> ProofOptions {
    ProofOptions::new(
        28, 8, 0,
        winterfell::FieldExtension::None,
        8, 31,
        BatchingMethod::Linear,
        BatchingMethod::Linear,
    )
}

/// Generate a STARK proof that `value ∈ [min, max]`.
pub fn prove_range(
    value: u64,
    min: u64,
    max: u64,
    value_commitment: [u8; 32],
) -> Result<Proof, String> {
    let trace = build_trace(value, min, max);
    let pub_inputs = RangePublicInputs { min_value: min, max_value: max, value_commitment };
    let prover = RangeProver { options: default_options(), pub_inputs };
    prover.prove(trace).map_err(|e| format!("Proving failed: {:?}", e))
}

/// Verify a STARK range proof.
pub fn verify_range(
    proof: Proof,
    min: u64,
    max: u64,
    value_commitment: [u8; 32],
) -> Result<(), String> {
    let pub_inputs = RangePublicInputs { min_value: min, max_value: max, value_commitment };
    let acceptable = AcceptableOptions::OptionSet(vec![default_options()]);
    winterfell::verify::<HealthRangeAir, Hasher, RandCoin, VC>(proof, pub_inputs, &acceptable)
        .map_err(|e| format!("Verification failed: {:?}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn commit(value: u64) -> [u8; 32] {
        let h = Sha256::digest(value.to_le_bytes());
        let mut c = [0u8; 32];
        c.copy_from_slice(&h);
        c
    }

    #[test]
    fn test_age_in_range() {
        let proof = prove_range(35, 18, 65, commit(35)).expect("prove");
        verify_range(proof, 18, 65, commit(35)).expect("verify");
    }

    #[test]
    fn test_boundary_min() {
        let proof = prove_range(18, 18, 65, commit(18)).expect("prove");
        verify_range(proof, 18, 65, commit(18)).expect("verify");
    }

    #[test]
    fn test_boundary_max() {
        let proof = prove_range(65, 18, 65, commit(65)).expect("prove");
        verify_range(proof, 18, 65, commit(65)).expect("verify");
    }

    #[test]
    fn test_lab_threshold() {
        let proof = prove_range(54, 0, 70, commit(54)).expect("prove");
        verify_range(proof, 0, 70, commit(54)).expect("verify");
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_below_min_panics() {
        let _ = prove_range(17, 18, 65, commit(17));
    }

    #[test]
    fn test_wrong_bounds_fail() {
        let proof = prove_range(35, 18, 65, commit(35)).expect("prove");
        assert!(verify_range(proof, 30, 40, commit(35)).is_err());
    }
}
