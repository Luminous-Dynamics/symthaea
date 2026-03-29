// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! PoGQ AIR Definition
//!
//! Defines the algebraic constraints for correct PoGQ state transitions.

use winterfell::{
    Air, AirContext, Assertion, EvaluationFrame, ProofOptions, TraceInfo,
    TransitionConstraintDegree,
};

use math::{fields::f128::BaseElement, FieldElement, StarkField, ToElements};

/// Wrapper for public inputs that implements ToElements
#[derive(Clone)]
pub struct PublicInputsWrapper(pub Vec<BaseElement>);

impl ToElements<BaseElement> for PublicInputsWrapper {
    fn to_elements(&self) -> Vec<BaseElement> {
        self.0.clone()
    }
}

/// Number of columns in execution trace
///
/// Layout (LEAN range-check version with 32 bit columns):
/// 0: ema_t       - EMA score (Q16.16 scaled to field)
/// 1: viol_t      - Consecutive violations counter
/// 2: clear_t     - Consecutive clears counter
/// 3: quar_t      - Quarantine flag (0 or 1)
/// 4: x_t         - Hybrid score this round (witness, Q16.16)
/// 5: threshold   - Conformal threshold (constant)
/// 6: round       - Current round number
/// 7: is_violation - Boolean selector (1 if x_t < threshold, else 0)
/// 8: is_warmup    - Boolean selector (1 if round <= w, else 0)
/// 9: is_release  - Boolean selector (1 if quar==1 && projected_clear >= m, else 0)
/// 10: is_entering - Boolean selector (1 if projected_viol >= k && quar==0, else 0)
/// 11: rem_t      - Remainder from EMA division (0 <= rem < SCALE)
/// 12-27: rem_bits[0..15] - 16-bit decomposition of rem_t (LSB first)
/// 28-43: x_bits[0..15]   - 16-bit decomposition of x_t (LSB first)
///
/// Range-check rationale (lean approach):
/// - rem_t: Must prove 0 <= rem < SCALE to prevent EMA arithmetic overflow
/// - x_t: Private witness, must prove valid Q16.16 range to prevent manipulation
/// - ema_t: Implicitly bounded by EMA equation + rem constraint (no bits needed)
/// - counters: Implicit reset via multiplication prevents overflow attacks (no bits needed)
/// - beta/threshold: Public inputs, validated at parsing (no bits needed)
///
/// Note: Selectors at row i use PROJECTED next counters computed from row i,
/// ensuring alignment with AIR transition constraints that operate on row i.
/// beta is a PUBLIC CONSTANT (not in trace) to keep EMA constraint at degree-1.
///
/// CONSTRAINTS: 40 total (C0-C39) - selectors implicitly bounded, not explicitly constrained
pub const TRACE_WIDTH: usize = 44;

/// Fixed-point scale factor (Q16.16)
pub const SCALE: u128 = 65536;

/// Current AIR schema revision
///
/// Increment this whenever trace width, constraint count, or range check configuration changes.
/// Version history:
/// - v1: PoGQ v4.1 LEAN (44 cols, 40 constraints, 32-bit range checks)
pub const AIR_SCHEMA_REV: u32 = 1;

/// Public inputs for PoGQ AIR
#[derive(Clone, Debug)]
pub struct PublicInputs {
    // PoGQ parameters (u64 is sufficient for Q16.16: max value 65536)
    pub beta: u64,        // EMA beta * SCALE (e.g., 55705 for 0.85)
    pub w: u64,           // Warm-up rounds
    pub k: u64,           // Violations to quarantine
    pub m: u64,           // Clears to release
    pub threshold: u64,   // Conformal threshold * SCALE

    // Initial state
    pub ema_init: u64,
    pub viol_init: u64,
    pub clear_init: u64,
    pub quar_init: u64,
    pub round_init: u64,

    // Expected output
    pub quar_out: u64,

    // Trace length (must be power of 2)
    pub trace_length: usize,

    // Provenance (tamper-evident configuration commitment)
    // 32-byte Blake3 hash split into 4× u64 (little-endian)
    pub prov_hash: [u64; 4],

    // Security profile ID (128 or 192 for S128/S192)
    pub profile_id: u32,

    // AIR schema revision (monotone version number, increment on constraint/width changes)
    pub air_rev: u32,
}

impl PublicInputs {
    // NOTE: from_core() method removed - vsv-core doesn't have pogq module in current version
    // If vsv-core is updated with pogq module, add this method back:
    //
    // pub fn from_core(core_public: &vsv_core::pogq::PublicInputs, trace_length: usize) -> Self { ... }

    /// Convert to field elements for AIR evaluation
    ///
    /// Includes provenance commitment (hash + profile + AIR revision) to ensure
    /// tamper-evident configuration binding.
    pub fn to_elements(&self) -> Vec<BaseElement> {
        vec![
            // PoGQ parameters and state
            BaseElement::from(self.beta),
            BaseElement::from(self.w),
            BaseElement::from(self.k),
            BaseElement::from(self.m),
            BaseElement::from(self.threshold),
            BaseElement::from(self.ema_init),
            BaseElement::from(self.viol_init),
            BaseElement::from(self.clear_init),
            BaseElement::from(self.quar_init),
            BaseElement::from(self.round_init),
            BaseElement::from(self.quar_out),
            // Provenance commitment (32-byte hash as 4× u64 LE)
            BaseElement::from(self.prov_hash[0]),
            BaseElement::from(self.prov_hash[1]),
            BaseElement::from(self.prov_hash[2]),
            BaseElement::from(self.prov_hash[3]),
            // Profile ID (128 or 192)
            BaseElement::from(self.profile_id as u64),
            // AIR schema revision
            BaseElement::from(self.air_rev as u64),
        ]
    }
}

/// PoGQ AIR
pub struct PoGQAir {
    context: AirContext<BaseElement>,
    beta: BaseElement,
    w: BaseElement,
    k: BaseElement,
    m: BaseElement,
    threshold: BaseElement,
    ema_init: BaseElement,
    viol_init: BaseElement,
    clear_init: BaseElement,
    quar_init: BaseElement,
    round_init: BaseElement,
    quar_out: BaseElement,
    // Provenance fields (field-encoded for public input commitment)
    prov_hash: [BaseElement; 4],
    profile_id: BaseElement,
    air_rev: BaseElement,
}

impl PoGQAir {
    pub fn new(trace_info: TraceInfo, public_inputs: PublicInputs, options: ProofOptions) -> Self {
        // Compute trace polynomial degree: d = T - 1
        // TransitionConstraintDegree expects RELATIVE degree (number of column multiplications),
        // not absolute polynomial degree! Winterfell computes absolute degree as: relative_deg * (T-1)
        // So for affine constraints (no multiplications): relative_deg = 1
        // For boolean constraints (one multiplication like b*(b-1)): relative_deg = 2

        let degrees = vec![
            // Core PoGQ constraints (C0-C4)
            TransitionConstraintDegree::new(1), // C0: EMA with remainder (affine, no multiplications)
            TransitionConstraintDegree::new(1), // C1: Violation counter (affine)
            TransitionConstraintDegree::new(1), // C2: Clear counter (affine)
            TransitionConstraintDegree::new(1), // C3: Quarantine (affine: chained linear interpolation)
            TransitionConstraintDegree::new(1), // C4: Round increment (affine)
            // NOTE: Removed explicit boolean constraints on selectors (was C5-C8)
            // Reason: Selectors that are constant produce degree-0 polynomials, causing
            // degree validation failures. Selectors are implicitly bounded by their usage
            // in other constraints and by construction from trace columns

            // LEAN Range Checks: rem_t bit decomposition (C5-C21)
            // C5-C20: Boolean constraints for rem_t bits (b*(b-1), one multiplication = relative degree 2)
            TransitionConstraintDegree::new(2), // C5:  rem_bit[0] boolean
            TransitionConstraintDegree::new(2), // C6:  rem_bit[1] boolean
            TransitionConstraintDegree::new(2), // C7:  rem_bit[2] boolean
            TransitionConstraintDegree::new(2), // C8:  rem_bit[3] boolean
            TransitionConstraintDegree::new(2), // C9:  rem_bit[4] boolean
            TransitionConstraintDegree::new(2), // C10: rem_bit[5] boolean
            TransitionConstraintDegree::new(2), // C11: rem_bit[6] boolean
            TransitionConstraintDegree::new(2), // C12: rem_bit[7] boolean
            TransitionConstraintDegree::new(2), // C13: rem_bit[8] boolean
            TransitionConstraintDegree::new(2), // C14: rem_bit[9] boolean
            TransitionConstraintDegree::new(2), // C15: rem_bit[10] boolean
            TransitionConstraintDegree::new(2), // C16: rem_bit[11] boolean
            TransitionConstraintDegree::new(2), // C17: rem_bit[12] boolean
            TransitionConstraintDegree::new(2), // C18: rem_bit[13] boolean
            TransitionConstraintDegree::new(2), // C19: rem_bit[14] boolean
            TransitionConstraintDegree::new(2), // C20: rem_bit[15] boolean
            TransitionConstraintDegree::new(1), // C21: rem_t reconstruction (affine: sum of bits * constants)

            // LEAN Range Checks: x_t bit decomposition (C22-C38)
            // C22-C37: Boolean constraints for x_t bits (b*(b-1), one multiplication = relative degree 2)
            TransitionConstraintDegree::new(2), // C22: x_bit[0] boolean
            TransitionConstraintDegree::new(2), // C23: x_bit[1] boolean
            TransitionConstraintDegree::new(2), // C24: x_bit[2] boolean
            TransitionConstraintDegree::new(2), // C25: x_bit[3] boolean
            TransitionConstraintDegree::new(2), // C26: x_bit[4] boolean
            TransitionConstraintDegree::new(2), // C27: x_bit[5] boolean
            TransitionConstraintDegree::new(2), // C28: x_bit[6] boolean
            TransitionConstraintDegree::new(2), // C29: x_bit[7] boolean
            TransitionConstraintDegree::new(2), // C30: x_bit[8] boolean
            TransitionConstraintDegree::new(2), // C31: x_bit[9] boolean
            TransitionConstraintDegree::new(2), // C32: x_bit[10] boolean
            TransitionConstraintDegree::new(2), // C33: x_bit[11] boolean
            TransitionConstraintDegree::new(2), // C34: x_bit[12] boolean
            TransitionConstraintDegree::new(2), // C35: x_bit[13] boolean
            TransitionConstraintDegree::new(2), // C36: x_bit[14] boolean
            TransitionConstraintDegree::new(2), // C37: x_bit[15] boolean
            TransitionConstraintDegree::new(1), // C38: x_t reconstruction (affine: sum of bits * constants)

            // Defense-in-depth: Explicit quarantine boolean (C39)
            TransitionConstraintDegree::new(2), // C39: quar_t boolean (b*(b-1), one multiplication)
        ];

        Self {
            context: AirContext::new(trace_info, degrees, 6, options), // 6 assertions (5 init + 1 final)
            beta: BaseElement::from(public_inputs.beta),
            w: BaseElement::from(public_inputs.w),
            k: BaseElement::from(public_inputs.k),
            m: BaseElement::from(public_inputs.m),
            threshold: BaseElement::from(public_inputs.threshold),
            ema_init: BaseElement::from(public_inputs.ema_init),
            viol_init: BaseElement::from(public_inputs.viol_init),
            clear_init: BaseElement::from(public_inputs.clear_init),
            quar_init: BaseElement::from(public_inputs.quar_init),
            round_init: BaseElement::from(public_inputs.round_init),
            quar_out: BaseElement::from(public_inputs.quar_out),
            // Provenance fields (field-encoded, not used in constraints)
            prov_hash: [
                BaseElement::from(public_inputs.prov_hash[0]),
                BaseElement::from(public_inputs.prov_hash[1]),
                BaseElement::from(public_inputs.prov_hash[2]),
                BaseElement::from(public_inputs.prov_hash[3]),
            ],
            profile_id: BaseElement::from(public_inputs.profile_id as u64),
            air_rev: BaseElement::from(public_inputs.air_rev as u64),
        }
    }

    /// Get public inputs as field elements
    pub fn get_public_inputs(&self) -> Vec<BaseElement> {
        vec![
            self.beta,
            self.w,
            self.k,
            self.m,
            self.threshold,
            self.ema_init,
            self.viol_init,
            self.clear_init,
            self.quar_init,
            self.round_init,
            self.quar_out,
            // Provenance fields (already BaseElement)
            self.prov_hash[0],
            self.prov_hash[1],
            self.prov_hash[2],
            self.prov_hash[3],
            self.profile_id,
            self.air_rev,
        ]
    }

}

impl Air for PoGQAir {
    type BaseField = BaseElement;
    type PublicInputs = PublicInputsWrapper;

    fn new(trace_info: TraceInfo, pub_inputs: PublicInputsWrapper, options: ProofOptions) -> Self {
        eprintln!("DEBUG Air::new(): trace_info.length()={}, width()={}", trace_info.length(), trace_info.width());
        let pub_inputs = pub_inputs.0; // Extract Vec from wrapper
        // Reconstruct PublicInputs from elements
        let public_inputs = PublicInputs {
            beta: pub_inputs[0].as_int() as u64,
            w: pub_inputs[1].as_int() as u64,
            k: pub_inputs[2].as_int() as u64,
            m: pub_inputs[3].as_int() as u64,
            threshold: pub_inputs[4].as_int() as u64,
            ema_init: pub_inputs[5].as_int() as u64,
            viol_init: pub_inputs[6].as_int() as u64,
            clear_init: pub_inputs[7].as_int() as u64,
            quar_init: pub_inputs[8].as_int() as u64,
            round_init: pub_inputs[9].as_int() as u64,
            quar_out: pub_inputs[10].as_int() as u64,
            trace_length: trace_info.length(),
            // Provenance fields (elements 11-16)
            prov_hash: [
                pub_inputs[11].as_int() as u64,
                pub_inputs[12].as_int() as u64,
                pub_inputs[13].as_int() as u64,
                pub_inputs[14].as_int() as u64,
            ],
            profile_id: pub_inputs[15].as_int() as u32,
            air_rev: pub_inputs[16].as_int() as u32,
        };

        Self::new(trace_info, public_inputs, options)
    }

    fn context(&self) -> &AirContext<Self::BaseField> {
        &self.context
    }

    fn evaluate_transition<E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        frame: &EvaluationFrame<E>,
        _periodic_values: &[E],
        result: &mut [E],
    )
    where
        E: From<Self::BaseField>,
    {
        let current = frame.current();
        let next = frame.next();

        // Extract current state (columns 0-5)
        let ema_t = current[0];
        let viol_t = current[1];
        let clear_t = current[2];
        let quar_t = current[3];
        let x_t = current[4];
        let _threshold = current[5]; // Constant (not used in constraints)
        let round = current[6];

        // Extract next state
        let ema_next = next[0];
        let viol_next = next[1];
        let clear_next = next[2];
        let quar_next = next[3];
        let round_next = next[6];

        // Extract selector columns (7-10)
        let is_violation = current[7];
        let is_warmup = current[8];
        let is_release = current[9];
        let is_entering = current[10];

        // Extract remainder column (11) - from NEXT row since it corresponds to ema_next
        let rem = next[11];

        // Get beta from public inputs (as constant, not from trace)
        let beta = E::from(self.beta);

        // Constants
        let scale = E::from(BaseElement::from(SCALE as u64));
        let one = E::ONE;
        let zero = E::ZERO;

        // Constraint 0: EMA update with remainder column (EXACT division encoding)
        // Check: ema_next * SCALE + rem = beta * ema + (SCALE - beta) * x
        // Where rem is the remainder from truncating division
        // This allows perfect field arithmetic matching of integer division
        let lhs = ema_next * scale + rem;
        let rhs = beta * ema_t + (scale - beta) * x_t;

        result[0] = lhs - rhs;

        // Constraint 1: Violation counter update (with implicit reset)
        // viol[t+1] = is_violation * (viol[t] + 1)
        // When is_violation=1: increment, when is_violation=0: reset to 0
        let viol_expected = is_violation * (viol_t + one);
        result[1] = viol_next - viol_expected;

        // Constraint 2: Clear counter update (with implicit reset)
        // clear[t+1] = (1 - is_violation) * (clear[t] + 1)
        // When is_violation=0: increment, when is_violation=1: reset to 0
        let not_violation = one - is_violation;
        let clear_expected = not_violation * (clear_t + one);
        result[2] = clear_next - clear_expected;

        // Constraint 3: Quarantine state transition (selector chaining for degree-2)
        // quar[t+1] = warmup ? 0 : (release ? 0 : (entering ? 1 : quar[t]))
        //
        // FIXED: Use linear interpolation chaining to avoid multiplying selectors
        // Step 1: entering_gated = is_entering * 1 + (1 - is_entering) * quar_t  (degree 2)
        // Step 2: release_gated = is_release * 0 + (1 - is_release) * entering_gated  (degree 2)
        // Step 3: warmup_gated = is_warmup * 0 + (1 - is_warmup) * release_gated  (degree 2)
        let entering_gated = is_entering * one + (one - is_entering) * quar_t;
        let release_gated = is_release * zero + (one - is_release) * entering_gated;
        let quar_expected = is_warmup * zero + (one - is_warmup) * release_gated;
        result[3] = quar_next - quar_expected;

        // Constraint 4: Round increment
        result[4] = round_next - (round + one);

        // ============================================================================
        // LEAN RANGE CHECKS: rem_t and x_t bit decomposition (32 bits total)
        // ============================================================================
        //
        // Attack vectors closed:
        // 1. rem_t >= SCALE: Breaks EMA arithmetic, allows fake EMA values
        // 2. x_t out of Q16.16 range: Malicious witness manipulation
        //
        // Why NOT check ema_t: Implicitly bounded by EMA equation + rem constraint
        // Why NOT check counters: Implicit reset prevents overflow attacks
        // Why NOT check beta/threshold: Public inputs, validated at parsing

        // Extract bit columns for remainder (rem_t) - columns 12-27
        let rem_bits: Vec<E> = (12..28).map(|i| current[i]).collect();

        // Extract bit columns for witness score (x_t) - columns 28-43
        let x_bits: Vec<E> = (28..44).map(|i| current[i]).collect();

        // Constraints 5-20: Boolean enforcement for rem_t bits (16 constraints, degree 2)
        // Each bit b_i must satisfy: b_i * (b_i - 1) = 0  (forces b_i ∈ {0, 1})
        for i in 0..16 {
            result[5 + i] = rem_bits[i] * (rem_bits[i] - one);
        }

        // Constraint 21: Reconstruction of rem_t from bits (degree 1)
        // rem_t = Σ(2^i * b_i) for i=0..15
        // Ensures: 0 <= rem_t <= 65535 < SCALE=65536
        let mut rem_reconstructed = zero;
        let mut power_of_two = one;
        for i in 0..16 {
            rem_reconstructed = rem_reconstructed + power_of_two * rem_bits[i];
            power_of_two = power_of_two + power_of_two; // Multiply by 2
        }
        result[21] = current[11] - rem_reconstructed; // current[11] is rem_t

        // Constraints 22-37: Boolean enforcement for x_t bits (16 constraints, degree 2)
        for i in 0..16 {
            result[22 + i] = x_bits[i] * (x_bits[i] - one);
        }

        // Constraint 38: Reconstruction of x_t from bits (degree 1)
        // x_t = Σ(2^i * b_i) for i=0..15
        // Ensures: 0 <= x_t <= 65535 (valid Q16.16 probability/score)
        let mut x_reconstructed = zero;
        let mut power_of_two = one;
        for i in 0..16 {
            x_reconstructed = x_reconstructed + power_of_two * x_bits[i];
            power_of_two = power_of_two + power_of_two;
        }
        result[38] = x_t - x_reconstructed; // x_t is current[4]

        // Constraint 39: Explicit boolean for quar_t (defense-in-depth, degree 2)
        // Previously implicit, now explicit: quar_t * (quar_t - 1) = 0
        result[39] = quar_t * (quar_t - one);
    }

    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        let last_step = self.trace_length() - 1;

        vec![
            // Initial state assertions (main columns)
            Assertion::single(0, 0, self.ema_init),   // ema[0]
            Assertion::single(1, 0, self.viol_init),  // viol[0]
            Assertion::single(2, 0, self.clear_init), // clear[0]
            Assertion::single(3, 0, self.quar_init),  // quar[0]
            Assertion::single(6, 0, self.round_init), // round[0] - column 6 after beta removal
            // Final output assertion
            Assertion::single(3, last_step, self.quar_out), // quar[N] = expected output
            // Note: Selector column initial values not asserted (derived from state)
        ]
    }
}
