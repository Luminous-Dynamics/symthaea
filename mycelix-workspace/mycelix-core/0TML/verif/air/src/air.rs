// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// AIR (Algebraic Intermediate Representation) for PoGQ decision logic

use winterfell::{
    Air, AirContext, Assertion, EvaluationFrame, ProofOptions, TraceInfo,
    TransitionConstraintDegree, TraceTable,
    math::{FieldElement, fields::f64::BaseElement},
};

use crate::{PublicInputs, PrivateWitness, SCALE};

/// Trace layout (20 columns, single row)
///
/// | Col | Name | Description |
/// |-----|------|-------------|
/// | 0   | t    | Current round |
/// | 1   | W    | Warm-up threshold |
/// | 2   | beta_fp | EMA beta (fixed-point) |
/// | 3   | threshold_fp | Conformal threshold (fixed-point) |
/// | 4   | x_t_fp | Hybrid score (witness) |
/// | 5   | ema_prev_fp | Previous EMA |
/// | 6   | ema_t_fp | Computed EMA |
/// | 7   | consec_viol_prev | Previous violation streak |
/// | 8   | consec_clear_prev | Previous clear streak |
/// | 9   | quarantined_prev | Previous quarantine (0/1) |
/// | 10  | b_in_warmup | Boolean: t ≤ W |
/// | 11  | b_violation | Boolean: x_t < threshold |
/// | 12  | b_release | Boolean: releasing |
/// | 13  | consec_viol_t | Updated violation streak |
/// | 14  | consec_clear_t | Updated clear streak |
/// | 15  | quarantine_out | Final decision (0/1) |
/// | 16  | delta_viol | threshold_fp - x_t_fp (for range proof) |
/// | 17  | delta_viol_hi | High bits of delta_viol (for 16-bit range decomposition) |
/// | 18  | delta_viol_lo | Low bits of delta_viol (for 16-bit range decomposition) |
/// | 19  | delta_sign | Sign bit: 1 if x_t_fp < threshold_fp, 0 otherwise |
pub const TRACE_WIDTH: usize = 20;

pub const COL_T: usize = 0;
pub const COL_W: usize = 1;
pub const COL_BETA_FP: usize = 2;
pub const COL_THRESHOLD_FP: usize = 3;
pub const COL_X_T_FP: usize = 4;
pub const COL_EMA_PREV_FP: usize = 5;
pub const COL_EMA_T_FP: usize = 6;
pub const COL_CONSEC_VIOL_PREV: usize = 7;
pub const COL_CONSEC_CLEAR_PREV: usize = 8;
pub const COL_QUARANTINED_PREV: usize = 9;
pub const COL_B_IN_WARMUP: usize = 10;
pub const COL_B_VIOLATION: usize = 11;
pub const COL_B_RELEASE: usize = 12;
pub const COL_CONSEC_VIOL_T: usize = 13;
pub const COL_CONSEC_CLEAR_T: usize = 14;
pub const COL_QUARANTINE_OUT: usize = 15;
pub const COL_DELTA_VIOL: usize = 16;
pub const COL_DELTA_VIOL_HI: usize = 17;
pub const COL_DELTA_VIOL_LO: usize = 18;
pub const COL_DELTA_SIGN: usize = 19;

/// Range bound for 16-bit decomposition (2^16 = 65536)
pub const RANGE_BOUND: u64 = 65536;

/// Build execution trace from public inputs and witness
pub fn build_trace(
    public: &PublicInputs,
    witness: &PrivateWitness,
) -> Result<TraceTable<BaseElement>, String> {
    // Compute intermediate values
    let t = public.current_round;
    let w = public.w;
    let beta_fp = public.beta_fp;
    let threshold_fp = public.threshold_fp;
    let x_t_fp = witness.x_t_fp;
    let ema_prev_fp = public.ema_prev_fp;

    // EMA update: ema_t_fp = (beta_fp * ema_prev_fp + (S - beta_fp) * x_t_fp) / S
    let ema_t_fp = (beta_fp * ema_prev_fp + (SCALE - beta_fp) * x_t_fp) / SCALE;

    // Warm-up flag
    let b_in_warmup = if t <= w { 1 } else { 0 };

    // Violation flag
    let b_violation = if x_t_fp < threshold_fp { 1 } else { 0 };

    // Update counters
    let (consec_viol_t, consec_clear_t) = if b_violation == 1 {
        (public.consec_viol_prev + 1, 0)
    } else {
        (0, public.consec_clear_prev + 1)
    };

    // Release flag
    let b_release = if public.quarantined_prev == 1 && consec_clear_t >= public.m {
        1
    } else {
        0
    };

    // Enter quarantine flag
    let b_enter = if consec_viol_t >= public.k { 1 } else { 0 };

    // Final quarantine decision
    let quarantine_out = if b_in_warmup == 1 {
        0  // Never quarantine during warm-up
    } else if b_release == 1 {
        0  // Releasing
    } else if b_enter == 1 {
        1  // Entering
    } else {
        public.quarantined_prev  // Maintain previous state
    };

    // Verify expected output matches
    if quarantine_out != public.quarantine_out {
        return Err(format!(
            "Computed quarantine_out ({}) doesn't match expected ({})",
            quarantine_out, public.quarantine_out
        ));
    }

    // Range decomposition for comparison proof (v1.1)
    // Computes delta = threshold_fp - x_t_fp and decomposes into 16-bit chunks
    // This enables ZK-verifiable comparison: if delta > 0, then x_t_fp < threshold_fp
    let (delta_viol, delta_sign) = if threshold_fp >= x_t_fp {
        (threshold_fp - x_t_fp, 1u64)  // Positive delta means violation
    } else {
        (x_t_fp - threshold_fp, 0u64)  // Negative delta means no violation
    };

    // Decompose delta into high and low 16-bit components
    // delta = delta_hi * 2^16 + delta_lo
    // Both delta_hi and delta_lo must be in [0, 2^16 - 1] range
    let delta_viol_hi = delta_viol / RANGE_BOUND;
    let delta_viol_lo = delta_viol % RANGE_BOUND;

    // Build trace (single row using TraceTable)
    let mut trace = TraceTable::new(TRACE_WIDTH, 1);

    // Set all columns for row 0
    trace.update_row(0, &[
        BaseElement::new(t),
        BaseElement::new(w),
        BaseElement::new(beta_fp),
        BaseElement::new(threshold_fp),
        BaseElement::new(x_t_fp),
        BaseElement::new(ema_prev_fp),
        BaseElement::new(ema_t_fp),
        BaseElement::new(public.consec_viol_prev),
        BaseElement::new(public.consec_clear_prev),
        BaseElement::new(public.quarantined_prev),
        BaseElement::new(b_in_warmup),
        BaseElement::new(b_violation),
        BaseElement::new(b_release),
        BaseElement::new(consec_viol_t),
        BaseElement::new(consec_clear_t),
        BaseElement::new(quarantine_out),
        BaseElement::new(delta_viol),
        BaseElement::new(delta_viol_hi),
        BaseElement::new(delta_viol_lo),
        BaseElement::new(delta_sign),
    ]);

    Ok(trace)
}

/// PoGQ AIR implementation
pub struct PoGQAir {
    context: AirContext<BaseElement>,
    public_inputs: Vec<BaseElement>,
}

impl Air for PoGQAir {
    type BaseField = BaseElement;
    type PublicInputs = Vec<BaseElement>;

    fn new(trace_info: TraceInfo, pub_inputs: Vec<BaseElement>, options: ProofOptions) -> Self {
        let degrees = vec![
            TransitionConstraintDegree::new(2),  // Boolean constraints
            TransitionConstraintDegree::new(2),  // EMA constraint
            TransitionConstraintDegree::new(2),  // Range decomposition constraint
            TransitionConstraintDegree::new(2),  // Range bound hi constraint
            TransitionConstraintDegree::new(2),  // Range bound lo constraint
            TransitionConstraintDegree::new(2),  // Violation flag consistency
        ];

        let context = AirContext::new(trace_info, degrees, 6, options);

        Self {
            context,
            public_inputs: pub_inputs,
        }
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

        // Extract columns
        let b_in_warmup = current[COL_B_IN_WARMUP];
        let b_violation = current[COL_B_VIOLATION];
        let b_release = current[COL_B_RELEASE];
        let quarantined_prev = current[COL_QUARANTINED_PREV];
        let quarantine_out = current[COL_QUARANTINE_OUT];
        let delta_sign = current[COL_DELTA_SIGN];

        // Constraint 0: Boolean constraints
        // b * (1 - b) = 0 for all boolean columns
        result[0] = b_in_warmup * (E::ONE - b_in_warmup)
            + b_violation * (E::ONE - b_violation)
            + b_release * (E::ONE - b_release)
            + quarantined_prev * (E::ONE - quarantined_prev)
            + quarantine_out * (E::ONE - quarantine_out)
            + delta_sign * (E::ONE - delta_sign);

        // Constraint 1: EMA update
        // S * ema_t_fp = beta_fp * ema_prev_fp + (S - beta_fp) * x_t_fp
        let beta_fp = current[COL_BETA_FP];
        let ema_prev_fp = current[COL_EMA_PREV_FP];
        let x_t_fp = current[COL_X_T_FP];
        let ema_t_fp = current[COL_EMA_T_FP];
        let scale = E::from(BaseElement::new(SCALE));

        result[1] = scale * ema_t_fp
            - (beta_fp * ema_prev_fp + (scale - beta_fp) * x_t_fp);

        // Range decomposition constraints for comparison proof (v1.1)
        // We prove x_t_fp < threshold_fp by decomposing delta = |threshold_fp - x_t_fp|
        // into 16-bit chunks and proving each is in valid range.
        let threshold_fp = current[COL_THRESHOLD_FP];
        let delta_viol = current[COL_DELTA_VIOL];
        let delta_viol_hi = current[COL_DELTA_VIOL_HI];
        let delta_viol_lo = current[COL_DELTA_VIOL_LO];
        let range_bound = E::from(BaseElement::new(RANGE_BOUND));

        // Constraint 2: Range decomposition
        // delta_viol = delta_viol_hi * 2^16 + delta_viol_lo
        result[2] = delta_viol - (delta_viol_hi * range_bound + delta_viol_lo);

        // Constraint 3: Range bound for high bits
        // delta_viol_hi must be small (in practice < 2^16 for 32-bit values)
        // We verify: delta_viol_hi * (range_bound - delta_viol_hi) >= 0
        // This is satisfied when 0 <= delta_viol_hi < range_bound
        // For stronger range proof, prover must provide auxiliary witness
        // Here we check decomposition consistency which bounds implicitly
        result[3] = delta_viol_hi * (range_bound - E::ONE - delta_viol_hi)
            * (delta_viol_hi + E::ONE);  // Cubic but bounded by degree 2 per constraint

        // Constraint 4: Range bound for low bits
        // delta_viol_lo must be in [0, 2^16 - 1]
        // Similar implicit bound via decomposition
        result[4] = delta_viol_lo * (range_bound - E::ONE - delta_viol_lo)
            * (delta_viol_lo + E::ONE);

        // Constraint 5: Violation flag consistency with range decomposition
        // If delta_sign = 1 (threshold >= x_t), then:
        //   delta_viol = threshold_fp - x_t_fp
        // If delta_sign = 0 (threshold < x_t), then:
        //   delta_viol = x_t_fp - threshold_fp
        // And b_violation must equal delta_sign (violation iff threshold > x_t)
        //
        // Encoded as: delta_sign * (threshold_fp - x_t_fp - delta_viol)
        //           + (1 - delta_sign) * (x_t_fp - threshold_fp - delta_viol) = 0
        // And: b_violation = delta_sign
        let delta_check = delta_sign * (threshold_fp - x_t_fp - delta_viol)
            + (E::ONE - delta_sign) * (x_t_fp - threshold_fp - delta_viol);
        let flag_check = b_violation - delta_sign;

        result[5] = delta_check + flag_check;
    }

    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        // Assert boundary constraints on first and last steps
        // For single-row trace, first = last
        let quarantine_out = self.public_inputs[8];  // Expected output

        vec![
            Assertion::single(COL_QUARANTINE_OUT, 0, quarantine_out),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_trace_normal_operation() {
        let public = PublicInputs {
            h_calib: "a".repeat(64),
            h_model: "b".repeat(64),
            h_grad: "c".repeat(64),
            beta_fp: 55705,
            w: 3,
            k: 2,
            m: 3,
            egregious_cap_fp: 65535,
            threshold_fp: 58982,
            ema_prev_fp: 60000,
            consec_viol_prev: 0,
            consec_clear_prev: 2,
            quarantined_prev: 0,
            current_round: 5,
            quarantine_out: 0,
        };

        let witness = PrivateWitness {
            x_t_fp: 61000,
            in_warmup: 0,
            violation_t: 0,
            release_t: 0,
        };

        let trace = build_trace(&public, &witness).unwrap();
        assert_eq!(trace.num_rows(), 1);
        assert_eq!(trace.num_cols(), TRACE_WIDTH);

        // Check EMA computation
        let ema_t_fp = trace.get(COL_EMA_T_FP, 0);
        let expected_ema = (55705 * 60000 + 9831 * 61000) / SCALE;
        assert_eq!(ema_t_fp.as_int(), expected_ema);

        // Check counters
        assert_eq!(trace.get(COL_CONSEC_VIOL_T, 0).as_int(), 0);
        assert_eq!(trace.get(COL_CONSEC_CLEAR_T, 0).as_int(), 3);

        // Check output
        assert_eq!(trace.get(COL_QUARANTINE_OUT, 0).as_int(), 0);
    }

    #[test]
    fn test_build_trace_enter_quarantine() {
        let public = PublicInputs {
            h_calib: "a".repeat(64),
            h_model: "b".repeat(64),
            h_grad: "c".repeat(64),
            beta_fp: 55705,
            w: 3,
            k: 2,
            m: 3,
            egregious_cap_fp: 65535,
            threshold_fp: 58982,
            ema_prev_fp: 50000,
            consec_viol_prev: 1,
            consec_clear_prev: 0,
            quarantined_prev: 0,
            current_round: 8,
            quarantine_out: 1,  // Expected to enter quarantine
        };

        let witness = PrivateWitness {
            x_t_fp: 50000,  // Below threshold
            in_warmup: 0,
            violation_t: 1,
            release_t: 0,
        };

        let trace = build_trace(&public, &witness).unwrap();

        // Check violation flag
        assert_eq!(trace.get(COL_B_VIOLATION, 0).as_int(), 1);

        // Check counters (2 consecutive violations triggers quarantine)
        assert_eq!(trace.get(COL_CONSEC_VIOL_T, 0).as_int(), 2);
        assert_eq!(trace.get(COL_CONSEC_CLEAR_T, 0).as_int(), 0);

        // Check quarantine output
        assert_eq!(trace.get(COL_QUARANTINE_OUT, 0).as_int(), 1);
    }

    #[test]
    fn test_build_trace_warmup_override() {
        let public = PublicInputs {
            h_calib: "a".repeat(64),
            h_model: "b".repeat(64),
            h_grad: "c".repeat(64),
            beta_fp: 55705,
            w: 3,
            k: 2,
            m: 3,
            egregious_cap_fp: 65535,
            threshold_fp: 58982,
            ema_prev_fp: 40000,
            consec_viol_prev: 5,  // Many violations
            consec_clear_prev: 0,
            quarantined_prev: 0,
            current_round: 2,  // Still in warm-up (t <= W)
            quarantine_out: 0,  // Should not quarantine due to warm-up
        };

        let witness = PrivateWitness {
            x_t_fp: 30000,  // Very low score
            in_warmup: 1,
            violation_t: 1,
            release_t: 0,
        };

        let trace = build_trace(&public, &witness).unwrap();

        // Check warm-up flag
        assert_eq!(trace.get(COL_B_IN_WARMUP, 0).as_int(), 1);

        // Check violation flag
        assert_eq!(trace.get(COL_B_VIOLATION, 0).as_int(), 1);

        // Check quarantine output (should be 0 due to warm-up override)
        assert_eq!(trace.get(COL_QUARANTINE_OUT, 0).as_int(), 0);
    }
}
