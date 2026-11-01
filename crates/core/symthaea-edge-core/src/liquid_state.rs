//! Allocation-free Q15 liquid-state interpolation.
//!
//! This is the fixed-point interpolation kernel only. The caller supplies alpha_q15,
//! an unsigned Q0.15 coefficient in [0, 32768] representing 1 - exp(-dt/tau).
//! A verified dt/tau or lookup-table policy is intentionally outside this primitive.

pub const Q15_ONE: u16 = 32_768;
const Q15_HALF: i64 = 16_384;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiquidStateError {
    InvalidInterpolationCoefficient,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedPointLiquidState<const DIM: usize> {
    values: [i16; DIM],
}

impl<const DIM: usize> FixedPointLiquidState<DIM> {
    pub const fn new() -> Self {
        Self { values: [0; DIM] }
    }

    pub const fn from_values(values: [i16; DIM]) -> Self {
        Self { values }
    }

    pub const fn values(&self) -> &[i16; DIM] {
        &self.values
    }

    pub fn reset(&mut self) {
        self.values = [0; DIM];
    }

    /// Apply convex Q15 interpolation with a caller-supplied coefficient.
    ///
    /// alpha=0 preserves state; alpha=32768 sets state to equilibrium.
    /// Intermediate values round to nearest with symmetric signed rounding.
    pub fn evolve_q15(
        &mut self,
        equilibrium: &[i16; DIM],
        alpha_q15: u16,
    ) -> Result<(), LiquidStateError> {
        if alpha_q15 > Q15_ONE {
            return Err(LiquidStateError::InvalidInterpolationCoefficient);
        }
        let alpha = alpha_q15 as i64;
        let retained = (Q15_ONE - alpha_q15) as i64;
        let mut index = 0usize;
        while index < DIM {
            let weighted = self.values[index] as i64 * retained
                + equilibrium[index] as i64 * alpha;
            let rounded = if weighted >= 0 {
                (weighted + Q15_HALF) / Q15_ONE as i64
            } else {
                (weighted - Q15_HALF) / Q15_ONE as i64
            };
            self.values[index] = rounded.clamp(i16::MIN as i64, i16::MAX as i64) as i16;
            index += 1;
        }
        Ok(())
    }

    pub const fn payload_bytes() -> Option<usize> {
        DIM.checked_mul(core::mem::size_of::<i16>())
    }
}

impl<const DIM: usize> Default for FixedPointLiquidState<DIM> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_zero_preserves_state() {
        let initial = [123, -456, i16::MAX, i16::MIN];
        let mut state = FixedPointLiquidState::from_values(initial);
        state.evolve_q15(&[-1, 2, 3, 4], 0).unwrap();
        assert_eq!(*state.values(), initial);
    }

    #[test]
    fn alpha_one_copies_equilibrium() {
        let mut state = FixedPointLiquidState::from_values([123, -456, 7]);
        let target = [i16::MIN, i16::MAX, -99];
        state.evolve_q15(&target, Q15_ONE).unwrap();
        assert_eq!(*state.values(), target);
    }

    #[test]
    fn alpha_half_interpolates_positive_and_negative_values() {
        let mut state = FixedPointLiquidState::from_values([0, 0, 100, -100]);
        let target = [100, -100, -100, 100];
        state.evolve_q15(&target, Q15_ONE / 2).unwrap();
        assert_eq!(*state.values(), [50, -50, 0, 0]);
    }

    #[test]
    fn interpolation_rounds_symmetrically_and_stays_in_range() {
        let mut state = FixedPointLiquidState::from_values([0, 0, i16::MAX, i16::MIN]);
        state.evolve_q15(&[1, -1, i16::MIN, i16::MAX], Q15_ONE / 2).unwrap();
        assert_eq!(state.values()[0], 1);
        assert_eq!(state.values()[1], -1);
        assert!(state.values().iter().all(|value| *value >= i16::MIN && *value <= i16::MAX));
    }

    #[test]
    fn invalid_coefficient_is_rejected_without_state_change() {
        let initial = [7, -8];
        let mut state = FixedPointLiquidState::from_values(initial);
        assert_eq!(
            state.evolve_q15(&[99, 99], Q15_ONE + 1),
            Err(LiquidStateError::InvalidInterpolationCoefficient)
        );
        assert_eq!(*state.values(), initial);
    }

    #[test]
    fn fixed_state_payload_is_exact() {
        assert_eq!(FixedPointLiquidState::<128>::payload_bytes(), Some(256));
        assert_eq!(FixedPointLiquidState::<0>::payload_bytes(), Some(0));
    }
}
