//! Small composed edge substrate: fixed-point liquid state plus packed exemplar memory.

use crate::liquid_state::{FixedPointLiquidState, LiquidStateError};
use crate::packed_hdc::{
    Classification, InsertOutcome, MemoryError, PackedHdcMemory, PackedHv,
};
use crate::time_constant::{alpha_from_dt_tau_us, TimeConstantError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeCoreUpdateError {
    ZeroTau,
    InvalidInterpolationCoefficient,
}

impl From<TimeConstantError> for EdgeCoreUpdateError {
    fn from(error: TimeConstantError) -> Self {
        match error {
            TimeConstantError::ZeroTau => Self::ZeroTau,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundedEdgeCore<
    const DIM: usize,
    const CLASSES: usize,
    const SLOTS: usize,
    const WORDS: usize,
> {
    liquid_state: FixedPointLiquidState<DIM>,
    associative_memory: PackedHdcMemory<CLASSES, SLOTS, WORDS>,
}

impl<const DIM: usize, const CLASSES: usize, const SLOTS: usize, const WORDS: usize>
    BoundedEdgeCore<DIM, CLASSES, SLOTS, WORDS>
{
    pub const fn new() -> Result<Self, MemoryError> {
        let memory = match PackedHdcMemory::new() {
            Ok(value) => value,
            Err(error) => return Err(error),
        };
        Ok(Self {
            liquid_state: FixedPointLiquidState::new(),
            associative_memory: memory,
        })
    }

    pub const fn liquid_state(&self) -> &FixedPointLiquidState<DIM> {
        &self.liquid_state
    }

    pub const fn associative_memory(&self) -> &PackedHdcMemory<CLASSES, SLOTS, WORDS> {
        &self.associative_memory
    }

    pub fn update_liquid_state(
        &mut self,
        equilibrium: &[i16; DIM],
        alpha_q15: u16,
    ) -> Result<(), LiquidStateError> {
        self.liquid_state.evolve_q15(equilibrium, alpha_q15)
    }

    /// Advance the liquid state using an integer dt/tau lookup coefficient.
    /// The coefficient policy is Q16.16 + table interpolation; no floating-point
    /// exponential is evaluated at runtime.
    pub fn update_liquid_state_from_dt_tau_us(
        &mut self,
        equilibrium: &[i16; DIM],
        dt_us: u32,
        tau_us: u32,
    ) -> Result<(), EdgeCoreUpdateError> {
        let alpha = alpha_from_dt_tau_us(dt_us, tau_us)?;
        self.liquid_state
            .evolve_q15(equilibrium, alpha)
            .map_err(|_| EdgeCoreUpdateError::InvalidInterpolationCoefficient)
    }

    pub fn remember(
        &mut self,
        class_index: usize,
        exemplar: PackedHv<WORDS>,
    ) -> Result<InsertOutcome, MemoryError> {
        self.associative_memory.insert(class_index, exemplar)
    }

    pub fn remember_with_replacement(
        &mut self,
        class_index: usize,
        exemplar: PackedHv<WORDS>,
    ) -> Result<InsertOutcome, MemoryError> {
        self.associative_memory.insert_or_replace_oldest(class_index, exemplar)
    }

    pub fn classify(&self, query: &PackedHv<WORDS>) -> Option<Classification> {
        self.associative_memory.predict(query)
    }

    /// Static payload size for the liquid state and fixed exemplar memory.
    /// Includes the per-class usize counters in the target's native usize width.
    pub const fn persistent_payload_bytes() -> Option<usize> {
        let liquid_bytes = match FixedPointLiquidState::<DIM>::payload_bytes() {
            Some(value) => value,
            None => return None,
        };
        let memory_bytes = match PackedHdcMemory::<CLASSES, SLOTS, WORDS>::payload_bytes() {
            Some(value) => value,
            None => return None,
        };
        liquid_bytes.checked_add(memory_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composed_core_updates_state_and_classifies_without_growth() {
        let mut core = BoundedEdgeCore::<4, 2, 2, 1>::new().unwrap();
        core.update_liquid_state(&[100, -100, 50, -50], 16_384).unwrap();
        assert_eq!(core.liquid_state().values(), &[50, -50, 25, -25]);

        let exemplar = PackedHv::<1>::from_words([0x55]);
        assert_eq!(core.remember(1, exemplar), Ok(InsertOutcome::Inserted));
        let result = core.classify(&exemplar).unwrap();
        assert_eq!(result.class_index, 1);
        assert_eq!(result.hamming_distance, 0);
        assert_eq!(result.similarity_q15, 32_767);
    }

    #[test]
    fn composed_core_uses_elapsed_time_and_tau_without_float_math() {
        let mut core = BoundedEdgeCore::<2, 1, 1, 1>::new().unwrap();
        core.update_liquid_state_from_dt_tau_us(&[10_000, -10_000], 10, 10).unwrap();
        assert_eq!(core.liquid_state().values(), &[6_321, -6_321]);
        assert_eq!(
            core.update_liquid_state_from_dt_tau_us(&[0, 0], 1, 0),
            Err(EdgeCoreUpdateError::ZeroTau)
        );
    }

    #[test]
    fn composed_static_payload_is_calculated_from_const_capacities() {
        // Liquid state (8 B) + exemplar words (32 B) + two usize arrays (4 * usize).
        // This is target-width aware: Cortex-M and 64-bit hosts have different usize widths.
        assert_eq!(
            BoundedEdgeCore::<4, 2, 2, 1>::persistent_payload_bytes(),
            Some(40 + 4 * core::mem::size_of::<usize>())
        );
    }

    #[test]
    fn invalid_memory_capacity_is_rejected() {
        assert!(BoundedEdgeCore::<4, 0, 1, 1>::new().is_err());
    }
}
