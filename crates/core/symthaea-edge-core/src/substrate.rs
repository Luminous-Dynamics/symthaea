//! Small composed edge substrate: fixed-point liquid state plus packed exemplar memory.

use crate::liquid_state::{FixedPointLiquidState, LiquidStateError};
use crate::packed_hdc::{
    Classification, InsertOutcome, MemoryError, PackedHdcMemory, PackedHv,
};

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
    fn composed_static_payload_is_calculated_from_const_capacities() {
        // 4 * 16-bit liquid state (8 B) + 2 * 2 * 8-byte HDC exemplars (32 B)
        // + two usize metadata arrays for 2 classes (32 B on a 64-bit host).
        assert_eq!(
            BoundedEdgeCore::<4, 2, 2, 1>::persistent_payload_bytes(),
            Some(72)
        );
    }

    #[test]
    fn invalid_memory_capacity_is_rejected() {
        assert!(BoundedEdgeCore::<4, 0, 1, 1>::new().is_err());
    }
}
