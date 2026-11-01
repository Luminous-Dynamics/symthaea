//! Fixed-capacity binary hypervectors and exemplar memory.
//! One 16,384-bit vector uses 2,048 bytes. All operations are allocation-free.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedHv<const WORDS: usize> {
    words: [u64; WORDS],
}

impl<const WORDS: usize> PackedHv<WORDS> {
    pub const fn zero() -> Self { Self { words: [0; WORDS] } }
    pub const fn from_words(words: [u64; WORDS]) -> Self { Self { words } }
    pub const fn words(&self) -> &[u64; WORDS] { &self.words }

    pub fn bind(&self, other: &Self) -> Self {
        let mut output = [0u64; WORDS];
        let mut i = 0;
        while i < WORDS { output[i] = self.words[i] ^ other.words[i]; i += 1; }
        Self { words: output }
    }

    pub fn hamming_distance(&self, other: &Self) -> usize {
        let mut distance = 0usize;
        let mut i = 0;
        while i < WORDS {
            distance += (self.words[i] ^ other.words[i]).count_ones() as usize;
            i += 1;
        }
        distance
    }

    /// Signed bipolar similarity in Q15, bounded to [-32767, 32767].
    pub fn similarity_q15(&self, other: &Self) -> i16 {
        let Some(bits) = WORDS.checked_mul(64) else { return 0 };
        if bits == 0 { return 0; }
        let distance = self.hamming_distance(other);
        let matches = bits.saturating_sub(distance);
        let signed = (matches as i128) * 2 - bits as i128;
        (signed * 32_767 / bits as i128).clamp(-32_767, 32_767) as i16
    }

    /// Cyclic left rotation across the complete packed bit string.
    /// Bit zero is the least-significant bit of word zero.
    pub fn permute_left(&self, positions: usize) -> Self {
        let Some(bits) = WORDS.checked_mul(64) else { return *self };
        if bits == 0 { return *self; }
        let shift = positions % bits;
        if shift == 0 { return *self; }
        let whole = shift / 64;
        let rem = shift % 64;
        let mut output = [0u64; WORDS];
        let mut source = 0usize;
        while source < WORDS {
            let destination = (source + whole) % WORDS;
            let value = self.words[source];
            output[destination] |= value << rem;
            if rem != 0 {
                let next = (destination + 1) % WORDS;
                output[next] |= value >> (64 - rem);
            }
            source += 1;
        }
        Self { words: output }
    }

    /// Majority bundle; exact ties resolve to zero. Empty input returns None.
    pub fn majority_bundle(inputs: &[Self]) -> Option<Self> {
        if inputs.is_empty() { return None; }
        let mut output = [0u64; WORDS];
        let mut word = 0usize;
        while word < WORDS {
            let mut bit = 0u32;
            while bit < 64 {
                let mask = 1u64 << bit;
                let mut ones = 0usize;
                let mut n = 0usize;
                while n < inputs.len() {
                    if inputs[n].words[word] & mask != 0 { ones += 1; }
                    n += 1;
                }
                if ones > inputs.len() / 2 { output[word] |= mask; }
                bit += 1;
            }
            word += 1;
        }
        Some(Self { words: output })
    }

    pub const fn payload_bytes() -> Option<usize> {
        WORDS.checked_mul(core::mem::size_of::<u64>())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryError { ZeroCapacity, ClassOutOfRange, CapacityFull }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertOutcome { Inserted, Replaced }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Classification {
    pub class_index: usize,
    pub exemplar_index: usize,
    pub hamming_distance: usize,
    pub similarity_q15: i16,
}

/// Statically allocated nearest-exemplar memory. Overflow is explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedHdcMemory<const CLASSES: usize, const SLOTS: usize, const WORDS: usize> {
    exemplars: [[PackedHv<WORDS>; SLOTS]; CLASSES],
    class_counts: [usize; CLASSES],
    next_replace: [usize; CLASSES],
}

impl<const CLASSES: usize, const SLOTS: usize, const WORDS: usize>
    PackedHdcMemory<CLASSES, SLOTS, WORDS>
{
    pub const fn new() -> Result<Self, MemoryError> {
        if CLASSES == 0 || SLOTS == 0 || WORDS == 0 {
            return Err(MemoryError::ZeroCapacity);
        }
        Ok(Self {
            exemplars: [[PackedHv::zero(); SLOTS]; CLASSES],
            class_counts: [0; CLASSES],
            next_replace: [0; CLASSES],
        })
    }

    pub const fn class_count(&self, class_index: usize) -> Option<usize> {
        if class_index >= CLASSES { None } else { Some(self.class_counts[class_index]) }
    }

    pub const fn total_exemplars(&self) -> usize {
        let mut total = 0usize;
        let mut i = 0usize;
        while i < CLASSES { total += self.class_counts[i]; i += 1; }
        total
    }

    pub fn insert(
        &mut self,
        class_index: usize,
        exemplar: PackedHv<WORDS>,
    ) -> Result<InsertOutcome, MemoryError> {
        if class_index >= CLASSES { return Err(MemoryError::ClassOutOfRange); }
        let count = self.class_counts[class_index];
        if count >= SLOTS { return Err(MemoryError::CapacityFull); }
        self.exemplars[class_index][count] = exemplar;
        self.class_counts[class_index] = count + 1;
        Ok(InsertOutcome::Inserted)
    }

    /// Explicit per-class round-robin replacement when capacity is full.
    pub fn insert_or_replace_oldest(
        &mut self,
        class_index: usize,
        exemplar: PackedHv<WORDS>,
    ) -> Result<InsertOutcome, MemoryError> {
        if class_index >= CLASSES { return Err(MemoryError::ClassOutOfRange); }
        let count = self.class_counts[class_index];
        if count < SLOTS {
            self.exemplars[class_index][count] = exemplar;
            self.class_counts[class_index] = count + 1;
            return Ok(InsertOutcome::Inserted);
        }
        let slot = self.next_replace[class_index];
        self.exemplars[class_index][slot] = exemplar;
        self.next_replace[class_index] = (slot + 1) % SLOTS;
        Ok(InsertOutcome::Replaced)
    }

    /// Exact ties keep the first slot in the lowest class index.
    pub fn predict(&self, query: &PackedHv<WORDS>) -> Option<Classification> {
        let mut best: Option<Classification> = None;
        let mut class = 0usize;
        while class < CLASSES {
            let mut slot = 0usize;
            while slot < self.class_counts[class] {
                let exemplar = &self.exemplars[class][slot];
                let distance = query.hamming_distance(exemplar);
                if best.is_none_or(|current| distance < current.hamming_distance) {
                    best = Some(Classification {
                        class_index: class,
                        exemplar_index: slot,
                        hamming_distance: distance,
                        similarity_q15: query.similarity_q15(exemplar),
                    });
                }
                slot += 1;
            }
            class += 1;
        }
        best
    }

    /// Exemplar words plus the two per-class usize metadata arrays.
    pub const fn payload_bytes() -> Option<usize> {
        let vector_bytes = match WORDS.checked_mul(core::mem::size_of::<u64>()) {
            Some(v) => v,
            None => return None,
        };
        let entries = match CLASSES.checked_mul(SLOTS) {
            Some(v) => v,
            None => return None,
        };
        let data = match entries.checked_mul(vector_bytes) {
            Some(v) => v,
            None => return None,
        };
        let metadata_words = match CLASSES.checked_mul(2) {
            Some(v) => v,
            None => return None,
        };
        let metadata = match metadata_words.checked_mul(core::mem::size_of::<usize>()) {
            Some(v) => v,
            None => return None,
        };
        data.checked_add(metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xor_binding_is_an_involution() {
        let a = PackedHv::<2>::from_words([0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210]);
        let b = PackedHv::<2>::from_words([0xffff_0000_ffff_0000, 0xaaaa_5555_aaaa_5555]);
        assert_eq!(a.bind(&b).bind(&b), a);
        assert_eq!(a.bind(&a), PackedHv::zero());
    }

    #[test]
    fn hamming_distance_and_similarity_are_exact() {
        let zero = PackedHv::<1>::zero();
        let all = PackedHv::<1>::from_words([u64::MAX]);
        let half = PackedHv::<1>::from_words([u64::MAX >> 1]);
        assert_eq!(zero.hamming_distance(&all), 64);
        assert_eq!(all.hamming_distance(&zero), 64);
        assert_eq!(zero.hamming_distance(&half), 63);
        assert_eq!(zero.similarity_q15(&all), -32_767);
        assert_eq!(zero.similarity_q15(&zero), 32_767);
    }

    #[test]
    fn permutation_wraps_across_word_boundary() {
        let source = PackedHv::<2>::from_words([1, 0]);
        assert_eq!(source.permute_left(64).words(), &[0, 1]);
        assert_eq!(source.permute_left(64).permute_left(64), source);
        assert_eq!(source.permute_left(128), source);
    }

    #[test]
    fn majority_bundle_uses_tie_to_zero() {
        let a = PackedHv::<1>::from_words([0b01]);
        let b = PackedHv::<1>::from_words([0b11]);
        let c = PackedHv::<1>::from_words([0b10]);
        assert_eq!(PackedHv::majority_bundle(&[a, b, c]).unwrap().words(), &[0b11]);
        assert_eq!(PackedHv::majority_bundle(&[a, c]).unwrap().words(), &[0]);
        assert!(PackedHv::<1>::majority_bundle(&[]).is_none());
    }

    #[test]
    fn packed_vector_has_exact_payload_size() {
        assert_eq!(PackedHv::<256>::payload_bytes(), Some(2048));
    }

    #[test]
    fn zero_word_permutation_is_defined() {
        let zero = PackedHv::<0>::zero();
        assert_eq!(zero.permute_left(5), zero);
        assert_eq!(zero.similarity_q15(&zero), 0);
    }

    #[test]
    fn memory_capacity_and_replacement_are_explicit() {
        let mut memory = PackedHdcMemory::<2, 2, 1>::new().unwrap();
        let first = PackedHv::<1>::from_words([1]);
        let second = PackedHv::<1>::from_words([3]);
        let third = PackedHv::<1>::from_words([7]);
        assert_eq!(memory.insert(0, first), Ok(InsertOutcome::Inserted));
        assert_eq!(memory.insert(0, second), Ok(InsertOutcome::Inserted));
        assert_eq!(memory.insert(0, third), Err(MemoryError::CapacityFull));
        assert_eq!(memory.insert(2, third), Err(MemoryError::ClassOutOfRange));
        assert_eq!(memory.insert_or_replace_oldest(0, third), Ok(InsertOutcome::Replaced));
        assert_eq!(memory.class_count(0), Some(2));
        assert_eq!(memory.total_exemplars(), 2);
    }

    #[test]
    fn round_robin_replacement_advances_one_slot_per_overflow() {
        let mut memory = PackedHdcMemory::<1, 2, 1>::new().unwrap();
        let first = PackedHv::<1>::from_words([1]);
        let second = PackedHv::<1>::from_words([3]);
        let third = PackedHv::<1>::from_words([7]);
        let fourth = PackedHv::<1>::from_words([15]);
        memory.insert(0, first).unwrap();
        memory.insert(0, second).unwrap();
        assert_eq!(memory.insert_or_replace_oldest(0, third), Ok(InsertOutcome::Replaced));
        assert_eq!(memory.insert_or_replace_oldest(0, fourth), Ok(InsertOutcome::Replaced));
        assert_eq!(memory.predict(&fourth).unwrap().exemplar_index, 1);
        assert_eq!(memory.predict(&third).unwrap().exemplar_index, 0);
    }

    #[test]
    fn large_static_memory_payload_is_exact() {
        // 4 * 16 exemplars * 2,048 bytes/vector, plus two per-class usize arrays.
        let expected = 131_072 + 8 * core::mem::size_of::<usize>();
        assert_eq!(PackedHdcMemory::<4, 16, 256>::payload_bytes(), Some(expected));
    }

    #[test]
    fn nearest_exemplar_prediction_is_deterministic() {
        let mut memory = PackedHdcMemory::<2, 2, 1>::new().unwrap();
        let left = PackedHv::<1>::from_words([0]);
        let right = PackedHv::<1>::from_words([u64::MAX]);
        memory.insert(0, left).unwrap();
        memory.insert(1, right).unwrap();
        let prediction = memory.predict(&PackedHv::<1>::from_words([0b11])).unwrap();
        assert_eq!(prediction.class_index, 0);
        assert_eq!(prediction.hamming_distance, 2);
    }

    #[test]
    fn zero_capacity_is_rejected() {
        assert_eq!(PackedHdcMemory::<0, 1, 1>::new(), Err(MemoryError::ZeroCapacity));
        assert_eq!(PackedHdcMemory::<1, 0, 1>::new(), Err(MemoryError::ZeroCapacity));
        assert_eq!(PackedHdcMemory::<1, 1, 0>::new(), Err(MemoryError::ZeroCapacity));
    }
}
