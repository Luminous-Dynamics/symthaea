//! Rollback-aware successor indexes for dynamic projected-BCE protected candidates.
//!
//! Each record keeps D4's exact candidate permutation. A hierarchical bitmap
//! tracks active clauses by their current positions; a reverse incidence index
//! updates only records containing a clause whose active state changes.
//! Backtracking changes active bits but deliberately does not undo promotions.
//! This is an algorithmic data structure, not yet a D4 binary refinement proof.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

/// Fail-closed errors from building or updating a protected-candidate index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CandidateIndexError {
    ActiveClauseLength { expected: usize, actual: usize },
    ClauseOutOfRange { clause: usize, clause_count: usize },
    EmptyCandidateSet { record: usize },
    SentinelMismatch { record: usize, expected: usize, actual: usize },
    SentinelStillActive { record: usize, clause: usize },
    InactivePromotionCandidate { record: usize, clause: usize, position: usize },
    DuplicateCandidate { record: usize, clause: usize },
    RecordOutOfRange { record: usize, record_count: usize },
    PositionOutOfRange { record: usize, position: usize, candidate_count: usize },
    InvalidState { record: usize, clause: usize },
    AllocationFailed { entries: usize },
}

impl fmt::Display for CandidateIndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ActiveClauseLength { expected, actual } => {
                write!(f, "active-clause table has length {actual}, expected {expected}")
            }
            Self::ClauseOutOfRange { clause, clause_count } => {
                write!(f, "clause {clause} is outside clause count {clause_count}")
            }
            Self::EmptyCandidateSet { record } => {
                write!(f, "protected record {record} has no candidate sentinel")
            }
            Self::SentinelMismatch { record, expected, actual } => write!(
                f,
                "record {record} is watched by sentinel {expected}, not deactivated clause {actual}"
            ),
            Self::SentinelStillActive { record, clause } => write!(
                f,
                "record {record} sentinel clause {clause} is still active"
            ),
            Self::InactivePromotionCandidate { record, clause, position } => write!(
                f,
                "record {record} cannot promote inactive candidate {clause} at position {position}"
            ),
            Self::DuplicateCandidate { record, clause } => {
                write!(f, "record {record} contains duplicate candidate clause {clause}")
            }
            Self::RecordOutOfRange { record, record_count } => {
                write!(f, "record {record} is outside record count {record_count}")
            }
            Self::PositionOutOfRange { record, position, candidate_count } => write!(
                f,
                "position {position} is outside candidate count {candidate_count} for record {record}"
            ),
            Self::InvalidState { record, clause } => {
                write!(f, "bitmap/reverse-index state is inconsistent at record {record}, clause {clause}")
            }
            Self::AllocationFailed { entries } => {
                write!(f, "unable to reserve {entries} candidate-index entries")
            }
        }
    }
}

impl Error for CandidateIndexError {}

/// Operation counters. These describe index work, not elapsed-time performance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CandidateIndexStats {
    /// Calls to find the first active candidate after the current sentinel.
    pub successor_queries: u64,
    /// Reverse-incidence entries visited on clause active-state changes.
    pub membership_status_updates: u64,
    /// Candidate swap-to-front operations matching D4's permutation.
    pub promotions: u64,
    /// 64-bit words read by successor queries and point lookups.
    pub bitmap_words_read: u64,
    /// 64-bit words whose contents changed.
    pub bitmap_words_updated: u64,
}

/// Observable result of processing one deactivated current sentinel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SuccessorTransition {
    /// The first active candidate after the old sentinel was promoted by swap-to-front.
    Relocated {
        record: usize,
        previous_sentinel: usize,
        new_sentinel: usize,
        new_position: usize,
    },
    /// Every opposing candidate after the deactivated sentinel is inactive.
    /// The caller may now apply the theorem-backed target-clause deletion.
    Exhausted {
        record: usize,
        sentinel: usize,
    },
}

/// A multi-level bitmap. Level 0 stores candidate positions; each upper level
/// marks which words in the level below are non-empty. The final level has at
/// most one word, so successor lookup visits O(log_64 n) words.
#[derive(Clone, Debug, PartialEq, Eq)]
struct HierarchicalBitmap {
    bit_len: usize,
    levels: Vec<Vec<u64>>,
    words_read: u64,
    words_updated: u64,
}

impl HierarchicalBitmap {
    fn new(bit_len: usize) -> Self {
        let mut levels = Vec::new();
        let mut represented_bits = bit_len;
        loop {
            let word_count = represented_bits.saturating_add(63) / 64;
            levels.push(vec![0u64; word_count]);
            if word_count <= 1 {
                break;
            }
            represented_bits = word_count;
        }
        Self {
            bit_len,
            levels,
            words_read: 0,
            words_updated: 0,
        }
    }

    fn get(&mut self, position: usize) -> Option<bool> {
        if position >= self.bit_len {
            return None;
        }
        let word_index = position / 64;
        let bit = position % 64;
        self.words_read = self.words_read.saturating_add(1);
        Some((self.levels[0][word_index] & (1u64 << bit)) != 0)
    }

    fn set(&mut self, position: usize, value: bool) -> bool {
        if position >= self.bit_len {
            return false;
        }
        let word_index = position / 64;
        let mask = 1u64 << (position % 64);
        let before = self.levels[0][word_index];
        let after = if value { before | mask } else { before & !mask };
        if before == after {
            return true;
        }
        self.levels[0][word_index] = after;
        self.words_updated = self.words_updated.saturating_add(1);
        self.propagate_nonempty_change(0, word_index, before == 0, after == 0);
        true
    }

    fn propagate_nonempty_change(
        &mut self,
        changed_level: usize,
        changed_word_index: usize,
        was_empty: bool,
        is_empty: bool,
    ) {
        let mut level = changed_level + 1;
        let mut child_word_index = changed_word_index;
        let mut child_was_empty = was_empty;
        let mut child_is_empty = is_empty;

        while child_was_empty != child_is_empty && level < self.levels.len() {
            let parent_word_index = child_word_index / 64;
            let parent_mask = 1u64 << (child_word_index % 64);
            let before = self.levels[level][parent_word_index];
            let after = if child_is_empty {
                before & !parent_mask
            } else {
                before | parent_mask
            };
            if before == after {
                break;
            }
            self.levels[level][parent_word_index] = after;
            self.words_updated = self.words_updated.saturating_add(1);
            child_was_empty = before == 0;
            child_is_empty = after == 0;
            child_word_index = parent_word_index;
            level += 1;
        }
    }

    fn swap_bits(&mut self, left: usize, right: usize) -> bool {
        if left >= self.bit_len || right >= self.bit_len {
            return false;
        }
        if left == right {
            return true;
        }
        let left_value = match self.get(left) {
            Some(value) => value,
            None => return false,
        };
        let right_value = match self.get(right) {
            Some(value) => value,
            None => return false,
        };
        self.set(left, right_value) && self.set(right, left_value)
    }

    /// Find the least set-bit position at or after the requested position.
    fn find_next_set(&mut self, start: usize) -> Option<usize> {
        if start >= self.bit_len || self.levels.is_empty() {
            return None;
        }
        self.find_next_in_level(0, start)
    }

    fn find_next_in_level(&mut self, level: usize, start: usize) -> Option<usize> {
        let word_count = self.levels.get(level)?.len();
        let word_index = start / 64;
        if word_index >= word_count {
            return None;
        }
        let offset = start % 64;
        self.words_read = self.words_read.saturating_add(1);
        let mask = u64::MAX.checked_shl(offset as u32).unwrap_or(0);
        let word = self.levels[level][word_index] & mask;
        if word != 0 {
            let result = word_index * 64 + word.trailing_zeros() as usize;
            return (result < self.bit_len_for_level(level)).then_some(result);
        }

        if level + 1 >= self.levels.len() {
            return None;
        }
        let next_nonempty_word = self.find_next_in_level(level + 1, word_index + 1)?;
        self.words_read = self.words_read.saturating_add(1);
        let next_word = self.levels[level][next_nonempty_word];
        if next_word == 0 {
            return None;
        }
        let result = next_nonempty_word * 64 + next_word.trailing_zeros() as usize;
        (result < self.bit_len_for_level(level)).then_some(result)
    }

    fn bit_len_for_level(&self, level: usize) -> usize {
        if level == 0 {
            self.bit_len
        } else {
            self.levels[level - 1].len()
        }
    }
}

/// One protected triple's candidate order, inverse positions and active bitmap.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ProtectedCandidateSet {
    candidates: Vec<usize>,
    positions: HashMap<usize, usize>,
    active_positions: HierarchicalBitmap,
}

/// Reverse-indexed successor queries for D4-style protected candidate lists.
///
/// A record's candidate order is never sorted or reconstructed. promote_to_head
/// performs the same swap between index 0 and the first active candidate found
/// after index 0. set_clause_active updates every protected record containing
/// that clause, including after a backtrack. It does not move sentinels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateSuccessorIndex {
    clause_count: usize,
    records: Vec<ProtectedCandidateSet>,
    reverse_incidence: Vec<Vec<usize>>,
    stats: CandidateIndexStats,
}

impl CandidateSuccessorIndex {
    /// Build the index from D4-style candidate lists and current clause activity.
    ///
    /// Candidate lists must contain unique in-range clause IDs. Candidate order
    /// is preserved exactly.
    pub fn new(
        clause_count: usize,
        candidate_lists: Vec<Vec<usize>>,
        active_clauses: &[bool],
    ) -> Result<Self, CandidateIndexError> {
        if active_clauses.len() != clause_count {
            return Err(CandidateIndexError::ActiveClauseLength {
                expected: clause_count,
                actual: active_clauses.len(),
            });
        }

        let mut reverse_incidence = Vec::new();
        reverse_incidence.try_reserve_exact(clause_count).map_err(|_| {
            CandidateIndexError::AllocationFailed { entries: clause_count }
        })?;
        reverse_incidence.resize_with(clause_count, Vec::<usize>::new);

        let mut records = Vec::new();
        records.try_reserve_exact(candidate_lists.len()).map_err(|_| {
            CandidateIndexError::AllocationFailed { entries: candidate_lists.len() }
        })?;

        for (record_id, candidates) in candidate_lists.into_iter().enumerate() {
            let mut positions = HashMap::new();
            positions.try_reserve(candidates.len()).map_err(|_| {
                CandidateIndexError::AllocationFailed { entries: candidates.len() }
            })?;
            let mut active_positions = HierarchicalBitmap::new(candidates.len());
            for (position, &clause) in candidates.iter().enumerate() {
                if clause >= clause_count {
                    return Err(CandidateIndexError::ClauseOutOfRange {
                        clause,
                        clause_count,
                    });
                }
                if positions.insert(clause, position).is_some() {
                    return Err(CandidateIndexError::DuplicateCandidate {
                        record: record_id,
                        clause,
                    });
                }
                reverse_incidence[clause].push(record_id);
                if active_clauses[clause] {
                    active_positions.set(position, true);
                }
            }
            records.push(ProtectedCandidateSet {
                candidates,
                positions,
                active_positions,
            });
        }

        Ok(Self {
            clause_count,
            records,
            reverse_incidence,
            stats: CandidateIndexStats::default(),
        })
    }

    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    pub fn candidates(&self, record: usize) -> Option<&[usize]> {
        self.records.get(record).map(|set| set.candidates.as_slice())
    }

    pub fn current_head(&self, record: usize) -> Option<usize> {
        self.records.get(record)?.candidates.first().copied()
    }

    /// Update all indexed candidate positions for a clause active-state change.
    ///
    /// Call this on every transition and every backtrack reactivation. Repeated
    /// assignment of the same state is safe and does not alter bitmap contents.
    pub fn set_clause_active(
        &mut self,
        clause: usize,
        active: bool,
    ) -> Result<(), CandidateIndexError> {
        if clause >= self.clause_count {
            return Err(CandidateIndexError::ClauseOutOfRange {
                clause,
                clause_count: self.clause_count,
            });
        }
        let CandidateSuccessorIndex {
            records,
            reverse_incidence,
            stats,
            ..
        } = self;
        for &record_id in &reverse_incidence[clause] {
            let position = *records[record_id]
                .positions
                .get(&clause)
                .ok_or(CandidateIndexError::InvalidState {
                    record: record_id,
                    clause,
                })?;
            let set = &mut records[record_id];
            if !set.active_positions.set(position, active) {
                return Err(CandidateIndexError::PositionOutOfRange {
                    record: record_id,
                    position,
                    candidate_count: set.candidates.len(),
                });
            }
            stats.membership_status_updates =
                stats.membership_status_updates.saturating_add(1);
        }
        Ok(())
    }

    /// Return the first active candidate after the current head (position 0).
    ///
    /// This query is valid only after the current sentinel has been deactivated.
    /// The result includes the current array position so the caller can perform
    /// the exact D4 swap-to-front promotion.
    pub fn next_active_after_head(
        &mut self,
        record: usize,
    ) -> Result<Option<(usize, usize)>, CandidateIndexError> {
        let record_count = self.records.len();
        let result = {
            let set = self.records.get_mut(record).ok_or(
                CandidateIndexError::RecordOutOfRange { record, record_count },
            )?;
            let sentinel = set.candidates.first().copied().ok_or(
                CandidateIndexError::EmptyCandidateSet { record },
            )?;
            match set.active_positions.get(0) {
                Some(false) => {}
                Some(true) => {
                    return Err(CandidateIndexError::SentinelStillActive {
                        record,
                        clause: sentinel,
                    });
                }
                None => {
                    return Err(CandidateIndexError::InvalidState {
                        record,
                        clause: sentinel,
                    });
                }
            }
            self.stats.successor_queries = self.stats.successor_queries.saturating_add(1);
            set.active_positions
                .find_next_set(1)
                .map(|position| (set.candidates[position], position))
        };
        Ok(result)
    }

    /// Swap the current inactive sentinel with an active candidate, exactly as D4 does.
    ///
    /// This only changes the candidate permutation. Activity changes are still
    /// driven by set_clause_active, including when a branch is backtracked.
    /// No mutation occurs unless both sentinel and replacement preconditions hold.
    pub fn promote_to_head(
        &mut self,
        record: usize,
        position: usize,
    ) -> Result<(), CandidateIndexError> {
        let record_count = self.records.len();
        let set = self.records.get_mut(record).ok_or(
            CandidateIndexError::RecordOutOfRange { record, record_count },
        )?;
        if position == 0 || position >= set.candidates.len() {
            return Err(CandidateIndexError::PositionOutOfRange {
                record,
                position,
                candidate_count: set.candidates.len(),
            });
        }

        let old_head = set.candidates.first().copied().ok_or(
            CandidateIndexError::EmptyCandidateSet { record },
        )?;
        match set.active_positions.get(0) {
            Some(false) => {}
            Some(true) => {
                return Err(CandidateIndexError::SentinelStillActive {
                    record,
                    clause: old_head,
                });
            }
            None => {
                return Err(CandidateIndexError::InvalidState {
                    record,
                    clause: old_head,
                });
            }
        }

        let new_head = set.candidates[position];
        if set.active_positions.get(position) != Some(true) {
            return Err(CandidateIndexError::InactivePromotionCandidate {
                record,
                clause: new_head,
                position,
            });
        }
        if set.positions.get(&old_head).copied() != Some(0)
            || set.positions.get(&new_head).copied() != Some(position)
        {
            return Err(CandidateIndexError::InvalidState {
                record,
                clause: new_head,
            });
        }

        if !set.active_positions.swap_bits(0, position) {
            return Err(CandidateIndexError::InvalidState { record, clause: new_head });
        }
        set.candidates.swap(0, position);
        set.positions.insert(old_head, position);
        set.positions.insert(new_head, 0);
        self.stats.promotions = self.stats.promotions.saturating_add(1);
        Ok(())
    }

    /// Process one deactivated current sentinel as a single checked transition.
    ///
    /// The caller first updates authoritative activity with set_clause_active.
    /// This method then checks that the deactivated clause is exactly the
    /// protected record's current sentinel, finds the first active successor,
    /// and performs D4's swap-to-front promotion. Exhaustion is returned only
    /// when every candidate after the sentinel is inactive.
    pub fn advance_after_deactivation(
        &mut self,
        record: usize,
        deactivated_clause: usize,
    ) -> Result<SuccessorTransition, CandidateIndexError> {
        let record_count = self.records.len();
        let sentinel = self.records.get(record)
            .ok_or(CandidateIndexError::RecordOutOfRange { record, record_count })?
            .candidates
            .first()
            .copied()
            .ok_or(CandidateIndexError::EmptyCandidateSet { record })?;

        if sentinel != deactivated_clause {
            return Err(CandidateIndexError::SentinelMismatch {
                record,
                expected: sentinel,
                actual: deactivated_clause,
            });
        }

        match self.next_active_after_head(record)? {
            Some((new_sentinel, new_position)) => {
                self.promote_to_head(record, new_position)?;
                Ok(SuccessorTransition::Relocated {
                    record,
                    previous_sentinel: sentinel,
                    new_sentinel,
                    new_position,
                })
            }
            None => Ok(SuccessorTransition::Exhausted { record, sentinel }),
        }
    }

    pub fn stats(&self) -> CandidateIndexStats {
        let mut stats = self.stats;
        for record in &self.records {
            stats.bitmap_words_read = stats
                .bitmap_words_read
                .saturating_add(record.active_positions.words_read);
            stats.bitmap_words_updated = stats
                .bitmap_words_updated
                .saturating_add(record.active_positions.words_updated);
        }
        stats
    }

    /// Fail-closed consistency check against authoritative clause activity.
    pub fn validate(&mut self, active_clauses: &[bool]) -> Result<(), CandidateIndexError> {
        if active_clauses.len() != self.clause_count {
            return Err(CandidateIndexError::ActiveClauseLength {
                expected: self.clause_count,
                actual: active_clauses.len(),
            });
        }
        for (record_id, set) in self.records.iter_mut().enumerate() {
            if set.positions.len() != set.candidates.len() {
                return Err(CandidateIndexError::InvalidState {
                    record: record_id,
                    clause: usize::MAX,
                });
            }
            for (position, &clause) in set.candidates.iter().enumerate() {
                if set.positions.get(&clause).copied() != Some(position)
                    || set.active_positions.get(position) != Some(active_clauses[clause])
                {
                    return Err(CandidateIndexError::InvalidState { record: record_id, clause });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn next_random(state: &mut u64) -> u64 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        (*state).wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    #[test]
    fn hierarchical_bitmap_matches_ordered_set_for_boundary_sizes() {
        let mut seed = 0xE7_030_2026u64;
        for size in [0usize, 1, 2, 63, 64, 65, 127, 128, 129, 4096, 8193] {
            let mut bitmap = HierarchicalBitmap::new(size);
            let mut oracle = BTreeSet::new();
            for step in 0..20_000usize {
                if size > 0 {
                    let position = (next_random(&mut seed) as usize) % size;
                    let value = next_random(&mut seed) & 1 == 1;
                    assert!(bitmap.set(position, value));
                    if value {
                        oracle.insert(position);
                    } else {
                        oracle.remove(&position);
                    }
                }
                let start = if size == 0 {
                    0
                } else {
                    (next_random(&mut seed) as usize) % (size + 1)
                };
                let actual = bitmap.find_next_set(start);
                let expected = oracle.range(start..).next().copied();
                assert_eq!(actual, expected, "size={size}, step={step}, start={start}");
            }
        }
    }

    #[test]
    fn candidate_index_preserves_swap_to_front_and_backtrack_activity() {
        let mut index = CandidateSuccessorIndex::new(
            4,
            vec![vec![0, 1, 2, 3]],
            &[true, true, true, true],
        )
        .unwrap();

        index.set_clause_active(0, false).unwrap();
        assert_eq!(
            index.advance_after_deactivation(0, 0).unwrap(),
            SuccessorTransition::Relocated {
                record: 0,
                previous_sentinel: 0,
                new_sentinel: 1,
                new_position: 1,
            }
        );
        assert_eq!(index.candidates(0), Some(&[1, 0, 2, 3][..]));

        index.set_clause_active(1, false).unwrap();
        assert_eq!(
            index.advance_after_deactivation(0, 1).unwrap(),
            SuccessorTransition::Relocated {
                record: 0,
                previous_sentinel: 1,
                new_sentinel: 2,
                new_position: 2,
            }
        );
        assert_eq!(index.candidates(0), Some(&[2, 0, 1, 3][..]));

        // Backtracking restores active status, not candidate ordering.
        index.set_clause_active(0, true).unwrap();
        index.set_clause_active(1, true).unwrap();
        assert_eq!(index.candidates(0), Some(&[2, 0, 1, 3][..]));

        // Once the persistent sentinel is later deactivated, the reactivated
        // former sentinel is selected from its current non-head position.
        index.set_clause_active(2, false).unwrap();
        assert_eq!(
            index.advance_after_deactivation(0, 2).unwrap(),
            SuccessorTransition::Relocated {
                record: 0,
                previous_sentinel: 2,
                new_sentinel: 0,
                new_position: 1,
            }
        );
        assert_eq!(index.candidates(0), Some(&[0, 2, 1, 3][..]));
        index.validate(&[true, true, false, true]).unwrap();
    }

    #[test]
    fn reverse_incidence_updates_only_records_that_contain_the_clause() {
        let mut index = CandidateSuccessorIndex::new(
            5,
            vec![vec![0, 1], vec![2, 1], vec![3]],
            &[true; 5],
        )
        .unwrap();
        index.set_clause_active(1, false).unwrap();
        // Deactivating a non-sentinel candidate updates membership, but it must
        // not trigger that protected record's successor transition.
        assert!(matches!(
            index.advance_after_deactivation(0, 1),
            Err(CandidateIndexError::SentinelMismatch { .. })
        ));
        assert!(matches!(
            index.advance_after_deactivation(1, 1),
            Err(CandidateIndexError::SentinelMismatch { .. })
        ));

        index.set_clause_active(0, false).unwrap();
        assert_eq!(
            index.advance_after_deactivation(0, 0).unwrap(),
            SuccessorTransition::Exhausted { record: 0, sentinel: 0 }
        );
        index.set_clause_active(2, false).unwrap();
        assert_eq!(
            index.advance_after_deactivation(1, 2).unwrap(),
            SuccessorTransition::Exhausted { record: 1, sentinel: 2 }
        );
        index.validate(&[false, false, false, true, true]).unwrap();
        assert_eq!(index.stats().membership_status_updates, 4);
    }

    #[test]
    fn empty_candidates_have_no_successor() {
        let mut index = CandidateSuccessorIndex::new(0, vec![vec![]], &[]).unwrap();
        assert!(matches!(
            index.next_active_after_head(0),
            Err(CandidateIndexError::EmptyCandidateSet { record: 0 })
        ));
        assert_eq!(index.candidates(0), Some(&[][..]));
    }

    #[test]
    fn malformed_candidate_lists_fail_closed() {
        assert!(matches!(
            CandidateSuccessorIndex::new(2, vec![vec![0, 2]], &[true, true]),
            Err(CandidateIndexError::ClauseOutOfRange { .. })
        ));
        assert!(matches!(
            CandidateSuccessorIndex::new(2, vec![vec![0, 0]], &[true, true]),
            Err(CandidateIndexError::DuplicateCandidate { .. })
        ));
        assert!(matches!(
            CandidateSuccessorIndex::new(2, vec![vec![0]], &[true]),
            Err(CandidateIndexError::ActiveClauseLength { .. })
        ));
    }

    #[test]
    fn query_work_grows_by_bitmap_levels_not_candidate_prefix_length() {
        let size = 65_537usize;
        let mut active = vec![false; size];
        active[size - 1] = true;
        let mut index = CandidateSuccessorIndex::new(
            size,
            vec![(0..size).collect()],
            &active,
        )
        .unwrap();
        let (candidate, position) = index.next_active_after_head(0).unwrap().unwrap();
        assert_eq!((candidate, position), (size - 1, size - 1));
        let stats = index.stats();
        assert_eq!(stats.successor_queries, 1);
        // A three-level hierarchy reads at most one search word per level
        // and one selected child word on each return. This skips 65,535
        // inactive positions.
        assert!(stats.bitmap_words_read <= 5);
        assert_eq!(stats.promotions, 0);
    }

    #[test]
    fn successor_and_promotion_reject_invalid_sentinel_states() {
        let mut index = CandidateSuccessorIndex::new(
            3,
            vec![vec![0, 1, 2]],
            &[true, true, false],
        )
        .unwrap();

        assert!(matches!(
            index.next_active_after_head(0),
            Err(CandidateIndexError::SentinelStillActive { record: 0, clause: 0 })
        ));
        assert!(matches!(
            index.advance_after_deactivation(0, 1),
            Err(CandidateIndexError::SentinelMismatch {
                record: 0,
                expected: 0,
                actual: 1
            })
        ));

        index.set_clause_active(0, false).unwrap();
        assert!(matches!(
            index.promote_to_head(0, 2),
            Err(CandidateIndexError::InactivePromotionCandidate {
                record: 0,
                clause: 2,
                position: 2
            })
        ));
        assert_eq!(index.candidates(0), Some(&[0, 1, 2][..]));
        index.validate(&[false, true, false]).unwrap();

        assert_eq!(
            index.advance_after_deactivation(0, 0).unwrap(),
            SuccessorTransition::Relocated {
                record: 0,
                previous_sentinel: 0,
                new_sentinel: 1,
                new_position: 1
            }
        );
        index.validate(&[false, true, false]).unwrap();
    }

    #[test]
    fn randomized_updates_match_linear_successor_oracle_after_promotions() {
        let clause_count = 193usize;
        let mut seed = 0xE7_032_2026u64;
        let mut active: Vec<bool> = (0..clause_count)
            .map(|_| next_random(&mut seed) & 1 == 1)
            .collect();
        let mut candidate_lists = Vec::new();
        for record in 0..23usize {
            let mut candidates: Vec<usize> = (0..clause_count)
                .filter(|_| next_random(&mut seed) % 4 != 0)
                .collect();
            // Rotate records differently so reverse incidence is nontrivial.
            let rotate = record % candidates.len().max(1);
            if !candidates.is_empty() {
                candidates.rotate_left(rotate);
            }
            candidate_lists.push(candidates);
        }
        let mut index = CandidateSuccessorIndex::new(
            clause_count,
            candidate_lists.clone(),
            &active,
        )
        .unwrap();

        for step in 0..4_000usize {
            let clause = (next_random(&mut seed) as usize) % clause_count;
            let value = next_random(&mut seed) & 1 == 1;
            let was_active = active[clause];
            active[clause] = value;
            index.set_clause_active(clause, value).unwrap();

            // D4 processes a record only when its current sentinel is the
            // clause whose active state just changed to inactive. A candidate
            // elsewhere in the list becoming inactive only updates membership.
            if was_active && !value {
                for record in 0..candidate_lists.len() {
                    if index.current_head(record) != Some(clause) {
                        continue;
                    }
                    let current = index.candidates(record).unwrap().to_vec();
                    let expected_position = current
                        .iter()
                        .enumerate()
                        .skip(1)
                        .find_map(|(position, &candidate)| active[candidate].then_some(position));
                    let expected = match expected_position {
                        Some(position) => SuccessorTransition::Relocated {
                            record,
                            previous_sentinel: clause,
                            new_sentinel: current[position],
                            new_position: position,
                        },
                        None => SuccessorTransition::Exhausted {
                            record,
                            sentinel: clause,
                        },
                    };
                    let actual = index.advance_after_deactivation(record, clause).unwrap();
                    assert_eq!(actual, expected, "step={step}, record={record}");
                }
            }
            index.validate(&active).unwrap();
        }
    }
}
