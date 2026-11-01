//! Deterministic replay baselines for stability/plasticity experiments.
//!
//! Replay policies are intentionally simple references: bounded FIFO and
//! reservoir admission. The caller supplies bounded random draws explicitly,
//! making every trace reproducible and auditable. Draw generation is outside
//! this crate; this code validates ranges but does not certify RNG quality.
//!
//! Admission and replay accounting are preflighted before mutation. Logical
//! replay payload bytes are not physical device I/O.

use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayPolicy { Fifo, Reservoir }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplayItem {
    pub item_id: u64,
    pub task_id: u64,
    pub sequence_index: u64,
    pub payload_sha256: [u8; 32],
    pub payload_bytes: u64,
    pub replay_work_units: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Admission {
    Retained,
    Replaced { evicted_item_id: u64 },
    NotRetained,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReplayStats {
    pub seen_items: u64,
    pub retained_items: usize,
    pub retained_bytes: u64,
    pub retained_byte_time_us: u128,
    pub evictions: u64,
    pub not_retained: u64,
    pub replay_selections: u64,
    /// Logical selected payload bytes, not physical device I/O.
    pub replay_payload_bytes: u64,
    pub replay_work_units: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplayError {
    ZeroItemId,
    ZeroPayloadDigest,
    ZeroPayloadBytes,
    NonContiguousSequence { expected: u64, actual: u64 },
    NonMonotonicTime { previous: u64, actual: u64 },
    DuplicateActiveItem { item_id: u64 },
    RandomDrawRequired,
    UnexpectedRandomDraw,
    RandomDrawOutOfRange { draw: u64, upper_exclusive: u64 },
    WrongRandomDrawCount { expected: usize, actual: usize },
    SampleExceedsRetained { requested: usize, retained: usize },
    ReplayWorkBudgetExceeded { required: u64, budget: u64 },
    ReplayPayloadBudgetExceeded { required: u64, budget: u64 },
    ResourceArithmeticOverflow,
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroItemId => write!(f, "replay item ID must be nonzero"),
            Self::ZeroPayloadDigest => write!(f, "replay payload digest must be nonzero"),
            Self::ZeroPayloadBytes => write!(f, "replay payload size must be nonzero"),
            Self::NonContiguousSequence { expected, actual } => write!(f, "expected sequence index {expected}, received {actual}"),
            Self::NonMonotonicTime { previous, actual } => write!(f, "replay timestamp moved backwards from {previous} to {actual}"),
            Self::DuplicateActiveItem { item_id } => write!(f, "replay item {item_id} is already retained"),
            Self::RandomDrawRequired => write!(f, "reservoir admission requires a bounded random draw"),
            Self::UnexpectedRandomDraw => write!(f, "this replay operation does not consume a random draw"),
            Self::RandomDrawOutOfRange { draw, upper_exclusive } => write!(f, "random draw {draw} is outside [0, {upper_exclusive})"),
            Self::WrongRandomDrawCount { expected, actual } => write!(f, "sampling expected {expected} random draws, received {actual}"),
            Self::SampleExceedsRetained { requested, retained } => write!(f, "requested {requested} replay items but only {retained} are retained"),
            Self::ReplayWorkBudgetExceeded { required, budget } => write!(f, "replay work {required} exceeds budget {budget}"),
            Self::ReplayPayloadBudgetExceeded { required, budget } => write!(f, "replay payload bytes {required} exceed budget {budget}"),
            Self::ResourceArithmeticOverflow => write!(f, "replay resource accounting overflowed"),
        }
    }
}
impl std::error::Error for ReplayError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayBuffer {
    policy: ReplayPolicy,
    capacity: usize,
    entries: Vec<ReplayItem>,
    stats: ReplayStats,
    last_event_time_us: Option<u64>,
}

impl ReplayBuffer {
    pub fn new(policy: ReplayPolicy, capacity: usize) -> Self {
        Self { policy, capacity, entries: Vec::new(), stats: ReplayStats::default(), last_event_time_us: None }
    }
    pub fn policy(&self) -> ReplayPolicy { self.policy }
    pub fn capacity(&self) -> usize { self.capacity }
    pub fn retained(&self) -> &[ReplayItem] { &self.entries }
    pub fn stats(&self) -> ReplayStats {
        ReplayStats { retained_items: self.entries.len(), ..self.stats }
    }

    /// Admit one observation. Reservoir draws must be unbiased integers in
    /// [0, new_seen_count); this implementation never uses modulo reduction.
    pub fn observe(
        &mut self,
        item: ReplayItem,
        now_us: u64,
        random_draw: Option<u64>,
    ) -> Result<Admission, ReplayError> {
        if item.item_id == 0 { return Err(ReplayError::ZeroItemId); }
        if item.payload_sha256 == [0; 32] { return Err(ReplayError::ZeroPayloadDigest); }
        if item.payload_bytes == 0 { return Err(ReplayError::ZeroPayloadBytes); }
        if item.sequence_index != self.stats.seen_items {
            return Err(ReplayError::NonContiguousSequence { expected: self.stats.seen_items, actual: item.sequence_index });
        }
        if let Some(previous) = self.last_event_time_us {
            if now_us < previous { return Err(ReplayError::NonMonotonicTime { previous, actual: now_us }); }
        }
        if self.entries.iter().any(|existing| existing.item_id == item.item_id) {
            return Err(ReplayError::DuplicateActiveItem { item_id: item.item_id });
        }

        let next_seen = self.stats.seen_items.checked_add(1).ok_or(ReplayError::ResourceArithmeticOverflow)?;
        let (replace_index, admission) = match self.policy {
            ReplayPolicy::Fifo => {
                if random_draw.is_some() { return Err(ReplayError::UnexpectedRandomDraw); }
                if self.capacity == 0 { (None, Admission::NotRetained) }
                else if self.entries.len() < self.capacity { (None, Admission::Retained) }
                else { (Some(0), Admission::Replaced { evicted_item_id: self.entries[0].item_id }) }
            }
            ReplayPolicy::Reservoir => {
                if self.capacity == 0 || self.entries.len() < self.capacity {
                    if random_draw.is_some() { return Err(ReplayError::UnexpectedRandomDraw); }
                    (None, if self.capacity == 0 { Admission::NotRetained } else { Admission::Retained })
                } else {
                    let draw = random_draw.ok_or(ReplayError::RandomDrawRequired)?;
                    if draw >= next_seen {
                        return Err(ReplayError::RandomDrawOutOfRange { draw, upper_exclusive: next_seen });
                    }
                    if draw < u64::try_from(self.capacity).map_err(|_| ReplayError::ResourceArithmeticOverflow)? {
                        let index = usize::try_from(draw).map_err(|_| ReplayError::ResourceArithmeticOverflow)?;
                        (Some(index), Admission::Replaced { evicted_item_id: self.entries[index].item_id })
                    } else { (None, Admission::NotRetained) }
                }
            }
        };

        let previous_time = self.last_event_time_us.unwrap_or(now_us);
        let elapsed = now_us.checked_sub(previous_time).ok_or(ReplayError::NonMonotonicTime { previous: previous_time, actual: now_us })?;
        let accrued = u128::from(self.stats.retained_bytes).checked_mul(u128::from(elapsed))
            .ok_or(ReplayError::ResourceArithmeticOverflow)?;
        let next_byte_time = self.stats.retained_byte_time_us.checked_add(accrued)
            .ok_or(ReplayError::ResourceArithmeticOverflow)?;
        let evicted_bytes = replace_index.map(|index| self.entries[index].payload_bytes).unwrap_or(0);
        let retained_after_eviction = self.stats.retained_bytes.checked_sub(evicted_bytes)
            .ok_or(ReplayError::ResourceArithmeticOverflow)?;
        let next_retained_bytes = if matches!(admission, Admission::Retained | Admission::Replaced { .. }) {
            retained_after_eviction.checked_add(item.payload_bytes).ok_or(ReplayError::ResourceArithmeticOverflow)?
        } else { retained_after_eviction };
        let next_evictions = if replace_index.is_some() {
            self.stats.evictions.checked_add(1).ok_or(ReplayError::ResourceArithmeticOverflow)?
        } else { self.stats.evictions };
        let next_not_retained = if admission == Admission::NotRetained {
            self.stats.not_retained.checked_add(1).ok_or(ReplayError::ResourceArithmeticOverflow)?
        } else { self.stats.not_retained };

        // Every fallible check is complete before the entry vector/counters mutate.
        if let Some(index) = replace_index { self.entries.remove(index); }
        if matches!(admission, Admission::Retained | Admission::Replaced { .. }) { self.entries.push(item); }
        self.stats.seen_items = next_seen;
        self.stats.retained_bytes = next_retained_bytes;
        self.stats.retained_byte_time_us = next_byte_time;
        self.stats.evictions = next_evictions;
        self.stats.not_retained = next_not_retained;
        self.stats.retained_items = self.entries.len();
        self.last_event_time_us = Some(now_us);
        Ok(admission)
    }

    pub fn finish(&mut self, end_time_us: u64) -> Result<ReplayStats, ReplayError> {
        let previous = self.last_event_time_us.unwrap_or(end_time_us);
        if end_time_us < previous {
            return Err(ReplayError::NonMonotonicTime { previous, actual: end_time_us });
        }
        let elapsed = end_time_us - previous;
        let accrued = u128::from(self.stats.retained_bytes).checked_mul(u128::from(elapsed))
            .ok_or(ReplayError::ResourceArithmeticOverflow)?;
        let next_byte_time = self.stats.retained_byte_time_us.checked_add(accrued)
            .ok_or(ReplayError::ResourceArithmeticOverflow)?;
        self.stats.retained_byte_time_us = next_byte_time;
        self.last_event_time_us = Some(end_time_us);
        Ok(self.stats())
    }

    /// Partial Fisher-Yates without-replacement sampling. Both cumulative
    /// replay-work and logical payload-byte limits are checked before commit.
    pub fn sample_without_replacement(
        &mut self,
        requested: usize,
        bounded_draws: &[u64],
        maximum_work_units: u64,
        maximum_replay_payload_bytes: u64,
    ) -> Result<Vec<ReplayItem>, ReplayError> {
        if bounded_draws.len() != requested {
            return Err(ReplayError::WrongRandomDrawCount { expected: requested, actual: bounded_draws.len() });
        }
        if requested > self.entries.len() {
            return Err(ReplayError::SampleExceedsRetained { requested, retained: self.entries.len() });
        }
        let mut indices: Vec<usize> = (0..self.entries.len()).collect();
        for (i, draw) in bounded_draws.iter().enumerate() {
            let upper = self.entries.len() - i;
            let upper_exclusive = u64::try_from(upper).map_err(|_| ReplayError::ResourceArithmeticOverflow)?;
            if *draw >= upper_exclusive {
                return Err(ReplayError::RandomDrawOutOfRange { draw: *draw, upper_exclusive });
            }
            let offset = usize::try_from(*draw).map_err(|_| ReplayError::ResourceArithmeticOverflow)?;
            indices.swap(i, i + offset);
        }
        let mut work = 0_u64;
        let mut payload_bytes = 0_u64;
        let mut selected = Vec::with_capacity(requested);
        for index in indices.into_iter().take(requested) {
            let item = self.entries[index];
            work = work.checked_add(item.replay_work_units).ok_or(ReplayError::ResourceArithmeticOverflow)?;
            payload_bytes = payload_bytes.checked_add(item.payload_bytes).ok_or(ReplayError::ResourceArithmeticOverflow)?;
            selected.push(item);
        }
        let next_work = self.stats.replay_work_units.checked_add(work).ok_or(ReplayError::ResourceArithmeticOverflow)?;
        if next_work > maximum_work_units {
            return Err(ReplayError::ReplayWorkBudgetExceeded { required: next_work, budget: maximum_work_units });
        }
        let next_payload_bytes = self.stats.replay_payload_bytes.checked_add(payload_bytes).ok_or(ReplayError::ResourceArithmeticOverflow)?;
        if next_payload_bytes > maximum_replay_payload_bytes {
            return Err(ReplayError::ReplayPayloadBudgetExceeded { required: next_payload_bytes, budget: maximum_replay_payload_bytes });
        }
        let next_selections = self.stats.replay_selections.checked_add(u64::try_from(requested).map_err(|_| ReplayError::ResourceArithmeticOverflow)?)
            .ok_or(ReplayError::ResourceArithmeticOverflow)?;

        self.stats.replay_work_units = next_work;
        self.stats.replay_payload_bytes = next_payload_bytes;
        self.stats.replay_selections = next_selections;
        Ok(selected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(sequence_index: u64, item_id: u64, bytes: u64, work: u64) -> ReplayItem {
        let mut payload_sha256 = [0_u8; 32];
        payload_sha256[..8].copy_from_slice(&item_id.to_le_bytes());
        ReplayItem { item_id, task_id: sequence_index % 3, sequence_index, payload_sha256, payload_bytes: bytes, replay_work_units: work }
    }

    #[test]
    fn fifo_keeps_newest_capacity_and_counts_evictions() {
        let mut b=ReplayBuffer::new(ReplayPolicy::Fifo,2);
        assert_eq!(b.observe(item(0,10,4,2),0,None),Ok(Admission::Retained));
        assert_eq!(b.observe(item(1,11,5,3),1,None),Ok(Admission::Retained));
        assert_eq!(b.observe(item(2,12,6,4),2,None),Ok(Admission::Replaced{evicted_item_id:10}));
        assert_eq!(b.retained().iter().map(|x|x.item_id).collect::<Vec<_>>(),vec![11,12]);
        assert_eq!(b.stats().retained_bytes,11);
        assert_eq!(b.stats().evictions,1);
    }

    #[test]
    fn reservoir_admission_is_deterministic_for_bounded_draws() {
        let mut b=ReplayBuffer::new(ReplayPolicy::Reservoir,2);
        b.observe(item(0,10,4,1),0,None).unwrap();
        b.observe(item(1,11,5,1),1,None).unwrap();
        assert_eq!(b.observe(item(2,12,6,1),2,Some(1)),Ok(Admission::Replaced{evicted_item_id:11}));
        assert_eq!(b.observe(item(3,13,7,1),3,Some(3)),Ok(Admission::NotRetained));
        assert_eq!(b.retained().iter().map(|x|x.item_id).collect::<Vec<_>>(),vec![10,12]);
        assert_eq!(b.stats().seen_items,4);
        assert_eq!(b.stats().not_retained,1);
    }

    #[test]
    fn reservoir_random_draw_range_is_checked_transactionally() {
        let mut b=ReplayBuffer::new(ReplayPolicy::Reservoir,1);
        b.observe(item(0,10,4,1),0,None).unwrap();
        assert_eq!(b.observe(item(1,11,4,1),1,Some(2)),Err(ReplayError::RandomDrawOutOfRange{draw:2,upper_exclusive:2}));
        assert_eq!(b.stats().seen_items,1);
        assert_eq!(b.retained()[0].item_id,10);
        assert_eq!(b.observe(item(1,11,4,1),1,Some(1)),Ok(Admission::NotRetained));
    }

    #[test]
    fn zero_capacity_is_a_valid_memory_free_baseline() {
        let mut b=ReplayBuffer::new(ReplayPolicy::Reservoir,0);
        assert_eq!(b.observe(item(0,10,4,1),1,None),Ok(Admission::NotRetained));
        let stats=b.finish(11).unwrap();
        assert_eq!(stats.retained_items,0);
        assert_eq!(stats.retained_bytes,0);
        assert_eq!(stats.retained_byte_time_us,0);
        assert_eq!(stats.not_retained,1);
    }

    #[test]
    fn duplicate_active_item_is_rejected_without_mutation() {
        let mut b=ReplayBuffer::new(ReplayPolicy::Fifo,2);
        b.observe(item(0,10,4,1),0,None).unwrap();
        assert_eq!(b.observe(item(1,10,4,1),1,None),Err(ReplayError::DuplicateActiveItem{item_id:10}));
        assert_eq!(b.stats().seen_items,1);
        assert_eq!(b.stats().retained_byte_time_us,0);
    }

    #[test]
    fn sequence_and_timestamp_must_be_monotonic() {
        let mut b=ReplayBuffer::new(ReplayPolicy::Fifo,3);
        b.observe(item(0,10,4,1),5,None).unwrap();
        assert_eq!(b.observe(item(2,12,4,1),6,None),Err(ReplayError::NonContiguousSequence{expected:1,actual:2}));
        assert_eq!(b.observe(item(1,11,4,1),4,None),Err(ReplayError::NonMonotonicTime{previous:5,actual:4}));
        assert_eq!(b.stats().seen_items,1);
    }

    #[test]
    fn invalid_item_metadata_is_rejected() {
        let mut b=ReplayBuffer::new(ReplayPolicy::Fifo,2);
        let mut x=item(0,10,4,1); x.item_id=0;
        assert_eq!(b.observe(x,0,None),Err(ReplayError::ZeroItemId));
        let mut x=item(0,10,4,1); x.payload_sha256=[0;32];
        assert_eq!(b.observe(x,0,None),Err(ReplayError::ZeroPayloadDigest));
        assert_eq!(b.observe(item(0,10,0,1),0,None),Err(ReplayError::ZeroPayloadBytes));
    }

    #[test]
    fn retained_byte_time_integrates_piecewise_exactly() {
        let mut b=ReplayBuffer::new(ReplayPolicy::Fifo,2);
        b.observe(item(0,10,4,1),10,None).unwrap();
        b.observe(item(1,11,3,1),15,None).unwrap();
        assert_eq!(b.finish(20).unwrap().retained_byte_time_us,55);
    }

    #[test]
    fn sampling_is_without_replacement_and_replay_work_is_charged() {
        let mut b=ReplayBuffer::new(ReplayPolicy::Fifo,3);
        b.observe(item(0,10,4,2),0,None).unwrap();
        b.observe(item(1,11,4,3),0,None).unwrap();
        b.observe(item(2,12,4,5),0,None).unwrap();
        let sampled=b.sample_without_replacement(2,&[2,0],10,8).unwrap();
        assert_eq!(sampled.iter().map(|x|x.item_id).collect::<Vec<_>>(),vec![12,11]);
        assert_eq!(b.stats().replay_selections,2);
        assert_eq!(b.stats().replay_work_units,8);
        assert_eq!(b.stats().replay_payload_bytes,8);
    }

    #[test]
    fn replay_work_budget_failure_does_not_charge_partial_work() {
        let mut b=ReplayBuffer::new(ReplayPolicy::Fifo,2);
        b.observe(item(0,10,4,8),0,None).unwrap();
        b.observe(item(1,11,4,9),0,None).unwrap();
        assert_eq!(b.sample_without_replacement(2,&[0,0],10,100),Err(ReplayError::ReplayWorkBudgetExceeded{required:17,budget:10}));
        assert_eq!(b.stats().replay_selections,0);
        assert_eq!(b.stats().replay_work_units,0);
    }

    #[test]
    fn replay_payload_budget_is_separate_and_transactional() {
        let mut b=ReplayBuffer::new(ReplayPolicy::Fifo,2);
        b.observe(item(0,10,4,1),0,None).unwrap();
        b.observe(item(1,11,5,1),0,None).unwrap();
        assert_eq!(b.sample_without_replacement(2,&[0,0],100,8),Err(ReplayError::ReplayPayloadBudgetExceeded{required:9,budget:8}));
        assert_eq!(b.stats().replay_selections,0);
        assert_eq!(b.stats().replay_work_units,0);
        assert_eq!(b.stats().replay_payload_bytes,0);
    }

    #[test]
    fn sampling_requires_exact_bounded_draw_vector() {
        let mut b=ReplayBuffer::new(ReplayPolicy::Fifo,2);
        b.observe(item(0,10,4,1),0,None).unwrap();
        assert_eq!(b.sample_without_replacement(1,&[],10,100),Err(ReplayError::WrongRandomDrawCount{expected:1,actual:0}));
        assert_eq!(b.sample_without_replacement(1,&[1],10,100),Err(ReplayError::RandomDrawOutOfRange{draw:1,upper_exclusive:1}));
    }

    #[test]
    fn retained_byte_sum_overflow_fails_without_admitting_item() {
        let mut b=ReplayBuffer::new(ReplayPolicy::Fifo,2);
        b.observe(item(0,10,u64::MAX,1),0,None).unwrap();
        assert_eq!(b.observe(item(1,11,1,1),0,None),Err(ReplayError::ResourceArithmeticOverflow));
        assert_eq!(b.stats().seen_items,1);
        assert_eq!(b.retained().len(),1);
    }

    #[test]
    fn fifo_rejects_irrelevant_random_draws() {
        let mut b=ReplayBuffer::new(ReplayPolicy::Fifo,1);
        assert_eq!(b.observe(item(0,10,4,1),0,Some(0)),Err(ReplayError::UnexpectedRandomDraw));
        assert_eq!(b.stats().seen_items,0);
    }
}
