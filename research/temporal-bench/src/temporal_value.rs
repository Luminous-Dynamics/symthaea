//! Candidate-side temporal-value memory allocation controls.
//!
//! This module consumes only candidate-visible estimates and metadata. It has
//! no dependency on benchmark oracles, labels, evaluator seeds, or future
//! outcomes. The temporal-value score is a fixed-point heuristic, not an
//! exact knapsack solver and not evidence that the estimate is calibrated.

use std::collections::BTreeSet;
use std::fmt;

const PPM: u128 = 1_000_000;
const SCORE_SCALE: u128 = 1_000_000_000_000_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionPolicy {
    /// Estimated future wrong-belief exposure avoided per byte-microsecond.
    TemporalValuePerByteSecond,
    /// Most recently accessed item first.
    Lru,
    /// Most frequently accessed item first.
    Lfu,
    /// Retain items younger than the declared fixed TTL; newest first.
    FixedTtl { ttl_us: u64 },
    /// Highest candidate-estimated novelty first.
    NoveltyFirst,
    /// Highest candidate-estimated confidence first.
    ConfidenceFirst,
}

/// Candidate-visible information only. The value estimate is a prediction,
/// not an evaluator target or a hidden-oracle measurement.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TemporalMemoryCandidate {
    pub item_id: u64,
    pub bytes: u64,
    /// Candidate-estimated future wrong-belief exposure avoided, in microseconds.
    pub predicted_ted_reduction_us: u64,
    /// Fixed-point probabilities in parts per million.
    pub confidence_ppm: u32,
    pub relevance_ppm: u32,
    pub freshness_ppm: u32,
    pub novelty_ppm: u32,
    /// Candidate-observed timestamps in the task's declared clock units.
    pub last_access_us: u64,
    pub last_update_us: u64,
    /// Candidate-estimated useful retention horizon, in microseconds.
    pub expected_retention_us: u64,
    /// Hard validity boundary. An expired item is never selected by any policy.
    pub valid_until_us: Option<u64>,
    pub access_count: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RankedCandidate {
    pub item_id: u64,
    pub rank_score: u128,
    pub eligible: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetentionPlan {
    /// Selected IDs in policy priority order.
    pub selected_ids: Vec<u64>,
    /// All non-selected IDs in ascending ID order.
    pub evicted_ids: Vec<u64>,
    pub ranked: Vec<RankedCandidate>,
    pub bytes_used: u64,
    pub capacity_bytes: u64,
    /// Provenance digest for the candidate-side estimator that supplied
    /// predicted_ted_reduction_us. It is not an evaluator signature.
    pub estimate_model_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RetentionError {
    DuplicateItemId { item_id: u64 },
    ZeroItemSize { item_id: u64 },
    ZeroExpectedRetention { item_id: u64 },
    TimestampInFuture { item_id: u64 },
    InvalidProbability { item_id: u64, field: &'static str, value: u32 },
    ZeroTtl,
    ArithmeticOverflow,
}

impl fmt::Display for RetentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateItemId { item_id } => write!(f, "duplicate memory item ID {item_id}"),
            Self::ZeroItemSize { item_id } => write!(f, "memory item {item_id} has zero bytes"),
            Self::ZeroExpectedRetention { item_id } => {
                write!(f, "memory item {item_id} has zero expected retention")
            }
            Self::TimestampInFuture { item_id } => {
                write!(f, "memory item {item_id} has a timestamp later than now")
            }
            Self::InvalidProbability { item_id, field, value } => write!(
                f,
                "memory item {item_id} has invalid {field}={value}; expected <= 1000000"
            ),
            Self::ZeroTtl => write!(f, "fixed TTL must be greater than zero"),
            Self::ArithmeticOverflow => write!(f, "temporal-value score arithmetic overflowed"),
        }
    }
}

impl std::error::Error for RetentionError {}

/// Select items under a hard byte budget using one of the preregistered
/// candidate-side policies. All policies receive the same candidate records.
///
/// The temporal-value score is:
///
///   estimated_TED_reduction * confidence * relevance * freshness
///   ------------------------------------------------------------
///                 item_bytes * expected_retention_time
///
/// Confidence/relevance/freshness are fixed-point PPM values. Retention time
/// is capped at a hard validity boundary. Selection is a deterministic greedy
/// density heuristic; it is not a globally optimal knapsack solution.
pub fn select_retained_items(
    candidates: &[TemporalMemoryCandidate],
    capacity_bytes: u64,
    now_us: u64,
    policy: RetentionPolicy,
    estimate_model_sha256: [u8; 32],
) -> Result<RetentionPlan, RetentionError> {
    if matches!(policy, RetentionPolicy::FixedTtl { ttl_us: 0 }) {
        return Err(RetentionError::ZeroTtl);
    }

    let mut seen = BTreeSet::new();
    for candidate in candidates {
        if !seen.insert(candidate.item_id) {
            return Err(RetentionError::DuplicateItemId {
                item_id: candidate.item_id,
            });
        }
        if candidate.bytes == 0 {
            return Err(RetentionError::ZeroItemSize {
                item_id: candidate.item_id,
            });
        }
        if candidate.expected_retention_us == 0 {
            return Err(RetentionError::ZeroExpectedRetention {
                item_id: candidate.item_id,
            });
        }
        if candidate.last_access_us > now_us || candidate.last_update_us > now_us {
            return Err(RetentionError::TimestampInFuture {
                item_id: candidate.item_id,
            });
        }
        for (field, value) in [
            ("confidence_ppm", candidate.confidence_ppm),
            ("relevance_ppm", candidate.relevance_ppm),
            ("freshness_ppm", candidate.freshness_ppm),
            ("novelty_ppm", candidate.novelty_ppm),
        ] {
            if u128::from(value) > PPM {
                return Err(RetentionError::InvalidProbability {
                    item_id: candidate.item_id,
                    field,
                    value,
                });
            }
        }
    }

    let mut ranked = candidates
        .iter()
        .map(|candidate| score_candidate(candidate, now_us, policy))
        .collect::<Result<Vec<_>, _>>()?;

    ranked.sort_by(|left, right| {
        right
            .rank_score
            .cmp(&left.rank_score)
            .then_with(|| left.item_id.cmp(&right.item_id))
    });

    let by_id = candidates
        .iter()
        .map(|candidate| (candidate.item_id, candidate.bytes))
        .collect::<std::collections::BTreeMap<_, _>>();

    let mut selected_ids = Vec::new();
    let mut selected = BTreeSet::new();
    let mut bytes_used = 0_u64;
    for entry in ranked.iter().filter(|entry| entry.eligible) {
        let item_bytes = *by_id
            .get(&entry.item_id)
            .ok_or(RetentionError::ArithmeticOverflow)?;
        if item_bytes <= capacity_bytes.saturating_sub(bytes_used) {
            bytes_used = bytes_used
                .checked_add(item_bytes)
                .ok_or(RetentionError::ArithmeticOverflow)?;
            selected.insert(entry.item_id);
            selected_ids.push(entry.item_id);
        }
    }

    let mut evicted_ids = candidates
        .iter()
        .filter(|candidate| !selected.contains(&candidate.item_id))
        .map(|candidate| candidate.item_id)
        .collect::<Vec<_>>();
    evicted_ids.sort_unstable();

    Ok(RetentionPlan {
        selected_ids,
        evicted_ids,
        ranked,
        bytes_used,
        capacity_bytes,
        estimate_model_sha256,
    })
}

fn score_candidate(
    candidate: &TemporalMemoryCandidate,
    now_us: u64,
    policy: RetentionPolicy,
) -> Result<RankedCandidate, RetentionError> {
    let remaining_validity = candidate
        .valid_until_us
        .map(|deadline| deadline.saturating_sub(now_us));
    let hard_valid = remaining_validity.is_none_or(|remaining| remaining > 0);
    let effective_horizon = remaining_validity
        .map(|remaining| remaining.min(candidate.expected_retention_us))
        .unwrap_or(candidate.expected_retention_us);

    // Return before score arithmetic when a hard validity boundary has
    // expired; otherwise the capped retention horizon would be zero.
    if !hard_valid || effective_horizon == 0 {
        return Ok(RankedCandidate {
            item_id: candidate.item_id,
            rank_score: 0,
            eligible: false,
        });
    }

    let (rank_score, eligible) = match policy {
        RetentionPolicy::TemporalValuePerByteSecond => {
            let mut weighted_value = u128::from(candidate.predicted_ted_reduction_us);
            weighted_value = weighted_value
                .checked_mul(u128::from(candidate.confidence_ppm))
                .ok_or(RetentionError::ArithmeticOverflow)?
                / PPM;
            weighted_value = weighted_value
                .checked_mul(u128::from(candidate.relevance_ppm))
                .ok_or(RetentionError::ArithmeticOverflow)?
                / PPM;
            weighted_value = weighted_value
                .checked_mul(u128::from(candidate.freshness_ppm))
                .ok_or(RetentionError::ArithmeticOverflow)?
                / PPM;

            let denominator = u128::from(candidate.bytes)
                .checked_mul(u128::from(effective_horizon))
                .ok_or(RetentionError::ArithmeticOverflow)?;
            let scaled = weighted_value
                .checked_mul(SCORE_SCALE)
                .ok_or(RetentionError::ArithmeticOverflow)?
                / denominator;
            (scaled, hard_valid && weighted_value > 0)
        }
        RetentionPolicy::Lru => (u128::from(candidate.last_access_us), hard_valid),
        RetentionPolicy::Lfu => (u128::from(candidate.access_count), hard_valid),
        RetentionPolicy::FixedTtl { ttl_us } => {
            let age = now_us.saturating_sub(candidate.last_update_us);
            let eligible = hard_valid && age < ttl_us;
            let remaining_ttl = ttl_us.saturating_sub(age);
            let score = u128::from(remaining_ttl)
                .checked_mul(PPM)
                .ok_or(RetentionError::ArithmeticOverflow)?
                / u128::from(ttl_us);
            (score, eligible)
        }
        RetentionPolicy::NoveltyFirst => (u128::from(candidate.novelty_ppm), hard_valid),
        RetentionPolicy::ConfidenceFirst => {
            (u128::from(candidate.confidence_ppm), hard_valid)
        }
    };

    Ok(RankedCandidate {
        item_id: candidate.item_id,
        rank_score,
        eligible,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(item_id: u64, bytes: u64, value_us: u64) -> TemporalMemoryCandidate {
        TemporalMemoryCandidate {
            item_id,
            bytes,
            predicted_ted_reduction_us: value_us,
            confidence_ppm: 1_000_000,
            relevance_ppm: 1_000_000,
            freshness_ppm: 1_000_000,
            novelty_ppm: 500_000,
            last_access_us: item_id * 10,
            last_update_us: item_id * 10,
            expected_retention_us: 1_000_000,
            valid_until_us: None,
            access_count: item_id,
        }
    }

    #[test]
    fn temporal_value_prefers_highest_estimated_debt_reduction_per_byte_time() {
        let items = [candidate(1, 100, 1_000_000), candidate(2, 50, 100_000)];
        let plan = select_retained_items(
            &items,
            100,
            1_000_000,
            RetentionPolicy::TemporalValuePerByteSecond,
            [7; 32],
        )
        .unwrap();
        assert_eq!(plan.selected_ids, vec![1]);
        assert_eq!(plan.bytes_used, 100);
        assert_eq!(plan.estimate_model_sha256, [7; 32]);
    }

    #[test]
    fn confidence_relevance_and_freshness_discount_predicted_value() {
        let mut high = candidate(1, 10, 1_000_000);
        let mut low = candidate(2, 10, 1_000_000);
        low.confidence_ppm = 100_000;
        low.relevance_ppm = 100_000;
        low.freshness_ppm = 100_000;
        let plan = select_retained_items(
            &[low, high],
            10,
            1_000_000,
            RetentionPolicy::TemporalValuePerByteSecond,
            [1; 32],
        )
        .unwrap();
        assert_eq!(plan.selected_ids, vec![1]);
    }

    #[test]
    fn expired_items_are_hard_ineligible_even_with_large_value_estimate() {
        let mut expired = candidate(1, 1, u64::MAX);
        expired.valid_until_us = Some(99);
        let live = candidate(2, 1, 1);
        let plan = select_retained_items(
            &[expired, live],
            1,
            100,
            RetentionPolicy::TemporalValuePerByteSecond,
            [2; 32],
        )
        .unwrap();
        assert_eq!(plan.selected_ids, vec![2]);
        assert!(plan.ranked.iter().any(|row| row.item_id == 1 && !row.eligible));
    }

    #[test]
    fn lru_and_lfu_use_their_declared_observable_statistics() {
        let mut recently_used = candidate(1, 1, 1);
        recently_used.last_access_us = 90;
        recently_used.access_count = 1;
        let mut frequently_used = candidate(2, 1, 1);
        frequently_used.last_access_us = 50;
        frequently_used.access_count = 9;
        let items = [recently_used, frequently_used];
        let lru = select_retained_items(&items, 1, 100, RetentionPolicy::Lru, [0; 32]).unwrap();
        let lfu = select_retained_items(&items, 1, 100, RetentionPolicy::Lfu, [0; 32]).unwrap();
        assert_eq!(lru.selected_ids, vec![1]);
        assert_eq!(lfu.selected_ids, vec![2]);
    }

    #[test]
    fn hard_invalidation_overrides_every_retention_policy() {
        let mut expired = candidate(7, 1, u64::MAX);
        expired.valid_until_us = Some(99);
        let policies = [
            RetentionPolicy::TemporalValuePerByteSecond,
            RetentionPolicy::Lru,
            RetentionPolicy::Lfu,
            RetentionPolicy::FixedTtl { ttl_us: 10 },
            RetentionPolicy::NoveltyFirst,
            RetentionPolicy::ConfidenceFirst,
        ];
        for policy in policies {
            let plan = select_retained_items(&[expired.clone()], 1, 100, policy, [0; 32]).unwrap();
            assert!(plan.selected_ids.is_empty());
            assert_eq!(plan.evicted_ids, vec![7]);
        }
    }

    #[test]
    fn fixed_ttl_excludes_expired_items() {
        let mut expired = candidate(1, 1, 100);
        expired.last_update_us = 10;
        let mut live = candidate(2, 1, 1);
        live.last_update_us = 95;
        let plan = select_retained_items(
            &[expired, live],
            1,
            100,
            RetentionPolicy::FixedTtl { ttl_us: 20 },
            [0; 32],
        )
        .unwrap();
        assert_eq!(plan.selected_ids, vec![2]);
    }

    #[test]
    fn equal_scores_use_stable_ascending_item_id_tie_break() {
        let items = [candidate(9, 1, 10), candidate(3, 1, 10)];
        let plan = select_retained_items(
            &items,
            1,
            1_000_000,
            RetentionPolicy::TemporalValuePerByteSecond,
            [0; 32],
        )
        .unwrap();
        assert_eq!(plan.selected_ids, vec![3]);
    }

    #[test]
    fn selection_never_exceeds_budget_and_skips_oversized_items() {
        let items = [candidate(1, 20, 100), candidate(2, 6, 50), candidate(3, 4, 30)];
        let plan = select_retained_items(
            &items,
            10,
            1_000_000,
            RetentionPolicy::TemporalValuePerByteSecond,
            [0; 32],
        )
        .unwrap();
        assert!(plan.bytes_used <= 10);
        assert_eq!(plan.bytes_used, 10);
        assert!(plan.selected_ids.contains(&2));
        assert!(plan.selected_ids.contains(&3));
    }

    #[test]
    fn duplicate_ids_and_future_timestamps_fail_closed() {
        let duplicate = [candidate(1, 1, 1), candidate(1, 2, 1)];
        assert!(matches!(
            select_retained_items(
                &duplicate,
                10,
                100,
                RetentionPolicy::Lru,
                [0; 32]
            ),
            Err(RetentionError::DuplicateItemId { item_id: 1 })
        ));

        let mut future = candidate(2, 1, 1);
        future.last_access_us = 101;
        assert!(matches!(
            select_retained_items(
                &[future],
                10,
                100,
                RetentionPolicy::Lru,
                [0; 32]
            ),
            Err(RetentionError::TimestampInFuture { item_id: 2 })
        ));
    }

    #[test]
    fn invalid_probability_and_zero_ttl_are_rejected() {
        let mut invalid = candidate(1, 1, 1);
        invalid.confidence_ppm = 1_000_001;
        assert!(matches!(
            select_retained_items(
                &[invalid],
                10,
                100,
                RetentionPolicy::Lru,
                [0; 32]
            ),
            Err(RetentionError::InvalidProbability { .. })
        ));
        assert_eq!(
            select_retained_items(
                &[],
                10,
                100,
                RetentionPolicy::FixedTtl { ttl_us: 0 },
                [0; 32]
            ),
            Err(RetentionError::ZeroTtl)
        );
    }

    #[test]
    fn zero_predicted_value_does_not_consume_temporal_value_budget() {
        let no_value = candidate(1, 1, 0);
        let positive = candidate(2, 1, 1);
        let plan = select_retained_items(
            &[no_value, positive],
            1,
            100,
            RetentionPolicy::TemporalValuePerByteSecond,
            [0; 32],
        )
        .unwrap();
        assert_eq!(plan.selected_ids, vec![2]);
    }
}
