// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Global interval-first selection for exact ContinuousHV top-k refinement.
//!
//! Given one conservative score interval for every eligible record, this
//! module computes the kth-largest lower endpoint and retains every record
//! whose upper endpoint can still reach that threshold. The resulting set is
//! sufficient for exact reranking only if the caller has genuinely established
//! complete corpus coverage and valid, generation-bound intervals.
//!
//! This module checks coverage counts and interval shape; it does not prove
//! that the supplied counts reflect the pinned eligibility snapshot, nor does
//! it validate residual certificates. A caller must perform those checks
//! before passing intervals here. An approximate candidate index may order
//! work, but it cannot be substituted for the complete interval set.

use std::collections::BTreeMap;

use symthaea_core::hdc::continuous_hv_interval::ContinuousHVScoreInterval;

/// One eligible record and its conservative canonical similarity interval.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoreIntervalRecord<Id> {
    /// Stable record identifier used for deterministic output ordering.
    pub id: Id,
    /// Conservative interval for the canonical full-resolution score.
    pub interval: ContinuousHVScoreInterval,
}

/// Counts supplied by the caller for a pinned eligible-corpus snapshot.
///
/// These are consistency counters, not a cryptographic or semantic proof.
/// The caller must bind them to the manifest and eligibility generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntervalCoverageCounts {
    /// Number of eligible records in the pinned snapshot.
    pub eligible_records: usize,
    /// Number of eligible records with a validated interval supplied here.
    pub interval_records: usize,
    /// Number of unopened partitions that lack an admissible aggregate bound.
    pub unbounded_unopened_partitions: usize,
}

/// One exact canonical score and its stable record identifier.
#[derive(Debug, Clone, PartialEq)]
pub struct ExactScoreRecord<Id> {
    /// Stable record identifier.
    pub id: Id,
    /// Canonical finite similarity score in [-1, 1].
    pub score: f32,
}

/// The deterministic refinement set for one larger-is-better top-k query.
///
/// This plan selects records that must receive exact payload scoring. It is not
/// an ExactComplete receipt: the caller must exact-score every refinement ID,
/// rerank under the canonical stable-ID tie policy, and bind completion to the
/// manifest, score contract, certificate generation and eligibility snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct IntervalSelectionPlan<Id> {
    /// kth-largest lower endpoint across the complete eligible set.
    pub threshold: f32,
    /// IDs that must be exact-scored from their original payloads.
    pub refinement_ids: Vec<Id>,
    /// IDs proven unable to enter the exact top-k, assuming valid intervals.
    pub excluded_ids: Vec<Id>,
}

/// Structural errors that prevent interval-first selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntervalSelectionError {
    /// top_k must be in the closed interval [1, eligible_records].
    InvalidTopK {
        /// Requested result count.
        top_k: usize,
        /// Number of eligible records in the input.
        eligible_records: usize,
    },
    /// Supplied intervals do not cover exactly the eligible record population.
    IncompleteCoverage {
        /// Expected eligible record count.
        eligible_records: usize,
        /// Number of records supplied.
        supplied_records: usize,
        /// Number claimed to have intervals.
        interval_records: usize,
        /// Unopened partitions without safe aggregate bounds.
        unbounded_unopened_partitions: usize,
    },
    /// One or more records carry an invalid, non-finite or out-of-range interval.
    InvalidInterval {
        /// Index of the first invalid input record.
        record_index: usize,
    },
    /// An exact result score is non-finite or outside the canonical score range.
    InvalidExactScore {
        /// Index of the first invalid input score.
        record_index: usize,
    },
    /// Stable IDs are not unique within the eligible population.
    DuplicateId {
        /// First occurrence of the duplicate ID.
        first_index: usize,
        /// Later occurrence of the same ID.
        duplicate_index: usize,
    },
}

impl std::fmt::Display for IntervalSelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTopK { top_k, eligible_records } => write!(
                f,
                "top_k {top_k} is invalid for {eligible_records} eligible records"
            ),
            Self::IncompleteCoverage {
                eligible_records,
                supplied_records,
                interval_records,
                unbounded_unopened_partitions,
            } => write!(
                f,
                "interval coverage incomplete: eligible={eligible_records}, supplied={supplied_records}, intervals={interval_records}, unbounded_partitions={unbounded_unopened_partitions}"
            ),
            Self::InvalidInterval { record_index } => {
                write!(f, "record {record_index} has an invalid score interval")
            }
            Self::InvalidExactScore { record_index } => {
                write!(f, "record {record_index} has an invalid exact score")
            }
            Self::DuplicateId { first_index, duplicate_index } => write!(
                f,
                "duplicate record ID at indices {first_index} and {duplicate_index}"
            ),
        }
    }
}

impl std::error::Error for IntervalSelectionError {}

/// Compute the complete refinement set for larger-is-better similarity.
///
/// For intervals [L_i, U_i], this computes theta = kthLargest(L_i), retains
/// every record with U_i >= theta, and excludes only records with U_i < theta.
/// Equality is deliberately retained. Stable IDs break equal-lower-bound
/// ordering deterministically when selecting the scalar threshold, although
/// the threshold value itself is unaffected by that tie break.
///
/// This function does not support approximate/empirical intervals as proof.
/// It requires every eligible record to have a well-formed interval and
/// rejects incomplete record coverage or any unbounded unopened partition.
///
/// # Type-level boundary
///
/// The function returns a selection plan, not an ExactComplete status. The
/// caller remains responsible for validating interval provenance and exact
/// corpus/policy generations, loading and scoring every refinement payload,
/// and stable score-descending / ID-ascending reranking.
pub fn select_refinement_set<Id>(
    records: &[ScoreIntervalRecord<Id>],
    top_k: usize,
    coverage: IntervalCoverageCounts,
) -> Result<IntervalSelectionPlan<Id>, IntervalSelectionError>
where
    Id: Ord + Clone,
{
    let record_count = records.len();
    if coverage.eligible_records != record_count
        || coverage.interval_records != record_count
        || coverage.unbounded_unopened_partitions != 0
    {
        return Err(IntervalSelectionError::IncompleteCoverage {
            eligible_records: coverage.eligible_records,
            supplied_records: record_count,
            interval_records: coverage.interval_records,
            unbounded_unopened_partitions: coverage.unbounded_unopened_partitions,
        });
    }

    if top_k == 0 || top_k > record_count {
        return Err(IntervalSelectionError::InvalidTopK {
            top_k,
            eligible_records: record_count,
        });
    }

    let mut first_index_by_id = BTreeMap::new();
    let mut lower_order: Vec<&ScoreIntervalRecord<Id>> = Vec::with_capacity(record_count);
    for (record_index, record) in records.iter().enumerate() {
        if !record.interval.is_well_formed() {
            return Err(IntervalSelectionError::InvalidInterval { record_index });
        }
        if let Some(first_index) = first_index_by_id.get(&record.id).copied() {
            return Err(IntervalSelectionError::DuplicateId {
                first_index,
                duplicate_index: record_index,
            });
        }
        first_index_by_id.insert(record.id.clone(), record_index);
        lower_order.push(record);
    }

    lower_order.sort_by(|left, right| {
        let lower_order = if left.interval.lower == right.interval.lower {
            std::cmp::Ordering::Equal
        } else {
            right.interval.lower.total_cmp(&left.interval.lower)
        };
        lower_order.then_with(|| left.id.cmp(&right.id))
    });
    let threshold = lower_order[top_k - 1].interval.lower;

    let mut refinement_ids = Vec::new();
    let mut excluded_ids = Vec::new();
    for record in records {
        if record.interval.upper >= threshold {
            refinement_ids.push(record.id.clone());
        } else {
            // Strict comparison is essential: boundary equality stays in the
            // refinement set unless a separate canonical-ID dominance proof
            // can establish exclusion.
            excluded_ids.push(record.id.clone());
        }
    }
    refinement_ids.sort();
    excluded_ids.sort();

    Ok(IntervalSelectionPlan {
        threshold,
        refinement_ids,
        excluded_ids,
    })
}

/// Apply the canonical score-descending / stable-ID-ascending top-k order.
///
/// Numerically equal scores use the ID tiebreak; in particular, +0.0 and -0.0
/// compare as equal scores instead of being ordered by their sign bit. The
/// canonical score profile guarantees finite values in [-1, 1], and this
/// helper rejects violations rather than using a NaN-tolerant comparator.
pub fn stable_exact_top_k<Id>(
    scores: &[ExactScoreRecord<Id>],
    top_k: usize,
) -> Result<Vec<Id>, IntervalSelectionError>
where
    Id: Ord + Clone,
{
    if top_k == 0 || top_k > scores.len() {
        return Err(IntervalSelectionError::InvalidTopK {
            top_k,
            eligible_records: scores.len(),
        });
    }

    let mut first_index_by_id = BTreeMap::new();
    let mut ordered: Vec<&ExactScoreRecord<Id>> = Vec::with_capacity(scores.len());
    for (record_index, record) in scores.iter().enumerate() {
        if !record.score.is_finite() || !(-1.0..=1.0).contains(&record.score) {
            return Err(IntervalSelectionError::InvalidExactScore { record_index });
        }
        if let Some(first_index) = first_index_by_id.get(&record.id).copied() {
            return Err(IntervalSelectionError::DuplicateId {
                first_index,
                duplicate_index: record_index,
            });
        }
        first_index_by_id.insert(record.id.clone(), record_index);
        ordered.push(record);
    }

    ordered.sort_by(|left, right| {
        let score_order = if left.score == right.score {
            std::cmp::Ordering::Equal
        } else {
            right.score.total_cmp(&left.score)
        };
        score_order.then_with(|| left.id.cmp(&right.id))
    });

    Ok(ordered
        .into_iter()
        .take(top_k)
        .map(|record| record.id.clone())
        .collect())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    fn record(id: &str, lower: f32, upper: f32) -> ScoreIntervalRecord<String> {
        ScoreIntervalRecord {
            id: id.to_owned(),
            interval: ContinuousHVScoreInterval { lower, upper },
        }
    }

    fn complete_coverage(count: usize) -> IntervalCoverageCounts {
        IntervalCoverageCounts {
            eligible_records: count,
            interval_records: count,
            unbounded_unopened_partitions: 0,
        }
    }

    #[test]
    fn fixture_selects_global_refinement_set_and_preserves_threshold_equality() {
        let records = vec![
            record("a", 0.90, 0.98),
            record("b", 0.82, 0.94),
            record("c", 0.78, 0.86),
            record("d", 0.55, 0.82),
            record("e", 0.10, 0.30),
            record("f", -0.05, 0.81),
            record("g", 0.80, 0.82),
        ];
        let plan = select_refinement_set(&records, 2, complete_coverage(7)).unwrap();
        assert_eq!(plan.threshold, 0.82);
        assert_eq!(
            plan.refinement_ids,
            vec!["a".to_owned(), "b".to_owned(), "c".to_owned(), "d".to_owned(), "g".to_owned()]
        );
        assert_eq!(plan.excluded_ids, vec!["e".to_owned(), "f".to_owned()]);
    }

    #[test]
    fn output_is_independent_of_input_order() {
        let first = vec![
            record("a", 0.9, 0.98),
            record("b", 0.8, 0.9),
            record("c", 0.1, 0.2),
        ];
        let second = vec![first[2].clone(), first[0].clone(), first[1].clone()];
        let first_plan = select_refinement_set(&first, 2, complete_coverage(3)).unwrap();
        let second_plan = select_refinement_set(&second, 2, complete_coverage(3)).unwrap();
        assert_eq!(first_plan, second_plan);
    }

    #[test]
    fn duplicate_ids_and_invalid_intervals_fail_closed() {
        let duplicate = vec![record("same", 0.1, 0.2), record("same", 0.0, 0.3)];
        assert!(matches!(
            select_refinement_set(&duplicate, 1, complete_coverage(2)),
            Err(IntervalSelectionError::DuplicateId { .. })
        ));

        let invalid = vec![record("bad", f32::NAN, 0.5)];
        assert_eq!(
            select_refinement_set(&invalid, 1, complete_coverage(1)),
            Err(IntervalSelectionError::InvalidInterval { record_index: 0 })
        );
    }

    #[test]
    fn incomplete_metadata_or_unopened_partition_prevents_selection() {
        let records = vec![record("a", 0.1, 0.9), record("b", 0.0, 0.2)];
        let missing_record = IntervalCoverageCounts {
            eligible_records: 3,
            interval_records: 2,
            unbounded_unopened_partitions: 0,
        };
        assert!(matches!(
            select_refinement_set(&records, 1, missing_record),
            Err(IntervalSelectionError::IncompleteCoverage { .. })
        ));

        let unbounded_partition = IntervalCoverageCounts {
            eligible_records: 2,
            interval_records: 2,
            unbounded_unopened_partitions: 1,
        };
        assert!(matches!(
            select_refinement_set(&records, 1, unbounded_partition),
            Err(IntervalSelectionError::IncompleteCoverage { .. })
        ));
    }

    #[test]
    fn generated_interval_refinement_matches_full_exact_top_k() {
        let mut state = 0x5EED_1234u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };

        for case_index in 0..4096usize {
            let count = 1 + (next() as usize % 32);
            let top_k = 1 + (next() as usize % count);
            let mut exact_scores = Vec::with_capacity(count);
            let mut intervals = Vec::with_capacity(count);

            for item_index in 0..count {
                let score_raw = (next() % 2001) as i32 - 1000;
                let score = score_raw as f32 / 1000.0;
                let lower_slack = (next() % 501) as f32 / 1000.0;
                let upper_slack = (next() % 501) as f32 / 1000.0;
                let lower = (score - lower_slack).max(-1.0);
                let upper = (score + upper_slack).min(1.0);
                let id = format!("{item_index:04}");
                exact_scores.push((id.clone(), score));
                intervals.push(record(&id, lower, upper));
            }

            let plan = select_refinement_set(
                &intervals,
                top_k,
                complete_coverage(count),
            )
            .unwrap();

            let full_scores: Vec<ExactScoreRecord<String>> = exact_scores
                .iter()
                .map(|(id, score)| ExactScoreRecord {
                    id: id.clone(),
                    score: *score,
                })
                .collect();
            let expected = stable_exact_top_k(&full_scores, top_k).unwrap();

            let refinement_members: BTreeSet<String> =
                plan.refinement_ids.iter().cloned().collect();
            let refined_scores: Vec<ExactScoreRecord<String>> = exact_scores
                .iter()
                .filter(|(id, _)| refinement_members.contains(id))
                .map(|(id, score)| ExactScoreRecord {
                    id: id.clone(),
                    score: *score,
                })
                .collect();
            let actual = stable_exact_top_k(&refined_scores, top_k).unwrap();

            assert_eq!(
                actual, expected,
                "false-prune disagreement in generated case {case_index}"
            );
        }
    }

    #[test]
    fn threshold_ties_use_stable_id_even_for_signed_zero() {
        let records = vec![
            record("z", 0.0, 0.2),
            record("a", -0.0, 0.2),
        ];
        let plan = select_refinement_set(&records, 1, complete_coverage(2)).unwrap();
        assert_eq!(plan.threshold.to_bits(), (-0.0f32).to_bits());
        assert_eq!(plan.refinement_ids, vec!["a".to_owned(), "z".to_owned()]);
    }

    #[test]
    fn exact_rerank_uses_numeric_score_then_stable_id_including_signed_zero() {
        let scores = vec![
            ExactScoreRecord { id: "z".to_owned(), score: 0.0 },
            ExactScoreRecord { id: "b".to_owned(), score: -0.0 },
            ExactScoreRecord { id: "a".to_owned(), score: 0.0 },
            ExactScoreRecord { id: "c".to_owned(), score: 0.5 },
        ];
        assert_eq!(
            stable_exact_top_k(&scores, 3).unwrap(),
            vec!["c".to_owned(), "a".to_owned(), "b".to_owned()]
        );
    }

    #[test]
    fn exact_rerank_rejects_non_finite_or_out_of_range_scores() {
        let scores = vec![ExactScoreRecord {
            id: "nan".to_owned(),
            score: f32::NAN,
        }];
        assert_eq!(
            stable_exact_top_k(&scores, 1),
            Err(IntervalSelectionError::InvalidExactScore { record_index: 0 })
        );
        let out_of_range = vec![ExactScoreRecord {
            id: "large".to_owned(),
            score: 1.01,
        }];
        assert_eq!(
            stable_exact_top_k(&out_of_range, 1),
            Err(IntervalSelectionError::InvalidExactScore { record_index: 0 })
        );
    }

    #[test]
    fn invalid_top_k_is_rejected() {
        let records = vec![record("a", 0.1, 0.9)];
        assert!(matches!(
            select_refinement_set(&records, 0, complete_coverage(1)),
            Err(IntervalSelectionError::InvalidTopK { .. })
        ));
        assert!(matches!(
            select_refinement_set(&records, 2, complete_coverage(1)),
            Err(IntervalSelectionError::InvalidTopK { .. })
        ));
    }
}
