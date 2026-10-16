//! Deterministic, dependency-free metric kernel for CRISPR-OT-001.
//!
//! Compile and run tests without Cargo or network access:
//!   rustc --edition=2024 --test scripts/crispr_triage_verifier.rs -o /tmp/crispr-triage-tests
//!   /tmp/crispr-triage-tests
//!
//! This file intentionally does not load biological data or execute experiments.
//! The caller must separately verify source rights, artifact bytes/digests, cohort
//! reconstruction, and the provenance of all labels. The digest function below
//! validates a supplied SHA-256 hex claim; it does not hash the artifact bytes.

use std::collections::HashSet;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LabelState {
    Positive,
    AssayDefinedNegative,
    UnknownUnassayed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Candidate {
    pub id: String,
    /// One-based rank in the frozen, declared ranking universe.
    pub rank: usize,
    pub label: LabelState,
    /// True only if this candidate appeared in the native generator's output
    /// before any positive seeding for a shared ranking universe.
    pub native_nominated: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NotEstimableReason {
    NoEligiblePositives,
    ZeroBudget,
    TooFewCandidates { requested: usize, available: usize },
    UnknownOutcomesInSelectedSet { ids: Vec<String> },
    TargetNotReached,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Estimate<T> {
    Value(T),
    NotEstimable(NotEstimableReason),
}

#[derive(Clone, Debug, PartialEq)]
pub enum MetricError {
    DuplicateCandidateId(String),
    DuplicateRank(usize),
    DuplicateReferencePositive(String),
    PositiveOutsideReference(String),
    InvalidRecallTarget(f64),
}

impl fmt::Display for MetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateCandidateId(id) => write!(f, "duplicate candidate id: {id}"),
            Self::DuplicateRank(rank) => write!(f, "duplicate ranking position: {rank}"),
            Self::DuplicateReferencePositive(id) => {
                write!(f, "duplicate eligible positive id: {id}")
            }
            Self::PositiveOutsideReference(id) => {
                write!(f, "positive candidate is outside the declared reference: {id}")
            }
            Self::InvalidRecallTarget(target) => {
                write!(f, "recall target must be finite and within [0, 1], got {target}")
            }
        }
    }
}

impl std::error::Error for MetricError {}

/// Validate the frozen universe and return candidates ordered by rank.
/// Invalid identity/rank mappings fail closed; they are never silently deduplicated.
fn validated_ranked<'a>(
    candidates: &'a [Candidate],
    eligible_positive_ids: &[String],
) -> Result<Vec<&'a Candidate>, MetricError> {
    let mut reference = HashSet::with_capacity(eligible_positive_ids.len());
    for id in eligible_positive_ids {
        if !reference.insert(id.as_str()) {
            return Err(MetricError::DuplicateReferencePositive(id.clone()));
        }
    }

    let mut ids = HashSet::with_capacity(candidates.len());
    let mut ranks = HashSet::with_capacity(candidates.len());
    for candidate in candidates {
        if !ids.insert(candidate.id.as_str()) {
            return Err(MetricError::DuplicateCandidateId(candidate.id.clone()));
        }
        if !ranks.insert(candidate.rank) {
            return Err(MetricError::DuplicateRank(candidate.rank));
        }
        if candidate.label == LabelState::Positive && !reference.contains(candidate.id.as_str()) {
            return Err(MetricError::PositiveOutsideReference(candidate.id.clone()));
        }
    }

    let mut ranked: Vec<&Candidate> = candidates.iter().collect();
    ranked.sort_by_key(|candidate| candidate.rank);
    Ok(ranked)
}

/// Fraction of independently established eligible positives that occurred in
/// the *native* candidate generator output before positives were seeded.
pub fn native_positive_coverage(
    candidates: &[Candidate],
    eligible_positive_ids: &[String],
) -> Result<Estimate<f64>, MetricError> {
    let _ranked = validated_ranked(candidates, eligible_positive_ids)?;
    if eligible_positive_ids.is_empty() {
        return Ok(Estimate::NotEstimable(
            NotEstimableReason::NoEligiblePositives,
        ));
    }

    let nominated_positive_ids: HashSet<&str> = candidates
        .iter()
        .filter(|candidate| {
            candidate.native_nominated && candidate.label == LabelState::Positive
        })
        .map(|candidate| candidate.id.as_str())
        .collect();

    let hits = eligible_positive_ids
        .iter()
        .filter(|id| nominated_positive_ids.contains(id.as_str()))
        .count();

    Ok(Estimate::Value(hits as f64 / eligible_positive_ids.len() as f64))
}

/// Recall@K within a single, explicit ranking universe. The denominator is the
/// frozen eligible-positive reference, including positives absent from the
/// ranked candidate list; positive seeding must not alter native coverage.
pub fn recall_at_k(
    candidates: &[Candidate],
    eligible_positive_ids: &[String],
    k: usize,
) -> Result<Estimate<f64>, MetricError> {
    let ranked = validated_ranked(candidates, eligible_positive_ids)?;
    if eligible_positive_ids.is_empty() {
        return Ok(Estimate::NotEstimable(
            NotEstimableReason::NoEligiblePositives,
        ));
    }

    let selected = ranked.iter().take(k);
    let hits = selected
        .filter(|candidate| candidate.label == LabelState::Positive)
        .count();
    Ok(Estimate::Value(
        hits as f64 / eligible_positive_ids.len() as f64,
    ))
}

/// Precision@K is not estimable if an unknown/unassayed candidate is in the
/// review set or if the ranking cannot fill all K declared slots.
pub fn precision_at_k(
    candidates: &[Candidate],
    eligible_positive_ids: &[String],
    k: usize,
) -> Result<Estimate<f64>, MetricError> {
    let ranked = validated_ranked(candidates, eligible_positive_ids)?;
    if k == 0 {
        return Ok(Estimate::NotEstimable(NotEstimableReason::ZeroBudget));
    }
    if ranked.len() < k {
        return Ok(Estimate::NotEstimable(
            NotEstimableReason::TooFewCandidates {
                requested: k,
                available: ranked.len(),
            },
        ));
    }

    let selected: Vec<&Candidate> = ranked.into_iter().take(k).collect();
    let unknown_ids: Vec<String> = selected
        .iter()
        .filter(|candidate| candidate.label == LabelState::UnknownUnassayed)
        .map(|candidate| candidate.id.clone())
        .collect();
    if !unknown_ids.is_empty() {
        return Ok(Estimate::NotEstimable(
            NotEstimableReason::UnknownOutcomesInSelectedSet { ids: unknown_ids },
        ));
    }

    let hits = selected
        .iter()
        .filter(|candidate| candidate.label == LabelState::Positive)
        .count();
    Ok(Estimate::Value(hits as f64 / k as f64))
}

/// Return the smallest K at which the ranked list reaches the requested recall.
/// A target that cannot be reached is explicitly NotEstimable(TargetNotReached).
pub fn slots_at_recall(
    candidates: &[Candidate],
    eligible_positive_ids: &[String],
    target: f64,
) -> Result<Estimate<usize>, MetricError> {
    if !target.is_finite() || !(0.0..=1.0).contains(&target) {
        return Err(MetricError::InvalidRecallTarget(target));
    }
    let ranked = validated_ranked(candidates, eligible_positive_ids)?;
    if eligible_positive_ids.is_empty() {
        return Ok(Estimate::NotEstimable(
            NotEstimableReason::NoEligiblePositives,
        ));
    }
    if target == 0.0 {
        return Ok(Estimate::Value(0));
    }

    let mut hits = 0usize;
    for (index, candidate) in ranked.iter().enumerate() {
        if candidate.label == LabelState::Positive {
            hits += 1;
        }
        let recall = hits as f64 / eligible_positive_ids.len() as f64;
        if recall >= target {
            return Ok(Estimate::Value(index + 1));
        }
    }

    Ok(Estimate::NotEstimable(
        NotEstimableReason::TargetNotReached,
    ))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DigestError {
    MalformedExpected,
    MalformedObserved,
    Mismatch,
}

/// Check that two already-computed, canonical lowercase SHA-256 hex strings
/// match. This does not compute SHA-256 from bytes; callers must hash the
/// retrieved artifact bytes with a trusted implementation first.
pub fn verify_sha256_claim(expected: &str, observed: &str) -> Result<(), DigestError> {
    fn is_canonical_sha256(value: &str) -> bool {
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }

    if !is_canonical_sha256(expected) {
        return Err(DigestError::MalformedExpected);
    }
    if !is_canonical_sha256(observed) {
        return Err(DigestError::MalformedObserved);
    }
    if expected != observed {
        return Err(DigestError::Mismatch);
    }
    Ok(())
}

/// Reject group leakage in a declared train/holdout partition and return the
/// exact overlapping group IDs, sorted for deterministic diagnostics.
pub fn validate_group_holdout(
    train_group_ids: &[String],
    held_out_group_ids: &[String],
) -> Result<(), Vec<String>> {
    let train: HashSet<&str> = train_group_ids.iter().map(String::as_str).collect();
    let mut overlap: Vec<String> = held_out_group_ids
        .iter()
        .filter(|group| train.contains(group.as_str()))
        .cloned()
        .collect();
    overlap.sort();
    overlap.dedup();

    if overlap.is_empty() {
        Ok(())
    } else {
        Err(overlap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(
        id: &str,
        rank: usize,
        label: LabelState,
        native_nominated: bool,
    ) -> Candidate {
        Candidate {
            id: id.to_owned(),
            rank,
            label,
            native_nominated,
        }
    }

    fn positives(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|id| (*id).to_owned()).collect()
    }

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    /// Mirrors F1 in CRISPR_TRIAGE_SCORECARD_V1.json:
    /// native nomination remains 0.5 although post-seed Recall@5 becomes 1.0.
    #[test]
    fn f1_seeded_ranking_never_changes_native_nomination_coverage() {
        let reference = positives(&["p1", "p2", "p3", "p4"]);
        let candidates = vec![
            candidate("p1", 1, LabelState::Positive, true),
            candidate("n1", 2, LabelState::AssayDefinedNegative, true),
            candidate("p3", 3, LabelState::Positive, true),
            candidate("p2", 4, LabelState::Positive, false),
            candidate("p4", 5, LabelState::Positive, false),
            candidate("n2", 6, LabelState::AssayDefinedNegative, false),
        ];

        match native_positive_coverage(&candidates, &reference).unwrap() {
            Estimate::Value(value) => close(value, 0.5),
            other => panic!("expected coverage estimate, got {other:?}"),
        }
        match recall_at_k(&candidates, &reference, 3).unwrap() {
            Estimate::Value(value) => close(value, 0.5),
            other => panic!("expected Recall@3, got {other:?}"),
        }
        match recall_at_k(&candidates, &reference, 5).unwrap() {
            Estimate::Value(value) => close(value, 1.0),
            other => panic!("expected Recall@5, got {other:?}"),
        }
    }

    /// Mirrors F2: an unknown among reviewed candidates blocks precision.
    #[test]
    fn f2_unknown_outcome_is_not_a_true_negative() {
        let reference = positives(&["p1"]);
        let candidates = vec![
            candidate("u1", 1, LabelState::UnknownUnassayed, false),
            candidate("p1", 2, LabelState::Positive, true),
            candidate("n1", 3, LabelState::AssayDefinedNegative, true),
        ];

        assert_eq!(
            precision_at_k(&candidates, &reference, 2).unwrap(),
            Estimate::NotEstimable(
                NotEstimableReason::UnknownOutcomesInSelectedSet {
                    ids: vec!["u1".to_owned()]
                }
            )
        );
        assert_eq!(
            recall_at_k(&candidates, &reference, 2).unwrap(),
            Estimate::Value(1.0)
        );
    }

    #[test]
    fn f3_digest_mismatch_blocks_identity_claim() {
        let expected = "a".repeat(64);
        let observed = "b".repeat(64);
        assert_eq!(
            verify_sha256_claim(&expected, &observed),
            Err(DigestError::Mismatch)
        );
        assert_eq!(verify_sha256_claim(&expected, &expected), Ok(()));
        assert_eq!(
            verify_sha256_claim("not-a-digest", "not-a-digest"),
            Err(DigestError::MalformedExpected)
        );
    }

    #[test]
    fn f4_group_overlap_blocks_generalization_claim() {
        let train = positives(&["group-1", "group-7"]);
        let held_out = positives(&["group-7", "group-18"]);
        assert_eq!(
            validate_group_holdout(&train, &held_out),
            Err(vec!["group-7".to_owned()])
        );
        assert_eq!(
            validate_group_holdout(&train, &positives(&["group-18"])),
            Ok(())
        );
    }

    #[test]
    fn unreachable_recall_target_is_not_fabricated() {
        let reference = positives(&["p1", "p2"]);
        let candidates = vec![candidate("p1", 1, LabelState::Positive, true)];
        assert_eq!(
            slots_at_recall(&candidates, &reference, 1.0).unwrap(),
            Estimate::NotEstimable(NotEstimableReason::TargetNotReached)
        );
    }

    #[test]
    fn duplicate_candidate_ids_and_ranks_fail_closed() {
        let reference = positives(&["p1"]);
        let duplicate_id = vec![
            candidate("p1", 1, LabelState::Positive, true),
            candidate("p1", 2, LabelState::Positive, true),
        ];
        assert_eq!(
            recall_at_k(&duplicate_id, &reference, 1),
            Err(MetricError::DuplicateCandidateId("p1".to_owned()))
        );

        let duplicate_rank = vec![
            candidate("p1", 1, LabelState::Positive, true),
            candidate("n1", 1, LabelState::AssayDefinedNegative, false),
        ];
        assert_eq!(
            recall_at_k(&duplicate_rank, &reference, 1),
            Err(MetricError::DuplicateRank(1))
        );
    }

    #[test]
    fn positive_outside_reference_is_rejected() {
        let reference = positives(&["p1"]);
        let candidates = vec![candidate("p2", 1, LabelState::Positive, true)];
        assert_eq!(
            recall_at_k(&candidates, &reference, 1),
            Err(MetricError::PositiveOutsideReference("p2".to_owned()))
        );
    }

    #[test]
    fn invalid_recall_target_is_rejected() {
        let reference = positives(&["p1"]);
        let candidates = vec![candidate("p1", 1, LabelState::Positive, true)];
        assert!(matches!(
            slots_at_recall(&candidates, &reference, f64::NAN),
            Err(MetricError::InvalidRecallTarget(value)) if value.is_nan()
        ));
    }
}
