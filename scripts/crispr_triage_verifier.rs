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

use std::collections::{HashMap, HashSet};
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

/// Sharp best/worst-case bounds under unrestricted assignments of unknown
/// labels inside the declared candidate universe. These are not confidence
/// intervals and do not bound positives omitted from that candidate universe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetricBounds {
    pub lower: f64,
    pub upper: f64,
}

/// Known-label Recall@K plus its label-uncertainty bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecallLabelSensitivity {
    pub observed_known_label_recall: f64,
    pub bounds: MetricBounds,
}

/// Paired policy-delta sensitivity bounds under shared, unknown candidate labels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecallDeltaLabelSensitivity {
    /// Policy A minus policy B against the measured-positive reference.
    pub observed_known_label_delta: f64,
    /// Worst/best possible delta under arbitrary unknown-label assignments in
    /// this exact shared candidate universe.
    pub bounds: MetricBounds,
    pub policy_a_only_unknown_count: usize,
    pub policy_b_only_unknown_count: usize,
    pub neutral_unknown_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MetricError {
    DuplicateCandidateId(String),
    DuplicateRank(usize),
    DuplicateReferencePositive(String),
    PositiveOutsideReference(String),
    ReferenceLabelMismatch(String),
    CandidateUniverseMismatch(String),
    CandidateLabelMismatch(String),
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
            Self::ReferenceLabelMismatch(id) => {
                write!(f, "candidate label contradicts the positive reference: {id}")
            }
            Self::CandidateUniverseMismatch(id) => {
                write!(f, "policy candidate universes differ at candidate id: {id}")
            }
            Self::CandidateLabelMismatch(id) => {
                write!(f, "policy outcome labels differ for candidate id: {id}")
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
        let is_reference_positive = reference.contains(candidate.id.as_str());
        if candidate.label == LabelState::Positive && !is_reference_positive {
            return Err(MetricError::PositiveOutsideReference(candidate.id.clone()));
        }
        if is_reference_positive && candidate.label != LabelState::Positive {
            return Err(MetricError::ReferenceLabelMismatch(candidate.id.clone()));
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

    let selected = ranked.iter().take(k);
    let hits = selected
        .filter(|candidate| candidate.label == LabelState::Positive)
        .count();
    Ok(Estimate::Value(
        hits as f64 / eligible_positive_ids.len() as f64,
    ))
}

/// Return observed Recall@K against the explicit measured-positive reference,
/// plus sharp best/worst-case bounds if every unknown label in the fixed ranking
/// universe were allowed to be negative or positive. The lower bound assigns
/// selected unknowns negative and unselected unknowns positive; the upper bound
/// does the opposite. Sites never present in this universe are outside these bounds.
pub fn recall_label_sensitivity_at_k(
    candidates: &[Candidate],
    eligible_positive_ids: &[String],
    k: usize,
) -> Result<Estimate<RecallLabelSensitivity>, MetricError> {
    let ranked = validated_ranked(candidates, eligible_positive_ids)?;
    if eligible_positive_ids.is_empty() {
        return Ok(Estimate::NotEstimable(
            NotEstimableReason::NoEligiblePositives,
        ));
    }
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

    let selected: Vec<&Candidate> = ranked.iter().copied().take(k).collect();
    let selected_positive = selected
        .iter()
        .filter(|candidate| candidate.label == LabelState::Positive)
        .count();
    let selected_unknown = selected
        .iter()
        .filter(|candidate| candidate.label == LabelState::UnknownUnassayed)
        .count();
    let total_unknown = ranked
        .iter()
        .filter(|candidate| candidate.label == LabelState::UnknownUnassayed)
        .count();
    let unselected_unknown = total_unknown - selected_unknown;
    let known_positive_total = eligible_positive_ids.len();

    let observed = selected_positive as f64 / known_positive_total as f64;
    let lower_denominator = known_positive_total + unselected_unknown;
    let upper_denominator = known_positive_total + selected_unknown;
    let lower = selected_positive as f64 / lower_denominator as f64;
    let upper = (selected_positive + selected_unknown) as f64 / upper_denominator as f64;

    Ok(Estimate::Value(RecallLabelSensitivity {
        observed_known_label_recall: observed,
        bounds: MetricBounds { lower, upper },
    }))
}

/// Conservative precision interval under unrestricted label assignments for
/// unknown candidates in the selected K slots. It does not turn unknowns into
/// negatives and is not a statistical confidence interval.
pub fn precision_label_sensitivity_at_k(
    candidates: &[Candidate],
    eligible_positive_ids: &[String],
    k: usize,
) -> Result<Estimate<MetricBounds>, MetricError> {
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
    let known_positive = selected
        .iter()
        .filter(|candidate| candidate.label == LabelState::Positive)
        .count();
    let unknown = selected
        .iter()
        .filter(|candidate| candidate.label == LabelState::UnknownUnassayed)
        .count();

    Ok(Estimate::Value(MetricBounds {
        lower: known_positive as f64 / k as f64,
        upper: (known_positive + unknown) as f64 / k as f64,
    }))
}

/// Compare two policies on the *same candidate universe and label mapping*.
/// The same unknown candidate is assigned the same latent outcome for both
/// policies. This gives paired worst/best-case label bounds for Recall@K(A)-
/// Recall@K(B); it is not a statistical confidence interval and says nothing
/// about positive sites omitted from the shared universe.
pub fn paired_recall_delta_label_sensitivity_at_k(
    policy_a: &[Candidate],
    policy_b: &[Candidate],
    eligible_positive_ids: &[String],
    k: usize,
) -> Result<Estimate<RecallDeltaLabelSensitivity>, MetricError> {
    let ranked_a = validated_ranked(policy_a, eligible_positive_ids)?;
    let ranked_b = validated_ranked(policy_b, eligible_positive_ids)?;
    if eligible_positive_ids.is_empty() {
        return Ok(Estimate::NotEstimable(
            NotEstimableReason::NoEligiblePositives,
        ));
    }
    if k == 0 {
        return Ok(Estimate::NotEstimable(NotEstimableReason::ZeroBudget));
    }
    if ranked_a.len() < k {
        return Ok(Estimate::NotEstimable(
            NotEstimableReason::TooFewCandidates {
                requested: k,
                available: ranked_a.len(),
            },
        ));
    }

    let by_id_a: HashMap<&str, LabelState> = policy_a
        .iter()
        .map(|candidate| (candidate.id.as_str(), candidate.label))
        .collect();
    let by_id_b: HashMap<&str, LabelState> = policy_b
        .iter()
        .map(|candidate| (candidate.id.as_str(), candidate.label))
        .collect();

    let mut ids_a: Vec<&str> = by_id_a.keys().copied().collect();
    let mut ids_b: Vec<&str> = by_id_b.keys().copied().collect();
    ids_a.sort_unstable();
    ids_b.sort_unstable();

    for id in &ids_a {
        match by_id_b.get(*id) {
            None => return Err(MetricError::CandidateUniverseMismatch((*id).to_owned())),
            Some(label_b) if by_id_a.get(*id).copied() != Some(*label_b) => {
                return Err(MetricError::CandidateLabelMismatch((*id).to_owned()));
            }
            Some(_) => {}
        }
    }
    for id in &ids_b {
        if !by_id_a.contains_key(*id) {
            return Err(MetricError::CandidateUniverseMismatch((*id).to_owned()));
        }
    }

    let selected_a: HashSet<&str> = ranked_a
        .iter()
        .take(k)
        .map(|candidate| candidate.id.as_str())
        .collect();
    let selected_b: HashSet<&str> = ranked_b
        .iter()
        .take(k)
        .map(|candidate| candidate.id.as_str())
        .collect();

    let positives_a = ranked_a
        .iter()
        .filter(|candidate| {
            selected_a.contains(candidate.id.as_str())
                && candidate.label == LabelState::Positive
        })
        .count();
    let positives_b = ranked_b
        .iter()
        .filter(|candidate| {
            selected_b.contains(candidate.id.as_str())
                && candidate.label == LabelState::Positive
        })
        .count();

    let unknown_a_only = policy_a
        .iter()
        .filter(|candidate| {
            candidate.label == LabelState::UnknownUnassayed
                && selected_a.contains(candidate.id.as_str())
                && !selected_b.contains(candidate.id.as_str())
        })
        .count();
    let unknown_b_only = policy_b
        .iter()
        .filter(|candidate| {
            candidate.label == LabelState::UnknownUnassayed
                && selected_b.contains(candidate.id.as_str())
                && !selected_a.contains(candidate.id.as_str())
        })
        .count();
    let unknown_total = policy_a
        .iter()
        .filter(|candidate| candidate.label == LabelState::UnknownUnassayed)
        .count();
    let neutral_unknown = unknown_total - unknown_a_only - unknown_b_only;

    let p = eligible_positive_ids.len() as f64;
    let known_delta = positives_a as f64 - positives_b as f64;
    let lower_numerator = known_delta - unknown_b_only as f64;
    let lower_neutral = if lower_numerator > 0.0 {
        neutral_unknown
    } else {
        0
    };
    let lower_denominator =
        (eligible_positive_ids.len() + unknown_b_only + lower_neutral) as f64;

    let upper_numerator = known_delta + unknown_a_only as f64;
    let upper_neutral = if upper_numerator < 0.0 {
        neutral_unknown
    } else {
        0
    };
    let upper_denominator =
        (eligible_positive_ids.len() + unknown_a_only + upper_neutral) as f64;

    Ok(Estimate::Value(RecallDeltaLabelSensitivity {
        observed_known_label_delta: known_delta / p,
        bounds: MetricBounds {
            lower: lower_numerator / lower_denominator,
            upper: upper_numerator / upper_denominator,
        },
        policy_a_only_unknown_count: unknown_a_only,
        policy_b_only_unknown_count: unknown_b_only,
        neutral_unknown_count: neutral_unknown,
    }))
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
    fn fixed_budget_recall_is_not_estimable_when_budget_is_unavailable() {
        let reference = positives(&["p1", "p2"]);
        let candidates = vec![candidate("p1", 1, LabelState::Positive, true)];
        assert_eq!(
            recall_at_k(&candidates, &reference, 2).unwrap(),
            Estimate::NotEstimable(NotEstimableReason::TooFewCandidates {
                requested: 2,
                available: 1,
            })
        );
        assert_eq!(
            recall_label_sensitivity_at_k(&candidates, &reference, 2).unwrap(),
            Estimate::NotEstimable(NotEstimableReason::TooFewCandidates {
                requested: 2,
                available: 1,
            })
        );
        assert_eq!(
            recall_at_k(&candidates, &reference, 0).unwrap(),
            Estimate::NotEstimable(NotEstimableReason::ZeroBudget)
        );
    }

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

    /// F5: report bounds rather than treat unknown candidates as false positives.
    #[test]
    fn f5_partial_labels_produce_explicit_sensitivity_bounds() {
        let reference = positives(&["p1", "p2"]);
        let candidates = vec![
            candidate("p1", 1, LabelState::Positive, true),
            candidate("u1", 2, LabelState::UnknownUnassayed, false),
            candidate("p2", 3, LabelState::Positive, true),
            candidate("u2", 4, LabelState::UnknownUnassayed, false),
        ];

        match recall_label_sensitivity_at_k(&candidates, &reference, 2).unwrap() {
            Estimate::Value(result) => {
                close(result.observed_known_label_recall, 0.5);
                close(result.bounds.lower, 1.0 / 3.0);
                close(result.bounds.upper, 2.0 / 3.0);
            }
            other => panic!("expected recall sensitivity bounds, got {other:?}"),
        }

        match precision_label_sensitivity_at_k(&candidates, &reference, 2).unwrap() {
            Estimate::Value(bounds) => {
                close(bounds.lower, 0.5);
                close(bounds.upper, 1.0);
            }
            other => panic!("expected precision sensitivity bounds, got {other:?}"),
        }
    }

    /// F6: policy comparison pairs shared unknown labels by candidate identity.
    #[test]
    fn f6_paired_policy_delta_bounds_share_unknown_outcomes() {
        let reference = positives(&["p1", "p2"]);
        let policy_a = vec![
            candidate("p1", 1, LabelState::Positive, true),
            candidate("u1", 2, LabelState::UnknownUnassayed, true),
            candidate("p2", 3, LabelState::Positive, true),
            candidate("u2", 4, LabelState::UnknownUnassayed, true),
            candidate("u3", 5, LabelState::UnknownUnassayed, true),
        ];
        let policy_b = vec![
            candidate("p2", 1, LabelState::Positive, true),
            candidate("u2", 2, LabelState::UnknownUnassayed, true),
            candidate("p1", 3, LabelState::Positive, true),
            candidate("u1", 4, LabelState::UnknownUnassayed, true),
            candidate("u3", 5, LabelState::UnknownUnassayed, true),
        ];

        match paired_recall_delta_label_sensitivity_at_k(
            &policy_a, &policy_b, &reference, 2
        ).unwrap() {
            Estimate::Value(result) => {
                close(result.observed_known_label_delta, 0.0);
                close(result.bounds.lower, -1.0 / 3.0);
                close(result.bounds.upper, 1.0 / 3.0);
                assert_eq!(result.policy_a_only_unknown_count, 1);
                assert_eq!(result.policy_b_only_unknown_count, 1);
                assert_eq!(result.neutral_unknown_count, 1);
            }
            other => panic!("expected paired delta bounds, got {other:?}"),
        }
    }

    #[test]
    fn paired_delta_refuses_budget_larger_than_shared_universe() {
        let reference = positives(&["p1"]);
        let policy_a = vec![candidate("p1", 1, LabelState::Positive, true)];
        let policy_b = vec![candidate("p1", 1, LabelState::Positive, true)];
        assert_eq!(
            paired_recall_delta_label_sensitivity_at_k(
                &policy_a, &policy_b, &reference, 2
            ).unwrap(),
            Estimate::NotEstimable(NotEstimableReason::TooFewCandidates {
                requested: 2,
                available: 1,
            })
        );
    }

    #[test]
    fn paired_delta_rejects_different_candidate_universes() {
        let reference = positives(&["p1"]);
        let policy_a = vec![candidate("p1", 1, LabelState::Positive, true)];
        let policy_b = vec![candidate("different", 1, LabelState::AssayDefinedNegative, true)];
        assert_eq!(
            paired_recall_delta_label_sensitivity_at_k(
                &policy_a, &policy_b, &reference, 1
            ),
            Err(MetricError::CandidateUniverseMismatch("p1".to_owned()))
        );
    }

    #[test]
    fn paired_delta_rejects_conflicting_label_mapping() {
        let reference = positives(&["p1"]);
        let policy_a = vec![
            candidate("p1", 1, LabelState::Positive, true),
            candidate("u1", 2, LabelState::UnknownUnassayed, true),
        ];
        let policy_b = vec![
            candidate("p1", 1, LabelState::Positive, true),
            candidate("u1", 2, LabelState::AssayDefinedNegative, true),
        ];
        assert_eq!(
            paired_recall_delta_label_sensitivity_at_k(
                &policy_a, &policy_b, &reference, 1
            ),
            Err(MetricError::CandidateLabelMismatch("u1".to_owned()))
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
    fn reference_positive_cannot_be_marked_unknown_or_negative() {
        let reference = positives(&["p1"]);
        let candidates = vec![candidate("p1", 1, LabelState::UnknownUnassayed, false)];
        assert_eq!(
            recall_at_k(&candidates, &reference, 1),
            Err(MetricError::ReferenceLabelMismatch("p1".to_owned()))
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
