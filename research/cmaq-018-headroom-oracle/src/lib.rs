//! Exact finite repeated-continuation oracle for CMAQ-018.
//!
//! This module deliberately reports evaluator-oracle diagnostics, not a
//! deployable selector result. It does not estimate confidence intervals or
//! infer causal coupling from matching labels.

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CouplingDisposition {
    /// Candidate rows are aligned by the same exogenous continuation identity.
    /// The justification digest is retained as caller-supplied metadata; this
    /// module does not authenticate or independently validate that evidence.
    SharedInvariant { justification_digest: String },
    /// Candidate rows are independent. Column positions are replicate indexes,
    /// not matched random outcomes.
    Uncoupled,
    /// A pairing is retained for sensitivity analysis only and is not admitted
    /// as the primary variance decomposition.
    SensitivityOnly { justification_digest: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateOutcomes {
    pub candidate_id: String,
    /// One evaluator outcome per continuation in the owning batch.
    /// Higher values are better; failures must remain represented in this
    /// utility (for example, through the preregistered failure penalty).
    pub values: Vec<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContinuationBatch {
    /// Stable identities for the continuation columns. For SharedInvariant,
    /// each column denotes the same causally justified exogenous continuation
    /// across all candidate rows.
    pub continuation_ids: Vec<String>,
    pub candidates: Vec<CandidateOutcomes>,
    pub coupling: CouplingDisposition,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadroomDesign {
    /// Candidate IDs must be sorted lexicographically. This fixes tie-breaking
    /// and makes the result invariant to caller row ordering.
    pub candidate_ids: Vec<String>,
    pub baseline_candidate_id: String,
    /// Used for all candidate-selection statistics.
    pub selection: ContinuationBatch,
    /// Must be disjoint from selection continuations. Used only after selection.
    pub evaluation: ContinuationBatch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadroomError {
    EmptyCandidateSet,
    NonCanonicalCandidateOrder,
    EmptyIdentity(&'static str),
    MissingBaselineCandidate,
    EmptyContinuationSet,
    DuplicateContinuationId,
    CandidateRowCountMismatch,
    CandidateRowOrderMismatch,
    OutcomeCountMismatch,
    SelectionEvaluationOverlap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VarianceDecompositionDisposition {
    ComputedUnderDeclaredSharedInvariant,
    NotIdentifiedUncoupled,
    SensitivityOnlyNotPrimary,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CandidateContinuationDecomposition {
    pub candidate_sum_squares: f64,
    pub continuation_sum_squares: f64,
    pub interaction_sum_squares: f64,
    /// Interaction SS / (candidate SS + interaction SS). This is undefined
    /// when the denominator is zero.
    pub interaction_fraction_of_candidate_distinguishing_variation: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VarianceDecompositionResult {
    pub disposition: VarianceDecompositionDisposition,
    pub decomposition: Option<CandidateContinuationDecomposition>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OracleLane {
    pub selected_candidate_id: String,
    /// Score used to select. For the one-shot lane this is the first selection
    /// continuation; for repeated selection it is the mean across selection
    /// continuations.
    pub selected_selection_value: f64,
    pub baseline_selection_value: f64,
    pub selection_gain: f64,
    /// The selected candidate is scored on continuations never used to select.
    pub selected_heldout_value: f64,
    pub baseline_heldout_value: f64,
    pub heldout_gain: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActualSelectorStatus {
    /// This reference computes evaluator oracles only. It has not measured D3.
    NotMeasuredInThisReference,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HeadroomReport {
    pub candidate_count: usize,
    pub selection_continuation_count: usize,
    pub evaluation_continuation_count: usize,
    /// Select and score on the same first continuation. An optimistic
    /// diagnostic, never a deployable selector result.
    pub one_shot_oracle: OracleLane,
    /// Select and score on the same repeated selection set. An in-sample
    /// evaluator oracle, not a deployable selector or a formal upper bound.
    pub repeated_in_sample_oracle: OracleLane,
    /// Select on the selection set, then score on the disjoint evaluation set.
    /// This is a held-out evaluator-oracle diagnostic, not the learned D3 selector.
    pub heldout_repeated_oracle: OracleLane,
    /// Selected outcome on the selection set minus the same candidate's
    /// held-out mean. Positive values indicate in-sample optimism.
    pub one_shot_selection_optimism: f64,
    pub repeated_selection_optimism: f64,
    pub heldout_share_of_one_shot_apparent_gain: Option<f64>,
    pub heldout_share_of_repeated_in_sample_gain: Option<f64>,
    pub variance_decomposition: VarianceDecompositionResult,
    pub actual_selector_status: ActualSelectorStatus,
}

pub fn evaluate_headroom(design: &HeadroomDesign) -> Result<HeadroomReport, HeadroomError> {
    validate_design(design)?;

    let baseline_index = design
        .candidate_ids
        .iter()
        .position(|candidate| candidate == &design.baseline_candidate_id)
        .ok_or(HeadroomError::MissingBaselineCandidate)?;

    let first_selection_scores: Vec<f64> = design
        .selection
        .candidates
        .iter()
        .map(|candidate| candidate.values[0] as f64)
        .collect();
    let one_shot_index = argmax_canonical(&first_selection_scores);
    let one_shot = make_lane(design, one_shot_index, baseline_index, |candidate| {
        candidate.values[0] as f64
    });

    let selection_means: Vec<f64> = design
        .selection
        .candidates
        .iter()
        .map(|candidate| mean(&candidate.values))
        .collect();
    let repeated_index = argmax_canonical(&selection_means);
    let repeated_in_sample = make_lane(design, repeated_index, baseline_index, |candidate| {
        mean(&candidate.values)
    });

    let heldout_repeated = make_lane(design, repeated_index, baseline_index, |candidate| {
        mean(&candidate.values)
    });

    let one_shot_selection_optimism =
        one_shot.selected_selection_value - one_shot.selected_heldout_value;
    let repeated_selection_optimism =
        repeated_in_sample.selected_selection_value - repeated_in_sample.selected_heldout_value;

    let variance_decomposition = match &design.selection.coupling {
        CouplingDisposition::SharedInvariant { .. } => VarianceDecompositionResult {
            disposition: VarianceDecompositionDisposition::ComputedUnderDeclaredSharedInvariant,
            decomposition: Some(decompose_candidate_continuation(
                &design.selection.candidates,
            )),
        },
        CouplingDisposition::Uncoupled => VarianceDecompositionResult {
            disposition: VarianceDecompositionDisposition::NotIdentifiedUncoupled,
            decomposition: None,
        },
        CouplingDisposition::SensitivityOnly { .. } => VarianceDecompositionResult {
            disposition: VarianceDecompositionDisposition::SensitivityOnlyNotPrimary,
            decomposition: None,
        },
    };

    Ok(HeadroomReport {
        candidate_count: design.candidate_ids.len(),
        selection_continuation_count: design.selection.continuation_ids.len(),
        evaluation_continuation_count: design.evaluation.continuation_ids.len(),
        one_shot_oracle: one_shot,
        repeated_in_sample_oracle: repeated_in_sample,
        heldout_repeated_oracle: heldout_repeated,
        one_shot_selection_optimism,
        repeated_selection_optimism,
        heldout_share_of_one_shot_apparent_gain: positive_gain_share(
            one_shot.heldout_gain,
            one_shot.selection_gain,
        ),
        heldout_share_of_repeated_in_sample_gain: positive_gain_share(
            heldout_repeated.heldout_gain,
            repeated_in_sample.selection_gain,
        ),
        variance_decomposition,
        actual_selector_status: ActualSelectorStatus::NotMeasuredInThisReference,
    })
}

fn make_lane<F>(
    design: &HeadroomDesign,
    selected_index: usize,
    baseline_index: usize,
    selection_value: F,
) -> OracleLane
where
    F: Fn(&CandidateOutcomes) -> f64,
{
    let selected = &design.candidate_ids[selected_index];
    let selected_selection_value = selection_value(&design.selection.candidates[selected_index]);
    let baseline_selection_value = selection_value(&design.selection.candidates[baseline_index]);
    let selected_heldout_value = mean(&design.evaluation.candidates[selected_index].values);
    let baseline_heldout_value = mean(&design.evaluation.candidates[baseline_index].values);

    OracleLane {
        selected_candidate_id: selected.clone(),
        selected_selection_value,
        baseline_selection_value,
        selection_gain: selected_selection_value - baseline_selection_value,
        selected_heldout_value,
        baseline_heldout_value,
        heldout_gain: selected_heldout_value - baseline_heldout_value,
    }
}

fn validate_design(design: &HeadroomDesign) -> Result<(), HeadroomError> {
    if design.candidate_ids.is_empty() {
        return Err(HeadroomError::EmptyCandidateSet);
    }
    for candidate_id in &design.candidate_ids {
        require_identity(candidate_id, "candidate_id")?;
    }
    if design
        .candidate_ids
        .windows(2)
        .any(|pair| pair[0] >= pair[1])
    {
        return Err(HeadroomError::NonCanonicalCandidateOrder);
    }
    require_identity(&design.baseline_candidate_id, "baseline_candidate_id")?;
    if !design
        .candidate_ids
        .iter()
        .any(|candidate| candidate == &design.baseline_candidate_id)
    {
        return Err(HeadroomError::MissingBaselineCandidate);
    }

    validate_batch(&design.selection, &design.candidate_ids)?;
    validate_batch(&design.evaluation, &design.candidate_ids)?;

    let selection_ids: std::collections::BTreeSet<_> =
        design.selection.continuation_ids.iter().collect();
    if design
        .evaluation
        .continuation_ids
        .iter()
        .any(|continuation_id| selection_ids.contains(continuation_id))
    {
        return Err(HeadroomError::SelectionEvaluationOverlap);
    }
    Ok(())
}

fn validate_batch(
    batch: &ContinuationBatch,
    candidate_ids: &[String],
) -> Result<(), HeadroomError> {
    if batch.continuation_ids.is_empty() {
        return Err(HeadroomError::EmptyContinuationSet);
    }
    let mut seen_continuations = std::collections::BTreeSet::new();
    for continuation_id in &batch.continuation_ids {
        require_identity(continuation_id, "continuation_id")?;
        if !seen_continuations.insert(continuation_id) {
            return Err(HeadroomError::DuplicateContinuationId);
        }
    }
    if batch.candidates.len() != candidate_ids.len() {
        return Err(HeadroomError::CandidateRowCountMismatch);
    }
    for (row, expected_id) in batch.candidates.iter().zip(candidate_ids) {
        if row.candidate_id.as_str() != expected_id.as_str() {
            return Err(HeadroomError::CandidateRowOrderMismatch);
        }
        if row.values.len() != batch.continuation_ids.len() {
            return Err(HeadroomError::OutcomeCountMismatch);
        }
    }
    match &batch.coupling {
        CouplingDisposition::SharedInvariant {
            justification_digest,
        }
        | CouplingDisposition::SensitivityOnly {
            justification_digest,
        } => {
            require_identity(justification_digest, "coupling_justification_digest")?;
        }
        CouplingDisposition::Uncoupled => {}
    }
    Ok(())
}

fn require_identity(value: &str, field: &'static str) -> Result<(), HeadroomError> {
    if value.trim().is_empty() {
        Err(HeadroomError::EmptyIdentity(field))
    } else {
        Ok(())
    }
}

/// Candidate IDs must be in canonical lexical order before this function is
/// called. Ties therefore resolve to the lexicographically first candidate.
fn argmax_canonical(values: &[f64]) -> usize {
    let mut best = 0usize;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

fn mean(values: &[i64]) -> f64 {
    values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64
}

fn positive_gain_share(heldout_gain: f64, apparent_gain: f64) -> Option<f64> {
    (apparent_gain > 0.0).then_some(heldout_gain / apparent_gain)
}

fn decompose_candidate_continuation(
    candidates: &[CandidateOutcomes],
) -> CandidateContinuationDecomposition {
    let candidate_count = candidates.len();
    let continuation_count = candidates[0].values.len();

    let candidate_means: Vec<f64> = candidates
        .iter()
        .map(|candidate| mean(&candidate.values))
        .collect();
    let mut continuation_means = vec![0.0; continuation_count];
    for (continuation_index, continuation_mean) in continuation_means.iter_mut().enumerate() {
        *continuation_mean = candidates
            .iter()
            .map(|candidate| candidate.values[continuation_index] as f64)
            .sum::<f64>()
            / candidate_count as f64;
    }

    let grand_mean = candidate_means.iter().sum::<f64>() / candidate_count as f64;
    let candidate_sum_squares = continuation_count as f64
        * candidate_means
            .iter()
            .map(|candidate_mean| (candidate_mean - grand_mean).powi(2))
            .sum::<f64>();
    let continuation_sum_squares = candidate_count as f64
        * continuation_means
            .iter()
            .map(|continuation_mean| (continuation_mean - grand_mean).powi(2))
            .sum::<f64>();

    let mut interaction_sum_squares = 0.0;
    for (candidate_index, candidate) in candidates.iter().enumerate() {
        for (continuation_index, value) in candidate.values.iter().enumerate() {
            let residual = *value as f64
                - candidate_means[candidate_index]
                - continuation_means[continuation_index]
                + grand_mean;
            interaction_sum_squares += residual * residual;
        }
    }

    let candidate_distinguishing = candidate_sum_squares + interaction_sum_squares;
    let interaction_fraction = (candidate_distinguishing > 0.0)
        .then_some(interaction_sum_squares / candidate_distinguishing);

    CandidateContinuationDecomposition {
        candidate_sum_squares,
        continuation_sum_squares,
        interaction_sum_squares,
        interaction_fraction_of_candidate_distinguishing_variation: interaction_fraction,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(candidate_id: &str, values: &[i64]) -> CandidateOutcomes {
        CandidateOutcomes {
            candidate_id: candidate_id.to_owned(),
            values: values.to_vec(),
        }
    }

    fn batch(
        continuation_ids: &[&str],
        candidates: Vec<CandidateOutcomes>,
        coupling: CouplingDisposition,
    ) -> ContinuationBatch {
        ContinuationBatch {
            continuation_ids: continuation_ids.iter().map(|id| (*id).to_owned()).collect(),
            candidates,
            coupling,
        }
    }

    fn shared() -> CouplingDisposition {
        CouplingDisposition::SharedInvariant {
            justification_digest: "independent-causal-review:fixture-01".to_owned(),
        }
    }

    fn noise_only_design() -> HeadroomDesign {
        HeadroomDesign {
            candidate_ids: vec![
                "baseline".to_owned(),
                "candidate-a".to_owned(),
                "candidate-b".to_owned(),
            ],
            baseline_candidate_id: "baseline".to_owned(),
            selection: batch(
                &["selection-0", "selection-1", "selection-2", "selection-3"],
                vec![
                    candidate("baseline", &[0, 0, 0, 0]),
                    candidate("candidate-a", &[10, -10, 0, 0]),
                    candidate("candidate-b", &[-10, 10, 0, 0]),
                ],
                shared(),
            ),
            evaluation: batch(
                &["heldout-0", "heldout-1"],
                vec![
                    candidate("baseline", &[0, 0]),
                    candidate("candidate-a", &[0, 0]),
                    candidate("candidate-b", &[0, 0]),
                ],
                shared(),
            ),
        }
    }

    #[test]
    fn noise_only_best_of_one_has_apparent_gain_but_no_heldout_gain() {
        let report = evaluate_headroom(&noise_only_design()).unwrap();

        assert_eq!(report.one_shot_oracle.selected_candidate_id, "candidate-a");
        assert_eq!(report.one_shot_oracle.selection_gain, 10.0);
        assert_eq!(report.one_shot_oracle.heldout_gain, 0.0);

        assert_eq!(
            report.repeated_in_sample_oracle.selected_candidate_id,
            "baseline"
        );
        assert_eq!(report.repeated_in_sample_oracle.selection_gain, 0.0);
        assert_eq!(report.heldout_repeated_oracle.heldout_gain, 0.0);
        assert_eq!(report.heldout_share_of_one_shot_apparent_gain, Some(0.0));
        assert_eq!(
            report.actual_selector_status,
            ActualSelectorStatus::NotMeasuredInThisReference
        );
    }

    #[test]
    fn one_shot_heldout_share_scores_the_one_shot_selected_candidate() {
        let design = HeadroomDesign {
            candidate_ids: vec![
                "baseline".to_owned(),
                "candidate-a".to_owned(),
                "candidate-b".to_owned(),
            ],
            baseline_candidate_id: "baseline".to_owned(),
            selection: batch(
                &["s0", "s1", "s2"],
                vec![
                    candidate("baseline", &[0, 0, 0]),
                    candidate("candidate-a", &[2, 2, 2]),
                    candidate("candidate-b", &[10, -10, 0]),
                ],
                shared(),
            ),
            evaluation: batch(
                &["e0", "e1"],
                vec![
                    candidate("baseline", &[0, 0]),
                    candidate("candidate-a", &[2, 2]),
                    candidate("candidate-b", &[0, 0]),
                ],
                shared(),
            ),
        };

        let report = evaluate_headroom(&design).unwrap();
        assert_eq!(report.one_shot_oracle.selected_candidate_id, "candidate-b");
        assert_eq!(
            report.repeated_in_sample_oracle.selected_candidate_id,
            "candidate-a"
        );
        assert_eq!(report.one_shot_oracle.selection_gain, 10.0);
        assert_eq!(report.one_shot_oracle.heldout_gain, 0.0);
        assert_eq!(report.heldout_repeated_oracle.heldout_gain, 2.0);
        assert_eq!(report.heldout_share_of_one_shot_apparent_gain, Some(0.0));
        assert_eq!(report.heldout_share_of_repeated_in_sample_gain, Some(1.0));
    }

    #[test]
    fn genuine_candidate_effect_survives_repeated_heldout_scoring() {
        let design = HeadroomDesign {
            candidate_ids: vec![
                "baseline".to_owned(),
                "candidate-a".to_owned(),
                "candidate-b".to_owned(),
            ],
            baseline_candidate_id: "baseline".to_owned(),
            selection: batch(
                &["s0", "s1", "s2", "s3"],
                vec![
                    candidate("baseline", &[0, 0, 0, 0]),
                    candidate("candidate-a", &[2, 2, 2, 2]),
                    candidate("candidate-b", &[1, 1, 1, 1]),
                ],
                shared(),
            ),
            evaluation: batch(
                &["e0", "e1", "e2"],
                vec![
                    candidate("baseline", &[0, 0, 0]),
                    candidate("candidate-a", &[2, 2, 2]),
                    candidate("candidate-b", &[1, 1, 1]),
                ],
                shared(),
            ),
        };

        let report = evaluate_headroom(&design).unwrap();
        assert_eq!(
            report.heldout_repeated_oracle.selected_candidate_id,
            "candidate-a"
        );
        assert_eq!(report.heldout_repeated_oracle.heldout_gain, 2.0);
        assert_eq!(report.heldout_share_of_one_shot_apparent_gain, Some(1.0));
        let decomposition = report.variance_decomposition.decomposition.unwrap();
        assert_eq!(
            decomposition.interaction_fraction_of_candidate_distinguishing_variation,
            Some(0.0)
        );
    }

    #[test]
    fn candidate_by_continuation_reversal_is_reported_as_interaction() {
        let design = HeadroomDesign {
            candidate_ids: vec!["candidate-a".to_owned(), "candidate-b".to_owned()],
            baseline_candidate_id: "candidate-a".to_owned(),
            selection: batch(
                &["s0", "s1"],
                vec![
                    candidate("candidate-a", &[1, -1]),
                    candidate("candidate-b", &[-1, 1]),
                ],
                shared(),
            ),
            evaluation: batch(
                &["e0", "e1"],
                vec![
                    candidate("candidate-a", &[0, 0]),
                    candidate("candidate-b", &[0, 0]),
                ],
                shared(),
            ),
        };

        let report = evaluate_headroom(&design).unwrap();
        let decomposition = report.variance_decomposition.decomposition.unwrap();
        assert_eq!(decomposition.candidate_sum_squares, 0.0);
        assert!(decomposition.interaction_sum_squares > 0.0);
        assert_eq!(
            decomposition.interaction_fraction_of_candidate_distinguishing_variation,
            Some(1.0)
        );
    }

    #[test]
    fn unpaired_continuations_do_not_produce_a_paired_variance_decomposition() {
        let mut design = noise_only_design();
        design.selection.coupling = CouplingDisposition::Uncoupled;

        let report = evaluate_headroom(&design).unwrap();
        assert_eq!(
            report.variance_decomposition.disposition,
            VarianceDecompositionDisposition::NotIdentifiedUncoupled
        );
        assert!(report.variance_decomposition.decomposition.is_none());
    }

    #[test]
    fn sensitivity_only_pairing_is_not_promoted_to_primary_decomposition() {
        let mut design = noise_only_design();
        design.selection.coupling = CouplingDisposition::SensitivityOnly {
            justification_digest: "sensitivity-only:fixture-02".to_owned(),
        };

        let report = evaluate_headroom(&design).unwrap();
        assert_eq!(
            report.variance_decomposition.disposition,
            VarianceDecompositionDisposition::SensitivityOnlyNotPrimary
        );
        assert!(report.variance_decomposition.decomposition.is_none());
    }

    #[test]
    fn selection_and_evaluation_continuations_must_be_disjoint() {
        let mut design = noise_only_design();
        design.evaluation.continuation_ids[0] = "selection-0".to_owned();

        assert_eq!(
            evaluate_headroom(&design),
            Err(HeadroomError::SelectionEvaluationOverlap)
        );
    }

    #[test]
    fn candidate_order_is_canonical_and_controls_deterministic_ties() {
        let mut design = noise_only_design();
        design.selection.candidates[1].values = vec![0, 0, 0, 0];

        let report = evaluate_headroom(&design).unwrap();
        assert_eq!(
            report.repeated_in_sample_oracle.selected_candidate_id,
            "baseline"
        );

        design.candidate_ids.reverse();
        assert_eq!(
            evaluate_headroom(&design),
            Err(HeadroomError::NonCanonicalCandidateOrder)
        );
    }

    #[test]
    fn outcome_matrix_dimensions_are_checked() {
        let mut design = noise_only_design();
        design.selection.candidates[1].values.pop();

        assert_eq!(
            evaluate_headroom(&design),
            Err(HeadroomError::OutcomeCountMismatch)
        );
    }

    #[test]
    fn shared_pairing_requires_a_nonempty_justification_reference() {
        let mut design = noise_only_design();
        design.selection.coupling = CouplingDisposition::SharedInvariant {
            justification_digest: "  ".to_owned(),
        };

        assert_eq!(
            evaluate_headroom(&design),
            Err(HeadroomError::EmptyIdentity(
                "coupling_justification_digest"
            ))
        );
    }
}
