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
    OutcomeSumOverflow,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairingVarianceDisposition {
    /// Matched differences are computed under caller-declared invariant exogenous
    /// continuation identities. This module does not authenticate that claim.
    ComputedUnderDeclaredSharedInvariant,
    /// Marginal variances are available, but no candidate-by-candidate pairing
    /// is admitted because the continuation streams are independent.
    NotIdentifiedUncoupled,
    /// Paired statistics are shown only as sensitivity diagnostics.
    SensitivityOnlyNotPrimary,
    /// Fewer than two evaluation continuations were supplied.
    InsufficientContinuations,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CandidatePairVariance {
    pub candidate_sample_variance: Option<f64>,
    pub baseline_sample_variance: Option<f64>,
    pub sample_covariance: Option<f64>,
    pub paired_difference_sample_variance: Option<f64>,
    /// Variance of the difference in sample means under independent sampling:
    /// (s_candidate^2 + s_baseline^2) / M.
    pub independent_difference_of_means_variance: Option<f64>,
    /// Variance of the paired-difference sample mean: s_(candidate-baseline)^2 / M.
    pub paired_difference_of_means_variance: Option<f64>,
    /// Paired mean variance / independent mean variance. Values above 1 mean
    /// declared pairing increased variance.
    pub paired_to_independent_variance_ratio: Option<f64>,
    /// 1 - paired_to_independent_variance_ratio. Negative values are retained.
    pub common_random_number_variance_reduction_fraction: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CandidatePairVarianceRecord {
    pub candidate_id: String,
    pub baseline_candidate_id: String,
    pub continuation_count: usize,
    pub variance: CandidatePairVariance,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PairingVarianceReport {
    pub disposition: PairingVarianceDisposition,
    pub comparisons: Vec<CandidatePairVarianceRecord>,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateSupportDisposition {
    Supported,
    Unsupported,
    Unknown,
    Incomplete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SupportAssessmentIndependence {
    IndependentOfOutcomeSelection,
    OutcomeDependentOrUnknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateSupportRecord {
    pub candidate_id: String,
    pub disposition: CandidateSupportDisposition,
    pub independence: SupportAssessmentIndependence,
    /// Exact support-gate receipt reference; not authenticated by this crate.
    pub support_receipt_digest: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateUniverseScope {
    ExhaustiveWithinDeclaredScope,
    IncompleteOrBudgetLimited,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContinuationSamplingDisposition {
    IidAcrossContinuationIds,
    IndependenceNotEstablished,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContinuationSamplingContract {
    pub disposition: ContinuationSamplingDisposition,
    /// Caller-supplied evidence reference for the sampling design. This crate
    /// retains it but does not authenticate the source or prove independence.
    pub justification_digest: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectorScope {
    CompleteWithinDeclaredScope,
    EnumeratedBankOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SupportedSelectorDisposition {
    AlternativeSelected,
    BaselineRetained,
    Abstain,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectorAbstentionReason {
    BaselineSupportNotQualified,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SupportedSelectorError {
    InvalidHeadroomDesign(HeadroomError),
    InvalidUtilityBounds,
    InvalidFamilywiseDelta,
    SupportRecordCountMismatch,
    SupportRecordOrderMismatch,
    SupportAssessmentNotIndependentOfOutcomeSelection,
    EmptySupportReceiptDigest,
    SelectionContinuationIndependenceNotEstablished,
    EmptySamplingJustification,
    OutcomeOutsideDeclaredUtilityBounds,
    WorkCountOverflow,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SupportGatedSelectorDesign {
    pub headroom: HeadroomDesign,
    /// Must be generated by the independent SAM support gate for this context.
    /// This finite reference validates structure, not the authenticity of receipts.
    pub support: Vec<CandidateSupportRecord>,
    pub candidate_universe: CandidateUniverseScope,
    pub continuation_sampling: ContinuationSamplingContract,
    pub utility_lower_bound: i64,
    pub utility_upper_bound: i64,
    /// Familywise error budget in parts per million; 10_000 means delta=0.01.
    pub familywise_delta_ppm: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SupportGatedSelectorReport {
    pub disposition: SupportedSelectorDisposition,
    pub abstention_reason: Option<SelectorAbstentionReason>,
    pub selected_candidate_id: Option<String>,
    pub baseline_candidate_id: String,
    pub scope: SelectorScope,
    pub familywise_delta_ppm: u32,
    pub utility_lower_bound: i64,
    pub utility_upper_bound: i64,
    /// Source-side support dispositions and receipt references retained verbatim.
    pub support_records: Vec<CandidateSupportRecord>,
    /// Sampling assumption and caller-supplied justification retained verbatim.
    pub continuation_sampling: ContinuationSamplingContract,
    pub selection_continuation_count: usize,
    pub evaluation_continuation_count: usize,
    pub eligible_alternative_count: usize,
    pub excluded_candidate_ids: Vec<String>,
    /// Simultaneous Hoeffding radius for all eligible candidates plus baseline.
    pub simultaneous_confidence_radius: Option<f64>,
    pub selected_lower_bound: Option<f64>,
    pub baseline_upper_bound: Option<f64>,
    /// Descriptive held-out values; never used by the selection rule.
    pub heldout_selected_mean: Option<f64>,
    pub heldout_baseline_mean: Option<f64>,
    pub heldout_gain: Option<f64>,
    /// Number of candidate × continuation utility cells in each supplied matrix. This is not a simulator rollout count.
    pub selection_candidate_continuation_cells: usize,
    pub evaluation_candidate_continuation_cells: usize,
}

/// A finite reference selector using repeated continuation means and a
/// simultaneous bounded-outcome gate. It is not a learned D3 selector.
///
/// For each supported candidate and the supported baseline, a two-sided
/// Hoeffding interval is formed only when the caller declares that continuation
/// draws are independent across continuation IDs and utilities lie in
/// [lower, upper]. The contract's justification digest is retained as metadata,
/// not authenticated by this reference.
/// An alternative is eligible only when its lower bound strictly exceeds the
/// baseline upper bound. The union bound covers the baseline and every
/// supported alternative. Common random numbers may correlate candidates
/// within one continuation column; that does not by itself invalidate the
/// per-candidate marginal bounds, but serial dependence across columns does.
///
/// Held-out outcomes are accessed only after the selected candidate is fixed.
pub fn select_supported_candidate(
    design: &SupportGatedSelectorDesign,
) -> Result<SupportGatedSelectorReport, SupportedSelectorError> {
    validate_design(&design.headroom).map_err(SupportedSelectorError::InvalidHeadroomDesign)?;
    if design.continuation_sampling.disposition
        != ContinuationSamplingDisposition::IidAcrossContinuationIds
    {
        return Err(SupportedSelectorError::SelectionContinuationIndependenceNotEstablished);
    }
    if design
        .continuation_sampling
        .justification_digest
        .trim()
        .is_empty()
    {
        return Err(SupportedSelectorError::EmptySamplingJustification);
    }
    if design.utility_lower_bound >= design.utility_upper_bound {
        return Err(SupportedSelectorError::InvalidUtilityBounds);
    }
    if design.familywise_delta_ppm == 0 || design.familywise_delta_ppm >= 1_000_000 {
        return Err(SupportedSelectorError::InvalidFamilywiseDelta);
    }
    if design.support.len() != design.headroom.candidate_ids.len() {
        return Err(SupportedSelectorError::SupportRecordCountMismatch);
    }
    for (record, candidate_id) in design.support.iter().zip(&design.headroom.candidate_ids) {
        if record.candidate_id != *candidate_id {
            return Err(SupportedSelectorError::SupportRecordOrderMismatch);
        }
        if record.independence != SupportAssessmentIndependence::IndependentOfOutcomeSelection {
            return Err(SupportedSelectorError::SupportAssessmentNotIndependentOfOutcomeSelection);
        }
        if record.support_receipt_digest.trim().is_empty() {
            return Err(SupportedSelectorError::EmptySupportReceiptDigest);
        }
    }
    for batch in [&design.headroom.selection, &design.headroom.evaluation] {
        if batch.candidates.iter().any(|candidate| {
            candidate.values.iter().any(|value| {
                *value < design.utility_lower_bound || *value > design.utility_upper_bound
            })
        }) {
            return Err(SupportedSelectorError::OutcomeOutsideDeclaredUtilityBounds);
        }
    }

    let rollout_count = design.headroom.candidate_ids.len();
    let selection_candidate_continuation_cells = rollout_count
        .checked_mul(design.headroom.selection.continuation_ids.len())
        .ok_or(SupportedSelectorError::WorkCountOverflow)?;
    let evaluation_candidate_continuation_cells = rollout_count
        .checked_mul(design.headroom.evaluation.continuation_ids.len())
        .ok_or(SupportedSelectorError::WorkCountOverflow)?;

    let scope = match design.candidate_universe {
        CandidateUniverseScope::ExhaustiveWithinDeclaredScope => {
            SelectorScope::CompleteWithinDeclaredScope
        }
        CandidateUniverseScope::IncompleteOrBudgetLimited => SelectorScope::EnumeratedBankOnly,
    };

    let candidate_ids = &design.headroom.candidate_ids;
    let baseline_index = candidate_ids
        .iter()
        .position(|candidate_id| candidate_id == &design.headroom.baseline_candidate_id)
        .ok_or(SupportedSelectorError::InvalidHeadroomDesign(
            HeadroomError::MissingBaselineCandidate,
        ))?;
    let baseline_supported =
        design.support[baseline_index].disposition == CandidateSupportDisposition::Supported;

    let excluded_candidate_ids = design
        .support
        .iter()
        .filter(|record| record.disposition != CandidateSupportDisposition::Supported)
        .map(|record| record.candidate_id.clone())
        .collect::<Vec<_>>();

    if !baseline_supported {
        return Ok(SupportGatedSelectorReport {
            disposition: SupportedSelectorDisposition::Abstain,
            abstention_reason: Some(SelectorAbstentionReason::BaselineSupportNotQualified),
            selected_candidate_id: None,
            baseline_candidate_id: design.headroom.baseline_candidate_id.clone(),
            scope,
            familywise_delta_ppm: design.familywise_delta_ppm,
            utility_lower_bound: design.utility_lower_bound,
            utility_upper_bound: design.utility_upper_bound,
            support_records: design.support.clone(),
            continuation_sampling: design.continuation_sampling.clone(),
            selection_continuation_count: design.headroom.selection.continuation_ids.len(),
            evaluation_continuation_count: design.headroom.evaluation.continuation_ids.len(),
            eligible_alternative_count: 0,
            excluded_candidate_ids,
            simultaneous_confidence_radius: None,
            selected_lower_bound: None,
            baseline_upper_bound: None,
            heldout_selected_mean: None,
            heldout_baseline_mean: None,
            heldout_gain: None,
            selection_candidate_continuation_cells,
            evaluation_candidate_continuation_cells,
        });
    }

    let eligible_alternatives = design
        .support
        .iter()
        .enumerate()
        .filter(|(index, record)| {
            *index != baseline_index && record.disposition == CandidateSupportDisposition::Supported
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();

    let simultaneous_count = eligible_alternatives.len() + 1;
    let delta = design.familywise_delta_ppm as f64 / 1_000_000.0;
    let utility_range =
        (i128::from(design.utility_upper_bound) - i128::from(design.utility_lower_bound)) as f64;
    let selection_count = design.headroom.selection.continuation_ids.len();
    let radius = utility_range
        * ((2.0 * simultaneous_count as f64 / delta).ln() / (2.0 * selection_count as f64)).sqrt();

    let baseline_mean = mean(&design.headroom.selection.candidates[baseline_index].values);
    let baseline_upper_bound = baseline_mean + radius;
    let baseline_sum = checked_sum(&design.headroom.selection.candidates[baseline_index].values)
        .map_err(SupportedSelectorError::InvalidHeadroomDesign)?;

    let mut qualifying = Vec::<(usize, i128)>::new();
    for index in eligible_alternatives {
        let values = &design.headroom.selection.candidates[index].values;
        let candidate_sum =
            checked_sum(values).map_err(SupportedSelectorError::InvalidHeadroomDesign)?;
        let mean_difference = (candidate_sum
            .checked_sub(baseline_sum)
            .ok_or(SupportedSelectorError::WorkCountOverflow)?)
            as f64
            / selection_count as f64;
        if mean_difference > 2.0 * radius {
            qualifying.push((index, candidate_sum));
        }
    }

    qualifying.sort_by(|left, right| {
        right
            .1
            .cmp(&left.1)
            .then_with(|| candidate_ids[left.0].cmp(&candidate_ids[right.0]))
    });

    let (disposition, selected_index, selected_lower_bound) =
        if let Some((index, _)) = qualifying.first().copied() {
            let selected_mean = mean(&design.headroom.selection.candidates[index].values);
            (
                SupportedSelectorDisposition::AlternativeSelected,
                index,
                Some(selected_mean - radius),
            )
        } else {
            (
                SupportedSelectorDisposition::BaselineRetained,
                baseline_index,
                Some(baseline_mean - radius),
            )
        };

    let heldout_selected_mean = mean(&design.headroom.evaluation.candidates[selected_index].values);
    let heldout_baseline_mean = mean(&design.headroom.evaluation.candidates[baseline_index].values);

    Ok(SupportGatedSelectorReport {
        disposition,
        abstention_reason: None,
        selected_candidate_id: Some(candidate_ids[selected_index].clone()),
        baseline_candidate_id: design.headroom.baseline_candidate_id.clone(),
        scope,
        familywise_delta_ppm: design.familywise_delta_ppm,
        utility_lower_bound: design.utility_lower_bound,
        utility_upper_bound: design.utility_upper_bound,
        support_records: design.support.clone(),
        continuation_sampling: design.continuation_sampling.clone(),
        selection_continuation_count: design.headroom.selection.continuation_ids.len(),
        evaluation_continuation_count: design.headroom.evaluation.continuation_ids.len(),
        eligible_alternative_count: eligible_alternatives.len(),
        excluded_candidate_ids,
        simultaneous_confidence_radius: Some(radius),
        selected_lower_bound,
        baseline_upper_bound: Some(baseline_upper_bound),
        heldout_selected_mean: Some(heldout_selected_mean),
        heldout_baseline_mean: Some(heldout_baseline_mean),
        heldout_gain: Some(heldout_selected_mean - heldout_baseline_mean),
        selection_candidate_continuation_cells,
        evaluation_candidate_continuation_cells,
    })
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
    /// Paired-versus-independent variance comparison on held-out continuations.
    pub pairing_variance: PairingVarianceReport,
    pub actual_selector_status: ActualSelectorStatus,
}

pub fn evaluate_headroom(design: &HeadroomDesign) -> Result<HeadroomReport, HeadroomError> {
    validate_design(design)?;

    let baseline_index = design
        .candidate_ids
        .iter()
        .position(|candidate| candidate == &design.baseline_candidate_id)
        .ok_or(HeadroomError::MissingBaselineCandidate)?;

    let first_selection_scores: Vec<i64> = design
        .selection
        .candidates
        .iter()
        .map(|candidate| candidate.values[0])
        .collect();
    let one_shot_index = argmax_canonical(&first_selection_scores);
    let one_shot = make_lane(design, one_shot_index, baseline_index, |candidate| {
        candidate.values[0] as f64
    });

    // All candidates have the same continuation count, so comparing exact
    // integer sums is equivalent to comparing their means and avoids f64
    // ranking collapse for adjacent values above 2^53.
    let selection_totals: Vec<i128> = design
        .selection
        .candidates
        .iter()
        .map(|candidate| checked_sum(&candidate.values))
        .collect::<Result<_, _>>()?;
    let repeated_index = argmax_canonical(&selection_totals);
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

    // Capture scalar diagnostics before moving the lane structures into the report.
    let heldout_share_of_one_shot_apparent_gain =
        positive_gain_share(one_shot.heldout_gain, one_shot.selection_gain);
    let heldout_share_of_repeated_in_sample_gain = positive_gain_share(
        heldout_repeated.heldout_gain,
        repeated_in_sample.selection_gain,
    );

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
        heldout_share_of_one_shot_apparent_gain,
        heldout_share_of_repeated_in_sample_gain,
        variance_decomposition,
        pairing_variance: pairing_variance_report(&design.evaluation, baseline_index),
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
        checked_sum(&row.values)?;
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
fn argmax_canonical<T: PartialOrd>(values: &[T]) -> usize {
    let mut best = 0usize;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

fn checked_sum(values: &[i64]) -> Result<i128, HeadroomError> {
    values.iter().try_fold(0_i128, |total, value| {
        total
            .checked_add(i128::from(*value))
            .ok_or(HeadroomError::OutcomeSumOverflow)
    })
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

fn pairing_variance_report(
    batch: &ContinuationBatch,
    baseline_index: usize,
) -> PairingVarianceReport {
    let continuation_count = batch.continuation_ids.len();
    if continuation_count < 2 {
        return PairingVarianceReport {
            disposition: PairingVarianceDisposition::InsufficientContinuations,
            comparisons: Vec::new(),
        };
    }

    let baseline = &batch.candidates[baseline_index];
    let disposition = match &batch.coupling {
        CouplingDisposition::SharedInvariant { .. } => {
            PairingVarianceDisposition::ComputedUnderDeclaredSharedInvariant
        }
        CouplingDisposition::Uncoupled => PairingVarianceDisposition::NotIdentifiedUncoupled,
        CouplingDisposition::SensitivityOnly { .. } => {
            PairingVarianceDisposition::SensitivityOnlyNotPrimary
        }
    };

    let comparisons = batch
        .candidates
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != baseline_index)
        .map(|(_, candidate)| {
            let baseline_variance = sample_variance_i64(&baseline.values);
            let candidate_variance = sample_variance_i64(&candidate.values);
            let independent_mean_variance =
                (baseline_variance + candidate_variance) / continuation_count as f64;

            let (covariance, paired_difference_variance) = match &batch.coupling {
                CouplingDisposition::SharedInvariant { .. }
                | CouplingDisposition::SensitivityOnly { .. } => {
                    let covariance = sample_covariance_i64(&candidate.values, &baseline.values);
                    let paired = sample_variance_f64(
                        &candidate
                            .values
                            .iter()
                            .zip(&baseline.values)
                            .map(|(candidate, baseline)| *candidate as f64 - *baseline as f64)
                            .collect::<Vec<_>>(),
                    );
                    (Some(covariance), Some(paired))
                }
                CouplingDisposition::Uncoupled => (None, None),
            };

            let paired_mean_variance =
                paired_difference_variance.map(|variance| variance / continuation_count as f64);
            let ratio = paired_mean_variance.and_then(|paired| {
                (independent_mean_variance > 0.0).then_some(paired / independent_mean_variance)
            });
            let reduction = ratio.map(|value| 1.0 - value);

            CandidatePairVarianceRecord {
                candidate_id: candidate.candidate_id.clone(),
                baseline_candidate_id: baseline.candidate_id.clone(),
                continuation_count,
                variance: CandidatePairVariance {
                    candidate_sample_variance: Some(candidate_variance),
                    baseline_sample_variance: Some(baseline_variance),
                    sample_covariance: covariance,
                    paired_difference_sample_variance: paired_difference_variance,
                    independent_difference_of_means_variance: Some(independent_mean_variance),
                    paired_difference_of_means_variance: paired_mean_variance,
                    paired_to_independent_variance_ratio: ratio,
                    common_random_number_variance_reduction_fraction: reduction,
                },
            }
        })
        .collect();

    PairingVarianceReport {
        disposition,
        comparisons,
    }
}

fn sample_variance_i64(values: &[i64]) -> f64 {
    let average = mean(values);
    values
        .iter()
        .map(|value| (*value as f64 - average).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64
}

fn sample_variance_f64(values: &[f64]) -> f64 {
    let average = values.iter().sum::<f64>() / values.len() as f64;
    values
        .iter()
        .map(|value| (*value - average).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64
}

fn sample_covariance_i64(left: &[i64], right: &[i64]) -> f64 {
    let left_mean = mean(left);
    let right_mean = mean(right);
    left.iter()
        .zip(right)
        .map(|(left, right)| (*left as f64 - left_mean) * (*right as f64 - right_mean))
        .sum::<f64>()
        / (left.len() - 1) as f64
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
    fn repeated_selection_uses_exact_integer_order_above_f64_integer_precision() {
        let design = HeadroomDesign {
            candidate_ids: vec![
                "baseline".to_owned(),
                "candidate-a".to_owned(),
                "candidate-b".to_owned(),
            ],
            baseline_candidate_id: "baseline".to_owned(),
            selection: batch(
                &["s0", "s1"],
                vec![
                    candidate("baseline", &[0, 0]),
                    candidate(
                        "candidate-a",
                        &[9_007_199_254_740_992, 9_007_199_254_740_992],
                    ),
                    candidate(
                        "candidate-b",
                        &[9_007_199_254_740_993, 9_007_199_254_740_993],
                    ),
                ],
                shared(),
            ),
            evaluation: batch(
                &["e0"],
                vec![
                    candidate("baseline", &[0]),
                    candidate("candidate-a", &[1]),
                    candidate("candidate-b", &[2]),
                ],
                shared(),
            ),
        };

        let report = evaluate_headroom(&design).unwrap();
        // Both adjacent large means round to the same f64 value, but exact
        // integer sums correctly rank candidate-b above candidate-a.
        assert_eq!(
            report.repeated_in_sample_oracle.selected_candidate_id,
            "candidate-b"
        );
        assert_eq!(report.heldout_repeated_oracle.heldout_gain, 2.0);
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

    fn support_record(
        candidate_id: &str,
        disposition: CandidateSupportDisposition,
    ) -> CandidateSupportRecord {
        CandidateSupportRecord {
            candidate_id: candidate_id.to_owned(),
            disposition,
            independence: SupportAssessmentIndependence::IndependentOfOutcomeSelection,
            support_receipt_digest: format!(
                "synthetic-support-receipt:{candidate_id}:{disposition:?}"
            ),
        }
    }

    fn selector_design(
        headroom: HeadroomDesign,
        support: Vec<CandidateSupportRecord>,
        candidate_universe: CandidateUniverseScope,
    ) -> SupportGatedSelectorDesign {
        SupportGatedSelectorDesign {
            headroom,
            support,
            candidate_universe,
            continuation_sampling: ContinuationSamplingContract {
                disposition: ContinuationSamplingDisposition::IidAcrossContinuationIds,
                justification_digest: "synthetic-iid-continuation-fixture-v0.1".to_owned(),
            },
            utility_lower_bound: -10,
            utility_upper_bound: 10,
            familywise_delta_ppm: 10_000,
        }
    }

    #[test]
    fn repeated_support_gated_selector_requires_strict_simultaneous_separation() {
        let design = HeadroomDesign {
            candidate_ids: vec!["baseline".to_owned(), "candidate-a".to_owned()],
            baseline_candidate_id: "baseline".to_owned(),
            selection: batch(
                &[
                    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
                    "s12", "s13", "s14", "s15", "s16",
                ],
                vec![
                    candidate("baseline", &[0; 17]),
                    candidate("candidate-a", &[10; 17]),
                ],
                shared(),
            ),
            evaluation: batch(
                &["e0", "e1", "e2"],
                vec![
                    candidate("baseline", &[0, 0, 0]),
                    candidate("candidate-a", &[8, 9, 10]),
                ],
                shared(),
            ),
        };
        let mut selector = selector_design(
            design,
            vec![
                support_record("baseline", CandidateSupportDisposition::Supported),
                support_record("candidate-a", CandidateSupportDisposition::Supported),
            ],
            CandidateUniverseScope::ExhaustiveWithinDeclaredScope,
        );
        selector.utility_lower_bound = 0;
        selector.utility_upper_bound = 10;
        let report = select_supported_candidate(&selector).unwrap();

        assert_eq!(
            report.disposition,
            SupportedSelectorDisposition::AlternativeSelected
        );
        assert_eq!(report.selected_candidate_id.as_deref(), Some("candidate-a"));
        assert_eq!(report.scope, SelectorScope::CompleteWithinDeclaredScope);
        assert!(report.selected_lower_bound.unwrap() > report.baseline_upper_bound.unwrap());
        assert_eq!(report.heldout_gain, Some(9.0));
        assert_eq!(report.selection_candidate_continuation_cells, 34);
        assert_eq!(report.evaluation_candidate_continuation_cells, 6);
    }

    #[test]
    fn one_shot_noise_winner_is_not_selected_when_repeated_bounds_overlap() {
        let mut headroom = noise_only_design();
        headroom.selection = batch(
            &["s0", "s1", "s2", "s3"],
            vec![
                candidate("baseline", &[0, 0, 0, 0]),
                candidate("candidate-a", &[10, -10, 0, 0]),
                candidate("candidate-b", &[-10, 10, 0, 0]),
            ],
            shared(),
        );
        let selector = selector_design(
            headroom,
            vec![
                support_record("baseline", CandidateSupportDisposition::Supported),
                support_record("candidate-a", CandidateSupportDisposition::Supported),
                support_record("candidate-b", CandidateSupportDisposition::Supported),
            ],
            CandidateUniverseScope::ExhaustiveWithinDeclaredScope,
        );
        let report = select_supported_candidate(&selector).unwrap();

        assert_eq!(
            report.disposition,
            SupportedSelectorDisposition::BaselineRetained
        );
        assert_eq!(report.selected_candidate_id.as_deref(), Some("baseline"));
        assert!(report.selected_lower_bound.unwrap() <= report.baseline_upper_bound.unwrap());
    }

    #[test]
    fn unsupported_or_unknown_high_scoring_candidate_cannot_be_selected() {
        let headroom = HeadroomDesign {
            candidate_ids: vec![
                "baseline".to_owned(),
                "candidate-a".to_owned(),
                "candidate-b".to_owned(),
            ],
            baseline_candidate_id: "baseline".to_owned(),
            selection: batch(
                &[
                    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
                    "s12", "s13", "s14", "s15", "s16",
                ],
                vec![
                    candidate("baseline", &[0; 17]),
                    candidate("candidate-a", &[10; 17]),
                    candidate("candidate-b", &[9; 17]),
                ],
                shared(),
            ),
            evaluation: batch(
                &["e0", "e1"],
                vec![
                    candidate("baseline", &[0, 0]),
                    candidate("candidate-a", &[10, 10]),
                    candidate("candidate-b", &[9, 9]),
                ],
                shared(),
            ),
        };
        let selector = selector_design(
            headroom,
            vec![
                support_record("baseline", CandidateSupportDisposition::Supported),
                support_record("candidate-a", CandidateSupportDisposition::Unknown),
                support_record("candidate-b", CandidateSupportDisposition::Unsupported),
            ],
            CandidateUniverseScope::ExhaustiveWithinDeclaredScope,
        );
        let report = select_supported_candidate(&selector).unwrap();

        assert_eq!(
            report.disposition,
            SupportedSelectorDisposition::BaselineRetained
        );
        assert_eq!(report.selected_candidate_id.as_deref(), Some("baseline"));
        assert_eq!(report.eligible_alternative_count, 0);
        assert_eq!(
            report.excluded_candidate_ids,
            vec!["candidate-a".to_owned(), "candidate-b".to_owned(),]
        );
    }

    #[test]
    fn unqualified_baseline_support_forces_abstention() {
        let headroom = noise_only_design();
        let selector = selector_design(
            headroom,
            vec![
                support_record("baseline", CandidateSupportDisposition::Unknown),
                support_record("candidate-a", CandidateSupportDisposition::Supported),
                support_record("candidate-b", CandidateSupportDisposition::Supported),
            ],
            CandidateUniverseScope::ExhaustiveWithinDeclaredScope,
        );
        let report = select_supported_candidate(&selector).unwrap();

        assert_eq!(report.disposition, SupportedSelectorDisposition::Abstain);
        assert_eq!(
            report.abstention_reason,
            Some(SelectorAbstentionReason::BaselineSupportNotQualified)
        );
        assert_eq!(report.selected_candidate_id, None);
        assert_eq!(report.heldout_gain, None);
    }

    #[test]
    fn incomplete_candidate_universe_is_labelled_bank_local_not_global_optimum() {
        let headroom = HeadroomDesign {
            candidate_ids: vec!["baseline".to_owned(), "candidate-a".to_owned()],
            baseline_candidate_id: "baseline".to_owned(),
            selection: batch(
                &[
                    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
                    "s12", "s13", "s14", "s15", "s16",
                ],
                vec![
                    candidate("baseline", &[0; 17]),
                    candidate("candidate-a", &[10; 17]),
                ],
                shared(),
            ),
            evaluation: batch(
                &["e0", "e1"],
                vec![
                    candidate("baseline", &[0, 0]),
                    candidate("candidate-a", &[7, 7]),
                ],
                shared(),
            ),
        };
        let mut selector = selector_design(
            headroom,
            vec![
                support_record("baseline", CandidateSupportDisposition::Supported),
                support_record("candidate-a", CandidateSupportDisposition::Supported),
            ],
            CandidateUniverseScope::IncompleteOrBudgetLimited,
        );
        selector.utility_lower_bound = 0;
        selector.utility_upper_bound = 10;
        let report = select_supported_candidate(&selector).unwrap();

        assert_eq!(
            report.disposition,
            SupportedSelectorDisposition::AlternativeSelected
        );
        assert_eq!(report.scope, SelectorScope::EnumeratedBankOnly);
    }

    #[test]
    fn support_gate_requires_and_retains_receipt_references() {
        let mut selector = selector_design(
            noise_only_design(),
            vec![
                support_record("baseline", CandidateSupportDisposition::Supported),
                support_record("candidate-a", CandidateSupportDisposition::Unknown),
                support_record("candidate-b", CandidateSupportDisposition::Unsupported),
            ],
            CandidateUniverseScope::ExhaustiveWithinDeclaredScope,
        );
        selector.support[1].support_receipt_digest.clear();
        assert_eq!(
            select_supported_candidate(&selector),
            Err(SupportedSelectorError::EmptySupportReceiptDigest)
        );

        selector.support[1].support_receipt_digest =
            "sam-support-receipt:candidate-a:unknown".to_owned();
        selector.support[2].independence = SupportAssessmentIndependence::OutcomeDependentOrUnknown;
        assert_eq!(
            select_supported_candidate(&selector),
            Err(SupportedSelectorError::SupportAssessmentNotIndependentOfOutcomeSelection)
        );

        selector.support[2].independence =
            SupportAssessmentIndependence::IndependentOfOutcomeSelection;
        let report = select_supported_candidate(&selector).unwrap();
        assert_eq!(report.support_records, selector.support);
        assert_eq!(report.continuation_sampling, selector.continuation_sampling);
        assert_eq!(report.utility_lower_bound, selector.utility_lower_bound);
        assert_eq!(report.utility_upper_bound, selector.utility_upper_bound);
    }

    #[test]
    fn unqualified_continuation_sampling_cannot_issue_simultaneous_bounds() {
        let selector = selector_design(
            noise_only_design(),
            vec![
                support_record("baseline", CandidateSupportDisposition::Supported),
                support_record("candidate-a", CandidateSupportDisposition::Supported),
                support_record("candidate-b", CandidateSupportDisposition::Supported),
            ],
            CandidateUniverseScope::ExhaustiveWithinDeclaredScope,
        );
        let mut selector = selector;
        selector.continuation_sampling.disposition =
            ContinuationSamplingDisposition::IndependenceNotEstablished;

        assert_eq!(
            select_supported_candidate(&selector),
            Err(SupportedSelectorError::SelectionContinuationIndependenceNotEstablished)
        );
    }

    #[test]
    fn support_gated_selector_rejects_outcomes_outside_declared_utility_bounds() {
        let mut headroom = noise_only_design();
        headroom.selection.candidates[1].values[0] = 11;
        let selector = selector_design(
            headroom,
            vec![
                support_record("baseline", CandidateSupportDisposition::Supported),
                support_record("candidate-a", CandidateSupportDisposition::Supported),
                support_record("candidate-b", CandidateSupportDisposition::Supported),
            ],
            CandidateUniverseScope::ExhaustiveWithinDeclaredScope,
        );
        assert_eq!(
            select_supported_candidate(&selector),
            Err(SupportedSelectorError::OutcomeOutsideDeclaredUtilityBounds)
        );
    }

    #[test]
    fn industry_candidate_ladder_includes_k64_without_changing_selection_semantics() {
        let candidate_ids = (0..64)
            .map(|index| format!("candidate-{index:03}"))
            .collect::<Vec<_>>();
        let selection_ids = (0..17)
            .map(|index| format!("selection-{index:02}"))
            .collect::<Vec<_>>();
        let evaluation_ids = (0..17)
            .map(|index| format!("evaluation-{index:02}"))
            .collect::<Vec<_>>();
        let candidates = candidate_ids
            .iter()
            .enumerate()
            .map(|(index, id)| candidate(id, &[index as i64; 17]))
            .collect::<Vec<_>>();
        let evaluation = candidate_ids
            .iter()
            .enumerate()
            .map(|(index, id)| candidate(id, &[index as i64; 17]))
            .collect::<Vec<_>>();

        let design = HeadroomDesign {
            candidate_ids: candidate_ids.clone(),
            baseline_candidate_id: candidate_ids[0].clone(),
            selection: ContinuationBatch {
                continuation_ids: selection_ids,
                candidates,
                coupling: shared(),
            },
            evaluation: ContinuationBatch {
                continuation_ids: evaluation_ids,
                candidates: evaluation,
                coupling: shared(),
            },
        };

        let report = evaluate_headroom(&design).unwrap();
        assert_eq!(report.candidate_count, 64);
        assert_eq!(report.selection_continuation_count, 17);
        assert_eq!(report.evaluation_continuation_count, 17);
        assert_eq!(
            report.repeated_in_sample_oracle.selected_candidate_id,
            "candidate-063"
        );
        assert_eq!(report.heldout_repeated_oracle.heldout_gain, 63.0);
    }

    #[test]
    fn positive_common_noise_covariance_reduces_difference_variance() {
        let design = HeadroomDesign {
            candidate_ids: vec!["baseline".to_owned(), "candidate-a".to_owned()],
            baseline_candidate_id: "baseline".to_owned(),
            selection: batch(
                &["s0", "s1", "s2"],
                vec![
                    candidate("baseline", &[0, 10, 20]),
                    candidate("candidate-a", &[2, 12, 22]),
                ],
                shared(),
            ),
            evaluation: batch(
                &["e0", "e1", "e2"],
                vec![
                    candidate("baseline", &[0, 10, 20]),
                    candidate("candidate-a", &[2, 12, 22]),
                ],
                shared(),
            ),
        };

        let report = evaluate_headroom(&design).unwrap();
        let pairing = &report.pairing_variance;
        assert_eq!(
            pairing.disposition,
            PairingVarianceDisposition::ComputedUnderDeclaredSharedInvariant
        );
        let comparison = &pairing.comparisons[0].variance;
        assert_eq!(comparison.paired_difference_sample_variance, Some(0.0));
        assert!(comparison.independent_difference_of_means_variance.unwrap() > 0.0);
        assert_eq!(comparison.paired_to_independent_variance_ratio, Some(0.0));
        assert_eq!(
            comparison.common_random_number_variance_reduction_fraction,
            Some(1.0)
        );
    }

    #[test]
    fn negative_common_noise_covariance_reports_variance_increase() {
        let design = HeadroomDesign {
            candidate_ids: vec!["baseline".to_owned(), "candidate-a".to_owned()],
            baseline_candidate_id: "baseline".to_owned(),
            selection: batch(
                &["s0", "s1", "s2"],
                vec![
                    candidate("baseline", &[0, 10, 20]),
                    candidate("candidate-a", &[20, 10, 0]),
                ],
                shared(),
            ),
            evaluation: batch(
                &["e0", "e1", "e2"],
                vec![
                    candidate("baseline", &[0, 10, 20]),
                    candidate("candidate-a", &[20, 10, 0]),
                ],
                shared(),
            ),
        };

        let report = evaluate_headroom(&design).unwrap();
        let comparison = &report.pairing_variance.comparisons[0].variance;
        assert_eq!(comparison.sample_covariance, Some(-100.0));
        assert_eq!(comparison.paired_to_independent_variance_ratio, Some(2.0));
        assert_eq!(
            comparison.common_random_number_variance_reduction_fraction,
            Some(-1.0)
        );
    }

    #[test]
    fn unpaired_continuations_estimate_independent_variance_but_not_paired_effect() {
        let mut design = noise_only_design();
        design.evaluation.coupling = CouplingDisposition::Uncoupled;

        let report = evaluate_headroom(&design).unwrap();
        assert_eq!(
            report.pairing_variance.disposition,
            PairingVarianceDisposition::NotIdentifiedUncoupled
        );
        let comparison = &report.pairing_variance.comparisons[0].variance;
        assert!(
            comparison
                .independent_difference_of_means_variance
                .is_some()
        );
        assert!(comparison.sample_covariance.is_none());
        assert!(comparison.paired_difference_of_means_variance.is_none());
        assert!(
            comparison
                .common_random_number_variance_reduction_fraction
                .is_none()
        );
    }

    #[test]
    fn sensitivity_only_pairing_keeps_paired_statistics_out_of_primary_lane() {
        let mut design = noise_only_design();
        design.evaluation.coupling = CouplingDisposition::SensitivityOnly {
            justification_digest: "sensitivity-only-evaluation-pairing".to_owned(),
        };

        let report = evaluate_headroom(&design).unwrap();
        assert_eq!(
            report.pairing_variance.disposition,
            PairingVarianceDisposition::SensitivityOnlyNotPrimary
        );
        assert!(
            report.pairing_variance.comparisons[0]
                .variance
                .paired_difference_of_means_variance
                .is_some()
        );
    }

    #[test]
    fn paired_variance_is_not_reported_from_a_single_evaluation_continuation() {
        let mut design = noise_only_design();
        design.evaluation.continuation_ids.truncate(1);
        for candidate in &mut design.evaluation.candidates {
            candidate.values.truncate(1);
        }

        let report = evaluate_headroom(&design).unwrap();
        assert_eq!(
            report.pairing_variance.disposition,
            PairingVarianceDisposition::InsufficientContinuations
        );
        assert!(report.pairing_variance.comparisons.is_empty());
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
