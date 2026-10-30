//! Multi-rater reliability and reviewer-calibration audit for connectome proofreading.
//!
//! This evaluator is deliberately outside the candidate ranker. It consumes frozen
//! proposals/rankings and evaluation-only multi-rater records. It preserves raw votes,
//! refuses to force consensus on ties, and reports descriptive agreement/calibration
//! metrics. It does not establish biological truth or causal policy efficiency.

use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Parser)]
#[command(
    name = "connectome-multi-rater-audit",
    version,
    about = "Audit multi-rater reliability, adjudication uncertainty, confidence calibration, and review cost"
)]
struct Args {
    /// Exact proposal input previously passed to connectome-query-impact.
    #[arg(long)]
    proposals: PathBuf,

    /// Exact frozen ranking report emitted by connectome-query-impact.
    #[arg(long)]
    ranking_report: PathBuf,

    /// Separate evaluation-only multi-rater outcomes file.
    #[arg(long)]
    outcomes: PathBuf,

    /// Optional deterministic JSON report path; stdout when omitted.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct ProposalInput {
    schema_version: u32,
    candidates: Vec<ProposalCandidate>,
}

#[derive(Debug, Deserialize)]
struct ProposalCandidate {
    candidate_id: String,
}

#[derive(Debug, Deserialize)]
struct RankingInput {
    schema_version: u32,
    input_sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct OutcomeFile {
    schema_version: u32,
    source_input_sha256: String,
    source_ranking_report_sha256: String,
    adjudication_protocol_sha256: String,
    candidate_outcomes: Vec<CandidateOutcome>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CandidateOutcome {
    candidate_id: String,
    adjudicator_votes: Vec<AdjudicatorVote>,
    reviewer_reviews: Vec<ReviewerReview>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AdjudicatorVote {
    adjudicator_id: String,
    adjudication: Adjudication,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ReviewerReview {
    reviewer_id: String,
    reviewer_decision: ReviewerDecision,
    /// Confidence that the recorded accept/reject decision is the correct action.
    /// Must be null for abstentions and in [0,1] for accept/reject decisions.
    decision_confidence: Option<f64>,
    /// Measured elapsed reviewer minutes for this candidate.
    actual_review_minutes: f64,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, Eq, Ord, PartialEq, PartialOrd)]
#[serde(rename_all = "snake_case")]
enum Adjudication {
    Correct,
    Incorrect,
    Indeterminate,
}

impl Adjudication {
    fn label(self) -> &'static str {
        match self {
            Self::Correct => "correct",
            Self::Incorrect => "incorrect",
            Self::Indeterminate => "indeterminate",
        }
    }

    fn is_determinate(self) -> bool {
        matches!(self, Self::Correct | Self::Incorrect)
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, Eq, Ord, PartialEq, PartialOrd)]
#[serde(rename_all = "snake_case")]
enum ReviewerDecision {
    Accept,
    Reject,
    Abstain,
}

impl ReviewerDecision {
    fn label(self) -> &'static str {
        match self {
            Self::Accept => "accept",
            Self::Reject => "reject",
            Self::Abstain => "abstain",
        }
    }
}

#[derive(Debug, Serialize)]
struct ReliabilityMetrics {
    nominal_krippendorff_alpha: Option<f64>,
    observed_pairwise_disagreement: Option<f64>,
    expected_pairwise_disagreement: Option<f64>,
    rated_candidate_count: usize,
    rating_count: usize,
    candidate_pair_count: usize,
}

#[derive(Debug, Serialize)]
struct CandidateSummary {
    candidate_id: String,
    adjudicator_count: usize,
    adjudication_votes: BTreeMap<String, usize>,
    adjudication_consensus: Option<String>,
    adjudication_majority_fraction: f64,
    reviewer_count: usize,
    reviewer_decision_votes: BTreeMap<String, usize>,
    reviewer_consensus_decision: Option<String>,
    reviewer_majority_fraction: f64,
    determinate_reviewer_consensus_correct: Option<bool>,
    total_panel_review_minutes: f64,
    mean_reviewer_minutes: f64,
}

#[derive(Debug, Serialize)]
struct DecisionMetrics {
    review_event_count: usize,
    accepted_count: usize,
    rejected_count: usize,
    abstained_count: usize,
    determinate_adjudication_reference_review_count: usize,
    actionable_determinate_review_count: usize,
    correct_action_count: usize,
    actionable_decision_accuracy: Option<f64>,
    accepted_decision_precision: Option<f64>,
    false_accept_rate_on_incorrect_proposals: Option<f64>,
    correct_proposal_accept_rate: Option<f64>,
    abstention_rate: Option<f64>,
    total_review_minutes: f64,
    mean_review_minutes: Option<f64>,
    confidence_calibration_sample_count: usize,
    confidence_brier_score: Option<f64>,
    confidence_expected_calibration_error_10_bins: Option<f64>,
}

#[derive(Debug, Serialize)]
struct ReviewerMetrics {
    reviewer_id: String,
    #[serde(flatten)]
    metrics: DecisionMetrics,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    schema_version: u32,
    tool: &'static str,
    proposal_input_sha256: String,
    ranking_report_sha256: String,
    outcomes_sha256: String,
    adjudication_protocol_sha256: String,
    candidate_count: usize,
    reviewer_count: usize,
    adjudicator_count: usize,
    candidate_summaries: Vec<CandidateSummary>,
    reviewer_agreement: ReliabilityMetrics,
    adjudicator_agreement: ReliabilityMetrics,
    pooled_reviewer_metrics: DecisionMetrics,
    reviewer_metrics: Vec<ReviewerMetrics>,
    total_panel_review_minutes: f64,
    warnings: Vec<String>,
    status: &'static str,
    scope_note: &'static str,
}

#[derive(Debug, Clone)]
struct ReviewObservation {
    reviewer_id: String,
    decision: ReviewerDecision,
    confidence: Option<f64>,
    minutes: f64,
    adjudication_reference: Option<Adjudication>,
}

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn majority<T: Copy + Ord>(votes: &[T]) -> (Option<T>, f64) {
    if votes.is_empty() {
        return (None, 0.0);
    }
    let mut counts = BTreeMap::<T, usize>::new();
    for vote in votes {
        *counts.entry(*vote).or_default() += 1;
    }
    let mut ranked: Vec<(T, usize)> = counts.into_iter().collect();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    let (winner, winner_count) = ranked[0];
    let strict_majority = winner_count * 2 > votes.len();
    let tied_for_first = ranked.get(1).is_some_and(|second| second.1 == winner_count);
    (
        (strict_majority && !tied_for_first).then_some(winner),
        winner_count as f64 / votes.len() as f64,
    )
}

fn vote_counts<T: Copy + Ord>(
    votes: &[T],
    label: impl Fn(T) -> &'static str,
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::<String, usize>::new();
    for vote in votes {
        *counts.entry(label(*vote).to_string()).or_default() += 1;
    }
    counts
}

/// Nominal Krippendorff alpha using the finite-sample coincidence formulation.
///
/// For candidate i with n_i ratings and d_i disagreeing unordered pairs,
/// Do = sum_i(2*d_i/(n_i-1)) / N. For N pooled ratings and D disagreeing
/// unordered pairs across the pooled category counts, De = 2*D/(N*(N-1)).
/// Ties/abstentions remain categories; missing ratings are not imputed.
fn nominal_alpha(units: &[Vec<String>]) -> ReliabilityMetrics {
    let mut pooled = BTreeMap::<String, usize>::new();
    let mut rating_count = 0usize;
    let mut rated_candidate_count = 0usize;
    let mut candidate_pair_count = 0usize;
    let mut observed_disagreement_sum = 0.0f64;

    for unit in units {
        if unit.len() < 2 {
            continue;
        }
        rated_candidate_count += 1;
        rating_count += unit.len();
        let pair_count = unit.len() * (unit.len() - 1) / 2;
        candidate_pair_count += pair_count;
        let mut local = BTreeMap::<String, usize>::new();
        for label in unit {
            *local.entry(label.clone()).or_default() += 1;
            *pooled.entry(label.clone()).or_default() += 1;
        }
        let local_counts: Vec<usize> = local.values().copied().collect();
        let mut local_disagreeing_pairs = 0usize;
        for left in 0..local_counts.len() {
            for right in (left + 1)..local_counts.len() {
                local_disagreeing_pairs += local_counts[left] * local_counts[right];
            }
        }
        observed_disagreement_sum +=
            2.0 * local_disagreeing_pairs as f64 / (unit.len() - 1) as f64;
    }

    let mut pooled_disagreeing_pairs = 0usize;
    let pooled_counts: Vec<usize> = pooled.values().copied().collect();
    for left in 0..pooled_counts.len() {
        for right in (left + 1)..pooled_counts.len() {
            pooled_disagreeing_pairs += pooled_counts[left] * pooled_counts[right];
        }
    }

    let observed = (rating_count > 0).then_some(observed_disagreement_sum / rating_count as f64);
    let pooled_pair_count = rating_count * rating_count.saturating_sub(1) / 2;
    let expected = (pooled_pair_count > 0).then_some(
        2.0 * pooled_disagreeing_pairs as f64
            / (rating_count * rating_count.saturating_sub(1)) as f64,
    );
    let alpha = match (observed, expected) {
        (Some(do_value), Some(de_value)) if de_value > f64::EPSILON => {
            Some(1.0 - do_value / de_value)
        }
        _ => None,
    };

    ReliabilityMetrics {
        nominal_krippendorff_alpha: alpha,
        observed_pairwise_disagreement: observed,
        expected_pairwise_disagreement: expected,
        rated_candidate_count,
        rating_count,
        candidate_pair_count,
    }
}

fn decision_is_correct(decision: ReviewerDecision, gold: Adjudication) -> Option<bool> {
    match (decision, gold) {
        (ReviewerDecision::Accept, Adjudication::Correct)
        | (ReviewerDecision::Reject, Adjudication::Incorrect) => Some(true),
        (ReviewerDecision::Accept, Adjudication::Incorrect)
        | (ReviewerDecision::Reject, Adjudication::Correct) => Some(false),
        (ReviewerDecision::Abstain, _) | (_, Adjudication::Indeterminate) => None,
    }
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn decision_metrics(observations: &[ReviewObservation]) -> DecisionMetrics {
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let mut abstained = 0usize;
    let mut determinate_gold = 0usize;
    let mut actionable_determinate = 0usize;
    let mut correct_actions = 0usize;
    let mut accepted_correct = 0usize;
    let mut accepted_determinate = 0usize;
    let mut incorrect_gold_reviews = 0usize;
    let mut incorrect_gold_accepts = 0usize;
    let mut correct_gold_reviews = 0usize;
    let mut correct_gold_accepts = 0usize;
    let mut calibration: Vec<(f64, f64)> = Vec::new();
    let mut minutes: Vec<f64> = Vec::new();

    for observation in observations {
        minutes.push(observation.minutes);
        match observation.decision {
            ReviewerDecision::Accept => accepted += 1,
            ReviewerDecision::Reject => rejected += 1,
            ReviewerDecision::Abstain => abstained += 1,
        }

        let Some(reference) = observation.adjudication_reference.filter(|value| value.is_determinate()) else {
            continue;
        };
        determinate_gold += 1;

        match reference {
            Adjudication::Incorrect => {
                incorrect_gold_reviews += 1;
                if observation.decision == ReviewerDecision::Accept {
                    incorrect_gold_accepts += 1;
                }
            }
            Adjudication::Correct => {
                correct_gold_reviews += 1;
                if observation.decision == ReviewerDecision::Accept {
                    correct_gold_accepts += 1;
                }
            }
            Adjudication::Indeterminate => {}
        }

        if let Some(is_correct) = decision_is_correct(observation.decision, reference) {
            actionable_determinate += 1;
            if is_correct {
                correct_actions += 1;
            }
            if observation.decision == ReviewerDecision::Accept {
                accepted_determinate += 1;
                if is_correct {
                    accepted_correct += 1;
                }
            }
            if let Some(confidence) = observation.confidence {
                calibration.push((confidence, if is_correct { 1.0 } else { 0.0 }));
            }
        }
    }

    let brier = (!calibration.is_empty()).then(|| {
        calibration
            .iter()
            .map(|(confidence, outcome)| (confidence - outcome).powi(2))
            .sum::<f64>()
            / calibration.len() as f64
    });

    let ece = if calibration.is_empty() {
        None
    } else {
        let mut bins = [(0usize, 0.0f64, 0.0f64); 10];
        for (confidence, outcome) in &calibration {
            let index = ((*confidence * 10.0).floor() as usize).min(9);
            bins[index].0 += 1;
            bins[index].1 += confidence;
            bins[index].2 += outcome;
        }
        Some(
            bins.iter()
                .filter(|(count, _, _)| *count > 0)
                .map(|(count, confidence_sum, outcome_sum)| {
                    (*count as f64 / calibration.len() as f64)
                        * ((confidence_sum / *count as f64) - (outcome_sum / *count as f64)).abs()
                })
                .sum(),
        )
    };

    DecisionMetrics {
        review_event_count: observations.len(),
        accepted_count: accepted,
        rejected_count: rejected,
        abstained_count: abstained,
        determinate_adjudication_reference_review_count: determinate_gold,
        actionable_determinate_review_count: actionable_determinate,
        correct_action_count: correct_actions,
        actionable_decision_accuracy: (actionable_determinate > 0)
            .then_some(correct_actions as f64 / actionable_determinate as f64),
        accepted_decision_precision: (accepted_determinate > 0)
            .then_some(accepted_correct as f64 / accepted_determinate as f64),
        false_accept_rate_on_incorrect_proposals: (incorrect_gold_reviews > 0)
            .then_some(incorrect_gold_accepts as f64 / incorrect_gold_reviews as f64),
        correct_proposal_accept_rate: (correct_gold_reviews > 0)
            .then_some(correct_gold_accepts as f64 / correct_gold_reviews as f64),
        abstention_rate: (!observations.is_empty())
            .then_some(abstained as f64 / observations.len() as f64),
        total_review_minutes: minutes.iter().sum(),
        mean_review_minutes: mean(&minutes),
        confidence_calibration_sample_count: calibration.len(),
        confidence_brier_score: brier,
        confidence_expected_calibration_error_10_bins: ece,
    }
}

fn validate(
    proposals: &ProposalInput,
    ranking: &RankingInput,
    outcomes: &OutcomeFile,
    proposal_hash: &str,
    ranking_hash: &str,
) -> Result<BTreeMap<String, CandidateOutcome>, String> {
    if proposals.schema_version != 1 || ranking.schema_version != 1 || outcomes.schema_version != 1 {
        return Err("proposal, ranking, and outcome schema_version values must all be 1".into());
    }
    if ranking.input_sha256 != proposal_hash || outcomes.source_input_sha256 != proposal_hash {
        return Err("ranking/outcome proposal SHA-256 does not match exact proposal bytes".into());
    }
    if outcomes.source_ranking_report_sha256 != ranking_hash {
        return Err("outcomes source_ranking_report_sha256 does not match exact ranking report bytes".into());
    }
    if !is_lower_hex(proposal_hash, 64)
        || !is_lower_hex(ranking_hash, 64)
        || !is_lower_hex(&outcomes.adjudication_protocol_sha256, 64)
    {
        return Err("proposal, ranking, or adjudication protocol SHA-256 is malformed".into());
    }

    let mut expected_ids = BTreeSet::new();
    for candidate in &proposals.candidates {
        if candidate.candidate_id.trim().is_empty()
            || !expected_ids.insert(candidate.candidate_id.clone())
        {
            return Err("proposal candidate IDs must be non-empty and unique".into());
        }
    }
    if expected_ids.is_empty() {
        return Err("proposal candidate pool must be non-empty".into());
    }

    let mut reviewer_ids = BTreeSet::new();
    let mut adjudicator_ids = BTreeSet::new();
    let mut mapped = BTreeMap::new();
    for row in &outcomes.candidate_outcomes {
        if row.candidate_id.trim().is_empty() {
            return Err("outcome candidate_id must be non-empty".into());
        }
        if !expected_ids.contains(&row.candidate_id) {
            return Err(format!("outcome contains candidate absent from frozen proposals: {}", row.candidate_id));
        }
        if row.adjudicator_votes.len() < 2 {
            return Err(format!("candidate {} needs at least two independent adjudicator votes", row.candidate_id));
        }
        if row.reviewer_reviews.len() < 2 {
            return Err(format!("candidate {} needs at least two independent reviewer reviews", row.candidate_id));
        }

        let mut local_adjudicators = BTreeSet::new();
        for vote in &row.adjudicator_votes {
            if vote.adjudicator_id.trim().is_empty()
                || !local_adjudicators.insert(vote.adjudicator_id.clone())
            {
                return Err(format!("candidate {} has an empty or duplicate adjudicator ID", row.candidate_id));
            }
            adjudicator_ids.insert(vote.adjudicator_id.clone());
        }

        let mut local_reviewers = BTreeSet::new();
        for review in &row.reviewer_reviews {
            if review.reviewer_id.trim().is_empty()
                || !local_reviewers.insert(review.reviewer_id.clone())
            {
                return Err(format!("candidate {} has an empty or duplicate reviewer ID", row.candidate_id));
            }
            if !review.actual_review_minutes.is_finite() || review.actual_review_minutes <= 0.0 {
                return Err(format!("candidate {} has invalid actual_review_minutes", row.candidate_id));
            }
            match (review.reviewer_decision, review.decision_confidence) {
                (ReviewerDecision::Abstain, None) => {}
                (ReviewerDecision::Abstain, Some(_)) => {
                    return Err(format!("candidate {} abstention confidence must be null", row.candidate_id));
                }
                (ReviewerDecision::Accept | ReviewerDecision::Reject, Some(confidence))
                    if confidence.is_finite() && (0.0..=1.0).contains(&confidence) => {}
                (ReviewerDecision::Accept | ReviewerDecision::Reject, _) => {
                    return Err(format!("candidate {} accept/reject needs decision_confidence in [0,1]", row.candidate_id));
                }
            }
            reviewer_ids.insert(review.reviewer_id.clone());
        }

        if mapped.insert(row.candidate_id.clone(), row.clone()).is_some() {
            return Err(format!("duplicate candidate outcome row: {}", row.candidate_id));
        }
    }

    if mapped.len() != expected_ids.len() {
        let present_ids: BTreeSet<String> = mapped.keys().cloned().collect();
        let missing: Vec<String> = expected_ids.difference(&present_ids).cloned().collect();
        return Err(format!("outcomes must cover the complete frozen candidate pool; missing: {}", missing.join(",")));
    }
    if !reviewer_ids.is_disjoint(&adjudicator_ids) {
        return Err("reviewer and adjudicator IDs must be disjoint to preserve independent roles".into());
    }
    Ok(mapped)
}

fn evaluate(
    proposals: &ProposalInput,
    outcomes: &OutcomeFile,
    by_candidate: &BTreeMap<String, CandidateOutcome>,
    proposal_hash: String,
    ranking_hash: String,
    outcomes_hash: String,
) -> AuditReport {
    let mut reviewer_units = Vec::new();
    let mut adjudicator_units = Vec::new();
    let mut all_observations = Vec::new();
    let mut per_reviewer = BTreeMap::<String, Vec<ReviewObservation>>::new();
    let mut candidate_summaries = Vec::new();
    let mut total_panel_review_minutes = 0.0;

    for proposal_candidate in &proposals.candidates {
        let row = &by_candidate[&proposal_candidate.candidate_id];
        let adjudication_values: Vec<Adjudication> =
            row.adjudicator_votes.iter().map(|vote| vote.adjudication).collect();
        let reviewer_values: Vec<ReviewerDecision> =
            row.reviewer_reviews.iter().map(|review| review.reviewer_decision).collect();
        let (adjudication_reference_consensus, adjudication_majority_fraction) = majority(&adjudication_values);
        let (review_consensus, review_fraction) = majority(&reviewer_values);

        adjudicator_units.push(adjudication_values.iter().map(|value| value.label().to_string()).collect());
        reviewer_units.push(reviewer_values.iter().map(|value| value.label().to_string()).collect());

        let mut panel_minutes = 0.0;
        for review in &row.reviewer_reviews {
            panel_minutes += review.actual_review_minutes;
            let observation = ReviewObservation {
                reviewer_id: review.reviewer_id.clone(),
                decision: review.reviewer_decision,
                confidence: review.decision_confidence,
                minutes: review.actual_review_minutes,
                adjudication_reference: adjudication_reference_consensus,
            };
            per_reviewer
                .entry(review.reviewer_id.clone())
                .or_default()
                .push(observation.clone());
            all_observations.push(observation);
        }
        total_panel_review_minutes += panel_minutes;

        let reviewer_consensus_correct = match (adjudication_reference_consensus, review_consensus) {
            (Some(Adjudication::Correct), Some(ReviewerDecision::Accept))
            | (Some(Adjudication::Incorrect), Some(ReviewerDecision::Reject)) => Some(true),
            (Some(Adjudication::Correct), Some(ReviewerDecision::Reject))
            | (Some(Adjudication::Incorrect), Some(ReviewerDecision::Accept)) => Some(false),
            _ => None,
        };

        candidate_summaries.push(CandidateSummary {
            candidate_id: proposal_candidate.candidate_id.clone(),
            adjudicator_count: row.adjudicator_votes.len(),
            adjudication_votes: vote_counts(&adjudication_values, Adjudication::label),
            adjudication_consensus: adjudication_reference_consensus.map(|value| value.label().to_string()),
            adjudication_majority_fraction,
            reviewer_count: row.reviewer_reviews.len(),
            reviewer_decision_votes: vote_counts(&reviewer_values, ReviewerDecision::label),
            reviewer_consensus_decision: review_consensus.map(|value| value.label().to_string()),
            reviewer_majority_fraction: review_fraction,
            determinate_reviewer_consensus_correct: reviewer_consensus_correct,
            total_panel_review_minutes: panel_minutes,
            mean_reviewer_minutes: panel_minutes / row.reviewer_reviews.len() as f64,
        });
    }

    let mut reviewer_metrics: Vec<ReviewerMetrics> = per_reviewer
        .into_iter()
        .map(|(reviewer_id, observations)| ReviewerMetrics {
            reviewer_id,
            metrics: decision_metrics(&observations),
        })
        .collect();
    reviewer_metrics.sort_by(|left, right| left.reviewer_id.cmp(&right.reviewer_id));

    let reviewer_count = reviewer_metrics.len();
    let adjudicator_count = by_candidate
            .values()
            .flat_map(|row| row.adjudicator_votes.iter().map(|vote| vote.adjudicator_id.clone()))
            .collect::<BTreeSet<_>>()
            .len();

    AuditReport {
        schema_version: 1,
        tool: "connectome-multi-rater-audit",
        proposal_input_sha256: proposal_hash,
        ranking_report_sha256: ranking_hash,
        outcomes_sha256: outcomes_hash,
        adjudication_protocol_sha256: outcomes.adjudication_protocol_sha256.clone(),
        candidate_count: proposals.candidates.len(),
        reviewer_count,
        adjudicator_count,
        candidate_summaries,
        reviewer_agreement: nominal_alpha(&reviewer_units),
        adjudicator_agreement: nominal_alpha(&adjudicator_units),
        pooled_reviewer_metrics: decision_metrics(&all_observations),
        reviewer_metrics,
        total_panel_review_minutes,
        warnings: vec![
            "Strict-majority consensus is a summary, not biological ground truth; tied and indeterminate adjudications remain unresolved.".into(),
            "Reviewer and adjudicator agreement are reported separately; agreement does not establish correctness.".into(),
            "Pooled reviewer observations share candidates and are descriptive; do not treat them as independent samples or attach naive row-level confidence intervals.".into(),
            "Panel minutes are the sum of all independent reviewer events, not the elapsed wall-clock time of a real queue.".into(),
            "This retrospective audit does not estimate causal policy effects or queue-order/learning effects; use a prospective randomized or blocked trial for those claims.".into(),
            "Confidence Brier score and 10-bin ECE are calculated only for accept/reject reviews with determinate adjudication consensus; abstentions and unresolved gold are excluded.".into(),
        ],
        status: "MULTI_RATER_OUTCOMES_AUDITED",
        scope_note: "Evaluation-only measurement artifact. It must not be read by connectome-query-impact or used to tune candidate ranking on the held-out outcome corpus.",
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    let proposal_bytes = fs::read(&args.proposals)
        .map_err(|error| format!("failed to read proposal input: {error}"))?;
    let ranking_bytes = fs::read(&args.ranking_report)
        .map_err(|error| format!("failed to read frozen ranking report: {error}"))?;
    let outcome_bytes = fs::read(&args.outcomes)
        .map_err(|error| format!("failed to read multi-rater outcomes: {error}"))?;

    let proposal_hash = sha256(&proposal_bytes);
    let ranking_hash = sha256(&ranking_bytes);
    let outcomes_hash = sha256(&outcome_bytes);
    let proposals: ProposalInput = serde_json::from_slice(&proposal_bytes)
        .map_err(|error| format!("invalid proposal JSON: {error}"))?;
    let ranking: RankingInput = serde_json::from_slice(&ranking_bytes)
        .map_err(|error| format!("invalid ranking report JSON: {error}"))?;
    let outcomes: OutcomeFile = serde_json::from_slice(&outcome_bytes)
        .map_err(|error| format!("invalid multi-rater outcomes JSON: {error}"))?;
    let by_candidate = validate(&proposals, &ranking, &outcomes, &proposal_hash, &ranking_hash)?;
    let report = evaluate(
        &proposals,
        &outcomes,
        &by_candidate,
        proposal_hash,
        ranking_hash,
        outcomes_hash,
    );

    let json = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("failed to serialize multi-rater report: {error}"))?
        + "\n";
    if let Some(output) = args.output {
        fs::write(&output, json.as_bytes())
            .map_err(|error| format!("failed to write report: {error}"))?;
    } else {
        print!("{json}");
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("connectome-multi-rater-audit: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> (ProposalInput, RankingInput, OutcomeFile, String, String) {
        let proposal_value = json!({
            "schema_version": 1,
            "dataset_id": "synthetic-multi-rater",
            "candidates": [
                {"candidate_id":"candidate-1","estimated_error_probability":0.9},
                {"candidate_id":"candidate-2","estimated_error_probability":0.8},
                {"candidate_id":"candidate-3","estimated_error_probability":0.7}
            ]
        });
        let proposal_bytes = serde_json::to_vec(&proposal_value).unwrap();
        let proposal_hash = sha256(&proposal_bytes);
        let ranking_value = json!({"schema_version":1,"input_sha256":proposal_hash});
        let ranking_bytes = serde_json::to_vec(&ranking_value).unwrap();
        let ranking_hash = sha256(&ranking_bytes);
        let outcome_value = json!({
            "schema_version":1,
            "source_input_sha256":proposal_hash,
            "source_ranking_report_sha256":ranking_hash,
            "adjudication_protocol_sha256":"9".repeat(64),
            "candidate_outcomes":[
                {
                    "candidate_id":"candidate-1",
                    "adjudicator_votes":[
                        {"adjudicator_id":"a1","adjudication":"correct"},
                        {"adjudicator_id":"a2","adjudication":"correct"},
                        {"adjudicator_id":"a3","adjudication":"incorrect"}
                    ],
                    "reviewer_reviews":[
                        {"reviewer_id":"r1","reviewer_decision":"accept","decision_confidence":0.9,"actual_review_minutes":2.2},
                        {"reviewer_id":"r2","reviewer_decision":"accept","decision_confidence":0.8,"actual_review_minutes":1.8},
                        {"reviewer_id":"r3","reviewer_decision":"reject","decision_confidence":0.7,"actual_review_minutes":2.5}
                    ]
                },
                {
                    "candidate_id":"candidate-2",
                    "adjudicator_votes":[
                        {"adjudicator_id":"a1","adjudication":"incorrect"},
                        {"adjudicator_id":"a2","adjudication":"incorrect"},
                        {"adjudicator_id":"a3","adjudication":"indeterminate"}
                    ],
                    "reviewer_reviews":[
                        {"reviewer_id":"r1","reviewer_decision":"reject","decision_confidence":0.8,"actual_review_minutes":1.2},
                        {"reviewer_id":"r2","reviewer_decision":"reject","decision_confidence":0.75,"actual_review_minutes":1.3},
                        {"reviewer_id":"r3","reviewer_decision":"abstain","decision_confidence":null,"actual_review_minutes":3.1}
                    ]
                },
                {
                    "candidate_id":"candidate-3",
                    "adjudicator_votes":[
                        {"adjudicator_id":"a1","adjudication":"correct"},
                        {"adjudicator_id":"a2","adjudication":"incorrect"},
                        {"adjudicator_id":"a3","adjudication":"indeterminate"}
                    ],
                    "reviewer_reviews":[
                        {"reviewer_id":"r1","reviewer_decision":"accept","decision_confidence":0.6,"actual_review_minutes":2.0},
                        {"reviewer_id":"r2","reviewer_decision":"reject","decision_confidence":0.6,"actual_review_minutes":2.1},
                        {"reviewer_id":"r3","reviewer_decision":"abstain","decision_confidence":null,"actual_review_minutes":3.0}
                    ]
                }
            ]
        });
        let proposals: ProposalInput = serde_json::from_value(proposal_value).unwrap();
        let ranking: RankingInput = serde_json::from_value(ranking_value).unwrap();
        let outcomes: OutcomeFile = serde_json::from_value(outcome_value).unwrap();
        (proposals, ranking, outcomes, proposal_hash, ranking_hash)
    }

    fn evaluate_fixture() -> AuditReport {
        let (proposals, _ranking, outcomes, proposal_hash, ranking_hash) = fixture();
        let rows = validate(&proposals, &_ranking, &outcomes, &proposal_hash, &ranking_hash).unwrap();
        evaluate(&proposals, &outcomes, &rows, proposal_hash, ranking_hash, "8".repeat(64))
    }

    #[test]
    fn preserves_raw_votes_and_reports_strict_majorities() {
        let report = evaluate_fixture();
        assert_eq!(report.candidate_count, 3);
        assert_eq!(report.reviewer_count, 3);
        assert_eq!(report.adjudicator_count, 3);
        assert_eq!(report.candidate_summaries[0].adjudication_consensus.as_deref(), Some("correct"));
        assert_eq!(report.candidate_summaries[0].adjudication_votes["incorrect"], 1);
        assert_eq!(report.candidate_summaries[0].reviewer_consensus_decision.as_deref(), Some("accept"));
        assert_eq!(report.candidate_summaries[0].reviewer_decision_votes["reject"], 1);
    }

    #[test]
    fn tied_adjudication_and_reviewer_panel_remain_unresolved() {
        let report = evaluate_fixture();
        let row = report.candidate_summaries.iter().find(|row| row.candidate_id == "candidate-3").unwrap();
        assert_eq!(row.adjudication_consensus, None);
        assert_eq!(row.reviewer_consensus_decision, None);
        assert_eq!(row.determinate_reviewer_consensus_correct, None);
    }

    #[test]
    fn computes_panel_time_and_separates_reviewer_metrics() {
        let report = evaluate_fixture();
        assert!((report.total_panel_review_minutes - 19.2).abs() < 1e-9);
        assert_eq!(report.pooled_reviewer_metrics.review_event_count, 9);
        assert_eq!(report.pooled_reviewer_metrics.abstained_count, 2);
        assert_eq!(report.reviewer_metrics.len(), 3);
    }

    #[test]
    fn abstentions_are_retained_but_excluded_from_confidence_calibration() {
        let report = evaluate_fixture();
        assert_eq!(report.pooled_reviewer_metrics.abstained_count, 2);
        assert!(report.pooled_reviewer_metrics.confidence_calibration_sample_count < 9);
        assert_eq!(report.pooled_reviewer_metrics.confidence_brier_score.is_some(), true);
    }

    #[test]
    fn computes_nominal_alpha_when_categories_vary_but_raters_agree() {
        let units = vec![
            vec!["accept".into(), "accept".into(), "accept".into()],
            vec!["reject".into(), "reject".into(), "reject".into()],
        ];
        let metrics = nominal_alpha(&units);
        assert_eq!(metrics.nominal_krippendorff_alpha, Some(1.0));
    }


    #[test]
    fn nominal_alpha_uses_finite_sample_correction_for_unequal_panel_sizes() {
        let units = vec![
            vec!["a".into(), "a".into()],
            vec!["a".into(), "b".into(), "b".into()],
        ];
        let metrics = nominal_alpha(&units);
        assert!((metrics.nominal_krippendorff_alpha.unwrap() - (1.0 / 3.0)).abs() < 1e-12);
        assert!((metrics.observed_pairwise_disagreement.unwrap() - 0.4).abs() < 1e-12);
        assert!((metrics.expected_pairwise_disagreement.unwrap() - 0.6).abs() < 1e-12);
    }

    #[test]
    fn alpha_is_not_estimable_when_expected_disagreement_is_zero() {
        let units = vec![
            vec!["accept".into(), "accept".into()],
            vec!["accept".into(), "accept".into()],
        ];
        let metrics = nominal_alpha(&units);
        assert_eq!(metrics.nominal_krippendorff_alpha, None);
        assert_eq!(metrics.expected_pairwise_disagreement, Some(0.0));
    }

    #[test]
    fn alpha_is_not_estimable_with_only_single_rater_units() {
        let metrics = nominal_alpha(&[vec!["accept".into()], vec!["reject".into()]]);
        assert_eq!(metrics.nominal_krippendorff_alpha, None);
        assert_eq!(metrics.rated_candidate_count, 0);
    }

    #[test]
    fn rejects_wrong_proposal_or_ranking_digest() {
        let (proposals, ranking, outcomes, proposal_hash, ranking_hash) = fixture();
        assert!(validate(&proposals, &ranking, &outcomes, &"0".repeat(64), &ranking_hash)
            .unwrap_err().contains("proposal SHA-256"));
        assert!(validate(&proposals, &ranking, &outcomes, &proposal_hash, &"0".repeat(64))
            .unwrap_err().contains("ranking_report_sha256"));
    }

    #[test]
    fn rejects_missing_or_extra_candidate_outcomes() {
        let (proposals, ranking, mut outcomes, proposal_hash, ranking_hash) = fixture();
        outcomes.candidate_outcomes.pop();
        assert!(validate(&proposals, &ranking, &outcomes, &proposal_hash, &ranking_hash)
            .unwrap_err().contains("complete frozen candidate pool"));
        outcomes.candidate_outcomes = fixture().2.candidate_outcomes;
        outcomes.candidate_outcomes[0].candidate_id = "not-in-pool".into();
        assert!(validate(&proposals, &ranking, &outcomes, &proposal_hash, &ranking_hash)
            .unwrap_err().contains("absent from frozen proposals"));
    }

    #[test]
    fn rejects_duplicate_reviewer_and_adjudicator_ids_within_candidate() {
        let (proposals, ranking, mut outcomes, proposal_hash, ranking_hash) = fixture();
        outcomes.candidate_outcomes[0].reviewer_reviews[1].reviewer_id = "r1".into();
        assert!(validate(&proposals, &ranking, &outcomes, &proposal_hash, &ranking_hash)
            .unwrap_err().contains("duplicate reviewer ID"));
        let (proposals, ranking, mut outcomes, proposal_hash, ranking_hash) = fixture();
        outcomes.candidate_outcomes[0].adjudicator_votes[1].adjudicator_id = "a1".into();
        assert!(validate(&proposals, &ranking, &outcomes, &proposal_hash, &ranking_hash)
            .unwrap_err().contains("duplicate adjudicator ID"));
    }

    #[test]
    fn rejects_reviewer_adjudicator_role_overlap() {
        let (proposals, ranking, mut outcomes, proposal_hash, ranking_hash) = fixture();
        outcomes.candidate_outcomes[0].reviewer_reviews[0].reviewer_id = "a1".into();
        assert!(validate(&proposals, &ranking, &outcomes, &proposal_hash, &ranking_hash)
            .unwrap_err().contains("IDs must be disjoint"));
    }

    #[test]
    fn rejects_confidence_outside_unit_interval_and_confident_abstention() {
        let (proposals, ranking, mut outcomes, proposal_hash, ranking_hash) = fixture();
        outcomes.candidate_outcomes[0].reviewer_reviews[0].decision_confidence = Some(1.01);
        assert!(validate(&proposals, &ranking, &outcomes, &proposal_hash, &ranking_hash)
            .unwrap_err().contains("decision_confidence"));
        let (proposals, ranking, mut outcomes, proposal_hash, ranking_hash) = fixture();
        outcomes.candidate_outcomes[1].reviewer_reviews[2].decision_confidence = Some(0.8);
        assert!(validate(&proposals, &ranking, &outcomes, &proposal_hash, &ranking_hash)
            .unwrap_err().contains("abstention confidence"));
    }

    #[test]
    fn rejects_non_positive_or_non_finite_review_time() {
        let (proposals, ranking, mut outcomes, proposal_hash, ranking_hash) = fixture();
        outcomes.candidate_outcomes[0].reviewer_reviews[0].actual_review_minutes = 0.0;
        assert!(validate(&proposals, &ranking, &outcomes, &proposal_hash, &ranking_hash)
            .unwrap_err().contains("actual_review_minutes"));
    }

    #[test]
    fn parses_checked_in_multi_rater_template_and_schema() {
        let template: OutcomeFile = serde_json::from_str(include_str!("../../../../docs/neuroscience/examples/CONNECTOME_MULTI_RATER_OUTCOMES.example.json"))
            .expect("multi-rater template parses");
        assert_eq!(template.candidate_outcomes.len(), 3);
        let schema: Value = serde_json::from_str(include_str!("../../../../docs/neuroscience/CONNECTOME_MULTI_RATER_OUTCOMES.schema.json"))
            .expect("multi-rater schema parses");
        assert_eq!(schema["title"], "Connectome multi-rater outcomes");
    }
}
