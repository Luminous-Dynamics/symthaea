//! Deterministic paired draft-to-final revision audit over the production
//! temporal-grounding scorer. It reuses the canonical one-to-one interval
//! matcher and never computes a second temporal alignment.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::temporal_grounding::{
    TemporalGroundingOutcomeV1, TemporalGroundingScoreV1,
};

const REVISION_SCHEMA: &str = "mel-temporal-revision-case-v1";
const EPSILON: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceConditionV1 {
    NoEvidence,
    LocalRetrieved,
    YokedRetrieved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RevisionActionV1 {
    NoChange,
    AddOccurrence,
    RemoveOccurrence,
    SplitMergedOccurrence,
    MergeDuplicateOccurrence,
    AdjustStartBoundary,
    AdjustEndBoundary,
    Other,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevisionEvidenceV1 {
    pub source_audio_sha256: String,
    pub source_sample_rate_hz: u32,
    pub evidence_payload_sha256: String,
    pub retrieval_config_sha256: String,
    pub start_frame: u64,
    pub end_frame: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevisionCaseV1 {
    pub schema: String,
    pub case_id: String,
    pub model_identity_sha256: String,
    pub prompt_sha256: String,
    pub adapter_sha256: String,
    pub evidence_condition: EvidenceConditionV1,
    pub evidence: Option<RevisionEvidenceV1>,
    pub actions: Vec<RevisionActionV1>,
    pub draft: TemporalGroundingScoreV1,
    pub final_score: TemporalGroundingScoreV1,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RevisionScoreSnapshotV1 {
    pub predicted_occurrences: usize,
    pub ground_truth_occurrences: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub f1: Option<f64>,
    pub exact_occurrence_count: Option<bool>,
    pub count_gated_f1: Option<f64>,
    pub correct_absence_claim: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RevisionReportV1 {
    pub case_id: String,
    pub evaluator_version: String,
    pub query_id: String,
    pub audio_sha256: String,
    pub evidence_condition: EvidenceConditionV1,
    pub actions: Vec<RevisionActionV1>,
    pub draft: RevisionScoreSnapshotV1,
    pub final_score: RevisionScoreSnapshotV1,
    pub f1_delta: Option<f64>,
    pub count_gated_f1_delta: Option<f64>,
    pub helpful_revision: Option<bool>,
    pub harmful_revision: Option<bool>,
    pub draft_was_perfect: bool,
    pub preservation_violation: bool,
    pub absence_improved: Option<bool>,
    pub absence_regressed: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RevisionAggregateV1 {
    pub evaluator_version: String,
    pub cases: usize,
    pub positive_cases: usize,
    pub absent_cases: usize,
    pub excluded_cases: usize,
    pub helpful_revisions: usize,
    pub harmful_revisions: usize,
    pub unchanged_positive_revisions: usize,
    pub preservation_violations: usize,
    pub mean_draft_f1: Option<f64>,
    pub mean_final_f1: Option<f64>,
    pub mean_f1_delta: Option<f64>,
    pub one_to_many_cases: usize,
    pub mean_draft_count_gated_f1: Option<f64>,
    pub mean_final_count_gated_f1: Option<f64>,
    pub mean_count_gated_f1_delta: Option<f64>,
    pub absent_claims_scored: usize,
    pub correct_absence_before: usize,
    pub correct_absence_after: usize,
    pub absence_improvements: usize,
    pub absence_regressions: usize,
}

pub fn audit_revision_case(input: &str) -> Result<RevisionReportV1, String> {
    let case: RevisionCaseV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid temporal revision case JSON: {error}"))?;
    audit_revision(&case)
}

pub fn audit_revision(case: &RevisionCaseV1) -> Result<RevisionReportV1, String> {
    if case.schema != REVISION_SCHEMA {
        return Err(format!("unsupported temporal revision schema: {}", case.schema));
    }
    require_text(&case.case_id, "case_id")?;
    validate_digest(&case.model_identity_sha256, "model_identity_sha256")?;
    validate_digest(&case.prompt_sha256, "prompt_sha256")?;
    validate_digest(&case.adapter_sha256, "adapter_sha256")?;
    validate_actions(&case.actions)?;
    validate_evidence(case)?;

    validate_score(&case.draft, "draft")?;
    validate_score(&case.final_score, "final_score")?;
    validate_score_pair(&case.draft, &case.final_score)?;

    let draft = snapshot(&case.draft);
    let final_score = snapshot(&case.final_score);
    let positive = case.draft.outcome == TemporalGroundingOutcomeV1::ScoredPresentQuery;
    let absent = case.draft.outcome == TemporalGroundingOutcomeV1::ScoredAbsentQuery;
    let f1_delta = positive.then(|| {
        final_score.f1.unwrap_or(0.0) - draft.f1.unwrap_or(0.0)
    });
    let count_gated_f1_delta = (positive && case.draft.ground_truth_occurrences > 1).then(|| {
        final_score.count_gated_f1.unwrap_or(0.0)
            - draft.count_gated_f1.unwrap_or(0.0)
    });
    let helpful_revision = f1_delta.map(|delta| delta > EPSILON);
    let harmful_revision = f1_delta.map(|delta| delta < -EPSILON);
    let draft_was_perfect = is_perfect(&case.draft);
    let preservation_violation = draft_was_perfect && !is_perfect(&case.final_score);
    let absence_improved = absent.then(|| {
        case.final_score.correct_absence_claim == Some(true)
            && case.draft.correct_absence_claim != Some(true)
    });
    let absence_regressed = absent.then(|| {
        case.draft.correct_absence_claim == Some(true)
            && case.final_score.correct_absence_claim != Some(true)
    });

    Ok(RevisionReportV1 {
        case_id: case.case_id.clone(),
        evaluator_version: case.draft.evaluator_version.clone(),
        query_id: case.draft.query_id.clone(),
        audio_sha256: case.draft.audio_sha256.clone(),
        evidence_condition: case.evidence_condition,
        actions: case.actions.clone(),
        draft,
        final_score,
        f1_delta,
        count_gated_f1_delta,
        helpful_revision,
        harmful_revision,
        draft_was_perfect,
        preservation_violation,
        absence_improved,
        absence_regressed,
    })
}

pub fn aggregate_revision_reports(
    reports: &[RevisionReportV1],
) -> Result<RevisionAggregateV1, String> {
    let evaluator_version = reports
        .first()
        .map(|report| report.evaluator_version.clone())
        .unwrap_or_default();
    let mut case_ids = BTreeSet::new();
    for report in reports {
        require_text(&report.case_id, "report.case_id")?;
        if report.evaluator_version != evaluator_version {
            return Err("revision aggregate mixes temporal-grounding evaluator versions".into());
        }
        if !case_ids.insert(report.case_id.as_str()) {
            return Err(format!("duplicate revision case_id: {}", report.case_id));
        }
    }

    let positive: Vec<_> = reports
        .iter()
        .filter(|report| report.f1_delta.is_some())
        .collect();
    let one_to_many: Vec<_> = reports
        .iter()
        .filter(|report| report.count_gated_f1_delta.is_some())
        .collect();
    let absent: Vec<_> = reports
        .iter()
        .filter(|report| report.absence_improved.is_some())
        .collect();

    let draft_f1: Vec<_> = positive
        .iter()
        .map(|report| report.draft.f1.unwrap_or(0.0))
        .collect();
    let final_f1: Vec<_> = positive
        .iter()
        .map(|report| report.final_score.f1.unwrap_or(0.0))
        .collect();
    let f1_deltas: Vec<_> = positive
        .iter()
        .map(|report| report.f1_delta.unwrap_or(0.0))
        .collect();
    let draft_gated: Vec<_> = one_to_many
        .iter()
        .map(|report| report.draft.count_gated_f1.unwrap_or(0.0))
        .collect();
    let final_gated: Vec<_> = one_to_many
        .iter()
        .map(|report| report.final_score.count_gated_f1.unwrap_or(0.0))
        .collect();
    let gated_deltas: Vec<_> = one_to_many
        .iter()
        .map(|report| report.count_gated_f1_delta.unwrap_or(0.0))
        .collect();

    Ok(RevisionAggregateV1 {
        evaluator_version,
        cases: reports.len(),
        positive_cases: positive.len(),
        absent_cases: absent.len(),
        excluded_cases: reports.len() - positive.len() - absent.len(),
        helpful_revisions: positive
            .iter()
            .filter(|report| report.helpful_revision == Some(true))
            .count(),
        harmful_revisions: positive
            .iter()
            .filter(|report| report.harmful_revision == Some(true))
            .count(),
        unchanged_positive_revisions: positive
            .iter()
            .filter(|report| {
                report.helpful_revision == Some(false) && report.harmful_revision == Some(false)
            })
            .count(),
        preservation_violations: reports
            .iter()
            .filter(|report| report.preservation_violation)
            .count(),
        mean_draft_f1: mean(&draft_f1),
        mean_final_f1: mean(&final_f1),
        mean_f1_delta: mean(&f1_deltas),
        one_to_many_cases: one_to_many.len(),
        mean_draft_count_gated_f1: mean(&draft_gated),
        mean_final_count_gated_f1: mean(&final_gated),
        mean_count_gated_f1_delta: mean(&gated_deltas),
        absent_claims_scored: absent
            .iter()
            .filter(|report| report.draft.correct_absence_claim.is_some()
                || report.final_score.correct_absence_claim.is_some())
            .count(),
        correct_absence_before: absent
            .iter()
            .filter(|report| report.draft.correct_absence_claim == Some(true))
            .count(),
        correct_absence_after: absent
            .iter()
            .filter(|report| report.final_score.correct_absence_claim == Some(true))
            .count(),
        absence_improvements: absent
            .iter()
            .filter(|report| report.absence_improved == Some(true))
            .count(),
        absence_regressions: absent
            .iter()
            .filter(|report| report.absence_regressed == Some(true))
            .count(),
    })
}

fn validate_actions(actions: &[RevisionActionV1]) -> Result<(), String> {
    if actions.is_empty() {
        return Err("revision actions must contain at least one typed action".into());
    }
    if actions.len() > 8 {
        return Err("revision actions exceed the bounded maximum of eight".into());
    }
    if actions.contains(&RevisionActionV1::NoChange) && actions.len() != 1 {
        return Err("NoChange cannot be combined with other revision actions".into());
    }
    let mut seen = BTreeSet::new();
    for action in actions {
        if !seen.insert(format!("{action:?}")) {
            return Err("revision actions must not repeat".into());
        }
    }
    Ok(())
}

fn validate_evidence(case: &RevisionCaseV1) -> Result<(), String> {
    match (case.evidence_condition, &case.evidence) {
        (EvidenceConditionV1::NoEvidence, None) => Ok(()),
        (EvidenceConditionV1::NoEvidence, Some(_)) => {
            Err("NoEvidence condition must not carry an evidence reference".into())
        }
        (_, None) => Err("retrieved/yoked evidence condition requires an evidence reference".into()),
        (condition, Some(evidence)) => {
            validate_digest(&evidence.source_audio_sha256, "evidence.source_audio_sha256")?;
            validate_digest(
                &evidence.evidence_payload_sha256,
                "evidence.evidence_payload_sha256",
            )?;
            validate_digest(
                &evidence.retrieval_config_sha256,
                "evidence.retrieval_config_sha256",
            )?;
            if evidence.source_sample_rate_hz == 0 || evidence.end_frame <= evidence.start_frame {
                return Err("evidence sample rate and half-open frame window must be valid".into());
            }
            let same_audio = evidence.source_audio_sha256 == case.draft.audio_sha256
                && evidence.source_sample_rate_hz == case.draft.sample_rate_hz;
            match condition {
                EvidenceConditionV1::LocalRetrieved if !same_audio => {
                    Err("local evidence must bind the exact scored audio digest and sample rate".into())
                }
                EvidenceConditionV1::YokedRetrieved if same_audio => {
                    Err("yoked evidence must originate from a different audio artifact".into())
                }
                _ => Ok(()),
            }
        }
    }
}

fn validate_score(
    score: &TemporalGroundingScoreV1,
    label: &str,
) -> Result<(), String> {
    require_text(&score.evaluator_version, &format!("{label}.evaluator_version"))?;
    require_text(&score.query_id, &format!("{label}.query_id"))?;
    validate_digest(&score.audio_sha256, &format!("{label}.audio_sha256"))?;
    if score.sample_rate_hz == 0 {
        return Err(format!("{label}.sample_rate_hz must be positive"));
    }
    for (metric, value) in [
        ("precision", score.precision),
        ("recall", score.recall),
        ("f1", score.f1),
        ("matched_interval_iou_mean", score.matched_interval_iou_mean),
    ] {
        if value.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
            return Err(format!("{label}.{metric} must be finite and in [0,1]"));
        }
    }
    if score.true_positives + score.false_positives != score.predicted_occurrences
        && score.outcome == TemporalGroundingOutcomeV1::ScoredPresentQuery
    {
        return Err(format!("{label} prediction count does not equal TP + FP"));
    }
    if score.true_positives + score.false_negatives != score.ground_truth_occurrences
        && score.outcome == TemporalGroundingOutcomeV1::ScoredPresentQuery
    {
        return Err(format!("{label} truth count does not equal TP + FN"));
    }
    match score.outcome {
        TemporalGroundingOutcomeV1::ScoredPresentQuery if !score.scored || score.f1.is_none() => {
            Err(format!("{label} present query must be scored with F1"))
        }
        TemporalGroundingOutcomeV1::ScoredAbsentQuery if !score.scored => {
            Err(format!("{label} absent query must be scored"))
        }
        TemporalGroundingOutcomeV1::ExcludedAmbiguousGroundTruth
        | TemporalGroundingOutcomeV1::ExcludedNotEstimableGroundTruth
            if score.scored || score.f1.is_some() =>
        {
            Err(format!("{label} excluded truth must not be scored"))
        }
        _ => Ok(()),
    }
}

fn validate_score_pair(
    draft: &TemporalGroundingScoreV1,
    final_score: &TemporalGroundingScoreV1,
) -> Result<(), String> {
    if draft.evaluator_version != final_score.evaluator_version {
        return Err("draft/final scores use different evaluator versions".into());
    }
    if draft.query_id != final_score.query_id {
        return Err("draft/final scores reference different query IDs".into());
    }
    if draft.audio_sha256 != final_score.audio_sha256 {
        return Err("draft/final scores reference different audio artifacts".into());
    }
    if draft.sample_rate_hz != final_score.sample_rate_hz {
        return Err("draft/final scores use different sample rates".into());
    }
    if draft.outcome != final_score.outcome
        || draft.scored != final_score.scored
        || draft.ground_truth_occurrences != final_score.ground_truth_occurrences
    {
        return Err("draft/final score cohort or ground-truth cardinality changed".into());
    }
    Ok(())
}

fn snapshot(score: &TemporalGroundingScoreV1) -> RevisionScoreSnapshotV1 {
    let count_gated_f1 = if score.outcome == TemporalGroundingOutcomeV1::ScoredPresentQuery
        && score.ground_truth_occurrences > 1
    {
        Some(if score.exact_occurrence_count == Some(true) {
            score.f1.unwrap_or(0.0)
        } else {
            0.0
        })
    } else {
        None
    };
    RevisionScoreSnapshotV1 {
        predicted_occurrences: score.predicted_occurrences,
        ground_truth_occurrences: score.ground_truth_occurrences,
        true_positives: score.true_positives,
        false_positives: score.false_positives,
        false_negatives: score.false_negatives,
        f1: score.f1,
        exact_occurrence_count: score.exact_occurrence_count,
        count_gated_f1,
        correct_absence_claim: score.correct_absence_claim,
    }
}

fn is_perfect(score: &TemporalGroundingScoreV1) -> bool {
    match score.outcome {
        TemporalGroundingOutcomeV1::ScoredPresentQuery => {
            score.f1 == Some(1.0)
                && score.false_positives == 0
                && score.false_negatives == 0
                && score.exact_occurrence_count == Some(true)
                && !score.parse_failure
                && !score.abstained
        }
        TemporalGroundingOutcomeV1::ScoredAbsentQuery => {
            score.correct_absence_claim == Some(true) && !score.abstained && !score.parse_failure
        }
        TemporalGroundingOutcomeV1::ExcludedAmbiguousGroundTruth
        | TemporalGroundingOutcomeV1::ExcludedNotEstimableGroundTruth => false,
    }
}

fn validate_digest(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} must be a lowercase SHA-256 digest"));
    }
    Ok(())
}

fn require_text(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{label} must not be empty"));
    }
    Ok(())
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal_grounding::{
        GroundingPredictionKindV1, TemporalGroundingOutcomeV1,
    };

    const EVALUATOR: &str = "mel-temporal-grounding-v1";
    const AUDIO: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const MODEL: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const PROMPT: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const ADAPTER: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const EVIDENCE: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const RETRIEVAL: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn positive_score(
        f1: f64,
        tp: usize,
        fp: usize,
        fn_: usize,
    ) -> TemporalGroundingScoreV1 {
        let predicted = tp + fp;
        let truth = tp + fn_;
        let precision = if predicted == 0 { 0.0 } else { tp as f64 / predicted as f64 };
        let recall = if truth == 0 { 0.0 } else { tp as f64 / truth as f64 };
        TemporalGroundingScoreV1 {
            evaluator_version: EVALUATOR.into(),
            query_id: "query-1".into(),
            audio_sha256: AUDIO.into(),
            sample_rate_hz: 48_000,
            outcome: TemporalGroundingOutcomeV1::ScoredPresentQuery,
            prediction_kind: GroundingPredictionKindV1::Occurrences,
            scored: true,
            abstained: false,
            parse_failure: false,
            ground_truth_occurrences: truth,
            predicted_occurrences: predicted,
            ambiguous_candidate_count: 0,
            true_positives: tp,
            false_positives: fp,
            false_negatives: fn_,
            precision: Some(precision),
            recall: Some(recall),
            f1: Some(f1),
            matched_interval_iou_mean: Some(f1),
            count_absolute_error: Some(fp.abs_diff(fn_)),
            exact_occurrence_count: Some(fp == fn_),
            correct_absence_claim: None,
            matches: Vec::new(),
        }
    }

    fn case(draft: TemporalGroundingScoreV1, final_score: TemporalGroundingScoreV1) -> RevisionCaseV1 {
        RevisionCaseV1 {
            schema: REVISION_SCHEMA.into(),
            case_id: "case-1".into(),
            model_identity_sha256: MODEL.into(),
            prompt_sha256: PROMPT.into(),
            adapter_sha256: ADAPTER.into(),
            evidence_condition: EvidenceConditionV1::NoEvidence,
            evidence: None,
            actions: vec![RevisionActionV1::Other],
            draft,
            final_score,
        }
    }

    #[test]
    fn better_final_revision_reports_gain_without_claiming_causality() {
        let draft = positive_score(0.5, 1, 1, 1);
        let final_score = positive_score(1.0, 2, 0, 0);
        let report = audit_revision(&case(draft, final_score)).unwrap();
        assert_eq!(report.f1_delta, Some(0.5));
        assert_eq!(report.helpful_revision, Some(true));
        assert_eq!(report.harmful_revision, Some(false));
        assert_eq!(report.draft.count_gated_f1, Some(0.0));
        assert_eq!(report.final_score.count_gated_f1, Some(1.0));
        assert_eq!(report.count_gated_f1_delta, Some(1.0));
    }

    #[test]
    fn revision_that_damages_a_perfect_draft_is_a_preservation_violation() {
        let draft = positive_score(1.0, 2, 0, 0);
        let final_score = positive_score(0.5, 1, 1, 1);
        let report = audit_revision(&case(draft, final_score)).unwrap();
        assert!(report.draft_was_perfect);
        assert!(report.preservation_violation);
        assert_eq!(report.harmful_revision, Some(true));
    }

    #[test]
    fn cross_query_or_cross_audio_draft_final_pairs_are_rejected() {
        let draft = positive_score(0.5, 1, 1, 1);
        let mut final_score = positive_score(1.0, 2, 0, 0);
        final_score.audio_sha256 = "1212121212121212121212121212121212121212121212121212121212121212".into();
        assert!(audit_revision(&case(draft, final_score)).is_err());
    }

    #[test]
    fn absence_revision_reports_improvement_without_relabeling_abstention_as_absence() {
        let absent = audit_revision(&case(
            absent_score(Some(false)),
            absent_score(Some(true)),
        ))
        .unwrap();
        assert_eq!(absent.absence_improved, Some(true));
        assert_eq!(absent.absence_regressed, Some(false));
        assert_eq!(absent.f1_delta, None);

        let abstained = audit_revision(&case(
            absent_score(Some(false)),
            absent_score(None),
        ))
        .unwrap();
        assert_eq!(abstained.absence_improved, Some(false));
        assert_eq!(abstained.absence_regressed, Some(false));
        assert_eq!(abstained.final_score.correct_absence_claim, None);
    }

    #[test]
    fn valid_yoked_evidence_requires_a_distinct_source_recording() {
        let mut yoked = case(positive_score(0.5, 1, 1, 1), positive_score(1.0, 2, 0, 0));
        yoked.evidence_condition = EvidenceConditionV1::YokedRetrieved;
        yoked.evidence = Some(RevisionEvidenceV1 {
            source_audio_sha256: "1212121212121212121212121212121212121212121212121212121212121212".into(),
            source_sample_rate_hz: 44_100,
            evidence_payload_sha256: EVIDENCE.into(),
            retrieval_config_sha256: RETRIEVAL.into(),
            start_frame: 100,
            end_frame: 200,
        });
        assert!(audit_revision(&yoked).is_ok());
    }

    #[test]
    fn local_and_yoked_evidence_have_distinct_source_binding_rules() {
        let mut local = case(positive_score(0.5, 1, 1, 1), positive_score(1.0, 2, 0, 0));
        local.evidence_condition = EvidenceConditionV1::LocalRetrieved;
        local.evidence = Some(RevisionEvidenceV1 {
            source_audio_sha256: AUDIO.into(),
            source_sample_rate_hz: 48_000,
            evidence_payload_sha256: EVIDENCE.into(),
            retrieval_config_sha256: RETRIEVAL.into(),
            start_frame: 100,
            end_frame: 200,
        });
        assert!(audit_revision(&local).is_ok());
        local.evidence_condition = EvidenceConditionV1::YokedRetrieved;
        assert!(audit_revision(&local).is_err());
    }

    #[test]
    fn aggregate_reports_helpful_harmful_and_preservation_denominators_separately() {
        let improved = audit_revision(&case(
            positive_score(0.5, 1, 1, 1),
            positive_score(1.0, 2, 0, 0),
        )).unwrap();
        let harmed = audit_revision(&case(
            positive_score(1.0, 2, 0, 0),
            positive_score(0.5, 1, 1, 1),
        )).unwrap();
        let mut harmed = harmed;
        harmed.case_id = "case-2".into();
        let aggregate = aggregate_revision_reports(&[improved, harmed]).unwrap();
        assert_eq!(aggregate.cases, 2);
        assert_eq!(aggregate.positive_cases, 2);
        assert_eq!(aggregate.helpful_revisions, 1);
        assert_eq!(aggregate.harmful_revisions, 1);
        assert_eq!(aggregate.preservation_violations, 1);
        assert_eq!(aggregate.mean_f1_delta, Some(0.0));
    }
}
