use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

const SCHEMA_VERSION: &str = "mel-event-universe-panel-v1";
const LOG_EPSILON: f64 = 1e-15;
const CALIBRATION_BINS: usize = 10;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EventUniversePanelV1 {
    schema_version: String,
    experiment_id: String,
    universe: OpportunityUniverseV1,
    candidate: CandidateForecastV1,
    policy: ScoringPolicyV1,
    predictions: Vec<SlotPredictionV1>,
    claim_ceiling: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpportunityUniverseV1 {
    universe_id: String,
    universe_frozen_at_ns: u64,
    labels_adjudicated_at_ns: u64,
    task_family: String,
    canonical_time_map_id: String,
    canonical_time_map_version: String,
    canonical_time_map_kind: String,
    canonical_time_map_clock_domain_id: String,
    evaluator_clock_domain_id: String,
    time_alignment_receipt_sha256: String,
    time_alignment_valid_from_ns: u64,
    time_alignment_valid_until_ns: u64,
    time_alignment_max_uncertainty_ns: u64,
    opportunity_manifest_sha256: String,
    label_protocol_sha256: String,
    adjudication_receipt_sha256: String,
    slots: Vec<OpportunitySlotV1>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpportunitySlotV1 {
    slot_id: String,
    role: String,
    beat_index: u64,
    target_start_ns: u64,
    target_end_ns: u64,
    local_beat_period_ns: u64,
    alternatives: Vec<String>,
    truth_status: String,
    actual_alternative: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateForecastV1 {
    model_id: String,
    model_version: String,
    model_sha256: String,
    forecast_payload_sha256: String,
    committed_at_ns: u64,
    input_snapshot_sha256: String,
    policy_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScoringPolicyV1 {
    presence_decision_threshold: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SlotPredictionV1 {
    slot_id: String,
    presence_probability: f64,
    alternative_probabilities: Vec<AlternativeProbabilityV1>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AlternativeProbabilityV1 {
    alternative_id: String,
    conditional_probability: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CalibrationBin {
    pub bin_index: usize,
    pub count: usize,
    pub mean_predicted_presence: Option<f64>,
    pub observed_presence_rate: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SlotFinding {
    pub slot_id: String,
    pub role: String,
    pub beat_index: u64,
    pub truth_status: String,
    pub actual_alternative: Option<String>,
    pub predicted: bool,
    pub omitted_prediction: bool,
    pub presence_probability: f64,
    pub predicted_alternative: Option<String>,
    pub predicted_alternative_probability: Option<f64>,
    pub observed_presence: Option<bool>,
    pub presence_brier_score: Option<f64>,
    pub categorical_brier_score: Option<f64>,
    pub categorical_log_loss: Option<f64>,
    pub top1_correct: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RoleSummary {
    pub role: String,
    pub slot_count: usize,
    pub scored_slot_count: usize,
    pub omitted_positive_count: usize,
    pub false_positive_count: usize,
    pub presence_brier_score: Option<f64>,
    pub recall: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EventUniverseReport {
    pub schema_version: String,
    pub experiment_id: String,
    pub universe_id: String,
    pub candidate_model_id: String,
    pub candidate_model_version: String,
    pub canonical_time_map_id: String,
    pub canonical_time_map_version: String,
    pub opportunity_slot_count: usize,
    pub scored_slot_count: usize,
    pub ambiguous_slot_count: usize,
    pub prediction_record_count: usize,
    pub prediction_coverage_rate: f64,
    pub omitted_slot_count: usize,
    pub omitted_positive_count: usize,
    pub false_positive_count: usize,
    pub true_positive_count: usize,
    pub false_negative_count: usize,
    pub true_negative_count: usize,
    pub presence_precision: Option<f64>,
    pub presence_recall: Option<f64>,
    pub presence_f1: Option<f64>,
    pub presence_brier_score: Option<f64>,
    pub categorical_brier_score: Option<f64>,
    pub categorical_log_loss: Option<f64>,
    pub top1_accuracy: Option<f64>,
    pub presence_expected_calibration_error: Option<f64>,
    pub calibration_bins: Vec<CalibrationBin>,
    pub per_role: Vec<RoleSummary>,
    pub findings: Vec<SlotFinding>,
    pub structurally_valid_panel: bool,
    pub qualification_findings: Vec<String>,
    pub claim_ceiling: Vec<String>,
}

#[derive(Debug, Default)]
struct ScoreAccumulator {
    count: usize,
    presence_brier_sum: f64,
    categorical_brier_sum: f64,
    categorical_log_loss_sum: f64,
    top1_correct: usize,
}

#[derive(Debug, Default)]
struct CalibrationAccumulator {
    count: usize,
    probability_sum: f64,
    observed_sum: f64,
}

#[derive(Debug, Default)]
struct RoleAccumulator {
    slots: usize,
    scored: usize,
    omitted_positive: usize,
    false_positive: usize,
    true_positive: usize,
    false_negative: usize,
    presence_brier_sum: f64,
}

pub fn parse_and_score(input: &str) -> Result<EventUniverseReport, String> {
    let panel: EventUniversePanelV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid event-universe panel JSON: {error}"))?;
    score_panel(&panel)
}

fn score_panel(panel: &EventUniversePanelV1) -> Result<EventUniverseReport, String> {
    validate_header(panel)?;

    let mut slot_by_id = BTreeMap::<&str, &OpportunitySlotV1>::new();
    for slot in &panel.universe.slots {
        validate_slot(slot, panel)?;
        if slot_by_id.insert(slot.slot_id.as_str(), slot).is_some() {
            return Err(format!("duplicate opportunity slot_id: {}", slot.slot_id));
        }
    }

    let mut prediction_by_slot = BTreeMap::<&str, &SlotPredictionV1>::new();
    for prediction in &panel.predictions {
        if prediction_by_slot
            .insert(prediction.slot_id.as_str(), prediction)
            .is_some()
        {
            return Err(format!("duplicate prediction for slot {}", prediction.slot_id));
        }
        let slot = slot_by_id
            .get(prediction.slot_id.as_str())
            .copied()
            .ok_or_else(|| format!("prediction references unknown opportunity slot {}", prediction.slot_id))?;
        validate_prediction(prediction, slot, &panel.policy)?;
    }

    let mut findings = Vec::new();
    let mut scoring = ScoreAccumulator::default();
    let mut bins = (0..CALIBRATION_BINS)
        .map(|_| CalibrationAccumulator::default())
        .collect::<Vec<_>>();
    let mut roles = BTreeMap::<String, RoleAccumulator>::new();

    let mut ambiguous_slot_count = 0usize;
    let mut omitted_slot_count = 0usize;
    let mut omitted_positive_count = 0usize;
    let mut false_positive_count = 0usize;
    let mut true_positive_count = 0usize;
    let mut false_negative_count = 0usize;
    let mut true_negative_count = 0usize;

    for slot in &panel.universe.slots {
        let prediction = prediction_by_slot.get(slot.slot_id.as_str()).copied();
        let omitted = prediction.is_none();
        let presence_probability = prediction.map_or(0.0, |item| item.presence_probability);
        let role = roles.entry(slot.role.clone()).or_default();
        role.slots += 1;

        if omitted {
            omitted_slot_count += 1;
        }

        if slot.truth_status == "ambiguous" {
            ambiguous_slot_count += 1;
            findings.push(SlotFinding {
                slot_id: slot.slot_id.clone(),
                role: slot.role.clone(),
                beat_index: slot.beat_index,
                truth_status: slot.truth_status.clone(),
                actual_alternative: None,
                predicted: !omitted && presence_probability >= panel.policy.presence_decision_threshold,
                omitted_prediction: omitted,
                presence_probability,
                predicted_alternative: None,
                predicted_alternative_probability: None,
                observed_presence: None,
                presence_brier_score: None,
                categorical_brier_score: None,
                categorical_log_loss: None,
                top1_correct: None,
            });
            continue;
        }

        let observed_presence = slot.truth_status == "present";
        let prediction_positive = !omitted
            && presence_probability >= panel.policy.presence_decision_threshold;
        if omitted && observed_presence {
            omitted_positive_count += 1;
            false_negative_count += 1;
            role.false_negative += 1;
            role.omitted_positive += 1;
        } else if prediction_positive && observed_presence {
            true_positive_count += 1;
            role.true_positive += 1;
        } else if prediction_positive && !observed_presence {
            false_positive_count += 1;
            role.false_positive += 1;
        } else if !prediction_positive && !observed_presence {
            true_negative_count += 1;
        } else {
            false_negative_count += 1;
            role.false_negative += 1;
        }

        let presence_brier = (presence_probability - if observed_presence { 1.0 } else { 0.0 }).powi(2);
        let mut class_probabilities = Vec::with_capacity(slot.alternatives.len() + 1);
        class_probabilities.push(1.0 - presence_probability);
        if let Some(prediction) = prediction {
            let alternative_map: BTreeMap<&str, f64> = prediction
                .alternative_probabilities
                .iter()
                .map(|item| (item.alternative_id.as_str(), item.conditional_probability))
                .collect();
            for alternative in &slot.alternatives {
                class_probabilities.push(
                    presence_probability
                        * alternative_map.get(alternative.as_str()).copied().unwrap_or(0.0),
                );
            }
        } else {
            class_probabilities.extend(std::iter::repeat_n(0.0, slot.alternatives.len()));
        }

        let truth_index = if observed_presence {
            slot.alternatives
                .iter()
                .position(|alternative| Some(alternative.as_str()) == slot.actual_alternative.as_deref())
                .ok_or_else(|| format!("slot {} has an invalid present-label alternative", slot.slot_id))?
                + 1
        } else {
            0
        };

        let mut categorical_brier = 0.0;
        for (index, probability) in class_probabilities.iter().enumerate() {
            let target = if index == truth_index { 1.0 } else { 0.0 };
            categorical_brier += (probability - target).powi(2);
        }
        let categorical_log_loss = -class_probabilities[truth_index].max(LOG_EPSILON).ln();
        let top1_index = class_probabilities
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .unwrap_or(0);
        let top1_correct = top1_index == truth_index;

        scoring.count += 1;
        scoring.presence_brier_sum += presence_brier;
        scoring.categorical_brier_sum += categorical_brier;
        scoring.categorical_log_loss_sum += categorical_log_loss;
        scoring.top1_correct += usize::from(top1_correct);

        role.scored += 1;
        role.presence_brier_sum += presence_brier;

        let bin_index = ((presence_probability * CALIBRATION_BINS as f64).floor() as usize)
            .min(CALIBRATION_BINS - 1);
        let bin = &mut bins[bin_index];
        bin.count += 1;
        bin.probability_sum += presence_probability;
        bin.observed_sum += if observed_presence { 1.0 } else { 0.0 };

        let predicted_alternative = prediction.and_then(|item| {
            item.alternative_probabilities
                .iter()
                .max_by(|left, right| left.conditional_probability.total_cmp(&right.conditional_probability))
                .map(|item| item.alternative_id.clone())
        });
        let predicted_alternative_probability = prediction.and_then(|item| {
            item.alternative_probabilities
                .iter()
                .max_by(|left, right| left.conditional_probability.total_cmp(&right.conditional_probability))
                .map(|item| item.conditional_probability)
        });

        findings.push(SlotFinding {
            slot_id: slot.slot_id.clone(),
            role: slot.role.clone(),
            beat_index: slot.beat_index,
            truth_status: slot.truth_status.clone(),
            actual_alternative: slot.actual_alternative.clone(),
            predicted: prediction_positive,
            omitted_prediction: omitted,
            presence_probability,
            predicted_alternative,
            predicted_alternative_probability,
            observed_presence: Some(observed_presence),
            presence_brier_score: Some(presence_brier),
            categorical_brier_score: Some(categorical_brier),
            categorical_log_loss: Some(categorical_log_loss),
            top1_correct: Some(top1_correct),
        });
    }

    let prediction_coverage_rate = if panel.universe.slots.is_empty() {
        0.0
    } else {
        prediction_by_slot.len() as f64 / panel.universe.slots.len() as f64
    };
    let ece = expected_calibration_error(&bins, scoring.count);
    let calibration_bins = bins
        .iter()
        .enumerate()
        .map(|(index, item)| CalibrationBin {
            bin_index: index,
            count: item.count,
            mean_predicted_presence: average(item.probability_sum, item.count),
            observed_presence_rate: average(item.observed_sum, item.count),
        })
        .collect();

    let per_role = roles
        .into_iter()
        .map(|(role, item)| RoleSummary {
            role,
            slot_count: item.slots,
            scored_slot_count: item.scored,
            omitted_positive_count: item.omitted_positive,
            false_positive_count: item.false_positive,
            presence_brier_score: average(item.presence_brier_sum, item.scored),
            recall: ratio(item.true_positive, item.true_positive + item.false_negative),
        })
        .collect();

    let precision = ratio(true_positive_count, true_positive_count + false_positive_count);
    let recall = ratio(true_positive_count, true_positive_count + false_negative_count);
    let f1 = match (precision, recall) {
        (Some(p), Some(r)) if p + r > 0.0 => Some(2.0 * p * r / (p + r)),
        (Some(_), Some(_)) => Some(0.0),
        _ => None,
    };

    let structurally_valid = findings.len() == panel.universe.slots.len();
    Ok(EventUniverseReport {
        schema_version: "mel-event-universe-report-v1".to_owned(),
        experiment_id: panel.experiment_id.clone(),
        universe_id: panel.universe.universe_id.clone(),
        candidate_model_id: panel.candidate.model_id.clone(),
        candidate_model_version: panel.candidate.model_version.clone(),
        canonical_time_map_id: panel.universe.canonical_time_map_id.clone(),
        canonical_time_map_version: panel.universe.canonical_time_map_version.clone(),
        opportunity_slot_count: panel.universe.slots.len(),
        scored_slot_count: scoring.count,
        ambiguous_slot_count,
        prediction_record_count: panel.predictions.len(),
        prediction_coverage_rate,
        omitted_slot_count,
        omitted_positive_count,
        false_positive_count,
        true_positive_count,
        false_negative_count,
        true_negative_count,
        presence_precision: precision,
        presence_recall: recall,
        presence_f1: f1,
        presence_brier_score: average(scoring.presence_brier_sum, scoring.count),
        categorical_brier_score: average(scoring.categorical_brier_sum, scoring.count),
        categorical_log_loss: average(scoring.categorical_log_loss_sum, scoring.count),
        top1_accuracy: ratio(scoring.top1_correct, scoring.count),
        presence_expected_calibration_error: ece,
        calibration_bins,
        per_role,
        findings,
        structurally_valid_panel: structurally_valid,
        qualification_findings: Vec::new(),
        claim_ceiling: panel.claim_ceiling.clone(),
    })
}

fn validate_header(panel: &EventUniversePanelV1) -> Result<(), String> {
    if panel.schema_version != SCHEMA_VERSION || panel.experiment_id.trim().is_empty() {
        return Err("schema_version or experiment_id is invalid".to_owned());
    }
    let universe = &panel.universe;
    require_text(&universe.universe_id, "universe_id")?;
    require_text(&universe.task_family, "task_family")?;
    require_text(&universe.canonical_time_map_id, "canonical_time_map_id")?;
    require_text(&universe.canonical_time_map_version, "canonical_time_map_version")?;
    require_text(&universe.canonical_time_map_clock_domain_id, "canonical_time_map_clock_domain_id")?;
    require_text(&universe.evaluator_clock_domain_id, "evaluator_clock_domain_id")?;
    if universe.canonical_time_map_kind != "performance_time" {
        return Err("event-universe onset scoring requires canonical performance_time".to_owned());
    }
    require_digest(&universe.time_alignment_receipt_sha256, "time alignment receipt")?;
    require_digest(&universe.opportunity_manifest_sha256, "opportunity manifest")?;
    require_digest(&universe.label_protocol_sha256, "label protocol")?;
    require_digest(&universe.adjudication_receipt_sha256, "adjudication receipt")?;
    if universe.time_alignment_valid_from_ns >= universe.time_alignment_valid_until_ns {
        return Err("time alignment validity interval must have positive duration".to_owned());
    }
    if universe.slots.is_empty() {
        return Err("opportunity universe must contain at least one slot".to_owned());
    }
    if universe.labels_adjudicated_at_ns < universe.slots.iter().map(|slot| slot.target_end_ns).max().unwrap_or(0) {
        return Err("labels were adjudicated before all target windows ended".to_owned());
    }
    require_text(&panel.candidate.model_id, "candidate model_id")?;
    require_text(&panel.candidate.model_version, "candidate model_version")?;
    require_digest(&panel.candidate.model_sha256, "candidate model digest")?;
    require_digest(&panel.candidate.forecast_payload_sha256, "forecast payload digest")?;
    require_digest(&panel.candidate.input_snapshot_sha256, "candidate input snapshot")?;
    require_digest(&panel.candidate.policy_sha256, "candidate policy digest")?;
    if panel.candidate.committed_at_ns <= universe.universe_frozen_at_ns {
        return Err("candidate forecast must be committed after the opportunity universe was frozen".to_owned());
    }
    if panel.candidate.committed_at_ns >= panel.universe.slots.iter().map(|slot| slot.target_start_ns).min().unwrap_or(0) {
        return Err("candidate forecast was committed after an opportunity target began".to_owned());
    }
    if panel.candidate.committed_at_ns >= universe.labels_adjudicated_at_ns {
        return Err("candidate forecast was committed after reference labels were adjudicated (label leakage)".to_owned());
    }
    if !panel.policy.presence_decision_threshold.is_finite()
        || !(0.0..=1.0).contains(&panel.policy.presence_decision_threshold)
    {
        return Err("invalid event-universe scoring policy".to_owned());
    }
    if panel.claim_ceiling.is_empty() || panel.claim_ceiling.iter().any(|claim| claim.trim().is_empty()) {
        return Err("claim_ceiling must contain explicit limits".to_owned());
    }
    Ok(())
}

fn validate_slot(slot: &OpportunitySlotV1, panel: &EventUniversePanelV1) -> Result<(), String> {
    require_text(&slot.slot_id, "slot_id")?;
    require_text(&slot.role, "slot role")?;
    if slot.target_start_ns >= slot.target_end_ns || slot.local_beat_period_ns == 0 {
        return Err(format!("slot {} has invalid time or beat period", slot.slot_id));
    }
    if slot.target_start_ns < panel.universe.time_alignment_valid_from_ns
        || slot.target_end_ns > panel.universe.time_alignment_valid_until_ns
    {
        return Err(format!("slot {} falls outside the time alignment validity interval", slot.slot_id));
    }
    if slot.alternatives.is_empty() {
        return Err(format!("slot {} must declare at least one admissible alternative", slot.slot_id));
    }
    let mut alternatives = BTreeSet::new();
    for alternative in &slot.alternatives {
        require_text(alternative, "alternative ID")?;
        if !alternatives.insert(alternative.as_str()) {
            return Err(format!("slot {} contains duplicate alternative", slot.slot_id));
        }
    }
    match slot.truth_status.as_str() {
        "present" => {
            let actual = slot.actual_alternative.as_deref().ok_or_else(|| format!("present slot {} needs an actual alternative", slot.slot_id))?;
            if !alternatives.contains(actual) {
                return Err(format!("slot {} actual alternative is not in its admissible set", slot.slot_id));
            }
        }
        "absent" | "ambiguous" => {
            if slot.actual_alternative.is_some() {
                return Err(format!("{} slot {} must not carry an actual alternative", slot.truth_status, slot.slot_id));
            }
        }
        other => return Err(format!("unsupported truth_status {other}")),
    }
    Ok(())
}

fn validate_prediction(prediction: &SlotPredictionV1, slot: &OpportunitySlotV1, policy: &ScoringPolicyV1) -> Result<(), String> {
    require_text(&prediction.slot_id, "prediction slot_id")?;
    if !prediction.presence_probability.is_finite()
        || !(0.0..=1.0).contains(&prediction.presence_probability)
    {
        return Err(format!("slot {} presence probability is invalid", slot.slot_id));
    }
    if prediction.presence_probability == 0.0 {
        if !prediction.alternative_probabilities.is_empty() {
            return Err(format!("slot {} near-zero presence probability must not carry conditional alternatives", slot.slot_id));
        }
        return Ok(());
    }
    let mut probabilities = BTreeMap::new();
    for item in &prediction.alternative_probabilities {
        require_text(&item.alternative_id, "alternative probability ID")?;
        if !item.conditional_probability.is_finite()
            || !(0.0..=1.0).contains(&item.conditional_probability)
        {
            return Err(format!("slot {} has invalid conditional alternative probability", slot.slot_id));
        }
        if probabilities.insert(item.alternative_id.as_str(), item.conditional_probability).is_some() {
            return Err(format!("slot {} has duplicate alternative probability", slot.slot_id));
        }
    }
    if probabilities.len() != slot.alternatives.len()
        || slot.alternatives.iter().any(|alternative| !probabilities.contains_key(alternative.as_str()))
    {
        return Err(format!("slot {} must predict every admissible alternative exactly once", slot.slot_id));
    }
    let sum: f64 = probabilities.values().sum();
    if (sum - 1.0).abs() > 1e-6 {
        return Err(format!("slot {} conditional alternative probabilities must sum to one", slot.slot_id));
    }
    Ok(())
}

fn expected_calibration_error(bins: &[CalibrationAccumulator], total: usize) -> Option<f64> {
    if total == 0 {
        return None;
    }
    let error = bins.iter().filter(|bin| bin.count > 0).map(|bin| {
        (bin.count as f64 / total as f64)
            * ((bin.probability_sum / bin.count as f64) - (bin.observed_sum / bin.count as f64)).abs()
    }).sum();
    Some(error)
}

fn average(sum: f64, count: usize) -> Option<f64> {
    if count == 0 { None } else { Some(sum / count as f64) }
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    if denominator == 0 { None } else { Some(numerator as f64 / denominator as f64) }
}

fn require_text(value: &str, name: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{name} must be non-empty"))
    } else {
        Ok(())
    }
}

fn require_digest(value: &str, name: &str) -> Result<(), String> {
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)) {
        Ok(())
    } else {
        Err(format!("{name} must be a lowercase 64-character SHA-256 digest"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const FIXTURE: &str = include_str!("../fixtures/valid-event-universe-v1.json");

    fn panel() -> Value {
        serde_json::from_str(FIXTURE).expect("known-answer event universe fixture")
    }

    fn score(value: &Value) -> Result<EventUniverseReport, String> {
        parse_and_score(&value.to_string())
    }

    #[test]
    fn known_answer_scores_every_universe_slot_including_omissions() {
        let report = parse_and_score(FIXTURE).expect("known event universe must score");
        assert_eq!(report.opportunity_slot_count, 6);
        assert_eq!(report.scored_slot_count, 5);
        assert_eq!(report.ambiguous_slot_count, 1);
        assert_eq!(report.omitted_positive_count, 1);
        assert_eq!(report.omitted_slot_count, 2);
        assert_eq!(report.false_positive_count, 1);
        assert_eq!(report.true_positive_count, 2);
        assert_eq!(report.false_negative_count, 1);
        assert_eq!(report.true_negative_count, 1);
        assert!(report.presence_brier_score.is_some());
        assert!(report.categorical_brier_score.is_some());
        assert!(report.categorical_log_loss.is_some());
        assert!(report.presence_expected_calibration_error.is_some());
        assert!(report.structurally_valid_panel);
    }

    #[test]
    fn missing_prediction_is_scored_as_zero_presence_not_dropped() {
        let mut value = panel();
        value["predictions"].as_array_mut().unwrap().retain(|item| item["slot_id"] != "slot-kick");
        let report = score(&value).expect("missing candidate prediction is a measurable omission");
        assert_eq!(report.omitted_positive_count, 2);
        assert_eq!(report.scored_slot_count, 5);
    }

    #[test]
    fn unknown_prediction_slot_is_rejected() {
        let mut value = panel();
        value["predictions"][0]["slot_id"] = serde_json::json!("not-in-universe");
        let error = score(&value).expect_err("unknown slot cannot be smuggled into score");
        assert!(error.contains("unknown opportunity slot"));
    }

    #[test]
    fn duplicate_prediction_slot_is_rejected() {
        let mut value = panel();
        let duplicate = value["predictions"][0].clone();
        value["predictions"].as_array_mut().unwrap().push(duplicate);
        let error = score(&value).expect_err("duplicate prediction must fail");
        assert!(error.contains("duplicate prediction"));
    }

    #[test]
    fn candidate_cannot_commit_after_reference_labels_are_adjudicated() {
        let mut value = panel();
        value["candidate"]["committed_at_ns"] = value["universe"]["labels_adjudicated_at_ns"].clone();
        let error = score(&value).expect_err("forecast after labels is leakage");
        assert!(error.contains("after reference labels were adjudicated"));
    }

    #[test]
    fn candidate_must_commit_after_opportunity_universe_is_frozen() {
        let mut value = panel();
        value["candidate"]["committed_at_ns"] = value["universe"]["universe_frozen_at_ns"].clone();
        let error = score(&value).expect_err("candidate cannot predate universe freeze");
        assert!(error.contains("after the opportunity universe was frozen"));
    }

    #[test]
    fn present_slot_must_name_admissible_actual_alternative() {
        let mut value = panel();
        value["universe"]["slots"][0]["actual_alternative"] = serde_json::json!("not-admissible");
        let error = score(&value).expect_err("actual alternative outside reference set fails");
        assert!(error.contains("not in its admissible set"));
    }

    #[test]
    fn conditional_alternative_probability_must_sum_to_one() {
        let mut value = panel();
        value["predictions"][0]["alternative_probabilities"][0]["conditional_probability"] =
            serde_json::json!(0.2);
        let error = score(&value).expect_err("conditional distribution must normalize");
        assert!(error.contains("sum to one"));
    }

    #[test]
    fn ambiguous_reference_slots_are_counted_but_not_scored() {
        let report = parse_and_score(FIXTURE).expect("ambiguous event should be preserved");
        let ambiguous = report.findings.iter().find(|finding| finding.truth_status == "ambiguous").unwrap();
        assert!(ambiguous.presence_brier_score.is_none());
        assert_eq!(report.ambiguous_slot_count, 1);
    }

    #[test]
    fn zero_presence_prediction_does_not_require_conditional_labels() {
        let mut value = panel();
        value["predictions"][0]["presence_probability"] = serde_json::json!(0.0);
        value["predictions"][0]["alternative_probabilities"] = serde_json::json!([]);
        let report = score(&value).expect("zero presence forecast has all mass on no-event");
        assert!(report.categorical_log_loss.is_some());
    }

    #[test]
    fn time_alignment_domain_mismatch_is_rejected() {
        let mut value = panel();
        value["universe"]["evaluator_clock_domain_id"] = serde_json::json!("wrong-clock");
        let error = score(&value).expect_err("time domains cannot be mixed");
        assert!(error.contains("time alignment source/target"));
    }
}
