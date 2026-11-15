//! Evaluator-only D6.8 multi-shift evidence-continuity scorer.
//!
//! This module consumes a synthetic or independently adjudicated evaluator
//! bundle. It does not create a general evidence graph, authenticate receipts,
//! execute checks, or enforce filesystem custody. Never expose its reports or
//! gold-label inputs to candidate agents.

use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, FixedOffset};
use serde::Serialize;
use serde_json::Value;

const STATE_DIMENSIONS: &[&str] = &[
    "source_tree_digest",
    "runner_image_digest",
    "toolchain_digest",
    "policy_digest",
    "issuer_key_digest",
    "api_contract_digest",
    "docs_contract_digest",
];

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ReceiptIssue {
    pub obligation_id: String,
    pub receipt_kind: String,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct MultiShiftEpochReport {
    pub epoch: u64,
    pub event_id: String,
    pub decision_cutoff: String,
    pub required_revalidation_obligations: Vec<String>,
    pub dependency_implied_revalidation_obligations: Vec<String>,
    pub dependency_only_impact_obligations: Vec<String>,
    pub gold_only_impact_obligations: Vec<String>,
    pub fresh_revalidated_obligations: Vec<String>,
    pub fresh_passing_obligations: Vec<String>,
    pub reused_current_obligations: Vec<String>,
    pub missing_required_revalidation_obligations: Vec<String>,
    pub required_without_fresh_pass_obligations: Vec<String>,
    pub unnecessary_revalidation_obligations: Vec<String>,
    pub false_resurrection_obligations: Vec<String>,
    pub stale_carryover_obligations: Vec<String>,
    pub future_evidence_leakage_obligations: Vec<String>,
    pub unsupported_current_claims: Vec<String>,
    pub invalid_fresh_receipts: Vec<ReceiptIssue>,
    pub invalid_reuse_receipts: Vec<ReceiptIssue>,
    pub critical_false_resurrection_count: usize,
    pub required_revalidation_units: usize,
    pub fresh_revalidation_units: usize,
    pub over_revalidation_units: usize,
    pub under_revalidation_units: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct MultiShiftTraceReport {
    pub trace_id: String,
    pub epochs: Vec<MultiShiftEpochReport>,
    pub total_required_revalidation_units: usize,
    pub total_fresh_revalidation_units: usize,
    pub total_over_revalidation_units: usize,
    pub total_under_revalidation_units: usize,
    pub total_false_resurrections: usize,
    pub total_critical_false_resurrections: usize,
    pub total_stale_carryovers: usize,
    pub total_future_evidence_leakages: usize,
    pub total_unsupported_current_claims: usize,
    pub total_dependency_only_impact_labels: usize,
    pub total_gold_only_impact_labels: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct MultiShiftRestorationReport {
    pub control_id: String,
    pub active_obligations: Vec<String>,
    pub valid_reused_obligations: Vec<String>,
    pub required_revalidation_obligations: Vec<String>,
    pub false_resurrection_obligations: Vec<String>,
    pub invalid_reuse_receipts: Vec<ReceiptIssue>,
    pub false_resurrection_count: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct MultiShiftSequenceReport {
    pub scenario_id: String,
    pub gold_adjudication_status: String,
    pub trace_reports: Vec<MultiShiftTraceReport>,
    pub restoration_control: MultiShiftRestorationReport,
    pub claim_ceiling: String,
}

#[derive(Debug, Clone)]
struct Obligation {
    critical: bool,
    introduced_epoch: u64,
    dependency_dimensions: BTreeSet<String>,
}

#[derive(Debug, Clone)]
struct ReceiptAssessment {
    outcome: String,
    reasons: Vec<String>,
    context_mismatch_dimensions: Vec<String>,
    unavailable_at_cutoff: bool,
}

fn required_str<'a>(value: &'a Value, key: &str, label: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|item| !item.trim().is_empty())
        .ok_or_else(|| format!("{label}.{key} must be a non-empty string"))
}

fn required_u64(value: &Value, key: &str, label: &str) -> Result<u64, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label}.{key} must be an unsigned integer"))
}

fn unique_string_set(value: &Value, key: &str, label: &str) -> Result<BTreeSet<String>, String> {
    let values = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label}.{key} must be an array"))?;
    let mut output = BTreeSet::new();
    for (index, item) in values.iter().enumerate() {
        let id = item
            .as_str()
            .filter(|item| !item.trim().is_empty())
            .ok_or_else(|| format!("{label}.{key}[{index}] must be a non-empty string"))?;
        if !output.insert(id.to_owned()) {
            return Err(format!("{label}.{key} contains duplicate value {id:?}"));
        }
    }
    Ok(output)
}

fn parse_time(value: &Value, key: &str, label: &str) -> Result<DateTime<FixedOffset>, String> {
    let raw = required_str(value, key, label)?;
    DateTime::parse_from_rfc3339(raw)
        .map_err(|error| format!("{label}.{key} must be RFC 3339: {error}"))
}

fn validate_state(state: &Value, label: &str) -> Result<(), String> {
    let object = state
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    if object.len() != STATE_DIMENSIONS.len() {
        return Err(format!(
            "{label} must contain exactly {} identity dimensions",
            STATE_DIMENSIONS.len()
        ));
    }
    for dimension in STATE_DIMENSIONS {
        let digest = required_str(state, dimension, label)?;
        if digest.len() != 64
            || !digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(format!(
                "{label}.{dimension} must be a lowercase 64-character hexadecimal digest"
            ));
        }
    }
    for key in object.keys() {
        if !STATE_DIMENSIONS.contains(&key.as_str()) {
            return Err(format!("{label} contains unknown identity dimension {key:?}"));
        }
    }
    Ok(())
}

fn states_equal(left: &Value, right: &Value) -> bool {
    STATE_DIMENSIONS
        .iter()
        .all(|dimension| left.get(dimension) == right.get(dimension))
}

fn parse_obligations(candidate_view: &Value) -> Result<BTreeMap<String, Obligation>, String> {
    let values = candidate_view
        .get("obligations")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_view.obligations must be an array".to_owned())?;
    let mut obligations = BTreeMap::new();
    for (index, value) in values.iter().enumerate() {
        let label = format!("candidate_view.obligations[{index}]");
        let id = required_str(value, "obligation_id", &label)?.to_owned();
        let critical = value
            .get("critical")
            .and_then(Value::as_bool)
            .ok_or_else(|| format!("{label}.critical must be boolean"))?;
        let introduced_epoch = required_u64(value, "introduced_epoch", &label)?;
        let dimensions = unique_string_set(value, "dependency_dimensions", &label)?;
        if dimensions.is_empty() {
            return Err(format!("{label}.dependency_dimensions must not be empty"));
        }
        for dimension in &dimensions {
            if !STATE_DIMENSIONS.contains(&dimension.as_str()) {
                return Err(format!(
                    "{label} uses unsupported dependency dimension {dimension:?}"
                ));
            }
        }
        if obligations
            .insert(
                id.clone(),
                Obligation {
                    critical,
                    introduced_epoch,
                    dependency_dimensions: dimensions,
                },
            )
            .is_some()
        {
            return Err(format!("duplicate obligation definition {id:?}"));
        }
    }
    if obligations.is_empty() {
        return Err("at least one obligation definition is required".to_owned());
    }
    Ok(obligations)
}

fn assess_receipt(
    receipt: &Value,
    kind: &str,
    epoch: u64,
    decision_cutoff: &DateTime<FixedOffset>,
    current_state: &Value,
    obligation: &Obligation,
    minimum_observed_at: Option<&DateTime<FixedOffset>>,
) -> Result<ReceiptAssessment, String> {
    let label = format!("{kind} receipt");
    let obligation_id = required_str(receipt, "obligation_id", &label)?.to_owned();
    let source_epoch = required_u64(receipt, "source_epoch", &label)?;
    let outcome = required_str(receipt, "outcome", &label)?.to_owned();
    if !["pass", "fail", "unknown"].contains(&outcome.as_str()) {
        return Err(format!("{label} has unsupported outcome {outcome:?}"));
    }
    let context = receipt
        .get("context")
        .ok_or_else(|| format!("{label}.context is required"))?;
    validate_state(context, &format!("{label}.context"))?;

    let observed_at = parse_time(receipt, "observed_at", &label)?;
    let available_at = parse_time(receipt, "available_at", &label)?;
    let valid_until = parse_time(receipt, "valid_until", &label)?;

    let mut reasons = Vec::new();
    let is_fresh = kind == "fresh";
    if is_fresh && source_epoch != epoch {
        reasons.push(format!(
            "wrong_fresh_source_epoch:expected_{epoch}_got_{source_epoch}"
        ));
    }
    if !is_fresh && source_epoch >= epoch {
        reasons.push(format!(
            "reused_receipt_not_from_prior_epoch:current_{epoch}_source_{source_epoch}"
        ));
    }
    if observed_at > available_at {
        reasons.push("observation_after_availability".to_owned());
    }
    let unavailable_at_cutoff = available_at > *decision_cutoff;
    if unavailable_at_cutoff {
        reasons.push("unavailable_at_decision_cutoff".to_owned());
    }
    if *decision_cutoff > valid_until {
        reasons.push("expired_at_decision_cutoff".to_owned());
    }
    if let Some(minimum) = minimum_observed_at {
        if observed_at < *minimum {
            reasons.push("observed_before_intervention_available".to_owned());
        }
    }

    let mut context_mismatch_dimensions = Vec::new();
    for dimension in &obligation.dependency_dimensions {
        if context.get(dimension) != current_state.get(dimension) {
            context_mismatch_dimensions.push(dimension.clone());
            reasons.push(format!("context_mismatch:{dimension}"));
        }
    }
    context_mismatch_dimensions.sort();
    reasons.sort();
    reasons.dedup();

    Ok(ReceiptAssessment {
        outcome,
        reasons,
        context_mismatch_dimensions,
        unavailable_at_cutoff,
    })
}

fn parse_receipt_array<'a>(
    checkpoint: &'a Value,
    key: &str,
    label: &str,
) -> Result<&'a Vec<Value>, String> {
    checkpoint
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label}.{key} must be an array"))
}

fn assess_initial_receipts(
    candidate_view: &Value,
    obligations: &BTreeMap<String, Obligation>,
    first_event_time: &DateTime<FixedOffset>,
    baseline: &Value,
) -> Result<(), String> {
    let receipts = candidate_view
        .get("initial_receipts")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_view.initial_receipts must be an array".to_owned())?;
    let mut seen = BTreeSet::new();
    for (index, receipt) in receipts.iter().enumerate() {
        let label = format!("candidate_view.initial_receipts[{index}]");
        let id = required_str(receipt, "obligation_id", &label)?.to_owned();
        if !seen.insert(id.clone()) {
            return Err(format!("duplicate initial receipt for {id:?}"));
        }
        let obligation = obligations
            .get(&id)
            .ok_or_else(|| format!("initial receipt references unknown obligation {id:?}"))?;
        if obligation.introduced_epoch != 0 {
            return Err(format!(
                "initial receipt exists for obligation {id:?} before its introduction epoch"
            ));
        }
        let assessment = assess_receipt(
            receipt,
            "reused",
            1,
            first_event_time,
            baseline,
            obligation,
            None,
        )?;
        if !assessment.reasons.is_empty() || assessment.outcome != "pass" {
            return Err(format!(
                "initial receipt for {id:?} is not valid at baseline: {:?}",
                assessment.reasons
            ));
        }
    }
    Ok(())
}

fn validate_event_sequence(
    candidate_view: &Value,
    obligations: &BTreeMap<String, Obligation>,
) -> Result<Vec<(String, Value, Value, DateTime<FixedOffset>, DateTime<FixedOffset>)>, String> {
    let baseline = candidate_view
        .get("baseline")
        .ok_or_else(|| "candidate_view.baseline is required".to_owned())?;
    validate_state(baseline, "candidate_view.baseline")?;

    let events = candidate_view
        .get("interventions")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_view.interventions must be an array".to_owned())?;
    if !(2..=5).contains(&events.len()) {
        return Err("D6.8 v0.1 requires between two and five interventions".to_owned());
    }

    let mut seen_event_ids = BTreeSet::new();
    let mut sequence = Vec::new();
    let mut previous_state = baseline.clone();
    let mut previous_event_time: Option<DateTime<FixedOffset>> = None;
    let mut previous_available_at: Option<DateTime<FixedOffset>> = None;

    for (index, event) in events.iter().enumerate() {
        let label = format!("candidate_view.interventions[{index}]");
        let epoch = index as u64 + 1;
        let sequence_index = required_u64(event, "sequence_index", &label)?;
        if sequence_index != epoch {
            return Err(format!("{label}.sequence_index must be {epoch}"));
        }
        let event_id = required_str(event, "event_id", &label)?.to_owned();
        if !seen_event_ids.insert(event_id.clone()) {
            return Err(format!("duplicate event_id {event_id:?}"));
        }
        let event_time = parse_time(event, "event_time", &label)?;
        let available_at = parse_time(event, "available_at", &label)?;
        if event_time > available_at {
            return Err(format!("{label}: available_at precedes event_time"));
        }
        if let Some(previous) = &previous_event_time {
            if event_time <= *previous {
                return Err(format!("{label}: event_time must strictly increase in v0.1 sequence order"));
            }
        }
        if let Some(previous) = &previous_available_at {
            if available_at < *previous {
                return Err(format!("{label}: event availability must not move backwards"));
            }
        }

        let before = event
            .get("before")
            .ok_or_else(|| format!("{label}.before is required"))?;
        let after = event
            .get("after")
            .ok_or_else(|| format!("{label}.after is required"))?;
        validate_state(before, &format!("{label}.before"))?;
        validate_state(after, &format!("{label}.after"))?;
        if !states_equal(before, &previous_state) {
            return Err(format!("{label}.before does not match the prior sequence state"));
        }

        let introduced = unique_string_set(event, "newly_introduced_obligations", &label)?;
        let expected_introduced: BTreeSet<String> = obligations
            .iter()
            .filter(|(_, obligation)| obligation.introduced_epoch == epoch)
            .map(|(id, _)| id.clone())
            .collect();
        if introduced != expected_introduced {
            return Err(format!(
                "{label}.newly_introduced_obligations does not match obligation introduced_epoch values"
            ));
        }
        sequence.push((event_id, before.clone(), after.clone(), event_time.clone(), available_at.clone()));
        previous_state = after.clone();
        previous_event_time = Some(event_time);
        previous_available_at = Some(available_at);
    }

    for (id, obligation) in obligations {
        if obligation.introduced_epoch > events.len() as u64 {
            return Err(format!(
                "obligation {id:?} is introduced after the final event"
            ));
        }
    }
    Ok(sequence)
}

fn validate_gold_overlay(
    bundle: &Value,
    obligations: &BTreeMap<String, Obligation>,
    sequence_len: usize,
) -> Result<Vec<BTreeSet<String>>, String> {
    let gold = bundle
        .get("gold_overlay")
        .ok_or_else(|| "gold_overlay is required".to_owned())?;
    if required_str(gold, "visibility", "gold_overlay")? != "evaluator_only" {
        return Err("gold_overlay.visibility must be evaluator_only".to_owned());
    }
    let adjudication = required_str(gold, "adjudication_status", "gold_overlay")?;
    if ![
        "synthetic_known_answer_not_independently_adjudicated",
        "independently_adjudicated",
    ]
    .contains(&adjudication)
    {
        return Err(format!("unsupported gold adjudication status {adjudication:?}"));
    }
    let epochs = gold
        .get("required_revalidation_by_epoch")
        .and_then(Value::as_array)
        .ok_or_else(|| "gold_overlay.required_revalidation_by_epoch must be an array".to_owned())?;
    if epochs.len() != sequence_len {
        return Err("gold impact overlay must have one entry per intervention".to_owned());
    }

    let mut result = Vec::new();
    for (index, item) in epochs.iter().enumerate() {
        let epoch = index as u64 + 1;
        let label = format!("gold_overlay.required_revalidation_by_epoch[{index}]");
        if required_u64(item, "epoch", &label)? != epoch {
            return Err(format!("{label}.epoch must be {epoch}"));
        }
        let required = unique_string_set(item, "required_revalidation_obligations", &label)?;
        for id in &required {
            let obligation = obligations
                .get(id)
                .ok_or_else(|| format!("{label} references unknown obligation {id:?}"))?;
            if obligation.introduced_epoch > epoch {
                return Err(format!(
                    "{label} requires obligation {id:?} before it is introduced"
                ));
            }
        }

        let reasons = item
            .get("reason_by_obligation")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{label}.reason_by_obligation must be an array"))?;
        let mut reason_ids = BTreeSet::new();
        for (reason_index, reason) in reasons.iter().enumerate() {
            let reason_label = format!("{label}.reason_by_obligation[{reason_index}]");
            let id = required_str(reason, "obligation_id", &reason_label)?;
            required_str(reason, "reason", &reason_label)?;
            if !reason_ids.insert(id.to_owned()) {
                return Err(format!("{label} has duplicate reason for {id:?}"));
            }
        }
        if reason_ids != required {
            return Err(format!(
                "{label}.reason_by_obligation must have exactly the required obligation IDs"
            ));
        }
        result.push(required);
    }
    Ok(result)
}

fn score_trace(
    trace: &Value,
    sequence: &[(String, Value, Value, DateTime<FixedOffset>, DateTime<FixedOffset>)],
    gold_required: &[BTreeSet<String>],
    obligations: &BTreeMap<String, Obligation>,
    enforce_synthetic_dependency_parity: bool,
) -> Result<MultiShiftTraceReport, String> {
    let trace_id = required_str(trace, "trace_id", "candidate_trace")?.to_owned();
    let checkpoints = trace
        .get("checkpoints")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("trace {trace_id}: checkpoints must be an array"))?;
    if checkpoints.len() != sequence.len() {
        return Err(format!(
            "trace {trace_id}: checkpoint count must equal intervention count"
        ));
    }

    let mut epoch_reports = Vec::new();
    let mut total_required = 0usize;
    let mut total_fresh = 0usize;
    let mut total_over = 0usize;
    let mut total_under = 0usize;
    let mut total_false_resurrections = 0usize;
    let mut total_critical_false_resurrections = 0usize;
    let mut total_stale_carryovers = 0usize;
    let mut total_future_leakages = 0usize;
    let mut total_unsupported = 0usize;
    let mut total_dependency_only = 0usize;
    let mut total_gold_only = 0usize;
    let mut previous_cutoff: Option<DateTime<FixedOffset>> = None;

    for (index, checkpoint) in checkpoints.iter().enumerate() {
        let epoch = index as u64 + 1;
        let label = format!("trace {trace_id} checkpoint[{index}]");
        if required_u64(checkpoint, "epoch", &label)? != epoch {
            return Err(format!("{label}.epoch must be {epoch}"));
        }
        let (event_id, before_state, current_state, _event_time, event_available_at) = &sequence[index];
        let decision_cutoff = parse_time(checkpoint, "decision_cutoff", &label)?;
        if decision_cutoff < *event_available_at {
            return Err(format!("{label}: decision cutoff precedes intervention availability"));
        }
        if let Some(previous) = &previous_cutoff {
            if decision_cutoff <= *previous {
                return Err(format!("{label}: decision cutoffs must strictly increase"));
            }
        }
        previous_cutoff = Some(decision_cutoff.clone());

        let declared_current = unique_string_set(
            checkpoint,
            "declared_current_obligations",
            &label,
        )?;
        let active: BTreeSet<String> = obligations
            .iter()
            .filter(|(_, obligation)| obligation.introduced_epoch <= epoch)
            .map(|(id, _)| id.clone())
            .collect();
        for id in &declared_current {
            if !active.contains(id) {
                return Err(format!(
                    "{label} declares unknown or inactive obligation {id:?} current"
                ));
            }
        }

        let fresh_receipts = parse_receipt_array(checkpoint, "fresh_receipts", &label)?;
        let reused_receipts = parse_receipt_array(checkpoint, "reused_receipts", &label)?;
        let mut seen_receipts = BTreeSet::new();
        let mut valid_fresh = BTreeSet::new();
        let mut fresh_pass = BTreeSet::new();
        let mut valid_reused_pass = BTreeSet::new();
        let mut false_resurrection = BTreeSet::new();
        let mut stale_carryover = BTreeSet::new();
        let mut future_leakage = BTreeSet::new();
        let mut invalid_fresh_receipts = Vec::new();
        let mut invalid_reuse_receipts = Vec::new();

        for (kind, receipts) in [("fresh", fresh_receipts), ("reused", reused_receipts)] {
            for (receipt_index, receipt) in receipts.iter().enumerate() {
                let receipt_label = format!("{label}.{kind}_receipts[{receipt_index}]");
                let id = required_str(receipt, "obligation_id", &receipt_label)?.to_owned();
                if !active.contains(&id) {
                    return Err(format!("{receipt_label} references inactive or unknown obligation {id:?}"));
                }
                if !seen_receipts.insert(id.clone()) {
                    return Err(format!("{label} has more than one receipt for obligation {id:?}"));
                }
                let obligation = obligations
                    .get(&id)
                    .ok_or_else(|| format!("{receipt_label} references unknown obligation {id:?}"))?;
                let assessment = assess_receipt(
                    receipt,
                    kind,
                    epoch,
                    &decision_cutoff,
                    current_state,
                    obligation,
                    if kind == "fresh" {
                        Some(event_available_at)
                    } else {
                        None
                    },
                )?;

                if declared_current.contains(&id) && assessment.unavailable_at_cutoff {
                    future_leakage.insert(id.clone());
                }
                if kind == "fresh" {
                    if assessment.reasons.is_empty() {
                        valid_fresh.insert(id.clone());
                        if assessment.outcome == "pass" {
                            fresh_pass.insert(id.clone());
                        }
                    } else {
                        let mut reasons = assessment.reasons.clone();
                        if assessment.outcome != "pass" {
                            reasons.push(format!("non_passing_outcome:{}", assessment.outcome));
                        }
                        reasons.sort();
                        reasons.dedup();
                        invalid_fresh_receipts.push(ReceiptIssue {
                            obligation_id: id,
                            receipt_kind: "fresh".to_owned(),
                            reasons,
                        });
                    }
                } else if assessment.reasons.is_empty() && assessment.outcome == "pass" {
                    valid_reused_pass.insert(id.clone());
                } else {
                    if declared_current.contains(&id) {
                        false_resurrection.insert(id.clone());
                    }
                    if declared_current.contains(&id)
                        && !assessment.context_mismatch_dimensions.is_empty()
                    {
                        stale_carryover.insert(id.clone());
                    }
                    if !assessment.reasons.is_empty() || assessment.outcome != "pass" {
                        let mut reasons = assessment.reasons.clone();
                        if assessment.outcome != "pass" {
                            reasons.push(format!("non_passing_outcome:{}", assessment.outcome));
                        }
                        reasons.sort();
                        reasons.dedup();
                        invalid_reuse_receipts.push(ReceiptIssue {
                            obligation_id: id,
                            receipt_kind: "reused".to_owned(),
                            reasons,
                        });
                    }
                }
            }
        }

        let required = gold_required[index].clone();
        let changed_dimensions: BTreeSet<String> = STATE_DIMENSIONS
            .iter()
            .filter(|dimension| before_state.get(**dimension) != current_state.get(**dimension))
            .map(|dimension| (*dimension).to_owned())
            .collect();
        let dependency_implied: BTreeSet<String> = obligations
            .iter()
            .filter(|(_, obligation)| {
                obligation.introduced_epoch <= epoch
                    && (obligation.introduced_epoch == epoch
                        || obligation
                            .dependency_dimensions
                            .iter()
                            .any(|dimension| changed_dimensions.contains(dimension)))
            })
            .map(|(id, _)| id.clone())
            .collect();
        let dependency_only: BTreeSet<String> =
            dependency_implied.difference(&required).cloned().collect();
        let gold_only: BTreeSet<String> =
            required.difference(&dependency_implied).cloned().collect();
        if enforce_synthetic_dependency_parity && (!dependency_only.is_empty() || !gold_only.is_empty()) {
            return Err(format!(
                "synthetic gold impact disagrees with dependency-implied impact at epoch {epoch}: dependency_only={:?}, gold_only={:?}",
                dependency_only, gold_only
            ));
        }
        let missing: BTreeSet<String> =
            required.difference(&valid_fresh).cloned().collect();
        let required_without_pass: BTreeSet<String> =
            required.difference(&fresh_pass).cloned().collect();
        let unnecessary: BTreeSet<String> =
            valid_fresh.difference(&required).cloned().collect();
        let supported_current: BTreeSet<String> = fresh_pass
            .union(&valid_reused_pass)
            .cloned()
            .collect();
        let unsupported_current: BTreeSet<String> =
            declared_current.difference(&supported_current).cloned().collect();

        // A reused receipt can be current only if the complete set of
        // dependency dimensions declared by this obligation still matches.
        // This compares content identities, not a signature or execution truth.
        let critical_false_resurrections = false_resurrection
            .iter()
            .filter(|id| obligations.get(*id).is_some_and(|item| item.critical))
            .count();

        let report = MultiShiftEpochReport {
            epoch,
            event_id: event_id.clone(),
            decision_cutoff: required_str(checkpoint, "decision_cutoff", &label)?.to_owned(),
            required_revalidation_obligations: required.iter().cloned().collect(),
            dependency_implied_revalidation_obligations: dependency_implied.iter().cloned().collect(),
            dependency_only_impact_obligations: dependency_only.iter().cloned().collect(),
            gold_only_impact_obligations: gold_only.iter().cloned().collect(),
            fresh_revalidated_obligations: valid_fresh.iter().cloned().collect(),
            fresh_passing_obligations: fresh_pass.iter().cloned().collect(),
            reused_current_obligations: valid_reused_pass.iter().cloned().collect(),
            missing_required_revalidation_obligations: missing.iter().cloned().collect(),
            required_without_fresh_pass_obligations: required_without_pass.iter().cloned().collect(),
            unnecessary_revalidation_obligations: unnecessary.iter().cloned().collect(),
            false_resurrection_obligations: false_resurrection.iter().cloned().collect(),
            stale_carryover_obligations: stale_carryover.iter().cloned().collect(),
            future_evidence_leakage_obligations: future_leakage.iter().cloned().collect(),
            unsupported_current_claims: unsupported_current.iter().cloned().collect(),
            invalid_fresh_receipts,
            invalid_reuse_receipts,
            critical_false_resurrection_count: critical_false_resurrections,
            required_revalidation_units: required.len(),
            fresh_revalidation_units: valid_fresh.len(),
            over_revalidation_units: unnecessary.len(),
            under_revalidation_units: missing.len(),
        };

        total_required += report.required_revalidation_units;
        total_fresh += report.fresh_revalidation_units;
        total_over += report.over_revalidation_units;
        total_under += report.under_revalidation_units;
        total_false_resurrections += report.false_resurrection_obligations.len();
        total_critical_false_resurrections += report.critical_false_resurrection_count;
        total_stale_carryovers += report.stale_carryover_obligations.len();
        total_future_leakages += report.future_evidence_leakage_obligations.len();
        total_unsupported += report.unsupported_current_claims.len();
        total_dependency_only += report.dependency_only_impact_obligations.len();
        total_gold_only += report.gold_only_impact_obligations.len();
        epoch_reports.push(report);
    }

    Ok(MultiShiftTraceReport {
        trace_id,
        epochs: epoch_reports,
        total_required_revalidation_units: total_required,
        total_fresh_revalidation_units: total_fresh,
        total_over_revalidation_units: total_over,
        total_under_revalidation_units: total_under,
        total_false_resurrections,
        total_critical_false_resurrections,
        total_stale_carryovers,
        total_future_evidence_leakages: total_future_leakages,
        total_unsupported_current_claims: total_unsupported,
        total_dependency_only_impact_labels: total_dependency_only,
        total_gold_only_impact_labels: total_gold_only,
    })
}

fn score_restoration_control(
    bundle: &Value,
    candidate_view: &Value,
    obligations: &BTreeMap<String, Obligation>,
    sequence_len: usize,
) -> Result<MultiShiftRestorationReport, String> {
    let control = bundle
        .get("restoration_control")
        .ok_or_else(|| "restoration_control is required".to_owned())?;
    let control_id = required_str(control, "control_id", "restoration_control")?.to_owned();
    if required_str(control, "reuse_policy", "restoration_control")?
        != "permitted_only_when_all_declared_identity_and_validity_dimensions_match"
    {
        return Err("restoration_control has unsupported reuse policy".to_owned());
    }
    if control.get("receipt_validity_window_open").and_then(Value::as_bool) != Some(true) {
        return Err("restoration_control.receipt_validity_window_open must be true".to_owned());
    }
    let final_state = control
        .get("final_state")
        .ok_or_else(|| "restoration_control.final_state is required".to_owned())?;
    validate_state(final_state, "restoration_control.final_state")?;
    let baseline = candidate_view
        .get("baseline")
        .ok_or_else(|| "candidate_view.baseline is required".to_owned())?;
    if !states_equal(final_state, baseline) {
        return Err("restoration control must restore all declared baseline identity dimensions".to_owned());
    }
    let decision_cutoff = parse_time(control, "decision_cutoff", "restoration_control")?;
    let active = unique_string_set(control, "active_obligation_ids", "restoration_control")?;
    for id in &active {
        let obligation = obligations
            .get(id)
            .ok_or_else(|| format!("restoration control references unknown obligation {id:?}"))?;
        if obligation.introduced_epoch != 0 {
            return Err(format!(
                "restoration control baseline arm cannot activate later-introduced obligation {id:?}"
            ));
        }
    }

    let receipts = control
        .get("reused_receipts")
        .and_then(Value::as_array)
        .ok_or_else(|| "restoration_control.reused_receipts must be an array".to_owned())?;
    let mut seen = BTreeSet::new();
    let mut valid_reused = BTreeSet::new();
    let mut false_resurrection = BTreeSet::new();
    let mut invalid_reuse_receipts = Vec::new();
    for (index, receipt) in receipts.iter().enumerate() {
        let label = format!("restoration_control.reused_receipts[{index}]");
        let id = required_str(receipt, "obligation_id", &label)?.to_owned();
        if !active.contains(&id) {
            return Err(format!("restoration receipt references inactive obligation {id:?}"));
        }
        if !seen.insert(id.clone()) {
            return Err(format!("duplicate restoration receipt for {id:?}"));
        }
        let obligation = obligations
            .get(&id)
            .ok_or_else(|| format!("restoration receipt references unknown obligation {id:?}"))?;
        let assessment = assess_receipt(
            receipt,
            "reused",
            sequence_len as u64 + 1,
            &decision_cutoff,
            final_state,
            obligation,
            None,
        )?;
        if assessment.reasons.is_empty() && assessment.outcome == "pass" {
            valid_reused.insert(id);
        } else {
            false_resurrection.insert(id.clone());
            let mut reasons = assessment.reasons;
            if assessment.outcome != "pass" {
                reasons.push(format!("non_passing_outcome:{}", assessment.outcome));
            }
            reasons.sort();
            reasons.dedup();
            invalid_reuse_receipts.push(ReceiptIssue {
                obligation_id: id,
                receipt_kind: "restoration_reuse".to_owned(),
                reasons,
            });
        }
    }

    let required: BTreeSet<String> = active.difference(&valid_reused).cloned().collect();
    let expected_required =
        unique_string_set(control, "expected_required_revalidation", "restoration_control")?;
    let expected_false_count = required_u64(
        control,
        "expected_false_resurrection_count",
        "restoration_control",
    )? as usize;
    if required != expected_required {
        return Err(format!(
            "restoration expected_required_revalidation mismatch: computed {:?}, declared {:?}",
            required, expected_required
        ));
    }
    if false_resurrection.len() != expected_false_count {
        return Err(format!(
            "restoration expected_false_resurrection_count mismatch: computed {}, declared {expected_false_count}",
            false_resurrection.len()
        ));
    }

    Ok(MultiShiftRestorationReport {
        control_id,
        active_obligations: active.iter().cloned().collect(),
        valid_reused_obligations: valid_reused.iter().cloned().collect(),
        required_revalidation_obligations: required.iter().cloned().collect(),
        false_resurrection_obligations: false_resurrection.iter().cloned().collect(),
        invalid_reuse_receipts,
        false_resurrection_count: expected_false_count,
    })
}

/// Score a D6.8 multi-shift bundle without mutating the supplied JSON.
///
/// The bundle includes evaluator-side gold labels and candidate traces. The
/// caller must keep both the input bundle and this output outside candidate
/// workspaces. This function checks declared dependency dimensions and timing;
/// it does not authenticate receipts or independently adjudicate gold labels.
pub fn score_multi_shift_sequence_json(input: &str) -> Result<MultiShiftSequenceReport, String> {
    let bundle: Value =
        serde_json::from_str(input).map_err(|error| format!("bundle JSON parse failed: {error}"))?;
    if required_str(&bundle, "bundle_id", "bundle")?.is_empty() {
        return Err("bundle_id must not be empty".to_owned());
    }
    if required_str(&bundle, "bundle_visibility", "bundle")?
        != "evaluator_only_never_candidate_visible"
    {
        return Err("D6.8 evaluator bundle must be explicitly evaluator-only".to_owned());
    }
    let candidate_view = bundle
        .get("candidate_view")
        .ok_or_else(|| "candidate_view is required".to_owned())?;
    if candidate_view.get("gold_overlay").is_some() {
        return Err("candidate_view must not contain gold_overlay".to_owned());
    }
    let scenario_id = required_str(candidate_view, "scenario_id", "candidate_view")?.to_owned();
    let obligations = parse_obligations(candidate_view)?;
    let sequence = validate_event_sequence(candidate_view, &obligations)?;
    let baseline = candidate_view
        .get("baseline")
        .ok_or_else(|| "candidate_view.baseline is required".to_owned())?;
    let first_event_time = sequence
        .first()
        .map(|item| item.3.clone())
        .ok_or_else(|| "at least one intervention is required".to_owned())?;
    assess_initial_receipts(candidate_view, &obligations, &first_event_time, baseline)?;

    let gold_required = validate_gold_overlay(&bundle, &obligations, sequence.len())?;
    let gold_adjudication_status =
        required_str(&bundle["gold_overlay"], "adjudication_status", "gold_overlay")?.to_owned();
    let enforce_synthetic_dependency_parity =
        gold_adjudication_status == "synthetic_known_answer_not_independently_adjudicated";
    let traces = bundle
        .get("candidate_traces")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_traces must be an array".to_owned())?;
    if traces.len() < 2 {
        return Err("at least two candidate traces are required for the paired known-answer bundle".to_owned());
    }
    let mut trace_ids = BTreeSet::new();
    let mut trace_reports = Vec::new();
    for trace in traces {
        let id = required_str(trace, "trace_id", "candidate_trace")?.to_owned();
        if !trace_ids.insert(id) {
            return Err("candidate trace identifiers must be unique".to_owned());
        }
        trace_reports.push(score_trace(
            trace,
            &sequence,
            &gold_required,
            &obligations,
            enforce_synthetic_dependency_parity,
        )?);
    }
    trace_reports.sort_by(|left, right| left.trace_id.cmp(&right.trace_id));

    let restoration_control =
        score_restoration_control(&bundle, candidate_view, &obligations, sequence.len())?;

    Ok(MultiShiftSequenceReport {
        scenario_id,
        gold_adjudication_status,
        trace_reports,
        restoration_control,
        claim_ceiling: "Deterministic evaluation of supplied multi-shift traces against evaluator-side impact labels and declared receipt dependency/time semantics. Does not authenticate execution or authority, independently adjudicate gold labels, prove causal completeness, or establish agent performance.".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const FIXTURE: &str = include_str!(
        "../../../docs/development-map/fixtures/egddm-multi-shift-sequence-v0.1.json"
    );

    fn fixture_value() -> Value {
        serde_json::from_str(FIXTURE).expect("D6.8 fixture JSON")
    }

    fn report_for(value: &Value) -> MultiShiftSequenceReport {
        score_multi_shift_sequence_json(&value.to_string()).expect("D6.8 score")
    }

    fn trace<'a>(report: &'a MultiShiftSequenceReport, id: &str) -> &'a MultiShiftTraceReport {
        report.trace_reports.iter().find(|item| item.trace_id == id).expect("trace")
    }

    #[test]
    fn matches_complete_machine_readable_known_answer_report() {
        const EXPECTED: &str = include_str!(
            "../../../docs/development-map/fixtures/egddm-multi-shift-sequence-v0.1.expected.json"
        );
        let actual = score_multi_shift_sequence_json(FIXTURE).expect("D6.8 fixture must score");
        let actual_value = serde_json::to_value(actual).expect("serialize report");
        let expected_value: Value = serde_json::from_str(EXPECTED).expect("expected report JSON");
        assert_eq!(actual_value, expected_value);
    }

    #[test]
    fn matches_multishift_known_answer_and_restoration_control() {
        let report = score_multi_shift_sequence_json(FIXTURE).expect("D6.8 fixture must score");
        let conformant = trace(&report, "conformant-sequence");
        assert_eq!(conformant.total_required_revalidation_units, 10);
        assert_eq!(conformant.total_fresh_revalidation_units, 10);
        assert_eq!(conformant.total_over_revalidation_units, 0);
        assert_eq!(conformant.total_under_revalidation_units, 0);
        assert_eq!(conformant.total_false_resurrections, 0);
        assert_eq!(conformant.total_critical_false_resurrections, 0);
        assert_eq!(conformant.total_dependency_only_impact_labels, 0);
        assert_eq!(conformant.total_gold_only_impact_labels, 0);
        assert_eq!(report.gold_adjudication_status, "synthetic_known_answer_not_independently_adjudicated");
        let adversarial = trace(&report, "false-resurrection-sequence");
        assert_eq!(adversarial.total_required_revalidation_units, 10);
        assert_eq!(adversarial.total_fresh_revalidation_units, 9);
        assert_eq!(adversarial.total_under_revalidation_units, 1);
        assert_eq!(adversarial.total_false_resurrections, 1);
        assert_eq!(adversarial.total_critical_false_resurrections, 1);
        assert_eq!(
            adversarial.epochs[2].false_resurrection_obligations,
            vec!["release-authority".to_owned()]
        );
        assert_eq!(
            report.restoration_control.required_revalidation_obligations,
            Vec::<String>::new()
        );
        assert_eq!(report.restoration_control.false_resurrection_count, 0);
        assert_eq!(report.restoration_control.valid_reused_obligations.len(), 4);
    }


    #[test]
    fn rejects_synthetic_gold_label_disagreement_with_declared_dependencies() {
        let mut value = fixture_value();
        value["gold_overlay"]["required_revalidation_by_epoch"][0]["required_revalidation_obligations"]
            .as_array_mut().expect("gold set")
            .retain(|item| item.as_str() != Some("build-repro"));
        value["gold_overlay"]["required_revalidation_by_epoch"][0]["reason_by_obligation"]
            .as_array_mut().expect("reasons")
            .retain(|item| item["obligation_id"].as_str() != Some("build-repro"));
        let error = score_multi_shift_sequence_json(&value.to_string()).unwrap_err();
        assert!(error.contains("synthetic gold impact disagrees with dependency-implied impact"));
    }

    #[test]
    fn independent_gold_disagreement_is_reported_not_silently_rewritten() {
        let mut value = fixture_value();
        value["gold_overlay"]["adjudication_status"] = json!("independently_adjudicated");
        value["gold_overlay"]["required_revalidation_by_epoch"][0]["required_revalidation_obligations"]
            .as_array_mut().expect("gold set")
            .retain(|item| item != "build-repro");
        value["gold_overlay"]["required_revalidation_by_epoch"][0]["reason_by_obligation"]
            .as_array_mut().expect("reasons")
            .retain(|item| item["obligation_id"] != "build-repro");
        let report = report_for(&value);
        assert_eq!(report.gold_adjudication_status, "independently_adjudicated");
        let conformant = trace(&report, "conformant-sequence");
        assert_eq!(conformant.epochs[0].dependency_only_impact_obligations, vec!["build-repro".to_owned()]);
        assert!(conformant.epochs[0].gold_only_impact_obligations.is_empty());
        assert_eq!(conformant.total_dependency_only_impact_labels, 1);
    }

    #[test]
    fn source_dependency_change_invalidates_old_build_receipt() {
        let mut value = fixture_value();
        let old_source = value["candidate_view"]["baseline"]["source_tree_digest"].clone();
        value["candidate_traces"][0]["checkpoints"][0]["fresh_receipts"]
            .as_array_mut().expect("fresh receipts")
            .retain(|receipt| receipt["obligation_id"].as_str() != Some("build-repro"));
        let reused = value["candidate_traces"][0]["checkpoints"][0]["reused_receipts"]
            .as_array_mut().expect("reused receipts");
        let mut stale = value["candidate_view"]["initial_receipts"][0].clone();
        stale["context"]["source_tree_digest"] = old_source;
        reused.push(stale);
        value["candidate_traces"][0]["checkpoints"][0]["declared_current_obligations"] =
            json!(["build-repro","api-compat","audit-logging","release-authority","docs-format"]);
        let report = report_for(&value);
        let result = trace(&report, "conformant-sequence");
        assert!(result.epochs[0].false_resurrection_obligations.contains(&"build-repro".to_owned()));
        assert!(result.epochs[0].stale_carryover_obligations.contains(&"build-repro".to_owned()));
        assert!(result.epochs[0].missing_required_revalidation_obligations.contains(&"build-repro".to_owned()));
    }

    #[test]
    fn detects_future_available_evidence_claimed_current() {
        let mut value = fixture_value();
        value["candidate_traces"][0]["checkpoints"][0]["fresh_receipts"][0]["available_at"] =
            json!("2026-11-15T10:02:00Z");
        let report = report_for(&value);
        let result = trace(&report, "conformant-sequence");
        assert!(result.epochs[0].future_evidence_leakage_obligations.contains(
            &value["candidate_traces"][0]["checkpoints"][0]["fresh_receipts"][0]["obligation_id"]
                .as_str().unwrap().to_owned()
        ));
        assert!(result.epochs[0].unsupported_current_claims.contains(
            &value["candidate_traces"][0]["checkpoints"][0]["fresh_receipts"][0]["obligation_id"]
                .as_str().unwrap().to_owned()
        ));
    }

    #[test]
    fn unchanged_documentation_receipt_remains_reusable() {
        let report = score_multi_shift_sequence_json(FIXTURE).expect("D6.8 fixture");
        let result = trace(&report, "conformant-sequence");
        assert!(result.epochs[2].reused_current_obligations.contains(&"docs-format".to_owned()));
        assert_eq!(result.epochs[2].false_resurrection_obligations, Vec::<String>::new());
    }

    #[test]
    fn rejects_a_broken_intervention_state_chain() {
        let mut value = fixture_value();
        value["candidate_view"]["interventions"][1]["before"]["runner_image_digest"] =
            json!("00000000000000000000000000000000000000000000000000000000000000ff");
        let error = score_multi_shift_sequence_json(&value.to_string()).unwrap_err();
        assert!(error.contains("does not match the prior sequence state"));
    }

    #[test]
    fn rejects_duplicate_receipts_for_one_checkpoint_obligation() {
        let mut value = fixture_value();
        let receipt = value["candidate_traces"][0]["checkpoints"][0]["fresh_receipts"][0].clone();
        value["candidate_traces"][0]["checkpoints"][0]["reused_receipts"]
            .as_array_mut().expect("reused")
            .push(receipt);
        let error = score_multi_shift_sequence_json(&value.to_string()).unwrap_err();
        assert!(error.contains("more than one receipt"));
    }

    #[test]
    fn rejects_gold_reason_omission() {
        let mut value = fixture_value();
        value["gold_overlay"]["required_revalidation_by_epoch"][0]["reason_by_obligation"]
            .as_array_mut().expect("reason list")
            .pop();
        let error = score_multi_shift_sequence_json(&value.to_string()).unwrap_err();
        assert!(error.contains("exactly the required obligation IDs"));
    }

    #[test]
    fn restoration_reuse_requires_full_identity_match() {
        let mut value = fixture_value();
        value["restoration_control"]["reused_receipts"][0]["context"]["runner_image_digest"] =
            json!("00000000000000000000000000000000000000000000000000000000000000ff");
        value["restoration_control"]["expected_required_revalidation"] =
            json!(["build-repro"]);
        value["restoration_control"]["expected_false_resurrection_count"] = json!(1);
        let report = report_for(&value);
        assert_eq!(
            report.restoration_control.required_revalidation_obligations,
            vec!["build-repro".to_owned()]
        );
        assert_eq!(report.restoration_control.false_resurrection_count, 1);
    }

    #[test]
    fn nonpassing_fresh_check_is_refreshed_but_does_not_close_requirement() {
        let mut value = fixture_value();
        value["candidate_traces"][0]["checkpoints"][0]["fresh_receipts"][0]["outcome"] =
            json!("fail");
        let report = report_for(&value);
        let result = trace(&report, "conformant-sequence");
        let id = value["candidate_traces"][0]["checkpoints"][0]["fresh_receipts"][0]["obligation_id"]
            .as_str().unwrap().to_owned();
        assert!(result.epochs[0].fresh_revalidated_obligations.contains(&id));
        assert!(!result.epochs[0].fresh_passing_obligations.contains(&id));
        assert!(result.epochs[0].required_without_fresh_pass_obligations.contains(&id));
    }
}
