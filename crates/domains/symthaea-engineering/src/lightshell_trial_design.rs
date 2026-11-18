// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pre-registered, counterbalanced trial-design integrity for Atelier Lightshell.
//!
//! This module validates experimental allocation, per-period reset/carryover
//! records and block-level drift evidence. It links to installed-system runs;
//! it does not execute trials, analyze qualification metrics, control robots,
//! evaluate functional safety, or authorize operation.

use serde::Deserialize;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrialDesignManifest {
    pub schema_version: String,
    pub campaign_id: String,
    pub target_qualification_campaign_ref: String,
    pub state: String,
    pub design_mode: String,
    pub frozen_before_first_run: bool,
    pub preregistration_ref: String,
    pub preregistered_at: String,
    pub protocol_digest: String,
    pub condition_ids: Vec<String>,
    pub incumbent_comparator_ref: Option<String>,
    pub strata: Vec<RobotStratum>,
    pub allocation_policy: AllocationPolicy,
    pub reset_policy: ResetPolicy,
    pub drift_policy: DriftPolicy,
    pub blocks: Vec<TrialBlock>,
    pub independent_review: Option<IndependentReview>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobotStratum {
    pub stratum_id: String,
    pub robot_serial: String,
    pub firmware_revision: String,
    pub robot_configuration_digest: String,
    pub tool_payload_digest: String,
    pub mission_profile_ref: String,
    pub environment_stratum_ref: String,
    pub minimum_blocks: u32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AllocationPolicy {
    pub algorithm_ref: String,
    pub seed_commitment_sha256: String,
    pub seed_reveal_ref: Option<String>,
    pub allocation_custodian_ref: String,
    pub allocation_method_ref: String,
    pub locked_before_outcomes: bool,
    pub no_outcome_adaptive_reassignment: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResetPolicy {
    pub reset_procedure_ref: String,
    pub stabilization_rule_ref: String,
    pub reset_before_every_period: bool,
    pub carryover_check_between_periods: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DriftPolicy {
    pub sentinel_metric_id: String,
    pub maximum_abs_drift: f64,
    pub uncertainty_guarded: bool,
    pub start_end_measurements_required: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrialBlock {
    pub block_id: String,
    pub stratum_id: String,
    pub sequence_id: String,
    pub allocation_receipt_ref: String,
    pub periods: Vec<TrialPeriod>,
    pub drift_check: Option<BlockDriftCheck>,
    pub block_evidence_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrialPeriod {
    pub period_index: u32,
    pub condition_id: String,
    pub linked_system_run_ref: Option<String>,
    pub reset_before: Option<ResetCheck>,
    pub carryover_check: Option<CarryoverCheck>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResetCheck {
    pub disposition: String,
    pub evidence_ref: String,
    pub rule_ref: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CarryoverCheck {
    pub disposition: String,
    pub evidence_ref: String,
    pub rule_ref: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockDriftCheck {
    pub start_value: f64,
    pub end_value: f64,
    pub expanded_uncertainty: f64,
    pub start_evidence_ref: String,
    pub end_evidence_ref: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndependentReview {
    pub disposition: String,
    pub reviewer_ref: String,
    pub reviewed_at: String,
    pub review_record_ref: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationReport {
    pub findings: Vec<Finding>,
    pub stratum_count: usize,
    pub block_count: usize,
    pub period_count: usize,
    pub executed_period_count: usize,
    pub balanced_stratum_count: usize,
    pub trial_evidence_admitted: bool,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.findings.is_empty()
    }

    /// Admits only protocol-conformant, independently reviewed trial evidence.
    /// This does not admit the underlying installed-system qualification result.
    pub fn can_admit_trial_evidence(&self) -> bool {
        self.is_valid() && self.trial_evidence_admitted
    }
}

pub fn parse_and_validate(input: &str) -> Result<ValidationReport, serde_json::Error> {
    let manifest: TrialDesignManifest = serde_json::from_str(input)?;
    Ok(manifest.validate())
}

impl TrialDesignManifest {
    pub fn validate(&self) -> ValidationReport {
        let mut findings = Vec::new();
        let mut strata = HashSet::new();
        let mut block_ids = HashSet::new();
        let mut allocation_receipts = HashSet::new();
        let mut system_runs = HashSet::new();
        let mut blocks_per_stratum: HashMap<&str, usize> = HashMap::new();
        let mut sequence_counts: HashMap<(&str, &str), usize> = HashMap::new();
        let mut balanced_strata = 0usize;
        let mut period_count = 0usize;
        let mut executed_period_count = 0usize;

        if self.schema_version != "0.1.0" {
            push(&mut findings, "unsupported_schema_version", "schema_version",
                "Only trial-design schema version 0.1.0 is admitted.");
        }
        if is_blank(&self.campaign_id) || is_blank(&self.target_qualification_campaign_ref) {
            push(&mut findings, "campaign_binding_missing", "campaign_id",
                "Campaign and target installed-system qualification references are required.");
        }
        if !matches!(self.state.as_str(), "proposed" | "observed" | "adjudicated") {
            push(&mut findings, "invalid_campaign_state", "state",
                "State must be proposed, observed or adjudicated.");
        }
        if self.design_mode != "randomized_counterbalanced_within_robot" {
            push(&mut findings, "unsupported_design_mode", "design_mode",
                "P0 requires a randomized, counterbalanced within-robot blocked design.");
        }
        if !self.frozen_before_first_run || is_blank(&self.preregistration_ref)
            || is_blank(&self.preregistered_at) || !is_sha256(&self.protocol_digest)
        {
            push(&mut findings, "preregistration_not_frozen", "preregistration_ref",
                "A pre-run frozen protocol record, timestamp and SHA-256 digest are required.");
        }

        let condition_set: HashSet<&str> = self.condition_ids.iter().map(String::as_str).collect();
        if condition_set.len() != self.condition_ids.len() {
            push(&mut findings, "duplicate_condition_id", "condition_ids",
                "Condition IDs must be unique.");
        }
        let sequence_catalog = sequence_catalog(&self.condition_ids);
        if sequence_catalog.is_empty() {
            push(&mut findings, "unsupported_condition_set", "condition_ids",
                "Use baseline_uncovered + lightshell_installed, optionally with incumbent_cover.");
        }
        let has_incumbent = condition_set.contains("incumbent_cover");
        if has_incumbent && self.incumbent_comparator_ref.as_deref().is_none_or(is_blank) {
            push(&mut findings, "incumbent_comparator_binding_missing", "incumbent_comparator_ref",
                "An incumbent condition requires an exact comparable product/configuration reference.");
        }
        if !has_incumbent && self.incumbent_comparator_ref.is_some() {
            push(&mut findings, "orphan_incumbent_comparator_binding", "incumbent_comparator_ref",
                "Do not declare an incumbent comparator when the trial has no incumbent condition.");
        }

        validate_allocation_policy(&self.allocation_policy, self.state.as_str(), &mut findings);
        validate_reset_policy(&self.reset_policy, &mut findings);
        if is_blank(&self.drift_policy.sentinel_metric_id)
            || !self.drift_policy.maximum_abs_drift.is_finite()
            || self.drift_policy.maximum_abs_drift <= 0.0
            || !self.drift_policy.uncertainty_guarded
            || !self.drift_policy.start_end_measurements_required
        {
            push(&mut findings, "invalid_drift_policy", "drift_policy",
                "A named sentinel, positive finite limit, start/end measurements and uncertainty guard are mandatory.");
        }

        for (index, stratum) in self.strata.iter().enumerate() {
            let path = format!("strata[{index}]");
            if is_blank(&stratum.stratum_id) || !strata.insert(stratum.stratum_id.as_str()) {
                push(&mut findings, "duplicate_or_blank_stratum_id", format!("{path}.stratum_id"),
                    "Stratum IDs must be non-empty and unique.");
            }
            if is_blank(&stratum.robot_serial)
                || is_blank(&stratum.firmware_revision)
                || is_blank(&stratum.robot_configuration_digest)
                || is_blank(&stratum.tool_payload_digest)
                || is_blank(&stratum.mission_profile_ref)
                || is_blank(&stratum.environment_stratum_ref)
            {
                push(&mut findings, "stratum_identity_incomplete", &path,
                    "Every stratum must freeze robot serial, firmware, configuration, payload, profile and environment.");
            }
            if !is_sha256(&stratum.robot_configuration_digest)
                || !is_sha256(&stratum.tool_payload_digest)
            {
                push(&mut findings, "stratum_digest_invalid", &path,
                    "Robot configuration and tool/payload identities must be SHA-256 digests.");
            }
            if !sequence_catalog.is_empty() && stratum.minimum_blocks < sequence_catalog.len() as u32 {
                push(&mut findings, "minimum_blocks_below_balance_floor", format!("{path}.minimum_blocks"),
                    "Minimum blocks must cover every counterbalance sequence at least once.");
            }
        }
        if self.strata.is_empty() {
            push(&mut findings, "no_robot_strata", "strata",
                "At least one frozen robot/configuration/environment stratum is required.");
        }

        for (block_index, block) in self.blocks.iter().enumerate() {
            let path = format!("blocks[{block_index}]");
            if is_blank(&block.block_id) || !block_ids.insert(block.block_id.as_str()) {
                push(&mut findings, "duplicate_or_blank_block_id", format!("{path}.block_id"),
                    "Block IDs must be non-empty and unique.");
            }
            if !strata.contains(block.stratum_id.as_str()) {
                push(&mut findings, "unknown_block_stratum", format!("{path}.stratum_id"),
                    "Block references an undeclared stratum.");
            }
            *blocks_per_stratum.entry(block.stratum_id.as_str()).or_default() += 1;
            if is_blank(&block.allocation_receipt_ref)
                || !allocation_receipts.insert(block.allocation_receipt_ref.as_str())
            {
                push(&mut findings, "missing_or_duplicate_allocation_receipt",
                    format!("{path}.allocation_receipt_ref"),
                    "Every block requires a unique allocation receipt.");
            }

            let expected_order = sequence_catalog.iter()
                .find(|(id, _)| id == &block.sequence_id)
                .map(|(_, order)| order);
            let Some(expected_order) = expected_order else {
                push(&mut findings, "unknown_counterbalance_sequence", format!("{path}.sequence_id"),
                    "Sequence must be a member of the declared two- or three-arm counterbalance catalog.");
                continue;
            };
            *sequence_counts.entry((block.stratum_id.as_str(), block.sequence_id.as_str())).or_default() += 1;

            if &block.periods.iter().map(|p| p.condition_id.clone()).collect::<Vec<_>>() != expected_order {
                push(&mut findings, "period_order_does_not_match_allocation", format!("{path}.periods"),
                    "Observed/planned period order must exactly match the pre-registered allocation sequence.");
            }
            if block.periods.len() != expected_order.len() {
                push(&mut findings, "period_count_mismatch", format!("{path}.periods"),
                    "Each block must contain exactly one period for every declared condition.");
            }

            for (period_index, period) in block.periods.iter().enumerate() {
                period_count += 1;
                let period_path = format!("{path}.periods[{period_index}]");
                if period.period_index as usize != period_index {
                    push(&mut findings, "period_index_not_contiguous", format!("{period_path}.period_index"),
                        "Period indices must be contiguous and start at zero.");
                }
                if !condition_set.contains(period.condition_id.as_str()) {
                    push(&mut findings, "period_condition_unknown", format!("{period_path}.condition_id"),
                        "Period condition is not declared in the frozen condition set.");
                }
                match self.state.as_str() {
                    "proposed" => {
                        if period.linked_system_run_ref.is_some()
                            || period.reset_before.is_some()
                            || period.carryover_check.is_some()
                        {
                            push(&mut findings, "proposed_design_contains_observations", &period_path,
                                "A proposed design cannot contain executed run, reset or carryover observations.");
                        }
                    }
                    "observed" | "adjudicated" => {
                        match period.linked_system_run_ref.as_deref() {
                            Some(reference) if !is_blank(reference) => {
                                executed_period_count += 1;
                                if !system_runs.insert(reference) {
                                    push(&mut findings, "duplicate_system_run_reference",
                                        format!("{period_path}.linked_system_run_ref"),
                                        "Every period must link to a unique installed-system run.");
                                }
                            }
                            _ => push(&mut findings, "period_system_run_missing",
                                format!("{period_path}.linked_system_run_ref"),
                                "Every observed period must link to an exact #12831 system-run record."),
                        }
                        validate_reset(period.reset_before.as_ref(), &period_path, &mut findings);
                        if period_index > 0 && self.reset_policy.carryover_check_between_periods {
                            validate_carryover(period.carryover_check.as_ref(), &period_path, &mut findings);
                        } else if period_index == 0 && period.carryover_check.is_some() {
                            push(&mut findings, "unexpected_first_period_carryover_check",
                                format!("{period_path}.carryover_check"),
                                "The first period has no preceding trial period to carry over.");
                        }
                    }
                    _ => {}
                }
            }

            match self.state.as_str() {
                "proposed" => {
                    if block.drift_check.is_some() || block.block_evidence_ref.is_some() {
                        push(&mut findings, "proposed_block_contains_outcome_evidence", &path,
                            "A proposed block cannot contain drift measurements or execution evidence.");
                    }
                }
                "observed" | "adjudicated" => {
                    validate_drift_check(
                        block.drift_check.as_ref(),
                        &self.drift_policy,
                        &path,
                        &mut findings,
                    );
                    if block.block_evidence_ref.as_deref().is_none_or(is_blank) {
                        push(&mut findings, "block_evidence_missing", format!("{path}.block_evidence_ref"),
                            "Observed blocks require a block-level evidence receipt.");
                    }
                }
                _ => {}
            }
        }

        if self.blocks.is_empty() {
            push(&mut findings, "no_trial_blocks", "blocks",
                "At least one pre-registered trial block is required.");
        }

        for stratum in &self.strata {
            let count = blocks_per_stratum.get(stratum.stratum_id.as_str()).copied().unwrap_or(0);
            if count < stratum.minimum_blocks as usize {
                push(&mut findings, "stratum_below_minimum_blocks", "blocks",
                    format!("Stratum {} has {count} blocks; {} are required.",
                        stratum.stratum_id, stratum.minimum_blocks));
            }
            if !sequence_catalog.is_empty() {
                let mut counts = Vec::new();
                for (sequence_id, _) in &sequence_catalog {
                    counts.push(sequence_counts.get(&(stratum.stratum_id.as_str(), sequence_id.as_str()))
                        .copied().unwrap_or(0));
                }
                if let Some(first) = counts.first().copied() {
                    if first > 0 && counts.iter().all(|count| *count == first)
                        && count % sequence_catalog.len() == 0
                    {
                        balanced_strata += 1;
                    } else {
                        push(&mut findings, "counterbalance_allocation_unbalanced", "blocks",
                            format!("Stratum {} must allocate every sequence equally; observed {:?}.",
                                stratum.stratum_id, counts));
                    }
                } else {
                    push(&mut findings, "counterbalance_allocation_missing", "blocks",
                        format!("Stratum {} has no counterbalanced allocation.", stratum.stratum_id));
                }
            }
        }

        if self.state != "proposed" && self.allocation_policy.seed_reveal_ref.as_deref().is_none_or(is_blank) {
            push(&mut findings, "allocation_seed_reveal_missing", "allocation_policy.seed_reveal_ref",
                "Observed campaigns must disclose the precommitted allocation seed or an auditable reveal receipt.");
        }
        if self.state == "proposed" && self.allocation_policy.seed_reveal_ref.is_some() {
            push(&mut findings, "proposed_design_reveals_execution_seed", "allocation_policy.seed_reveal_ref",
                "Do not reveal the allocation seed as an execution outcome in a proposed-only record.");
        }

        if self.state == "adjudicated" {
            match self.independent_review.as_ref() {
                Some(review) if review.disposition == "admit"
                    && !is_blank(&review.reviewer_ref)
                    && !is_blank(&review.reviewed_at)
                    && !is_blank(&review.review_record_ref) => {}
                Some(review) if review.disposition == "reject" => {
                    push(&mut findings, "independent_trial_review_rejected", "independent_review",
                        "Independent reviewer rejected trial-design evidence admission.");
                }
                _ => push(&mut findings, "independent_trial_review_missing", "independent_review",
                    "Adjudication requires an independent reviewer, timestamp, record and admit disposition."),
            }
        } else if self.independent_review.is_some() {
            push(&mut findings, "premature_independent_review", "independent_review",
                "Independent trial-design adjudication is only valid after observed execution.");
        }

        let trial_evidence_admitted = findings.is_empty()
            && self.state == "adjudicated"
            && self.independent_review.as_ref().is_some_and(|r| r.disposition == "admit");

        ValidationReport {
            findings,
            stratum_count: self.strata.len(),
            block_count: self.blocks.len(),
            period_count,
            executed_period_count,
            balanced_stratum_count: balanced_strata,
            trial_evidence_admitted,
        }
    }
}

fn sequence_catalog(conditions: &[String]) -> Vec<(String, Vec<String>)> {
    let set: HashSet<&str> = conditions.iter().map(String::as_str).collect();
    if conditions.len() == 2
        && set.len() == 2
        && set.contains("baseline_uncovered")
        && set.contains("lightshell_installed")
    {
        return vec![
            ("U-L".into(), vec!["baseline_uncovered".into(), "lightshell_installed".into()]),
            ("L-U".into(), vec!["lightshell_installed".into(), "baseline_uncovered".into()]),
        ];
    }
    if conditions.len() == 3
        && set.len() == 3
        && set.contains("baseline_uncovered")
        && set.contains("incumbent_cover")
        && set.contains("lightshell_installed")
    {
        return vec![
            ("U-I-L".into(), vec!["baseline_uncovered".into(), "incumbent_cover".into(), "lightshell_installed".into()]),
            ("U-L-I".into(), vec!["baseline_uncovered".into(), "lightshell_installed".into(), "incumbent_cover".into()]),
            ("I-U-L".into(), vec!["incumbent_cover".into(), "baseline_uncovered".into(), "lightshell_installed".into()]),
            ("I-L-U".into(), vec!["incumbent_cover".into(), "lightshell_installed".into(), "baseline_uncovered".into()]),
            ("L-U-I".into(), vec!["lightshell_installed".into(), "baseline_uncovered".into(), "incumbent_cover".into()]),
            ("L-I-U".into(), vec!["lightshell_installed".into(), "incumbent_cover".into(), "baseline_uncovered".into()]),
        ];
    }
    Vec::new()
}

fn validate_allocation_policy(policy: &AllocationPolicy, state: &str, findings: &mut Vec<Finding>) {
    if is_blank(&policy.algorithm_ref)
        || !is_sha256(&policy.seed_commitment_sha256)
        || is_blank(&policy.allocation_custodian_ref)
        || is_blank(&policy.allocation_method_ref)
        || !policy.locked_before_outcomes
        || !policy.no_outcome_adaptive_reassignment
    {
        push(findings, "allocation_policy_invalid", "allocation_policy",
            "Require an algorithm, precommitted SHA-256 seed, custodian, method, pre-outcome lock and no outcome-adaptive reassignment.");
    }
    if state == "proposed" && policy.seed_reveal_ref.is_some() {
        // Reported separately by the state-specific validator.
    }
}

fn validate_reset_policy(policy: &ResetPolicy, findings: &mut Vec<Finding>) {
    if is_blank(&policy.reset_procedure_ref)
        || is_blank(&policy.stabilization_rule_ref)
        || !policy.reset_before_every_period
        || !policy.carryover_check_between_periods
    {
        push(findings, "reset_policy_incomplete", "reset_policy",
            "Every period requires a declared reset/stabilization rule and inter-period carryover checks.");
    }
}

fn validate_reset(check: Option<&ResetCheck>, path: &str, findings: &mut Vec<Finding>) {
    match check {
        Some(check) if check.disposition == "passed"
            && !is_blank(&check.evidence_ref) && !is_blank(&check.rule_ref) => {}
        Some(check) => push(findings, "reset_check_not_passed", format!("{path}.reset_before"),
            format!("Reset must pass with evidence and rule references; got {}.", check.disposition)),
        None => push(findings, "reset_check_missing", format!("{path}.reset_before"),
            "Every observed period requires a successful reset/stabilization check."),
    }
}

fn validate_carryover(check: Option<&CarryoverCheck>, path: &str, findings: &mut Vec<Finding>) {
    match check {
        Some(check) if check.disposition == "passed"
            && !is_blank(&check.evidence_ref) && !is_blank(&check.rule_ref) => {}
        Some(check) => push(findings, "carryover_check_not_passed", format!("{path}.carryover_check"),
            format!("Carryover must pass with evidence and rule references; got {}.", check.disposition)),
        None => push(findings, "carryover_check_missing", format!("{path}.carryover_check"),
            "Every period after the first requires a successful carryover check."),
    }
}

fn validate_drift_check(
    check: Option<&BlockDriftCheck>,
    policy: &DriftPolicy,
    path: &str,
    findings: &mut Vec<Finding>,
) {
    let Some(check) = check else {
        push(findings, "block_drift_check_missing", format!("{path}.drift_check"),
            "Every observed block requires start/end sentinel measurements.");
        return;
    };
    if !check.start_value.is_finite()
        || !check.end_value.is_finite()
        || !check.expanded_uncertainty.is_finite()
        || check.expanded_uncertainty < 0.0
        || is_blank(&check.start_evidence_ref)
        || is_blank(&check.end_evidence_ref)
    {
        push(findings, "block_drift_evidence_invalid", format!("{path}.drift_check"),
            "Drift values must be finite, uncertainty non-negative and both measurements evidence-bound.");
        return;
    }
    let guarded_drift = (check.end_value - check.start_value).abs() + check.expanded_uncertainty;
    if guarded_drift > policy.maximum_abs_drift {
        push(findings, "block_drift_exceeds_limit", format!("{path}.drift_check"),
            format!("Conservative drift bound {guarded_drift} exceeds frozen limit {}.",
                policy.maximum_abs_drift));
    }
}

fn is_blank(value: &str) -> bool {
    value.trim().is_empty()
}

fn is_sha256(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("sha256:") else { return false; };
    hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn push(
    findings: &mut Vec<Finding>,
    code: &'static str,
    path: impl Into<String>,
    message: impl Into<String>,
) {
    findings.push(Finding { code, path: path.into(), message: message.into() });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn base_manifest() -> Value {
        let sequences: [(&str, [&str; 3]); 6] = [
            ("U-I-L", ["baseline_uncovered", "incumbent_cover", "lightshell_installed"]),
            ("U-L-I", ["baseline_uncovered", "lightshell_installed", "incumbent_cover"]),
            ("I-U-L", ["incumbent_cover", "baseline_uncovered", "lightshell_installed"]),
            ("I-L-U", ["incumbent_cover", "lightshell_installed", "baseline_uncovered"]),
            ("L-U-I", ["lightshell_installed", "baseline_uncovered", "incumbent_cover"]),
            ("L-I-U", ["lightshell_installed", "incumbent_cover", "baseline_uncovered"]),
        ];
        let blocks: Vec<Value> = sequences.iter().enumerate().map(|(block_index, (sequence_id, order))| {
            let periods: Vec<Value> = order.iter().enumerate().map(|(period_index, condition)| json!({
                "period_index": period_index,
                "condition_id": condition,
                "linked_system_run_ref": null,
                "reset_before": null,
                "carryover_check": null
            })).collect();
            json!({
                "block_id": format!("BLOCK-{block_index}"),
                "stratum_id": "ROBOT-STRATUM-A",
                "sequence_id": sequence_id,
                "allocation_receipt_ref": format!("ALLOCATION-{block_index}"),
                "periods": periods,
                "drift_check": null,
                "block_evidence_ref": null
            })
        }).collect();

        json!({
            "schema_version": "0.1.0",
            "campaign_id": "LIGHTSHELL-TRIAL-SYNTHETIC-001",
            "target_qualification_campaign_ref": "INSTALLED-SYSTEM-CAMPAIGN-SYNTHETIC-001",
            "state": "proposed",
            "design_mode": "randomized_counterbalanced_within_robot",
            "frozen_before_first_run": true,
            "preregistration_ref": "PREREGISTRATION-RECEIPT-SYNTHETIC",
            "preregistered_at": "2026-11-18T00:00:00Z",
            "protocol_digest": format!("sha256:{}", "a".repeat(64)),
            "condition_ids": ["baseline_uncovered", "incumbent_cover", "lightshell_installed"],
            "incumbent_comparator_ref": "INCUMBENT-COVER-REV-SYNTHETIC",
            "strata": [{
                "stratum_id": "ROBOT-STRATUM-A",
                "robot_serial": "ROBOT-SERIAL-SYNTHETIC",
                "firmware_revision": "FIRMWARE-REV-SYNTHETIC",
                "robot_configuration_digest": format!("sha256:{}", "b".repeat(64)),
                "tool_payload_digest": format!("sha256:{}", "c".repeat(64)),
                "mission_profile_ref": "MISSION-PROFILE-SYNTHETIC",
                "environment_stratum_ref": "ENVIRONMENT-STRATUM-SYNTHETIC",
                "minimum_blocks": 6
            }],
            "allocation_policy": {
                "algorithm_ref": "COUNTERBALANCE-ALLOCATOR-SYNTHETIC",
                "seed_commitment_sha256": format!("sha256:{}", "d".repeat(64)),
                "seed_reveal_ref": null,
                "allocation_custodian_ref": "INDEPENDENT-CUSTODIAN-SYNTHETIC",
                "allocation_method_ref": "ALLOCATION-METHOD-SYNTHETIC",
                "locked_before_outcomes": true,
                "no_outcome_adaptive_reassignment": true
            },
            "reset_policy": {
                "reset_procedure_ref": "RESET-PROCEDURE-SYNTHETIC",
                "stabilization_rule_ref": "STABILIZATION-RULE-SYNTHETIC",
                "reset_before_every_period": true,
                "carryover_check_between_periods": true
            },
            "drift_policy": {
                "sentinel_metric_id": "REFERENCE-SENTINEL-SYNTHETIC",
                "maximum_abs_drift": 0.5,
                "uncertainty_guarded": true,
                "start_end_measurements_required": true
            },
            "blocks": blocks,
            "independent_review": null
        })
    }

    fn make_observed(manifest: &mut Value) {
        manifest["state"] = json!("observed");
        manifest["allocation_policy"]["seed_reveal_ref"] = json!("ALLOCATION-SEED-REVEAL-SYNTHETIC");
        for (block_index, block) in manifest["blocks"].as_array_mut().unwrap().iter_mut().enumerate() {
            for (period_index, period) in block["periods"].as_array_mut().unwrap().iter_mut().enumerate() {
                period["linked_system_run_ref"] = json!(format!("SYSTEM-RUN-{block_index}-{period_index}"));
                period["reset_before"] = json!({
                    "disposition": "passed",
                    "evidence_ref": format!("RESET-{block_index}-{period_index}"),
                    "rule_ref": "RESET-RULE-SYNTHETIC"
                });
                if period_index > 0 {
                    period["carryover_check"] = json!({
                        "disposition": "passed",
                        "evidence_ref": format!("CARRYOVER-{block_index}-{period_index}"),
                        "rule_ref": "CARRYOVER-RULE-SYNTHETIC"
                    });
                }
            }
            block["drift_check"] = json!({
                "start_value": 20.0,
                "end_value": 20.1,
                "expanded_uncertainty": 0.1,
                "start_evidence_ref": format!("SENTINEL-START-{block_index}"),
                "end_evidence_ref": format!("SENTINEL-END-{block_index}")
            });
            block["block_evidence_ref"] = json!(format!("BLOCK-EVIDENCE-{block_index}"));
        }
    }

    #[test]
    fn proposed_counterbalanced_design_is_valid_without_trial_outcomes() {
        let report = parse_and_validate(&base_manifest().to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.block_count, 6);
        assert_eq!(report.period_count, 18);
        assert_eq!(report.executed_period_count, 0);
        assert_eq!(report.balanced_stratum_count, 1);
        assert!(!report.can_admit_trial_evidence());
    }

    #[test]
    fn all_six_three_arm_sequences_are_counterbalanced() {
        let report = parse_and_validate(&base_manifest().to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.balanced_stratum_count, 1);
    }

    #[test]
    fn unbalanced_sequence_allocation_is_rejected() {
        let mut manifest = base_manifest();
        manifest["blocks"].as_array_mut().unwrap().pop();
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|finding| finding.code == "counterbalance_allocation_unbalanced"));
    }

    #[test]
    fn period_order_must_match_pre_registered_sequence() {
        let mut manifest = base_manifest();
        manifest["blocks"][0]["periods"][0]["condition_id"] = json!("lightshell_installed");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|finding| finding.code == "period_order_does_not_match_allocation"));
    }

    #[test]
    fn incumbent_arm_requires_exact_comparator_binding() {
        let mut manifest = base_manifest();
        manifest["incumbent_comparator_ref"] = Value::Null;
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|finding| finding.code == "incumbent_comparator_binding_missing"));
    }

    #[test]
    fn two_arm_design_is_supported_without_an_incumbent() {
        let mut manifest = base_manifest();
        manifest["condition_ids"] = json!(["baseline_uncovered", "lightshell_installed"]);
        manifest["incumbent_comparator_ref"] = Value::Null;
        manifest["strata"][0]["minimum_blocks"] = json!(2);
        let orders: [(&str, [&str; 2]); 2] = [
            ("U-L", ["baseline_uncovered", "lightshell_installed"]),
            ("L-U", ["lightshell_installed", "baseline_uncovered"]),
        ];
        let blocks: Vec<Value> = orders.iter().enumerate().map(|(block_index, (sequence_id, order))| {
            let periods: Vec<Value> = order.iter().enumerate().map(|(period_index, condition)| json!({
                "period_index": period_index,
                "condition_id": condition,
                "linked_system_run_ref": null,
                "reset_before": null,
                "carryover_check": null
            })).collect();
            json!({
                "block_id": format!("TWO-ARM-BLOCK-{block_index}"),
                "stratum_id": "ROBOT-STRATUM-A",
                "sequence_id": sequence_id,
                "allocation_receipt_ref": format!("TWO-ARM-ALLOCATION-{block_index}"),
                "periods": periods,
                "drift_check": null,
                "block_evidence_ref": null
            })
        }).collect();
        manifest["blocks"] = json!(blocks);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
    }

    #[test]
    fn allocation_seed_commitment_must_be_sha256() {
        let mut manifest = base_manifest();
        manifest["allocation_policy"]["seed_commitment_sha256"] = json!("not-a-digest");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|finding| finding.code == "allocation_policy_invalid"));
    }

    #[test]
    fn design_must_be_frozen_before_first_run() {
        let mut manifest = base_manifest();
        manifest["frozen_before_first_run"] = json!(false);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|finding| finding.code == "preregistration_not_frozen"));
    }

    #[test]
    fn outcome_adaptive_reassignment_is_rejected() {
        let mut manifest = base_manifest();
        manifest["allocation_policy"]["no_outcome_adaptive_reassignment"] = json!(false);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|finding| finding.code == "allocation_policy_invalid"));
    }

    #[test]
    fn observed_periods_require_unique_exact_system_run_links() {
        let mut manifest = base_manifest();
        make_observed(&mut manifest);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.executed_period_count, 18);
        let first_run_ref = manifest["blocks"][0]["periods"][0]["linked_system_run_ref"].clone();
        manifest["blocks"][1]["periods"][0]["linked_system_run_ref"] = first_run_ref;
        let duplicate = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(duplicate.findings.iter().any(|finding| finding.code == "duplicate_system_run_reference"));
    }

    #[test]
    fn missing_system_run_link_blocks_observed_trial() {
        let mut manifest = base_manifest();
        make_observed(&mut manifest);
        manifest["blocks"][0]["periods"][1]["linked_system_run_ref"] = Value::Null;
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|finding| finding.code == "period_system_run_missing"));
    }

    #[test]
    fn failed_reset_blocks_observed_trial() {
        let mut manifest = base_manifest();
        make_observed(&mut manifest);
        manifest["blocks"][0]["periods"][1]["reset_before"]["disposition"] = json!("failed");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|finding| finding.code == "reset_check_not_passed"));
    }

    #[test]
    fn missing_carryover_check_blocks_observed_trial() {
        let mut manifest = base_manifest();
        make_observed(&mut manifest);
        manifest["blocks"][0]["periods"][2]["carryover_check"] = Value::Null;
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|finding| finding.code == "carryover_check_missing"));
    }

    #[test]
    fn drift_plus_uncertainty_must_fit_inside_frozen_limit() {
        let mut manifest = base_manifest();
        make_observed(&mut manifest);
        manifest["blocks"][0]["drift_check"]["end_value"] = json!(20.4);
        manifest["blocks"][0]["drift_check"]["expanded_uncertainty"] = json!(0.2);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|finding| finding.code == "block_drift_exceeds_limit"));
    }

    #[test]
    fn drift_check_requires_both_evidence_receipts() {
        let mut manifest = base_manifest();
        make_observed(&mut manifest);
        manifest["blocks"][0]["drift_check"]["end_evidence_ref"] = json!("");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|finding| finding.code == "block_drift_evidence_invalid"));
    }

    #[test]
    fn observed_campaign_requires_seed_reveal_receipt() {
        let mut manifest = base_manifest();
        make_observed(&mut manifest);
        manifest["allocation_policy"]["seed_reveal_ref"] = Value::Null;
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|finding| finding.code == "allocation_seed_reveal_missing"));
    }

    #[test]
    fn adjudicated_trial_requires_independent_review() {
        let mut manifest = base_manifest();
        make_observed(&mut manifest);
        manifest["state"] = json!("adjudicated");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|finding| finding.code == "independent_trial_review_missing"));
        assert!(!report.can_admit_trial_evidence());
    }

    #[test]
    fn independently_reviewed_adjudicated_trial_can_be_admitted() {
        let mut manifest = base_manifest();
        make_observed(&mut manifest);
        manifest["state"] = json!("adjudicated");
        manifest["independent_review"] = json!({
            "disposition": "admit",
            "reviewer_ref": "INDEPENDENT-REVIEWER-SYNTHETIC",
            "reviewed_at": "2026-11-18T12:00:00Z",
            "review_record_ref": "INDEPENDENT-REVIEW-RECEIPT-SYNTHETIC"
        });
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert!(report.can_admit_trial_evidence());
    }

    #[test]
    fn proposed_design_cannot_contain_observed_runs() {
        let mut manifest = base_manifest();
        manifest["blocks"][0]["periods"][0]["linked_system_run_ref"] = json!("UNPLANNED-RUN");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|finding| finding.code == "proposed_design_contains_observations"));
    }

    #[test]
    fn unknown_strata_and_unknown_fields_are_rejected() {
        let mut manifest = base_manifest();
        manifest["blocks"][0]["stratum_id"] = json!("UNDECLARED-STRATUM");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|finding| finding.code == "unknown_block_stratum"));
        manifest["invented_robot_authority"] = json!(true);
        assert!(parse_and_validate(&manifest.to_string()).is_err());
    }

    #[test]
    fn block_allocation_receipts_must_be_unique() {
        let mut manifest = base_manifest();
        let first_receipt = manifest["blocks"][0]["allocation_receipt_ref"].clone();
        manifest["blocks"][1]["allocation_receipt_ref"] = first_receipt;
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|finding| finding.code == "missing_or_duplicate_allocation_receipt"));
    }
}
