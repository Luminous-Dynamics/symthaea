// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Structural verifier for the NANO-013 prospective rescue-controller contract.
//!
//! This verifier checks protocol receipts and internally consistent sampling/accounting.
//! It cannot prove that a declared sample was externally timestamped, that a reference
//! service produced a label, that costs were billed, or that an evaluated trajectory is
//! physically correct. Those require independent source-system and scientific evidence.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufRead, BufReader, Cursor};
use std::path::{Component, Path, PathBuf};

const MAX_FILE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_JSONL_ROWS: usize = 1_000_000;
const MAX_FRAME_ROWS: usize = 5_000_000;

#[derive(Debug, Clone)]
pub(crate) struct ContractFinding {
    pub code: String,
    pub detail: String,
}

#[derive(Debug, Default)]
pub(crate) struct RescueContractAudit {
    pub provided: bool,
    pub findings: Vec<ContractFinding>,
    pub checks_completed: Vec<String>,
}

impl RescueContractAudit {
    fn fail(&mut self, code: &str, detail: impl Into<String>) {
        self.findings.push(ContractFinding {
            code: code.to_owned(),
            detail: detail.into(),
        });
    }

    fn checked(&mut self, check: &str) {
        self.checks_completed.push(check.to_owned());
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DecisionRecord {
    schema_version: String,
    run_id: String,
    trajectory_id: String,
    parent_group_id: String,
    window_id: String,
    sequence: u64,
    decision_step: u64,
    max_observed_step: u64,
    action: String,
    future_data_read: bool,
    heldout_label_read: bool,
    observed_state_sha256: String,
    available_inputs_manifest_sha256: String,
    policy_revision_sha256: String,
    trigger_reason: String,
    // Value is intentional: these keys are required by contract but may contain JSON null.
    risk_score: serde_json::Value,
    reference_request_id: serde_json::Value,
    policy_cost: f64,
    policy_cost_unit: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrameRecord {
    window_id: String,
    trajectory_id: String,
    parent_group_id: String,
    stratum_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SamplingManifest {
    schema_version: String,
    run_id: String,
    evaluation_id: String,
    policy_revision_sha256: String,
    sampling_frame_path: String,
    sampling_frame_sha256: String,
    preregistration_receipt_path: String,
    manifest_visibility: String,
    seed_reveal_path: String,
    seed_commitment_sha256: String,
    frame_population_size: usize,
    sampling_method: String,
    frozen_before_policy_run: bool,
    policy_blind: bool,
    strata: Vec<Stratum>,
    sampled_windows: Vec<SampledWindow>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Stratum {
    stratum_id: String,
    eligible_count: usize,
    sampled_count: usize,
    inclusion_probability: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SampledWindow {
    window_id: String,
    parent_group_id: String,
    trajectory_id: String,
    stratum_id: String,
    inclusion_probability: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PreregistrationReceipt {
    schema_version: String,
    run_id: String,
    audit_sampling_manifest_sha256: String,
    policy_revision_sha256: String,
    sampling_frame_sha256: String,
    seed_commitment_sha256: String,
    sampling_method: String,
    frame_population_size: usize,
    frozen_before_policy_run: bool,
    policy_blind: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuditResult {
    schema_version: String,
    run_id: String,
    sampling_manifest_sha256: String,
    window_id: String,
    parent_group_id: String,
    trajectory_id: String,
    stratum_id: String,
    inclusion_probability: f64,
    reference_status: String,
    // Required nullable fields: missing and explicit null have different meanings.
    severe_event: serde_json::Value,
    physical_integrity_pass: serde_json::Value,
    policy_triggered_reference: bool,
    labels_revealed_to_policy: bool,
    audit_reference_cost: f64,
    audit_cost_unit: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SeedReveal {
    schema_version: String,
    run_id: String,
    seed: String,
}

/// Audit an optional rescue-controller bundle under run_dir/rescue-controller/.
///
/// If no rescue-controller artifacts exist, this is a no-op so legacy MADE runs remain
/// auditable. Once any rescue-controller artifact is present, all required artifacts must
/// exist and agree; partial bundles fail closed.
pub(crate) fn audit_optional_rescue_contract(run_dir: &Path) -> RescueContractAudit {
    let mut audit = RescueContractAudit::default();
    let root = run_dir.join("rescue-controller");
    let required = [
        "decision-ledger.jsonl",
        "eligible-window-frame.jsonl",
        "audit-sampling-manifest.json",
        "preregistration-receipt.json",
        "sampling-seed-reveal.json",
        "blinded-audit-results.jsonl",
    ];
    let present = required
        .iter()
        .filter(|name| fs::symlink_metadata(root.join(name)).is_ok())
        .count();
    if present == 0 {
        return audit;
    }
    audit.provided = true;
    if present != required.len() {
        audit.fail(
            "rescue_contract_incomplete_bundle",
            format!("found {present} of {} required rescue-controller artifacts", required.len()),
        );
        return audit;
    }

    let manifest_bytes = match read_regular_file(run_dir, Path::new("rescue-controller/audit-sampling-manifest.json")) {
        Ok(bytes) => bytes,
        Err(error) => {
            audit.fail("rescue_manifest_unreadable", error);
            return audit;
        }
    };
    let manifest_digest = sha256_hex(&manifest_bytes);
    let manifest: SamplingManifest = match serde_json::from_slice(&manifest_bytes) {
        Ok(value) => value,
        Err(error) => {
            audit.fail("rescue_manifest_invalid_json_or_schema", error.to_string());
            return audit;
        }
    };

    if manifest.schema_version != "nano-rescue-audit-sampling-v1" {
        audit.fail("rescue_manifest_version", "unsupported audit-sampling manifest schema_version");
    }
    if manifest.run_id.trim().is_empty() || manifest.evaluation_id.trim().is_empty() {
        audit.fail("rescue_manifest_identity_empty", "run_id and evaluation_id must be non-empty");
    }
    if !is_sha256(&manifest.policy_revision_sha256)
        || !is_sha256(&manifest.sampling_frame_sha256)
        || !is_sha256(&manifest.seed_commitment_sha256)
    {
        audit.fail("rescue_manifest_bad_digest", "policy/frame/receipt/seed digests must be lowercase SHA-256");
    }
    if manifest.manifest_visibility != "evaluator_sealed_until_policy_run_complete" {
        audit.fail("rescue_manifest_not_sealed", "sample IDs must remain evaluator-sealed until the policy run completes");
    }
    if !manifest.frozen_before_policy_run || !manifest.policy_blind {
        audit.fail(
            "rescue_manifest_not_preregistered_blind",
            "audit sample must declare frozen_before_policy_run=true and policy_blind=true",
        );
    }
    if !matches!(
        manifest.sampling_method.as_str(),
        "simple_random_without_replacement" | "stratified_random_without_replacement"
    ) {
        audit.fail("rescue_sampling_method_unsupported", "unsupported or adaptive sampling method");
    }
    if manifest.sampling_method == "simple_random_without_replacement" && manifest.strata.len() != 1 {
        audit.fail("rescue_simple_random_multiple_strata", "simple random sampling must declare exactly one stratum");
    }
    if manifest.frame_population_size == 0 || manifest.sampled_windows.is_empty() {
        audit.fail("rescue_sampling_empty", "sampling frame and blinded audit sample must be non-empty");
    }
    if manifest.sampled_windows.len() > MAX_FRAME_ROWS || manifest.strata.len() > MAX_FRAME_ROWS {
        audit.fail("rescue_sampling_too_large", "sampling manifest exceeds row-count limit");
        return audit;
    }

    let frame_bytes = match read_regular_file(run_dir, Path::new(&manifest.sampling_frame_path)) {
        Ok(bytes) => bytes,
        Err(error) => {
            audit.fail("rescue_frame_unreadable", error);
            return audit;
        }
    };
    if sha256_hex(&frame_bytes) != manifest.sampling_frame_sha256 {
        audit.fail("rescue_frame_digest_mismatch", "eligible-window frame digest does not match the frozen manifest");
    }
    let frame_rows: Vec<FrameRecord> = match parse_jsonl(&frame_bytes, MAX_FRAME_ROWS) {
        Ok(rows) => rows,
        Err(error) => {
            audit.fail("rescue_frame_invalid_jsonl", error);
            return audit;
        }
    };
    if frame_rows.len() != manifest.frame_population_size {
        audit.fail(
            "rescue_frame_population_mismatch",
            format!("manifest population={} but frame has {} rows", manifest.frame_population_size, frame_rows.len()),
        );
    }
    let mut frame_by_id = BTreeMap::new();
    let mut frame_strata: BTreeMap<String, usize> = BTreeMap::new();
    for row in &frame_rows {
        if row.window_id.trim().is_empty()
            || row.trajectory_id.trim().is_empty()
            || row.parent_group_id.trim().is_empty()
            || row.stratum_id.trim().is_empty()
        {
            audit.fail("rescue_frame_empty_identity", "frame rows require window, trajectory, parent group and stratum IDs");
        }
        if frame_by_id.insert(row.window_id.clone(), row).is_some() {
            audit.fail("rescue_frame_duplicate_window", format!("duplicate frame window_id {}", row.window_id));
        }
        *frame_strata.entry(row.stratum_id.clone()).or_default() += 1;
    }
    audit.checked("rescue_sampling_frame_digest_and_rows_checked");

    let receipt_bytes = match read_regular_file(run_dir, Path::new(&manifest.preregistration_receipt_path)) {
        Ok(bytes) => bytes,
        Err(error) => {
            audit.fail("rescue_receipt_unreadable", error);
            return audit;
        }
    };
    let receipt: PreregistrationReceipt = match serde_json::from_slice(&receipt_bytes) {
        Ok(value) => value,
        Err(error) => {
            audit.fail("rescue_receipt_invalid_json_or_schema", error.to_string());
            return audit;
        }
    };
    if receipt.schema_version != "nano-rescue-preregistration-v1"
        || receipt.run_id != manifest.run_id
        || receipt.audit_sampling_manifest_sha256 != manifest_digest
        || receipt.policy_revision_sha256 != manifest.policy_revision_sha256
        || receipt.sampling_frame_sha256 != manifest.sampling_frame_sha256
        || receipt.seed_commitment_sha256 != manifest.seed_commitment_sha256
        || receipt.sampling_method != manifest.sampling_method
        || receipt.frame_population_size != manifest.frame_population_size
        || !receipt.frozen_before_policy_run
        || !receipt.policy_blind
    {
        audit.fail("rescue_preregistration_receipt_mismatch", "preregistration receipt does not bind the current frozen evaluation manifest");
    }

    let seed_bytes = match read_regular_file(run_dir, Path::new(&manifest.seed_reveal_path)) {
        Ok(bytes) => bytes,
        Err(error) => {
            audit.fail("rescue_seed_reveal_unreadable", error);
            return audit;
        }
    };
    let seed_reveal: SeedReveal = match serde_json::from_slice(&seed_bytes) {
        Ok(value) => value,
        Err(error) => {
            audit.fail("rescue_seed_reveal_invalid_json_or_schema", error.to_string());
            return audit;
        }
    };
    if seed_reveal.schema_version != "nano-rescue-seed-reveal-v1"
        || seed_reveal.run_id != manifest.run_id
        || seed_reveal.seed.trim().is_empty()
        || sha256_hex(seed_reveal.seed.as_bytes()) != manifest.seed_commitment_sha256
    {
        audit.fail("rescue_seed_commitment_mismatch", "post-run seed reveal does not match the preregistered commitment");
    }

    let mut stratum_specs = BTreeMap::new();
    let mut manifest_eligible_sum = 0usize;
    let mut manifest_sample_sum = 0usize;
    for stratum in &manifest.strata {
        if stratum.stratum_id.trim().is_empty()
            || stratum.eligible_count == 0
            || stratum.sampled_count == 0
            || stratum.sampled_count > stratum.eligible_count
            || !probability(stratum.inclusion_probability)
        {
            audit.fail("rescue_stratum_invalid", format!("invalid stratum {}", stratum.stratum_id));
            continue;
        }
        let expected = stratum.sampled_count as f64 / stratum.eligible_count as f64;
        if !close(stratum.inclusion_probability, expected) {
            audit.fail("rescue_stratum_probability_mismatch", format!("stratum {} inclusion probability must equal n_h/N_h", stratum.stratum_id));
        }
        if frame_strata.get(&stratum.stratum_id).copied().unwrap_or(0) != stratum.eligible_count {
            audit.fail("rescue_stratum_population_mismatch", format!("stratum {} population count differs from eligible-window frame", stratum.stratum_id));
        }
        if stratum_specs.insert(stratum.stratum_id.clone(), stratum).is_some() {
            audit.fail("rescue_duplicate_stratum", format!("duplicate stratum_id {}", stratum.stratum_id));
        }
        manifest_eligible_sum = manifest_eligible_sum.saturating_add(stratum.eligible_count);
        manifest_sample_sum = manifest_sample_sum.saturating_add(stratum.sampled_count);
    }
    if manifest_eligible_sum != manifest.frame_population_size {
        audit.fail("rescue_strata_population_sum", "stratum eligible counts do not sum to frame_population_size");
    }
    if manifest_sample_sum != manifest.sampled_windows.len() {
        audit.fail("rescue_strata_sample_sum", "stratum sampled counts do not sum to sampled_windows length");
    }
    if frame_strata.len() != stratum_specs.len() || frame_strata.keys().any(|id| !stratum_specs.contains_key(id)) {
        audit.fail("rescue_frame_strata_mismatch", "strata in frame and manifest do not match exactly");
    }

    let mut sample_by_id = BTreeMap::new();
    let mut sampled_by_stratum: BTreeMap<String, usize> = BTreeMap::new();
    for sample in &manifest.sampled_windows {
        if !probability(sample.inclusion_probability) {
            audit.fail("rescue_sample_probability_invalid", format!("window {} has an invalid inclusion probability", sample.window_id));
        }
        if sample_by_id.insert(sample.window_id.clone(), sample).is_some() {
            audit.fail("rescue_sample_duplicate_window", format!("duplicate sampled window {}", sample.window_id));
        }
        let Some(frame) = frame_by_id.get(&sample.window_id) else {
            audit.fail("rescue_sample_not_in_frame", format!("sampled window {} is absent from frozen frame", sample.window_id));
            continue;
        };
        if frame.parent_group_id != sample.parent_group_id
            || frame.trajectory_id != sample.trajectory_id
            || frame.stratum_id != sample.stratum_id
        {
            audit.fail("rescue_sample_frame_identity_mismatch", format!("sampled window {} differs from its frame identity", sample.window_id));
        }
        let Some(stratum) = stratum_specs.get(&sample.stratum_id) else {
            audit.fail("rescue_sample_unknown_stratum", format!("sampled window {} names unknown stratum {}", sample.window_id, sample.stratum_id));
            continue;
        };
        if !close(sample.inclusion_probability, stratum.inclusion_probability) {
            audit.fail("rescue_sample_probability_mismatch", format!("window {} probability differs from its stratum probability", sample.window_id));
        }
        *sampled_by_stratum.entry(sample.stratum_id.clone()).or_default() += 1;
    }
    for (id, stratum) in &stratum_specs {
        if sampled_by_stratum.get(id).copied().unwrap_or(0) != stratum.sampled_count {
            audit.fail("rescue_sample_count_mismatch", format!("sample count differs for stratum {id}"));
        }
    }

    let decision_bytes = match read_regular_file(run_dir, Path::new("rescue-controller/decision-ledger.jsonl")) {
        Ok(bytes) => bytes,
        Err(error) => {
            audit.fail("rescue_decision_ledger_unreadable", error);
            return audit;
        }
    };
    let decisions: Vec<DecisionRecord> = match parse_jsonl(&decision_bytes, MAX_JSONL_ROWS) {
        Ok(rows) => rows,
        Err(error) => {
            audit.fail("rescue_decision_ledger_invalid_jsonl", error);
            return audit;
        }
    };
    if decisions.is_empty() {
        audit.fail("rescue_decision_ledger_empty", "decision ledger must have at least one decision");
    }
    let mut decision_by_window = BTreeMap::new();
    let mut seen_decisions = BTreeSet::new();
    let mut sequence_by_trajectory: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    let mut decision_run: Option<&str> = None;
    let mut decision_policy: Option<&str> = None;
    let mut decision_cost_unit: Option<&str> = None;
    for row in &decisions {
        if row.schema_version != "nano-rescue-decision-v1" || row.run_id != manifest.run_id {
            audit.fail("rescue_decision_identity_mismatch", format!("decision {} has wrong schema version or run_id", row.window_id));
        }
        if row.trajectory_id.trim().is_empty() || row.parent_group_id.trim().is_empty() || row.window_id.trim().is_empty() || row.trigger_reason.trim().is_empty() {
            audit.fail("rescue_decision_empty_identity", "decision records require trajectory/group/window IDs and trigger_reason");
        }
        if row.max_observed_step > row.decision_step {
            audit.fail("rescue_future_step_observed", format!("window {} observes step {} after decision step {}", row.window_id, row.max_observed_step, row.decision_step));
        }
        if !matches!(
            row.trigger_reason.as_str(),
            "nominal" | "uncertainty" | "shift_alarm" | "budget_guard" | "reference_result" | "integrity_guard" | "operator_override"
        ) {
            audit.fail("rescue_trigger_reason_invalid", format!("unknown trigger reason at window {}", row.window_id));
        }
        if row.future_data_read || row.heldout_label_read {
            audit.fail("rescue_test_leakage_declared", format!("window {} declares future-data or held-out-label access", row.window_id));
        }
        if !matches!(row.action.as_str(), "continue" | "conservative_continue" | "abstain_stop" | "request_reference") {
            audit.fail("rescue_action_invalid", format!("invalid action {} at window {}", row.action, row.window_id));
        }
        let reference_request_id = row.reference_request_id.as_str().unwrap_or("");
        if row.action == "request_reference" && reference_request_id.trim().is_empty() {
            audit.fail("rescue_reference_request_id_missing", format!("window {} requests reference without request ID", row.window_id));
        }
        if row.action != "request_reference" && !row.reference_request_id.is_null() {
            audit.fail("rescue_unexpected_reference_request_id", format!("window {} has request ID but action is not request_reference", row.window_id));
        }
        if !is_sha256(&row.observed_state_sha256)
            || !is_sha256(&row.available_inputs_manifest_sha256)
            || !is_sha256(&row.policy_revision_sha256)
        {
            audit.fail("rescue_decision_digest_invalid", format!("decision {} contains invalid SHA-256", row.window_id));
        }
        if row.policy_revision_sha256 != manifest.policy_revision_sha256 {
            audit.fail("rescue_policy_revision_mismatch", format!("decision {} is from a different policy revision", row.window_id));
        }
        if !row.risk_score.is_null()
            && row.risk_score.as_f64().map_or(true, |value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            audit.fail("rescue_risk_score_invalid", format!("window {} risk_score must be null or finite in [0,1]", row.window_id));
        }
        if !row.policy_cost.is_finite() || row.policy_cost < 0.0 || row.policy_cost_unit.trim().is_empty() {
            audit.fail("rescue_policy_cost_invalid", format!("window {} has invalid policy cost/unit", row.window_id));
        }
        if decision_run.replace(row.run_id.as_str()).is_some_and(|old| old != row.run_id) {
            audit.fail("rescue_multiple_run_ids", "decision ledger contains more than one run_id");
        }
        if decision_policy.replace(row.policy_revision_sha256.as_str()).is_some_and(|old| old != row.policy_revision_sha256) {
            audit.fail("rescue_multiple_policy_revisions", "decision ledger mixes policy revisions");
        }
        if decision_cost_unit.replace(row.policy_cost_unit.as_str()).is_some_and(|old| old != row.policy_cost_unit) {
            audit.fail("rescue_mixed_policy_cost_units", "policy decision costs must use one declared unit per run");
        }
        sequence_by_trajectory.entry(row.trajectory_id.clone()).or_default().push(row.sequence);
        let key = format!("{}\u{1f}{}", row.trajectory_id, row.sequence);
        if !seen_decisions.insert(key) {
            audit.fail("rescue_decision_sequence_duplicate", format!("duplicate sequence for trajectory {}", row.trajectory_id));
        }
        if decision_by_window.insert(row.window_id.clone(), row).is_some() {
            audit.fail("rescue_decision_window_duplicate", format!("multiple decision records for window {}", row.window_id));
        }
    }
    for (trajectory, sequences) in &mut sequence_by_trajectory {
        sequences.sort_unstable();
        for (index, value) in sequences.iter().enumerate() {
            if *value != (index as u64) + 1 {
                audit.fail("rescue_decision_sequence_gap", format!("trajectory {trajectory} sequence must be contiguous from 1"));
                break;
            }
        }
    }
    audit.checked("rescue_decision_ledger_semantics_checked");

    let audit_bytes = match read_regular_file(run_dir, Path::new("rescue-controller/blinded-audit-results.jsonl")) {
        Ok(bytes) => bytes,
        Err(error) => {
            audit.fail("rescue_blinded_audit_unreadable", error);
            return audit;
        }
    };
    let results: Vec<AuditResult> = match parse_jsonl(&audit_bytes, MAX_JSONL_ROWS) {
        Ok(rows) => rows,
        Err(error) => {
            audit.fail("rescue_blinded_audit_invalid_jsonl", error);
            return audit;
        }
    };
    let mut results_by_id = BTreeMap::new();
    let mut audit_cost_unit: Option<&str> = None;
    for row in &results {
        if row.schema_version != "nano-rescue-audit-result-v1" || row.run_id != manifest.run_id {
            audit.fail("rescue_audit_result_identity_mismatch", format!("audit result {} has wrong schema version or run_id", row.window_id));
        }
        if row.sampling_manifest_sha256 != manifest_digest {
            audit.fail("rescue_audit_manifest_digest_mismatch", format!("audit result {} is not bound to the raw sampling-manifest bytes", row.window_id));
        }
        if row.labels_revealed_to_policy {
            audit.fail("rescue_audit_labels_revealed", format!("audit labels for window {} were revealed to policy", row.window_id));
        }
        if !probability(row.inclusion_probability) || !row.audit_reference_cost.is_finite() || row.audit_reference_cost < 0.0 || row.audit_cost_unit.trim().is_empty() {
            audit.fail("rescue_audit_cost_or_probability_invalid", format!("audit result {} has invalid probability or cost", row.window_id));
        }
        if audit_cost_unit.replace(row.audit_cost_unit.as_str()).is_some_and(|old| old != row.audit_cost_unit) {
            audit.fail("rescue_mixed_audit_cost_units", "evaluation-audit costs must use one declared unit per run");
        }
        match row.reference_status.as_str() {
            "evaluated" => {
                if row.severe_event.as_bool().is_none() || row.physical_integrity_pass.as_bool().is_none() {
                    audit.fail("rescue_audit_label_missing", format!("evaluated window {} needs severe_event and physical_integrity_pass", row.window_id));
                }
            }
            "reference_failed" | "reference_unavailable" => {
                if !row.severe_event.is_null() || !row.physical_integrity_pass.is_null() {
                    audit.fail("rescue_audit_failed_reference_has_labels", format!("failed reference for {} cannot carry outcome labels", row.window_id));
                }
            }
            _ => audit.fail("rescue_audit_reference_status_invalid", format!("invalid reference status for {}", row.window_id)),
        }
        if results_by_id.insert(row.window_id.clone(), row).is_some() {
            audit.fail("rescue_audit_result_duplicate_window", format!("duplicate audit-result window {}", row.window_id));
        }
    }

    if results_by_id.len() != sample_by_id.len() {
        audit.fail("rescue_audit_sample_count_mismatch", format!("manifest samples={} but audit results={}", sample_by_id.len(), results_by_id.len()));
    }
    for (window_id, sample) in &sample_by_id {
        let Some(result) = results_by_id.get(window_id) else {
            audit.fail("rescue_audit_sample_missing", format!("sampled window {window_id} has no audit result"));
            continue;
        };
        if result.parent_group_id != sample.parent_group_id
            || result.trajectory_id != sample.trajectory_id
            || result.stratum_id != sample.stratum_id
            || !close(result.inclusion_probability, sample.inclusion_probability)
        {
            audit.fail("rescue_audit_sample_identity_mismatch", format!("audit result {window_id} differs from sampled-window manifest"));
        }
        let Some(decision) = decision_by_window.get(window_id) else {
            audit.fail("rescue_audit_decision_missing", format!("audited window {window_id} has no logged policy decision"));
            continue;
        };
        if decision.parent_group_id != result.parent_group_id
            || decision.trajectory_id != result.trajectory_id
            || (decision.action == "request_reference") != result.policy_triggered_reference
        {
            audit.fail("rescue_policy_action_reconciliation_failed", format!("policy decision and blinded audit outcome disagree for {window_id}"));
        }
    }
    for window_id in results_by_id.keys() {
        if !sample_by_id.contains_key(window_id) {
            audit.fail("rescue_audit_unplanned_window", format!("audit result {window_id} was not in the frozen sample"));
        }
    }
    audit.checked("rescue_blinded_sample_and_decision_reconciliation_checked");
    audit.checked("rescue_policy_and_audit_costs_kept_separate");

    audit
}

fn read_regular_file(root: &Path, relative: &Path) -> Result<Vec<u8>, String> {
    if relative.as_os_str().is_empty()
        || relative.components().any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("unsafe relative path: {}", relative.display()));
    }
    let mut current = PathBuf::from(root);
    for component in relative.components() {
        if let Component::Normal(name) = component {
            current.push(name);
            let metadata = fs::symlink_metadata(&current).map_err(|error| format!("{}: {error}", current.display()))?;
            if metadata.file_type().is_symlink() {
                return Err(format!("symlink path component is forbidden: {}", current.display()));
            }
            if current != root.join(relative) && !metadata.is_dir() {
                return Err(format!("non-directory parent component: {}", current.display()));
            }
        }
    }
    let metadata = fs::symlink_metadata(&current).map_err(|error| format!("{}: {error}", current.display()))?;
    if !metadata.is_file() || metadata.len() > MAX_FILE_BYTES {
        return Err(format!("not a regular file or exceeds size limit: {}", current.display()));
    }
    fs::read(&current).map_err(|error| format!("{}: {error}", current.display()))
}

fn parse_jsonl<T: for<'de> Deserialize<'de>>(bytes: &[u8], maximum_rows: usize) -> Result<Vec<T>, String> {
    let reader = BufReader::new(Cursor::new(bytes));
    let mut rows = Vec::new();
    for (line_index, line) in reader.lines().enumerate() {
        let line = line.map_err(|error| error.to_string())?;
        if line.trim().is_empty() {
            return Err(format!("blank JSONL line at {}", line_index + 1));
        }
        if rows.len() >= maximum_rows {
            return Err(format!("JSONL row limit exceeded ({maximum_rows})"));
        }
        let row = serde_json::from_str(&line).map_err(|error| format!("line {}: {error}", line_index + 1))?;
        rows.push(row);
    }
    Ok(rows)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn probability(value: f64) -> bool {
    value.is_finite() && value > 0.0 && value <= 1.0
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1.0e-12
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("nano-rescue-valid")
    }

    fn copy_fixture_to_temp() -> (PathBuf, PathBuf) {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        let parent = std::env::temp_dir().join(format!(
            "symthaea-nano-rescue-test-{}-{stamp}",
            std::process::id()
        ));
        let dest = parent.join("rescue-controller");
        fs::create_dir_all(&dest).expect("create test output directory");
        let source = fixture_root().join("rescue-controller");
        for entry in fs::read_dir(source).expect("read fixture directory") {
            let entry = entry.expect("fixture directory entry");
            fs::copy(entry.path(), dest.join(entry.file_name())).expect("copy fixture artifact");
        }
        (parent, dest)
    }

    #[test]
    fn digest_validator_accepts_only_lowercase_sha256() {
        assert!(is_sha256(&"a".repeat(64)));
        assert!(!is_sha256(&"A".repeat(64)));
        assert!(!is_sha256(&"g".repeat(64)));
        assert!(!is_sha256(&"a".repeat(63)));
    }

    #[test]
    fn inclusion_probabilities_are_strictly_positive_and_bounded() {
        assert!(probability(0.5));
        assert!(probability(1.0));
        assert!(!probability(0.0));
        assert!(!probability(-0.2));
        assert!(!probability(1.01));
        assert!(!probability(f64::NAN));
    }

    #[test]
    fn inclusion_probability_reconciles_with_frozen_stratum_counts() {
        assert!(close(2.0 / 10.0, 0.2));
        assert!(!close(3.0 / 10.0, 0.2));
    }

    #[test]
    fn valid_fixture_reconciles_and_blind_audit_exposes_a_missed_severe_event() {
        let audit = audit_optional_rescue_contract(&fixture_root());
        assert!(audit.provided);
        assert!(
            audit.findings.is_empty(),
            "unexpected findings: {:?}",
            audit.findings.iter().map(|f| (&f.code, &f.detail)).collect::<Vec<_>>()
        );
        assert!(audit.checks_completed.iter().any(|c| c == "rescue_blinded_sample_and_decision_reconciliation_checked"));

        let results_path = fixture_root().join("rescue-controller").join("blinded-audit-results.jsonl");
        let rows: Vec<AuditResult> = parse_jsonl(
            &fs::read(results_path).expect("read synthetic audit results"),
            MAX_JSONL_ROWS,
        )
        .expect("fixture audit results parse");
        let missed = rows.iter().find(|row| row.window_id == "w-003").expect("missed event fixture exists");
        assert_eq!(missed.severe_event.as_bool(), Some(true));
        assert!(!missed.policy_triggered_reference);
        assert_eq!(missed.physical_integrity_pass.as_bool(), Some(false));
        // Structural integrity can pass while the synthetic policy plainly misses an event.
    }

    #[test]
    fn future_data_and_heldout_label_access_fail_closed() {
        let (parent, run_dir) = copy_fixture_to_temp();
        let path = run_dir.join("decision-ledger.jsonl");
        let raw = fs::read_to_string(&path).expect("read copied decision ledger");
        let changed = raw.replacen("\\"future_data_read\\":false", "\\"future_data_read\\":true", 1);
        assert_ne!(raw, changed);
        fs::write(&path, changed).expect("write manipulated decision ledger");
        let audit = audit_optional_rescue_contract(&parent);
        assert!(audit.findings.iter().any(|f| f.code == "rescue_test_leakage_declared"));
        fs::remove_dir_all(parent).expect("remove temporary fixture");
    }

    #[test]
    fn audit_result_must_bind_exact_sampling_manifest_bytes() {
        let (parent, run_dir) = copy_fixture_to_temp();
        let path = run_dir.join("audit-sampling-manifest.json");
        let mut raw = fs::read(&path).expect("read sampling manifest");
        raw.push(b'\n');
        fs::write(&path, raw).expect("mutate sampling-manifest bytes");
        let audit = audit_optional_rescue_contract(&parent);
        assert!(audit.findings.iter().any(|f| f.code == "rescue_audit_manifest_digest_mismatch"));
        fs::remove_dir_all(parent).expect("remove temporary fixture");
    }

    #[test]
    fn sample_manifest_must_remain_evaluator_sealed() {
        let (parent, run_dir) = copy_fixture_to_temp();
        let path = run_dir.join("audit-sampling-manifest.json");
        let raw = fs::read_to_string(&path).expect("read copied sampling manifest");
        let changed = raw.replace(
            "\\"manifest_visibility\\": \\"evaluator_sealed_until_policy_run_complete\\"",
            "\\"manifest_visibility\\": \\"public_before_run\\"",
        );
        assert_ne!(raw, changed);
        fs::write(&path, changed).expect("write exposed-sample manifest");
        let audit = audit_optional_rescue_contract(&parent);
        assert!(audit.findings.iter().any(|f| f.code == "rescue_manifest_not_sealed"));
        fs::remove_dir_all(parent).expect("remove temporary fixture");
    }

    #[test]
    fn audit_labels_revealed_to_policy_fail_closed() {
        let (parent, run_dir) = copy_fixture_to_temp();
        let path = run_dir.join("blinded-audit-results.jsonl");
        let raw = fs::read_to_string(&path).expect("read copied blinded outcomes");
        let changed = raw.replacen("\\"labels_revealed_to_policy\\":false", "\\"labels_revealed_to_policy\\":true", 1);
        assert_ne!(raw, changed);
        fs::write(&path, changed).expect("write revealed-label audit result");
        let audit = audit_optional_rescue_contract(&parent);
        assert!(audit.findings.iter().any(|f| f.code == "rescue_audit_labels_revealed"));
        fs::remove_dir_all(parent).expect("remove temporary fixture");
    }

    #[test]
    fn unsafe_relative_paths_are_rejected_before_reading() {
        let result = read_regular_file(Path::new("/tmp"), Path::new("../outside.json"));
        assert!(result.unwrap_err().contains("unsafe relative path"));
    }
}
