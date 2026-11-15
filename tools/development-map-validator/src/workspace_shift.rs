//! Read-only evaluator for matched clean/shifted engineering continuations.
//!
//! The evaluator checks trace structure, evidence scope, selective invalidation,
//! retained functionality and active-requirement conformance. Receipt digests
//! provide deterministic integrity checks only; they do not authenticate a
//! producer, source observation, test execution, authority or CI result.

use std::collections::{BTreeMap, HashMap, HashSet};

use chrono::{DateTime, FixedOffset};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const SCHEMA_VERSION: &str = "0.1.0";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShiftScenario {
    pub schema_version: String,
    pub scenario_id: String,
    pub repository_full_name: String,
    pub initial_snapshot: Snapshot,
    pub obligations: Vec<Obligation>,
    pub arms: Vec<ShiftArm>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Snapshot {
    pub revision: String,
    pub tree_oid: String,
    pub configuration_world_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Obligation {
    pub obligation_id: String,
    pub check_id: String,
    pub critical: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArmCondition {
    Clean,
    Shifted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShiftArm {
    pub arm_id: String,
    pub condition: ArmCondition,
    pub epochs: Vec<StateEpoch>,
    pub checkpoints: Vec<AgentCheckpoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateEpoch {
    pub epoch: u64,
    pub observed_at: String,
    pub snapshot: Snapshot,
    /// "initial", "none", or a change-family label such as "dependency_api".
    pub change_kind: String,
    /// Evaluator-side gold impact labels. Production benchmark custody must
    /// keep these outside candidate-readable task artifacts.
    pub affected_obligation_ids: Vec<String>,
    /// Requirements active after this epoch's change.
    pub active_obligation_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentCheckpoint {
    pub epoch: u64,
    /// Agent prediction of which obligations the change affected.
    pub predicted_affected_obligation_ids: Vec<String>,
    /// One current receipt per obligation, including retained older receipts.
    pub receipts: Vec<CheckReceipt>,
    pub revalidation_cost_units: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckReceipt {
    pub receipt_id: String,
    pub obligation_id: String,
    pub check_id: String,
    pub outcome: CheckOutcome,
    pub source_epoch: u64,
    pub revision: String,
    pub tree_oid: String,
    pub configuration_world_digest: String,
    pub receipt_digest: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckOutcome {
    Pass,
    Fail,
    Unknown,
}

#[derive(Serialize)]
struct ReceiptMaterial<'a> {
    schema_version: &'a str,
    scenario_id: &'a str,
    repository_full_name: &'a str,
    arm_id: &'a str,
    receipt_id: &'a str,
    obligation_id: &'a str,
    check_id: &'a str,
    outcome: CheckOutcome,
    source_epoch: u64,
    revision: &'a str,
    tree_oid: &'a str,
    configuration_world_digest: &'a str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmScore {
    pub arm_id: String,
    pub condition: ArmCondition,
    pub retained_functionality: Option<f64>,
    pub active_requirement_conformance: Option<f64>,
    pub impact_precision: Option<f64>,
    pub impact_recall: Option<f64>,
    pub stale_active_receipts: usize,
    pub missing_active_receipts: Vec<String>,
    pub critical_unverified: Vec<String>,
    pub total_revalidation_cost_units: f64,
    pub full_rerun_reference_units: f64,
    pub revalidation_savings_fraction: Option<f64>,
    pub final_checkpoint_success: bool,
    pub claim_ceiling: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShiftBenchmarkReport {
    pub scenario_id: String,
    pub arm_scores: Vec<ArmScore>,
    pub paired_success: bool,
    pub retained_functionality_delta: Option<f64>,
    pub active_requirement_conformance_delta: Option<f64>,
    pub claim_ceiling: String,
}

fn valid_hex(value: &str, allowed_lengths: &[usize]) -> bool {
    allowed_lengths.contains(&value.len()) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn parse_time(value: &str) -> Result<DateTime<FixedOffset>, String> {
    DateTime::parse_from_rfc3339(value)
        .map_err(|_| format!("timestamp is not RFC 3339: {value:?}"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn receipt_digest(
    scenario: &ShiftScenario,
    arm: &ShiftArm,
    receipt: &CheckReceipt,
) -> Result<String, String> {
    let material = ReceiptMaterial {
        schema_version: &scenario.schema_version,
        scenario_id: &scenario.scenario_id,
        repository_full_name: &scenario.repository_full_name,
        arm_id: &arm.arm_id,
        receipt_id: &receipt.receipt_id,
        obligation_id: &receipt.obligation_id,
        check_id: &receipt.check_id,
        outcome: receipt.outcome,
        source_epoch: receipt.source_epoch,
        revision: &receipt.revision,
        tree_oid: &receipt.tree_oid,
        configuration_world_digest: &receipt.configuration_world_digest,
    };
    let bytes = serde_json::to_vec(&material).map_err(|error| error.to_string())?;
    Ok(sha256_hex(&bytes))
}

fn unique_nonempty<'a>(
    values: impl Iterator<Item = &'a String>,
    label: &str,
) -> Result<HashSet<&'a str>, String> {
    let mut found = HashSet::new();
    for value in values {
        if value.trim().is_empty() || !found.insert(value.as_str()) {
            return Err(format!("{label} values must be non-empty and unique"));
        }
    }
    Ok(found)
}

fn validate_snapshot(snapshot: &Snapshot, label: &str) -> Result<(), String> {
    if !valid_hex(&snapshot.revision, &[40, 64]) {
        return Err(format!("{label}: revision must be a 40- or 64-character hexadecimal object ID"));
    }
    if !valid_hex(&snapshot.tree_oid, &[40, 64]) || snapshot.tree_oid.len() != snapshot.revision.len() {
        return Err(format!("{label}: tree_oid must match revision object-ID length"));
    }
    if !valid_hex(&snapshot.configuration_world_digest, &[64]) {
        return Err(format!("{label}: configuration_world_digest must be a SHA-256 digest"));
    }
    Ok(())
}

fn validate_scenario(scenario: &ShiftScenario) -> Result<(), String> {
    if scenario.schema_version != SCHEMA_VERSION {
        return Err(format!("unsupported schema_version {:?}", scenario.schema_version));
    }
    if scenario.scenario_id.trim().is_empty() {
        return Err("scenario_id must not be empty".to_owned());
    }
    let mut repo_parts = scenario.repository_full_name.split('/');
    if repo_parts.next().is_none_or(str::is_empty)
        || repo_parts.next().is_none_or(str::is_empty)
        || repo_parts.next().is_some()
        || scenario.repository_full_name.chars().any(char::is_whitespace)
    {
        return Err("repository_full_name must be owner/repository".to_owned());
    }
    validate_snapshot(&scenario.initial_snapshot, "initial_snapshot")?;

    let mut obligations = HashMap::new();
    for obligation in &scenario.obligations {
        if obligation.obligation_id.trim().is_empty()
            || obligation.check_id.trim().is_empty()
            || obligations.insert(obligation.obligation_id.as_str(), obligation).is_some()
        {
            return Err("obligation IDs and check IDs must be non-empty; obligation IDs must be unique".to_owned());
        }
    }
    if obligations.is_empty() {
        return Err("at least one obligation is required".to_owned());
    }
    if scenario.arms.len() != 2 {
        return Err("exactly two matched arms (clean and shifted) are required".to_owned());
    }
    let conditions: HashSet<_> = scenario.arms.iter().map(|arm| arm.condition).collect();
    if conditions.len() != 2
        || !conditions.contains(&ArmCondition::Clean)
        || !conditions.contains(&ArmCondition::Shifted)
    {
        return Err("one clean arm and one shifted arm are required".to_owned());
    }
    unique_nonempty(scenario.arms.iter().map(|arm| &arm.arm_id), "arm_id")?;

    for arm in &scenario.arms {
        if arm.epochs.len() < 2 || arm.checkpoints.len() != arm.epochs.len() {
            return Err(format!("arm {} must have at least two epochs and exactly one checkpoint per epoch", arm.arm_id));
        }
        let mut prior_time: Option<DateTime<FixedOffset>> = None;
        for (index, epoch) in arm.epochs.iter().enumerate() {
            if epoch.epoch != index as u64 {
                return Err(format!("arm {} epochs must be contiguous from zero", arm.arm_id));
            }
            let observed = parse_time(&epoch.observed_at)?;
            if prior_time.as_ref().is_some_and(|prior| &observed <= prior) {
                return Err(format!("arm {} epoch timestamps must increase strictly", arm.arm_id));
            }
            prior_time = Some(observed);
            validate_snapshot(&epoch.snapshot, &format!("arm {} epoch {}", arm.arm_id, epoch.epoch))?;
            if index == 0 {
                if epoch.change_kind != "initial"
                    || !epoch.affected_obligation_ids.is_empty()
                    || epoch.snapshot != scenario.initial_snapshot
                {
                    return Err(format!("arm {} epoch zero must be the shared initial snapshot", arm.arm_id));
                }
            } else if epoch.change_kind.trim().is_empty() {
                return Err(format!("arm {} epoch {} requires a change_kind", arm.arm_id, epoch.epoch));
            }
            unique_nonempty(epoch.affected_obligation_ids.iter(), "affected obligation")?;
            unique_nonempty(epoch.active_obligation_ids.iter(), "active obligation")?;
            for id in epoch.affected_obligation_ids.iter().chain(epoch.active_obligation_ids.iter()) {
                if !obligations.contains_key(id.as_str()) {
                    return Err(format!("arm {} epoch {} references unknown obligation {id}", arm.arm_id, epoch.epoch));
                }
            }
            if arm.condition == ArmCondition::Clean && index > 0
                && (epoch.change_kind != "none" || !epoch.affected_obligation_ids.is_empty())
            {
                return Err(format!("clean arm {} must not contain a material workspace shift", arm.arm_id));
            }
        }
        if arm.condition == ArmCondition::Shifted
            && !arm.epochs.iter().skip(1).any(|epoch| !epoch.affected_obligation_ids.is_empty())
        {
            return Err(format!("shifted arm {} must include at least one affected obligation", arm.arm_id));
        }

        for (index, checkpoint) in arm.checkpoints.iter().enumerate() {
            let epoch = &arm.epochs[index];
            if checkpoint.epoch != epoch.epoch {
                return Err(format!("arm {} checkpoint {} does not match its epoch", arm.arm_id, index));
            }
            if !checkpoint.revalidation_cost_units.is_finite() || checkpoint.revalidation_cost_units < 0.0 {
                return Err(format!("arm {} checkpoint {} has invalid revalidation cost", arm.arm_id, index));
            }
            unique_nonempty(checkpoint.predicted_affected_obligation_ids.iter(), "predicted affected obligation")?;
            for id in &checkpoint.predicted_affected_obligation_ids {
                if !obligations.contains_key(id.as_str()) {
                    return Err(format!("arm {} predicts unknown obligation {id}", arm.arm_id));
                }
            }
            if index == 0 && !checkpoint.predicted_affected_obligation_ids.is_empty() {
                return Err(format!("arm {} initial checkpoint cannot predict a prior change", arm.arm_id));
            }

            let mut receipt_ids = HashSet::new();
            let mut receipt_obligations = HashSet::new();
            for receipt in &checkpoint.receipts {
                if receipt.receipt_id.trim().is_empty()
                    || !receipt_ids.insert(receipt.receipt_id.as_str())
                    || !receipt_obligations.insert(receipt.obligation_id.as_str())
                {
                    return Err(format!("arm {} checkpoint {} receipt IDs and obligation references must be unique", arm.arm_id, index));
                }
                let obligation = obligations.get(receipt.obligation_id.as_str())
                    .ok_or_else(|| format!("receipt references unknown obligation {}", receipt.obligation_id))?;
                if receipt.check_id != obligation.check_id {
                    return Err(format!("receipt {} uses the wrong check_id", receipt.receipt_id));
                }
                if receipt.source_epoch > checkpoint.epoch {
                    return Err(format!("receipt {} uses future evidence", receipt.receipt_id));
                }
                let source = &arm.epochs[receipt.source_epoch as usize];
                if receipt.revision != source.snapshot.revision
                    || receipt.tree_oid != source.snapshot.tree_oid
                    || receipt.configuration_world_digest != source.snapshot.configuration_world_digest
                {
                    return Err(format!("receipt {} is not bound to its claimed source epoch", receipt.receipt_id));
                }
                if !source.active_obligation_ids.contains(&receipt.obligation_id) {
                    return Err(format!("receipt {} refers to an obligation that was not active at its source epoch", receipt.receipt_id));
                }
                if !valid_hex(&receipt.receipt_digest, &[64])
                    || receipt_digest(scenario, arm, receipt)? != receipt.receipt_digest.to_ascii_lowercase()
                {
                    return Err(format!("receipt {} digest mismatch", receipt.receipt_id));
                }
            }
        }
    }
    Ok(())
}

fn last_affected_epoch(arm: &ShiftArm, obligation_id: &str, through_epoch: u64) -> u64 {
    arm.epochs.iter()
        .take((through_epoch + 1) as usize)
        .filter(|epoch| epoch.affected_obligation_ids.iter().any(|id| id == obligation_id))
        .map(|epoch| epoch.epoch)
        .max()
        .unwrap_or(0)
}

fn latest_receipts(checkpoint: &AgentCheckpoint) -> HashMap<&str, &CheckReceipt> {
    checkpoint.receipts.iter().map(|receipt| (receipt.obligation_id.as_str(), receipt)).collect()
}

fn score_arm(scenario: &ShiftScenario, arm: &ShiftArm) -> Result<ArmScore, String> {
    let initial_receipts = latest_receipts(&arm.checkpoints[0]);
    let mut baseline_passed = HashSet::new();
    for id in &arm.epochs[0].active_obligation_ids {
        if initial_receipts.get(id.as_str()).is_some_and(|receipt| receipt.outcome == CheckOutcome::Pass) {
            baseline_passed.insert(id.as_str());
        }
    }

    let final_epoch = arm.epochs.last().expect("validated non-empty epoch list");
    let final_checkpoint = arm.checkpoints.last().expect("validated checkpoint list");
    let receipts = latest_receipts(final_checkpoint);

    let still_active_baseline: Vec<_> = baseline_passed.iter()
        .filter(|id| final_epoch.active_obligation_ids.iter().any(|active| active.as_str() == **id))
        .copied()
        .collect();
    let mut retained = 0usize;
    for id in &still_active_baseline {
        if let Some(receipt) = receipts.get(id) {
            let last_change = last_affected_epoch(arm, id, final_epoch.epoch);
            if receipt.outcome == CheckOutcome::Pass && receipt.source_epoch >= last_change {
                retained += 1;
            }
        }
    }
    let retained_functionality = (!still_active_baseline.is_empty())
        .then_some(retained as f64 / still_active_baseline.len() as f64);

    let mut passing_active = 0usize;
    let mut stale_active_receipts = 0usize;
    let mut missing_active_receipts = Vec::new();
    let mut critical_unverified = Vec::new();
    for id in &final_epoch.active_obligation_ids {
        let receipt = receipts.get(id.as_str()).copied();
        let fresh_pass = receipt.is_some_and(|item| {
            item.outcome == CheckOutcome::Pass
                && item.source_epoch >= last_affected_epoch(arm, id, final_epoch.epoch)
        });
        if fresh_pass {
            passing_active += 1;
        } else {
            if receipt.is_none() {
                missing_active_receipts.push(id.clone());
            } else if receipt.is_some_and(|item| item.source_epoch < last_affected_epoch(arm, id, final_epoch.epoch)) {
                stale_active_receipts += 1;
            }
            if scenario.obligations.iter().any(|obligation| obligation.obligation_id == *id && obligation.critical) {
                critical_unverified.push(id.clone());
            }
        }
    }
    let active_requirement_conformance = (!final_epoch.active_obligation_ids.is_empty())
        .then_some(passing_active as f64 / final_epoch.active_obligation_ids.len() as f64);

    let mut true_positive = 0usize;
    let mut predicted = 0usize;
    let mut actual = 0usize;
    for (index, checkpoint) in arm.checkpoints.iter().enumerate().skip(1) {
        let actual_ids: HashSet<_> = arm.epochs[index].affected_obligation_ids.iter().map(String::as_str).collect();
        let predicted_ids: HashSet<_> = checkpoint.predicted_affected_obligation_ids.iter().map(String::as_str).collect();
        true_positive += predicted_ids.intersection(&actual_ids).count();
        predicted += predicted_ids.len();
        actual += actual_ids.len();
    }
    let impact_precision = (predicted > 0).then_some(true_positive as f64 / predicted as f64);
    let impact_recall = (actual > 0).then_some(true_positive as f64 / actual as f64);

    let total_revalidation_cost_units: f64 = arm.checkpoints.iter()
        .map(|checkpoint| checkpoint.revalidation_cost_units)
        .sum();
    let full_rerun_reference_units: f64 = arm.epochs.iter().skip(1)
        .map(|epoch| epoch.active_obligation_ids.len() as f64)
        .sum();
    let revalidation_savings_fraction = (full_rerun_reference_units > 0.0)
        .then_some(1.0 - total_revalidation_cost_units / full_rerun_reference_units);

    let retained_ok = retained_functionality.is_none_or(|value| value >= 1.0);
    let conformance_ok = active_requirement_conformance.is_some_and(|value| value >= 1.0);
    let final_checkpoint_success = retained_ok
        && conformance_ok
        && stale_active_receipts == 0
        && missing_active_receipts.is_empty()
        && critical_unverified.is_empty();

    Ok(ArmScore {
        arm_id: arm.arm_id.clone(),
        condition: arm.condition,
        retained_functionality,
        active_requirement_conformance,
        impact_precision,
        impact_recall,
        stale_active_receipts,
        missing_active_receipts,
        critical_unverified,
        total_revalidation_cost_units,
        full_rerun_reference_units,
        revalidation_savings_fraction,
        final_checkpoint_success,
        claim_ceiling: "Deterministic scoring of a supplied synthetic/benchmark trace only. Receipt digests are not signatures; the evaluator does not authenticate test execution, source observations, gold custody, authority, CI, or deployed outcomes.".to_owned(),
    })
}

/// Validate and score a matched clean/shifted workspace-continuation trace.
pub fn evaluate_shift_scenario(scenario: &ShiftScenario) -> Result<ShiftBenchmarkReport, String> {
    validate_scenario(scenario)?;
    let mut scores = Vec::new();
    for arm in &scenario.arms {
        scores.push(score_arm(scenario, arm)?);
    }
    scores.sort_by_key(|score| match score.condition {
        ArmCondition::Clean => 0,
        ArmCondition::Shifted => 1,
    });
    let clean = scores.iter().find(|score| score.condition == ArmCondition::Clean)
        .expect("validated clean arm");
    let shifted = scores.iter().find(|score| score.condition == ArmCondition::Shifted)
        .expect("validated shifted arm");
    let retained_functionality_delta = shifted.retained_functionality.zip(clean.retained_functionality)
        .map(|(shift, base)| shift - base);
    let active_requirement_conformance_delta = shifted.active_requirement_conformance.zip(clean.active_requirement_conformance)
        .map(|(shift, base)| shift - base);
    Ok(ShiftBenchmarkReport {
        scenario_id: scenario.scenario_id.clone(),
        arm_scores: scores,
        paired_success: clean.final_checkpoint_success && shifted.final_checkpoint_success,
        retained_functionality_delta,
        active_requirement_conformance_delta,
        claim_ceiling: "Synthetic/benchmark trace evaluation only. No model capability, causal effect, operational safety, or production outcome is established by structural scoring alone.".to_owned(),
    })
}

pub fn evaluate_shift_scenario_json(input: &str) -> Result<ShiftBenchmarkReport, String> {
    let scenario: ShiftScenario = serde_json::from_str(input)
        .map_err(|error| format!("invalid shift scenario JSON: {error}"))?;
    evaluate_shift_scenario(&scenario)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../../../docs/development-map/fixtures/egddm-shiftbench-v0.1.json");

    fn fixture() -> ShiftScenario {
        serde_json::from_str(FIXTURE).expect("fixture parses")
    }

    #[test]
    fn known_answer_fixture_has_perfect_clean_and_shifted_continuity() {
        let report = evaluate_shift_scenario(&fixture()).expect("valid fixture");
        assert!(report.paired_success);
        assert_eq!(report.retained_functionality_delta, Some(0.0));
        assert_eq!(report.active_requirement_conformance_delta, Some(0.0));
        assert_eq!(report.arm_scores.len(), 2);
        assert!(report.arm_scores.iter().all(|score| score.final_checkpoint_success));
    }

    #[test]
    fn unaffected_evidence_remains_reusable_after_workspace_change() {
        let report = evaluate_shift_scenario(&fixture()).expect("valid fixture");
        let shifted = report.arm_scores.iter()
            .find(|score| score.condition == ArmCondition::Shifted).expect("shifted arm");
        assert_eq!(shifted.retained_functionality, Some(1.0));
        assert_eq!(shifted.stale_active_receipts, 0);
    }

    #[test]
    fn impacted_obligation_requires_post_change_evidence() {
        let mut scenario = fixture();
        let digest_scenario = scenario.clone();
        let shifted = scenario.arms.iter_mut()
            .find(|arm| arm.condition == ArmCondition::Shifted).expect("shifted arm");
        let checkpoint = shifted.checkpoints.last_mut().expect("final checkpoint");
        let receipt = checkpoint.receipts.iter_mut()
            .find(|receipt| receipt.obligation_id == "api-contract").expect("api receipt");
        receipt.source_epoch = 0;
        let source = &shifted.epochs[0].snapshot;
        receipt.revision = source.revision.clone();
        receipt.tree_oid = source.tree_oid.clone();
        receipt.configuration_world_digest = source.configuration_world_digest.clone();
        receipt.receipt_digest = receipt_digest(&digest_scenario, shifted, receipt).expect("reseal test receipt");
        let report = evaluate_shift_scenario(&scenario).expect("trace remains structurally valid");
        let shifted_score = report.arm_scores.iter()
            .find(|score| score.condition == ArmCondition::Shifted).expect("shifted score");
        assert_eq!(shifted_score.stale_active_receipts, 1);
        assert!(!shifted_score.final_checkpoint_success);
    }

    #[test]
    fn new_active_requirement_is_counted_in_conformance() {
        let mut scenario = fixture();
        let shifted = scenario.arms.iter_mut()
            .find(|arm| arm.condition == ArmCondition::Shifted).expect("shifted arm");
        shifted.checkpoints.last_mut().expect("final checkpoint").receipts
            .retain(|receipt| receipt.obligation_id != "audit-requirement");
        let report = evaluate_shift_scenario(&scenario).expect("missing receipt is scoreable");
        let shifted_score = report.arm_scores.iter()
            .find(|score| score.condition == ArmCondition::Shifted).expect("shifted score");
        assert_eq!(shifted_score.active_requirement_conformance, Some(2.0 / 3.0));
        assert_eq!(shifted_score.missing_active_receipts, vec!["audit-requirement"]);
        assert!(!shifted_score.final_checkpoint_success);
    }

    #[test]
    fn new_obligation_cannot_reuse_a_pre_introduction_receipt() {
        let mut scenario = fixture();
        let digest_scenario = scenario.clone();
        let shifted = scenario.arms.iter_mut()
            .find(|arm| arm.condition == ArmCondition::Shifted).expect("shifted arm");
        let receipt = shifted.checkpoints.last_mut().expect("final checkpoint").receipts.iter_mut()
            .find(|receipt| receipt.obligation_id == "audit-requirement").expect("audit receipt");
        receipt.source_epoch = 0;
        let source = &shifted.epochs[0].snapshot;
        receipt.revision = source.revision.clone();
        receipt.tree_oid = source.tree_oid.clone();
        receipt.configuration_world_digest = source.configuration_world_digest.clone();
        receipt.receipt_digest = receipt_digest(&digest_scenario, shifted, receipt).expect("reseal");
        assert!(evaluate_shift_scenario(&scenario).is_err());
    }

    #[test]
    fn future_evidence_is_rejected() {
        let mut scenario = fixture();
        let shifted = scenario.arms.iter_mut()
            .find(|arm| arm.condition == ArmCondition::Shifted).expect("shifted arm");
        let checkpoint = shifted.checkpoints.first_mut().expect("initial checkpoint");
        let receipt = checkpoint.receipts.first_mut().expect("receipt");
        receipt.source_epoch = 1;
        assert!(evaluate_shift_scenario(&scenario).is_err());
    }

    #[test]
    fn receipt_from_wrong_snapshot_is_rejected() {
        let mut scenario = fixture();
        let clean = scenario.arms.iter_mut()
            .find(|arm| arm.condition == ArmCondition::Clean).expect("clean arm");
        let receipt = clean.checkpoints[0].receipts.first_mut().expect("receipt");
        receipt.tree_oid = "f".repeat(40);
        assert!(evaluate_shift_scenario(&scenario).is_err());
    }

    #[test]
    fn duplicate_obligation_receipts_are_rejected() {
        let mut scenario = fixture();
        let digest_scenario = scenario.clone();
        let clean = scenario.arms.iter_mut()
            .find(|arm| arm.condition == ArmCondition::Clean).expect("clean arm");
        let mut duplicate = clean.checkpoints[0].receipts[0].clone();
        duplicate.receipt_id = "duplicate-receipt".to_owned();
        duplicate.receipt_digest = receipt_digest(&digest_scenario, clean, &duplicate).expect("reseal");
        clean.checkpoints[0].receipts.push(duplicate);
        assert!(evaluate_shift_scenario(&scenario).is_err());
    }

    #[test]
    fn shifted_arm_must_contain_a_real_affected_scope() {
        let mut scenario = fixture();
        let shifted = scenario.arms.iter_mut()
            .find(|arm| arm.condition == ArmCondition::Shifted).expect("shifted arm");
        shifted.epochs[1].affected_obligation_ids.clear();
        assert!(evaluate_shift_scenario(&scenario).is_err());
    }
}
