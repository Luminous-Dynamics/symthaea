//! Read-only evaluator for matched clean/shifted engineering continuations.
//!
//! The evaluator checks trace structure, evidence scope, selective invalidation,
//! retained functionality, active-requirement conformance and longitudinal
//! time spent with obligations lacking current passing evidence. Receipt digests
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
    /// Exact cutoff at which this checkpoint represents agent-visible state.
    pub checkpoint_at: String,
    /// Agent prediction of which obligations the change affected.
    pub predicted_affected_obligation_ids: Vec<String>,
    /// One current receipt per obligation, including retained older receipts.
    pub receipts: Vec<CheckReceipt>,
    pub revalidation_cost_units: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckReceipt {
    pub receipt_id: String,
    pub obligation_id: String,
    pub check_id: String,
    pub outcome: CheckOutcome,
    pub source_epoch: u64,
    /// When the check result was observed in its pinned source snapshot.
    pub observed_at: String,
    /// When the receipt became available to the evaluated agent.
    pub available_at: String,
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
    observed_at: &'a str,
    available_at: &'a str,
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
    pub active_obligation_seconds: u64,
    pub unverified_obligation_seconds: u64,
    pub critical_active_obligation_seconds: u64,
    pub critical_unverified_obligation_seconds: u64,
    pub time_weighted_requirement_conformance: Option<f64>,
    pub critical_time_weighted_conformance: Option<f64>,
    pub recovery_event_count: usize,
    pub max_unverified_gap_seconds: u64,
    pub max_critical_unverified_gap_seconds: u64,
    pub obligation_exposures: Vec<ObligationExposure>,
    pub final_checkpoint_success: bool,
    pub claim_ceiling: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObligationExposure {
    pub obligation_id: String,
    pub critical: bool,
    pub active_seconds: u64,
    pub unverified_seconds: u64,
    pub recovery_events: usize,
    pub max_unverified_gap_seconds: u64,
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
    arm_id: &str,
    receipt: &CheckReceipt,
) -> Result<String, String> {
    let material = ReceiptMaterial {
        schema_version: &scenario.schema_version,
        scenario_id: &scenario.scenario_id,
        repository_full_name: &scenario.repository_full_name,
        arm_id,
        receipt_id: &receipt.receipt_id,
        obligation_id: &receipt.obligation_id,
        check_id: &receipt.check_id,
        outcome: receipt.outcome,
        source_epoch: receipt.source_epoch,
        observed_at: &receipt.observed_at,
        available_at: &receipt.available_at,
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
                && (epoch.change_kind != "none"
                    || !epoch.affected_obligation_ids.is_empty()
                    || epoch.snapshot != scenario.initial_snapshot
                    || epoch.active_obligation_ids != arm.epochs[0].active_obligation_ids)
            {
                return Err(format!("clean arm {} must preserve its initial snapshot and active obligations without a material shift", arm.arm_id));
            }
        }
        if arm.condition == ArmCondition::Shifted
            && !arm.epochs.iter().skip(1).any(|epoch| !epoch.affected_obligation_ids.is_empty())
        {
            return Err(format!("shifted arm {} must include at least one affected obligation", arm.arm_id));
        }

        let mut prior_checkpoint_time: Option<DateTime<FixedOffset>> = None;
        for (index, checkpoint) in arm.checkpoints.iter().enumerate() {
            let epoch = &arm.epochs[index];
            if checkpoint.epoch != epoch.epoch {
                return Err(format!("arm {} checkpoint {} does not match its epoch", arm.arm_id, index));
            }
            let checkpoint_time = parse_time(&checkpoint.checkpoint_at)?;
            let epoch_time = parse_time(&epoch.observed_at)?;
            if checkpoint_time < epoch_time {
                return Err(format!("arm {} checkpoint {} predates its epoch observation", arm.arm_id, index));
            }
            if prior_checkpoint_time.as_ref().is_some_and(|prior| &checkpoint_time <= prior) {
                return Err(format!("arm {} checkpoint times must increase strictly", arm.arm_id));
            }
            if index + 1 < arm.epochs.len() {
                let next_epoch_time = parse_time(&arm.epochs[index + 1].observed_at)?;
                if checkpoint_time > next_epoch_time {
                    return Err(format!("arm {} checkpoint {} occurs after the next workspace intervention", arm.arm_id, index));
                }
            }
            prior_checkpoint_time = Some(checkpoint_time.clone());
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
                let source_time = parse_time(&source.observed_at)?;
                let observed_time = parse_time(&receipt.observed_at)?;
                let available_time = parse_time(&receipt.available_at)?;
                if observed_time < source_time {
                    return Err(format!("receipt {} predates its source epoch", receipt.receipt_id));
                }
                if observed_time > available_time {
                    return Err(format!("receipt {} became available before its observation", receipt.receipt_id));
                }
                if available_time > checkpoint_time {
                    return Err(format!("receipt {} was not available at the checkpoint cutoff", receipt.receipt_id));
                }
                if !valid_hex(&receipt.receipt_digest, &[64])
                    || receipt_digest(scenario, &arm.arm_id, receipt)? != receipt.receipt_digest.to_ascii_lowercase()
                {
                    return Err(format!("receipt {} digest mismatch", receipt.receipt_id));
                }
            }
        }

        let all_receipts = unique_arm_receipts(arm)?;
        for receipt in &all_receipts {
            let available_at = parse_time(&receipt.available_at)?;
            let first_available_checkpoint = arm.checkpoints.iter().position(|checkpoint| {
                parse_time(&checkpoint.checkpoint_at)
                    .is_ok_and(|cutoff| cutoff >= available_at)
            }).ok_or_else(|| format!("receipt {} is not available by any checkpoint", receipt.receipt_id))?;
            let first_seen_checkpoint = arm.checkpoints.iter().position(|checkpoint| {
                checkpoint.receipts.iter().any(|item| item.receipt_id == receipt.receipt_id)
            }).expect("unique receipts were collected from checkpoints");
            if first_seen_checkpoint != first_available_checkpoint {
                return Err(format!(
                    "receipt {} must first appear at the earliest checkpoint where it is available",
                    receipt.receipt_id
                ));
            }
        }
    }

    let clean_arm = scenario.arms.iter()
        .find(|arm| arm.condition == ArmCondition::Clean)
        .expect("scenario validation requires a clean arm");
    let shifted_arm = scenario.arms.iter()
        .find(|arm| arm.condition == ArmCondition::Shifted)
        .expect("scenario validation requires a shifted arm");
    if clean_arm.epochs[0].active_obligation_ids != shifted_arm.epochs[0].active_obligation_ids {
        return Err("matched clean and shifted arms must share the same initial active obligations".to_owned());
    }
    if clean_arm.epochs[0].observed_at != shifted_arm.epochs[0].observed_at
        || clean_arm.checkpoints[0].checkpoint_at != shifted_arm.checkpoints[0].checkpoint_at
    {
        return Err("matched clean and shifted arms must share initial observation and checkpoint cutoffs".to_owned());
    }
    if baseline_receipt_signature(clean_arm) != baseline_receipt_signature(shifted_arm) {
        return Err("matched clean and shifted arms must share the same initial evidence outcomes and scope".to_owned());
    }
    Ok(())
}

fn baseline_receipt_signature(
    arm: &ShiftArm,
) -> BTreeMap<String, (String, CheckOutcome, u64, String, String, String, String, String)> {
    arm.checkpoints[0].receipts.iter().map(|receipt| (
        receipt.obligation_id.clone(),
        (
            receipt.check_id.clone(),
            receipt.outcome,
            receipt.source_epoch,
            receipt.observed_at.clone(),
            receipt.available_at.clone(),
            receipt.revision.clone(),
            receipt.tree_oid.clone(),
            receipt.configuration_world_digest.clone(),
        ),
    )).collect()
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

fn obligation_starts_new_interval(arm: &ShiftArm, epoch_index: usize, obligation_id: &str) -> bool {
    let epoch = &arm.epochs[epoch_index];
    if !epoch.active_obligation_ids.iter().any(|id| id == obligation_id) {
        return false;
    }
    if epoch_index == 0 {
        return true;
    }
    let was_active = arm.epochs[epoch_index - 1].active_obligation_ids
        .iter().any(|id| id == obligation_id);
    !was_active || epoch.affected_obligation_ids.iter().any(|id| id == obligation_id)
}

fn unique_arm_receipts(arm: &ShiftArm) -> Result<Vec<CheckReceipt>, String> {
    let mut by_id: BTreeMap<String, CheckReceipt> = BTreeMap::new();
    for checkpoint in &arm.checkpoints {
        for receipt in &checkpoint.receipts {
            match by_id.get(&receipt.receipt_id) {
                Some(existing) if existing != receipt => {
                    return Err(format!("receipt ID {} changes content between checkpoints", receipt.receipt_id));
                }
                Some(_) => {}
                None => { by_id.insert(receipt.receipt_id.clone(), receipt.clone()); }
            }
        }
    }
    Ok(by_id.into_values().collect())
}

fn obligation_exposure(
    arm: &ShiftArm,
    obligation: &Obligation,
    receipts: &[CheckReceipt],
) -> Result<ObligationExposure, String> {
    let mut result = ObligationExposure {
        obligation_id: obligation.obligation_id.clone(),
        critical: obligation.critical,
        active_seconds: 0,
        unverified_seconds: 0,
        recovery_events: 0,
        max_unverified_gap_seconds: 0,
    };

    for start_epoch in 0..arm.epochs.len() {
        if !obligation_starts_new_interval(arm, start_epoch, &obligation.obligation_id) {
            continue;
        }
        let start_time = parse_time(&arm.epochs[start_epoch].observed_at)?;
        let mut boundary_index = None;
        for index in start_epoch + 1..arm.epochs.len() {
            let epoch = &arm.epochs[index];
            let still_active = epoch.active_obligation_ids.iter()
                .any(|id| id == &obligation.obligation_id);
            let invalidated = epoch.affected_obligation_ids.iter()
                .any(|id| id == &obligation.obligation_id);
            if !still_active || invalidated {
                boundary_index = Some(index);
                break;
            }
        }
        let end_time = match boundary_index {
            Some(index) => parse_time(&arm.epochs[index].observed_at)?,
            None => parse_time(&arm.checkpoints.last().expect("validated checkpoints").checkpoint_at)?,
        };
        let duration = end_time.signed_duration_since(start_time.clone()).num_seconds();
        if duration < 0 {
            return Err(format!("obligation {} has a negative active interval", obligation.obligation_id));
        }
        let duration = duration as u64;
        result.active_seconds += duration;

        let mut transitions: Vec<(&CheckReceipt, DateTime<FixedOffset>)> = receipts.iter()
            .filter(|receipt| {
                receipt.obligation_id == obligation.obligation_id
                    && receipt.source_epoch >= start_epoch as u64
            })
            .map(|receipt| Ok((receipt, parse_time(&receipt.available_at)?)))
            .collect::<Result<Vec<_>, String>>()?;
        transitions.retain(|(_, available)| available >= &start_time && available <= &end_time);
        transitions.sort_by(|left, right| {
            left.1.cmp(&right.1)
                .then_with(|| left.0.receipt_id.cmp(&right.0.receipt_id))
        });

        for pair in transitions.windows(2) {
            if pair[0].1 == pair[1].1 && pair[0].0.outcome != pair[1].0.outcome {
                return Err(format!(
                    "obligation {} has conflicting outcomes at the same availability timestamp",
                    obligation.obligation_id
                ));
            }
        }

        let mut verified = false;
        let mut gap_start = Some(start_time.clone());
        for (receipt, available_at) in transitions {
            match receipt.outcome {
                CheckOutcome::Pass => {
                    if let Some(gap) = gap_start.take() {
                        let gap_seconds = available_at.signed_duration_since(gap).num_seconds().max(0) as u64;
                        result.unverified_seconds += gap_seconds;
                        result.max_unverified_gap_seconds =
                            result.max_unverified_gap_seconds.max(gap_seconds);
                        if gap_seconds > 0 {
                            result.recovery_events += 1;
                        }
                    }
                    verified = true;
                }
                CheckOutcome::Fail | CheckOutcome::Unknown => {
                    if verified {
                        gap_start = Some(available_at);
                        verified = false;
                    }
                }
            }
        }
        if let Some(gap) = gap_start {
            let gap_seconds = end_time.signed_duration_since(gap).num_seconds().max(0) as u64;
            result.unverified_seconds += gap_seconds;
            result.max_unverified_gap_seconds =
                result.max_unverified_gap_seconds.max(gap_seconds);
        }
    }
    Ok(result)
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

    // The first checkpoint establishes baseline state; only later work is
    // counted as revalidation cost.
    let total_revalidation_cost_units: f64 = arm.checkpoints.iter().skip(1)
        .map(|checkpoint| checkpoint.revalidation_cost_units)
        .sum();
    let full_rerun_reference_units: f64 = arm.epochs.iter().skip(1)
        .map(|epoch| epoch.active_obligation_ids.len() as f64)
        .sum();
    let revalidation_savings_fraction = (full_rerun_reference_units > 0.0)
        .then_some(1.0 - total_revalidation_cost_units / full_rerun_reference_units);

    let all_receipts = unique_arm_receipts(arm)?;
    let mut obligation_exposures: Vec<ObligationExposure> = scenario.obligations.iter()
        .map(|obligation| obligation_exposure(arm, obligation, &all_receipts))
        .collect::<Result<Vec<_>, _>>()?;
    obligation_exposures.sort_by(|left, right| left.obligation_id.cmp(&right.obligation_id));
    let active_obligation_seconds: u64 = obligation_exposures.iter()
        .map(|item| item.active_seconds).sum();
    let unverified_obligation_seconds: u64 = obligation_exposures.iter()
        .map(|item| item.unverified_seconds).sum();
    let critical_active_obligation_seconds: u64 = obligation_exposures.iter()
        .filter(|item| item.critical).map(|item| item.active_seconds).sum();
    let critical_unverified_obligation_seconds: u64 = obligation_exposures.iter()
        .filter(|item| item.critical).map(|item| item.unverified_seconds).sum();
    let time_weighted_requirement_conformance = (active_obligation_seconds > 0).then_some(
        (1.0 - unverified_obligation_seconds as f64 / active_obligation_seconds as f64).clamp(0.0, 1.0)
    );
    let critical_time_weighted_conformance = (critical_active_obligation_seconds > 0).then_some(
        (1.0 - critical_unverified_obligation_seconds as f64 / critical_active_obligation_seconds as f64).clamp(0.0, 1.0)
    );
    let recovery_event_count: usize = obligation_exposures.iter()
        .map(|item| item.recovery_events).sum();
    let max_unverified_gap_seconds = obligation_exposures.iter()
        .map(|item| item.max_unverified_gap_seconds).max().unwrap_or(0);
    let max_critical_unverified_gap_seconds = obligation_exposures.iter()
        .filter(|item| item.critical)
        .map(|item| item.max_unverified_gap_seconds).max().unwrap_or(0);

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
        active_obligation_seconds,
        unverified_obligation_seconds,
        critical_active_obligation_seconds,
        critical_unverified_obligation_seconds,
        time_weighted_requirement_conformance,
        critical_time_weighted_conformance,
        recovery_event_count,
        max_unverified_gap_seconds,
        max_critical_unverified_gap_seconds,
        obligation_exposures,
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
    fn matched_arms_must_share_the_same_initial_obligations_and_evidence() {
        let mut scenario = fixture();
        let shifted = scenario.arms.iter_mut()
            .find(|arm| arm.condition == ArmCondition::Shifted).expect("shifted arm");
        shifted.epochs[0].active_obligation_ids.push("audit-requirement".to_owned());
        assert!(evaluate_shift_scenario(&scenario).is_err());
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
        let arm_id = shifted.arm_id.clone();
        let checkpoint = shifted.checkpoints.last_mut().expect("final checkpoint");
        let receipt = checkpoint.receipts.iter_mut()
            .find(|receipt| receipt.obligation_id == "api-contract").expect("api receipt");
        receipt.source_epoch = 0;
        let source = &shifted.epochs[0].snapshot;
        receipt.revision = source.revision.clone();
        receipt.tree_oid = source.tree_oid.clone();
        receipt.configuration_world_digest = source.configuration_world_digest.clone();
        receipt.receipt_digest = receipt_digest(&digest_scenario, &arm_id, receipt).expect("reseal test receipt");
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
        let arm_id = shifted.arm_id.clone();
        let receipt = shifted.checkpoints.last_mut().expect("final checkpoint").receipts.iter_mut()
            .find(|receipt| receipt.obligation_id == "audit-requirement").expect("audit receipt");
        receipt.source_epoch = 0;
        let source = &shifted.epochs[0].snapshot;
        receipt.revision = source.revision.clone();
        receipt.tree_oid = source.tree_oid.clone();
        receipt.configuration_world_digest = source.configuration_world_digest.clone();
        receipt.receipt_digest = receipt_digest(&digest_scenario, &arm_id, receipt).expect("reseal");
        assert!(evaluate_shift_scenario(&scenario).is_err());
    }

    #[test]
    fn receipt_unavailable_at_checkpoint_is_rejected() {
        let mut scenario = fixture();
        let digest_scenario = scenario.clone();
        let shifted = scenario.arms.iter_mut()
            .find(|arm| arm.condition == ArmCondition::Shifted).expect("shifted arm");
        let arm_id = shifted.arm_id.clone();
        let receipt = shifted.checkpoints.last_mut().expect("final checkpoint").receipts.iter_mut()
            .find(|receipt| receipt.obligation_id == "api-contract").expect("api receipt");
        receipt.available_at = "2026-11-15T07:16:00Z".to_owned();
        receipt.receipt_digest = receipt_digest(&digest_scenario, &arm_id, receipt).expect("reseal");
        assert!(evaluate_shift_scenario(&scenario).is_err());
    }

    #[test]
    fn receipt_observation_must_precede_agent_availability() {
        let mut scenario = fixture();
        let digest_scenario = scenario.clone();
        let shifted = scenario.arms.iter_mut()
            .find(|arm| arm.condition == ArmCondition::Shifted).expect("shifted arm");
        let arm_id = shifted.arm_id.clone();
        let receipt = shifted.checkpoints.last_mut().expect("final checkpoint").receipts.iter_mut()
            .find(|receipt| receipt.obligation_id == "api-contract").expect("api receipt");
        receipt.observed_at = "2026-11-15T07:14:00Z".to_owned();
        receipt.receipt_digest = receipt_digest(&digest_scenario, &arm_id, receipt).expect("reseal");
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
        let arm_id = clean.arm_id.clone();
        let mut duplicate = clean.checkpoints[0].receipts[0].clone();
        duplicate.receipt_id = "duplicate-receipt".to_owned();
        duplicate.receipt_digest = receipt_digest(&digest_scenario, &arm_id, &duplicate).expect("reseal");
        clean.checkpoints[0].receipts.push(duplicate);
        assert!(evaluate_shift_scenario(&scenario).is_err());
    }

    #[test]
    fn longitudinal_exposure_matches_hand_derived_fixture() {
        let report = evaluate_shift_scenario(&fixture()).expect("valid fixture");
        let clean = report.arm_scores.iter()
            .find(|score| score.condition == ArmCondition::Clean).expect("clean score");
        let shifted = report.arm_scores.iter()
            .find(|score| score.condition == ArmCondition::Shifted).expect("shifted score");
        assert_eq!(clean.active_obligation_seconds, 1_800);
        assert_eq!(clean.unverified_obligation_seconds, 240);
        assert_eq!(clean.critical_unverified_obligation_seconds, 120);
        assert_eq!(clean.time_weighted_requirement_conformance, Some(1.0 - 240.0 / 1_800.0));
        assert_eq!(clean.critical_time_weighted_conformance, Some(1.0 - 120.0 / 900.0));
        assert_eq!(shifted.active_obligation_seconds, 2_100);
        assert_eq!(shifted.unverified_obligation_seconds, 600);
        assert_eq!(shifted.critical_unverified_obligation_seconds, 480);
        assert_eq!(shifted.critical_time_weighted_conformance, Some(0.6));
        assert_eq!(shifted.max_unverified_gap_seconds, 180);
    }

    #[test]
    fn delayed_critical_recovery_worsens_exposure_even_when_terminal_state_passes() {
        let mut scenario = fixture();
        let digest_scenario = scenario.clone();
        let shifted = scenario.arms.iter_mut()
            .find(|arm| arm.condition == ArmCondition::Shifted).expect("shifted arm");
        let arm_id = shifted.arm_id.clone();
        let receipt = shifted.checkpoints.last_mut().expect("final checkpoint").receipts.iter_mut()
            .find(|receipt| receipt.obligation_id == "api-contract").expect("api receipt");
        receipt.available_at = "2026-11-15T07:14:00Z".to_owned();
        receipt.receipt_digest = receipt_digest(&digest_scenario, &arm_id, receipt).expect("reseal");
        let report = evaluate_shift_scenario(&scenario).expect("valid delayed trace");
        let shifted = report.arm_scores.iter()
            .find(|score| score.condition == ArmCondition::Shifted).expect("shifted score");
        assert!(shifted.final_checkpoint_success);
        assert_eq!(shifted.critical_unverified_obligation_seconds, 540);
        assert_eq!(shifted.max_unverified_gap_seconds, 240);
        assert_eq!(shifted.max_critical_unverified_gap_seconds, 240);
    }

    #[test]
    fn later_failure_reopens_an_unverified_gap_until_recovery() {
        let mut scenario = fixture();
        let digest_scenario = scenario.clone();
        for arm in &mut scenario.arms {
            let arm_id = arm.arm_id.clone();
            let shifted = arm.condition == ArmCondition::Shifted;
            let snapshot = arm.epochs[1].snapshot.clone();
            let active = arm.epochs[1].active_obligation_ids.clone();
            let second_epoch_at = if shifted { "2026-11-15T07:14:00Z" } else { "2026-11-15T07:16:00Z" };
            let third_epoch_at = if shifted { "2026-11-15T07:17:00Z" } else { "2026-11-15T07:18:00Z" };
            arm.epochs.push(StateEpoch {
                epoch: 2,
                observed_at: second_epoch_at.to_owned(),
                snapshot: snapshot.clone(),
                change_kind: "none".to_owned(),
                affected_obligation_ids: Vec::new(),
                active_obligation_ids: active.clone(),
            });
            arm.epochs.push(StateEpoch {
                epoch: 3,
                observed_at: third_epoch_at.to_owned(),
                snapshot,
                change_kind: "none".to_owned(),
                affected_obligation_ids: Vec::new(),
                active_obligation_ids: active,
            });

            if shifted {
                arm.checkpoints[1].checkpoint_at = "2026-11-15T07:13:00Z".to_owned();
                let initial_pass = arm.checkpoints[1].receipts.iter_mut()
                    .find(|receipt| receipt.obligation_id == "api-contract").expect("API pass receipt");
                initial_pass.observed_at = "2026-11-15T07:11:00Z".to_owned();
                initial_pass.available_at = "2026-11-15T07:12:00Z".to_owned();
                initial_pass.receipt_digest =
                    receipt_digest(&digest_scenario, &arm_id, initial_pass).expect("reseal initial pass");

                let mut failure_checkpoint = arm.checkpoints[1].clone();
                failure_checkpoint.epoch = 2;
                failure_checkpoint.checkpoint_at = "2026-11-15T07:16:00Z".to_owned();
                failure_checkpoint.predicted_affected_obligation_ids.clear();
                failure_checkpoint.revalidation_cost_units = 0.0;
                let failure = failure_checkpoint.receipts.iter_mut()
                    .find(|receipt| receipt.obligation_id == "api-contract").expect("API receipt");
                failure.receipt_id = "shifted-epoch2-api-fail".to_owned();
                failure.outcome = CheckOutcome::Fail;
                failure.source_epoch = 2;
                failure.observed_at = "2026-11-15T07:14:30Z".to_owned();
                failure.available_at = "2026-11-15T07:15:00Z".to_owned();
                failure.receipt_digest =
                    receipt_digest(&digest_scenario, &arm_id, failure).expect("reseal failure");

                let mut recovery_checkpoint = failure_checkpoint.clone();
                recovery_checkpoint.epoch = 3;
                recovery_checkpoint.checkpoint_at = "2026-11-15T07:20:00Z".to_owned();
                let recovery = recovery_checkpoint.receipts.iter_mut()
                    .find(|receipt| receipt.obligation_id == "api-contract").expect("API failure receipt");
                recovery.receipt_id = "shifted-epoch3-api-pass".to_owned();
                recovery.outcome = CheckOutcome::Pass;
                recovery.source_epoch = 3;
                recovery.observed_at = "2026-11-15T07:18:00Z".to_owned();
                recovery.available_at = "2026-11-15T07:19:00Z".to_owned();
                recovery.receipt_digest =
                    receipt_digest(&digest_scenario, &arm_id, recovery).expect("reseal recovery");

                arm.checkpoints.push(failure_checkpoint);
                arm.checkpoints.push(recovery_checkpoint);
            } else {
                let mut second_checkpoint = arm.checkpoints[1].clone();
                second_checkpoint.epoch = 2;
                second_checkpoint.checkpoint_at = "2026-11-15T07:17:00Z".to_owned();
                let mut third_checkpoint = second_checkpoint.clone();
                third_checkpoint.epoch = 3;
                third_checkpoint.checkpoint_at = "2026-11-15T07:20:00Z".to_owned();
                arm.checkpoints.push(second_checkpoint);
                arm.checkpoints.push(third_checkpoint);
            }
        }

        let report = evaluate_shift_scenario(&scenario).expect("valid transient-failure trace");
        let shifted = report.arm_scores.iter()
            .find(|score| score.condition == ArmCondition::Shifted).expect("shifted score");
        assert!(shifted.final_checkpoint_success);
        assert_eq!(shifted.critical_unverified_obligation_seconds, 660);
        assert_eq!(shifted.max_critical_unverified_gap_seconds, 240);
        assert_eq!(shifted.recovery_event_count, 5);
    }

    #[test]
    fn receipt_cannot_be_withheld_until_after_its_first_available_checkpoint() {
        let mut scenario = fixture();
        let digest_scenario = scenario.clone();
        let clean = scenario.arms.iter_mut()
            .find(|arm| arm.condition == ArmCondition::Clean).expect("clean arm");
        let arm_id = clean.arm_id.clone();
        let receipt = clean.checkpoints[1].receipts.iter_mut()
            .find(|receipt| receipt.obligation_id == "api-contract").expect("API receipt");
        receipt.receipt_id = "late-disclosed-baseline-api".to_owned();
        receipt.receipt_digest = receipt_digest(&digest_scenario, &arm_id, receipt).expect("reseal");
        assert!(evaluate_shift_scenario(&scenario).is_err());
    }

    #[test]
    fn clean_arm_cannot_drift_repository_or_active_requirements() {
        let mut scenario = fixture();
        let clean = scenario.arms.iter_mut()
            .find(|arm| arm.condition == ArmCondition::Clean).expect("clean arm");
        clean.epochs[1].snapshot.revision = "f".repeat(40);
        assert!(evaluate_shift_scenario(&scenario).is_err());
    }

    #[test]
    fn checkpoint_cannot_occur_after_the_next_intervention() {
        let mut scenario = fixture();
        let shifted = scenario.arms.iter_mut()
            .find(|arm| arm.condition == ArmCondition::Shifted).expect("shifted arm");
        shifted.checkpoints[0].checkpoint_at = "2026-11-15T07:11:00Z".to_owned();
        assert!(evaluate_shift_scenario(&scenario).is_err());
    }

    #[test]
    fn receipt_identity_cannot_change_between_checkpoints() {
        let mut scenario = fixture();
        let digest_scenario = scenario.clone();
        let clean = scenario.arms.iter_mut()
            .find(|arm| arm.condition == ArmCondition::Clean).expect("clean arm");
        let arm_id = clean.arm_id.clone();
        let baseline = clean.checkpoints[0].receipts.iter()
            .find(|receipt| receipt.obligation_id == "api-contract").expect("baseline API receipt").clone();
        let final_receipt = clean.checkpoints[1].receipts.iter_mut()
            .find(|receipt| receipt.obligation_id == "api-contract").expect("final API receipt");
        final_receipt.receipt_id = baseline.receipt_id;
        final_receipt.outcome = CheckOutcome::Fail;
        final_receipt.receipt_digest = receipt_digest(&digest_scenario, &arm_id, final_receipt).expect("reseal");
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
