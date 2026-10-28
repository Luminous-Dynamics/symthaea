//! Validate replay receipts for cognitive-loop counterfactual experiments.
//!
//! This is a receipt-integrity gate, not the replay engine and not proof that a runner is honest.
//! It checks that the submitted receipt is internally consistent, causal, and comparable across arms.
use anyhow::{ensure, Context};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::{BTreeMap, BTreeSet}, fs, path::Path};

const REQUIRED_ARMS: [&str; 5] = [
    "no-adapt",
    "detector-triggered",
    "detector-benefit-gated",
    "staged-verify-rollback",
    "oracle-timed",
];

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayReceipt {
    pub schema_version: u32,
    pub run_id: String,
    /// "synthetic-not-measured" for contract fixtures, "replay" for an executed replay.
    pub run_kind: String,
    pub dataset_id: String,
    pub dataset_artifact_sha256: String,
    pub transformation_config_sha256: String,
    pub stream_id: String,
    pub stream_length_ticks: u64,
    pub replay_driver_repository_head_sha: String,
    pub replay_driver_configuration_digest: String,
    pub environment_seed: u64,
    /// SHA-256 of serde_json's canonical serialization of the typed events vector.
    pub event_ledger_sha256: String,
    pub events: Vec<ExogenousEvent>,
    pub arms: Vec<ReplayArm>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExogenousEvent {
    pub sequence_index: u64,
    pub event_id: String,
    pub observed_at_tick: u64,
    pub available_at_tick: u64,
    pub payload_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DeliveryRecord {
    pub event_id: String,
    pub delivered_at_tick: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayArm {
    pub arm_id: String,
    pub repository_head_sha: String,
    pub configuration_digest: String,
    pub policy_seed: u64,
    /// An execution receipt declaration, not an independent sandbox proof.
    pub oracle_channel_access: bool,
    pub deliveries: Vec<DeliveryRecord>,
    pub decisions: Vec<ReplayDecision>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayDecision {
    pub decision_tick: u64,
    pub observed_event_ids: Vec<String>,
    /// One of no-op, abstain, stage, verify, commit, rollback.
    pub action: String,
    pub state_before_sha256: String,
    pub state_after_sha256: String,
    pub snapshot_sha256: Option<String>,
    pub candidate_state_sha256: Option<String>,
    /// Digest of the evidence artifact used to authorize a verify decision.
    pub verification_evidence_sha256: Option<String>,
    /// Realized per-cycle task loss attached by the replay driver, for post-hoc scoring.
    #[serde(default)]
    pub task_loss: Option<f64>,
    /// Realized per-cycle adaptation cost, in the registered cost unit.
    #[serde(default)]
    pub adaptation_cost_units: Option<f64>,
    /// Earliest tick when this task-loss outcome was actually available to the evaluator.
    #[serde(default)]
    pub task_loss_available_at_tick: Option<u64>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ArmValidationSummary {
    pub arm_id: String,
    pub decision_count: usize,
    pub delivered_event_count: usize,
    pub oracle_channel_access: bool,
    pub commits: usize,
    pub rollbacks: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReplayReceiptValidation {
    pub metric_version: String,
    pub canonical_receipt_sha256: String,
    /// SHA-256 of this report with result_sha256 set to null.
    pub result_sha256: Option<String>,
    pub run_id: String,
    pub run_kind: String,
    pub dataset_id: String,
    pub stream_id: String,
    pub event_ledger_sha256: String,
    pub event_count: usize,
    pub stream_length_ticks: u64,
    pub arm_count: usize,
    pub total_decision_count: usize,
    pub arm_summaries: Vec<ArmValidationSummary>,
    pub interpretation_guard: String,
}

#[derive(Debug, Clone)]
struct StagedCandidate {
    snapshot_sha256: String,
    candidate_sha256: String,
    verified: bool,
}

pub fn evaluate_file(path: &Path) -> anyhow::Result<ReplayReceiptValidation> {
    let bytes = fs::read(path)
        .with_context(|| format!("reading replay receipt {}", path.display()))?;
    let receipt: ReplayReceipt = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing replay receipt {}", path.display()))?;
    validate_receipt(receipt)
}

pub fn validate_receipt(receipt: ReplayReceipt) -> anyhow::Result<ReplayReceiptValidation> {
    ensure!(receipt.schema_version == 2, "replay receipt schema_version must be 2");
    nonempty(&receipt.run_id, "run_id")?;
    ensure!(
        ["synthetic-not-measured", "replay"].contains(&receipt.run_kind.as_str()),
        "run_kind must be synthetic-not-measured or replay"
    );
    nonempty(&receipt.dataset_id, "dataset_id")?;
    nonempty(&receipt.stream_id, "stream_id")?;
    validate_digest(&receipt.dataset_artifact_sha256, "dataset_artifact_sha256")?;
    validate_digest(&receipt.transformation_config_sha256, "transformation_config_sha256")?;
    validate_commit_sha(&receipt.replay_driver_repository_head_sha, "replay_driver_repository_head_sha")?;
    validate_digest(&receipt.replay_driver_configuration_digest, "replay_driver_configuration_digest")?;
    ensure!(receipt.stream_length_ticks > 0, "stream_length_ticks must be positive");
    ensure!(
        receipt.events.len() as u64 == receipt.stream_length_ticks,
        "event ledger must contain exactly one source event per registered stream tick"
    );
    ensure!(
        receipt.arms.len() == REQUIRED_ARMS.len(),
        "receipt must contain exactly five required policy arms"
    );

    let computed_ledger_sha256 = sha256(&serde_json::to_vec(&receipt.events)?);
    ensure!(
        computed_ledger_sha256 == receipt.event_ledger_sha256,
        "event_ledger_sha256 does not match the canonical event ledger"
    );

    let mut event_by_id = BTreeMap::new();
    for (index, event) in receipt.events.iter().enumerate() {
        ensure!(event.sequence_index == index as u64,
            "event sequence_index must be contiguous and match ledger order");
        nonempty(&event.event_id, "event_id")?;
        ensure!(
            event.observed_at_tick == event.sequence_index,
            "observed_at_tick must match the canonical source tick"
        );
        ensure!(
            event.available_at_tick >= event.observed_at_tick,
            "event cannot be available before it is observed"
        );
        ensure!(
            event.available_at_tick < receipt.stream_length_ticks,
            "event availability tick is outside the registered stream"
        );
        validate_digest(&event.payload_sha256, "event payload_sha256")?;
        ensure!(
            event_by_id.insert(event.event_id.as_str(), event).is_none(),
            "event_id values must be unique"
        );
    }

    let mut delivery_events: Vec<&ExogenousEvent> = receipt.events.iter().collect();
    delivery_events.sort_by_key(|event| (event.available_at_tick, event.sequence_index));
    let expected_deliveries: Vec<DeliveryRecord> = delivery_events.iter()
        .map(|event| DeliveryRecord {
            event_id: event.event_id.clone(),
            delivered_at_tick: event.available_at_tick,
        })
        .collect();

    let arm_ids: BTreeSet<&str> = receipt.arms.iter().map(|arm| arm.arm_id.as_str()).collect();
    ensure!(arm_ids.len() == REQUIRED_ARMS.len(), "arm_id values must be unique");
    for required in REQUIRED_ARMS {
        ensure!(arm_ids.contains(required), "missing required replay arm: {required}");
    }

    let mut summaries = Vec::with_capacity(receipt.arms.len());
    let mut reference_deliveries: Option<&[DeliveryRecord]> = None;
    let mut total_decisions = 0usize;
    for arm in &receipt.arms {
        nonempty(&arm.arm_id, "arm_id")?;
        validate_commit_sha(&arm.repository_head_sha, "arm.repository_head_sha")?;
        validate_digest(&arm.configuration_digest, "arm.configuration_digest")?;
        ensure!(
            arm.oracle_channel_access == (arm.arm_id == "oracle-timed"),
            "only oracle-timed may declare oracle_channel_access=true; all deployable arms must be isolated"
        );
        ensure!(
            arm.deliveries == expected_deliveries,
            "arm {} delivery ledger differs from the replay driver's canonical availability schedule",
            arm.arm_id
        );
        if let Some(reference) = reference_deliveries {
            ensure!(
                arm.deliveries.as_slice() == reference,
                "all policy arms must receive the same exogenous events at the same ticks"
            );
        } else {
            reference_deliveries = Some(arm.deliveries.as_slice());
        }
        let summary = validate_arm_decisions(arm, &event_by_id, receipt.stream_length_ticks)?;
        for decision in &arm.decisions {
            if let Some(task_loss) = decision.task_loss {
                ensure!(task_loss.is_finite() && task_loss >= 0.0,
                    "arm {} task_loss must be finite and non-negative", arm.arm_id);
            }
            if let Some(adaptation_cost) = decision.adaptation_cost_units {
                ensure!(adaptation_cost.is_finite() && adaptation_cost >= 0.0,
                    "arm {} adaptation_cost_units must be finite and non-negative", arm.arm_id);
            }
            if let Some(available_at) = decision.task_loss_available_at_tick {
                ensure!(available_at >= decision.decision_tick,
                    "arm {} task_loss_available_at_tick cannot precede decision_tick at tick {}",
                    arm.arm_id, decision.decision_tick);
            }
            if receipt.run_kind == "replay" {
                ensure!(decision.task_loss.is_some(),
                    "replay arm {} is missing task_loss at tick {}", arm.arm_id, decision.decision_tick);
                ensure!(decision.adaptation_cost_units.is_some(),
                    "replay arm {} is missing adaptation_cost_units at tick {}", arm.arm_id, decision.decision_tick);
                ensure!(decision.task_loss_available_at_tick.is_some(),
                    "replay arm {} is missing task_loss_available_at_tick at tick {}", arm.arm_id, decision.decision_tick);
            }
        }
        total_decisions = total_decisions.checked_add(summary.decision_count)
            .ok_or_else(|| anyhow::anyhow!("total decision count overflow"))?;
        summaries.push(summary);
    }

    let canonical_receipt_sha256 = sha256(&serde_json::to_vec(&receipt)?);
    let mut report = ReplayReceiptValidation {
        metric_version: "cognitive-replay-receipt-v2".to_owned(),
        canonical_receipt_sha256,
        result_sha256: None,
        run_id: receipt.run_id,
        run_kind: receipt.run_kind,
        dataset_id: receipt.dataset_id,
        stream_id: receipt.stream_id,
        event_ledger_sha256: receipt.event_ledger_sha256,
        event_count: receipt.events.len(),
        stream_length_ticks: receipt.stream_length_ticks,
        arm_count: summaries.len(),
        total_decision_count: total_decisions,
        arm_summaries: summaries,
        interpretation_guard: "Receipt integrity only: this validates declared event parity, timing, access flags, and state-transition records. It does not prove the driver executed honestly, that the oracle channel was sandboxed, or that reported task outcomes are valid. Synthetic receipts are not benchmark evidence.".to_owned(),
    };
    report.result_sha256 = Some(report_digest(&report)?);
    Ok(report)
}

fn validate_arm_decisions(
    arm: &ReplayArm,
    event_by_id: &BTreeMap<&str, &ExogenousEvent>,
    stream_length_ticks: u64,
) -> anyhow::Result<ArmValidationSummary> {
    ensure!(
        arm.decisions.len() as u64 == stream_length_ticks,
        "arm {} must have exactly one decision record per stream tick",
        arm.arm_id
    );
    let mut staged: Option<StagedCandidate> = None;
    let mut last_committed: Option<StagedCandidate> = None;
    let mut commits = 0usize;
    let mut rollbacks = 0usize;
    let mut previous_after: Option<&str> = None;

    for (index, decision) in arm.decisions.iter().enumerate() {
        ensure!(decision.decision_tick == index as u64,
            "arm {} decision ticks must be contiguous and ordered",
            arm.arm_id);
        validate_digest(&decision.state_before_sha256, "state_before_sha256")?;
        validate_digest(&decision.state_after_sha256, "state_after_sha256")?;
        if let Some(previous) = previous_after {
            ensure!(
                decision.state_before_sha256 == previous,
                "arm {} state digest chain breaks before tick {}",
                arm.arm_id, decision.decision_tick
            );
        }
        previous_after = Some(decision.state_after_sha256.as_str());

        let mut observed_ids = BTreeSet::new();
        for event_id in &decision.observed_event_ids {
            ensure!(observed_ids.insert(event_id.as_str()),
                "arm {} repeats an event ID in the decision information set",
                arm.arm_id);
            let event = event_by_id.get(event_id.as_str())
                .ok_or_else(|| anyhow::anyhow!("arm {} references unknown event {}", arm.arm_id, event_id))?;
            ensure!(
                event.available_at_tick <= decision.decision_tick,
                "arm {} uses future/unavailable event {} at decision tick {}",
                arm.arm_id, event_id, decision.decision_tick
            );
            ensure!(
                arm.deliveries.iter().any(|delivery|
                    delivery.event_id.as_str() == event_id.as_str()
                        && delivery.delivered_at_tick <= decision.decision_tick),
                "arm {} uses event {} before it was delivered",
                arm.arm_id, event_id
            );
        }

        if arm.arm_id == "no-adapt" {
            ensure!(
                decision.action == "no-op" || decision.action == "abstain",
                "no-adapt arm cannot execute adaptation action {}",
                decision.action
            );
        }

        if decision.action == "verify" {
            required_digest(
                decision.verification_evidence_sha256.as_deref(),
                "verify.verification_evidence_sha256",
            )?;
            ensure!(
                !decision.observed_event_ids.is_empty(),
                "verification must reference at least one observed event"
            );
        } else {
            ensure!(
                decision.verification_evidence_sha256.is_none(),
                "verification_evidence_sha256 is only valid on a verify action"
            );
        }

        match decision.action.as_str() {
            "no-op" | "abstain" => {
                ensure!(
                    decision.state_before_sha256 == decision.state_after_sha256,
                    "{} action cannot change state",
                    decision.action
                );
                ensure!(
                    decision.snapshot_sha256.is_none() && decision.candidate_state_sha256.is_none(),
                    "{} action must not carry adaptation transaction hashes",
                    decision.action
                );
            }
            "stage" => {
                ensure!(staged.is_none(), "arm {} stages a second candidate before resolving the first", arm.arm_id);
                let snapshot = required_digest(decision.snapshot_sha256.as_deref(), "stage.snapshot_sha256")?;
                let candidate = required_digest(decision.candidate_state_sha256.as_deref(), "stage.candidate_state_sha256")?;
                ensure!(snapshot == decision.state_before_sha256,
                    "stage snapshot must match active pre-adaptation state");
                ensure!(decision.state_after_sha256 == decision.state_before_sha256,
                    "staging must not silently replace the active state");
                staged = Some(StagedCandidate {
                    snapshot_sha256: snapshot.to_owned(),
                    candidate_sha256: candidate.to_owned(),
                    verified: false,
                });
            }
            "verify" => {
                let active = staged.as_mut()
                    .ok_or_else(|| anyhow::anyhow!("verify requires a staged candidate"))?;
                ensure_transaction_matches(decision, active)?;
                ensure!(
                    decision.state_before_sha256 == active.snapshot_sha256
                        && decision.state_after_sha256 == active.snapshot_sha256,
                    "verification must leave the active state unchanged"
                );
                active.verified = true;
            }
            "commit" => {
                let active = staged.take()
                    .ok_or_else(|| anyhow::anyhow!("commit requires a staged candidate"))?;
                ensure_transaction_matches(decision, &active)?;
                ensure!(active.verified, "candidate must pass verification before commit");
                ensure!(
                    decision.state_before_sha256 == active.snapshot_sha256
                        && decision.state_after_sha256 == active.candidate_sha256,
                    "commit must transition exactly from snapshot to verified candidate"
                );
                last_committed = Some(active);
                commits += 1;
            }
            "rollback" => {
                let snapshot = required_digest(decision.snapshot_sha256.as_deref(), "rollback.snapshot_sha256")?;
                let candidate = required_digest(decision.candidate_state_sha256.as_deref(), "rollback.candidate_state_sha256")?;
                if let Some(active) = staged.take() {
                    ensure_transaction_matches(decision, &active)?;
                    ensure!(
                        decision.state_before_sha256 == active.snapshot_sha256
                            && decision.state_after_sha256 == active.snapshot_sha256,
                        "pre-commit rollback must preserve/restore the original active state"
                    );
                } else if let Some(committed) = last_committed.take() {
                    ensure!(
                        snapshot == committed.snapshot_sha256 && candidate == committed.candidate_sha256,
                        "post-commit rollback must reference the exact committed snapshot/candidate pair"
                    );
                    ensure!(
                        decision.state_before_sha256 == committed.candidate_sha256
                            && decision.state_after_sha256 == committed.snapshot_sha256,
                        "post-commit rollback must restore the exact prior snapshot"
                    );
                } else {
                    anyhow::bail!("rollback has no matching staged or committed adaptation transaction");
                }
                rollbacks += 1;
            }
            other => anyhow::bail!("unknown action {other}; expected no-op, abstain, stage, verify, commit, or rollback"),
        }
    }
    ensure!(staged.is_none(), "arm {} ends with an unresolved staged candidate", arm.arm_id);
    Ok(ArmValidationSummary {
        arm_id: arm.arm_id.clone(),
        decision_count: arm.decisions.len(),
        delivered_event_count: arm.deliveries.len(),
        oracle_channel_access: arm.oracle_channel_access,
        commits,
        rollbacks,
    })
}

fn ensure_transaction_matches(decision: &ReplayDecision, transaction: &StagedCandidate) -> anyhow::Result<()> {
    ensure!(
        decision.snapshot_sha256.as_deref() == Some(transaction.snapshot_sha256.as_str()),
        "adaptation transaction snapshot digest changed between stages"
    );
    ensure!(
        decision.candidate_state_sha256.as_deref() == Some(transaction.candidate_sha256.as_str()),
        "adaptation candidate digest changed between stages"
    );
    Ok(())
}

fn required_digest<'a>(value: Option<&'a str>, name: &str) -> anyhow::Result<&'a str> {
    let value = value.ok_or_else(|| anyhow::anyhow!("{name} is required"))?;
    validate_digest(value, name)?;
    Ok(value)
}

fn nonempty(value: &str, name: &str) -> anyhow::Result<()> {
    ensure!(!value.trim().is_empty(), "{name} must not be empty");
    Ok(())
}

fn validate_commit_sha(value: &str, name: &str) -> anyhow::Result<()> {
    ensure!(
        value.len() == 40 && value.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "{name} must be a lowercase 40-character commit SHA"
    );
    Ok(())
}

fn validate_digest(value: &str, name: &str) -> anyhow::Result<()> {
    ensure!(
        value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "{name} must be a lowercase 64-character SHA-256 digest"
    );
    Ok(())
}

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        output.push_str(&format!("{byte:02x}"));
    }
    output
}

fn report_digest(report: &ReplayReceiptValidation) -> anyhow::Result<String> {
    let mut value = serde_json::to_value(report)?;
    value["result_sha256"] = serde_json::Value::Null;
    Ok(sha256(&serde_json::to_vec(&value)?))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn digest(c: char) -> String { c.to_string().repeat(64) }
    fn commit(c: char) -> String { c.to_string().repeat(40) }

    pub(crate) fn valid_receipt() -> ReplayReceipt {
        let events: Vec<ExogenousEvent> = (0..5).map(|index| ExogenousEvent {
            sequence_index: index,
            event_id: format!("e{index}"),
            observed_at_tick: index,
            available_at_tick: index,
            payload_sha256: digest('a'),
        }).collect();
        let event_ledger_sha256 = sha256(&serde_json::to_vec(&events).unwrap());
        let initial = digest('1');
        let candidate = digest('2');
        let mut arms = Vec::new();
        for (arm_id, oracle_channel_access) in [
            ("no-adapt", false),
            ("detector-triggered", false),
            ("detector-benefit-gated", false),
            ("staged-verify-rollback", false),
            ("oracle-timed", true),
        ] {
            let mut state = initial.clone();
            let mut decisions: Vec<ReplayDecision> = Vec::new();
            for tick in 0..5_u64 {
                let action = if arm_id == "staged-verify-rollback" {
                    match tick {
                        1 => "stage",
                        2 => "verify",
                        3 => "commit",
                        4 => "rollback",
                        _ => "no-op",
                    }
                } else { "no-op" };
                let (snapshot_sha256, candidate_state_sha256, after) = match action {
                    "stage" | "verify" => (Some(initial.clone()), Some(candidate.clone()), state.clone()),
                    "commit" => { state = candidate.clone(); (Some(initial.clone()), Some(candidate.clone()), state.clone()) },
                    "rollback" => { state = initial.clone(); (Some(initial.clone()), Some(candidate.clone()), state.clone()) },
                    _ => (None, None, state.clone()),
                };
                decisions.push(ReplayDecision {
                    decision_tick: tick,
                    observed_event_ids: (0..=tick).map(|i| format!("e{i}")).collect(),
                    action: action.to_owned(),
                    state_before_sha256: if action == "commit" { initial.clone() } else if action == "rollback" { candidate.clone() } else if tick == 0 { initial.clone() } else {
                        match decisions.last() { Some(previous) => previous.state_after_sha256.clone(), None => initial.clone() }
                    },
                    state_after_sha256: after,
                    snapshot_sha256,
                    candidate_state_sha256,
                    verification_evidence_sha256: if action == "verify" { Some(digest('3')) } else { None },
                    task_loss: None,
                    adaptation_cost_units: None,
                    task_loss_available_at_tick: None,
                });
            }
            arms.push(ReplayArm {
                arm_id: arm_id.to_owned(),
                repository_head_sha: commit('c'),
                configuration_digest: digest('d'),
                policy_seed: 100 + arms.len() as u64,
                oracle_channel_access,
                deliveries: events.iter().map(|event| DeliveryRecord {
                    event_id: event.event_id.clone(), delivered_at_tick: event.available_at_tick,
                }).collect(),
                decisions,
            });
        }
        ReplayReceipt {
            schema_version: 2,
            run_id: "synthetic-replay-receipt-not-measured".to_owned(),
            run_kind: "synthetic-not-measured".to_owned(),
            dataset_id: "synthetic-dataset-not-measured".to_owned(),
            dataset_artifact_sha256: digest('e'),
            transformation_config_sha256: digest('f'),
            stream_id: "synthetic-stream-not-measured".to_owned(),
            stream_length_ticks: 5,
            replay_driver_repository_head_sha: commit('a'),
            replay_driver_configuration_digest: digest('b'),
            environment_seed: 41,
            event_ledger_sha256,
            events,
            arms,
        }
    }

    #[test]
    fn accepts_well_formed_replay_receipt_and_reports_no_performance_claim() {
        let result = validate_receipt(valid_receipt()).expect("receipt validates");
        assert_eq!(result.arm_count, 5);
        assert_eq!(result.event_count, 5);
        assert_eq!(result.total_decision_count, 25);
        assert_eq!(result.result_sha256.as_ref().map(String::len), Some(64));
        assert!(result.interpretation_guard.contains("does not prove"));
    }

    #[test]
    fn executed_replay_requires_per_cycle_task_loss_and_cost() {
        let mut receipt = valid_receipt();
        receipt.run_kind = "replay".to_owned();
        let error = validate_receipt(receipt).expect_err("executed replay cannot omit outcome records");
        assert!(error.to_string().contains("missing task_loss at tick 0"));
    }

    #[test]
    fn accepts_delayed_outcomes_with_out_of_order_availability() {
        let mut receipt = valid_receipt();
        receipt.events[0].available_at_tick = 2;
        receipt.events[1].available_at_tick = 1;
        receipt.events[2].available_at_tick = 2;
        receipt.event_ledger_sha256 = sha256(&serde_json::to_vec(&receipt.events).unwrap());

        let mut delivery_events: Vec<&ExogenousEvent> = receipt.events.iter().collect();
        delivery_events.sort_by_key(|event| (event.available_at_tick, event.sequence_index));
        let expected_deliveries: Vec<DeliveryRecord> = delivery_events.iter().map(|event| DeliveryRecord {
            event_id: event.event_id.clone(),
            delivered_at_tick: event.available_at_tick,
        }).collect();
        let visible_by_tick: Vec<Vec<String>> = (0..receipt.stream_length_ticks).map(|tick| {
            receipt.events.iter()
                .filter(|event| event.available_at_tick <= tick)
                .map(|event| event.event_id.clone())
                .collect()
        }).collect();
        for arm in &mut receipt.arms {
            arm.deliveries = expected_deliveries.clone();
            for decision in &mut arm.decisions {
                decision.observed_event_ids = visible_by_tick[decision.decision_tick as usize].clone();
            }
        }

        let result = validate_receipt(receipt).expect("delayed, timestamp-correct receipt validates");
        assert_eq!(result.event_count, 5);
    }

    #[test]
    fn rejects_adaptation_action_from_no_adapt_control() {
        let mut receipt = valid_receipt();
        let decision = &mut receipt.arms[0].decisions[1];
        decision.action = "stage".to_owned();
        decision.snapshot_sha256 = Some(digest('1'));
        decision.candidate_state_sha256 = Some(digest('2'));
        assert!(validate_receipt(receipt).unwrap_err().to_string().contains("no-adapt arm cannot execute adaptation action"));
    }

    #[test]
    fn rejects_future_information_even_when_present_in_the_ledger() {
        let mut receipt = valid_receipt();
        receipt.arms[0].decisions[0].observed_event_ids.push("e1".to_owned());
        assert!(validate_receipt(receipt).unwrap_err().to_string().contains("future/unavailable"));
    }

    #[test]
    fn rejects_event_ledger_digest_tampering() {
        let mut receipt = valid_receipt();
        receipt.events[1].payload_sha256 = digest('9');
        assert!(validate_receipt(receipt).unwrap_err().to_string().contains("event_ledger_sha256"));
    }

    #[test]
    fn rejects_delivery_timing_or_event_mutation_in_one_arm() {
        let mut receipt = valid_receipt();
        receipt.arms[1].deliveries[2].delivered_at_tick = 0;
        assert!(validate_receipt(receipt).unwrap_err().to_string().contains("delivery ledger"));
    }

    #[test]
    fn rejects_oracle_access_from_deployable_arm() {
        let mut receipt = valid_receipt();
        receipt.arms[0].oracle_channel_access = true;
        assert!(validate_receipt(receipt).unwrap_err().to_string().contains("oracle_channel_access"));
    }

    #[test]
    fn rejects_verification_without_evidence_digest() {
        let mut receipt = valid_receipt();
        receipt.arms[3].decisions[2].verification_evidence_sha256 = None;
        assert!(validate_receipt(receipt).unwrap_err().to_string().contains("verify.verification_evidence_sha256"));
    }

    #[test]
    fn rejects_commit_without_a_verified_candidate() {
        let mut receipt = valid_receipt();
        let arm = &mut receipt.arms[3];
        arm.decisions[2].action = "commit".to_owned();
        arm.decisions[2].snapshot_sha256 = Some(digest('1'));
        arm.decisions[2].candidate_state_sha256 = Some(digest('2'));
        arm.decisions[2].verification_evidence_sha256 = None;
        assert!(validate_receipt(receipt).unwrap_err().to_string().contains("candidate must pass verification"));
    }

    #[test]
    fn rejects_rollback_that_does_not_restore_the_exact_snapshot() {
        let mut receipt = valid_receipt();
        receipt.arms[3].decisions[4].state_after_sha256 = digest('3');
        assert!(validate_receipt(receipt).unwrap_err().to_string().contains("post-commit rollback must restore"));
    }
}
