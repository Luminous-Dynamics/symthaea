//! Noisy multi-objective replay from versioned EventTape records.
//!
//! Consumes already-resolved replicate evidence bundles; performs no filesystem/network I/O.
//! Each successful observation must have one manifest, exact raw bytes, and a reconstructable
//! byte receipt. Failed attempts remain costed in EventTape and never enter Pareto statistics.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use super::event_tape::{
    ActionIntent, ActionOutcome, EventTape, EventTapeObjectiveContract, EventTapeObjectiveMode,
    NoisyObservationDisposition,
};
use super::noisy_replicate_evidence::{
    verify_noisy_replicate_artifact_receipt, NoisyReplicateEvidenceContext,
    NoisyReplicateEvidenceManifest, NoisyReplicateEvidenceReceipt,
};
use super::pareto_anytime::{AnytimeAction, AnytimeHistory, EvaluationOutcome};
use super::pareto_noisy_observations::{
    analyze_noisy_anytime_history, NoisyAnytimeCampaignReport, NoisyObjectiveSpec,
    NoisyObservationPlan, NoisyParetoReport, NoisyParetoStudy, NoisyReplicate,
};

const REPLAY_SCHEMA: &str = "symthaea.architecture-engineering.event-tape-noisy-replay";
const REPLAY_VERSION: &str = "0.1.0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventTapeNoisyError {
    pub path: String,
    pub message: String,
}

impl EventTapeNoisyError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self { path: path.into(), message: message.into() }
    }
}

impl fmt::Display for EventTapeNoisyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}
impl Error for EventTapeNoisyError {}

/// Caller-resolved evidence inputs for one successful noisy observation.
/// Keys in the bundle map are exact EventTape event IDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventTapeNoisyEvidenceBundle {
    pub manifest: NoisyReplicateEvidenceManifest,
    pub artifact_bytes: Vec<u8>,
    pub receipt: NoisyReplicateEvidenceReceipt,
}

/// Independent reconstruction of noisy Pareto samples directly from EventTape and raw bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventTapeNoisyReplayReport {
    pub schema: String,
    pub version: String,
    pub campaign_id: String,
    pub source_commit: String,
    pub objective_contract_sha256: String,
    pub noisy_plan_sha256: String,
    pub event_tape_tip_sha256: String,
    pub event_count: u64,
    pub attempted_observation_count: u64,
    pub successful_replicate_count: u64,
    pub failed_observation_count: u64,
    pub verified_artifact_count: u64,
    pub noisy_pareto_report: NoisyParetoReport,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct ReplayDigestPayload<'a> {
    schema: &'a str,
    version: &'a str,
    campaign_id: &'a str,
    source_commit: &'a str,
    objective_contract_sha256: &'a str,
    noisy_plan_sha256: &'a str,
    event_tape_tip_sha256: &'a str,
    event_count: u64,
    attempted_observation_count: u64,
    successful_replicate_count: u64,
    failed_observation_count: u64,
    verified_artifact_count: u64,
    noisy_pareto_report: &'a NoisyParetoReport,
}

/// Result after EventTape and AnytimeHistory independently reconstruct identical samples.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventTapeNoisyReconciliation {
    pub event_tape_replay: EventTapeNoisyReplayReport,
    pub anytime_campaign_report: NoisyAnytimeCampaignReport,
    pub reconciliation_sha256: String,
}

#[derive(Serialize)]
struct ReconciliationDigestPayload<'a> {
    event_tape_replay: &'a EventTapeNoisyReplayReport,
    anytime_campaign_report: &'a NoisyAnytimeCampaignReport,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ObservationStatus { Succeeded, Failed, Invalid, TimedOut }

#[derive(Debug, Clone, PartialEq)]
struct ObservationFingerprint {
    candidate_id: String,
    decision_sha256: String,
    replicate_id: String,
    sample_seed: u64,
    status: ObservationStatus,
    objective_values: Option<Vec<f64>>,
    evidence_sha256: Option<String>,
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest<T: Serialize>(value: &T) -> Result<String, EventTapeNoisyError> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        EventTapeNoisyError::new("serialization", format!("JSON encoding failed: {error}"))
    })?;
    Ok(sha256(&bytes))
}

fn plan_from_contract(
    contract: &EventTapeObjectiveContract,
    campaign_id: &str,
) -> Result<NoisyObservationPlan, EventTapeNoisyError> {
    if contract.mode != EventTapeObjectiveMode::NoisyReplicates {
        return Err(EventTapeNoisyError::new("header.objective_contract.mode", "noisy replay requires NoisyReplicates mode"));
    }
    let mut specs = Vec::with_capacity(contract.objective_specs.len());
    for (index, item) in contract.objective_specs.iter().enumerate() {
        specs.push(NoisyObjectiveSpec {
            objective_id: item.objective_id.clone(),
            direction: item.direction,
            observation_lower_bound: item.observation_lower_bound.ok_or_else(|| EventTapeNoisyError::new(
                format!("objective_specs[{index}].observation_lower_bound"), "missing noisy lower bound",
            ))?,
            observation_upper_bound: item.observation_upper_bound.ok_or_else(|| EventTapeNoisyError::new(
                format!("objective_specs[{index}].observation_upper_bound"), "missing noisy upper bound",
            ))?,
            bounds_certificate_sha256: item.bounds_certificate_sha256.clone().ok_or_else(|| EventTapeNoisyError::new(
                format!("objective_specs[{index}].bounds_certificate_sha256"), "missing bounds certificate",
            ))?,
        });
    }
    NoisyObservationPlan::new(
        campaign_id,
        contract.sampling_protocol_id.clone().unwrap_or_default(),
        contract.sampling_protocol_sha256.clone().unwrap_or_default(),
        contract.noise_model_id.clone().unwrap_or_default(),
        contract.noise_model_sha256.clone().unwrap_or_default(),
        specs,
        contract.familywise_alpha.unwrap_or_default(),
        contract.max_candidates.unwrap_or_default(),
        contract.minimum_replicates.unwrap_or_default(),
    ).map_err(|error| EventTapeNoisyError::new("objective_contract", error.to_string()))
}

fn contract_and_plan(
    tape: &EventTape,
) -> Result<(EventTapeObjectiveContract, String, NoisyObservationPlan), EventTapeNoisyError> {
    let header = tape.header();
    let contract = header.objective_contract.clone().ok_or_else(|| EventTapeNoisyError::new(
        "header.objective_contract", "missing frozen objective contract",
    ))?;
    if contract.mode != EventTapeObjectiveMode::NoisyReplicates {
        return Err(EventTapeNoisyError::new("header.objective_contract.mode", "campaign is not in noisy mode"));
    }
    let declared = header.objective_contract_sha256.as_deref().ok_or_else(|| EventTapeNoisyError::new(
        "header.objective_contract_sha256", "missing objective-contract digest",
    ))?;
    let actual = contract.contract_sha256().map_err(|error| EventTapeNoisyError::new(
        "header.objective_contract", error.to_string(),
    ))?;
    if declared != actual {
        return Err(EventTapeNoisyError::new("header.objective_contract_sha256", "digest does not reconstruct"));
    }
    let plan = plan_from_contract(&contract, &header.campaign_id)?;
    Ok((contract, actual, plan))
}

fn fingerprint_from_tape(
    intent: &ActionIntent,
    outcome: &ActionOutcome,
    path: &str,
) -> Result<ObservationFingerprint, EventTapeNoisyError> {
    let (candidate_id, decision_sha256, replicate_id, sample_seed) = match intent {
        ActionIntent::ObserveCandidateReplicate {
            candidate_sha256, decision_sha256, replicate_id, sample_seed, ..
        } => (candidate_sha256.clone(), decision_sha256.clone(), replicate_id.clone(), *sample_seed),
        _ => return Err(EventTapeNoisyError::new(format!("{path}.intent"), "not a noisy replicate intent")),
    };
    let (status, objective_values, evidence_sha256) = match outcome {
        ActionOutcome::CandidateNoisyObservation { disposition, objective_values, evidence_sha256, .. } => {
            let status = match disposition {
                NoisyObservationDisposition::Succeeded => ObservationStatus::Succeeded,
                NoisyObservationDisposition::Failed => ObservationStatus::Failed,
                NoisyObservationDisposition::Invalid | NoisyObservationDisposition::VerifierRejected => ObservationStatus::Invalid,
                NoisyObservationDisposition::TimedOut => ObservationStatus::TimedOut,
            };
            (status, objective_values.clone(), evidence_sha256.clone())
        }
        _ => return Err(EventTapeNoisyError::new(format!("{path}.outcome"), "wrong outcome type for noisy replicate intent")),
    };
    Ok(ObservationFingerprint {
        candidate_id, decision_sha256, replicate_id, sample_seed, status, objective_values, evidence_sha256,
    })
}

fn fingerprint_from_history(
    action: &AnytimeAction,
    path: &str,
) -> Result<Option<ObservationFingerprint>, EventTapeNoisyError> {
    let AnytimeAction::CandidateNoisyObservation {
        candidate_id, decision_sha256, replicate_id, sample_seed, outcome,
        objective_values, evidence_sha256, ..
    } = action else { return Ok(None) };
    let status = match outcome {
        EvaluationOutcome::Succeeded => ObservationStatus::Succeeded,
        EvaluationOutcome::Failed => ObservationStatus::Failed,
        EvaluationOutcome::Invalid => ObservationStatus::Invalid,
        EvaluationOutcome::TimedOut => ObservationStatus::TimedOut,
    };
    if status == ObservationStatus::Succeeded {
        if objective_values.is_none() || evidence_sha256.is_none() {
            return Err(EventTapeNoisyError::new(format!("{path}.action"), "successful history observation lacks values or evidence digest"));
        }
    } else if objective_values.is_some() || evidence_sha256.is_some() {
        return Err(EventTapeNoisyError::new(format!("{path}.action"), "failed history observation cannot publish values or evidence"));
    }
    Ok(Some(ObservationFingerprint {
        candidate_id: candidate_id.clone(),
        decision_sha256: decision_sha256.clone(),
        replicate_id: replicate_id.clone(),
        sample_seed: *sample_seed,
        status,
        objective_values: objective_values.clone(),
        evidence_sha256: evidence_sha256.clone(),
    }))
}

/// Rebuild the noisy Pareto report from EventTape, verifying one raw-byte receipt for every
/// successful replicate. The evidence map must contain exactly the successful event IDs.
pub fn replay_noisy_event_tape(
    tape: &EventTape,
    evidence: &BTreeMap<String, EventTapeNoisyEvidenceBundle>,
) -> Result<EventTapeNoisyReplayReport, EventTapeNoisyError> {
    let summary = tape.validate().map_err(|error| EventTapeNoisyError::new("event_tape", error.to_string()))?;
    let (_, contract_sha256, plan) = contract_and_plan(tape)?;
    let plan_sha256 = plan.plan_sha256().map_err(|error| EventTapeNoisyError::new("plan", error.to_string()))?;
    let header = tape.header();
    let mut study = NoisyParetoStudy::new(plan.clone()).map_err(|error| EventTapeNoisyError::new("study", error.to_string()))?;
    let mut attempted = 0u64;
    let mut successful = 0u64;
    let mut failed = 0u64;
    let mut verified = 0u64;
    let mut used = BTreeSet::new();

    for (index, envelope) in tape.events().iter().enumerate() {
        let payload = &envelope.payload;
        match (&payload.intent, &payload.outcome) {
            (
                ActionIntent::ObserveCandidateReplicate {
                    candidate_sha256, decision_sha256, replicate_id, sample_seed,
                    objective_contract_sha256: intent_contract, ..
                },
                ActionOutcome::CandidateNoisyObservation {
                    objective_contract_sha256: outcome_contract, disposition, objective_values,
                    evidence_sha256, evidence_manifest_sha256, evidence_byte_receipt_sha256, ..
                },
            ) => {
                let path = format!("events[{index}]");
                if intent_contract != &contract_sha256 || outcome_contract != &contract_sha256 {
                    return Err(EventTapeNoisyError::new(format!("{path}.objective_contract_sha256"), "event differs from frozen contract"));
                }
                attempted = attempted.checked_add(1).ok_or_else(|| EventTapeNoisyError::new("attempted_count", "counter overflow"))?;
                match disposition {
                    NoisyObservationDisposition::Succeeded => {
                        let values = objective_values.as_ref().ok_or_else(|| EventTapeNoisyError::new(format!("{path}.objective_values"), "missing successful vector"))?;
                        let artifact_sha = evidence_sha256.as_ref().ok_or_else(|| EventTapeNoisyError::new(format!("{path}.evidence_sha256"), "missing artifact digest"))?;
                        let manifest_sha = evidence_manifest_sha256.as_ref().ok_or_else(|| EventTapeNoisyError::new(format!("{path}.evidence_manifest_sha256"), "missing manifest digest"))?;
                        let receipt_sha = evidence_byte_receipt_sha256.as_ref().ok_or_else(|| EventTapeNoisyError::new(format!("{path}.evidence_byte_receipt_sha256"), "missing byte receipt digest"))?;
                        let bundle = evidence.get(&payload.event_id).ok_or_else(|| EventTapeNoisyError::new(format!("{path}.evidence"), "successful observation has no evidence bundle"))?;
                        if !used.insert(payload.event_id.clone()) {
                            return Err(EventTapeNoisyError::new(format!("{path}.evidence"), "bundle reused across events"));
                        }
                        let replicate = NoisyReplicate {
                            replicate_id: replicate_id.clone(),
                            sample_seed: *sample_seed,
                            objective_values: values.clone(),
                            evidence_sha256: artifact_sha.clone(),
                        };
                        let actual_manifest_sha = bundle.manifest.manifest_sha256().map_err(|error| EventTapeNoisyError::new(format!("{path}.manifest"), error.to_string()))?;
                        if &actual_manifest_sha != manifest_sha || &bundle.receipt.receipt_sha256 != receipt_sha {
                            return Err(EventTapeNoisyError::new(format!("{path}.evidence"), "manifest or receipt digest differs from event record"));
                        }
                        if &bundle.manifest.artifact.sha256 != artifact_sha {
                            return Err(EventTapeNoisyError::new(format!("{path}.manifest.artifact.sha256"), "artifact digest differs from event record"));
                        }
                        let context = NoisyReplicateEvidenceContext {
                            run_id: &header.campaign_id,
                            source_commit: &header.source_commit,
                            candidate_id: candidate_sha256,
                            decision_sha256,
                        };
                        verify_noisy_replicate_artifact_receipt(
                            &bundle.manifest, &plan, context, &replicate, &bundle.artifact_bytes, &bundle.receipt,
                        ).map_err(|error| EventTapeNoisyError::new(format!("{path}.evidence"), error.to_string()))?;
                        if bundle.receipt.run_id != header.campaign_id
                            || bundle.receipt.candidate_id != *candidate_sha256
                            || bundle.receipt.source_commit != header.source_commit
                            || bundle.receipt.decision_sha256 != *decision_sha256
                            || bundle.receipt.replicate_id != *replicate_id
                            || bundle.receipt.sample_seed != *sample_seed
                            || bundle.receipt.plan_sha256 != plan_sha256
                        {
                            return Err(EventTapeNoisyError::new(format!("{path}.receipt.identity"), "receipt identity differs from the exact EventTape action"));
                        }
                        study.append(candidate_sha256.clone(), decision_sha256.clone(), replicate)
                            .map_err(|error| EventTapeNoisyError::new(format!("{path}.study"), error.to_string()))?;
                        successful = successful.checked_add(1).ok_or_else(|| EventTapeNoisyError::new("successful_count", "counter overflow"))?;
                        verified = verified.checked_add(1).ok_or_else(|| EventTapeNoisyError::new("verified_count", "counter overflow"))?;
                    }
                    NoisyObservationDisposition::Failed | NoisyObservationDisposition::Invalid
                    | NoisyObservationDisposition::TimedOut | NoisyObservationDisposition::VerifierRejected => {
                        if objective_values.is_some() || evidence_sha256.is_some()
                            || evidence_manifest_sha256.is_some() || evidence_byte_receipt_sha256.is_some()
                        {
                            return Err(EventTapeNoisyError::new(format!("{path}.outcome"), "failed observation must not publish values or successful evidence"));
                        }
                        failed = failed.checked_add(1).ok_or_else(|| EventTapeNoisyError::new("failed_count", "counter overflow"))?;
                    }
                }
            }
            (ActionIntent::EvaluateCandidateVector { .. }, ActionOutcome::CandidateMultiObjectiveEvaluation { .. })
            | (ActionIntent::ProposeDirection { .. }, ActionOutcome::DirectionVerification { .. }) => {}
            _ => return Err(EventTapeNoisyError::new(format!("events[{index}]"), "unsupported or mismatched action in noisy replay")),
        }
    }
    if used.len() != evidence.len() {
        return Err(EventTapeNoisyError::new("evidence", "unreferenced evidence bundle supplied"));
    }
    if attempted != successful + failed || verified != successful {
        return Err(EventTapeNoisyError::new("counts", "noisy attempt, success, failure and byte-verification counts do not reconcile"));
    }
    let noisy_pareto_report = study.report().map_err(|error| EventTapeNoisyError::new("noisy_pareto_report", error.to_string()))?;
    let tip = tape.events().last().map(|event| event.event_sha256.clone())
        .ok_or_else(|| EventTapeNoisyError::new("event_tape.events", "empty tape cannot be replayed"))?;
    let mut report = EventTapeNoisyReplayReport {
        schema: REPLAY_SCHEMA.to_owned(), version: REPLAY_VERSION.to_owned(),
        campaign_id: header.campaign_id.clone(), source_commit: header.source_commit.clone(),
        objective_contract_sha256: contract_sha256, noisy_plan_sha256: plan_sha256,
        event_tape_tip_sha256: tip,
        event_count: u64::try_from(summary.event_count).map_err(|_| EventTapeNoisyError::new("event_count", "count exceeds u64"))?,
        attempted_observation_count: attempted, successful_replicate_count: successful,
        failed_observation_count: failed, verified_artifact_count: verified,
        noisy_pareto_report, report_sha256: String::new(),
    };
    report.report_sha256 = digest(&ReplayDigestPayload {
        schema: &report.schema, version: &report.version, campaign_id: &report.campaign_id,
        source_commit: &report.source_commit, objective_contract_sha256: &report.objective_contract_sha256,
        noisy_plan_sha256: &report.noisy_plan_sha256, event_tape_tip_sha256: &report.event_tape_tip_sha256,
        event_count: report.event_count, attempted_observation_count: report.attempted_observation_count,
        successful_replicate_count: report.successful_replicate_count, failed_observation_count: report.failed_observation_count,
        verified_artifact_count: report.verified_artifact_count, noisy_pareto_report: &report.noisy_pareto_report,
    })?;
    Ok(report)
}

/// Require exact attempt-level parity and identical independently reconstructed noisy Pareto
/// reports between EventTape and AnytimeHistory.
pub fn reconcile_noisy_event_tape_with_anytime_history(
    tape: &EventTape,
    evidence: &BTreeMap<String, EventTapeNoisyEvidenceBundle>,
    history: &AnytimeHistory,
) -> Result<EventTapeNoisyReconciliation, EventTapeNoisyError> {
    let replay = replay_noisy_event_tape(tape, evidence)?;
    let history_header = history.header();
    if history_header.run_id != tape.header().campaign_id || history_header.source_commit != tape.header().source_commit {
        return Err(EventTapeNoisyError::new("history.header", "run/campaign ID or source commit differs between histories"));
    }
    if history_header.noisy_observation_plan_sha256.as_deref() != Some(replay.noisy_plan_sha256.as_str()) {
        return Err(EventTapeNoisyError::new("history.header.noisy_observation_plan_sha256", "AnytimeHistory does not preregister the EventTape noisy plan"));
    }
    let (_, _, plan) = contract_and_plan(tape)?;
    if history_header.objective_specs.len() != plan.objective_specs.len()
        || history_header.objective_specs.iter().zip(&plan.objective_specs)
            .any(|(left, right)| left.objective_id != right.objective_id || left.direction != right.direction)
    {
        return Err(EventTapeNoisyError::new("history.header.objective_specs", "objective ID/order/direction differs"));
    }

    let mut tape_rows = Vec::new();
    for (index, envelope) in tape.events().iter().enumerate() {
        if matches!(envelope.payload.intent, ActionIntent::ObserveCandidateReplicate { .. }) {
            tape_rows.push(fingerprint_from_tape(&envelope.payload.intent, &envelope.payload.outcome, &format!("events[{index}]"))?);
        }
    }
    let mut history_rows = Vec::new();
    for (index, event) in history.events().iter().enumerate() {
        if let Some(row) = fingerprint_from_history(&event.action, &format!("history.events[{index}]"))? {
            history_rows.push(row);
        }
    }
    let key = |row: &ObservationFingerprint| (&row.candidate_id, &row.replicate_id);
    tape_rows.sort_by(|a,b| key(a).cmp(&key(b)));
    history_rows.sort_by(|a,b| key(a).cmp(&key(b)));
    if tape_rows != history_rows {
        return Err(EventTapeNoisyError::new("reconciliation.observations", "attempt identity/status/seed/objective/evidence fields differ across histories"));
    }

    let anytime_campaign_report = analyze_noisy_anytime_history(&plan, history)
        .map_err(|error| EventTapeNoisyError::new("anytime_campaign_report", error.to_string()))?;
    if anytime_campaign_report.noisy_pareto_report != replay.noisy_pareto_report
        || anytime_campaign_report.attempted_observation_count != replay.attempted_observation_count
        || anytime_campaign_report.successful_replicate_count != replay.successful_replicate_count
        || anytime_campaign_report.failed_observation_count != replay.failed_observation_count
    {
        return Err(EventTapeNoisyError::new("reconciliation.report", "independent Pareto reports or attempt counters differ"));
    }
    let mut reconciliation = EventTapeNoisyReconciliation {
        event_tape_replay: replay,
        anytime_campaign_report,
        reconciliation_sha256: String::new(),
    };
    reconciliation.reconciliation_sha256 = digest(&ReconciliationDigestPayload {
        event_tape_replay: &reconciliation.event_tape_replay,
        anytime_campaign_report: &reconciliation.anytime_campaign_report,
    })?;
    Ok(reconciliation)
}

/// Recompute the complete reconciliation; tampered stored outputs fail closed.
pub fn verify_noisy_event_tape_reconciliation(
    tape: &EventTape,
    evidence: &BTreeMap<String, EventTapeNoisyEvidenceBundle>,
    history: &AnytimeHistory,
    expected: &EventTapeNoisyReconciliation,
) -> Result<(), EventTapeNoisyError> {
    let actual = reconcile_noisy_event_tape_with_anytime_history(tape, evidence, history)?;
    if &actual != expected {
        return Err(EventTapeNoisyError::new("reconciliation", "stored reconciliation differs from independent reconstruction"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verifier_rejection_is_explicitly_downgraded_to_invalid_history_status() {
        let intent = ActionIntent::ObserveCandidateReplicate {
            candidate_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
            parent_candidate_sha256: None,
            semantic_delta_sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned(),
            direction_id: "A".to_owned(),
            objective_contract_sha256: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".to_owned(),
            decision_sha256: "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".to_owned(),
            replicate_id: "replicate-1".to_owned(),
            sample_seed: 17,
        };
        let outcome = ActionOutcome::CandidateNoisyObservation {
            task_recipe_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
            comparison_profile_sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned(),
            evaluator_sha256: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".to_owned(),
            verifier_sha256: "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".to_owned(),
            verifier_receipt_sha256: "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee".to_owned(),
            objective_contract_sha256: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".to_owned(),
            disposition: NoisyObservationDisposition::VerifierRejected,
            objective_values: None, evidence_sha256: None, evidence_manifest_sha256: None,
            evidence_byte_receipt_sha256: None, failure_reason: Some("receipt mismatch".to_owned()),
        };
        let row = fingerprint_from_tape(&intent, &outcome, "event").unwrap();
        assert_eq!(row.status, ObservationStatus::Invalid);
        assert_eq!(row.objective_values, None);
        assert_eq!(row.evidence_sha256, None);
    }
}
