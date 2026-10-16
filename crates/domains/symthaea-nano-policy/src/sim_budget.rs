//! Actual simulator-call accounting for budgeted scientific search.
//!
//! This module deliberately separates proposal attempts, preflight rejections,
//! simulator process invocations, and usable finite outputs. A failed, timed-out,
//! interrupted, or non-finite simulator invocation still consumes one simulator call.
//! The audit is structural: it validates the frozen ledger and identities, but does
//! not itself prove that referenced external receipt artifacts are truthful.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

/// Input ledger contract version.
pub const SIMULATOR_LEDGER_SCHEMA_VERSION: &str = "nano-simulator-call-ledger-v1";
/// Output audit contract version.
pub const SIMULATOR_AUDIT_SCHEMA_VERSION: &str = "nano-simulator-call-audit-v1";
/// Maximum raw JSON ledger size.
pub const MAX_SIMULATOR_LEDGER_BYTES: usize = 64 * 1024 * 1024;
/// Maximum proposal-attempt rows in one ledger.
pub const MAX_PROPOSAL_ATTEMPTS: usize = 100_000;
/// Maximum expected or reported simulator-call count.
pub const MAX_SIMULATOR_CALLS: usize = 100_000;
/// Maximum human-readable identifier or reason length, in bytes.
pub const MAX_SIM_LEDGER_TEXT_BYTES: usize = 256;
/// Maximum declared cost value per proposal or simulator invocation.
pub const MAX_SIM_LEDGER_COST: f64 = 1.0e12;
/// Maximum elapsed time recorded for one proposal or simulator call (seven days).
pub const MAX_SIM_LEDGER_ELAPSED_MS: u64 = 604_800_000;

/// Status declared by the process that wrote the run ledger.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    /// The orchestrator claims it completed the configured work.
    Completed,
    /// The orchestrator stopped without claiming completion.
    Incomplete,
    /// The orchestrator encountered a fatal run-level failure.
    Failed,
}

/// Whether a proposal attempt produced a candidate.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProposalStatus {
    /// A candidate was generated and has a stable identity/digest.
    Generated,
    /// Generation failed before a candidate existed.
    GenerationFailed,
}

/// Pre-simulation rule-check outcome. A rejected candidate consumes proposal
/// budget but **not** simulator-call budget.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PreflightStatus {
    /// No preflight was possible because generation failed.
    NotRun,
    /// Candidate passed preflight and could be submitted to the simulator.
    Passed,
    /// Candidate failed a rule check before the simulator started.
    Rejected,
}

/// Reason an eligible proposal did not start a simulator process.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NotCalledReason {
    /// Process launch failed before the simulator invocation began.
    LaunchFailed,
    /// The run was interrupted before a simulator process began.
    RunnerInterrupted,
    /// The orchestrator's explicit budget guard prevented an extra invocation.
    BudgetGuard,
    /// The upstream receipt cannot classify the non-invocation more specifically.
    Unknown,
}

/// Terminal result of an actual simulator invocation. Every variant consumes one
/// simulator-call budget unit, including failures and non-finite results.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SimulatorTerminalStatus {
    /// The simulator returned a finite response suitable for task metrics.
    ReturnedFiniteResult,
    /// The simulator returned an artifact, but its scientific result is non-finite.
    ReturnedNonfiniteResult,
    /// The simulator invocation started and then failed.
    Failed,
    /// The simulator invocation timed out.
    TimedOut,
    /// The invocation started but completion state was unresolved at finalization.
    Interrupted,
}

/// One actual invocation of the declared simulator.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SimulatorCall {
    /// Stable call identity allocated by the orchestrator.
    pub invocation_id: String,
    /// Digest of the raw receipt that records simulator process start.
    pub started_receipt_sha256: String,
    /// Digest of the terminal receipt, including failure/timeout/interruption.
    pub terminal_receipt_sha256: String,
    /// Terminal status. Failures still count as actual invocations.
    pub terminal_status: SimulatorTerminalStatus,
    /// Digest of the raw result artifact, required when a result was returned.
    #[serde(deserialize_with = "deserialize_required_option")]
    pub result_artifact_sha256: Option<String>,
    /// Wall-clock elapsed time for this invocation in milliseconds.
    pub elapsed_ms: u64,
    /// Cost of this invocation in the ledger-wide declared cost unit.
    pub cost: f64,
}

/// One proposal attempt, including any work rejected before simulation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProposalAttempt {
    /// One-based, contiguous sequence in the append order of the run.
    pub sequence: usize,
    /// Stable identity for this generation/proposal attempt.
    pub attempt_id: String,
    /// Candidate identity, absent only when generation failed.
    #[serde(deserialize_with = "deserialize_required_option")]
    pub candidate_id: Option<String>,
    /// SHA-256 of the canonical candidate representation, if a candidate exists.
    #[serde(deserialize_with = "deserialize_required_option")]
    pub candidate_sha256: Option<String>,
    /// Whether a candidate was actually generated.
    pub proposal_status: ProposalStatus,
    /// Result of preflight checks before launching the simulator.
    pub preflight_status: PreflightStatus,
    /// Required when generation or preflight rejection fails; absent on a clean pass.
    #[serde(deserialize_with = "deserialize_required_option")]
    pub attempt_reason: Option<String>,
    /// Required when preflight passed but no simulator invocation started.
    #[serde(deserialize_with = "deserialize_required_option")]
    pub not_called_reason: Option<NotCalledReason>,
    /// Candidate-generation and preflight cost, excluding simulator-call cost.
    pub proposal_cost: f64,
    /// Candidate-generation and preflight elapsed wall time in milliseconds.
    pub proposal_elapsed_ms: u64,
    /// Present only when the simulator actually started. The presence of this
    /// record consumes exactly one unit of simulator-call budget.
    #[serde(deserialize_with = "deserialize_required_option")]
    pub simulator_call: Option<SimulatorCall>,
}

/// Frozen per-policy run ledger for reconciling configured calls, actual invocations,
/// finite outputs, proposal attempts and costs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SimulatorBudgetLedger {
    /// Must equal SIMULATOR_LEDGER_SCHEMA_VERSION.
    pub schema_version: String,
    /// Benchmark identifier, e.g. the exact MEEP study version.
    pub benchmark_id: String,
    /// Unique run identifier.
    pub run_id: String,
    /// Policy arm identifier used in paired comparisons.
    pub arm_id: String,
    /// Policy name.
    pub policy_id: String,
    /// Immutable lowercase 40-character source revision for the policy/orchestrator.
    pub policy_revision: String,
    /// Digest of the frozen top-level run manifest.
    pub run_manifest_sha256: String,
    /// Digest of the task/objective/unit contract.
    pub task_contract_sha256: String,
    /// Digest of the evaluation target/split/group manifest.
    pub evaluation_manifest_sha256: String,
    /// Digest of the data/source/license provenance manifest.
    pub source_manifest_sha256: String,
    /// Digest of the exact simulator recipe/settings.
    pub simulator_recipe_sha256: String,
    /// Preregistered number of actual simulator invocations required by a complete run.
    pub expected_simulator_calls: usize,
    /// Number reported by the original run summary (e.g. a historical n_meep field).
    /// This is a self-reported value and is compared with the attempt ledger.
    pub reported_simulator_calls: usize,
    /// Maximum number of proposal/generation attempts allowed, including preflight rejections.
    pub proposal_attempt_cap: usize,
    /// One shared unit for proposal and simulator costs, e.g. core_seconds or cost_units.
    pub cost_unit: String,
    /// Status declared by the run orchestrator.
    pub run_status: RunStatus,
    /// Ordered, complete proposal-attempt ledger for this arm.
    pub attempts: Vec<ProposalAttempt>,
}

/// Computed audit classification. The main point is to avoid conflating configured
/// budget, actual calls, terminal results and finite task outputs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AuditDisposition {
    /// The run claims completion and has exactly the required calls, all terminal,
    /// with no uncalled eligible or post-budget proposal attempts.
    EqualBudgetComplete,
    /// Fewer actual invocations occurred than the required budget.
    IncompleteUnderBudget,
    /// More actual invocations occurred than the frozen budget.
    OverBudget,
    /// All required calls appear, but at least one invocation is unresolved.
    BudgetConsumedButUnresolved,
    /// The run claims completion but the attempt order/termination violates the protocol.
    CompletedClaimMismatch,
    /// The source run itself is marked failed.
    FailedRun,
    /// The source run is marked incomplete even though no simpler count classification applied.
    IncompleteRun,
}

/// Independently calculated counts and budget reconciliation for a frozen ledger.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SimulatorBudgetAudit {
    /// Must equal SIMULATOR_AUDIT_SCHEMA_VERSION.
    pub schema_version: String,
    /// SHA-256 of the exact raw ledger JSON bytes audited.
    pub ledger_sha256: String,
    /// Frozen run and arm identities.
    pub benchmark_id: String,
    pub run_id: String,
    pub arm_id: String,
    pub policy_id: String,
    pub run_status: RunStatus,
    /// Frozen target count for actual simulator process invocations.
    pub expected_simulator_calls: usize,
    /// Count recorded in the pre-existing summary/report.
    pub reported_simulator_calls: usize,
    /// Count independently recomputed from attempt rows with invocation records.
    pub observed_simulator_calls: usize,
    /// Whether the source summary's reported count matches invocation records.
    pub reported_count_matches_observed: bool,
    /// True only when observed calls equal the preregistered simulator budget.
    pub exact_call_budget_match: bool,
    /// Total proposal/generation attempts, including generation failures and preflight rejects.
    pub proposal_attempts: usize,
    pub proposal_attempt_cap: usize,
    pub generated_candidates: usize,
    pub generation_failures: usize,
    pub preflight_passed: usize,
    pub preflight_rejected: usize,
    /// Candidates that passed preflight but never started the simulator.
    pub eligible_candidates_not_called: usize,
    /// Proposal records that occur after the budget-th simulator invocation.
    pub proposal_attempts_after_budget_reached: usize,
    /// Unique canonical candidate digests, across generated candidates.
    pub unique_candidate_digests: usize,
    /// Repeat candidate attempts, retained in all attempt/call counts.
    pub repeated_candidate_attempts: usize,
    /// Actual call count by terminal status.
    pub returned_finite_results: usize,
    pub returned_nonfinite_results: usize,
    pub simulator_failures: usize,
    pub simulator_timeouts: usize,
    pub unresolved_simulator_calls: usize,
    /// Sum of costs and elapsed times; failed operations remain included.
    pub total_proposal_cost: f64,
    pub total_simulator_cost: f64,
    pub total_cost: f64,
    pub total_proposal_elapsed_ms: u64,
    pub total_simulator_elapsed_ms: u64,
    /// Computed disposition for denominator/budget audit.
    pub disposition: AuditDisposition,
    /// True only for a completed exact-budget run with the source summary reconciled.
    pub audit_pass: bool,
    /// Claim boundaries retained with every report.
    pub limitations: Vec<String>,
}

/// Parse, validate and audit a raw JSON ledger. The returned digest is computed from
/// the exact bytes, not from a caller-supplied field.
pub fn audit_json_ledger(raw_ledger: &[u8]) -> Result<SimulatorBudgetAudit, String> {
    if raw_ledger.len() > MAX_SIMULATOR_LEDGER_BYTES {
        return Err(format!(
            "simulator ledger exceeds the {MAX_SIMULATOR_LEDGER_BYTES}-byte limit"
        ));
    }
    let ledger_sha256 = sha256_hex(raw_ledger);
    let ledger: SimulatorBudgetLedger = serde_json::from_slice(raw_ledger)
        .map_err(|error| format!("invalid simulator-budget ledger JSON: {error}"))?;
    validate_ledger(&ledger)?;
    Ok(audit_ledger(&ledger, ledger_sha256))
}

fn validate_ledger(ledger: &SimulatorBudgetLedger) -> Result<(), String> {
    if ledger.schema_version != SIMULATOR_LEDGER_SCHEMA_VERSION {
        return Err(format!(
            "schema_version must be {SIMULATOR_LEDGER_SCHEMA_VERSION}"
        ));
    }
    validate_text(&ledger.benchmark_id, "benchmark_id")?;
    validate_text(&ledger.run_id, "run_id")?;
    validate_text(&ledger.arm_id, "arm_id")?;
    validate_text(&ledger.policy_id, "policy_id")?;
    validate_text(&ledger.cost_unit, "cost_unit")?;
    if !is_hex_digest(&ledger.policy_revision, 40) {
        return Err("policy_revision must be 40 lowercase hexadecimal characters".to_string());
    }
    for (name, value) in [
        ("run_manifest_sha256", &ledger.run_manifest_sha256),
        ("task_contract_sha256", &ledger.task_contract_sha256),
        ("evaluation_manifest_sha256", &ledger.evaluation_manifest_sha256),
        ("source_manifest_sha256", &ledger.source_manifest_sha256),
        ("simulator_recipe_sha256", &ledger.simulator_recipe_sha256),
    ] {
        if !is_hex_digest(value, 64) {
            return Err(format!("{name} must be 64 lowercase hexadecimal characters"));
        }
    }
    if ledger.expected_simulator_calls == 0 || ledger.expected_simulator_calls > MAX_SIMULATOR_CALLS {
        return Err(format!(
            "expected_simulator_calls must be in 1..={MAX_SIMULATOR_CALLS}"
        ));
    }
    if ledger.reported_simulator_calls > MAX_SIMULATOR_CALLS {
        return Err(format!(
            "reported_simulator_calls exceeds {MAX_SIMULATOR_CALLS}"
        ));
    }
    if ledger.proposal_attempt_cap == 0 || ledger.proposal_attempt_cap > MAX_PROPOSAL_ATTEMPTS {
        return Err(format!(
            "proposal_attempt_cap must be in 1..={MAX_PROPOSAL_ATTEMPTS}"
        ));
    }
    if ledger.attempts.len() > ledger.proposal_attempt_cap {
        return Err(format!(
            "attempt count {} exceeds proposal_attempt_cap {}",
            ledger.attempts.len(),
            ledger.proposal_attempt_cap
        ));
    }

    let mut attempt_ids = HashSet::with_capacity(ledger.attempts.len());
    let mut invocation_ids = HashSet::new();
    for (zero_index, attempt) in ledger.attempts.iter().enumerate() {
        let expected_sequence = zero_index + 1;
        if attempt.sequence != expected_sequence {
            return Err(format!(
                "attempt sequence must be contiguous and one-based: expected {expected_sequence}, got {}",
                attempt.sequence
            ));
        }
        validate_text(&attempt.attempt_id, "attempt_id")?;
        if !attempt_ids.insert(attempt.attempt_id.as_str()) {
            return Err(format!("duplicate attempt_id {:?}", attempt.attempt_id));
        }
        if !valid_cost(attempt.proposal_cost) {
            return Err(format!(
                "attempt {} has invalid proposal_cost",
                attempt.attempt_id
            ));
        }
        if attempt.proposal_elapsed_ms > MAX_SIM_LEDGER_ELAPSED_MS {
            return Err(format!(
                "attempt {} proposal_elapsed_ms exceeds the maximum",
                attempt.attempt_id
            ));
        }

        match attempt.proposal_status {
            ProposalStatus::GenerationFailed => {
                if attempt.candidate_id.is_some()
                    || attempt.candidate_sha256.is_some()
                    || attempt.preflight_status != PreflightStatus::NotRun
                    || attempt.simulator_call.is_some()
                    || attempt.not_called_reason.is_some()
                    || attempt.attempt_reason.as_ref().is_none_or(|reason| reason.trim().is_empty())
                {
                    return Err(format!(
                        "generation-failed attempt {} must have no candidate, preflight, simulator call, or not-called reason, and needs attempt_reason",
                        attempt.attempt_id
                    ));
                }
            }
            ProposalStatus::Generated => {
                let candidate_id = attempt.candidate_id.as_deref().ok_or_else(|| {
                    format!("generated attempt {} requires candidate_id", attempt.attempt_id)
                })?;
                validate_text(candidate_id, "candidate_id")?;
                let candidate_digest = attempt.candidate_sha256.as_deref().ok_or_else(|| {
                    format!(
                        "generated attempt {} requires candidate_sha256",
                        attempt.attempt_id
                    )
                })?;
                if !is_hex_digest(candidate_digest, 64) {
                    return Err(format!(
                        "attempt {} candidate_sha256 must be 64 lowercase hexadecimal characters",
                        attempt.attempt_id
                    ));
                }
                match attempt.preflight_status {
                    PreflightStatus::NotRun => {
                        return Err(format!(
                            "generated attempt {} cannot use preflight_status=not_run",
                            attempt.attempt_id
                        ));
                    }
                    PreflightStatus::Rejected => {
                        if attempt.simulator_call.is_some()
                            || attempt.not_called_reason.is_some()
                            || attempt.attempt_reason.as_ref().is_none_or(|reason| reason.trim().is_empty())
                        {
                            return Err(format!(
                                "preflight-rejected attempt {} must not have a simulator call or not-called reason and needs attempt_reason",
                                attempt.attempt_id
                            ));
                        }
                    }
                    PreflightStatus::Passed => {
                        if attempt.attempt_reason.is_some() {
                            return Err(format!(
                                "preflight-passed attempt {} must not have attempt_reason",
                                attempt.attempt_id
                            ));
                        }
                        match (&attempt.simulator_call, attempt.not_called_reason) {
                            (Some(_), None) | (None, Some(_)) => {}
                            _ => {
                                return Err(format!(
                                    "preflight-passed attempt {} must have exactly one of simulator_call or not_called_reason",
                                    attempt.attempt_id
                                ));
                            }
                        }
                    }
                }
            }
        }

        if let Some(call) = &attempt.simulator_call {
            validate_text(&call.invocation_id, "invocation_id")?;
            if !invocation_ids.insert(call.invocation_id.as_str()) {
                return Err(format!("duplicate simulator invocation_id {:?}", call.invocation_id));
            }
            for (name, digest) in [
                ("started_receipt_sha256", &call.started_receipt_sha256),
                ("terminal_receipt_sha256", &call.terminal_receipt_sha256),
            ] {
                if !is_hex_digest(digest, 64) {
                    return Err(format!(
                        "invocation {} {name} must be a 64-character lowercase SHA-256",
                        call.invocation_id
                    ));
                }
            }
            match call.terminal_status {
                SimulatorTerminalStatus::ReturnedFiniteResult
                | SimulatorTerminalStatus::ReturnedNonfiniteResult => {
                    let result_digest = call.result_artifact_sha256.as_deref().ok_or_else(|| {
                        format!(
                            "invocation {} returned a result but lacks result_artifact_sha256",
                            call.invocation_id
                        )
                    })?;
                    if !is_hex_digest(result_digest, 64) {
                        return Err(format!(
                            "invocation {} result_artifact_sha256 must be a 64-character lowercase SHA-256",
                            call.invocation_id
                        ));
                    }
                }
                SimulatorTerminalStatus::Failed
                | SimulatorTerminalStatus::TimedOut
                | SimulatorTerminalStatus::Interrupted => {
                    if call.result_artifact_sha256.is_some() {
                        return Err(format!(
                            "invocation {} has result_artifact_sha256 despite no usable returned result",
                            call.invocation_id
                        ));
                    }
                }
            }
            if call.elapsed_ms > MAX_SIM_LEDGER_ELAPSED_MS {
                return Err(format!(
                    "invocation {} elapsed_ms exceeds the maximum",
                    call.invocation_id
                ));
            }
            if !valid_cost(call.cost) {
                return Err(format!("invocation {} has invalid cost", call.invocation_id));
            }
        }
    }

    // Call-budget ordering is assessed by the audit report below. Structural
    // constraints (including the per-run cap) have already been checked above.
    Ok(())
}

fn audit_ledger(ledger: &SimulatorBudgetLedger, ledger_sha256: String) -> SimulatorBudgetAudit {
    let proposal_attempts = ledger.attempts.len();
    let generated_candidates = ledger
        .attempts
        .iter()
        .filter(|attempt| attempt.proposal_status == ProposalStatus::Generated)
        .count();
    let generation_failures = ledger
        .attempts
        .iter()
        .filter(|attempt| attempt.proposal_status == ProposalStatus::GenerationFailed)
        .count();
    let preflight_passed = ledger
        .attempts
        .iter()
        .filter(|attempt| attempt.preflight_status == PreflightStatus::Passed)
        .count();
    let preflight_rejected = ledger
        .attempts
        .iter()
        .filter(|attempt| attempt.preflight_status == PreflightStatus::Rejected)
        .count();
    let eligible_candidates_not_called = ledger
        .attempts
        .iter()
        .filter(|attempt| {
            attempt.preflight_status == PreflightStatus::Passed
                && attempt.simulator_call.is_none()
        })
        .count();
    let mut candidate_digests = HashSet::new();
    let mut generated_digest_count = 0usize;
    for attempt in &ledger.attempts {
        if let Some(digest) = &attempt.candidate_sha256 {
            generated_digest_count += 1;
            candidate_digests.insert(digest.as_str());
        }
    }
    let unique_candidate_digests = candidate_digests.len();
    let repeated_candidate_attempts = generated_digest_count.saturating_sub(unique_candidate_digests);

    let call_positions: Vec<usize> = ledger
        .attempts
        .iter()
        .enumerate()
        .filter_map(|(index, attempt)| attempt.simulator_call.as_ref().map(|_| index))
        .collect();
    let observed_simulator_calls = call_positions.len();
    let proposal_attempts_after_budget_reached = call_positions
        .get(ledger.expected_simulator_calls.saturating_sub(1))
        .map(|budget_call_index| ledger.attempts.len().saturating_sub(budget_call_index + 1))
        .unwrap_or(0);

    let mut returned_finite_results = 0usize;
    let mut returned_nonfinite_results = 0usize;
    let mut simulator_failures = 0usize;
    let mut simulator_timeouts = 0usize;
    let mut unresolved_simulator_calls = 0usize;
    let mut total_simulator_cost = 0.0;
    let mut total_simulator_elapsed_ms = 0u64;
    for attempt in &ledger.attempts {
        if let Some(call) = &attempt.simulator_call {
            match call.terminal_status {
                SimulatorTerminalStatus::ReturnedFiniteResult => returned_finite_results += 1,
                SimulatorTerminalStatus::ReturnedNonfiniteResult => returned_nonfinite_results += 1,
                SimulatorTerminalStatus::Failed => simulator_failures += 1,
                SimulatorTerminalStatus::TimedOut => simulator_timeouts += 1,
                SimulatorTerminalStatus::Interrupted => unresolved_simulator_calls += 1,
            }
            total_simulator_cost += call.cost;
            total_simulator_elapsed_ms =
                total_simulator_elapsed_ms.saturating_add(call.elapsed_ms);
        }
    }
    let total_proposal_cost = ledger
        .attempts
        .iter()
        .map(|attempt| attempt.proposal_cost)
        .sum::<f64>();
    let total_proposal_elapsed_ms = ledger
        .attempts
        .iter()
        .map(|attempt| attempt.proposal_elapsed_ms)
        .fold(0u64, u64::saturating_add);
    let exact_call_budget_match = observed_simulator_calls == ledger.expected_simulator_calls;
    let reported_count_matches_observed = ledger.reported_simulator_calls == observed_simulator_calls;
    let any_calls_unresolved = unresolved_simulator_calls > 0;

    let disposition = if observed_simulator_calls > ledger.expected_simulator_calls {
        AuditDisposition::OverBudget
    } else if ledger.run_status == RunStatus::Failed {
        AuditDisposition::FailedRun
    } else if observed_simulator_calls < ledger.expected_simulator_calls {
        AuditDisposition::IncompleteUnderBudget
    } else if any_calls_unresolved {
        AuditDisposition::BudgetConsumedButUnresolved
    } else if ledger.run_status == RunStatus::Completed
        && (eligible_candidates_not_called > 0 || proposal_attempts_after_budget_reached > 0)
    {
        AuditDisposition::CompletedClaimMismatch
    } else if ledger.run_status == RunStatus::Completed {
        AuditDisposition::EqualBudgetComplete
    } else {
        AuditDisposition::IncompleteRun
    };

    let total_cost = total_proposal_cost + total_simulator_cost;
    let audit_pass = disposition == AuditDisposition::EqualBudgetComplete
        && reported_count_matches_observed;

    SimulatorBudgetAudit {
        schema_version: SIMULATOR_AUDIT_SCHEMA_VERSION.to_string(),
        ledger_sha256,
        benchmark_id: ledger.benchmark_id.clone(),
        run_id: ledger.run_id.clone(),
        arm_id: ledger.arm_id.clone(),
        policy_id: ledger.policy_id.clone(),
        run_status: ledger.run_status,
        expected_simulator_calls: ledger.expected_simulator_calls,
        reported_simulator_calls: ledger.reported_simulator_calls,
        observed_simulator_calls,
        reported_count_matches_observed,
        exact_call_budget_match,
        proposal_attempts,
        proposal_attempt_cap: ledger.proposal_attempt_cap,
        generated_candidates,
        generation_failures,
        preflight_passed,
        preflight_rejected,
        eligible_candidates_not_called,
        proposal_attempts_after_budget_reached,
        unique_candidate_digests,
        repeated_candidate_attempts,
        returned_finite_results,
        returned_nonfinite_results,
        simulator_failures,
        simulator_timeouts,
        unresolved_simulator_calls,
        total_proposal_cost,
        total_simulator_cost,
        total_cost,
        total_proposal_elapsed_ms,
        total_simulator_elapsed_ms,
        disposition,
        audit_pass,
        limitations: vec![
            "The audit counts invocation records and checks receipt-digest syntax; it does not dereference receipt artifacts or prove that the external process actually ran.".to_string(),
            "Completeness depends on the simulator supervisor retaining one start and terminal receipt for every actual invocation, including crashes and timeouts.".to_string(),
            "The simulator recipe and source/evaluation identities are digest-bound but not scientifically validated by this structural audit.".to_string(),
            "Proposal and simulator costs must use the single declared cost_unit; elapsed wall time is reported separately.".to_string(),
            "A complete equal-call budget does not imply equal wall time, equal compute cost, model quality, or physical validity.".to_string(),
            "A simulator call returning non-finite output, failing, timing out, or being interrupted still consumes one actual-call budget unit.".to_string(),
        ],
    }
}

fn deserialize_required_option<'de, D, T>(
    deserializer: D,
) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

fn validate_text(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty()
        || value.len() > MAX_SIM_LEDGER_TEXT_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(format!(
            "{field} must be non-empty, at most {MAX_SIM_LEDGER_TEXT_BYTES} bytes, and contain no control characters"
        ));
    }
    Ok(())
}

fn valid_cost(value: f64) -> bool {
    value.is_finite() && (0.0..=MAX_SIM_LEDGER_COST).contains(&value)
}

fn is_hex_digest(value: &str, expected_len: usize) -> bool {
    value.len() == expected_len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(letter: char) -> String {
        std::iter::repeat_n(letter, 64).collect()
    }

    fn attempt(
        sequence: usize,
        attempt_id: &str,
        candidate_id: &str,
        candidate_digest: String,
        preflight_status: PreflightStatus,
        simulator_call: Option<SimulatorCall>,
        attempt_reason: Option<&str>,
        not_called_reason: Option<NotCalledReason>,
    ) -> ProposalAttempt {
        ProposalAttempt {
            sequence,
            attempt_id: attempt_id.to_string(),
            candidate_id: Some(candidate_id.to_string()),
            candidate_sha256: Some(candidate_digest),
            proposal_status: ProposalStatus::Generated,
            preflight_status,
            attempt_reason: attempt_reason.map(str::to_string),
            not_called_reason,
            proposal_cost: 0.1,
            proposal_elapsed_ms: 10,
            simulator_call,
        }
    }

    fn call(id: &str, terminal_status: SimulatorTerminalStatus, result_hash: Option<String>, cost: f64) -> SimulatorCall {
        SimulatorCall {
            invocation_id: id.to_string(),
            started_receipt_sha256: digest('d'),
            terminal_receipt_sha256: digest('e'),
            terminal_status,
            result_artifact_sha256: result_hash,
            elapsed_ms: 100,
            cost,
        }
    }

    fn sample_ledger(status: RunStatus) -> SimulatorBudgetLedger {
        let finite = call(
            "invocation-a",
            SimulatorTerminalStatus::ReturnedFiniteResult,
            Some(digest('f')),
            1.0,
        );
        let failed = call(
            "invocation-c",
            SimulatorTerminalStatus::Failed,
            None,
            2.0,
        );
        SimulatorBudgetLedger {
            schema_version: SIMULATOR_LEDGER_SCHEMA_VERSION.to_string(),
            benchmark_id: "synthetic-meep-budget-test".to_string(),
            run_id: "run-001".to_string(),
            arm_id: "random".to_string(),
            policy_id: "random_seeded".to_string(),
            policy_revision: std::iter::repeat_n('a', 40).collect(),
            run_manifest_sha256: digest('1'),
            task_contract_sha256: digest('2'),
            evaluation_manifest_sha256: digest('3'),
            source_manifest_sha256: digest('4'),
            simulator_recipe_sha256: digest('5'),
            expected_simulator_calls: 2,
            reported_simulator_calls: 2,
            proposal_attempt_cap: 10,
            cost_unit: "cost_units".to_string(),
            run_status: status,
            attempts: vec![
                attempt(
                    1, "attempt-a", "candidate-a", digest('a'),
                    PreflightStatus::Passed, Some(finite), None, None,
                ),
                attempt(
                    2, "attempt-b", "candidate-b", digest('a'),
                    PreflightStatus::Rejected, None, Some("minimum-feature-rule"), None,
                ),
                attempt(
                    3, "attempt-c", "candidate-c", digest('c'),
                    PreflightStatus::Passed, Some(failed), None, None,
                ),
            ],
        }
    }

    fn serialized(ledger: &SimulatorBudgetLedger) -> Vec<u8> {
        serde_json::to_vec(ledger).expect("test ledger serializes")
    }

    #[test]
    fn equal_budget_counts_failed_simulator_call_and_preflight_reject_separately() {
        let ledger = sample_ledger(RunStatus::Completed);
        let audit = audit_json_ledger(&serialized(&ledger)).expect("ledger audits");
        assert_eq!(audit.expected_simulator_calls, 2);
        assert_eq!(audit.observed_simulator_calls, 2);
        assert_eq!(audit.proposal_attempts, 3);
        assert_eq!(audit.preflight_rejected, 1);
        assert_eq!(audit.returned_finite_results, 1);
        assert_eq!(audit.simulator_failures, 1);
        assert_eq!(audit.disposition, AuditDisposition::EqualBudgetComplete);
        assert!(audit.audit_pass);
    }

    #[test]
    fn duplicate_candidate_geometry_is_counted_not_silently_dropped() {
        let ledger = sample_ledger(RunStatus::Completed);
        let audit = audit_json_ledger(&serialized(&ledger)).expect("ledger audits");
        assert_eq!(audit.generated_candidates, 3);
        assert_eq!(audit.unique_candidate_digests, 2);
        assert_eq!(audit.repeated_candidate_attempts, 1);
        assert_eq!(audit.observed_simulator_calls, 2);
    }

    #[test]
    fn reported_finite_row_count_cannot_substitute_for_actual_calls() {
        let mut ledger = sample_ledger(RunStatus::Completed);
        ledger.reported_simulator_calls = 1; // historical n_meep may count finite rows only
        let audit = audit_json_ledger(&serialized(&ledger)).expect("ledger audits");
        assert_eq!(audit.reported_simulator_calls, 1);
        assert_eq!(audit.observed_simulator_calls, 2);
        assert!(!audit.reported_count_matches_observed);
        assert!(!audit.audit_pass);
    }

    #[test]
    fn too_few_actual_calls_is_incomplete_even_if_all_returned_rows_are_finite() {
        let mut ledger = sample_ledger(RunStatus::Incomplete);
        ledger.attempts.pop();
        ledger.reported_simulator_calls = 1;
        let audit = audit_json_ledger(&serialized(&ledger)).expect("ledger audits");
        assert_eq!(audit.observed_simulator_calls, 1);
        assert_eq!(audit.disposition, AuditDisposition::IncompleteUnderBudget);
        assert!(!audit.exact_call_budget_match);
        assert!(!audit.audit_pass);
    }

    #[test]
    fn extra_call_is_over_budget_even_when_run_claims_completion() {
        let mut ledger = sample_ledger(RunStatus::Completed);
        let extra = call(
            "invocation-extra",
            SimulatorTerminalStatus::ReturnedFiniteResult,
            Some(digest('9')),
            1.0,
        );
        ledger.attempts.push(attempt(
            4, "attempt-extra", "candidate-extra", digest('9'),
            PreflightStatus::Passed, Some(extra), None, None,
        ));
        ledger.reported_simulator_calls = 3;
        let audit = audit_json_ledger(&serialized(&ledger)).expect("ledger audits");
        assert_eq!(audit.observed_simulator_calls, 3);
        assert_eq!(audit.disposition, AuditDisposition::OverBudget);
        assert!(!audit.audit_pass);
    }

    #[test]
    fn unresolved_started_call_consumes_budget_but_blocks_completion() {
        let mut ledger = sample_ledger(RunStatus::Incomplete);
        ledger.attempts[2].simulator_call = Some(call(
            "invocation-c",
            SimulatorTerminalStatus::Interrupted,
            None,
            2.0,
        ));
        let audit = audit_json_ledger(&serialized(&ledger)).expect("ledger audits");
        assert_eq!(audit.observed_simulator_calls, 2);
        assert_eq!(audit.unresolved_simulator_calls, 1);
        assert_eq!(audit.disposition, AuditDisposition::BudgetConsumedButUnresolved);
        assert!(!audit.audit_pass);
    }

    #[test]
    fn accepted_candidate_without_invocation_is_not_counted_as_call() {
        let mut ledger = sample_ledger(RunStatus::Completed);
        ledger.attempts[2].simulator_call = None;
        ledger.attempts[2].not_called_reason = Some(NotCalledReason::LaunchFailed);
        ledger.reported_simulator_calls = 1;
        let audit = audit_json_ledger(&serialized(&ledger)).expect("ledger audits");
        assert_eq!(audit.observed_simulator_calls, 1);
        assert_eq!(audit.eligible_candidates_not_called, 1);
        assert_eq!(audit.disposition, AuditDisposition::IncompleteUnderBudget);
        assert!(!audit.audit_pass);
    }

    #[test]
    fn completed_run_cannot_have_proposals_after_the_budget_is_reached() {
        let mut ledger = sample_ledger(RunStatus::Completed);
        ledger.attempts.push(attempt(
            4, "attempt-after-budget", "candidate-after", digest('8'),
            PreflightStatus::Rejected, None, Some("minimum-feature-rule"), None,
        ));
        let audit = audit_json_ledger(&serialized(&ledger)).expect("ledger audits");
        assert_eq!(audit.proposal_attempts_after_budget_reached, 1);
        assert_eq!(audit.disposition, AuditDisposition::CompletedClaimMismatch);
        assert!(!audit.audit_pass);
    }

    #[test]
    fn generation_failure_is_a_proposal_attempt_but_not_a_candidate_or_sim_call() {
        let mut ledger = sample_ledger(RunStatus::Incomplete);
        ledger.expected_simulator_calls = 3;
        ledger.attempts.insert(0, ProposalAttempt {
            sequence: 1,
            attempt_id: "generation-failed".to_string(),
            candidate_id: None,
            candidate_sha256: None,
            proposal_status: ProposalStatus::GenerationFailed,
            preflight_status: PreflightStatus::NotRun,
            attempt_reason: Some("decoder-error".to_string()),
            not_called_reason: None,
            proposal_cost: 0.05,
            proposal_elapsed_ms: 5,
            simulator_call: None,
        });
        for (index, item) in ledger.attempts.iter_mut().enumerate() { item.sequence = index + 1; }
        let audit = audit_json_ledger(&serialized(&ledger)).expect("ledger audits");
        assert_eq!(audit.proposal_attempts, 4);
        assert_eq!(audit.generation_failures, 1);
        assert_eq!(audit.observed_simulator_calls, 2);
        assert_eq!(audit.disposition, AuditDisposition::IncompleteUnderBudget);
    }

    #[test]
    fn rejects_non_contiguous_attempt_sequences() {
        let mut ledger = sample_ledger(RunStatus::Completed);
        ledger.attempts[1].sequence = 3;
        let error = audit_json_ledger(&serialized(&ledger)).unwrap_err();
        assert!(error.contains("contiguous"));
    }

    #[test]
    fn rejects_simulator_call_on_preflight_rejection() {
        let mut ledger = sample_ledger(RunStatus::Completed);
        ledger.attempts[1].simulator_call = Some(call(
            "invalid-call",
            SimulatorTerminalStatus::ReturnedFiniteResult,
            Some(digest('9')),
            1.0,
        ));
        let error = audit_json_ledger(&serialized(&ledger)).unwrap_err();
        assert!(error.contains("preflight-rejected"));
    }

    #[test]
    fn rejects_returned_result_without_artifact_digest() {
        let mut ledger = sample_ledger(RunStatus::Completed);
        ledger.attempts[0].simulator_call.as_mut().unwrap().result_artifact_sha256 = None;
        let error = audit_json_ledger(&serialized(&ledger)).unwrap_err();
        assert!(error.contains("lacks result_artifact_sha256"));
    }

    #[test]
    fn rejects_unknown_fields_fail_closed() {
        let serialized = String::from_utf8(serialized(&sample_ledger(RunStatus::Completed))).unwrap();
        let mutated = serialized.replacen(
            "\"expected_simulator_calls\":2",
            "\"expected_simulator_calls\":2,\"hidden_target\":0.5",
            1,
        );
        let error = audit_json_ledger(mutated.as_bytes()).unwrap_err();
        assert!(error.contains("unknown field"));
    }

    #[test]
    fn failed_and_nonfinite_calls_still_consume_budget() {
        let mut ledger = sample_ledger(RunStatus::Completed);
        ledger.attempts[0].simulator_call = Some(call(
            "invocation-a",
            SimulatorTerminalStatus::ReturnedNonfiniteResult,
            Some(digest('f')),
            1.0,
        ));
        let audit = audit_json_ledger(&serialized(&ledger)).expect("ledger audits");
        assert_eq!(audit.observed_simulator_calls, 2);
        assert_eq!(audit.returned_finite_results, 0);
        assert_eq!(audit.returned_nonfinite_results, 1);
        assert_eq!(audit.simulator_failures, 1);
        assert!(audit.audit_pass);
    }

    #[test]
    fn cost_includes_proposal_attempts_and_failed_simulator_calls() {
        let ledger = sample_ledger(RunStatus::Completed);
        let audit = audit_json_ledger(&serialized(&ledger)).expect("ledger audits");
        assert!((audit.total_proposal_cost - 0.3).abs() < 1e-12);
        assert!((audit.total_simulator_cost - 3.0).abs() < 1e-12);
        assert!((audit.total_cost - 3.3).abs() < 1e-12);
    }

    #[test]
    fn rejects_missing_nullable_key_instead_of_silently_defaulting_to_null() {
        let raw = String::from_utf8(serialized(&sample_ledger(RunStatus::Completed))).unwrap();
        let mutated = raw.replacen("\\"not_called_reason\\":null,", "", 1);
        assert_ne!(mutated, raw, "fixture must contain the nullable key");
        let error = audit_json_ledger(mutated.as_bytes()).unwrap_err();
        assert!(error.contains("missing field") || error.contains("not_called_reason"));
    }

    #[test]
    fn audit_binds_report_to_exact_input_bytes() {
        let ledger = sample_ledger(RunStatus::Completed);
        let mut raw = serde_json::to_vec_pretty(&ledger).unwrap();
        raw.push(b'\n');
        let audit = audit_json_ledger(&raw).expect("ledger audits");
        let expected = Sha256::digest(&raw);
        let expected_hex: String = expected.iter().map(|byte| format!("{byte:02x}")).collect();
        assert_eq!(audit.ledger_sha256, expected_hex);
    }
}
