//! Joint finite-grid calibration of two reasoning-stopping risks.
//!
//! The upper channel measures wrong answers at an upper-threshold exit.
//! The lower channel measures the fraction of future checkpoints that would
//! be correct when a lower-threshold rule abandons a task.
//!
//! This is a Symthaea joint-grid variant inspired by Conformal Thinking. It
//! evaluates all candidates and uses one simultaneous Hoeffding correction
//! over 2 * grid_size bounded losses. It does not reproduce the paper's
//! sequential two-stage threshold fitting procedure.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const PPM: u64 = 1_000_000;
const MAX_CANDIDATES: usize = 100_000;
const MAX_TRACES: usize = 1_000_000;
const MAX_STEPS: usize = 100_000;
const PLAN_DOMAIN: &[u8] = b"SYMTHAEA-COMPUTE-RISK-GRID-V0.1";
const RECEIPT_DOMAIN: &[u8] = b"SYMTHAEA-COMPUTE-RISK-RECEIPT-V0.1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeRiskError {
    pub field: String,
    pub message: String,
}
impl ComputeRiskError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }
}
impl std::fmt::Display for ComputeRiskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}
impl std::error::Error for ComputeRiskError {}
fn fail(field: impl Into<String>, message: impl Into<String>) -> ComputeRiskError {
    ComputeRiskError::new(field, message)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationDesign {
    /// Caller declares independent, identically distributed held-out tasks.
    /// The code cannot verify this design assumption.
    IidHeldOutTasksDeclared,
    /// A fixed synthetic/stratified panel. UCB values are diagnostics only.
    SyntheticFixedPanel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopCause {
    UpperThreshold,
    LowerThreshold,
    FullBudget,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReasoningStep {
    pub token_count: u64,
    /// Normalized confidence-like score in [0, 1], represented as integer PPM.
    pub confidence_ppm: u32,
    /// Evaluator-known truth: whether the current candidate answer is correct.
    pub correct: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReasoningTrace {
    pub task_id: String,
    pub budget_tokens: u64,
    pub signal_id: String,
    pub steps: Vec<ReasoningStep>,
}

impl ReasoningTrace {
    pub fn validate(&self) -> Result<(), ComputeRiskError> {
        if self.task_id.trim().is_empty() || self.signal_id.trim().is_empty() {
            return Err(fail("trace_identity", "task_id and signal_id must be non-empty"));
        }
        if self.budget_tokens == 0 || self.budget_tokens > 1_000_000_000_000 {
            return Err(fail("budget_tokens", "must be in 1..=1e12"));
        }
        if self.steps.len() < 2 || self.steps.len() > MAX_STEPS {
            return Err(fail("steps", format!("must contain 2..={MAX_STEPS} checkpoints")));
        }
        let mut previous = 0;
        for (index, step) in self.steps.iter().enumerate() {
            if step.token_count == 0 || step.token_count <= previous {
                return Err(fail(format!("steps[{index}].token_count"), "token counts must be positive and strictly increasing"));
            }
            if step.token_count > self.budget_tokens {
                return Err(fail(format!("steps[{index}].token_count"), "checkpoint exceeds the hard token budget"));
            }
            if step.confidence_ppm > PPM as u32 {
                return Err(fail(format!("steps[{index}].confidence_ppm"), "confidence must be in [0, 1] PPM"));
            }
            previous = step.token_count;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StoppingCandidate {
    pub candidate_id: String,
    pub signal_id: String,
    /// Upper threshold lambda_plus. The upper rule exits when confidence >= this.
    pub upper_threshold_ppm: u32,
    /// Lower asymptote l for the dynamic lower threshold.
    pub lower_floor_ppm: u32,
    /// Sigmoid slope c, in PPM per token. 500 means 0.0005 per token.
    pub lower_slope_ppm_per_token: u32,
    /// Shift s in [0,1], expressed in PPM of the hard token budget.
    pub lower_shift_fraction_ppm: u32,
}

impl StoppingCandidate {
    pub fn validate(&self) -> Result<(), ComputeRiskError> {
        if self.candidate_id.trim().is_empty() || self.signal_id.trim().is_empty() {
            return Err(fail("candidate_identity", "candidate_id and signal_id must be non-empty"));
        }
        if self.upper_threshold_ppm == 0 || self.upper_threshold_ppm > PPM as u32 {
            return Err(fail("upper_threshold_ppm", "must be in 1..=1,000,000"));
        }
        if self.lower_floor_ppm >= self.upper_threshold_ppm {
            return Err(fail("lower_floor_ppm", "must be strictly below the upper threshold"));
        }
        if self.lower_slope_ppm_per_token > 100_000 {
            return Err(fail("lower_slope_ppm_per_token", "slope exceeds the configured safe limit"));
        }
        if self.lower_shift_fraction_ppm > PPM as u32 {
            return Err(fail("lower_shift_fraction_ppm", "must be in [0, 1,000,000]"));
        }
        Ok(())
    }

    /// Dynamic lower threshold:
    /// l + (lambda_plus-l)/(1 + exp(-c*(tokens-s*budget))).
    ///
    /// The logistic result is rounded to the nearest integer PPM. A one-PPM
    /// gap is preserved so the upper and lower conditions cannot overlap.
    pub fn lower_threshold_at(&self, token_count: u64, budget_tokens: u64) -> u32 {
        let shift = (self.lower_shift_fraction_ppm as f64 / PPM as f64) * budget_tokens as f64;
        let z = (self.lower_slope_ppm_per_token as f64 / PPM as f64)
            * (token_count as f64 - shift);
        let z = z.clamp(-40.0, 40.0);
        let logistic = 1.0 / (1.0 + (-z).exp());
        let span = (self.upper_threshold_ppm - self.lower_floor_ppm) as f64;
        let raw = self.lower_floor_ppm as f64 + span * logistic;
        let rounded = raw.round().clamp(0.0, PPM as f64) as u32;
        rounded.min(self.upper_threshold_ppm.saturating_sub(1))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ComputeRiskFixture {
    pub schema: String,
    pub version: String,
    pub status: String,
    pub signal_id: String,
    pub task_count: usize,
    pub budget_tokens: u64,
    pub seed: u64,
    pub risk_upper_budget_ppm: u32,
    pub risk_lower_budget_ppm: u32,
    pub confidence_delta_ppm: u32,
    pub candidates: Vec<StoppingCandidate>,
    pub generator_id: String,
    pub scenario_ids: Vec<String>,
}

impl ComputeRiskFixture {
    pub fn validate(&self) -> Result<(), ComputeRiskError> {
        if self.schema != "symthaea.compute_risk_joint_grid" || self.version != "0.1.0" {
            return Err(fail("schema_version", "unsupported compute-risk fixture schema/version"));
        }
        if self.status != "synthetic_fixed_panel_not_iid_no_confidence_claim" {
            return Err(fail("status", "v0.1 fixture must be explicitly marked synthetic-only"));
        }
        if self.signal_id.trim().is_empty() || self.generator_id.trim().is_empty() {
            return Err(fail("fixture_identity", "signal_id and generator_id must be non-empty"));
        }
        if self.task_count < 2 || self.task_count > MAX_TRACES {
            return Err(fail("task_count", format!("must be in 2..={MAX_TRACES}")));
        }
        if self.budget_tokens == 0 || self.budget_tokens > 1_000_000_000_000 {
            return Err(fail("budget_tokens", "must be in 1..=1e12"));
        }
        if self.risk_upper_budget_ppm == 0 || self.risk_upper_budget_ppm > PPM as u32
            || self.risk_lower_budget_ppm == 0 || self.risk_lower_budget_ppm > PPM as u32 {
            return Err(fail("risk_budget", "both operational risk budgets must be in 1..=1,000,000 PPM"));
        }
        if self.confidence_delta_ppm == 0 || self.confidence_delta_ppm >= PPM as u32 {
            return Err(fail("confidence_delta_ppm", "must be in (0, 1,000,000) PPM"));
        }
        if self.candidates.is_empty() || self.candidates.len() > MAX_CANDIDATES {
            return Err(fail("candidates", "candidate grid must be non-empty and bounded"));
        }
        if self.scenario_ids.is_empty() {
            return Err(fail("scenario_ids", "at least one synthetic scenario is required"));
        }
        let mut previous: Option<&str> = None;
        let mut ids = BTreeSet::new();
        for candidate in &self.candidates {
            candidate.validate()?;
            if candidate.signal_id != self.signal_id {
                return Err(fail("candidate.signal_id", "all candidates must use the fixture's preregistered signal"));
            }
            if let Some(prior) = previous {
                if prior >= candidate.candidate_id.as_str() {
                    return Err(fail("candidates", "candidate IDs must be unique and strictly sorted"));
                }
            }
            previous = Some(&candidate.candidate_id);
            if !ids.insert(candidate.candidate_id.as_str()) {
                return Err(fail("candidates", "duplicate candidate ID"));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateTaskLoss {
    pub task_id: String,
    pub upper_wrong_answer_loss_ppm: u32,
    pub lower_premature_abandon_loss_ppm: u32,
    pub tokens_used: u64,
    pub stop_index: usize,
    pub stop_cause: StopCause,
    pub upper_efficiency_loss_ppm: u32,
    pub lower_efficiency_loss_ppm: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateEvaluation {
    pub candidate_id: String,
    pub upper_loss_sum_ppm: u128,
    pub lower_loss_sum_ppm: u128,
    pub sample_count: usize,
    pub upper_mean_loss_ceil_ppm: u32,
    pub lower_mean_loss_ceil_ppm: u32,
    pub simultaneous_ucb_correction_ppm: u32,
    pub upper_risk_ucb_ppm: u32,
    pub lower_risk_ucb_ppm: u32,
    pub mean_tokens_used_ppm: u32,
    pub mean_upper_efficiency_loss_ppm: u32,
    pub mean_lower_efficiency_loss_ppm: u32,
    pub meets_both_risk_budgets: bool,
    pub task_loss_receipt_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionStatus {
    FeasibleCandidateFoundOnValidationPanel,
    NoCandidateMeetsBothRiskBudgets,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputeRiskReport {
    pub schema: String,
    pub design: CalibrationDesign,
    pub confidence_claim_eligible_under_declared_design: bool,
    pub simultaneous_claim_count: usize,
    pub confidence_delta_ppm: u32,
    pub risk_upper_budget_ppm: u32,
    pub risk_lower_budget_ppm: u32,
    pub correction_ppm: u32,
    pub sample_count: usize,
    pub candidate_grid_sha256: String,
    pub validation_panel_sha256: String,
    pub candidate_evaluations: Vec<CandidateEvaluation>,
    pub selected_candidate_id: Option<String>,
    pub selection_status: SelectionStatus,
    pub formal_theorem_verified: bool,
    pub claim_ceiling: String,
    pub report_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputeRiskExperimentReceipt {
    pub schema: String,
    pub status: String,
    pub source_commit: String,
    pub fixture_sha256: String,
    pub validation_panel_sha256: String,
    pub report: ComputeRiskReport,
    pub receipt_sha256: String,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn is_git_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(output, "{byte:02x}");
    }
    output
}
fn domain_digest(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(output, "{byte:02x}");
    }
    output
}
fn json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, ComputeRiskError> {
    serde_json::to_vec(value).map_err(|e| fail("serialization", e.to_string()))
}
fn ceil_div(numerator: u128, denominator: u128) -> Result<u128, ComputeRiskError> {
    if denominator == 0 {
        return Err(fail("division", "denominator must be positive"));
    }
    let q = numerator / denominator;
    let r = numerator % denominator;
    if r == 0 { Ok(q) } else { q.checked_add(1).ok_or_else(|| fail("division", "ceil division overflow")) }
}

pub fn evaluate_candidate_on_trace(
    candidate: &StoppingCandidate,
    trace: &ReasoningTrace,
) -> Result<CandidateTaskLoss, ComputeRiskError> {
    candidate.validate()?;
    trace.validate()?;
    if candidate.signal_id != trace.signal_id {
        return Err(fail("signal_id", "candidate and trace use different confidence signals"));
    }

    let mut stop_index = trace.steps.len() - 1;
    let mut stop_cause = StopCause::FullBudget;
    for (index, step) in trace.steps.iter().enumerate() {
        let lower = candidate.lower_threshold_at(step.token_count, trace.budget_tokens);
        if step.confidence_ppm >= candidate.upper_threshold_ppm {
            stop_index = index;
            stop_cause = StopCause::UpperThreshold;
            break;
        }
        if step.confidence_ppm <= lower {
            stop_index = index;
            stop_cause = StopCause::LowerThreshold;
            break;
        }
    }

    let upper_wrong_answer_loss_ppm = if stop_cause == StopCause::UpperThreshold
        && !trace.steps[stop_index].correct { PPM as u32 } else { 0 };

    let lower_premature_abandon_loss_ppm = if stop_cause == StopCause::LowerThreshold {
        let future = &trace.steps[stop_index..];
        let correct = future.iter().filter(|step| step.correct).count() as u128;
        let count = future.len() as u128;
        u32::try_from(ceil_div(correct * u128::from(PPM), count)?)
            .map_err(|_| fail("lower_loss", "PPM conversion overflow"))?
    } else {
        0
    };

    let first_correct = trace.steps.iter().position(|step| step.correct);
    let upper_efficiency_loss_ppm = if stop_cause == StopCause::UpperThreshold {
        if let Some(first_correct_index) = first_correct {
            let delay = stop_index.saturating_sub(first_correct_index) as u128;
            u32::try_from(ceil_div(delay * u128::from(PPM), trace.steps.len() as u128)?)
                .map_err(|_| fail("upper_efficiency", "PPM conversion overflow"))?
        } else { 0 }
    } else { 0 };

    let lower_efficiency_loss_ppm = if stop_cause == StopCause::LowerThreshold {
        let incorrect_prefix = trace.steps[..=stop_index].iter().filter(|step| !step.correct).count() as u128;
        u32::try_from(ceil_div(incorrect_prefix * u128::from(PPM), trace.steps.len() as u128)?)
            .map_err(|_| fail("lower_efficiency", "PPM conversion overflow"))?
    } else { 0 };

    let tokens_used = trace.steps[stop_index].token_count;
    Ok(CandidateTaskLoss {
        task_id: trace.task_id.clone(),
        upper_wrong_answer_loss_ppm,
        lower_premature_abandon_loss_ppm,
        tokens_used,
        stop_index,
        stop_cause,
        upper_efficiency_loss_ppm,
        lower_efficiency_loss_ppm,
    })
}

fn simultaneous_hoeffding_correction_ppm(
    candidate_count: usize,
    sample_count: usize,
    confidence_delta_ppm: u32,
) -> Result<u32, ComputeRiskError> {
    if candidate_count == 0 || sample_count == 0 || confidence_delta_ppm == 0
        || confidence_delta_ppm >= PPM as u32 {
        return Err(fail("hoeffding", "grid, sample count, or delta is invalid"));
    }
    // 2G one-sided claims: one upper-risk and one lower-risk statement per
    // candidate. Equal Bonferroni allocation gives delta/(2G) per claim.
    // One extra PPM is added after ceil to avoid optimistic output rounding.
    let claims = (candidate_count as f64) * 2.0;
    let delta = confidence_delta_ppm as f64 / PPM as f64;
    let log_term = (claims / delta).ln();
    let correction = (log_term / (2.0 * sample_count as f64)).sqrt();
    let correction_ppm = (correction * PPM as f64).ceil() as u64 + 1;
    Ok(correction_ppm.min(PPM) as u32)
}

fn aggregate_candidate(
    candidate: &StoppingCandidate,
    traces: &[ReasoningTrace],
    risk_upper_budget_ppm: u32,
    risk_lower_budget_ppm: u32,
    correction_ppm: u32,
) -> Result<CandidateEvaluation, ComputeRiskError> {
    let mut losses = Vec::with_capacity(traces.len());
    let mut upper_sum = 0u128;
    let mut lower_sum = 0u128;
    let mut tokens_fraction_sum = 0u128;
    let mut upper_efficiency_sum = 0u128;
    let mut lower_efficiency_sum = 0u128;

    for trace in traces {
        let loss = evaluate_candidate_on_trace(candidate, trace)?;
        upper_sum = upper_sum.checked_add(u128::from(loss.upper_wrong_answer_loss_ppm))
            .ok_or_else(|| fail("upper_sum", "overflow"))?;
        lower_sum = lower_sum.checked_add(u128::from(loss.lower_premature_abandon_loss_ppm))
            .ok_or_else(|| fail("lower_sum", "overflow"))?;
        let token_fraction = ceil_div(
            u128::from(loss.tokens_used) * u128::from(PPM),
            u128::from(trace.budget_tokens),
        )?;
        tokens_fraction_sum = tokens_fraction_sum.checked_add(token_fraction)
            .ok_or_else(|| fail("token_fraction_sum", "overflow"))?;
        upper_efficiency_sum = upper_efficiency_sum.checked_add(u128::from(loss.upper_efficiency_loss_ppm))
            .ok_or_else(|| fail("upper_efficiency_sum", "overflow"))?;
        lower_efficiency_sum = lower_efficiency_sum.checked_add(u128::from(loss.lower_efficiency_loss_ppm))
            .ok_or_else(|| fail("lower_efficiency_sum", "overflow"))?;
        losses.push(loss);
    }
    let n = traces.len() as u128;
    let upper_mean = u32::try_from(ceil_div(upper_sum, n)?)
        .map_err(|_| fail("upper_mean", "overflow"))?;
    let lower_mean = u32::try_from(ceil_div(lower_sum, n)?)
        .map_err(|_| fail("lower_mean", "overflow"))?;
    let upper_ucb = upper_mean.saturating_add(correction_ppm).min(PPM as u32);
    let lower_ucb = lower_mean.saturating_add(correction_ppm).min(PPM as u32);
    let mean_tokens = u32::try_from(ceil_div(tokens_fraction_sum, n)?)
        .map_err(|_| fail("mean_tokens", "overflow"))?;
    let mean_upper_efficiency = u32::try_from(ceil_div(upper_efficiency_sum, n)?)
        .map_err(|_| fail("mean_upper_efficiency", "overflow"))?;
    let mean_lower_efficiency = u32::try_from(ceil_div(lower_efficiency_sum, n)?)
        .map_err(|_| fail("mean_lower_efficiency", "overflow"))?;

    let loss_digest = domain_digest(RECEIPT_DOMAIN, &json_bytes(&losses)?);
    Ok(CandidateEvaluation {
        candidate_id: candidate.candidate_id.clone(),
        upper_loss_sum_ppm: upper_sum,
        lower_loss_sum_ppm: lower_sum,
        sample_count: traces.len(),
        upper_mean_loss_ceil_ppm: upper_mean,
        lower_mean_loss_ceil_ppm: lower_mean,
        simultaneous_ucb_correction_ppm: correction_ppm,
        upper_risk_ucb_ppm: upper_ucb,
        lower_risk_ucb_ppm: lower_ucb,
        mean_tokens_used_ppm: mean_tokens,
        mean_upper_efficiency_loss_ppm: mean_upper_efficiency,
        mean_lower_efficiency_loss_ppm: mean_lower_efficiency,
        meets_both_risk_budgets: upper_ucb <= risk_upper_budget_ppm && lower_ucb <= risk_lower_budget_ppm,
        task_loss_receipt_sha256: loss_digest,
    })
}

pub fn calibrate_joint_candidate_grid(
    candidates: &[StoppingCandidate],
    traces: &[ReasoningTrace],
    design: CalibrationDesign,
    risk_upper_budget_ppm: u32,
    risk_lower_budget_ppm: u32,
    confidence_delta_ppm: u32,
) -> Result<ComputeRiskReport, ComputeRiskError> {
    if candidates.is_empty() || candidates.len() > MAX_CANDIDATES {
        return Err(fail("candidates", "candidate grid must be non-empty and bounded"));
    }
    if traces.is_empty() || traces.len() > MAX_TRACES {
        return Err(fail("traces", "validation panel must be non-empty and bounded"));
    }
    if risk_upper_budget_ppm == 0 || risk_upper_budget_ppm > PPM as u32
        || risk_lower_budget_ppm == 0 || risk_lower_budget_ppm > PPM as u32 {
        return Err(fail("risk_budget", "both budgets must be in 1..=1,000,000 PPM"));
    }
    let mut ids = BTreeSet::new();
    let mut previous: Option<&str> = None;
    for candidate in candidates {
        candidate.validate()?;
        if let Some(prior) = previous {
            if prior >= candidate.candidate_id.as_str() {
                return Err(fail("candidates", "candidate grid must be uniquely sorted by candidate_id"));
            }
        }
        previous = Some(&candidate.candidate_id);
        if !ids.insert(candidate.candidate_id.as_str()) {
            return Err(fail("candidates", "duplicate candidate ID"));
        }
    }
    let expected_signal = candidates[0].signal_id.as_str();
    let mut task_ids = BTreeSet::new();
    for trace in traces {
        trace.validate()?;
        if trace.signal_id != expected_signal {
            return Err(fail("traces.signal_id", "all validation tasks must use the candidate grid's frozen signal"));
        }
        if !task_ids.insert(trace.task_id.as_str()) {
            return Err(fail("traces.task_id", "task IDs must be unique within a validation panel"));
        }
        if trace.budget_tokens != traces[0].budget_tokens {
            return Err(fail("traces.budget_tokens", "all tasks must share the same hard compute budget in v0.1"));
        }
    }
    if candidates.iter().any(|candidate| candidate.signal_id != expected_signal) {
        return Err(fail("candidates.signal_id", "candidate grid mixes different signal identities"));
    }

    let correction = simultaneous_hoeffding_correction_ppm(candidates.len(), traces.len(), confidence_delta_ppm)?;
    let mut evaluations = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        evaluations.push(aggregate_candidate(candidate, traces, risk_upper_budget_ppm, risk_lower_budget_ppm, correction)?);
    }

    let selected = evaluations.iter()
        .filter(|evaluation| evaluation.meets_both_risk_budgets)
        .min_by(|left, right| {
            left.mean_tokens_used_ppm.cmp(&right.mean_tokens_used_ppm)
                .then_with(|| left.candidate_id.cmp(&right.candidate_id))
        })
        .map(|evaluation| evaluation.candidate_id.clone());

    let candidate_json = json_bytes(&candidates)?;
    let trace_json = json_bytes(traces)?;
    let mut report = ComputeRiskReport {
        schema: "symthaea.compute_risk_joint_report.v0.1".to_string(),
        design,
        confidence_claim_eligible_under_declared_design: design == CalibrationDesign::IidHeldOutTasksDeclared,
        simultaneous_claim_count: candidates.len() * 2,
        confidence_delta_ppm,
        risk_upper_budget_ppm,
        risk_lower_budget_ppm,
        correction_ppm: correction,
        sample_count: traces.len(),
        candidate_grid_sha256: domain_digest(PLAN_DOMAIN, &candidate_json),
        validation_panel_sha256: domain_digest(RECEIPT_DOMAIN, &trace_json),
        candidate_evaluations: evaluations,
        selected_candidate_id: selected.clone(),
        selection_status: if selected.is_some() {
            SelectionStatus::FeasibleCandidateFoundOnValidationPanel
        } else {
            SelectionStatus::NoCandidateMeetsBothRiskBudgets
        },
        formal_theorem_verified: false,
        claim_ceiling: "simultaneous Hoeffding UCB arithmetic over the frozen finite candidate grid; any population-level statement is conditional on independent iid held-out task traces, bounded losses, and correct preregistration, none of which is verified by this runner".to_string(),
        report_sha256: String::new(),
    };
    report.report_sha256 = domain_digest(RECEIPT_DOMAIN, &json_bytes(&report)?);
    Ok(report)
}

pub fn verify_compute_risk_report(report: &ComputeRiskReport) -> Result<(), ComputeRiskError> {
    if report.schema != "symthaea.compute_risk_joint_report.v0.1" {
        return Err(fail("report.schema", "unsupported report schema"));
    }
    if report.formal_theorem_verified {
        return Err(fail("formal_theorem_verified", "v0.1 cannot claim theorem verification"));
    }
    if report.report_sha256.len() != 64 || !is_sha256(&report.report_sha256) {
        return Err(fail("report_sha256", "must be lowercase hexadecimal SHA-256"));
    }
    let mut copy = report.clone();
    let recorded = copy.report_sha256.clone();
    copy.report_sha256.clear();
    let expected = domain_digest(RECEIPT_DOMAIN, &json_bytes(&copy)?);
    if recorded != expected {
        return Err(fail("report_sha256", "does not match report contents"));
    }
    Ok(())
}

pub const BUILTIN_FIXTURE_JSON: &str = include_str!("../fixtures/compute_risk_joint_grid_v0.1.json");
pub fn load_builtin_fixture() -> Result<ComputeRiskFixture, ComputeRiskError> {
    let fixture: ComputeRiskFixture = serde_json::from_str(BUILTIN_FIXTURE_JSON)
        .map_err(|e| fail("fixture", format!("JSON parse failed: {e}")))?;
    fixture.validate()?;
    Ok(fixture)
}

/// Deterministic fixed-panel generator. Scenarios are deliberately stratified;
/// this panel is not an iid sample and cannot support a population confidence claim.
pub fn generate_synthetic_validation_panel(
    fixture: &ComputeRiskFixture,
) -> Result<Vec<ReasoningTrace>, ComputeRiskError> {
    fixture.validate()?;
    let patterns: [(&str, usize, [u32; 8]); 8] = [
        ("early_solved", 2, [220_000, 360_000, 760_000, 880_000, 920_000, 950_000, 970_000, 980_000]),
        ("quickly_solved", 1, [180_000, 820_000, 900_000, 930_000, 950_000, 965_000, 975_000, 985_000]),
        ("delayed_solved", 6, [100_000, 160_000, 250_000, 360_000, 500_000, 700_000, 890_000, 950_000]),
        ("never_solved_low", usize::MAX, [90_000, 140_000, 190_000, 230_000, 260_000, 290_000, 320_000, 350_000]),
        ("overconfident_wrong", 7, [900_000, 920_000, 930_000, 945_000, 955_000, 965_000, 975_000, 985_000]),
        ("low_confidence_solved", 6, [80_000, 110_000, 140_000, 170_000, 210_000, 260_000, 420_000, 650_000]),
        ("oscillatory_delayed", 5, [100_000, 850_000, 320_000, 880_000, 450_000, 910_000, 940_000, 970_000]),
        ("never_solved_medium", usize::MAX, [160_000, 270_000, 350_000, 440_000, 500_000, 570_000, 610_000, 650_000]),
    ];
    if fixture.task_count % patterns.len() != 0 {
        return Err(fail("task_count", "synthetic stratified panel task_count must be divisible by eight"));
    }
    let mut traces = Vec::with_capacity(fixture.task_count);
    for task_index in 0..fixture.task_count {
        let scenario_index = task_index % patterns.len();
        let replicate_index = task_index / patterns.len();
        let (scenario_id, correct_from, base_confidence) = &patterns[scenario_index];
        let mut steps = Vec::with_capacity(8);
        for step_index in 0..8 {
            let jitter_seed = (task_index as u64)
                .wrapping_mul(37)
                .wrapping_add((step_index as u64).wrapping_mul(53))
                .wrapping_add(fixture.seed);
            let jitter = (jitter_seed % 21) as i64 - 10;
            let small_replicate = (replicate_index % 5) as i64 - 2;
            let confidence = (base_confidence[step_index] as i64 + jitter * 1_000 + small_replicate * 500)
                .clamp(0, PPM as i64) as u32;
            steps.push(ReasoningStep {
                token_count: (step_index as u64 + 1) * (fixture.budget_tokens / 8),
                confidence_ppm: confidence,
                correct: *correct_from != usize::MAX && step_index >= *correct_from,
            });
        }
        let trace = ReasoningTrace {
            task_id: format!("synthetic-{}-{:04}", scenario_id, task_index),
            budget_tokens: fixture.budget_tokens,
            signal_id: fixture.signal_id.clone(),
            steps,
        };
        trace.validate()?;
        traces.push(trace);
    }
    Ok(traces)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: &str, upper: u32, floor: u32, slope: u32, shift: u32) -> StoppingCandidate {
        StoppingCandidate {
            candidate_id: id.into(),
            signal_id: "confidence_v1".into(),
            upper_threshold_ppm: upper,
            lower_floor_ppm: floor,
            lower_slope_ppm_per_token: slope,
            lower_shift_fraction_ppm: shift,
        }
    }

    fn trace(correct: &[bool], confidence: &[u32]) -> ReasoningTrace {
        ReasoningTrace {
            task_id: "task".into(),
            budget_tokens: 8_000,
            signal_id: "confidence_v1".into(),
            steps: correct.iter().zip(confidence).enumerate().map(|(i,(correct,confidence))| ReasoningStep {
                token_count: (i as u64+1)*1_000, confidence_ppm:*confidence, correct:*correct
            }).collect()
        }
    }

    #[test]
    fn upper_threshold_counts_wrong_emission_only() {
        let c=candidate("upper",900_000,100_000,500,300_000);
        let t=trace(&[false,false,true,true,true,true,true,true], &[200_000,920_000,930_000,940_000,950_000,960_000,970_000,980_000]);
        let loss=evaluate_candidate_on_trace(&c,&t).unwrap();
        assert_eq!(loss.stop_cause,StopCause::UpperThreshold);
        assert_eq!(loss.stop_index,1);
        assert_eq!(loss.upper_wrong_answer_loss_ppm,1_000_000);
        assert_eq!(loss.lower_premature_abandon_loss_ppm,0);
    }

    #[test]
    fn upper_threshold_correct_answer_has_zero_wrong_answer_loss() {
        let c=candidate("upper",800_000,100_000,300,500_000);
        let t=trace(&[false,true,true,true,true,true,true,true], &[100_000,850_000,900_000,930_000,950_000,960_000,970_000,980_000]);
        let loss=evaluate_candidate_on_trace(&c,&t).unwrap();
        assert_eq!(loss.stop_cause,StopCause::UpperThreshold);
        assert_eq!(loss.upper_wrong_answer_loss_ppm,0);
    }

    #[test]
    fn lower_threshold_loss_uses_future_correct_checkpoints() {
        let c=candidate("lower",990_000,500_000,300,700_000);
        let t=trace(&[false,false,false,true,true,true,true,true], &[100_000,100_000,100_000,100_000,100_000,100_000,100_000,100_000]);
        let loss=evaluate_candidate_on_trace(&c,&t).unwrap();
        assert_eq!(loss.stop_cause,StopCause::LowerThreshold);
        assert!(loss.lower_premature_abandon_loss_ppm>0);
        assert_eq!(loss.upper_wrong_answer_loss_ppm,0);
    }

    #[test]
    fn full_budget_fallback_has_no_early_stop_loss() {
        let c=candidate("late",1_000_000,0,0,500_000);
        let t=trace(&[false,false,false,false,false,false,false,false], &[100_000,200_000,300_000,400_000,500_000,600_000,700_000,800_000]);
        let loss=evaluate_candidate_on_trace(&c,&t).unwrap();
        assert_eq!(loss.stop_cause,StopCause::FullBudget);
        assert_eq!(loss.tokens_used,8_000);
        assert_eq!(loss.upper_wrong_answer_loss_ppm,0);
        assert_eq!(loss.lower_premature_abandon_loss_ppm,0);
    }

    #[test]
    fn lower_threshold_is_bounded_and_strictly_below_upper() {
        let c=candidate("lower",900_000,200_000,1_000,500_000);
        for tokens in [1,1_000,4_000,8_000] {
            let threshold=c.lower_threshold_at(tokens,8_000);
            assert!(threshold>=c.lower_floor_ppm);
            assert!(threshold<c.upper_threshold_ppm);
        }
    }

    #[test]
    fn task_budget_must_not_be_exceeded() {
        let mut t=trace(&[false,true], &[100_000,200_000]);
        t.steps[1].token_count=9_000;
        assert!(t.validate().is_err());
    }

    #[test]
    fn simultaneous_correction_increases_with_grid_size_and_delta_tightness() {
        let small=simultaneous_hoeffding_correction_ppm(2,256,100_000).unwrap();
        let large=simultaneous_hoeffding_correction_ppm(8,256,100_000).unwrap();
        let tighter=simultaneous_hoeffding_correction_ppm(2,256,10_000).unwrap();
        assert!(large>small);
        assert!(tighter>small);
    }

    #[test]
    fn complete_grid_is_evaluated_without_monotonicity_assumption() {
        let candidates=vec![
            candidate("a",700_000,100_000,100,400_000),
            candidate("b",800_000,150_000,200,500_000),
            candidate("c",900_000,200_000,300,600_000),
        ];
        let traces=(0..64).map(|i| {
            let mut t=trace(&[false,false,true,true,true,true,true,true], &[200_000,300_000,500_000,800_000,850_000,900_000,930_000,950_000]);
            t.task_id=format!("t{i:03}");
            t
        }).collect::<Vec<_>>();
        let report=calibrate_joint_candidate_grid(&candidates,&traces,CalibrationDesign::IidHeldOutTasksDeclared,400_000,400_000,100_000).unwrap();
        assert_eq!(report.candidate_evaluations.len(),3);
        assert_eq!(report.simultaneous_claim_count,6);
        assert!(report.candidate_evaluations.iter().all(|x|x.sample_count==64));
    }

    #[test]
    fn no_candidate_meets_budgets_is_explicit_not_silently_relaxed() {
        let candidates=vec![candidate("a",500_000,100_000,500,400_000)];
        let traces=(0..16).map(|i| {
            let mut t=trace(&[false,false,false,false,true,true,true,true], &[900_000,900_000,900_000,900_000,900_000,900_000,900_000,900_000]);
            t.task_id=format!("t{i:03}");
            t
        }).collect::<Vec<_>>();
        let report=calibrate_joint_candidate_grid(&candidates,&traces,CalibrationDesign::IidHeldOutTasksDeclared,10_000,10_000,50_000).unwrap();
        assert_eq!(report.selected_candidate_id,None);
        assert_eq!(report.selection_status,SelectionStatus::NoCandidateMeetsBothRiskBudgets);
    }

    #[test]
    fn validation_trace_ids_must_be_unique() {
        let candidates=vec![candidate("a",900_000,100_000,300,500_000)];
        let t=trace(&[false,true,true,true,true,true,true,true], &[100_000,850_000,900_000,930_000,950_000,960_000,970_000,980_000]);
        assert!(calibrate_joint_candidate_grid(&candidates,&[t.clone(),t],CalibrationDesign::SyntheticFixedPanel,500_000,500_000,100_000).is_err());
    }

    #[test]
    fn report_digest_detects_mutation() {
        let candidates=vec![candidate("a",900_000,100_000,300,500_000)];
        let traces=(0..16).map(|i| {
            let mut t=trace(&[false,true,true,true,true,true,true,true], &[100_000,850_000,900_000,930_000,950_000,960_000,970_000,980_000]);
            t.task_id=format!("t{i:03}"); t
        }).collect::<Vec<_>>();
        let report=calibrate_joint_candidate_grid(&candidates,&traces,CalibrationDesign::SyntheticFixedPanel,500_000,500_000,100_000).unwrap();
        verify_compute_risk_report(&report).unwrap();
        let mut tampered=report.clone(); tampered.correction_ppm+=1;
        assert!(verify_compute_risk_report(&tampered).is_err());
    }

    #[test]
    fn fixture_generates_stratified_non_iid_panel() {
        let fixture=load_builtin_fixture().unwrap();
        let traces=generate_synthetic_validation_panel(&fixture).unwrap();
        assert_eq!(traces.len(),fixture.task_count);
        assert!(traces.iter().all(|t|t.steps.len()==8));
    }

    #[test]
    fn sorted_unique_candidate_grid_required() {
        let candidates=vec![
            candidate("b",800_000,100_000,200,500_000),
            candidate("a",900_000,100_000,300,500_000),
        ];
        let traces=vec![trace(&[false,true,true,true,true,true,true,true], &[100_000,850_000,900_000,930_000,950_000,960_000,970_000,980_000])];
        assert!(calibrate_joint_candidate_grid(&candidates,&traces,CalibrationDesign::SyntheticFixedPanel,500_000,500_000,100_000).is_err());
    }
}

