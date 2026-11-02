//! Joint finite-grid risk control for dual-threshold reasoning compute.
//!
//! The calibration unit is one complete reasoning trajectory. Steps inside a
//! trajectory are deliberately not treated as independent observations.
//! Every frozen candidate policy is evaluated on every calibration trajectory.
//! One-sided Hoeffding bounds are simultaneous over both risks and all candidates.
//!
//! This is an independent reference implementation of the loss definitions in
//! Conformal Thinking (ICML 2026), not a full reproduction and not a shift-robust
//! or anytime guarantee.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const PPM: u64 = 1_000_000;
const FIXTURE_DOMAIN: &[u8] = b"SYMTHAEA-COMPUTE-RISK-FIXTURE-V0.1";
const REPORT_DOMAIN: &[u8] = b"SYMTHAEA-COMPUTE-RISK-REPORT-V0.1";
const MAX_CANDIDATES: usize = 10_000;
const MAX_TRAJECTORIES: usize = 1_000_000;
const MAX_STEPS: usize = 100_000;

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PaperReference {
    pub paper_url: String,
    pub author_code_url: String,
    pub author_code_commit: String,
    pub author_code_status: String,
    pub method_note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CandidatePolicy {
    pub candidate_id: String,
    pub upper_threshold_ppm: u32,
    /// One preregistered lower threshold for each reasoning step.
    pub lower_threshold_by_step_ppm: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CalibrationTrajectory {
    pub trajectory_id: String,
    pub scenario_id: String,
    /// Stabilized confidence signal in [0, 1], represented as integer PPM.
    pub confidence_ppm: Vec<u32>,
    /// Whether each intermediate answer equals the ground-truth answer.
    pub correct_by_step: Vec<bool>,
    /// Cumulative tokens consumed at the end of each reasoning step.
    pub cumulative_tokens: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ComputeRiskFixture {
    pub schema: String,
    pub version: String,
    pub status: String,
    pub paper_reference: PaperReference,
    pub upper_risk_budget_ppm: u32,
    pub lower_risk_budget_ppm: u32,
    /// Total statistical failure probability allocated across 2 * M risk bounds.
    pub statistical_delta_ppm: u32,
    /// This complete policy set is frozen before calibration outcomes are inspected.
    pub candidate_policies: Vec<CandidatePolicy>,
    /// One independent/exchangeable unit is one complete trajectory, not one step.
    pub calibration_trajectories: Vec<CalibrationTrajectory>,
    pub notes: Vec<String>,
}

impl ComputeRiskFixture {
    pub fn horizon_steps(&self) -> Result<usize, ComputeRiskError> {
        self.calibration_trajectories.first()
            .map(|x| x.confidence_ppm.len())
            .ok_or_else(|| fail("calibration_trajectories", "must be non-empty"))
    }

    pub fn validate(&self) -> Result<(), ComputeRiskError> {
        if self.schema != "symthaea.compute_risk_simultaneous_ucb.v0.1" || self.version != "0.1.0" {
            return Err(fail("schema_version", "unsupported compute-risk fixture schema/version"));
        }
        if self.status != "synthetic_joint_grid_reference_not_a_model_or_distribution_shift_claim" {
            return Err(fail("status", "unsupported fixture status"));
        }
        for (field, value) in [
            ("paper_url", self.paper_reference.paper_url.as_str()),
            ("author_code_url", self.paper_reference.author_code_url.as_str()),
        ] {
            if !value.starts_with("https://") {
                return Err(fail(field, "must use HTTPS"));
            }
        }
        if self.paper_reference.author_code_commit.len() != 40
            || !self.paper_reference.author_code_commit.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
            return Err(fail("author_code_commit", "must be a 40-character lowercase Git SHA"));
        }
        if self.paper_reference.author_code_status.trim().is_empty()
            || self.paper_reference.method_note.trim().is_empty() {
            return Err(fail("paper_reference", "status and method note must be non-empty"));
        }
        for (field, value) in [
            ("upper_risk_budget_ppm", self.upper_risk_budget_ppm),
            ("lower_risk_budget_ppm", self.lower_risk_budget_ppm),
        ] {
            if value > PPM as u32 {
                return Err(fail(field, "must be in [0, 1,000,000]"));
            }
        }
        if self.statistical_delta_ppm == 0 || self.statistical_delta_ppm >= PPM as u32 {
            return Err(fail("statistical_delta_ppm", "must be in (0, 1,000,000)"));
        }
        if self.candidate_policies.is_empty() || self.candidate_policies.len() > MAX_CANDIDATES {
            return Err(fail("candidate_policies", "candidate count outside supported limits"));
        }
        let horizon = self.horizon_steps()?;
        if horizon == 0 || horizon > MAX_STEPS {
            return Err(fail("horizon_steps", "reasoning horizon outside supported limits"));
        }
        let mut ids = BTreeSet::new();
        let mut prior_id: Option<&str> = None;
        for (index, candidate) in self.candidate_policies.iter().enumerate() {
            if candidate.candidate_id.trim().is_empty() {
                return Err(fail(format!("candidate_policies[{index}].candidate_id"), "must be non-empty"));
            }
            if let Some(prior) = prior_id {
                if prior >= candidate.candidate_id.as_str() {
                    return Err(fail("candidate_policies", "candidate IDs must be unique and strictly sorted"));
                }
            }
            prior_id = Some(candidate.candidate_id.as_str());
            if !ids.insert(candidate.candidate_id.as_str()) {
                return Err(fail("candidate_policies", "duplicate candidate ID"));
            }
            if candidate.upper_threshold_ppm > PPM as u32 {
                return Err(fail(format!("candidate_policies[{index}].upper_threshold_ppm"), "must be in [0, 1,000,000]"));
            }
            if candidate.lower_threshold_by_step_ppm.len() != horizon {
                return Err(fail(format!("candidate_policies[{index}].lower_threshold_by_step_ppm"), "must have one lower threshold for every reasoning step"));
            }
            for (step, lower) in candidate.lower_threshold_by_step_ppm.iter().enumerate() {
                if *lower > PPM as u32 || *lower >= candidate.upper_threshold_ppm {
                    return Err(fail(
                        format!("candidate_policies[{index}].lower_threshold_by_step_ppm[{step}]"),
                        "lower threshold must be in [0, 1] and strictly below the upper threshold",
                    ));
                }
            }
        }
        if self.calibration_trajectories.is_empty() || self.calibration_trajectories.len() > MAX_TRAJECTORIES {
            return Err(fail("calibration_trajectories", "trajectory count outside supported limits"));
        }
        let mut trajectory_ids = BTreeSet::new();
        for (index, trajectory) in self.calibration_trajectories.iter().enumerate() {
            if trajectory.trajectory_id.trim().is_empty() || trajectory.scenario_id.trim().is_empty() {
                return Err(fail(format!("calibration_trajectories[{index}]"), "trajectory and scenario IDs must be non-empty"));
            }
            if !trajectory_ids.insert(trajectory.trajectory_id.as_str()) {
                return Err(fail("calibration_trajectories", "duplicate trajectory ID"));
            }
            if trajectory.confidence_ppm.len() != horizon
                || trajectory.correct_by_step.len() != horizon
                || trajectory.cumulative_tokens.len() != horizon {
                return Err(fail(format!("calibration_trajectories[{index}]"), "confidence, correctness and token arrays must match the common horizon"));
            }
            if trajectory.confidence_ppm.iter().any(|x| *x > PPM as u32) {
                return Err(fail(format!("calibration_trajectories[{index}].confidence_ppm"), "confidence signal outside [0, 1,000,000]"));
            }
            if trajectory.cumulative_tokens[0] == 0
                || trajectory.cumulative_tokens.windows(2).any(|w| w[0] >= w[1]) {
                return Err(fail(format!("calibration_trajectories[{index}].cumulative_tokens"), "cumulative token counts must be positive and strictly increasing"));
            }
        }
        Ok(())
    }
}

pub const BUILTIN_FIXTURE_JSON: &str = include_str!("../fixtures/dual_threshold_compute_risk_v0.1.json");
pub fn load_builtin_fixture() -> Result<ComputeRiskFixture, ComputeRiskError> {
    let fixture: ComputeRiskFixture = serde_json::from_str(BUILTIN_FIXTURE_JSON)
        .map_err(|e| fail("fixture", format!("JSON parse failed: {e}")))?;
    fixture.validate()?;
    Ok(fixture)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExitReason {
    UpperThreshold,
    LowerThreshold,
    MaximumBudget,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrajectoryOutcome {
    pub trajectory_id: String,
    pub exit_step_index: usize,
    pub exit_reason: ExitReason,
    /// Bounded [0,1] loss: an upper-threshold exit with an incorrect answer.
    pub upper_false_positive_loss_ppm: u32,
    /// Bounded [0,1] farsighted loss from the lower-threshold exit.
    pub lower_false_negative_loss_ppm: u32,
    /// Descriptive efficiency losses from the paper; not statistically certified here.
    pub upper_efficiency_regret_ppm: u32,
    pub lower_efficiency_regret_ppm: u32,
    pub normalized_compute_ppm: u32,
}

fn ceil_div(numerator: u128, denominator: u128) -> Result<u128, ComputeRiskError> {
    if denominator == 0 {
        return Err(fail("division", "zero denominator"));
    }
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    quotient.checked_add(u128::from(remainder != 0))
        .ok_or_else(|| fail("division", "ceiling division overflow"))
}

fn scale_fraction_ceil(numerator: u128, denominator: usize) -> Result<u32, ComputeRiskError> {
    let scaled = numerator.checked_mul(u128::from(PPM))
        .ok_or_else(|| fail("loss", "PPM scaling overflow"))?;
    u32::try_from(ceil_div(scaled, denominator as u128)?)
        .map_err(|_| fail("loss", "PPM loss exceeds u32"))
}

/// Evaluate one frozen policy against one complete trajectory. Upper and lower
/// conditions are mutually exclusive by fixture validation.
pub fn evaluate_trajectory(
    candidate: &CandidatePolicy,
    trajectory: &CalibrationTrajectory,
) -> Result<TrajectoryOutcome, ComputeRiskError> {
    let horizon = trajectory.confidence_ppm.len();
    if horizon == 0 || candidate.lower_threshold_by_step_ppm.len() != horizon {
        return Err(fail("trajectory", "candidate and trajectory horizons must match and be non-empty"));
    }
    let mut exit_step = horizon - 1;
    let mut reason = ExitReason::MaximumBudget;
    for step in 0..horizon {
        let signal = trajectory.confidence_ppm[step];
        if signal >= candidate.upper_threshold_ppm {
            exit_step = step;
            reason = ExitReason::UpperThreshold;
            break;
        }
        if signal <= candidate.lower_threshold_by_step_ppm[step] {
            exit_step = step;
            reason = ExitReason::LowerThreshold;
            break;
        }
    }

    let upper_loss = if reason == ExitReason::UpperThreshold && !trajectory.correct_by_step[exit_step] {
        PPM as u32
    } else {
        0
    };

    let lower_loss = if reason == ExitReason::LowerThreshold {
        let correct_future = trajectory.correct_by_step[exit_step..]
            .iter().filter(|x| **x).count() as u128;
        scale_fraction_ceil(correct_future, horizon - exit_step)?
    } else {
        0
    };

    let first_correct = trajectory.correct_by_step.iter().position(|x| *x);
    let wasted_steps = first_correct
        .map(|first| exit_step.saturating_sub(first))
        .unwrap_or(0);
    let upper_efficiency = scale_fraction_ceil(u128::from(wasted_steps as u64), horizon)?;

    let incorrect_through_exit = trajectory.correct_by_step[..=exit_step]
        .iter().filter(|x| !**x).count() as u128;
    let lower_efficiency = scale_fraction_ceil(incorrect_through_exit, horizon)?;

    let tokens_at_exit = trajectory.cumulative_tokens[exit_step];
    let max_tokens = *trajectory.cumulative_tokens.last()
        .ok_or_else(|| fail("cumulative_tokens", "missing maximum token count"))?;
    let normalized_compute = u32::try_from(ceil_div(
        u128::from(tokens_at_exit) * u128::from(PPM),
        u128::from(max_tokens),
    )?).map_err(|_| fail("normalized_compute_ppm", "compute fraction exceeds u32"))?;

    Ok(TrajectoryOutcome {
        trajectory_id: trajectory.trajectory_id.clone(),
        exit_step_index: exit_step,
        exit_reason: reason,
        upper_false_positive_loss_ppm: upper_loss,
        lower_false_negative_loss_ppm: lower_loss,
        upper_efficiency_regret_ppm: upper_efficiency,
        lower_efficiency_regret_ppm: lower_efficiency,
        normalized_compute_ppm: normalized_compute,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RiskFailureAllocation {
    /// Exact Bonferroni allocation is numerator_ppm / denominator.
    pub numerator_ppm: u32,
    pub denominator: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateRiskReceipt {
    pub candidate_id: String,
    pub trajectory_count: usize,
    pub upper_exit_count: usize,
    pub lower_exit_count: usize,
    pub maximum_budget_exit_count: usize,
    pub empirical_upper_risk_ppm: u32,
    pub empirical_lower_risk_ppm: u32,
    pub simultaneous_ucb_radius_ppm: u32,
    pub upper_risk_ucb_ppm: u32,
    pub lower_risk_ucb_ppm: u32,
    pub upper_risk_budget_ppm: u32,
    pub lower_risk_budget_ppm: u32,
    pub upper_risk_feasible: bool,
    pub lower_risk_feasible: bool,
    pub risk_feasible: bool,
    pub mean_normalized_compute_ppm: u32,
    pub mean_upper_efficiency_regret_ppm: u32,
    pub mean_lower_efficiency_regret_ppm: u32,
    pub upper_ucb_failure_allocation: RiskFailureAllocation,
    pub lower_ucb_failure_allocation: RiskFailureAllocation,
    pub outcome_digest_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionStatus {
    Selected,
    NoCandidateMeetsBothRiskBudgets,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComputeRiskSelectionReport {
    pub schema: String,
    pub status: SelectionStatus,
    pub selected_candidate_id: Option<String>,
    pub statistical_delta_ppm: u32,
    pub candidate_count: usize,
    pub calibration_trajectory_count: usize,
    pub simultaneous_risk_bound_count: usize,
    pub upper_risk_budget_ppm: u32,
    pub lower_risk_budget_ppm: u32,
    pub candidate_receipts: Vec<CandidateRiskReceipt>,
    pub selection_rule: String,
    pub confidence_statement: String,
    pub claim_ceiling: String,
    pub report_sha256: String,
}

fn mean_ppm(sum: u128, count: usize) -> Result<u32, ComputeRiskError> {
    if count == 0 {
        return Err(fail("mean", "cannot compute mean of empty sample"));
    }
    u32::try_from(ceil_div(sum, count as u128)?)
        .map_err(|_| fail("mean", "mean exceeds u32"))
}

/// One-sided Hoeffding radius with Bonferroni allocation across both risks
/// for every frozen candidate. The ceiling makes PPM rounding conservative.
pub fn simultaneous_hoeffding_radius_ppm(
    trajectory_count: usize,
    candidate_count: usize,
    statistical_delta_ppm: u32,
) -> Result<u32, ComputeRiskError> {
    if trajectory_count == 0 || candidate_count == 0 {
        return Err(fail("sample_or_candidate_count", "must be positive"));
    }
    if statistical_delta_ppm == 0 || statistical_delta_ppm >= PPM as u32 {
        return Err(fail("statistical_delta_ppm", "must be in (0, 1,000,000)"));
    }
    let family_size = (candidate_count as f64) * 2.0;
    let delta = f64::from(statistical_delta_ppm) / PPM as f64;
    let log_term = (family_size / delta).ln();
    let radius = (log_term / (2.0 * trajectory_count as f64)).sqrt();
    if !radius.is_finite() || radius < 0.0 {
        return Err(fail("hoeffding_radius", "non-finite or negative radius"));
    }
    let ppm = (radius * PPM as f64).ceil().min(PPM as f64);
    Ok(ppm as u32)
}

fn digest_hex(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    let hash = hasher.finalize();
    let mut output = String::with_capacity(64);
    for b in hash {
        use std::fmt::Write;
        let _ = write!(output, "{b:02x}");
    }
    output
}
pub fn sha256_hex(bytes: &[u8]) -> String {
    let hash = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for b in hash {
        use std::fmt::Write;
        let _ = write!(output, "{b:02x}");
    }
    output
}
fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>, ComputeRiskError> {
    serde_json::to_vec(value).map_err(|e| fail("serialization", format!("JSON serialization failed: {e}")))
}
fn report_digest(report: &ComputeRiskSelectionReport) -> Result<String, ComputeRiskError> {
    let mut copy = report.clone();
    copy.report_sha256.clear();
    Ok(digest_hex(REPORT_DOMAIN, &canonical_json(&copy)?))
}

pub fn evaluate_candidate_grid(
    fixture: &ComputeRiskFixture,
) -> Result<ComputeRiskSelectionReport, ComputeRiskError> {
    fixture.validate()?;
    let n = fixture.calibration_trajectories.len();
    let m = fixture.candidate_policies.len();
    let radius = simultaneous_hoeffding_radius_ppm(n, m, fixture.statistical_delta_ppm)?;

    let mut receipts = Vec::with_capacity(m);
    for candidate in &fixture.candidate_policies {
        let mut upper_sum = 0u128;
        let mut lower_sum = 0u128;
        let mut compute_sum = 0u128;
        let mut upper_efficiency_sum = 0u128;
        let mut lower_efficiency_sum = 0u128;
        let mut upper_exits = 0usize;
        let mut lower_exits = 0usize;
        let mut max_budget_exits = 0usize;
        let mut outcomes = Vec::with_capacity(n);

        for trajectory in &fixture.calibration_trajectories {
            let outcome = evaluate_trajectory(candidate, trajectory)?;
            match outcome.exit_reason {
                ExitReason::UpperThreshold => upper_exits += 1,
                ExitReason::LowerThreshold => lower_exits += 1,
                ExitReason::MaximumBudget => max_budget_exits += 1,
            }
            upper_sum = upper_sum.checked_add(u128::from(outcome.upper_false_positive_loss_ppm))
                .ok_or_else(|| fail("upper_loss_sum", "integer overflow"))?;
            lower_sum = lower_sum.checked_add(u128::from(outcome.lower_false_negative_loss_ppm))
                .ok_or_else(|| fail("lower_loss_sum", "integer overflow"))?;
            compute_sum = compute_sum.checked_add(u128::from(outcome.normalized_compute_ppm))
                .ok_or_else(|| fail("compute_sum", "integer overflow"))?;
            upper_efficiency_sum = upper_efficiency_sum.checked_add(u128::from(outcome.upper_efficiency_regret_ppm))
                .ok_or_else(|| fail("upper_efficiency_sum", "integer overflow"))?;
            lower_efficiency_sum = lower_efficiency_sum.checked_add(u128::from(outcome.lower_efficiency_regret_ppm))
                .ok_or_else(|| fail("lower_efficiency_sum", "integer overflow"))?;
            outcomes.push(outcome);
        }

        let upper_mean = mean_ppm(upper_sum, n)?;
        let lower_mean = mean_ppm(lower_sum, n)?;
        let upper_ucb = upper_mean.saturating_add(radius).min(PPM as u32);
        let lower_ucb = lower_mean.saturating_add(radius).min(PPM as u32);
        let upper_feasible = upper_ucb <= fixture.upper_risk_budget_ppm;
        let lower_feasible = lower_ucb <= fixture.lower_risk_budget_ppm;
        let outcome_digest = digest_hex(b"SYMTHAEA-COMPUTE-RISK-CANDIDATE-OUTCOMES-V0.1", &canonical_json(&outcomes)?);

        receipts.push(CandidateRiskReceipt {
            candidate_id: candidate.candidate_id.clone(),
            trajectory_count: n,
            upper_exit_count: upper_exits,
            lower_exit_count: lower_exits,
            maximum_budget_exit_count: max_budget_exits,
            empirical_upper_risk_ppm: upper_mean,
            empirical_lower_risk_ppm: lower_mean,
            simultaneous_ucb_radius_ppm: radius,
            upper_risk_ucb_ppm: upper_ucb,
            lower_risk_ucb_ppm: lower_ucb,
            upper_risk_budget_ppm: fixture.upper_risk_budget_ppm,
            lower_risk_budget_ppm: fixture.lower_risk_budget_ppm,
            upper_risk_feasible: upper_feasible,
            lower_risk_feasible: lower_feasible,
            risk_feasible: upper_feasible && lower_feasible,
            mean_normalized_compute_ppm: mean_ppm(compute_sum, n)?,
            mean_upper_efficiency_regret_ppm: mean_ppm(upper_efficiency_sum, n)?,
            mean_lower_efficiency_regret_ppm: mean_ppm(lower_efficiency_sum, n)?,
            upper_ucb_failure_allocation: RiskFailureAllocation {
                numerator_ppm: fixture.statistical_delta_ppm,
                denominator: (2 * m) as u32,
            },
            lower_ucb_failure_allocation: RiskFailureAllocation {
                numerator_ppm: fixture.statistical_delta_ppm,
                denominator: (2 * m) as u32,
            },
            outcome_digest_sha256: outcome_digest,
        });
    }

    // Selection may use empirical efficiency on the same validation set:
    // simultaneous risk bounds cover every frozen candidate, so risk validity
    // is not conditioned on a single candidate being preselected.
    let selected = receipts.iter()
        .filter(|x| x.risk_feasible)
        .min_by(|a, b| a.mean_normalized_compute_ppm.cmp(&b.mean_normalized_compute_ppm)
            .then_with(|| a.candidate_id.cmp(&b.candidate_id)))
        .map(|x| x.candidate_id.clone());

    let mut report = ComputeRiskSelectionReport {
        schema: "symthaea.compute_risk_selection_report".to_string(),
        status: if selected.is_some() { SelectionStatus::Selected } else { SelectionStatus::NoCandidateMeetsBothRiskBudgets },
        selected_candidate_id: selected,
        statistical_delta_ppm: fixture.statistical_delta_ppm,
        candidate_count: m,
        calibration_trajectory_count: n,
        simultaneous_risk_bound_count: 2 * m,
        upper_risk_budget_ppm: fixture.upper_risk_budget_ppm,
        lower_risk_budget_ppm: fixture.lower_risk_budget_ppm,
        candidate_receipts: receipts,
        selection_rule: "among all frozen candidates whose simultaneous upper and lower risk UCBs meet their separate budgets, choose the candidate with minimum mean normalized compute; ties break by candidate_id".to_string(),
        confidence_statement: format!("Under iid/exchangeable complete-trajectory calibration units, each loss bounded in [0,1], and a candidate grid frozen independently of calibration outcomes, all 2M risk UCBs simultaneously cover their corresponding population means with probability at least 1 - delta = {:.6}.", 1.0 - f64::from(fixture.statistical_delta_ppm) / PPM as f64),
        claim_ceiling: "finite frozen-grid simultaneous Hoeffding risk bounds under stated trajectory-level exchangeability only; no distribution-shift, anytime, subgroup, or model-correctness guarantee".to_string(),
        report_sha256: String::new(),
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

pub fn verify_report(report: &ComputeRiskSelectionReport) -> Result<(), ComputeRiskError> {
    if report.schema != "symthaea.compute_risk_selection_report" {
        return Err(fail("schema", "unsupported selection report schema"));
    }
    if report.report_sha256.len() != 64
        || !report.report_sha256.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err(fail("report_sha256", "must be lowercase hexadecimal SHA-256"));
    }
    if report.simultaneous_risk_bound_count != 2 * report.candidate_count
        || report.candidate_receipts.len() != report.candidate_count {
        return Err(fail("candidate_count", "report candidate/risk-bound counts are inconsistent"));
    }
    if report.formal_placeholder_forbidden() {
        return Err(fail("claim_ceiling", "unsupported claim ceiling"));
    }
    let expected = report_digest(report)?;
    if expected != report.report_sha256 {
        return Err(fail("report_sha256", "does not match report contents"));
    }
    Ok(())
}

impl ComputeRiskSelectionReport {
    fn formal_placeholder_forbidden(&self) -> bool {
        self.claim_ceiling.is_empty()
            || !self.claim_ceiling.contains("no distribution-shift")
            || !self.claim_ceiling.contains("no anytime")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_fixture_validates_and_has_full_frozen_grid() {
        let fixture = load_builtin_fixture().unwrap();
        assert_eq!(fixture.calibration_trajectories.len(), 512);
        assert_eq!(fixture.candidate_policies.len(), 4);
        assert_eq!(fixture.horizon_steps().unwrap(), 8);
    }

    #[test]
    fn simultaneous_hoeffding_radius_has_candidate_family_correction() {
        let radius = simultaneous_hoeffding_radius_ppm(512, 4, 50_000).unwrap();
        assert_eq!(radius, 70_401);
        let one_candidate = simultaneous_hoeffding_radius_ppm(512, 1, 50_000).unwrap();
        assert!(one_candidate < radius);
    }

    #[test]
    fn upper_exit_loss_is_false_positive_indicator_only() {
        let candidate = CandidatePolicy {
            candidate_id: "upper".into(),
            upper_threshold_ppm: 800_000,
            lower_threshold_by_step_ppm: vec![0, 0],
        };
        let wrong = CalibrationTrajectory {
            trajectory_id: "wrong".into(), scenario_id: "test".into(),
            confidence_ppm: vec![900_000, 950_000],
            correct_by_step: vec![false, true],
            cumulative_tokens: vec![10, 20],
        };
        let out = evaluate_trajectory(&candidate, &wrong).unwrap();
        assert_eq!(out.exit_reason, ExitReason::UpperThreshold);
        assert_eq!(out.upper_false_positive_loss_ppm, 1_000_000);
        assert_eq!(out.lower_false_negative_loss_ppm, 0);
    }

    #[test]
    fn lower_exit_uses_farsighted_present_and_future_correct_fraction() {
        let candidate = CandidatePolicy {
            candidate_id: "lower".into(),
            upper_threshold_ppm: 900_000,
            lower_threshold_by_step_ppm: vec![0, 400_000, 500_000],
        };
        let trajectory = CalibrationTrajectory {
            trajectory_id: "late-solve".into(), scenario_id: "test".into(),
            confidence_ppm: vec![300_000, 350_000, 800_000],
            correct_by_step: vec![false, false, true],
            cumulative_tokens: vec![10, 20, 30],
        };
        let out = evaluate_trajectory(&candidate, &trajectory).unwrap();
        assert_eq!(out.exit_step_index, 1);
        assert_eq!(out.exit_reason, ExitReason::LowerThreshold);
        assert_eq!(out.lower_false_negative_loss_ppm, 500_000);
        assert_eq!(out.upper_false_positive_loss_ppm, 0);
    }

    #[test]
    fn equality_at_upper_and_lower_thresholds_triggers_exit() {
        let upper = CandidatePolicy { candidate_id:"upper".into(), upper_threshold_ppm:800_000, lower_threshold_by_step_ppm:vec![0] };
        let x = CalibrationTrajectory { trajectory_id:"a".into(),scenario_id:"s".into(),confidence_ppm:vec![800_000],correct_by_step:vec![true],cumulative_tokens:vec![5] };
        assert_eq!(evaluate_trajectory(&upper,&x).unwrap().exit_reason,ExitReason::UpperThreshold);
        let lower = CandidatePolicy { candidate_id:"lower".into(), upper_threshold_ppm:900_000, lower_threshold_by_step_ppm:vec![400_000] };
        let y = CalibrationTrajectory { trajectory_id:"b".into(),scenario_id:"s".into(),confidence_ppm:vec![400_000],correct_by_step:vec![false],cumulative_tokens:vec![5] };
        assert_eq!(evaluate_trajectory(&lower,&y).unwrap().exit_reason,ExitReason::LowerThreshold);
    }

    #[test]
    fn selected_candidate_is_feasible_for_both_simultaneous_risks() {
        let fixture = load_builtin_fixture().unwrap();
        let report = evaluate_candidate_grid(&fixture).unwrap();
        assert_eq!(report.selected_candidate_id.as_deref(),Some("aggressive_dual"));
        let selected = report.candidate_receipts.iter().find(|x|Some(x.candidate_id.as_str())==report.selected_candidate_id.as_deref()).unwrap();
        assert!(selected.risk_feasible);
        assert!(selected.upper_risk_ucb_ppm <= fixture.upper_risk_budget_ppm);
        assert!(selected.lower_risk_ucb_ppm <= fixture.lower_risk_budget_ppm);
        assert_eq!(selected.upper_risk_ucb_ppm,216_886);
        assert_eq!(selected.lower_risk_ucb_ppm,127_042);
    }

    #[test]
    fn no_feasible_policy_returns_none_instead_of_weakening_risk_budget() {
        let mut fixture = load_builtin_fixture().unwrap();
        fixture.upper_risk_budget_ppm = 1;
        fixture.lower_risk_budget_ppm = 1;
        let report = evaluate_candidate_grid(&fixture).unwrap();
        assert_eq!(report.selected_candidate_id,None);
        assert_eq!(report.status,SelectionStatus::NoCandidateMeetsBothRiskBudgets);
    }

    #[test]
    fn fixture_rejects_candidate_grid_reordering_and_duplicate_ids() {
        let mut fixture = load_builtin_fixture().unwrap();
        fixture.candidate_policies.swap(0,1);
        assert!(fixture.validate().is_err());
        let mut fixture = load_builtin_fixture().unwrap();
        fixture.candidate_policies[1].candidate_id = fixture.candidate_policies[0].candidate_id.clone();
        assert!(fixture.validate().is_err());
    }

    #[test]
    fn fixture_rejects_lower_threshold_not_below_upper() {
        let mut fixture = load_builtin_fixture().unwrap();
        fixture.candidate_policies[0].lower_threshold_by_step_ppm[2] = fixture.candidate_policies[0].upper_threshold_ppm;
        assert!(fixture.validate().is_err());
    }

    #[test]
    fn fixture_rejects_future_or_mismatched_trajectory_shapes() {
        let mut fixture = load_builtin_fixture().unwrap();
        fixture.calibration_trajectories[0].correct_by_step.pop();
        assert!(fixture.validate().is_err());
        let mut fixture = load_builtin_fixture().unwrap();
        fixture.calibration_trajectories[0].confidence_ppm[0] = 1_000_001;
        assert!(fixture.validate().is_err());
    }

    #[test]
    fn repeated_trajectory_ids_and_nonmonotone_tokens_fail_closed() {
        let mut fixture = load_builtin_fixture().unwrap();
        fixture.calibration_trajectories[1].trajectory_id = fixture.calibration_trajectories[0].trajectory_id.clone();
        assert!(fixture.validate().is_err());
        let mut fixture = load_builtin_fixture().unwrap();
        fixture.calibration_trajectories[0].cumulative_tokens[2] = 1;
        assert!(fixture.validate().is_err());
    }

    #[test]
    fn maximum_budget_fallback_does_not_claim_early_exit_risk() {
        let c = CandidatePolicy { candidate_id:"never".into(),upper_threshold_ppm:1_000_000,lower_threshold_by_step_ppm:vec![0,0] };
        let x = CalibrationTrajectory { trajectory_id:"x".into(),scenario_id:"s".into(),confidence_ppm:vec![500_000,500_000],correct_by_step:vec![false,true],cumulative_tokens:vec![5,10] };
        let out = evaluate_trajectory(&c,&x).unwrap();
        assert_eq!(out.exit_reason,ExitReason::MaximumBudget);
        assert_eq!(out.upper_false_positive_loss_ppm,0);
        assert_eq!(out.lower_false_negative_loss_ppm,0);
        assert_eq!(out.normalized_compute_ppm,1_000_000);
    }

    #[test]
    fn report_digest_detects_candidate_receipt_tampering() {
        let fixture = load_builtin_fixture().unwrap();
        let report = evaluate_candidate_grid(&fixture).unwrap();
        verify_report(&report).unwrap();
        let mut changed = report.clone();
        changed.candidate_receipts[0].mean_normalized_compute_ppm += 1;
        assert!(verify_report(&changed).is_err());
    }

    #[test]
    fn all_candidates_receive_identical_simultaneous_radius() {
        let fixture = load_builtin_fixture().unwrap();
        let report = evaluate_candidate_grid(&fixture).unwrap();
        let radii: BTreeSet<u32> = report.candidate_receipts.iter().map(|x|x.simultaneous_ucb_radius_ppm).collect();
        assert_eq!(radii.len(),1);
        assert_eq!(radii.first().copied(),Some(70_401));
        assert!(report.candidate_receipts.iter().all(|x|x.upper_ucb_failure_allocation.numerator_ppm==50_000 && x.upper_ucb_failure_allocation.denominator==8));
        assert!(report.candidate_receipts.iter().all(|x|x.lower_ucb_failure_allocation.numerator_ppm==50_000 && x.lower_ucb_failure_allocation.denominator==8));
    }
}
