//! Finite optional-continuation significance ledger.
//!
//! A first-stage [0,1]-valued test outcome can serve as the conditional significance
//! budget for a later test. This finite reference verifies the tower-property contract:
//! E0[phi_1] <= alpha and E0[phi_2 | first-stage outcome] <= phi_1(outcome)
//! for every reachable first-stage outcome. It is a finite simple-null ledger, not a
//! general stopping-time or composite-null proof.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, error::Error, fmt};

const PLAN_SCHEMA: &str = "symthaea.architecture-engineering.optional-continuation-plan";
const PLAN_VERSION: &str = "0.1.0";
const REPORT_SCHEMA: &str = "symthaea.architecture-engineering.optional-continuation-report";
const REPORT_VERSION: &str = "0.1.0";
const PROBABILITY_TOLERANCE: f64 = 1e-10;
const CONDITIONAL_TOLERANCE: f64 = 1e-10;
const MAX_FIRST_STAGE_OUTCOMES: usize = 128;
const MAX_SECOND_STAGE_OUTCOMES: usize = 128;
const MAX_TOTAL_BRANCHES: usize = 16_384;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptionalContinuationError {
    pub path: String,
    pub message: String,
}
impl OptionalContinuationError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self { path: path.into(), message: message.into() }
    }
}
impl fmt::Display for OptionalContinuationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}
impl Error for OptionalContinuationError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptionalContinuationFirstOutcome {
    pub outcome_id: String,
    pub null_probability: f64,
    /// The interim randomized-test value and conditional significance budget.
    pub interim_test_value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptionalContinuationSecondOutcome {
    pub outcome_id: String,
    pub conditional_null_probability: f64,
    /// Probability of rejection after this second-stage outcome, including randomization.
    pub rejection_probability: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptionalContinuationBranch {
    pub first_stage_outcome_id: String,
    pub second_stage_outcomes: Vec<OptionalContinuationSecondOutcome>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptionalContinuationPlan {
    pub schema: String,
    pub version: String,
    pub continuation_id: String,
    pub significance_level: f64,
    pub first_stage_outcomes: Vec<OptionalContinuationFirstOutcome>,
    /// Exactly one conditional continuation test for each first-stage outcome.
    pub continuation_branches: Vec<OptionalContinuationBranch>,
}

impl OptionalContinuationPlan {
    pub fn new(
        continuation_id: impl Into<String>,
        significance_level: f64,
        first_stage_outcomes: Vec<OptionalContinuationFirstOutcome>,
        continuation_branches: Vec<OptionalContinuationBranch>,
    ) -> Result<Self, OptionalContinuationError> {
        let plan = Self {
            schema: PLAN_SCHEMA.into(),
            version: PLAN_VERSION.into(),
            continuation_id: continuation_id.into(),
            significance_level,
            first_stage_outcomes,
            continuation_branches,
        };
        plan.clone().canonicalize()?;
        Ok(plan)
    }

    /// Canonicalize every ID-indexed table and validate the finite null experiment.
    pub fn canonicalize(mut self) -> Result<Self, OptionalContinuationError> {
        self.first_stage_outcomes.sort_by(|a,b| a.outcome_id.cmp(&b.outcome_id));
        self.continuation_branches.sort_by(|a,b| a.first_stage_outcome_id.cmp(&b.first_stage_outcome_id));
        for branch in &mut self.continuation_branches {
            branch.second_stage_outcomes.sort_by(|a,b| a.outcome_id.cmp(&b.outcome_id));
        }
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), OptionalContinuationError> {
        if self.schema != PLAN_SCHEMA || self.version != PLAN_VERSION {
            return Err(OptionalContinuationError::new("plan.schema", "unsupported optional-continuation schema/version"));
        }
        if self.continuation_id.trim().is_empty() {
            return Err(OptionalContinuationError::new("plan.continuation_id", "continuation ID must be non-empty"));
        }
        if !self.significance_level.is_finite() || !(0.0..1.0).contains(&self.significance_level) {
            return Err(OptionalContinuationError::new("plan.significance_level", "alpha must be finite and strictly between zero and one"));
        }
        if self.first_stage_outcomes.is_empty() || self.first_stage_outcomes.len() > MAX_FIRST_STAGE_OUTCOMES {
            return Err(OptionalContinuationError::new("plan.first_stage_outcomes", "first-stage outcome count is outside supported bounds"));
        }
        if self.continuation_branches.len() != self.first_stage_outcomes.len() {
            return Err(OptionalContinuationError::new("plan.continuation_branches", "there must be exactly one continuation branch per first-stage outcome"));
        }

        let mut first_ids = BTreeSet::new();
        let mut first_mass = 0.0;
        let mut first_compensation = 0.0;
        for (index, outcome) in self.first_stage_outcomes.iter().enumerate() {
            if outcome.outcome_id.trim().is_empty() || !first_ids.insert(outcome.outcome_id.as_str()) {
                return Err(OptionalContinuationError::new(format!("plan.first_stage_outcomes[{index}].outcome_id"), "first-stage outcome IDs must be non-empty and unique"));
            }
            validate_probability(outcome.null_probability, &format!("plan.first_stage_outcomes[{index}].null_probability"))?;
            validate_probability(outcome.interim_test_value, &format!("plan.first_stage_outcomes[{index}].interim_test_value"))?;
            kahan_add(&mut first_mass, &mut first_compensation, outcome.null_probability);
        }
        if (first_mass - 1.0).abs() > PROBABILITY_TOLERANCE {
            return Err(OptionalContinuationError::new("plan.first_stage_outcomes.null_probability", format!("first-stage null probabilities must sum to one; found {first_mass}")));
        }
        let interim_size = compensated_sum(self.first_stage_outcomes.iter().map(|row| {
            row.null_probability * row.interim_test_value
        }));
        if !interim_size.is_finite() || interim_size > self.significance_level + CONDITIONAL_TOLERANCE {
            return Err(OptionalContinuationError::new("plan.first_stage_outcomes.interim_test_value", format!("first-stage test is not valid at declared alpha: size={interim_size}, alpha={}", self.significance_level)));
        }

        let mut branch_ids = BTreeSet::new();
        let total_second_outcomes: usize = self.continuation_branches.iter().map(|branch| branch.second_stage_outcomes.len()).sum();
        if total_second_outcomes > MAX_TOTAL_BRANCHES {
            return Err(OptionalContinuationError::new("plan.continuation_branches", "total second-stage outcome count exceeds the supported ceiling"));
        }
        for (branch_index, branch) in self.continuation_branches.iter().enumerate() {
            if branch.first_stage_outcome_id.trim().is_empty()
                || !branch_ids.insert(branch.first_stage_outcome_id.as_str())
                || !first_ids.contains(branch.first_stage_outcome_id.as_str())
            {
                return Err(OptionalContinuationError::new(format!("plan.continuation_branches[{branch_index}].first_stage_outcome_id"), "branch must uniquely reference a known first-stage outcome"));
            }
            if branch.second_stage_outcomes.is_empty() || branch.second_stage_outcomes.len() > MAX_SECOND_STAGE_OUTCOMES {
                return Err(OptionalContinuationError::new(format!("plan.continuation_branches[{branch_index}].second_stage_outcomes"), "second-stage outcome count is outside supported bounds"));
            }
            let mut second_ids = BTreeSet::new();
            let mut conditional_mass = 0.0;
            let mut mass_compensation = 0.0;
            let mut conditional_rejection = 0.0;
            let mut rejection_compensation = 0.0;
            for (outcome_index, second) in branch.second_stage_outcomes.iter().enumerate() {
                let path = format!("plan.continuation_branches[{branch_index}].second_stage_outcomes[{outcome_index}]");
                if second.outcome_id.trim().is_empty() || !second_ids.insert(second.outcome_id.as_str()) {
                    return Err(OptionalContinuationError::new(format!("{path}.outcome_id"), "second-stage outcome IDs must be non-empty and unique within the branch"));
                }
                validate_probability(second.conditional_null_probability, &format!("{path}.conditional_null_probability"))?;
                validate_probability(second.rejection_probability, &format!("{path}.rejection_probability"))?;
                kahan_add(&mut conditional_mass, &mut mass_compensation, second.conditional_null_probability);
                kahan_add(&mut conditional_rejection, &mut rejection_compensation,
                    second.conditional_null_probability * second.rejection_probability);
            }
            if (conditional_mass - 1.0).abs() > PROBABILITY_TOLERANCE {
                return Err(OptionalContinuationError::new(format!("plan.continuation_branches[{branch_index}].second_stage_outcomes.conditional_null_probability"), format!("conditional null probabilities must sum to one; found {conditional_mass}")));
            }
            let first = self.first_stage_outcomes.iter().find(|row| row.outcome_id == branch.first_stage_outcome_id)
                .ok_or_else(|| OptionalContinuationError::new("plan.continuation_branches", "validated first-stage outcome disappeared"))?;
            if conditional_rejection > first.interim_test_value + CONDITIONAL_TOLERANCE {
                return Err(OptionalContinuationError::new(
                    format!("plan.continuation_branches[{branch_index}]"),
                    format!("conditional continuation size {conditional_rejection} exceeds interim significance budget {}", first.interim_test_value),
                ));
            }
        }
        if branch_ids.len() != first_ids.len() {
            return Err(OptionalContinuationError::new("plan.continuation_branches", "one or more first-stage outcomes have no continuation branch"));
        }
        Ok(())
    }

    pub fn plan_sha256(&self) -> Result<String, OptionalContinuationError> {
        self.clone().canonicalize()?;
        digest(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptionalContinuationBranchReport {
    pub first_stage_outcome_id: String,
    pub first_stage_null_probability: f64,
    pub interim_test_value: f64,
    pub conditional_second_stage_size: f64,
    pub conditional_budget_slack: f64,
    pub joint_contribution_to_continued_size: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptionalContinuationReport {
    pub schema: String,
    pub version: String,
    pub continuation_id: String,
    pub plan_sha256: String,
    pub first_stage_size: f64,
    pub significance_level: f64,
    pub continued_unconditional_size: f64,
    pub unconditional_budget_slack: f64,
    pub branch_reports: Vec<OptionalContinuationBranchReport>,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct ReportPayload<'a> {
    schema: &'a str,
    version: &'a str,
    continuation_id: &'a str,
    plan_sha256: &'a str,
    first_stage_size: f64,
    significance_level: f64,
    continued_unconditional_size: f64,
    unconditional_budget_slack: f64,
    branch_reports: &'a [OptionalContinuationBranchReport],
}

pub fn analyze_optional_continuation(
    plan: &OptionalContinuationPlan,
) -> Result<OptionalContinuationReport, OptionalContinuationError> {
    let plan = plan.clone().canonicalize()?;
    let mut branch_reports = Vec::with_capacity(plan.continuation_branches.len());
    let mut first_stage_size = 0.0;
    let mut first_compensation = 0.0;
    let mut continued_size = 0.0;
    let mut continued_compensation = 0.0;

    for outcome in &plan.first_stage_outcomes {
        kahan_add(&mut first_stage_size, &mut first_compensation,
            outcome.null_probability * outcome.interim_test_value);
        let branch = plan.continuation_branches.iter().find(|candidate| candidate.first_stage_outcome_id == outcome.outcome_id)
            .ok_or_else(|| OptionalContinuationError::new("plan.continuation_branches", "continuation branch missing after validation"))?;
        let conditional_size = compensated_sum(branch.second_stage_outcomes.iter().map(|second| {
            second.conditional_null_probability * second.rejection_probability
        }));
        let joint_contribution = outcome.null_probability * conditional_size;
        kahan_add(&mut continued_size, &mut continued_compensation, joint_contribution);
        branch_reports.push(OptionalContinuationBranchReport {
            first_stage_outcome_id: outcome.outcome_id.clone(),
            first_stage_null_probability: outcome.null_probability,
            interim_test_value: outcome.interim_test_value,
            conditional_second_stage_size: conditional_size,
            conditional_budget_slack: outcome.interim_test_value - conditional_size,
            joint_contribution_to_continued_size: joint_contribution,
        });
    }

    if !first_stage_size.is_finite() || !continued_size.is_finite()
        || continued_size > first_stage_size + CONDITIONAL_TOLERANCE
        || continued_size > plan.significance_level + CONDITIONAL_TOLERANCE
    {
        return Err(OptionalContinuationError::new("report.continued_unconditional_size", "tower-property budget check failed"));
    }
    let plan_sha256 = plan.plan_sha256()?;
    let payload = ReportPayload {
        schema: REPORT_SCHEMA, version: REPORT_VERSION,
        continuation_id: &plan.continuation_id, plan_sha256: &plan_sha256,
        first_stage_size, significance_level: plan.significance_level,
        continued_unconditional_size: continued_size,
        unconditional_budget_slack: plan.significance_level - continued_size,
        branch_reports: &branch_reports,
    };
    let report_sha256 = digest(&payload)?;
    Ok(OptionalContinuationReport {
        schema: REPORT_SCHEMA.into(), version: REPORT_VERSION.into(),
        continuation_id: plan.continuation_id, plan_sha256, first_stage_size,
        significance_level: plan.significance_level,
        continued_unconditional_size: continued_size,
        unconditional_budget_slack: plan.significance_level - continued_size,
        branch_reports, report_sha256,
    })
}

pub fn verify_optional_continuation_report(
    plan: &OptionalContinuationPlan,
    report: &OptionalContinuationReport,
) -> Result<(), OptionalContinuationError> {
    if analyze_optional_continuation(plan)? != *report {
        return Err(OptionalContinuationError::new("report", "stored report differs from finite optional-continuation reconstruction"));
    }
    Ok(())
}

fn validate_probability(value: f64, path: &str) -> Result<(), OptionalContinuationError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(OptionalContinuationError::new(path, "value must be finite and lie in [0, 1]"));
    }
    Ok(())
}
fn compensated_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut compensation = 0.0;
    for value in values { kahan_add(&mut sum, &mut compensation, value); }
    sum
}
fn kahan_add(sum: &mut f64, compensation: &mut f64, value: f64) {
    let adjusted = value - *compensation;
    let next = *sum + adjusted;
    *compensation = (next - *sum) - adjusted;
    *sum = next;
}
fn digest<T: Serialize>(value: &T) -> Result<String, OptionalContinuationError> {
    let bytes = serde_json::to_vec(value).map_err(|error| OptionalContinuationError::new("serialization", error.to_string()))?;
    Ok(Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> OptionalContinuationPlan {
        OptionalContinuationPlan::new(
            "optional-continue-after-exploration",
            0.05,
            vec![
                OptionalContinuationFirstOutcome { outcome_id: "ambiguous".into(), null_probability: 0.6, interim_test_value: 0.07 },
                OptionalContinuationFirstOutcome { outcome_id: "strong-evidence".into(), null_probability: 0.4, interim_test_value: 0.02 },
            ],
            vec![
                OptionalContinuationBranch { first_stage_outcome_id: "ambiguous".into(), second_stage_outcomes: vec![
                    OptionalContinuationSecondOutcome { outcome_id: "confirm-no".into(), conditional_null_probability: 0.7, rejection_probability: 0.0 },
                    OptionalContinuationSecondOutcome { outcome_id: "confirm-yes".into(), conditional_null_probability: 0.3, rejection_probability: 0.2 },
                ]},
                OptionalContinuationBranch { first_stage_outcome_id: "strong-evidence".into(), second_stage_outcomes: vec![
                    OptionalContinuationSecondOutcome { outcome_id: "replicate-no".into(), conditional_null_probability: 0.9, rejection_probability: 0.0 },
                    OptionalContinuationSecondOutcome { outcome_id: "replicate-yes".into(), conditional_null_probability: 0.1, rejection_probability: 0.1 },
                ]},
            ],
        ).unwrap()
    }

    #[test]
    fn conditional_significance_budgets_preserve_unconditional_alpha() {
        let plan = plan();
        let report = analyze_optional_continuation(&plan).unwrap();
        assert!((report.first_stage_size - 0.05).abs() < 1e-12);
        assert!((report.continued_unconditional_size - 0.04).abs() < 1e-12);
        assert!((report.unconditional_budget_slack - 0.01).abs() < 1e-12);
        verify_optional_continuation_report(&plan, &report).unwrap();
    }

    #[test]
    fn branch_specific_conditional_overrun_fails_closed() {
        let mut plan = plan();
        plan.continuation_branches.iter_mut()
            .find(|branch| branch.first_stage_outcome_id == "ambiguous").unwrap()
            .second_stage_outcomes[1].rejection_probability = 0.4;
        assert!(plan.validate().is_err());
    }

    #[test]
    fn first_stage_alpha_overrun_fails_closed() {
        let mut plan = plan();
        plan.first_stage_outcomes[0].interim_test_value = 0.09;
        assert!(plan.validate().is_err());
    }

    #[test]
    fn missing_duplicate_and_malformed_branches_fail_closed() {
        let mut missing = plan();
        missing.continuation_branches.pop();
        assert!(missing.validate().is_err());
        let mut duplicate = plan();
        duplicate.continuation_branches[1].first_stage_outcome_id = "ambiguous".into();
        assert!(duplicate.validate().is_err());
        let mut malformed = plan();
        malformed.continuation_branches[0].second_stage_outcomes[0].conditional_null_probability = f64::NAN;
        assert!(malformed.validate().is_err());
    }

    #[test]
    fn canonical_order_and_report_mutations_are_detected() {
        let baseline = plan();
        let report = analyze_optional_continuation(&baseline).unwrap();
        let mut reordered = baseline.clone();
        reordered.first_stage_outcomes.reverse();
        reordered.continuation_branches.reverse();
        for branch in &mut reordered.continuation_branches { branch.second_stage_outcomes.reverse(); }
        assert_eq!(report, analyze_optional_continuation(&reordered).unwrap());
        let mut altered = report.clone();
        altered.continued_unconditional_size += 0.01;
        assert!(verify_optional_continuation_report(&baseline, &altered).is_err());
    }
}
