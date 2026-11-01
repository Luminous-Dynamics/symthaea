//! Finite composite-null pointwise optional-continuation reference.
//!
//! For each null distribution P in a finite family H, this module verifies:
//!   E_P[phi_1^P] <= alpha
//!   E_P[phi_2^P | F_1] <= phi_1^P, at every declared first-stage history.
//!
//! It then constructs phi_2^H(path) = min_{P in H} phi_2^P(path), the finite
//! analogue of the essential infimum. Since phi_2^H <= phi_2^P pointwise for
//! every P, its expectation under every null model is bounded by alpha.
//!
//! This is a finite two-stage reference, not a continuous-family or arbitrary-
//! filtration implementation. Conditional validity of supplied tests remains
//! an assumption represented by the plan, not something inferred from data.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::{BTreeMap, BTreeSet}, error::Error, fmt};

const PLAN_SCHEMA: &str = "symthaea.architecture-engineering.composite-null-continuation-plan";
const PLAN_VERSION: &str = "0.1.0";
const REPORT_SCHEMA: &str = "symthaea.architecture-engineering.composite-null-continuation-report";
const REPORT_VERSION: &str = "0.1.0";
const PROBABILITY_TOLERANCE: f64 = 1e-10;
const CONDITIONAL_TOLERANCE: f64 = 1e-10;
const MAX_NULL_MODELS: usize = 32;
const MAX_FIRST_STAGE_OUTCOMES: usize = 64;
const MAX_SECOND_STAGE_OUTCOMES: usize = 64;
const MAX_TOTAL_BRANCHES: usize = 4_096;
const MAX_TERMINAL_PATHS: usize = 4_096;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeContinuationError {
    pub path: String,
    pub message: String,
}

impl CompositeContinuationError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self { path: path.into(), message: message.into() }
    }
}

impl fmt::Display for CompositeContinuationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for CompositeContinuationError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompositeFirstStageOutcome {
    pub outcome_id: String,
    pub null_probability: f64,
    /// Model-specific interim [0,1] test value and continuation budget.
    pub interim_test_value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompositeSecondStageOutcome {
    pub outcome_id: String,
    pub conditional_null_probability: f64,
    /// Model-specific terminal [0,1] test value, including randomization.
    pub rejection_probability: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompositeContinuationBranch {
    pub first_stage_outcome_id: String,
    pub second_stage_outcomes: Vec<CompositeSecondStageOutcome>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompositeNullModel {
    pub model_id: String,
    pub first_stage_outcomes: Vec<CompositeFirstStageOutcome>,
    pub continuation_branches: Vec<CompositeContinuationBranch>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompositeNullContinuationPlan {
    pub schema: String,
    pub version: String,
    pub continuation_id: String,
    pub significance_level: f64,
    pub null_models: Vec<CompositeNullModel>,
}

impl CompositeNullContinuationPlan {
    pub fn new(
        continuation_id: impl Into<String>,
        significance_level: f64,
        null_models: Vec<CompositeNullModel>,
    ) -> Result<Self, CompositeContinuationError> {
        Self {
            schema: PLAN_SCHEMA.into(),
            version: PLAN_VERSION.into(),
            continuation_id: continuation_id.into(),
            significance_level,
            null_models,
        }
        .canonicalize()
    }

    /// Canonicalize all ID-indexed tables before validating and hashing.
    pub fn canonicalize(mut self) -> Result<Self, CompositeContinuationError> {
        self.null_models.sort_by(|a, b| a.model_id.cmp(&b.model_id));
        for model in &mut self.null_models {
            model.first_stage_outcomes.sort_by(|a, b| a.outcome_id.cmp(&b.outcome_id));
            model.continuation_branches.sort_by(|a, b| {
                a.first_stage_outcome_id.cmp(&b.first_stage_outcome_id)
            });
            for branch in &mut model.continuation_branches {
                branch.second_stage_outcomes.sort_by(|a, b| a.outcome_id.cmp(&b.outcome_id));
            }
        }
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), CompositeContinuationError> {
        if self.schema != PLAN_SCHEMA || self.version != PLAN_VERSION {
            return Err(CompositeContinuationError::new(
                "plan.schema",
                "unsupported composite-null continuation schema/version",
            ));
        }
        if self.continuation_id.trim().is_empty() {
            return Err(CompositeContinuationError::new(
                "plan.continuation_id",
                "continuation ID must be non-empty",
            ));
        }
        if !self.significance_level.is_finite()
            || !(0.0..1.0).contains(&self.significance_level)
        {
            return Err(CompositeContinuationError::new(
                "plan.significance_level",
                "alpha must be finite and strictly between zero and one",
            ));
        }
        if self.null_models.is_empty() || self.null_models.len() > MAX_NULL_MODELS {
            return Err(CompositeContinuationError::new(
                "plan.null_models",
                "null-model count is outside supported bounds",
            ));
        }

        let first_support = self.null_models[0]
            .first_stage_outcomes
            .iter()
            .map(|row| row.outcome_id.clone())
            .collect::<Vec<_>>();
        if first_support.is_empty() || first_support.len() > MAX_FIRST_STAGE_OUTCOMES {
            return Err(CompositeContinuationError::new(
                "plan.null_models[0].first_stage_outcomes",
                "first-stage support is empty or exceeds the supported ceiling",
            ));
        }

        let total_branches = self.null_models.len().saturating_mul(first_support.len());
        if total_branches > MAX_TOTAL_BRANCHES {
            return Err(CompositeContinuationError::new(
                "plan.null_models",
                "model-by-history branch count exceeds the supported ceiling",
            ));
        }

        let mut model_ids = BTreeSet::new();
        for (model_index, model) in self.null_models.iter().enumerate() {
            let model_path = format!("plan.null_models[{model_index}]");
            if model.model_id.trim().is_empty() || !model_ids.insert(model.model_id.as_str()) {
                return Err(CompositeContinuationError::new(
                    format!("{model_path}.model_id"),
                    "null-model IDs must be non-empty and unique",
                ));
            }
            if model.first_stage_outcomes.len() != first_support.len()
                || model.first_stage_outcomes.iter().map(|row| row.outcome_id.clone())
                    .collect::<Vec<_>>() != first_support
            {
                return Err(CompositeContinuationError::new(
                    format!("{model_path}.first_stage_outcomes"),
                    "every null model must use the same canonical first-stage outcome support",
                ));
            }
            if model.continuation_branches.len() != first_support.len() {
                return Err(CompositeContinuationError::new(
                    format!("{model_path}.continuation_branches"),
                    "every first-stage history must have exactly one continuation branch",
                ));
            }

            let mut first_mass = 0.0;
            let mut first_compensation = 0.0;
            let mut interim_size = 0.0;
            let mut interim_compensation = 0.0;
            for (outcome_index, outcome) in model.first_stage_outcomes.iter().enumerate() {
                let path = format!("{model_path}.first_stage_outcomes[{outcome_index}]");
                if outcome.outcome_id.trim().is_empty() {
                    return Err(CompositeContinuationError::new(
                        format!("{path}.outcome_id"),
                        "first-stage outcome ID must be non-empty",
                    ));
                }
                validate_probability(outcome.null_probability, &format!("{path}.null_probability"))?;
                validate_probability(outcome.interim_test_value, &format!("{path}.interim_test_value"))?;
                kahan_add(&mut first_mass, &mut first_compensation, outcome.null_probability);
                kahan_add(
                    &mut interim_size,
                    &mut interim_compensation,
                    outcome.null_probability * outcome.interim_test_value,
                );
            }
            if (first_mass - 1.0).abs() > PROBABILITY_TOLERANCE {
                return Err(CompositeContinuationError::new(
                    format!("{model_path}.first_stage_outcomes.null_probability"),
                    format!("first-stage probabilities must sum to one; found {first_mass}"),
                ));
            }
            if !interim_size.is_finite()
                || interim_size > self.significance_level + CONDITIONAL_TOLERANCE
            {
                return Err(CompositeContinuationError::new(
                    format!("{model_path}.first_stage_outcomes.interim_test_value"),
                    format!(
                        "first-stage test exceeds alpha for this null model: size={interim_size}, alpha={}",
                        self.significance_level
                    ),
                ));
            }

            let mut branch_ids = BTreeSet::new();
            for (branch_index, branch) in model.continuation_branches.iter().enumerate() {
                let branch_path = format!("{model_path}.continuation_branches[{branch_index}]");
                if branch.first_stage_outcome_id.trim().is_empty()
                    || !branch_ids.insert(branch.first_stage_outcome_id.as_str())
                    || !first_support.iter().any(|id| id == &branch.first_stage_outcome_id)
                {
                    return Err(CompositeContinuationError::new(
                        format!("{branch_path}.first_stage_outcome_id"),
                        "branch must uniquely reference a declared first-stage outcome",
                    ));
                }
                if branch.second_stage_outcomes.is_empty()
                    || branch.second_stage_outcomes.len() > MAX_SECOND_STAGE_OUTCOMES
                {
                    return Err(CompositeContinuationError::new(
                        format!("{branch_path}.second_stage_outcomes"),
                        "second-stage support is empty or exceeds the supported ceiling",
                    ));
                }

                let mut second_ids = BTreeSet::new();
                let mut conditional_mass = 0.0;
                let mut mass_compensation = 0.0;
                let mut conditional_rejection = 0.0;
                let mut rejection_compensation = 0.0;
                for (second_index, second) in branch.second_stage_outcomes.iter().enumerate() {
                    let path = format!("{branch_path}.second_stage_outcomes[{second_index}]");
                    if second.outcome_id.trim().is_empty()
                        || !second_ids.insert(second.outcome_id.as_str())
                    {
                        return Err(CompositeContinuationError::new(
                            format!("{path}.outcome_id"),
                            "second-stage outcome IDs must be non-empty and unique within the branch",
                        ));
                    }
                    validate_probability(
                        second.conditional_null_probability,
                        &format!("{path}.conditional_null_probability"),
                    )?;
                    validate_probability(
                        second.rejection_probability,
                        &format!("{path}.rejection_probability"),
                    )?;
                    kahan_add(
                        &mut conditional_mass,
                        &mut mass_compensation,
                        second.conditional_null_probability,
                    );
                    kahan_add(
                        &mut conditional_rejection,
                        &mut rejection_compensation,
                        second.conditional_null_probability * second.rejection_probability,
                    );
                }
                if (conditional_mass - 1.0).abs() > PROBABILITY_TOLERANCE {
                    return Err(CompositeContinuationError::new(
                        format!("{branch_path}.second_stage_outcomes.conditional_null_probability"),
                        format!("conditional probabilities must sum to one; found {conditional_mass}"),
                    ));
                }

                let expected_second_support = self.null_models[0]
                    .continuation_branches
                    .iter()
                    .find(|candidate| candidate.first_stage_outcome_id == branch.first_stage_outcome_id)
                    .map(|candidate| {
                        candidate.second_stage_outcomes.iter()
                            .map(|row| row.outcome_id.clone())
                            .collect::<Vec<_>>()
                    });
                let actual_second_support = branch.second_stage_outcomes.iter()
                    .map(|row| row.outcome_id.clone())
                    .collect::<Vec<_>>();
                if let Some(expected) = expected_second_support {
                    if actual_second_support != expected {
                        return Err(CompositeContinuationError::new(
                            format!("{branch_path}.second_stage_outcomes"),
                            "all null models must share the same second-stage outcome support for a given first-stage history",
                        ));
                    }
                }

                let interim = model.first_stage_outcomes
                    .iter()
                    .find(|row| row.outcome_id == branch.first_stage_outcome_id)
                    .map(|row| row.interim_test_value)
                    .ok_or_else(|| CompositeContinuationError::new(
                        &branch_path,
                        "validated first-stage outcome disappeared",
                    ))?;
                if conditional_rejection > interim + CONDITIONAL_TOLERANCE {
                    return Err(CompositeContinuationError::new(
                        &branch_path,
                        format!(
                            "conditional continuation size {conditional_rejection} exceeds this model's interim budget {interim}"
                        ),
                    ));
                }
            }
            if branch_ids.len() != first_support.len() {
                return Err(CompositeContinuationError::new(
                    format!("{model_path}.continuation_branches"),
                    "one or more first-stage histories have no continuation branch",
                ));
            }
        }

        let terminal_path_count = self.null_models[0].continuation_branches
            .iter()
            .try_fold(0usize, |sum, branch| sum.checked_add(branch.second_stage_outcomes.len()))
            .ok_or_else(|| CompositeContinuationError::new(
                "plan.null_models",
                "terminal path count overflow",
            ))?;
        if terminal_path_count > MAX_TERMINAL_PATHS {
            return Err(CompositeContinuationError::new(
                "plan.null_models",
                "terminal path count exceeds the supported ceiling",
            ));
        }
        Ok(())
    }

    pub fn plan_sha256(&self) -> Result<String, CompositeContinuationError> {
        let canonical = self.clone().canonicalize()?;
        digest(&canonical)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompositeContinuationBranchReport {
    pub first_stage_outcome_id: String,
    pub first_stage_null_probability: f64,
    pub interim_test_value: f64,
    pub conditional_second_stage_size: f64,
    pub conditional_budget_slack: f64,
    pub joint_contribution_to_model_specific_size: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompositeModelContinuationReport {
    pub model_id: String,
    pub first_stage_size: f64,
    pub model_specific_continued_size: f64,
    pub composite_continued_size: f64,
    pub first_stage_budget_slack: f64,
    pub branches: Vec<CompositeContinuationBranchReport>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompositePathModelValue {
    pub model_id: String,
    pub rejection_probability: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompositeContinuationPathReport {
    pub first_stage_outcome_id: String,
    pub second_stage_outcome_id: String,
    pub model_rejection_probabilities: Vec<CompositePathModelValue>,
    pub composite_rejection_probability: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompositeNullContinuationReport {
    pub schema: String,
    pub version: String,
    pub continuation_id: String,
    pub plan_sha256: String,
    pub significance_level: f64,
    pub models: Vec<CompositeModelContinuationReport>,
    pub terminal_paths: Vec<CompositeContinuationPathReport>,
    pub maximum_composite_size_over_null_family: f64,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct ReportPayload<'a> {
    schema: &'a str,
    version: &'a str,
    continuation_id: &'a str,
    plan_sha256: &'a str,
    significance_level: f64,
    models: &'a [CompositeModelContinuationReport],
    terminal_paths: &'a [CompositeContinuationPathReport],
    maximum_composite_size_over_null_family: f64,
}

pub fn analyze_composite_null_continuation(
    plan: &CompositeNullContinuationPlan,
) -> Result<CompositeNullContinuationReport, CompositeContinuationError> {
    let plan = plan.clone().canonicalize()?;
    let first_support = &plan.null_models[0].first_stage_outcomes;
    let mut terminal_paths = Vec::new();

    for first in first_support {
        let base_branch = find_branch(&plan.null_models[0], &first.outcome_id)?;
        for second in &base_branch.second_stage_outcomes {
            let mut model_values = Vec::with_capacity(plan.null_models.len());
            let mut composite_value = 1.0_f64;
            for model in &plan.null_models {
                let branch = find_branch(model, &first.outcome_id)?;
                let outcome = branch.second_stage_outcomes.iter()
                    .find(|candidate| candidate.outcome_id == second.outcome_id)
                    .ok_or_else(|| CompositeContinuationError::new(
                        "report.terminal_paths",
                        "shared terminal path disappeared after validation",
                    ))?;
                composite_value = composite_value.min(outcome.rejection_probability);
                model_values.push(CompositePathModelValue {
                    model_id: model.model_id.clone(),
                    rejection_probability: outcome.rejection_probability,
                });
            }
            terminal_paths.push(CompositeContinuationPathReport {
                first_stage_outcome_id: first.outcome_id.clone(),
                second_stage_outcome_id: second.outcome_id.clone(),
                model_rejection_probabilities: model_values,
                composite_rejection_probability: composite_value,
            });
        }
    }

    let composite_by_path = terminal_paths.iter().map(|path| {
        (
            (path.first_stage_outcome_id.as_str(), path.second_stage_outcome_id.as_str()),
            path.composite_rejection_probability,
        )
    }).collect::<BTreeMap<_, _>>();

    let mut model_reports = Vec::with_capacity(plan.null_models.len());
    let mut maximum_composite_size = 0.0_f64;
    for model in &plan.null_models {
        let mut first_stage_size = 0.0;
        let mut first_compensation = 0.0;
        let mut own_continued_size = 0.0;
        let mut own_compensation = 0.0;
        let mut composite_size = 0.0;
        let mut composite_compensation = 0.0;
        let mut branch_reports = Vec::with_capacity(model.first_stage_outcomes.len());

        for first in &model.first_stage_outcomes {
            kahan_add(
                &mut first_stage_size,
                &mut first_compensation,
                first.null_probability * first.interim_test_value,
            );
            let branch = find_branch(model, &first.outcome_id)?;
            let conditional_size = compensated_sum(branch.second_stage_outcomes.iter().map(|second| {
                second.conditional_null_probability * second.rejection_probability
            }));
            let joint_own = first.null_probability * conditional_size;
            kahan_add(&mut own_continued_size, &mut own_compensation, joint_own);

            let mut composite_conditional_size = 0.0;
            let mut composite_conditional_compensation = 0.0;
            for second in &branch.second_stage_outcomes {
                let composite = composite_by_path
                    .get(&(first.outcome_id.as_str(), second.outcome_id.as_str()))
                    .copied()
                    .ok_or_else(|| CompositeContinuationError::new(
                        "report.terminal_paths",
                        "composite terminal path is missing from the reconstructed path table",
                    ))?;
                kahan_add(
                    &mut composite_conditional_size,
                    &mut composite_conditional_compensation,
                    second.conditional_null_probability * composite,
                );
            }
            let joint_composite = first.null_probability * composite_conditional_size;
            kahan_add(&mut composite_size, &mut composite_compensation, joint_composite);

            branch_reports.push(CompositeContinuationBranchReport {
                first_stage_outcome_id: first.outcome_id.clone(),
                first_stage_null_probability: first.null_probability,
                interim_test_value: first.interim_test_value,
                conditional_second_stage_size: conditional_size,
                conditional_budget_slack: first.interim_test_value - conditional_size,
                joint_contribution_to_model_specific_size: joint_own,
            });
        }

        if !first_stage_size.is_finite()
            || !own_continued_size.is_finite()
            || !composite_size.is_finite()
            || own_continued_size > first_stage_size + CONDITIONAL_TOLERANCE
            || composite_size > first_stage_size + CONDITIONAL_TOLERANCE
            || composite_size > plan.significance_level + CONDITIONAL_TOLERANCE
        {
            return Err(CompositeContinuationError::new(
                "report.models",
                format!("tower-property or pointwise-minimum size check failed for null model {}", model.model_id),
            ));
        }
        maximum_composite_size = maximum_composite_size.max(composite_size);
        model_reports.push(CompositeModelContinuationReport {
            model_id: model.model_id.clone(),
            first_stage_size,
            model_specific_continued_size: own_continued_size,
            composite_continued_size: composite_size,
            first_stage_budget_slack: plan.significance_level - first_stage_size,
            branches: branch_reports,
        });
    }

    let plan_sha256 = plan.plan_sha256()?;
    let payload = ReportPayload {
        schema: REPORT_SCHEMA,
        version: REPORT_VERSION,
        continuation_id: &plan.continuation_id,
        plan_sha256: &plan_sha256,
        significance_level: plan.significance_level,
        models: &model_reports,
        terminal_paths: &terminal_paths,
        maximum_composite_size_over_null_family: maximum_composite_size,
    };
    let report_sha256 = digest(&payload)?;
    Ok(CompositeNullContinuationReport {
        schema: REPORT_SCHEMA.into(),
        version: REPORT_VERSION.into(),
        continuation_id: plan.continuation_id,
        plan_sha256,
        significance_level: plan.significance_level,
        models: model_reports,
        terminal_paths,
        maximum_composite_size_over_null_family: maximum_composite_size,
        report_sha256,
    })
}

pub fn verify_composite_null_continuation_report(
    plan: &CompositeNullContinuationPlan,
    report: &CompositeNullContinuationReport,
) -> Result<(), CompositeContinuationError> {
    if analyze_composite_null_continuation(plan)? != *report {
        return Err(CompositeContinuationError::new(
            "report",
            "stored report differs from composite-null pointwise reconstruction",
        ));
    }
    Ok(())
}

fn find_branch<'a>(
    model: &'a CompositeNullModel,
    first_stage_outcome_id: &str,
) -> Result<&'a CompositeContinuationBranch, CompositeContinuationError> {
    model.continuation_branches.iter()
        .find(|branch| branch.first_stage_outcome_id == first_stage_outcome_id)
        .ok_or_else(|| CompositeContinuationError::new(
            "plan.continuation_branches",
            format!("missing branch for first-stage outcome {first_stage_outcome_id:?}"),
        ))
}

fn validate_probability(value: f64, path: &str) -> Result<(), CompositeContinuationError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(CompositeContinuationError::new(
            path,
            "value must be finite and lie in [0, 1]",
        ));
    }
    Ok(())
}

fn compensated_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut compensation = 0.0;
    for value in values {
        kahan_add(&mut sum, &mut compensation, value);
    }
    sum
}

fn kahan_add(sum: &mut f64, compensation: &mut f64, value: f64) {
    let adjusted = value - *compensation;
    let next = *sum + adjusted;
    *compensation = (next - *sum) - adjusted;
    *sum = next;
}

fn digest<T: Serialize>(value: &T) -> Result<String, CompositeContinuationError> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        CompositeContinuationError::new("serialization", error.to_string())
    })?;
    Ok(Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> CompositeNullContinuationPlan {
        let model_a = CompositeNullModel {
            model_id: "null-a".into(),
            first_stage_outcomes: vec![
                CompositeFirstStageOutcome {
                    outcome_id: "x".into(),
                    null_probability: 0.8,
                    interim_test_value: 0.05,
                },
                CompositeFirstStageOutcome {
                    outcome_id: "y".into(),
                    null_probability: 0.2,
                    interim_test_value: 0.01,
                },
            ],
            continuation_branches: vec![
                CompositeContinuationBranch {
                    first_stage_outcome_id: "x".into(),
                    second_stage_outcomes: vec![
                        CompositeSecondStageOutcome {
                            outcome_id: "hit".into(),
                            conditional_null_probability: 0.5,
                            rejection_probability: 0.08,
                        },
                        CompositeSecondStageOutcome {
                            outcome_id: "miss".into(),
                            conditional_null_probability: 0.5,
                            rejection_probability: 0.0,
                        },
                    ],
                },
                CompositeContinuationBranch {
                    first_stage_outcome_id: "y".into(),
                    second_stage_outcomes: vec![
                        CompositeSecondStageOutcome {
                            outcome_id: "hit".into(),
                            conditional_null_probability: 0.5,
                            rejection_probability: 0.02,
                        },
                        CompositeSecondStageOutcome {
                            outcome_id: "miss".into(),
                            conditional_null_probability: 0.5,
                            rejection_probability: 0.0,
                        },
                    ],
                },
            ],
        };
        let model_b = CompositeNullModel {
            model_id: "null-b".into(),
            first_stage_outcomes: vec![
                CompositeFirstStageOutcome {
                    outcome_id: "x".into(),
                    null_probability: 0.2,
                    interim_test_value: 0.03,
                },
                CompositeFirstStageOutcome {
                    outcome_id: "y".into(),
                    null_probability: 0.8,
                    interim_test_value: 0.05,
                },
            ],
            continuation_branches: vec![
                CompositeContinuationBranch {
                    first_stage_outcome_id: "x".into(),
                    second_stage_outcomes: vec![
                        CompositeSecondStageOutcome {
                            outcome_id: "hit".into(),
                            conditional_null_probability: 0.4,
                            rejection_probability: 0.04,
                        },
                        CompositeSecondStageOutcome {
                            outcome_id: "miss".into(),
                            conditional_null_probability: 0.6,
                            rejection_probability: 0.0,
                        },
                    ],
                },
                CompositeContinuationBranch {
                    first_stage_outcome_id: "y".into(),
                    second_stage_outcomes: vec![
                        CompositeSecondStageOutcome {
                            outcome_id: "hit".into(),
                            conditional_null_probability: 0.4,
                            rejection_probability: 0.08,
                        },
                        CompositeSecondStageOutcome {
                            outcome_id: "miss".into(),
                            conditional_null_probability: 0.6,
                            rejection_probability: 0.0,
                        },
                    ],
                },
            ],
        };
        CompositeNullContinuationPlan::new(
            "composite-null-pointwise-continuation",
            0.05,
            vec![model_b, model_a],
        )
        .unwrap()
    }

    #[test]
    fn pointwise_minimum_is_valid_under_every_declared_null() {
        let plan = fixture();
        let report = analyze_composite_null_continuation(&plan).unwrap();
        assert_eq!(report.models.len(), 2);
        assert_eq!(report.terminal_paths.len(), 4);
        assert!((report.models[0].first_stage_size - 0.042).abs() < 1e-12);
        assert!((report.models[1].first_stage_size - 0.046).abs() < 1e-12);
        assert!((report.models[0].model_specific_continued_size - 0.034).abs() < 1e-12);
        assert!((report.models[1].model_specific_continued_size - 0.0288).abs() < 1e-12);
        assert!((report.models[0].composite_continued_size - 0.018).abs() < 1e-12);
        assert!((report.models[1].composite_continued_size - 0.0096).abs() < 1e-12);
        assert!(report.models.iter().all(|model| {
            model.composite_continued_size <= model.first_stage_size + 1e-12
                && model.composite_continued_size <= report.significance_level + 1e-12
        }));
        for path in &report.terminal_paths {
            assert!(path.model_rejection_probabilities.iter().all(|value| {
                path.composite_rejection_probability <= value.rejection_probability + 1e-12
            }));
        }
        verify_composite_null_continuation_report(&plan, &report).unwrap();
    }

    #[test]
    fn every_model_must_respect_its_own_conditional_budget() {
        let mut plan = fixture();
        let model = plan.null_models.iter_mut().find(|model| model.model_id == "null-a").unwrap();
        let branch = model.continuation_branches.iter_mut()
            .find(|branch| branch.first_stage_outcome_id == "x").unwrap();
        branch.second_stage_outcomes.iter_mut()
            .find(|outcome| outcome.outcome_id == "hit").unwrap()
            .rejection_probability = 0.12;
        assert!(plan.validate().is_err());
    }

    #[test]
    fn all_null_models_must_share_observable_path_support() {
        let mut plan = fixture();
        let model = plan.null_models.iter_mut().find(|model| model.model_id == "null-b").unwrap();
        model.continuation_branches[0].second_stage_outcomes[0].outcome_id = "other".into();
        assert!(plan.validate().is_err());
        let mut first_stage_mismatch = fixture();
        first_stage_mismatch.null_models[1].first_stage_outcomes[0].outcome_id = "other".into();
        assert!(first_stage_mismatch.validate().is_err());
    }

    #[test]
    fn first_stage_alpha_and_probability_mass_overruns_fail_closed() {
        let mut alpha_overrun = fixture();
        alpha_overrun.null_models[0].first_stage_outcomes[0].interim_test_value = 0.2;
        assert!(alpha_overrun.validate().is_err());
        let mut mass_overrun = fixture();
        mass_overrun.null_models[0].first_stage_outcomes[0].null_probability = 0.9;
        assert!(mass_overrun.validate().is_err());
    }

    #[test]
    fn duplicate_models_and_missing_branches_fail_closed() {
        let mut duplicate_model = fixture();
        duplicate_model.null_models[1].model_id = duplicate_model.null_models[0].model_id.clone();
        assert!(duplicate_model.validate().is_err());
        let mut missing_branch = fixture();
        missing_branch.null_models[0].continuation_branches.pop();
        assert!(missing_branch.validate().is_err());
    }

    #[test]
    fn canonical_order_and_report_tampering_are_detected() {
        let plan = fixture();
        let report = analyze_composite_null_continuation(&plan).unwrap();
        let mut reordered = plan.clone();
        reordered.null_models.reverse();
        for model in &mut reordered.null_models {
            model.first_stage_outcomes.reverse();
            model.continuation_branches.reverse();
            for branch in &mut model.continuation_branches {
                branch.second_stage_outcomes.reverse();
            }
        }
        assert_eq!(report, analyze_composite_null_continuation(&reordered).unwrap());
        let mut altered = report.clone();
        altered.maximum_composite_size_over_null_family += 0.01;
        assert!(verify_composite_null_continuation_report(&plan, &altered).is_err());
    }
}
