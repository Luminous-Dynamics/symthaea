//! Provenance-rich campaign results for fair benchmark comparison.
//!
//! A CampaignRecord is an experiment artifact, not a benchmark claim. It records
//! what ran and allows matched policy comparisons only when the dataset, model,
//! evaluator, target, seed, and budget contract align.

use crate::contract::EvidenceReference;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

pub const CAMPAIGN_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CampaignStatus {
    Completed,
    Failed,
    Incomplete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveDirection {
    Maximize,
    Minimize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignRecord {
    pub schema_version: u16,
    pub run_id: String,
    pub lane_id: String,
    pub policy_id: String,
    pub seed: u64,
    pub dataset_digest: String,
    pub model_digest: String,
    pub evaluator_digest: String,
    /// Digest of the shared execution environment, excluding policy-specific config.
    pub environment_digest: String,
    pub config_digest: String,
    pub target_definition_digest: String,
    pub budget_definition_digest: String,
    pub objective_name: String,
    pub objective_unit: String,
    pub objective_direction: ObjectiveDirection,
    pub used_evaluations: u64,
    pub max_evaluations: u64,
    pub elapsed_seconds: f64,
    /// Aggregated cost in a documented, versioned budget unit.
    pub total_cost_value: f64,
    pub total_cost_unit: String,
    pub terminal_status: CampaignStatus,
    pub target_reached: bool,
    pub best_validated_objective: Option<f64>,
    pub evidence: Vec<EvidenceReference>,
    pub failure_reasons: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CampaignError {
    UnsupportedSchemaVersion,
    EmptyRequiredField(&'static str),
    InvalidDigest(&'static str),
    InvalidBudget,
    InvalidElapsedTime,
    InvalidCost,
    InvalidCostTolerance,
    InvalidObjective,
    CompletedRunMissingEvidence,
    CompletedRunHasFailureReason,
    FailedRunMissingReason,
    TargetReachedOnNonCompletedRun,
    MismatchedPolicies,
    TargetNotReached,
    UnequalEvaluationUse,
}

impl fmt::Display for CampaignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion => {
                write!(f, "unsupported campaign record schema version")
            }
            Self::EmptyRequiredField(name) => write!(f, "campaign field must not be empty: {name}"),
            Self::InvalidDigest(name) => write!(
                f,
                "campaign field must be a lowercase SHA-256 digest: {name}"
            ),
            Self::InvalidBudget => write!(
                f,
                "campaign evaluation usage must not exceed its positive budget"
            ),
            Self::InvalidElapsedTime => {
                write!(f, "campaign elapsed time must be finite and non-negative")
            }
            Self::InvalidCost => write!(
                f,
                "campaign cost must be finite and non-negative and have a unit"
            ),
            Self::InvalidCostTolerance => write!(
                f,
                "cost-comparison tolerance must be finite and non-negative"
            ),
            Self::InvalidObjective => {
                write!(f, "completed campaign needs a finite objective value")
            }
            Self::CompletedRunMissingEvidence => {
                write!(f, "completed campaign needs traceable evidence")
            }
            Self::CompletedRunHasFailureReason => write!(
                f,
                "completed campaign cannot contain terminal failure reasons"
            ),
            Self::FailedRunMissingReason => {
                write!(f, "failed or incomplete campaign needs a failure reason")
            }
            Self::TargetReachedOnNonCompletedRun => {
                write!(f, "only a completed campaign may claim to reach target")
            }
            Self::MismatchedPolicies => {
                write!(f, "policy runs are not matched on their benchmark contract")
            }
            Self::TargetNotReached => write!(
                f,
                "both completed policy runs must reach the preregistered target for AF"
            ),
            Self::UnequalEvaluationUse => write!(
                f,
                "equal-evaluation comparison requires the same evaluation count"
            ),
        }
    }
}

impl Error for CampaignError {}

impl CampaignRecord {
    pub fn validate(&self) -> Result<(), CampaignError> {
        if self.schema_version != CAMPAIGN_SCHEMA_VERSION {
            return Err(CampaignError::UnsupportedSchemaVersion);
        }
        for (value, name) in [
            (&self.run_id, "run_id"),
            (&self.lane_id, "lane_id"),
            (&self.policy_id, "policy_id"),
            (&self.objective_name, "objective_name"),
            (&self.objective_unit, "objective_unit"),
            (&self.total_cost_unit, "total_cost_unit"),
        ] {
            if value.trim().is_empty() {
                return Err(CampaignError::EmptyRequiredField(name));
            }
        }
        for (value, name) in [
            (&self.dataset_digest, "dataset_digest"),
            (&self.model_digest, "model_digest"),
            (&self.evaluator_digest, "evaluator_digest"),
            (&self.environment_digest, "environment_digest"),
            (&self.config_digest, "config_digest"),
            (&self.target_definition_digest, "target_definition_digest"),
            (&self.budget_definition_digest, "budget_definition_digest"),
        ] {
            validate_digest(value, name)?;
        }
        if self.max_evaluations == 0 || self.used_evaluations > self.max_evaluations {
            return Err(CampaignError::InvalidBudget);
        }
        if !self.elapsed_seconds.is_finite() || self.elapsed_seconds < 0.0 {
            return Err(CampaignError::InvalidElapsedTime);
        }
        if !self.total_cost_value.is_finite() || self.total_cost_value < 0.0 {
            return Err(CampaignError::InvalidCost);
        }
        if let Some(value) = self.best_validated_objective
            && !value.is_finite()
        {
            return Err(CampaignError::InvalidObjective);
        }

        match self.terminal_status {
            CampaignStatus::Completed => {
                if self.used_evaluations == 0 {
                    return Err(CampaignError::InvalidBudget);
                }
                if self.best_validated_objective.is_none() {
                    return Err(CampaignError::InvalidObjective);
                }
                if self.evidence.is_empty() {
                    return Err(CampaignError::CompletedRunMissingEvidence);
                }
                if !self.failure_reasons.is_empty() {
                    return Err(CampaignError::CompletedRunHasFailureReason);
                }
            }
            CampaignStatus::Failed | CampaignStatus::Incomplete => {
                if self.failure_reasons.is_empty()
                    || self
                        .failure_reasons
                        .iter()
                        .any(|reason| reason.trim().is_empty())
                {
                    return Err(CampaignError::FailedRunMissingReason);
                }
                if self.target_reached {
                    return Err(CampaignError::TargetReachedOnNonCompletedRun);
                }
            }
        }

        for evidence in &self.evidence {
            if evidence.evidence_id.trim().is_empty() || evidence.role.trim().is_empty() {
                return Err(CampaignError::EmptyRequiredField("evidence"));
            }
            validate_digest(&evidence.sha256, "evidence.sha256")?;
        }
        Ok(())
    }
}

/// Verify that matched policy runs both reached the same target.
///
/// Use this for acceleration-factor comparisons. A completed run that did not
/// reach the target is valid evidence, but it cannot supply a target-reaching AF.
pub fn validate_target_reached_pair(
    left: &CampaignRecord,
    right: &CampaignRecord,
) -> Result<(), CampaignError> {
    validate_matched_policy_pair(left, right)?;
    if left.terminal_status != CampaignStatus::Completed
        || right.terminal_status != CampaignStatus::Completed
        || !left.target_reached
        || !right.target_reached
    {
        return Err(CampaignError::TargetNotReached);
    }
    Ok(())
}

/// Verify that two policy runs have equal total cost in a common declared unit.
///
/// The caller supplies an absolute tolerance in that unit. This avoids treating
/// the same number of evaluations as equal-cost when query costs differ.
pub fn validate_equal_cost_pair(
    left: &CampaignRecord,
    right: &CampaignRecord,
    tolerance: f64,
) -> Result<(), CampaignError> {
    validate_matched_policy_pair(left, right)?;
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(CampaignError::InvalidCostTolerance);
    }
    if left.total_cost_unit != right.total_cost_unit
        || (left.total_cost_value - right.total_cost_value).abs() > tolerance
    {
        return Err(CampaignError::UnequalEvaluationUse);
    }
    Ok(())
}

/// Verify that two policies can be compared at the same number of evaluations.
pub fn validate_equal_evaluation_pair(
    left: &CampaignRecord,
    right: &CampaignRecord,
) -> Result<(), CampaignError> {
    validate_matched_policy_pair(left, right)?;
    if left.used_evaluations != right.used_evaluations {
        return Err(CampaignError::UnequalEvaluationUse);
    }
    Ok(())
}

/// Verify that two policies can be compared as a matched active-learning pair.
///
/// The seed is deliberately required to match so the comparison can be paired.
/// Used evaluations may differ for target-reaching (AF) comparisons; equal-budget
/// enhancement comparisons must additionally compare runs at the same budget.
/// Model digests must match: comparing policies and comparing models are separate
/// experiments.
pub fn validate_matched_policy_pair(
    left: &CampaignRecord,
    right: &CampaignRecord,
) -> Result<(), CampaignError> {
    left.validate()?;
    right.validate()?;
    let matched = left.lane_id == right.lane_id
        && left.policy_id != right.policy_id
        && left.seed == right.seed
        && left.dataset_digest == right.dataset_digest
        && left.model_digest == right.model_digest
        && left.evaluator_digest == right.evaluator_digest
        && left.environment_digest == right.environment_digest
        && left.target_definition_digest == right.target_definition_digest
        && left.budget_definition_digest == right.budget_definition_digest
        && left.objective_name == right.objective_name
        && left.objective_unit == right.objective_unit
        && left.objective_direction == right.objective_direction
        && left.total_cost_unit == right.total_cost_unit;
    if matched {
        Ok(())
    } else {
        Err(CampaignError::MismatchedPolicies)
    }
}

fn validate_digest(value: &str, field: &'static str) -> Result<(), CampaignError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(CampaignError::InvalidDigest(field));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn completed(policy: &str) -> CampaignRecord {
        CampaignRecord {
            schema_version: CAMPAIGN_SCHEMA_VERSION,
            run_id: format!("run-{policy}-seed-7"),
            lane_id: "active-learning-policy".to_owned(),
            policy_id: policy.to_owned(),
            seed: 7,
            dataset_digest: "a".repeat(64),
            model_digest: "b".repeat(64),
            evaluator_digest: "c".repeat(64),
            environment_digest: "9".repeat(64),
            config_digest: "d".repeat(64),
            target_definition_digest: "e".repeat(64),
            budget_definition_digest: "f".repeat(64),
            objective_name: "best_validated_objective".to_owned(),
            objective_unit: "score".to_owned(),
            objective_direction: ObjectiveDirection::Maximize,
            used_evaluations: 12,
            max_evaluations: 20,
            elapsed_seconds: 3.5,
            total_cost_value: 12.0,
            total_cost_unit: "reference_eval_equivalent".to_owned(),
            terminal_status: CampaignStatus::Completed,
            target_reached: true,
            best_validated_objective: Some(0.75),
            evidence: vec![EvidenceReference {
                evidence_id: format!("predictions-{policy}"),
                sha256: "1".repeat(64),
                role: "raw_prediction_table".to_owned(),
            }],
            failure_reasons: Vec::new(),
        }
    }

    #[test]
    fn completed_campaign_requires_traceable_finite_result() {
        let record = completed("random");
        record
            .validate()
            .expect("valid campaign record should pass");

        let mut no_evidence = completed("random");
        no_evidence.evidence.clear();
        assert_eq!(
            no_evidence.validate(),
            Err(CampaignError::CompletedRunMissingEvidence)
        );

        let mut nonfinite = completed("random");
        nonfinite.best_validated_objective = Some(f64::NAN);
        assert_eq!(nonfinite.validate(), Err(CampaignError::InvalidObjective));
    }

    #[test]
    fn completed_campaign_cannot_claim_free_evaluations() {
        let mut record = completed("random");
        record.used_evaluations = 0;
        assert_eq!(record.validate(), Err(CampaignError::InvalidBudget));
    }

    #[test]
    fn failed_campaign_requires_reason_and_cannot_claim_target_success() {
        let mut record = completed("random");
        record.terminal_status = CampaignStatus::Failed;
        record.target_reached = false;
        record.failure_reasons = vec!["solver_failed".to_owned()];
        record
            .validate()
            .expect("failed run should remain recordable");

        record.failure_reasons.clear();
        assert_eq!(
            record.validate(),
            Err(CampaignError::FailedRunMissingReason)
        );

        record.failure_reasons = vec!["solver_failed".to_owned()];
        record.target_reached = true;
        assert_eq!(
            record.validate(),
            Err(CampaignError::TargetReachedOnNonCompletedRun)
        );
    }

    #[test]
    fn campaign_record_round_trips_and_schema_is_standard_json() {
        let record = completed("random");
        let json = serde_json::to_string(&record).expect("record should serialize");
        let decoded: CampaignRecord =
            serde_json::from_str(&json).expect("record should deserialize");
        decoded
            .validate()
            .expect("round-tripped record should validate");
        assert_eq!(decoded, record);

        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../schemas/campaign-record-v1.schema.json"))
                .expect("campaign JSON schema should parse");
        assert_eq!(schema["properties"]["schema_version"]["const"], 1);
        assert_eq!(
            schema["$schema"],
            "https://json-schema.org/draft/2020-12/schema"
        );
    }

    #[test]
    fn matched_policy_comparison_requires_same_model_and_task() {
        let random = completed("random");
        let active = completed("cost_aware");
        validate_matched_policy_pair(&random, &active)
            .expect("policy comparison with held-constant model should pass");

        let mut changed_model = active.clone();
        changed_model.model_digest = "9".repeat(64);
        assert_eq!(
            validate_matched_policy_pair(&random, &changed_model),
            Err(CampaignError::MismatchedPolicies)
        );

        let mut changed_seed = active;
        changed_seed.seed = 8;
        assert_eq!(
            validate_matched_policy_pair(&random, &changed_seed),
            Err(CampaignError::MismatchedPolicies)
        );
    }

    #[test]
    fn matched_policies_require_same_environment_and_comparable_cost_unit() {
        let left = completed("random");
        let mut right = completed("cost_aware");
        right.environment_digest = "8".repeat(64);
        assert_eq!(
            validate_matched_policy_pair(&left, &right),
            Err(CampaignError::MismatchedPolicies)
        );

        right.environment_digest = left.environment_digest.clone();
        right.total_cost_unit = "seconds".to_owned();
        assert_eq!(
            validate_equal_cost_pair(&left, &right, 0.0),
            Err(CampaignError::MismatchedPolicies)
        );
        right.total_cost_unit = left.total_cost_unit.clone();
        right.total_cost_value = 12.00001;
        assert!(validate_equal_cost_pair(&left, &right, 0.0001).is_ok());
        assert_eq!(
            validate_equal_cost_pair(&left, &right, -1.0),
            Err(CampaignError::InvalidCostTolerance)
        );
    }

    #[test]
    fn comparisons_enforce_target_reached_and_equal_evaluation_modes() {
        let random = completed("random");
        let mut active = completed("cost_aware");
        active.used_evaluations = 9;
        assert!(validate_target_reached_pair(&random, &active).is_ok());
        assert_eq!(
            validate_equal_evaluation_pair(&random, &active),
            Err(CampaignError::UnequalEvaluationUse)
        );

        active.target_reached = false;
        assert_eq!(
            validate_target_reached_pair(&random, &active),
            Err(CampaignError::TargetNotReached)
        );
    }

    #[test]
    fn campaign_cost_must_be_finite_and_non_negative() {
        let mut record = completed("random");
        record.total_cost_value = f64::INFINITY;
        assert_eq!(record.validate(), Err(CampaignError::InvalidCost));
        record.total_cost_value = -1.0;
        assert_eq!(record.validate(), Err(CampaignError::InvalidCost));
    }

    #[test]
    fn campaign_rejects_budget_overrun_and_invalid_digest() {
        let mut record = completed("random");
        record.used_evaluations = 21;
        assert_eq!(record.validate(), Err(CampaignError::InvalidBudget));

        let mut record = completed("random");
        record.dataset_digest = "not-a-digest".to_owned();
        assert_eq!(
            record.validate(),
            Err(CampaignError::InvalidDigest("dataset_digest"))
        );
    }
}
