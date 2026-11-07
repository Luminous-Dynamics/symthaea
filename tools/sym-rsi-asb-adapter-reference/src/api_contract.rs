//! Typed, fixture-driven contract for the pinned Autonomous Science Bench Lab API v1.
//!
//! This module is deliberately transport-agnostic: it parses and validates
//! API payloads but makes no network requests. It validates only public lab
//! responses and agent-visible requests. Hidden truth and verifier artifacts
//! are intentionally absent from every type in this module.

use crate::job_lifecycle::{JobContractError, JobStatus, LabJob};
use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

const API_VERSION: u32 = 1;
const BUDGET_UNIT: &str = "measured designs";

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignCardV1 {
    pub ok: bool,
    pub api_version: u32,
    pub campaign_id: String,
    pub source: String,
    pub objective: String,
    pub design_id_field: String,
    pub measurements: Vec<MeasurementSpecV1>,
    pub stages: Vec<StageSpecV1>,
    pub budget: BudgetSpecV1,
    pub deliverable: DeliverableSpecV1,
    pub design_space: Option<DesignSpaceSpecV1>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementSpecV1 {
    pub name: String,
    pub description: String,
    pub higher_is_better: Option<bool>,
    pub kind: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageSpecV1 {
    pub stage: u32,
    pub label: String,
    pub batch_size: u64,
    pub catalog_size: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetSpecV1 {
    pub unit: String,
    pub total: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeliverableSpecV1 {
    pub kind: String,
    pub format_version: u32,
    pub max_bytes: u64,
    pub input_field: String,
    pub description: String,
    pub requires: String,
    pub content: Option<String>,
    pub requires_stages: Option<u32>,
    pub constraints: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DesignSpaceSpecV1 {
    pub kind: String,
    pub shared_pool: Option<bool>,
    pub format: Option<String>,
    pub max_bytes: Option<u64>,
    pub constraints: Option<Value>,
    pub shape: Option<Vec<u64>>,
    pub id_format: Option<String>,
    pub description: Option<String>,
    pub rules: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignStatusV1 {
    pub ok: bool,
    pub api_version: u32,
    pub campaign_id: String,
    pub stages_total: u32,
    pub stages_completed: u32,
    pub next_stage: Option<u32>,
    pub active_job: Option<String>,
    pub budget: StatusBudgetV1,
    pub deliverable_submitted: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StatusBudgetV1 {
    pub unit: String,
    pub total: u64,
    pub spent: u64,
    pub remaining: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogResponseV1 {
    pub ok: bool,
    pub stage: u32,
    pub designs: Vec<BTreeMap<String, String>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobResponseV1 {
    pub ok: bool,
    pub replayed: Option<bool>,
    pub job: LabJob,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ErrorResponseV1 {
    pub ok: bool,
    pub error: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApiContractError {
    ResponseNotOk,
    WrongApiVersion,
    EmptyIdentifier(&'static str),
    EmptyCampaign,
    InvalidStageOrder,
    InvalidBatchSize,
    InvalidCatalogSize,
    InvalidBudgetUnit,
    BudgetMismatch,
    InvalidMeasurementKind,
    DuplicateMeasurementName,
    InvalidDesignSpace,
    InvalidDeliverable,
    CampaignIdentityMismatch,
    StageCountMismatch,
    InvalidStageProgress,
    InvalidNextStage,
    InvalidBudgetAccounting,
    InvalidActiveJob,
    InvalidCatalogStage,
    CatalogSizeMismatch,
    MissingDesignIdField,
    DuplicateCatalogDesign(String),
    InvalidCatalogDesign(String),
    WrongSubmissionStage,
    WrongBatchSize { expected: usize, actual: usize },
    DuplicateSubmission(String),
    IneligibleDesign(String),
    AlreadyMeasured(String),
    InsufficientRemainingBudget { required: u64, remaining: u64 },
    ResultIdentityMissing { index: usize, field: String },
    ResultIdentityMismatch(String),
    DuplicateResultIdentity(String),
    Job(JobContractError),
}

impl std::fmt::Display for ApiContractError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ApiContractError {}

impl From<JobContractError> for ApiContractError {
    fn from(error: JobContractError) -> Self {
        Self::Job(error)
    }
}

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

impl CampaignCardV1 {
    pub fn parse(contents: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(contents)
    }

    pub fn validate(&self) -> Result<(), ApiContractError> {
        if !self.ok {
            return Err(ApiContractError::ResponseNotOk);
        }
        if self.api_version != API_VERSION {
            return Err(ApiContractError::WrongApiVersion);
        }
        if !non_empty(&self.campaign_id) || !non_empty(&self.source)
            || !non_empty(&self.objective) || !non_empty(&self.design_id_field)
        {
            return Err(ApiContractError::EmptyCampaign);
        }
        if self.stages.is_empty() {
            return Err(ApiContractError::EmptyCampaign);
        }

        let mut total_budget = 0u64;
        for (index, stage) in self.stages.iter().enumerate() {
            if stage.stage != index as u32 + 1 {
                return Err(ApiContractError::InvalidStageOrder);
            }
            if stage.batch_size == 0 {
                return Err(ApiContractError::InvalidBatchSize);
            }
            if let Some(size) = stage.catalog_size {
                if size < stage.batch_size {
                    return Err(ApiContractError::InvalidCatalogSize);
                }
            }
            total_budget = total_budget
                .checked_add(stage.batch_size)
                .ok_or(ApiContractError::BudgetMismatch)?;
        }
        if self.budget.unit != BUDGET_UNIT {
            return Err(ApiContractError::InvalidBudgetUnit);
        }
        if self.budget.total != total_budget {
            return Err(ApiContractError::BudgetMismatch);
        }

        let mut measurement_names = BTreeSet::new();
        for measurement in &self.measurements {
            if !non_empty(&measurement.name) {
                return Err(ApiContractError::EmptyIdentifier("measurement"));
            }
            if !measurement_names.insert(measurement.name.as_str()) {
                return Err(ApiContractError::DuplicateMeasurementName);
            }
            if measurement.kind.as_deref().is_some_and(|kind| kind != "value" && kind != "file") {
                return Err(ApiContractError::InvalidMeasurementKind);
            }
        }

        if let Some(space) = &self.design_space {
            match space.kind.as_str() {
                "catalog" => {
                    if space.shared_pool == Some(true)
                        && space.rules.as_deref().is_none_or(|rules| !non_empty(rules))
                    {
                        return Err(ApiContractError::InvalidDesignSpace);
                    }
                }
                "grid" => {
                    if space.shape.as_ref().is_none_or(|shape| shape.len() != 2 || shape.contains(&0))
                        || space.id_format.as_deref() != Some("r{row}c{col}")
                    {
                        return Err(ApiContractError::InvalidDesignSpace);
                    }
                }
                "upload" => {
                    if space.max_bytes.is_none_or(|bytes| bytes == 0)
                        || space.format.as_deref().is_none_or(|format| !non_empty(format))
                    {
                        return Err(ApiContractError::InvalidDesignSpace);
                    }
                }
                _ => return Err(ApiContractError::InvalidDesignSpace),
            }
        }

        if !non_empty(&self.deliverable.kind)
            || self.deliverable.format_version == 0
            || self.deliverable.max_bytes == 0
            || !non_empty(&self.deliverable.requires)
            || self.deliverable.requires_stages.is_some_and(|count| count as usize > self.stages.len())
            || self.deliverable.content.as_deref().is_some_and(|value| value != "binary")
        {
            return Err(ApiContractError::InvalidDeliverable);
        }
        Ok(())
    }

    pub fn stage(&self, number: u32) -> Option<&StageSpecV1> {
        self.stages.iter().find(|stage| stage.stage == number)
    }

    pub fn is_shared_pool_catalog(&self) -> bool {
        self.design_space.as_ref().is_some_and(|space| {
            space.kind == "catalog" && space.shared_pool == Some(true)
        })
    }
}

impl CampaignStatusV1 {
    pub fn parse(contents: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(contents)
    }

    pub fn validate_against(&self, campaign: &CampaignCardV1) -> Result<(), ApiContractError> {
        campaign.validate()?;
        if !self.ok {
            return Err(ApiContractError::ResponseNotOk);
        }
        if self.api_version != API_VERSION {
            return Err(ApiContractError::WrongApiVersion);
        }
        if self.campaign_id != campaign.campaign_id {
            return Err(ApiContractError::CampaignIdentityMismatch);
        }
        if self.stages_total as usize != campaign.stages.len() {
            return Err(ApiContractError::StageCountMismatch);
        }
        if self.stages_completed > self.stages_total {
            return Err(ApiContractError::InvalidStageProgress);
        }
        let expected_next = if self.stages_completed == self.stages_total {
            None
        } else {
            Some(self.stages_completed + 1)
        };
        if self.next_stage != expected_next {
            return Err(ApiContractError::InvalidNextStage);
        }
        if self.budget.unit != campaign.budget.unit
            || self.budget.total != campaign.budget.total
            || self.budget.spent.checked_add(self.budget.remaining) != Some(self.budget.total)
            || self.budget.spent > self.budget.total
        {
            return Err(ApiContractError::InvalidBudgetAccounting);
        }
        if self.active_job.as_deref().is_some_and(|job| !non_empty(job)) {
            return Err(ApiContractError::InvalidActiveJob);
        }
        Ok(())
    }
}

impl CatalogResponseV1 {
    pub fn parse(contents: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(contents)
    }

    pub fn validate_against(
        &self,
        campaign: &CampaignCardV1,
        requested_stage: u32,
    ) -> Result<BTreeSet<String>, ApiContractError> {
        campaign.validate()?;
        if !self.ok {
            return Err(ApiContractError::ResponseNotOk);
        }
        if self.stage != requested_stage || campaign.stage(requested_stage).is_none() {
            return Err(ApiContractError::InvalidCatalogStage);
        }
        if let Some(expected_size) = campaign.stage(requested_stage).and_then(|stage| stage.catalog_size) {
            if self.designs.len() != expected_size as usize {
                return Err(ApiContractError::CatalogSizeMismatch);
            }
        }
        let mut ids = BTreeSet::new();
        for row in &self.designs {
            let Some(id) = row.get(&campaign.design_id_field) else {
                return Err(ApiContractError::MissingDesignIdField);
            };
            if !non_empty(id) {
                return Err(ApiContractError::InvalidCatalogDesign(id.clone()));
            }
            if !ids.insert(id.clone()) {
                return Err(ApiContractError::DuplicateCatalogDesign(id.clone()));
            }
        }
        Ok(ids)
    }
}

/// Validate an agent's catalog batch before sending POST /v1/experiments.
///
/// The lab remains authoritative. This is a local fail-fast check, not a
/// replacement for server-side stage, eligibility, uniqueness or budget checks.
pub fn validate_catalog_submission(
    campaign: &CampaignCardV1,
    stage_number: u32,
    design_ids: &[String],
    catalog_ids: &BTreeSet<String>,
    previously_measured: &BTreeSet<String>,
    remaining_budget: u64,
) -> Result<(), ApiContractError> {
    campaign.validate()?;
    let Some(stage) = campaign.stage(stage_number) else {
        return Err(ApiContractError::WrongSubmissionStage);
    };
    if design_ids.len() != stage.batch_size as usize {
        return Err(ApiContractError::WrongBatchSize {
            expected: stage.batch_size as usize,
            actual: design_ids.len(),
        });
    }
    if remaining_budget < design_ids.len() as u64 {
        return Err(ApiContractError::InsufficientRemainingBudget {
            required: design_ids.len() as u64,
            remaining: remaining_budget,
        });
    }

    let mut seen = BTreeSet::new();
    for design_id in design_ids {
        if !non_empty(design_id) {
            return Err(ApiContractError::InvalidCatalogDesign(design_id.clone()));
        }
        if !seen.insert(design_id.as_str()) {
            return Err(ApiContractError::DuplicateSubmission(design_id.clone()));
        }
        if !catalog_ids.contains(design_id) {
            return Err(ApiContractError::IneligibleDesign(design_id.clone()));
        }
        if campaign.is_shared_pool_catalog() && previously_measured.contains(design_id) {
            return Err(ApiContractError::AlreadyMeasured(design_id.clone()));
        }
    }
    Ok(())
}

impl JobResponseV1 {
    pub fn parse(contents: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(contents)
    }

    pub fn validate_against(
        &self,
        campaign: &CampaignCardV1,
    ) -> Result<(), ApiContractError> {
        if !self.ok {
            return Err(ApiContractError::ResponseNotOk);
        }
        self.job.validate()?;
        if self.job.stage > campaign.stages.len() as u32 {
            return Err(ApiContractError::WrongSubmissionStage);
        }
        let stage = campaign.stage(self.job.stage).ok_or(ApiContractError::WrongSubmissionStage)?;
        if self.job.designs.len() != stage.batch_size as usize {
            return Err(ApiContractError::WrongBatchSize {
                expected: stage.batch_size as usize,
                actual: self.job.designs.len(),
            });
        }
        if let Some(results) = self.job.completed_measurements()? {
            let mut result_ids = BTreeSet::new();
            for (index, result) in results.iter().enumerate() {
                let Some(id) = result.get(&campaign.design_id_field).and_then(Value::as_str) else {
                    return Err(ApiContractError::ResultIdentityMissing {
                        index,
                        field: campaign.design_id_field.clone(),
                    });
                };
                if !result_ids.insert(id.to_owned()) {
                    return Err(ApiContractError::DuplicateResultIdentity(id.to_owned()));
                }
                if !self.job.designs.iter().any(|design| design == id) {
                    return Err(ApiContractError::ResultIdentityMismatch(id.to_owned()));
                }
            }
            if result_ids.len() != self.job.designs.len() {
                return Err(ApiContractError::ResultIdentityMismatch(
                    "completed result identities do not equal submitted design identities".to_owned(),
                ));
            }
        }
        Ok(())
    }

    pub fn advances_stage(&self, campaign: &CampaignCardV1) -> Result<bool, ApiContractError> {
        self.validate_against(campaign)?;
        Ok(self.job.advances_stage()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CAMPAIGN: &str = include_str!("../fixtures/campaign-card-v1.json");
    const STATUS: &str = include_str!("../fixtures/status-v1.json");
    const CATALOG: &str = include_str!("../fixtures/catalog-stage-1-v1.json");
    const COMPLETED_JOB: &str = include_str!("../fixtures/completed-job-v1.json");

    fn campaign() -> CampaignCardV1 {
        CampaignCardV1::parse(CAMPAIGN).expect("campaign fixture parses")
    }

    fn catalog() -> CatalogResponseV1 {
        CatalogResponseV1::parse(CATALOG).expect("catalog fixture parses")
    }

    fn completed_job() -> JobResponseV1 {
        JobResponseV1::parse(COMPLETED_JOB).expect("job fixture parses")
    }

    #[test]
    fn accepts_campaign_card_and_computes_total_budget() {
        let card = campaign();
        assert!(card.validate().is_ok());
        assert_eq!(card.budget.total, 4);
        assert!(card.is_shared_pool_catalog());
    }

    #[test]
    fn accepts_status_and_reconciles_budget() {
        let card = campaign();
        let status = CampaignStatusV1::parse(STATUS).expect("status fixture parses");
        assert!(status.validate_against(&card).is_ok());
    }

    #[test]
    fn rejects_status_budget_conservation_failure() {
        let card = campaign();
        let mut value: Value = serde_json::from_str(STATUS).expect("status JSON parses");
        value["budget"]["remaining"] = Value::from(3);
        let bad: CampaignStatusV1 = serde_json::from_value(value).expect("mutated status parses");
        assert_eq!(
            bad.validate_against(&card).expect_err("budget must reconcile"),
            ApiContractError::InvalidBudgetAccounting
        );
    }

    #[test]
    fn rejects_status_from_another_campaign() {
        let card = campaign();
        let mut value: Value = serde_json::from_str(STATUS).expect("status JSON parses");
        value["campaign_id"] = Value::String("other".to_owned());
        let bad: CampaignStatusV1 = serde_json::from_value(value).expect("mutated status parses");
        assert_eq!(
            bad.validate_against(&card).expect_err("campaign identity must match"),
            ApiContractError::CampaignIdentityMismatch
        );
    }

    #[test]
    fn accepts_catalog_with_unique_design_ids() {
        let card = campaign();
        let ids = catalog().validate_against(&card, 1).expect("catalog is valid");
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn rejects_catalog_with_duplicate_design_ids() {
        let card = campaign();
        let mut value: Value = serde_json::from_str(CATALOG).expect("catalog JSON parses");
        value["designs"][2]["design_id"] = Value::String("d-02".to_owned());
        let bad: CatalogResponseV1 = serde_json::from_value(value).expect("mutated catalog parses");
        assert!(matches!(
            bad.validate_against(&card, 1),
            Err(ApiContractError::DuplicateCatalogDesign(_))
        ));
    }

    #[test]
    fn rejects_catalog_missing_design_id_field() {
        let card = campaign();
        let mut value: Value = serde_json::from_str(CATALOG).expect("catalog JSON parses");
        value["designs"][0].as_object_mut().expect("row object").remove("design_id");
        let bad: CatalogResponseV1 = serde_json::from_value(value).expect("mutated catalog parses");
        assert_eq!(
            bad.validate_against(&card, 1).expect_err("design id field required"),
            ApiContractError::MissingDesignIdField
        );
    }

    #[test]
    fn accepts_valid_exact_size_eligible_catalog_batch() {
        let card = campaign();
        let ids = catalog().validate_against(&card, 1).expect("catalog is valid");
        assert!(validate_catalog_submission(
            &card, 1, &["d-01".to_owned(), "d-02".to_owned()], &ids, &BTreeSet::new(), 4
        ).is_ok());
    }

    #[test]
    fn rejects_wrong_size_and_duplicate_batches() {
        let card = campaign();
        let ids = catalog().validate_against(&card, 1).expect("catalog is valid");
        let wrong_size = validate_catalog_submission(&card, 1, &["d-01".to_owned()], &ids, &BTreeSet::new(), 4);
        assert!(matches!(wrong_size, Err(ApiContractError::WrongBatchSize { .. })));
        let duplicate = validate_catalog_submission(
            &card, 1, &["d-01".to_owned(), "d-01".to_owned()], &ids, &BTreeSet::new(), 4
        );
        assert!(matches!(duplicate, Err(ApiContractError::DuplicateSubmission(_))));
    }

    #[test]
    fn rejects_ineligible_repeated_and_over_budget_designs() {
        let card = campaign();
        let ids = catalog().validate_against(&card, 1).expect("catalog is valid");
        let ineligible = validate_catalog_submission(
            &card, 1, &["d-01".to_owned(), "d-99".to_owned()], &ids, &BTreeSet::new(), 4
        );
        assert!(matches!(ineligible, Err(ApiContractError::IneligibleDesign(_))));
        let prior = BTreeSet::from(["d-01".to_owned()]);
        let repeated = validate_catalog_submission(
            &card, 1, &["d-01".to_owned(), "d-02".to_owned()], &ids, &prior, 4
        );
        assert!(matches!(repeated, Err(ApiContractError::AlreadyMeasured(_))));
        let over_budget = validate_catalog_submission(
            &card, 1, &["d-01".to_owned(), "d-02".to_owned()], &ids, &BTreeSet::new(), 1
        );
        assert!(matches!(over_budget, Err(ApiContractError::InsufficientRemainingBudget { .. })));
    }

    #[test]
    fn accepts_completed_job_and_checks_result_identity_set() {
        let card = campaign();
        let job = completed_job();
        assert!(job.validate_against(&card).is_ok());
        assert!(job.advances_stage(&card).expect("completed job"));
    }

    #[test]
    fn rejects_completed_job_with_unknown_result_identity() {
        let card = campaign();
        let mut value: Value = serde_json::from_str(COMPLETED_JOB).expect("job JSON parses");
        value["job"]["results"][1]["design_id"] = Value::String("not-submitted".to_owned());
        let bad = JobResponseV1::parse(&value.to_string()).expect("mutated job parses");
        assert!(matches!(
            bad.validate_against(&card),
            Err(ApiContractError::ResultIdentityMismatch(_))
        ));
    }

    #[test]
    fn pending_job_never_advances_a_stage() {
        let card = campaign();
        let mut value: Value = serde_json::from_str(COMPLETED_JOB).expect("job JSON parses");
        value["job"]["status"] = Value::String("running".to_owned());
        value["job"]["results"] = Value::Null;
        let pending = JobResponseV1::parse(&value.to_string()).expect("pending job parses");
        assert!(!pending.advances_stage(&card).expect("pending is valid"));
    }

    #[test]
    fn rejects_unknown_api_fields_fail_closed() {
        let mut value: Value = serde_json::from_str(CAMPAIGN).expect("campaign JSON parses");
        value["backend"] = Value::String("replay".to_owned());
        assert!(CampaignCardV1::parse(&value.to_string()).is_err());
    }
}
