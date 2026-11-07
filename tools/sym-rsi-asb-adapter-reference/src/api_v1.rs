//! Transport-independent typed contracts for Autonomous Science Bench Lab API v1.
//!
//! This module models only the agent-visible public API. It deliberately
//! refuses a campaign card containing a private `backend` field, preserving
//! the upstream boundary between public campaign information and lab internals.
//! The remote lab remains authoritative for catalog eligibility, stage order,
//! budget enforcement, ledger state and all verification.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

use crate::{JobContractError, JobStatus, LabJob};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct PublicBudget {
    pub unit: String,
    pub total: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct PublicDesignSpace {
    pub kind: String,
    pub shared_pool: Option<bool>,
    pub format: Option<String>,
    pub max_bytes: Option<u64>,
    pub description: Option<String>,
    pub rules: Option<String>,
    pub shape: Option<Vec<u64>>,
    pub id_format: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct PublicMeasurement {
    pub name: String,
    pub description: String,
    pub higher_is_better: Option<bool>,
    pub kind: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct PublicStage {
    pub stage: u32,
    pub label: String,
    pub batch_size: u64,
    pub catalog_size: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct PublicDeliverable {
    pub kind: String,
    pub format_version: u32,
    pub max_bytes: u64,
    pub input_field: String,
    pub description: String,
    pub requires: String,
    pub content: Option<String>,
    pub requires_stages: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct CampaignCard {
    pub ok: bool,
    pub api_version: u32,
    pub campaign_id: String,
    pub source: String,
    pub objective: String,
    pub design_id_field: String,
    pub design_space: Option<PublicDesignSpace>,
    pub measurements: Vec<PublicMeasurement>,
    pub stages: Vec<PublicStage>,
    pub budget: PublicBudget,
    pub deliverable: PublicDeliverable,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct StatusBudget {
    pub unit: String,
    pub total: u64,
    pub spent: u64,
    pub remaining: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct CampaignStatus {
    pub ok: bool,
    pub api_version: u32,
    pub campaign_id: String,
    pub stages_total: u32,
    pub stages_completed: u32,
    pub next_stage: Option<u32>,
    pub active_job: Option<String>,
    pub budget: StatusBudget,
    pub deliverable_submitted: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct CatalogResponse {
    pub ok: bool,
    pub stage: u32,
    pub designs: Vec<std::collections::BTreeMap<String, String>>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct SubmitExperimentRequest {
    pub designs: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SubmitExperimentResponse {
    pub ok: bool,
    pub replayed: bool,
    pub job: LabJob,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct JobSummary {
    pub job_id: String,
    pub stage: u32,
    pub status: JobStatus,
    /// The list endpoint returns a count; the per-job endpoint returns design IDs.
    pub designs: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ListJobsResponse {
    pub ok: bool,
    pub jobs: Vec<JobSummary>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct TextDeliverableRequest {
    pub kind: String,
    pub format_version: u32,
    pub source: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct BinaryDeliverableRequest {
    pub kind: String,
    pub format_version: u32,
    pub encoding: String,
    pub data: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum DeliverableRequest {
    Text(TextDeliverableRequest),
    Binary(BinaryDeliverableRequest),
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct DeliverableResponse {
    pub ok: bool,
    pub status: String,
    pub seq: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct LabApiError {
    pub ok: bool,
    pub error: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApiContractError {
    Json(String),
    PrivateBackendLeaked,
    ApiNotOk,
    UnsupportedApiVersion(u32),
    EmptyIdentifier(&'static str),
    EmptyCollection(&'static str),
    InvalidStageSequence,
    InvalidBatchSize,
    BudgetDoesNotMatchStages,
    CampaignStatusMismatch(&'static str),
    BudgetAccountingMismatch,
    StageMismatch { expected: u32, actual: u32 },
    UnknownStage(u32),
    CatalogSizeMismatch { expected: u64, actual: usize },
    MissingDesignIdField(String),
    DuplicateCatalogDesign(String),
    DuplicateSelectedDesign(String),
    SelectedDesignNotInCatalog(String),
    DesignAlreadyMeasured(String),
    WrongDeliverableKind,
    WrongDeliverableVersion,
    DeliverableContentMismatch,
    EmptyDeliverable,
    DeliverableTooLarge { max_bytes: u64, actual_bytes: usize },
    InvalidDeliverableEncoding,
    InvalidDeliverableReceipt,
    InvalidJob(JobContractError),
}

impl std::fmt::Display for ApiContractError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ApiContractError {}

impl From<JobContractError> for ApiContractError {
    fn from(error: JobContractError) -> Self {
        Self::InvalidJob(error)
    }
}

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

impl CampaignCard {
    /// Parse the public campaign card and reject private backend leakage.
    pub fn parse_public(contents: &str) -> Result<Self, ApiContractError> {
        let raw: Value =
            serde_json::from_str(contents).map_err(|error| ApiContractError::Json(error.to_string()))?;
        if raw.get("backend").is_some() {
            return Err(ApiContractError::PrivateBackendLeaked);
        }
        let card: Self =
            serde_json::from_value(raw).map_err(|error| ApiContractError::Json(error.to_string()))?;
        card.validate()?;
        Ok(card)
    }

    pub fn validate(&self) -> Result<(), ApiContractError> {
        if !self.ok {
            return Err(ApiContractError::ApiNotOk);
        }
        if self.api_version != 1 {
            return Err(ApiContractError::UnsupportedApiVersion(self.api_version));
        }
        if !non_empty(&self.campaign_id) {
            return Err(ApiContractError::EmptyIdentifier("campaign_id"));
        }
        if !non_empty(&self.design_id_field) {
            return Err(ApiContractError::EmptyIdentifier("design_id_field"));
        }
        if self.stages.is_empty() {
            return Err(ApiContractError::EmptyCollection("stages"));
        }
        if self.measurements.is_empty() {
            return Err(ApiContractError::EmptyCollection("measurements"));
        }
        if self.stages.iter().any(|stage| stage.batch_size == 0 || !non_empty(&stage.label)) {
            return Err(ApiContractError::InvalidBatchSize);
        }
        for (index, stage) in self.stages.iter().enumerate() {
            if stage.stage as usize != index + 1 {
                return Err(ApiContractError::InvalidStageSequence);
            }
        }
        let total = self.stages.iter().try_fold(0u64, |sum, stage| {
            sum.checked_add(stage.batch_size).ok_or(ApiContractError::BudgetDoesNotMatchStages)
        })?;
        if total != self.budget.total {
            return Err(ApiContractError::BudgetDoesNotMatchStages);
        }
        if !non_empty(&self.source) || !non_empty(&self.objective) {
            return Err(ApiContractError::EmptyIdentifier("source/objective"));
        }
        if self.measurements.iter().any(|measurement| !non_empty(&measurement.name)) {
            return Err(ApiContractError::EmptyIdentifier("measurement.name"));
        }
        if !non_empty(&self.budget.unit)
            || !non_empty(&self.deliverable.kind)
            || !non_empty(&self.deliverable.requires)
            || self.deliverable.format_version == 0
            || self.deliverable.max_bytes == 0
            || self.deliverable.content.as_deref().is_some_and(|value| value != "binary")
        {
            return Err(ApiContractError::EmptyIdentifier("budget/deliverable"));
        }
        if let Some(space) = &self.design_space {
            if !non_empty(&space.kind) {
                return Err(ApiContractError::EmptyIdentifier("design_space.kind"));
            }
            if space.kind == "catalog" && space.shared_pool == Some(true) {
                for stage in &self.stages {
                    if stage.catalog_size.unwrap_or(0) == 0 {
                        return Err(ApiContractError::EmptyCollection("shared-pool catalog"));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn stage(&self, stage_number: u32) -> Option<&PublicStage> {
        self.stages.iter().find(|stage| stage.stage == stage_number)
    }
}

impl CampaignStatus {
    pub fn validate_against(&self, campaign: &CampaignCard) -> Result<(), ApiContractError> {
        if !self.ok {
            return Err(ApiContractError::ApiNotOk);
        }
        if self.api_version != 1 {
            return Err(ApiContractError::UnsupportedApiVersion(self.api_version));
        }
        if self.campaign_id != campaign.campaign_id {
            return Err(ApiContractError::CampaignStatusMismatch("campaign_id"));
        }
        if self.stages_total as usize != campaign.stages.len()
            || self.stages_completed > self.stages_total
        {
            return Err(ApiContractError::CampaignStatusMismatch("stage_counts"));
        }
        let expected_next = if self.stages_completed < self.stages_total {
            Some(self.stages_completed + 1)
        } else {
            None
        };
        if self.next_stage != expected_next {
            return Err(ApiContractError::CampaignStatusMismatch("next_stage"));
        }
        if self.budget.total != campaign.budget.total {
            return Err(ApiContractError::CampaignStatusMismatch("budget.total"));
        }
        if self.budget.spent.checked_add(self.budget.remaining) != Some(self.budget.total) {
            return Err(ApiContractError::BudgetAccountingMismatch);
        }
        if self.active_job.as_ref().is_some_and(|job| !non_empty(job)) {
            return Err(ApiContractError::EmptyIdentifier("active_job"));
        }
        Ok(())
    }
}

impl CatalogResponse {
    /// Validate a catalog against the agent-visible campaign card and stage.
    pub fn validate_for_stage(
        &self,
        campaign: &CampaignCard,
        stage_number: u32,
    ) -> Result<(), ApiContractError> {
        if !self.ok {
            return Err(ApiContractError::ApiNotOk);
        }
        if self.stage != stage_number {
            return Err(ApiContractError::StageMismatch {
                expected: stage_number,
                actual: self.stage,
            });
        }
        let stage = campaign
            .stage(stage_number)
            .ok_or(ApiContractError::UnknownStage(stage_number))?;
        if let Some(expected) = stage.catalog_size {
            if expected != self.designs.len() as u64 {
                return Err(ApiContractError::CatalogSizeMismatch {
                    expected,
                    actual: self.designs.len(),
                });
            }
        }
        let mut ids = BTreeSet::new();
        for row in &self.designs {
            let design_id = row
                .get(&campaign.design_id_field)
                .filter(|value| non_empty(value))
                .ok_or_else(|| ApiContractError::MissingDesignIdField(campaign.design_id_field.clone()))?;
            if !ids.insert(design_id.as_str()) {
                return Err(ApiContractError::DuplicateCatalogDesign(design_id.clone()));
            }
        }
        Ok(())
    }

    pub fn design_ids<'a>(&'a self, field: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        self.designs.iter().filter_map(move |row| row.get(field).map(String::as_str))
    }
}

impl SubmitExperimentRequest {
    /// Client-side fail-fast check. The lab is still authoritative and must revalidate.
    pub fn validate_against_catalog(
        &self,
        catalog: &CatalogResponse,
        stage: &PublicStage,
        already_measured: &BTreeSet<String>,
        design_id_field: &str,
    ) -> Result<(), ApiContractError> {
        if self.designs.len() as u64 != stage.batch_size {
            return Err(ApiContractError::InvalidBatchSize);
        }
        if !catalog.ok {
            return Err(ApiContractError::ApiNotOk);
        }
        if catalog.stage != stage.stage {
            return Err(ApiContractError::StageMismatch {
                expected: stage.stage,
                actual: catalog.stage,
            });
        }
        let mut allowed = BTreeSet::new();
        for design_id in catalog.design_ids(design_id_field) {
            if !non_empty(design_id) {
                return Err(ApiContractError::MissingDesignIdField(design_id_field.to_owned()));
            }
            if !allowed.insert(design_id) {
                return Err(ApiContractError::DuplicateCatalogDesign(design_id.to_owned()));
            }
        }
        let mut selected = BTreeSet::new();
        for design in &self.designs {
            if !non_empty(design) {
                return Err(ApiContractError::EmptyIdentifier("selected design"));
            }
            if !selected.insert(design.as_str()) {
                return Err(ApiContractError::DuplicateSelectedDesign(design.clone()));
            }
            if !allowed.contains(design.as_str()) {
                return Err(ApiContractError::SelectedDesignNotInCatalog(design.clone()));
            }
            if already_measured.contains(design) {
                return Err(ApiContractError::DesignAlreadyMeasured(design.clone()));
            }
        }
        Ok(())
    }
}

impl SubmitExperimentResponse {
    pub fn validate(&self) -> Result<(), ApiContractError> {
        if !self.ok {
            return Err(ApiContractError::ApiNotOk);
        }
        self.job.validate()?;
        Ok(())
    }
}

impl ListJobsResponse {
    pub fn validate(&self) -> Result<(), ApiContractError> {
        if !self.ok {
            return Err(ApiContractError::ApiNotOk);
        }
        let mut ids = BTreeSet::new();
        for job in &self.jobs {
            if !non_empty(&job.job_id) || job.stage == 0 || job.designs == 0 {
                return Err(ApiContractError::EmptyIdentifier("job summary"));
            }
            if !ids.insert(job.job_id.as_str()) {
                return Err(ApiContractError::DuplicateSelectedDesign(job.job_id.clone()));
            }
        }
        Ok(())
    }
}

impl DeliverableRequest {
    pub fn validate_against(&self, spec: &PublicDeliverable) -> Result<(), ApiContractError> {
        match self {
            Self::Text(request) => {
                if spec.content.as_deref() == Some("binary") {
                    return Err(ApiContractError::DeliverableContentMismatch);
                }
                if request.kind != spec.kind {
                    return Err(ApiContractError::WrongDeliverableKind);
                }
                if request.format_version != spec.format_version {
                    return Err(ApiContractError::WrongDeliverableVersion);
                }
                if request.source.is_empty() {
                    return Err(ApiContractError::EmptyDeliverable);
                }
                if request.source.len() > spec.max_bytes as usize {
                    return Err(ApiContractError::DeliverableTooLarge {
                        max_bytes: spec.max_bytes,
                        actual_bytes: request.source.len(),
                    });
                }
            }
            Self::Binary(request) => {
                if spec.content.as_deref() != Some("binary") {
                    return Err(ApiContractError::DeliverableContentMismatch);
                }
                if request.kind != spec.kind {
                    return Err(ApiContractError::WrongDeliverableKind);
                }
                if request.format_version != spec.format_version {
                    return Err(ApiContractError::WrongDeliverableVersion);
                }
                if !matches!(request.encoding.as_str(), "base64" | "zlib+base64") || request.data.is_empty() {
                    return Err(ApiContractError::InvalidDeliverableEncoding);
                }
            }
        }
        Ok(())
    }
}

impl DeliverableResponse {
    pub fn validate(&self) -> Result<(), ApiContractError> {
        if !self.ok || self.status != "received" || self.seq == 0
            || self.sha256.len() != 64
            || !self.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(ApiContractError::InvalidDeliverableReceipt);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn campaign_card() -> CampaignCard {
        CampaignCard {
            ok: true,
            api_version: 1,
            campaign_id: "deepmind/nuclease-active-learning".to_owned(),
            source: "deepmind".to_owned(),
            objective: "Discover and rank nuclease variants".to_owned(),
            design_id_field: "variant_id".to_owned(),
            design_space: Some(PublicDesignSpace {
                kind: "catalog".to_owned(),
                shared_pool: Some(true),
                format: None,
                max_bytes: None,
                description: Some("shared pool".to_owned()),
                rules: Some("each design once".to_owned()),
                shape: None,
                id_format: None,
            }),
            measurements: vec![PublicMeasurement {
                name: "activity_class".to_owned(),
                description: "ordered class".to_owned(),
                higher_is_better: Some(true),
                kind: None,
            }],
            stages: vec![
                PublicStage { stage: 1, label: "round 1".to_owned(), batch_size: 2, catalog_size: Some(3) },
                PublicStage { stage: 2, label: "round 2".to_owned(), batch_size: 2, catalog_size: Some(3) },
            ],
            budget: PublicBudget { unit: "measured designs".to_owned(), total: 4 },
            deliverable: PublicDeliverable {
                kind: "js-predictor".to_owned(),
                format_version: 1,
                max_bytes: 4_194_304,
                input_field: "sequence".to_owned(),
                description: "deterministic predictor".to_owned(),
                requires: "all stages completed".to_owned(),
                content: None,
                requires_stages: None,
            },
        }
    }

    fn catalog() -> CatalogResponse {
        CatalogResponse {
            ok: true,
            stage: 1,
            designs: vec![
                [("variant_id".to_owned(), "a".to_owned())].into_iter().collect(),
                [("variant_id".to_owned(), "b".to_owned())].into_iter().collect(),
                [("variant_id".to_owned(), "c".to_owned())].into_iter().collect(),
            ],
        }
    }

    #[test]
    fn accepts_valid_public_campaign_card_contract() {
        assert!(campaign_card().validate().is_ok());
    }

    #[test]
    fn rejects_private_backend_leak_in_public_campaign_card() {
        let raw = serde_json::json!({
            "ok": true,
            "api_version": 1,
            "campaign_id": "deepmind/nuclease-active-learning",
            "backend": {"kind":"replay"}
        });
        assert_eq!(
            CampaignCard::parse_public(&raw.to_string()).expect_err("backend is private"),
            ApiContractError::PrivateBackendLeaked
        );
    }

    #[test]
    fn accepts_public_campaign_card_without_backend() {
        let raw = serde_json::json!({
            "ok": true,
            "api_version": 1,
            "campaign_id": "deepmind/nuclease-active-learning",
            "source": "deepmind",
            "objective": "Discover variants",
            "design_id_field": "variant_id",
            "design_space": {"kind":"catalog","shared_pool":true},
            "measurements": [{"name":"activity_class","description":"ordered","higher_is_better":true}],
            "stages": [
                {"stage":1,"label":"round 1","batch_size":2,"catalog_size":3},
                {"stage":2,"label":"round 2","batch_size":2,"catalog_size":3}
            ],
            "budget": {"unit":"measured designs","total":4},
            "deliverable": {"kind":"js-predictor","format_version":1,"max_bytes":4096,"input_field":"sequence","description":"predict","requires":"all stages completed"}
        });
        assert!(CampaignCard::parse_public(&raw.to_string()).is_ok());
    }

    #[test]
    fn status_requires_consistent_stages_and_budget_accounting() {
        let card = campaign_card();
        let status = CampaignStatus {
            ok: true,
            api_version: 1,
            campaign_id: card.campaign_id.clone(),
            stages_total: 2,
            stages_completed: 1,
            next_stage: Some(2),
            active_job: None,
            budget: StatusBudget {
                unit: "measured designs".to_owned(),
                total: 4,
                spent: 2,
                remaining: 2,
            },
            deliverable_submitted: false,
        };
        assert!(status.validate_against(&card).is_ok());
    }

    #[test]
    fn rejects_status_budget_mismatch() {
        let card = campaign_card();
        let status = CampaignStatus {
            ok: true,
            api_version: 1,
            campaign_id: card.campaign_id.clone(),
            stages_total: 2,
            stages_completed: 0,
            next_stage: Some(1),
            active_job: None,
            budget: StatusBudget {
                unit: "measured designs".to_owned(),
                total: 4,
                spent: 3,
                remaining: 0,
            },
            deliverable_submitted: false,
        };
        assert_eq!(
            status.validate_against(&card).expect_err("budget accounting mismatch"),
            ApiContractError::BudgetAccountingMismatch
        );
    }

    #[test]
    fn validates_catalog_identity_and_unique_rows() {
        let card = campaign_card();
        assert!(catalog().validate_for_stage(&card, 1).is_ok());
    }

    #[test]
    fn rejects_duplicate_catalog_design_ids() {
        let card = campaign_card();
        let mut duplicate = catalog();
        duplicate.designs[2].insert("variant_id".to_owned(), "a".to_owned());
        assert!(matches!(
            duplicate.validate_for_stage(&card, 1),
            Err(ApiContractError::DuplicateCatalogDesign(id)) if id == "a"
        ));
    }

    #[test]
    fn rejects_duplicate_or_previously_measured_selections() {
        let stage = campaign_card().stages[0].clone();
        let request = SubmitExperimentRequest { designs: vec!["a".to_owned(), "a".to_owned()] };
        assert_eq!(
            request.validate_against_catalog(&catalog(), &stage, &BTreeSet::new(), "variant_id")
                .expect_err("duplicate designs"),
            ApiContractError::DuplicateSelectedDesign("a".to_owned())
        );
        let request = SubmitExperimentRequest { designs: vec!["a".to_owned(), "b".to_owned()] };
        let already = BTreeSet::from(["b".to_owned()]);
        assert_eq!(
            request.validate_against_catalog(&catalog(), &stage, &already, "variant_id")
                .expect_err("already measured design"),
            ApiContractError::DesignAlreadyMeasured("b".to_owned())
        );
    }

    #[test]
    fn rejects_selection_outside_catalog() {
        let stage = campaign_card().stages[0].clone();
        let request = SubmitExperimentRequest { designs: vec!["a".to_owned(), "unknown".to_owned()] };
        assert_eq!(
            request.validate_against_catalog(&catalog(), &stage, &BTreeSet::new(), "variant_id")
                .expect_err("unknown design"),
            ApiContractError::SelectedDesignNotInCatalog("unknown".to_owned())
        );
    }

    #[test]
    fn deliverable_request_uses_exact_public_card_contract() {
        let spec = campaign_card().deliverable;
        let request = DeliverableRequest::Text(TextDeliverableRequest {
            kind: "js-predictor".to_owned(),
            format_version: 1,
            source: "function predict(x) { return 0; }".to_owned(),
        });
        assert!(request.validate_against(&spec).is_ok());
        let invalid = DeliverableRequest::Text(TextDeliverableRequest {
            kind: "wrong-kind".to_owned(),
            format_version: 1,
            source: "predict".to_owned(),
        });
        assert_eq!(invalid.validate_against(&spec).expect_err("wrong kind"), ApiContractError::WrongDeliverableKind);
    }

    #[test]
    fn deliverable_receipt_requires_sha256_and_received_state() {
        let receipt = DeliverableResponse {
            ok: true,
            status: "received".to_owned(),
            seq: 1,
            sha256: "a".repeat(64),
        };
        assert!(receipt.validate().is_ok());
    }
}
