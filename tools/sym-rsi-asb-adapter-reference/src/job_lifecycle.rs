//! Pinned AS-Bench Lab API v1 job lifecycle semantics.
//!
//! Pending and failed jobs never expose usable measurements or advance the
//! campaign stage. Only a completed job with one result row per requested
//! design is treated as a completed measurement batch.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Completed,
    Failed,
}

impl JobStatus {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed)
    }

    pub fn has_completed_measurements(self) -> bool {
        matches!(self, Self::Completed)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LabJob {
    pub job_id: String,
    pub stage: u32,
    pub status: JobStatus,
    pub designs: Vec<String>,
    /// AS-Bench returns null for queued/running/failed jobs and rows on completion.
    pub results: Option<Vec<Value>>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobContractError {
    EmptyJobId,
    InvalidStage,
    EmptyDesignBatch,
    EmptyDesignId,
    DuplicateDesignId(String),
    PendingJobHasResults,
    CompletedJobMissingResults,
    CompletedResultCountMismatch { designs: usize, results: usize },
    FailedJobHasResults,
}

impl std::fmt::Display for JobContractError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for JobContractError {}

impl LabJob {
    pub fn validate(&self) -> Result<(), JobContractError> {
        if self.job_id.trim().is_empty() {
            return Err(JobContractError::EmptyJobId);
        }
        if self.stage == 0 {
            return Err(JobContractError::InvalidStage);
        }
        if self.designs.is_empty() {
            return Err(JobContractError::EmptyDesignBatch);
        }

        let mut seen = BTreeSet::new();
        for design in &self.designs {
            if design.trim().is_empty() {
                return Err(JobContractError::EmptyDesignId);
            }
            if !seen.insert(design.as_str()) {
                return Err(JobContractError::DuplicateDesignId(design.clone()));
            }
        }

        match (self.status, self.results.as_ref()) {
            (JobStatus::Queued | JobStatus::Running, Some(_)) => {
                Err(JobContractError::PendingJobHasResults)
            }
            (JobStatus::Queued | JobStatus::Running, None) => Ok(()),
            (JobStatus::Completed, None) => Err(JobContractError::CompletedJobMissingResults),
            (JobStatus::Completed, Some(results)) if results.len() != self.designs.len() => {
                Err(JobContractError::CompletedResultCountMismatch {
                    designs: self.designs.len(),
                    results: results.len(),
                })
            }
            (JobStatus::Completed, Some(_)) => Ok(()),
            (JobStatus::Failed, Some(_)) => Err(JobContractError::FailedJobHasResults),
            (JobStatus::Failed, None) => Ok(()),
        }
    }

    /// Results are returned only for a valid, completed job.
    pub fn completed_measurements(&self) -> Result<Option<&[Value]>, JobContractError> {
        self.validate()?;
        match (self.status, self.results.as_ref()) {
            (JobStatus::Completed, Some(results)) => Ok(Some(results.as_slice())),
            _ => Ok(None),
        }
    }

    /// A job advances the stage only when a complete measurement batch is present.
    pub fn advances_stage(&self) -> Result<bool, JobContractError> {
        Ok(self.completed_measurements()?.is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(status: JobStatus, results: Option<Vec<Value>>) -> LabJob {
        LabJob {
            job_id: "job-0001".to_owned(),
            stage: 1,
            status,
            designs: vec!["design-a".to_owned(), "design-b".to_owned()],
            results,
            error: None,
        }
    }

    fn two_results() -> Vec<Value> {
        vec![
            serde_json::json!({"design_id":"design-a","measurement":1.0}),
            serde_json::json!({"design_id":"design-b","measurement":2.0}),
        ]
    }

    #[test]
    fn queued_job_is_pending_and_does_not_advance_stage() {
        let pending = job(JobStatus::Queued, None);
        assert!(pending.validate().is_ok());
        assert_eq!(pending.completed_measurements().expect("valid job"), None);
        assert!(!pending.advances_stage().expect("valid job"));
        assert!(!pending.status.is_terminal());
    }

    #[test]
    fn running_job_is_pending_and_does_not_advance_stage() {
        let pending = job(JobStatus::Running, None);
        assert!(!pending.advances_stage().expect("valid job"));
        assert!(!pending.status.is_terminal());
    }

    #[test]
    fn failed_job_does_not_advance_stage() {
        let failed = job(JobStatus::Failed, None);
        assert!(failed.status.is_terminal());
        assert!(!failed.advances_stage().expect("valid failed job"));
    }

    #[test]
    fn completed_job_advances_only_with_one_row_per_design() {
        let completed = job(JobStatus::Completed, Some(two_results()));
        assert!(completed.advances_stage().expect("complete batch"));
        assert_eq!(completed.completed_measurements().expect("complete batch").unwrap().len(), 2);
    }

    #[test]
    fn rejects_completed_job_without_results() {
        let completed = job(JobStatus::Completed, None);
        assert_eq!(
            completed.validate().expect_err("completed job needs results"),
            JobContractError::CompletedJobMissingResults
        );
    }

    #[test]
    fn rejects_completed_batch_with_wrong_result_count() {
        let completed = job(JobStatus::Completed, Some(vec![serde_json::json!({"measurement": 1})]));
        assert_eq!(
            completed.validate().expect_err("result count must match batch"),
            JobContractError::CompletedResultCountMismatch { designs: 2, results: 1 }
        );
    }

    #[test]
    fn rejects_pending_job_with_results() {
        let pending = job(JobStatus::Running, Some(two_results()));
        assert_eq!(
            pending.validate().expect_err("pending jobs cannot expose completed results"),
            JobContractError::PendingJobHasResults
        );
    }

    #[test]
    fn rejects_failed_job_with_results() {
        let failed = job(JobStatus::Failed, Some(two_results()));
        assert_eq!(
            failed.validate().expect_err("failed job cannot be a completed batch"),
            JobContractError::FailedJobHasResults
        );
    }

    #[test]
    fn rejects_repeated_design_ids() {
        let mut repeated = job(JobStatus::Queued, None);
        repeated.designs[1] = repeated.designs[0].clone();
        assert_eq!(
            repeated.validate().expect_err("campaign cannot remeasure design"),
            JobContractError::DuplicateDesignId("design-a".to_owned())
        );
    }
}
