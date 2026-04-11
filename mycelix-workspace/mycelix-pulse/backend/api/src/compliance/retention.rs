// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Track BZ: Data Retention Policies
// Automated data lifecycle management with configurable retention rules

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{warn, debug};
use uuid::Uuid;

/// Data category for retention classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DataCategory {
    /// Primary email content
    EmailContent,
    /// Email attachments
    Attachments,
    /// Email metadata (headers, timestamps)
    EmailMetadata,
    /// User account data
    AccountData,
    /// Authentication logs
    AuthLogs,
    /// Audit trails
    AuditLogs,
    /// Analytics data
    Analytics,
    /// Spam/quarantine
    SpamQuarantine,
    /// Deleted items (soft delete)
    DeletedItems,
    /// Draft emails
    Drafts,
    /// Trust score history
    TrustHistory,
    /// Encryption keys (expired)
    ExpiredKeys,
    /// Session data
    SessionData,
    /// Search indexes
    SearchIndexes,
    /// Backup archives
    BackupArchives,
}

/// Retention action to take when policy triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetentionAction {
    /// Permanently delete data
    Delete,
    /// Move to archive storage (cold tier)
    Archive,
    /// Anonymize personal identifiers
    Anonymize,
    /// Compress for long-term storage
    Compress,
    /// Export before deletion
    ExportThenDelete,
    /// Mark for manual review
    FlagForReview,
}

/// Legal hold preventing data deletion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegalHold {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub matter_id: Option<String>,
    pub custodians: Vec<Uuid>,
    pub query: Option<String>,
    pub created_at: DateTime<Utc>,
    pub created_by: Uuid,
    pub expires_at: Option<DateTime<Utc>>,
    pub active: bool,
}

/// A retention policy rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub category: DataCategory,
    pub retention_days: i64,
    pub action: RetentionAction,
    pub priority: i32,
    pub conditions: RetentionConditions,
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Conditions for policy application
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RetentionConditions {
    /// Only apply to specific users
    pub user_ids: Option<Vec<Uuid>>,
    /// Only apply to specific domains
    pub domains: Option<Vec<String>>,
    /// Minimum trust score threshold
    pub min_trust_score: Option<f64>,
    /// Maximum trust score threshold
    pub max_trust_score: Option<f64>,
    /// Label/folder filter
    pub labels: Option<Vec<String>>,
    /// Size threshold in bytes
    pub min_size_bytes: Option<i64>,
    /// Exclude starred/important items
    pub exclude_starred: bool,
    /// Exclude items with attachments
    pub exclude_with_attachments: bool,
}

/// Retention job execution record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionJob {
    pub id: Uuid,
    pub policy_id: Uuid,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub status: JobStatus,
    pub items_processed: u64,
    pub items_affected: u64,
    pub bytes_reclaimed: u64,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Data item marked for retention action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionCandidate {
    pub id: Uuid,
    pub category: DataCategory,
    pub owner_id: Uuid,
    pub created_at: DateTime<Utc>,
    pub last_accessed: Option<DateTime<Utc>>,
    pub size_bytes: i64,
    pub policy_id: Uuid,
    pub scheduled_action: RetentionAction,
    pub scheduled_at: DateTime<Utc>,
    pub legal_holds: Vec<Uuid>,
}

/// Retention policy service
pub struct RetentionService {
    policies: HashMap<Uuid, RetentionPolicy>,
    legal_holds: HashMap<Uuid, LegalHold>,
    jobs: HashMap<Uuid, RetentionJob>,
    candidates: Vec<RetentionCandidate>,
}

impl Default for RetentionService {
    fn default() -> Self {
        Self::new()
    }
}

impl RetentionService {
    pub fn new() -> Self {
        let mut service = Self {
            policies: HashMap::new(),
            legal_holds: HashMap::new(),
            jobs: HashMap::new(),
            candidates: Vec::new(),
        };

        // Initialize with default policies
        service.create_default_policies();
        service
    }

    /// Create sensible default retention policies
    fn create_default_policies(&mut self) {
        let defaults = vec![
            // Deleted items: 30 days then permanent delete
            RetentionPolicy {
                id: Uuid::new_v4(),
                name: "Deleted Items Cleanup".to_string(),
                description: "Permanently delete items in trash after 30 days".to_string(),
                category: DataCategory::DeletedItems,
                retention_days: 30,
                action: RetentionAction::Delete,
                priority: 100,
                conditions: RetentionConditions::default(),
                enabled: true,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
            // Spam: 14 days
            RetentionPolicy {
                id: Uuid::new_v4(),
                name: "Spam Cleanup".to_string(),
                description: "Delete spam after 14 days".to_string(),
                category: DataCategory::SpamQuarantine,
                retention_days: 14,
                action: RetentionAction::Delete,
                priority: 90,
                conditions: RetentionConditions::default(),
                enabled: true,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
            // Drafts: 90 days if unmodified
            RetentionPolicy {
                id: Uuid::new_v4(),
                name: "Stale Drafts Cleanup".to_string(),
                description: "Delete unmodified drafts after 90 days".to_string(),
                category: DataCategory::Drafts,
                retention_days: 90,
                action: RetentionAction::Delete,
                priority: 80,
                conditions: RetentionConditions::default(),
                enabled: true,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
            // Session data: 7 days
            RetentionPolicy {
                id: Uuid::new_v4(),
                name: "Session Cleanup".to_string(),
                description: "Clear expired session data after 7 days".to_string(),
                category: DataCategory::SessionData,
                retention_days: 7,
                action: RetentionAction::Delete,
                priority: 100,
                conditions: RetentionConditions::default(),
                enabled: true,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
            // Auth logs: 1 year then archive
            RetentionPolicy {
                id: Uuid::new_v4(),
                name: "Auth Log Archival".to_string(),
                description: "Archive authentication logs after 1 year".to_string(),
                category: DataCategory::AuthLogs,
                retention_days: 365,
                action: RetentionAction::Archive,
                priority: 50,
                conditions: RetentionConditions::default(),
                enabled: true,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
            // Audit logs: 7 years (compliance)
            RetentionPolicy {
                id: Uuid::new_v4(),
                name: "Audit Log Retention".to_string(),
                description: "Retain audit logs for 7 years for compliance".to_string(),
                category: DataCategory::AuditLogs,
                retention_days: 2555, // ~7 years
                action: RetentionAction::Archive,
                priority: 10,
                conditions: RetentionConditions::default(),
                enabled: true,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
            // Analytics: 2 years then anonymize
            RetentionPolicy {
                id: Uuid::new_v4(),
                name: "Analytics Anonymization".to_string(),
                description: "Anonymize analytics data after 2 years".to_string(),
                category: DataCategory::Analytics,
                retention_days: 730,
                action: RetentionAction::Anonymize,
                priority: 60,
                conditions: RetentionConditions::default(),
                enabled: true,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
            // Old emails: Archive after 3 years
            RetentionPolicy {
                id: Uuid::new_v4(),
                name: "Email Archival".to_string(),
                description: "Archive old emails to cold storage after 3 years".to_string(),
                category: DataCategory::EmailContent,
                retention_days: 1095,
                action: RetentionAction::Archive,
                priority: 30,
                conditions: RetentionConditions {
                    exclude_starred: true,
                    ..Default::default()
                },
                enabled: true,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
            // Expired encryption keys: 90 days
            RetentionPolicy {
                id: Uuid::new_v4(),
                name: "Expired Key Cleanup".to_string(),
                description: "Delete expired encryption keys after 90 days".to_string(),
                category: DataCategory::ExpiredKeys,
                retention_days: 90,
                action: RetentionAction::Delete,
                priority: 70,
                conditions: RetentionConditions::default(),
                enabled: true,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
            // Backup archives: 1 year
            RetentionPolicy {
                id: Uuid::new_v4(),
                name: "Backup Rotation".to_string(),
                description: "Delete backup archives older than 1 year".to_string(),
                category: DataCategory::BackupArchives,
                retention_days: 365,
                action: RetentionAction::Delete,
                priority: 40,
                conditions: RetentionConditions::default(),
                enabled: true,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
        ];

        for policy in defaults {
            self.policies.insert(policy.id, policy);
        }
    }

    /// Create a new retention policy
    pub async fn create_policy(
        &mut self,
        name: String,
        description: String,
        category: DataCategory,
        retention_days: i64,
        action: RetentionAction,
        conditions: RetentionConditions,
    ) -> Result<RetentionPolicy, RetentionError> {
        if retention_days < 0 {
            return Err(RetentionError::InvalidRetentionPeriod);
        }

        let policy = RetentionPolicy {
            id: Uuid::new_v4(),
            name,
            description,
            category,
            retention_days,
            action,
            priority: 50,
            conditions,
            enabled: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        self.policies.insert(policy.id, policy.clone());
        Ok(policy)
    }

    /// Update existing policy
    pub async fn update_policy(
        &mut self,
        policy_id: Uuid,
        updates: PolicyUpdate,
    ) -> Result<RetentionPolicy, RetentionError> {
        let policy = self.policies.get_mut(&policy_id)
            .ok_or(RetentionError::PolicyNotFound)?;

        if let Some(name) = updates.name {
            policy.name = name;
        }
        if let Some(retention_days) = updates.retention_days {
            if retention_days < 0 {
                return Err(RetentionError::InvalidRetentionPeriod);
            }
            policy.retention_days = retention_days;
        }
        if let Some(action) = updates.action {
            policy.action = action;
        }
        if let Some(enabled) = updates.enabled {
            policy.enabled = enabled;
        }
        if let Some(priority) = updates.priority {
            policy.priority = priority;
        }

        policy.updated_at = Utc::now();

        Ok(policy.clone())
    }

    /// Delete a policy
    pub async fn delete_policy(&mut self, policy_id: Uuid) -> Result<(), RetentionError> {
        self.policies.remove(&policy_id)
            .ok_or(RetentionError::PolicyNotFound)?;
        Ok(())
    }

    /// Create a legal hold
    pub async fn create_legal_hold(
        &mut self,
        name: String,
        description: String,
        matter_id: Option<String>,
        custodians: Vec<Uuid>,
        query: Option<String>,
        created_by: Uuid,
        expires_at: Option<DateTime<Utc>>,
    ) -> Result<LegalHold, RetentionError> {
        let hold = LegalHold {
            id: Uuid::new_v4(),
            name,
            description,
            matter_id,
            custodians,
            query,
            created_at: Utc::now(),
            created_by,
            expires_at,
            active: true,
        };

        self.legal_holds.insert(hold.id, hold.clone());
        Ok(hold)
    }

    /// Release a legal hold
    pub async fn release_legal_hold(&mut self, hold_id: Uuid) -> Result<(), RetentionError> {
        let hold = self.legal_holds.get_mut(&hold_id)
            .ok_or(RetentionError::LegalHoldNotFound)?;

        hold.active = false;
        Ok(())
    }

    /// Check if item is under legal hold
    pub fn is_under_legal_hold(&self, owner_id: Uuid) -> bool {
        self.legal_holds.values().any(|hold| {
            hold.active && hold.custodians.contains(&owner_id)
        })
    }

    /// Get all active legal holds for a user
    pub fn get_active_holds(&self, user_id: Uuid) -> Vec<&LegalHold> {
        self.legal_holds.values()
            .filter(|hold| hold.active && hold.custodians.contains(&user_id))
            .collect()
    }

    /// Scan for retention candidates based on policies
    pub async fn scan_for_candidates(&mut self) -> Result<Vec<RetentionCandidate>, RetentionError> {
        let now = Utc::now();
        let mut candidates = Vec::new();

        for policy in self.policies.values().filter(|p| p.enabled) {
            let cutoff = now - Duration::days(policy.retention_days);

            // In real implementation, this would query the database
            // Here we simulate finding candidates
            let simulated_candidates = self.find_candidates_for_policy(policy, cutoff).await?;

            for mut candidate in simulated_candidates {
                // Check legal holds
                if self.is_under_legal_hold(candidate.owner_id) {
                    let holds: Vec<Uuid> = self.get_active_holds(candidate.owner_id)
                        .iter()
                        .map(|h| h.id)
                        .collect();
                    candidate.legal_holds = holds;
                    // Don't skip - just mark as held
                }

                candidate.policy_id = policy.id;
                candidate.scheduled_action = policy.action.clone();
                candidate.scheduled_at = now;
                candidates.push(candidate);
            }
        }

        self.candidates = candidates.clone();
        Ok(candidates)
    }

    /// Find candidates matching a specific policy
    async fn find_candidates_for_policy(
        &self,
        policy: &RetentionPolicy,
        _cutoff: DateTime<Utc>,
    ) -> Result<Vec<RetentionCandidate>, RetentionError> {
        // Simulated - in production would query database
        // Based on policy.category, policy.conditions, and cutoff date
        Ok(Vec::new())
    }

    /// Execute retention job for a policy
    pub async fn execute_retention_job(
        &mut self,
        policy_id: Uuid,
    ) -> Result<RetentionJob, RetentionError> {
        let policy = self.policies.get(&policy_id)
            .ok_or(RetentionError::PolicyNotFound)?
            .clone();

        if !policy.enabled {
            return Err(RetentionError::PolicyDisabled);
        }

        let job = RetentionJob {
            id: Uuid::new_v4(),
            policy_id,
            started_at: Utc::now(),
            completed_at: None,
            status: JobStatus::Running,
            items_processed: 0,
            items_affected: 0,
            bytes_reclaimed: 0,
            errors: Vec::new(),
        };

        self.jobs.insert(job.id, job.clone());

        // Process candidates for this policy
        let candidates: Vec<_> = self.candidates.iter()
            .filter(|c| c.policy_id == policy_id && c.legal_holds.is_empty())
            .cloned()
            .collect();

        let mut processed = 0u64;
        let mut affected = 0u64;
        let mut reclaimed = 0u64;
        let mut errors = Vec::new();

        for candidate in candidates {
            processed += 1;

            match self.execute_action(&candidate, &policy.action).await {
                Ok(bytes) => {
                    affected += 1;
                    reclaimed += bytes;
                }
                Err(e) => {
                    errors.push(format!("Failed to process {}: {}", candidate.id, e));
                }
            }
        }

        // Update job status
        if let Some(job) = self.jobs.get_mut(&job.id) {
            job.completed_at = Some(Utc::now());
            job.status = if errors.is_empty() {
                JobStatus::Completed
            } else if affected > 0 {
                JobStatus::Completed // Partial success
            } else {
                JobStatus::Failed
            };
            job.items_processed = processed;
            job.items_affected = affected;
            job.bytes_reclaimed = reclaimed;
            job.errors = errors;
        }

        Ok(self.jobs.get(&job.id).unwrap().clone())
    }

    /// Execute retention action on a candidate
    async fn execute_action(
        &self,
        candidate: &RetentionCandidate,
        action: &RetentionAction,
    ) -> Result<u64, RetentionError> {
        match action {
            RetentionAction::Delete => {
                // In production: DELETE FROM table WHERE id = candidate.id
                Ok(candidate.size_bytes as u64)
            }
            RetentionAction::Archive => {
                // In production: Move to cold storage (S3 Glacier, etc.)
                Ok(0) // Space reclaimed after archive transition
            }
            RetentionAction::Anonymize => {
                // In production: UPDATE table SET pii_fields = hash(pii_fields)
                Ok(0)
            }
            RetentionAction::Compress => {
                // In production: Compress and replace original
                Ok((candidate.size_bytes as f64 * 0.3) as u64) // ~70% compression
            }
            RetentionAction::ExportThenDelete => {
                // In production: Export to file, then delete
                Ok(candidate.size_bytes as u64)
            }
            RetentionAction::FlagForReview => {
                // In production: Add to review queue
                Ok(0)
            }
        }
    }

    /// Run all enabled retention policies
    pub async fn run_all_policies(&mut self) -> Vec<RetentionJob> {
        let policy_ids: Vec<Uuid> = self.policies.values()
            .filter(|p| p.enabled)
            .map(|p| p.id)
            .collect();

        let mut jobs = Vec::new();

        // First scan for candidates
        let _ = self.scan_for_candidates().await;

        // Then execute each policy
        let mut failed_policies = 0;
        for policy_id in &policy_ids {
            match self.execute_retention_job(policy_id.clone()).await {
                Ok(job) => jobs.push(job),
                Err(e) => {
                    failed_policies += 1;
                    warn!(
                        policy_id = %policy_id,
                        error = %e,
                        "Failed to execute retention job for policy"
                    );
                    continue;
                }
            }
        }

        if failed_policies > 0 {
            warn!(
                total_policies = policy_ids.len(),
                failed_policies = failed_policies,
                successful_jobs = jobs.len(),
                "Batch retention execution completed with failures"
            );
        } else {
            debug!(
                total_policies = policy_ids.len(),
                successful_jobs = jobs.len(),
                "Batch retention execution completed successfully"
            );
        }

        jobs
    }

    /// Get retention statistics
    pub fn get_statistics(&self) -> RetentionStatistics {
        let total_policies = self.policies.len();
        let enabled_policies = self.policies.values().filter(|p| p.enabled).count();
        let active_holds = self.legal_holds.values().filter(|h| h.active).count();
        let pending_candidates = self.candidates.len();
        let held_candidates = self.candidates.iter().filter(|c| !c.legal_holds.is_empty()).count();

        let completed_jobs = self.jobs.values()
            .filter(|j| j.status == JobStatus::Completed)
            .count();
        let total_reclaimed: u64 = self.jobs.values()
            .map(|j| j.bytes_reclaimed)
            .sum();

        RetentionStatistics {
            total_policies,
            enabled_policies,
            active_legal_holds: active_holds,
            pending_candidates,
            held_candidates,
            completed_jobs,
            total_bytes_reclaimed: total_reclaimed,
        }
    }

    /// List all policies
    pub fn list_policies(&self) -> Vec<&RetentionPolicy> {
        let mut policies: Vec<_> = self.policies.values().collect();
        policies.sort_by(|a, b| a.priority.cmp(&b.priority));
        policies
    }

    /// Get policy by ID
    pub fn get_policy(&self, policy_id: Uuid) -> Option<&RetentionPolicy> {
        self.policies.get(&policy_id)
    }

    /// List all legal holds
    pub fn list_legal_holds(&self) -> Vec<&LegalHold> {
        self.legal_holds.values().collect()
    }

    /// Get job history
    pub fn get_job_history(&self, limit: usize) -> Vec<&RetentionJob> {
        let mut jobs: Vec<_> = self.jobs.values().collect();
        jobs.sort_by(|a, b| b.started_at.cmp(&a.started_at));
        jobs.truncate(limit);
        jobs
    }
}

/// Policy update fields
#[derive(Debug, Default)]
pub struct PolicyUpdate {
    pub name: Option<String>,
    pub retention_days: Option<i64>,
    pub action: Option<RetentionAction>,
    pub enabled: Option<bool>,
    pub priority: Option<i32>,
}

/// Retention statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionStatistics {
    pub total_policies: usize,
    pub enabled_policies: usize,
    pub active_legal_holds: usize,
    pub pending_candidates: usize,
    pub held_candidates: usize,
    pub completed_jobs: usize,
    pub total_bytes_reclaimed: u64,
}

/// Retention errors
#[derive(Debug, Clone)]
pub enum RetentionError {
    PolicyNotFound,
    PolicyDisabled,
    LegalHoldNotFound,
    InvalidRetentionPeriod,
    ExecutionFailed(String),
    DatabaseError(String),
}

impl std::fmt::Display for RetentionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PolicyNotFound => write!(f, "Retention policy not found"),
            Self::PolicyDisabled => write!(f, "Retention policy is disabled"),
            Self::LegalHoldNotFound => write!(f, "Legal hold not found"),
            Self::InvalidRetentionPeriod => write!(f, "Invalid retention period"),
            Self::ExecutionFailed(msg) => write!(f, "Execution failed: {}", msg),
            Self::DatabaseError(msg) => write!(f, "Database error: {}", msg),
        }
    }
}

impl std::error::Error for RetentionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_default_policies() {
        let service = RetentionService::new();
        let policies = service.list_policies();

        assert!(!policies.is_empty());
        assert!(policies.iter().any(|p| p.category == DataCategory::DeletedItems));
        assert!(policies.iter().any(|p| p.category == DataCategory::SpamQuarantine));
    }

    #[tokio::test]
    async fn test_create_policy() {
        let mut service = RetentionService::new();

        let policy = service.create_policy(
            "Test Policy".to_string(),
            "A test retention policy".to_string(),
            DataCategory::EmailContent,
            180,
            RetentionAction::Archive,
            RetentionConditions::default(),
        ).await.unwrap();

        assert_eq!(policy.name, "Test Policy");
        assert_eq!(policy.retention_days, 180);
    }

    #[tokio::test]
    async fn test_legal_hold() {
        let mut service = RetentionService::new();
        let user_id = Uuid::new_v4();

        let hold = service.create_legal_hold(
            "Investigation Hold".to_string(),
            "Hold for legal investigation".to_string(),
            Some("CASE-2024-001".to_string()),
            vec![user_id],
            None,
            Uuid::new_v4(),
            None,
        ).await.unwrap();

        assert!(service.is_under_legal_hold(user_id));

        service.release_legal_hold(hold.id).await.unwrap();
        assert!(!service.is_under_legal_hold(user_id));
    }

    #[tokio::test]
    async fn test_statistics() {
        let service = RetentionService::new();
        let stats = service.get_statistics();

        assert!(stats.total_policies > 0);
        assert_eq!(stats.active_legal_holds, 0);
    }
}
