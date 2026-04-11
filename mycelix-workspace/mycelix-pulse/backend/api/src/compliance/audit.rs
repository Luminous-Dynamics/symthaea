// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Immutable Audit Logging for Mycelix Mail
//!
//! Comprehensive audit trail for security, compliance, and forensics.
//! Supports tamper-evident logging and compliance reporting.

use std::collections::HashMap;
use std::sync::Arc;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Audit event categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AuditCategory {
    Authentication,
    Authorization,
    DataAccess,
    DataModification,
    DataDeletion,
    Configuration,
    Security,
    Compliance,
    System,
    Integration,
}

/// Audit event severity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AuditSeverity {
    Debug = 0,
    Info = 1,
    Warning = 2,
    Error = 3,
    Critical = 4,
}

/// Audit event outcome
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AuditOutcome {
    Success,
    Failure,
    Denied,
    Error,
    Unknown,
}

/// A single audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub sequence: u64,
    pub category: AuditCategory,
    pub action: String,
    pub severity: AuditSeverity,
    pub outcome: AuditOutcome,

    // Actor information
    pub actor_id: Option<Uuid>,
    pub actor_type: ActorType,
    pub actor_email: Option<String>,
    pub actor_ip: Option<String>,
    pub actor_user_agent: Option<String>,
    pub actor_session_id: Option<String>,

    // Resource information
    pub resource_type: Option<String>,
    pub resource_id: Option<String>,
    pub resource_name: Option<String>,

    // Context
    pub request_id: Option<String>,
    pub correlation_id: Option<String>,
    pub service_name: String,
    pub environment: String,

    // Details
    pub message: String,
    pub details: HashMap<String, serde_json::Value>,
    pub old_values: Option<serde_json::Value>,
    pub new_values: Option<serde_json::Value>,

    // Integrity
    pub checksum: String,
    pub previous_checksum: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ActorType {
    User,
    System,
    ApiKey,
    Service,
    Anonymous,
}

/// Audit entry builder
pub struct AuditEntryBuilder {
    category: AuditCategory,
    action: String,
    severity: AuditSeverity,
    outcome: AuditOutcome,
    actor_id: Option<Uuid>,
    actor_type: ActorType,
    actor_email: Option<String>,
    actor_ip: Option<String>,
    actor_user_agent: Option<String>,
    actor_session_id: Option<String>,
    resource_type: Option<String>,
    resource_id: Option<String>,
    resource_name: Option<String>,
    request_id: Option<String>,
    correlation_id: Option<String>,
    message: String,
    details: HashMap<String, serde_json::Value>,
    old_values: Option<serde_json::Value>,
    new_values: Option<serde_json::Value>,
}

impl AuditEntryBuilder {
    pub fn new(category: AuditCategory, action: impl Into<String>) -> Self {
        Self {
            category,
            action: action.into(),
            severity: AuditSeverity::Info,
            outcome: AuditOutcome::Success,
            actor_id: None,
            actor_type: ActorType::Anonymous,
            actor_email: None,
            actor_ip: None,
            actor_user_agent: None,
            actor_session_id: None,
            resource_type: None,
            resource_id: None,
            resource_name: None,
            request_id: None,
            correlation_id: None,
            message: String::new(),
            details: HashMap::new(),
            old_values: None,
            new_values: None,
        }
    }

    pub fn severity(mut self, severity: AuditSeverity) -> Self {
        self.severity = severity;
        self
    }

    pub fn outcome(mut self, outcome: AuditOutcome) -> Self {
        self.outcome = outcome;
        self
    }

    pub fn actor_user(mut self, user_id: Uuid, email: Option<String>) -> Self {
        self.actor_id = Some(user_id);
        self.actor_type = ActorType::User;
        self.actor_email = email;
        self
    }

    pub fn actor_system(mut self, service: &str) -> Self {
        self.actor_type = ActorType::System;
        self.actor_email = Some(service.to_string());
        self
    }

    pub fn actor_api_key(mut self, key_id: Uuid) -> Self {
        self.actor_id = Some(key_id);
        self.actor_type = ActorType::ApiKey;
        self
    }

    pub fn actor_ip(mut self, ip: impl Into<String>) -> Self {
        self.actor_ip = Some(ip.into());
        self
    }

    pub fn actor_user_agent(mut self, ua: impl Into<String>) -> Self {
        self.actor_user_agent = Some(ua.into());
        self
    }

    pub fn session(mut self, session_id: impl Into<String>) -> Self {
        self.actor_session_id = Some(session_id.into());
        self
    }

    pub fn resource(mut self, resource_type: &str, resource_id: &str) -> Self {
        self.resource_type = Some(resource_type.to_string());
        self.resource_id = Some(resource_id.to_string());
        self
    }

    pub fn resource_name(mut self, name: impl Into<String>) -> Self {
        self.resource_name = Some(name.into());
        self
    }

    pub fn request_id(mut self, id: impl Into<String>) -> Self {
        self.request_id = Some(id.into());
        self
    }

    pub fn correlation_id(mut self, id: impl Into<String>) -> Self {
        self.correlation_id = Some(id.into());
        self
    }

    pub fn message(mut self, msg: impl Into<String>) -> Self {
        self.message = msg.into();
        self
    }

    pub fn detail(mut self, key: &str, value: impl Serialize) -> Self {
        if let Ok(v) = serde_json::to_value(value) {
            self.details.insert(key.to_string(), v);
        }
        self
    }

    pub fn old_values(mut self, values: impl Serialize) -> Self {
        self.old_values = serde_json::to_value(values).ok();
        self
    }

    pub fn new_values(mut self, values: impl Serialize) -> Self {
        self.new_values = serde_json::to_value(values).ok();
        self
    }
}

/// Audit log configuration
#[derive(Debug, Clone)]
pub struct AuditConfig {
    pub service_name: String,
    pub environment: String,
    pub min_severity: AuditSeverity,
    pub categories: Option<Vec<AuditCategory>>, // None = all
    pub retention_days: u32,
    pub enable_checksums: bool,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            service_name: "mycelix-mail".to_string(),
            environment: "production".to_string(),
            min_severity: AuditSeverity::Info,
            categories: None,
            retention_days: 365,
            enable_checksums: true,
        }
    }
}

/// Audit log service
pub struct AuditService {
    config: AuditConfig,
    sequence: Arc<RwLock<u64>>,
    last_checksum: Arc<RwLock<Option<String>>>,
    entries: Arc<RwLock<Vec<AuditEntry>>>,
}

impl AuditService {
    pub fn new(config: AuditConfig) -> Self {
        Self {
            config,
            sequence: Arc::new(RwLock::new(0)),
            last_checksum: Arc::new(RwLock::new(None)),
            entries: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Log an audit entry
    pub async fn log(&self, builder: AuditEntryBuilder) -> Uuid {
        // Check if this event should be logged
        if builder.severity < self.config.min_severity {
            return Uuid::nil();
        }

        if let Some(ref categories) = self.config.categories {
            if !categories.contains(&builder.category) {
                return Uuid::nil();
            }
        }

        // Get next sequence number
        let mut seq = self.sequence.write().await;
        *seq += 1;
        let sequence = *seq;

        // Get previous checksum
        let prev_checksum = self.last_checksum.read().await.clone();

        // Build entry
        let id = Uuid::new_v4();
        let timestamp = Utc::now();

        let mut entry = AuditEntry {
            id,
            timestamp,
            sequence,
            category: builder.category,
            action: builder.action,
            severity: builder.severity,
            outcome: builder.outcome,
            actor_id: builder.actor_id,
            actor_type: builder.actor_type,
            actor_email: builder.actor_email,
            actor_ip: builder.actor_ip,
            actor_user_agent: builder.actor_user_agent,
            actor_session_id: builder.actor_session_id,
            resource_type: builder.resource_type,
            resource_id: builder.resource_id,
            resource_name: builder.resource_name,
            request_id: builder.request_id,
            correlation_id: builder.correlation_id,
            service_name: self.config.service_name.clone(),
            environment: self.config.environment.clone(),
            message: builder.message,
            details: builder.details,
            old_values: builder.old_values,
            new_values: builder.new_values,
            checksum: String::new(),
            previous_checksum: prev_checksum,
        };

        // Calculate checksum
        if self.config.enable_checksums {
            entry.checksum = self.calculate_checksum(&entry);
            *self.last_checksum.write().await = Some(entry.checksum.clone());
        }

        // Store entry
        self.entries.write().await.push(entry);

        id
    }

    /// Calculate tamper-evident checksum
    fn calculate_checksum(&self, entry: &AuditEntry) -> String {
        let mut hasher = Sha256::new();

        // Include key fields in checksum
        hasher.update(entry.id.as_bytes());
        hasher.update(entry.timestamp.to_rfc3339().as_bytes());
        hasher.update(entry.sequence.to_le_bytes());
        hasher.update(entry.action.as_bytes());
        hasher.update(entry.message.as_bytes());

        if let Some(ref prev) = entry.previous_checksum {
            hasher.update(prev.as_bytes());
        }

        format!("{:x}", hasher.finalize())
    }

    /// Verify audit log integrity
    pub async fn verify_integrity(&self) -> IntegrityReport {
        let entries = self.entries.read().await;
        let mut issues = Vec::new();
        let mut prev_checksum: Option<String> = None;
        let mut prev_sequence: u64 = 0;

        for (idx, entry) in entries.iter().enumerate() {
            // Check sequence continuity
            if idx > 0 && entry.sequence != prev_sequence + 1 {
                issues.push(IntegrityIssue {
                    entry_id: entry.id,
                    issue_type: IntegrityIssueType::SequenceGap,
                    description: format!(
                        "Expected sequence {}, got {}",
                        prev_sequence + 1,
                        entry.sequence
                    ),
                });
            }

            // Check checksum chain
            if entry.previous_checksum != prev_checksum {
                issues.push(IntegrityIssue {
                    entry_id: entry.id,
                    issue_type: IntegrityIssueType::ChecksumMismatch,
                    description: "Previous checksum does not match chain".to_string(),
                });
            }

            // Verify entry checksum
            let computed = self.calculate_checksum(entry);
            if computed != entry.checksum {
                issues.push(IntegrityIssue {
                    entry_id: entry.id,
                    issue_type: IntegrityIssueType::TamperedEntry,
                    description: "Entry checksum does not match computed value".to_string(),
                });
            }

            prev_checksum = Some(entry.checksum.clone());
            prev_sequence = entry.sequence;
        }

        IntegrityReport {
            verified_at: Utc::now(),
            total_entries: entries.len(),
            issues,
            is_valid: issues.is_empty(),
        }
    }

    /// Query audit logs
    pub async fn query(&self, filter: AuditFilter) -> Vec<AuditEntry> {
        let entries = self.entries.read().await;

        entries.iter()
            .filter(|e| {
                // Category filter
                if let Some(ref cats) = filter.categories {
                    if !cats.contains(&e.category) {
                        return false;
                    }
                }

                // Severity filter
                if let Some(min_sev) = filter.min_severity {
                    if e.severity < min_sev {
                        return false;
                    }
                }

                // Actor filter
                if let Some(ref actor_id) = filter.actor_id {
                    if e.actor_id.as_ref() != Some(actor_id) {
                        return false;
                    }
                }

                // Resource filter
                if let Some(ref res_type) = filter.resource_type {
                    if e.resource_type.as_ref() != Some(res_type) {
                        return false;
                    }
                }

                if let Some(ref res_id) = filter.resource_id {
                    if e.resource_id.as_ref() != Some(res_id) {
                        return false;
                    }
                }

                // Date range filter
                if let Some(from) = filter.from_date {
                    if e.timestamp < from {
                        return false;
                    }
                }

                if let Some(to) = filter.to_date {
                    if e.timestamp > to {
                        return false;
                    }
                }

                // Action filter
                if let Some(ref action) = filter.action {
                    if !e.action.contains(action) {
                        return false;
                    }
                }

                // Outcome filter
                if let Some(ref outcome) = filter.outcome {
                    if &e.outcome != outcome {
                        return false;
                    }
                }

                true
            })
            .cloned()
            .collect()
    }

    /// Generate compliance report
    pub async fn generate_compliance_report(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> ComplianceReport {
        let entries = self.query(AuditFilter {
            from_date: Some(from),
            to_date: Some(to),
            ..Default::default()
        }).await;

        let mut category_counts: HashMap<AuditCategory, usize> = HashMap::new();
        let mut outcome_counts: HashMap<AuditOutcome, usize> = HashMap::new();
        let mut severity_counts: HashMap<AuditSeverity, usize> = HashMap::new();
        let mut failed_logins = 0;
        let mut data_exports = 0;
        let mut data_deletions = 0;
        let mut security_events = 0;

        for entry in &entries {
            *category_counts.entry(entry.category.clone()).or_insert(0) += 1;
            *outcome_counts.entry(entry.outcome.clone()).or_insert(0) += 1;
            *severity_counts.entry(entry.severity).or_insert(0) += 1;

            if entry.category == AuditCategory::Authentication &&
               entry.outcome == AuditOutcome::Failure {
                failed_logins += 1;
            }

            if entry.action.contains("export") {
                data_exports += 1;
            }

            if entry.action.contains("delete") || entry.action.contains("erase") {
                data_deletions += 1;
            }

            if entry.category == AuditCategory::Security {
                security_events += 1;
            }
        }

        ComplianceReport {
            generated_at: Utc::now(),
            period_start: from,
            period_end: to,
            total_events: entries.len(),
            category_breakdown: category_counts,
            outcome_breakdown: outcome_counts,
            severity_breakdown: severity_counts,
            failed_login_attempts: failed_logins,
            data_export_requests: data_exports,
            data_deletion_requests: data_deletions,
            security_events,
        }
    }
}

/// Filter for querying audit logs
#[derive(Debug, Clone, Default)]
pub struct AuditFilter {
    pub categories: Option<Vec<AuditCategory>>,
    pub min_severity: Option<AuditSeverity>,
    pub actor_id: Option<Uuid>,
    pub resource_type: Option<String>,
    pub resource_id: Option<String>,
    pub from_date: Option<DateTime<Utc>>,
    pub to_date: Option<DateTime<Utc>>,
    pub action: Option<String>,
    pub outcome: Option<AuditOutcome>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// Integrity verification report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityReport {
    pub verified_at: DateTime<Utc>,
    pub total_entries: usize,
    pub issues: Vec<IntegrityIssue>,
    pub is_valid: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityIssue {
    pub entry_id: Uuid,
    pub issue_type: IntegrityIssueType,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntegrityIssueType {
    SequenceGap,
    ChecksumMismatch,
    TamperedEntry,
}

/// Compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    pub generated_at: DateTime<Utc>,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub total_events: usize,
    pub category_breakdown: HashMap<AuditCategory, usize>,
    pub outcome_breakdown: HashMap<AuditOutcome, usize>,
    pub severity_breakdown: HashMap<AuditSeverity, usize>,
    pub failed_login_attempts: usize,
    pub data_export_requests: usize,
    pub data_deletion_requests: usize,
    pub security_events: usize,
}

// Convenience logging functions
pub async fn log_login(service: &AuditService, user_id: Uuid, email: &str, ip: &str, success: bool) -> Uuid {
    service.log(
        AuditEntryBuilder::new(AuditCategory::Authentication, "user.login")
            .actor_user(user_id, Some(email.to_string()))
            .actor_ip(ip)
            .outcome(if success { AuditOutcome::Success } else { AuditOutcome::Failure })
            .message(if success { "User logged in" } else { "Login failed" })
    ).await
}

pub async fn log_data_access(
    service: &AuditService,
    user_id: Uuid,
    resource_type: &str,
    resource_id: &str,
) -> Uuid {
    service.log(
        AuditEntryBuilder::new(AuditCategory::DataAccess, format!("{}.read", resource_type))
            .actor_user(user_id, None)
            .resource(resource_type, resource_id)
            .message(format!("Accessed {} {}", resource_type, resource_id))
    ).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_audit_logging() {
        let service = AuditService::new(AuditConfig::default());

        let id = service.log(
            AuditEntryBuilder::new(AuditCategory::Authentication, "user.login")
                .actor_user(Uuid::new_v4(), Some("test@example.com".to_string()))
                .outcome(AuditOutcome::Success)
                .message("User logged in successfully")
        ).await;

        assert_ne!(id, Uuid::nil());

        let entries = service.query(AuditFilter::default()).await;
        assert_eq!(entries.len(), 1);
    }

    #[tokio::test]
    async fn test_integrity_verification() {
        let service = AuditService::new(AuditConfig::default());

        for i in 0..5 {
            service.log(
                AuditEntryBuilder::new(AuditCategory::System, format!("test.action.{}", i))
                    .message(format!("Test entry {}", i))
            ).await;
        }

        let report = service.verify_integrity().await;
        assert!(report.is_valid);
        assert_eq!(report.total_entries, 5);
    }
}
