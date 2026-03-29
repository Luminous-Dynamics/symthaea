// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! GDPR Compliance Tooling for Mycelix Mail
//!
//! Implements data subject rights: access, erasure, portability,
//! consent management, and data processing records.

use std::collections::HashMap;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Types of data subject requests
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DataSubjectRequestType {
    /// Right to access (Article 15)
    Access,
    /// Right to rectification (Article 16)
    Rectification,
    /// Right to erasure / "right to be forgotten" (Article 17)
    Erasure,
    /// Right to restriction of processing (Article 18)
    Restriction,
    /// Right to data portability (Article 20)
    Portability,
    /// Right to object (Article 21)
    Objection,
}

/// Status of a data subject request
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RequestStatus {
    Pending,
    InProgress,
    AwaitingVerification,
    Completed,
    Rejected,
    Expired,
}

/// A data subject request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSubjectRequest {
    pub id: Uuid,
    pub user_id: Uuid,
    pub request_type: DataSubjectRequestType,
    pub status: RequestStatus,
    pub email: String,
    pub description: Option<String>,
    pub verification_token: Option<String>,
    pub verified_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub due_date: DateTime<Utc>, // GDPR requires response within 30 days
    pub completed_at: Option<DateTime<Utc>>,
    pub processed_by: Option<Uuid>,
    pub notes: Vec<RequestNote>,
    pub export_file_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestNote {
    pub id: Uuid,
    pub author_id: Uuid,
    pub content: String,
    pub created_at: DateTime<Utc>,
}

/// User consent record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRecord {
    pub id: Uuid,
    pub user_id: Uuid,
    pub consent_type: ConsentType,
    pub granted: bool,
    pub granted_at: Option<DateTime<Utc>>,
    pub revoked_at: Option<DateTime<Utc>>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub version: String, // Policy version consented to
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ConsentType {
    TermsOfService,
    PrivacyPolicy,
    MarketingEmails,
    AnalyticsCookies,
    ThirdPartySharing,
    AIFeatures,
    DataProcessing,
}

/// Data processing activity record (Article 30)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingActivityRecord {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub purpose: String,
    pub legal_basis: LegalBasis,
    pub data_categories: Vec<String>,
    pub data_subjects: Vec<String>,
    pub recipients: Vec<String>,
    pub retention_period: String,
    pub security_measures: Vec<String>,
    pub cross_border_transfers: bool,
    pub transfer_safeguards: Option<String>,
    pub dpia_required: bool,
    pub dpia_completed: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LegalBasis {
    Consent,
    Contract,
    LegalObligation,
    VitalInterests,
    PublicInterest,
    LegitimateInterests(String), // Must document the interest
}

/// Data export format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataExport {
    pub export_id: Uuid,
    pub user_id: Uuid,
    pub generated_at: DateTime<Utc>,
    pub format: ExportFormat,
    pub sections: Vec<ExportSection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    Json,
    Csv,
    Xml,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportSection {
    pub name: String,
    pub description: String,
    pub data: serde_json::Value,
    pub record_count: usize,
}

/// Erasure task tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErasureTask {
    pub id: Uuid,
    pub request_id: Uuid,
    pub user_id: Uuid,
    pub data_type: String,
    pub status: ErasureStatus,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ErasureStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Skipped, // For data that must be retained (legal obligation)
}

/// GDPR compliance service
pub struct GdprService {
    requests: HashMap<Uuid, DataSubjectRequest>,
    consents: HashMap<Uuid, Vec<ConsentRecord>>,
    processing_activities: Vec<ProcessingActivityRecord>,
    erasure_tasks: HashMap<Uuid, Vec<ErasureTask>>,
}

impl GdprService {
    pub fn new() -> Self {
        Self {
            requests: HashMap::new(),
            consents: HashMap::new(),
            processing_activities: Vec::new(),
            erasure_tasks: HashMap::new(),
        }
    }

    // ========================================================================
    // Data Subject Requests
    // ========================================================================

    /// Create a new data subject request
    pub fn create_request(
        &mut self,
        user_id: Uuid,
        email: String,
        request_type: DataSubjectRequestType,
        description: Option<String>,
    ) -> DataSubjectRequest {
        let now = Utc::now();
        let verification_token = Uuid::new_v4().to_string();

        let request = DataSubjectRequest {
            id: Uuid::new_v4(),
            user_id,
            request_type,
            status: RequestStatus::AwaitingVerification,
            email,
            description,
            verification_token: Some(verification_token),
            verified_at: None,
            created_at: now,
            due_date: now + Duration::days(30), // GDPR 30-day deadline
            completed_at: None,
            processed_by: None,
            notes: Vec::new(),
            export_file_path: None,
        };

        self.requests.insert(request.id, request.clone());
        request
    }

    /// Verify a data subject request
    pub fn verify_request(&mut self, request_id: Uuid, token: &str) -> Result<(), GdprError> {
        let request = self.requests.get_mut(&request_id)
            .ok_or(GdprError::RequestNotFound(request_id))?;

        if request.verification_token.as_deref() != Some(token) {
            return Err(GdprError::InvalidVerificationToken);
        }

        request.verified_at = Some(Utc::now());
        request.status = RequestStatus::Pending;
        request.verification_token = None;

        Ok(())
    }

    /// Process a data subject request
    pub async fn process_request(
        &mut self,
        request_id: Uuid,
        processor_id: Uuid,
    ) -> Result<(), GdprError> {
        let request = self.requests.get_mut(&request_id)
            .ok_or(GdprError::RequestNotFound(request_id))?;

        if request.verified_at.is_none() {
            return Err(GdprError::NotVerified);
        }

        request.status = RequestStatus::InProgress;
        request.processed_by = Some(processor_id);

        match request.request_type {
            DataSubjectRequestType::Access | DataSubjectRequestType::Portability => {
                // Data export will be handled separately
            }
            DataSubjectRequestType::Erasure => {
                // Queue erasure tasks
                self.queue_erasure_tasks(request_id, request.user_id);
            }
            DataSubjectRequestType::Restriction => {
                // Mark user data as restricted
            }
            DataSubjectRequestType::Rectification => {
                // Handled manually with notes
            }
            DataSubjectRequestType::Objection => {
                // Stop specific processing
            }
        }

        Ok(())
    }

    /// Complete a request
    pub fn complete_request(
        &mut self,
        request_id: Uuid,
        export_path: Option<String>,
    ) -> Result<(), GdprError> {
        let request = self.requests.get_mut(&request_id)
            .ok_or(GdprError::RequestNotFound(request_id))?;

        request.status = RequestStatus::Completed;
        request.completed_at = Some(Utc::now());
        request.export_file_path = export_path;

        Ok(())
    }

    /// Get pending requests due soon
    pub fn get_requests_due_soon(&self, days: i64) -> Vec<&DataSubjectRequest> {
        let threshold = Utc::now() + Duration::days(days);

        self.requests.values()
            .filter(|r| {
                r.status != RequestStatus::Completed &&
                r.status != RequestStatus::Rejected &&
                r.due_date <= threshold
            })
            .collect()
    }

    // ========================================================================
    // Data Export (Article 15 & 20)
    // ========================================================================

    /// Generate a complete data export for a user
    pub fn generate_data_export(
        &self,
        user_id: Uuid,
        format: ExportFormat,
        // In real implementation, these would come from database queries
        user_data: serde_json::Value,
        emails: Vec<serde_json::Value>,
        contacts: Vec<serde_json::Value>,
        settings: serde_json::Value,
        trust_data: serde_json::Value,
        activity_log: Vec<serde_json::Value>,
    ) -> DataExport {
        let sections = vec![
            ExportSection {
                name: "Profile".to_string(),
                description: "Your account information and profile data".to_string(),
                data: user_data,
                record_count: 1,
            },
            ExportSection {
                name: "Emails".to_string(),
                description: "All emails in your mailbox".to_string(),
                record_count: emails.len(),
                data: serde_json::json!(emails),
            },
            ExportSection {
                name: "Contacts".to_string(),
                description: "Your contact list".to_string(),
                record_count: contacts.len(),
                data: serde_json::json!(contacts),
            },
            ExportSection {
                name: "Settings".to_string(),
                description: "Your account settings and preferences".to_string(),
                data: settings,
                record_count: 1,
            },
            ExportSection {
                name: "Trust Network".to_string(),
                description: "Your trust relationships and scores".to_string(),
                data: trust_data,
                record_count: 1,
            },
            ExportSection {
                name: "Activity Log".to_string(),
                description: "History of your account activity".to_string(),
                record_count: activity_log.len(),
                data: serde_json::json!(activity_log),
            },
        ];

        DataExport {
            export_id: Uuid::new_v4(),
            user_id,
            generated_at: Utc::now(),
            format,
            sections,
        }
    }

    // ========================================================================
    // Data Erasure (Article 17)
    // ========================================================================

    /// Queue erasure tasks for a user
    fn queue_erasure_tasks(&mut self, request_id: Uuid, user_id: Uuid) {
        let data_types = vec![
            "emails",
            "contacts",
            "attachments",
            "drafts",
            "settings",
            "trust_relationships",
            "activity_logs",
            "search_history",
            "api_keys",
            "sessions",
        ];

        let tasks: Vec<ErasureTask> = data_types.iter().map(|dt| {
            ErasureTask {
                id: Uuid::new_v4(),
                request_id,
                user_id,
                data_type: dt.to_string(),
                status: ErasureStatus::Pending,
                started_at: None,
                completed_at: None,
                error: None,
            }
        }).collect();

        self.erasure_tasks.insert(request_id, tasks);
    }

    /// Execute an erasure task
    pub fn execute_erasure_task(
        &mut self,
        request_id: Uuid,
        task_id: Uuid,
    ) -> Result<(), GdprError> {
        let tasks = self.erasure_tasks.get_mut(&request_id)
            .ok_or(GdprError::RequestNotFound(request_id))?;

        let task = tasks.iter_mut()
            .find(|t| t.id == task_id)
            .ok_or(GdprError::TaskNotFound(task_id))?;

        task.status = ErasureStatus::InProgress;
        task.started_at = Some(Utc::now());

        // In real implementation, execute actual deletion
        // For now, mark as completed
        task.status = ErasureStatus::Completed;
        task.completed_at = Some(Utc::now());

        Ok(())
    }

    /// Get erasure progress
    pub fn get_erasure_progress(&self, request_id: Uuid) -> Result<ErasureProgress, GdprError> {
        let tasks = self.erasure_tasks.get(&request_id)
            .ok_or(GdprError::RequestNotFound(request_id))?;

        let total = tasks.len();
        let completed = tasks.iter().filter(|t| t.status == ErasureStatus::Completed).count();
        let failed = tasks.iter().filter(|t| t.status == ErasureStatus::Failed).count();

        Ok(ErasureProgress {
            total,
            completed,
            failed,
            pending: total - completed - failed,
            tasks: tasks.clone(),
        })
    }

    // ========================================================================
    // Consent Management
    // ========================================================================

    /// Record user consent
    pub fn record_consent(
        &mut self,
        user_id: Uuid,
        consent_type: ConsentType,
        granted: bool,
        ip_address: Option<String>,
        user_agent: Option<String>,
        policy_version: String,
    ) -> ConsentRecord {
        let now = Utc::now();

        let record = ConsentRecord {
            id: Uuid::new_v4(),
            user_id,
            consent_type: consent_type.clone(),
            granted,
            granted_at: if granted { Some(now) } else { None },
            revoked_at: if !granted { Some(now) } else { None },
            ip_address,
            user_agent,
            version: policy_version,
        };

        self.consents.entry(user_id).or_default().push(record.clone());
        record
    }

    /// Get current consent status for a user
    pub fn get_consent_status(&self, user_id: Uuid) -> HashMap<ConsentType, bool> {
        let mut status = HashMap::new();

        if let Some(records) = self.consents.get(&user_id) {
            for record in records.iter().rev() {
                if !status.contains_key(&record.consent_type) {
                    status.insert(record.consent_type.clone(), record.granted);
                }
            }
        }

        status
    }

    /// Check if user has given specific consent
    pub fn has_consent(&self, user_id: Uuid, consent_type: &ConsentType) -> bool {
        self.get_consent_status(user_id)
            .get(consent_type)
            .copied()
            .unwrap_or(false)
    }

    /// Revoke consent
    pub fn revoke_consent(
        &mut self,
        user_id: Uuid,
        consent_type: ConsentType,
    ) -> Result<(), GdprError> {
        self.record_consent(
            user_id,
            consent_type,
            false,
            None,
            None,
            String::new(),
        );
        Ok(())
    }

    /// Get consent history for a user
    pub fn get_consent_history(&self, user_id: Uuid) -> Vec<&ConsentRecord> {
        self.consents.get(&user_id)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    // ========================================================================
    // Processing Activities (Article 30)
    // ========================================================================

    /// Register a processing activity
    pub fn register_processing_activity(&mut self, activity: ProcessingActivityRecord) {
        self.processing_activities.push(activity);
    }

    /// Get all processing activities (for ROPA)
    pub fn get_processing_activities(&self) -> &[ProcessingActivityRecord] {
        &self.processing_activities
    }

    /// Generate Records of Processing Activities report
    pub fn generate_ropa_report(&self) -> RopaReport {
        RopaReport {
            generated_at: Utc::now(),
            controller_name: "Mycelix Mail".to_string(),
            controller_contact: "privacy@mycelix.mail".to_string(),
            dpo_contact: Some("dpo@mycelix.mail".to_string()),
            activities: self.processing_activities.clone(),
        }
    }
}

impl Default for GdprService {
    fn default() -> Self {
        Self::new()
    }
}

/// Erasure progress tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErasureProgress {
    pub total: usize,
    pub completed: usize,
    pub failed: usize,
    pub pending: usize,
    pub tasks: Vec<ErasureTask>,
}

/// Records of Processing Activities report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopaReport {
    pub generated_at: DateTime<Utc>,
    pub controller_name: String,
    pub controller_contact: String,
    pub dpo_contact: Option<String>,
    pub activities: Vec<ProcessingActivityRecord>,
}

/// GDPR-related errors
#[derive(Debug, Clone)]
pub enum GdprError {
    RequestNotFound(Uuid),
    TaskNotFound(Uuid),
    InvalidVerificationToken,
    NotVerified,
    AlreadyProcessed,
    ExportFailed(String),
    ErasureFailed(String),
}

impl std::fmt::Display for GdprError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RequestNotFound(id) => write!(f, "Request not found: {}", id),
            Self::TaskNotFound(id) => write!(f, "Task not found: {}", id),
            Self::InvalidVerificationToken => write!(f, "Invalid verification token"),
            Self::NotVerified => write!(f, "Request not verified"),
            Self::AlreadyProcessed => write!(f, "Request already processed"),
            Self::ExportFailed(e) => write!(f, "Export failed: {}", e),
            Self::ErasureFailed(e) => write!(f, "Erasure failed: {}", e),
        }
    }
}

impl std::error::Error for GdprError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_verify_request() {
        let mut service = GdprService::new();
        let user_id = Uuid::new_v4();

        let request = service.create_request(
            user_id,
            "user@example.com".to_string(),
            DataSubjectRequestType::Access,
            None,
        );

        assert_eq!(request.status, RequestStatus::AwaitingVerification);

        let token = request.verification_token.clone().unwrap();
        service.verify_request(request.id, &token).unwrap();

        let updated = service.requests.get(&request.id).unwrap();
        assert_eq!(updated.status, RequestStatus::Pending);
    }

    #[test]
    fn test_consent_management() {
        let mut service = GdprService::new();
        let user_id = Uuid::new_v4();

        service.record_consent(
            user_id,
            ConsentType::MarketingEmails,
            true,
            Some("192.168.1.1".to_string()),
            None,
            "1.0".to_string(),
        );

        assert!(service.has_consent(user_id, &ConsentType::MarketingEmails));
        assert!(!service.has_consent(user_id, &ConsentType::AnalyticsCookies));

        service.revoke_consent(user_id, ConsentType::MarketingEmails).unwrap();
        assert!(!service.has_consent(user_id, &ConsentType::MarketingEmails));
    }
}
