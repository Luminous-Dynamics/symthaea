// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Track AD: Data Sovereignty
//!
//! GDPR compliance, data export, right to deletion, audit trails,
//! and full user control over their data.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ============================================================================
// Data Subject Rights (GDPR)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSubjectRequest {
    pub id: Uuid,
    pub user_id: Uuid,
    pub request_type: DataSubjectRequestType,
    pub status: RequestStatus,
    pub submitted_at: DateTime<Utc>,
    pub processed_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub notes: Vec<RequestNote>,
    pub verification_status: VerificationStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DataSubjectRequestType {
    /// Right to access (Article 15)
    Access,
    /// Right to rectification (Article 16)
    Rectification,
    /// Right to erasure (Article 17)
    Erasure,
    /// Right to restriction (Article 18)
    Restriction,
    /// Right to portability (Article 20)
    Portability,
    /// Right to object (Article 21)
    Objection,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RequestStatus {
    Pending,
    IdentityVerification,
    Processing,
    AwaitingApproval,
    Completed,
    Rejected,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum VerificationStatus {
    NotStarted,
    EmailSent,
    Verified,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestNote {
    pub timestamp: DateTime<Utc>,
    pub author: String,
    pub content: String,
    pub is_internal: bool,
}

// ============================================================================
// Data Export
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataExport {
    pub id: Uuid,
    pub user_id: Uuid,
    pub format: ExportFormat,
    pub scope: ExportScope,
    pub status: ExportStatus,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
    pub download_url: Option<String>,
    pub file_size_bytes: Option<u64>,
    pub checksum: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// Machine-readable JSON
    Json,
    /// Email-standard MBOX format
    Mbox,
    /// Individual EML files (zipped)
    Eml,
    /// CSV for structured data
    Csv,
    /// Human-readable PDF report
    Pdf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportScope {
    pub emails: bool,
    pub contacts: bool,
    pub calendar: bool,
    pub settings: bool,
    pub trust_data: bool,
    pub activity_logs: bool,
    pub attachments: bool,
    pub date_range: Option<DateRange>,
    pub folders: Option<Vec<String>>,
}

impl Default for ExportScope {
    fn default() -> Self {
        Self {
            emails: true,
            contacts: true,
            calendar: true,
            settings: true,
            trust_data: true,
            activity_logs: true,
            attachments: true,
            date_range: None,
            folders: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ExportStatus {
    Queued,
    Processing,
    Packaging,
    Ready,
    Downloaded,
    Expired,
    Failed,
}

// ============================================================================
// Data Deletion
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeletionRequest {
    pub id: Uuid,
    pub user_id: Uuid,
    pub scope: DeletionScope,
    pub status: DeletionStatus,
    pub requested_at: DateTime<Utc>,
    pub scheduled_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub retention_days: u32,
    pub confirmation_code: Option<String>,
    pub confirmed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeletionScope {
    pub account: bool,
    pub emails: bool,
    pub contacts: bool,
    pub calendar: bool,
    pub attachments: bool,
    pub trust_data: bool,
    pub activity_logs: bool,
    pub backups: bool,
}

impl DeletionScope {
    pub fn full_account() -> Self {
        Self {
            account: true,
            emails: true,
            contacts: true,
            calendar: true,
            attachments: true,
            trust_data: true,
            activity_logs: true,
            backups: true,
        }
    }

    pub fn emails_only() -> Self {
        Self {
            account: false,
            emails: true,
            contacts: false,
            calendar: false,
            attachments: true,
            trust_data: false,
            activity_logs: false,
            backups: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DeletionStatus {
    Requested,
    PendingConfirmation,
    Scheduled,
    InProgress,
    PartiallyCompleted,
    Completed,
    Failed,
    Cancelled,
}

// ============================================================================
// Audit Trail
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub id: Uuid,
    pub user_id: Option<Uuid>,
    pub actor_id: Option<Uuid>,
    pub actor_type: ActorType,
    pub action: AuditAction,
    pub resource_type: String,
    pub resource_id: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub details: serde_json::Value,
    pub success: bool,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ActorType {
    User,
    System,
    Admin,
    Api,
    Automated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditAction {
    // Authentication
    Login,
    Logout,
    PasswordChange,
    TwoFactorEnabled,
    TwoFactorDisabled,
    PasskeyAdded,
    PasskeyRemoved,
    SessionRevoked,

    // Email
    EmailRead,
    EmailDeleted,
    EmailSent,
    EmailMoved,
    EmailExported,
    AttachmentDownloaded,

    // Data
    DataExported,
    DataDeleted,
    SettingsChanged,
    ProfileUpdated,

    // Trust
    TrustAttestationCreated,
    TrustAttestationRevoked,

    // Admin
    AccountCreated,
    AccountSuspended,
    AccountDeleted,
    PermissionChanged,

    // Compliance
    DataSubjectRequest,
    ConsentUpdated,

    // Other
    Custom(String),
}

// ============================================================================
// Consent Management
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRecord {
    pub id: Uuid,
    pub user_id: Uuid,
    pub consent_type: ConsentType,
    pub granted: bool,
    pub granted_at: Option<DateTime<Utc>>,
    pub revoked_at: Option<DateTime<Utc>>,
    pub version: String,
    pub ip_address: Option<String>,
    pub proof: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConsentType {
    TermsOfService,
    PrivacyPolicy,
    DataProcessing,
    Marketing,
    Analytics,
    ThirdPartySharing,
    CrossBorderTransfer,
    AiProcessing,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentPreferences {
    pub user_id: Uuid,
    pub consents: Vec<ConsentRecord>,
    pub last_updated: DateTime<Utc>,
}

// ============================================================================
// Data Retention
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub data_type: DataType,
    pub retention_days: u32,
    pub action: RetentionAction,
    pub enabled: bool,
    pub conditions: Vec<RetentionCondition>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataType {
    Emails,
    Attachments,
    Contacts,
    CalendarEvents,
    AuditLogs,
    SessionLogs,
    TrustAttestations,
    Backups,
    TempFiles,
    Custom(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RetentionAction {
    Delete,
    Archive,
    Anonymize,
    Notify,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionCondition {
    pub field: String,
    pub operator: ConditionOperator,
    pub value: serde_json::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ConditionOperator {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    Contains,
    IsNull,
    IsNotNull,
}

// ============================================================================
// Privacy Dashboard
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyDashboard {
    pub user_id: Uuid,
    pub data_summary: DataSummary,
    pub consent_status: Vec<ConsentStatus>,
    pub pending_requests: Vec<DataSubjectRequest>,
    pub exports: Vec<DataExport>,
    pub retention_info: RetentionInfo,
    pub third_party_access: Vec<ThirdPartyAccess>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSummary {
    pub total_emails: u64,
    pub total_contacts: u64,
    pub total_events: u64,
    pub total_attachments: u64,
    pub storage_used_bytes: u64,
    pub account_created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentStatus {
    pub consent_type: ConsentType,
    pub granted: bool,
    pub granted_at: Option<DateTime<Utc>>,
    pub required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionInfo {
    pub policies: Vec<RetentionPolicySummary>,
    pub next_cleanup: Option<DateTime<Utc>>,
    pub items_pending_deletion: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicySummary {
    pub data_type: DataType,
    pub retention_days: u32,
    pub items_affected: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThirdPartyAccess {
    pub service_name: String,
    pub permissions: Vec<String>,
    pub authorized_at: DateTime<Utc>,
    pub last_accessed: Option<DateTime<Utc>>,
    pub can_revoke: bool,
}

// ============================================================================
// Services
// ============================================================================

#[async_trait]
pub trait DataExportService: Send + Sync {
    /// Create a new data export request
    async fn create_export(
        &self,
        user_id: Uuid,
        format: ExportFormat,
        scope: ExportScope,
    ) -> Result<DataExport, SovereigntyError>;

    /// Get export status
    async fn get_export(&self, export_id: Uuid) -> Result<Option<DataExport>, SovereigntyError>;

    /// Generate the export file
    async fn generate_export(&self, export_id: Uuid) -> Result<(), SovereigntyError>;

    /// Get download URL
    async fn get_download_url(&self, export_id: Uuid) -> Result<String, SovereigntyError>;
}

#[async_trait]
pub trait DataDeletionService: Send + Sync {
    /// Request data deletion
    async fn request_deletion(
        &self,
        user_id: Uuid,
        scope: DeletionScope,
    ) -> Result<DeletionRequest, SovereigntyError>;

    /// Confirm deletion request
    async fn confirm_deletion(
        &self,
        request_id: Uuid,
        confirmation_code: &str,
    ) -> Result<(), SovereigntyError>;

    /// Execute deletion
    async fn execute_deletion(&self, request_id: Uuid) -> Result<(), SovereigntyError>;

    /// Cancel deletion request
    async fn cancel_deletion(&self, request_id: Uuid) -> Result<(), SovereigntyError>;
}

#[async_trait]
pub trait AuditService: Send + Sync {
    /// Log an audit entry
    async fn log(&self, entry: AuditEntry) -> Result<(), SovereigntyError>;

    /// Query audit log
    async fn query(
        &self,
        filters: AuditFilters,
        pagination: Pagination,
    ) -> Result<Vec<AuditEntry>, SovereigntyError>;

    /// Export audit log
    async fn export(
        &self,
        user_id: Uuid,
        date_range: DateRange,
        format: ExportFormat,
    ) -> Result<String, SovereigntyError>;
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuditFilters {
    pub user_id: Option<Uuid>,
    pub actor_id: Option<Uuid>,
    pub action: Option<AuditAction>,
    pub resource_type: Option<String>,
    pub start_date: Option<DateTime<Utc>>,
    pub end_date: Option<DateTime<Utc>>,
    pub success: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pagination {
    pub offset: u32,
    pub limit: u32,
}

#[async_trait]
pub trait ConsentService: Send + Sync {
    /// Get user's consent preferences
    async fn get_preferences(&self, user_id: Uuid) -> Result<ConsentPreferences, SovereigntyError>;

    /// Grant consent
    async fn grant_consent(
        &self,
        user_id: Uuid,
        consent_type: ConsentType,
        version: &str,
        proof: Option<String>,
    ) -> Result<(), SovereigntyError>;

    /// Revoke consent
    async fn revoke_consent(
        &self,
        user_id: Uuid,
        consent_type: ConsentType,
    ) -> Result<(), SovereigntyError>;

    /// Check if consent is granted
    async fn has_consent(
        &self,
        user_id: Uuid,
        consent_type: ConsentType,
    ) -> Result<bool, SovereigntyError>;
}

#[async_trait]
pub trait RetentionService: Send + Sync {
    /// Get retention policies
    async fn get_policies(&self) -> Result<Vec<RetentionPolicy>, SovereigntyError>;

    /// Apply retention policies
    async fn apply_policies(&self) -> Result<RetentionResult, SovereigntyError>;

    /// Get items pending deletion
    async fn get_pending_items(
        &self,
        data_type: DataType,
    ) -> Result<Vec<PendingDeletion>, SovereigntyError>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionResult {
    pub items_deleted: u64,
    pub items_archived: u64,
    pub items_anonymized: u64,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingDeletion {
    pub resource_type: String,
    pub resource_id: String,
    pub scheduled_for: DateTime<Utc>,
    pub reason: String,
}

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug, thiserror::Error)]
pub enum SovereigntyError {
    #[error("User not found: {0}")]
    UserNotFound(Uuid),
    #[error("Export not found: {0}")]
    ExportNotFound(Uuid),
    #[error("Request not found: {0}")]
    RequestNotFound(Uuid),
    #[error("Invalid confirmation code")]
    InvalidConfirmationCode,
    #[error("Request already processed")]
    AlreadyProcessed,
    #[error("Operation not permitted: {0}")]
    NotPermitted(String),
    #[error("Export generation failed: {0}")]
    ExportFailed(String),
    #[error("Deletion failed: {0}")]
    DeletionFailed(String),
    #[error("Database error: {0}")]
    DatabaseError(String),
    #[error("Internal error: {0}")]
    InternalError(String),
}

// ============================================================================
// Implementation Stubs
// ============================================================================

pub struct DataExportServiceImpl {
    // db: sqlx::PgPool,
    // storage: Arc<dyn StorageBackend>,
}

impl DataExportServiceImpl {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl DataExportService for DataExportServiceImpl {
    async fn create_export(
        &self,
        user_id: Uuid,
        format: ExportFormat,
        scope: ExportScope,
    ) -> Result<DataExport, SovereigntyError> {
        let export = DataExport {
            id: Uuid::new_v4(),
            user_id,
            format,
            scope,
            status: ExportStatus::Queued,
            created_at: Utc::now(),
            completed_at: None,
            expires_at: None,
            download_url: None,
            file_size_bytes: None,
            checksum: None,
        };

        Ok(export)
    }

    async fn get_export(&self, _export_id: Uuid) -> Result<Option<DataExport>, SovereigntyError> {
        Ok(None)
    }

    async fn generate_export(&self, _export_id: Uuid) -> Result<(), SovereigntyError> {
        Ok(())
    }

    async fn get_download_url(&self, _export_id: Uuid) -> Result<String, SovereigntyError> {
        Err(SovereigntyError::ExportNotFound(Uuid::nil()))
    }
}

impl Default for DataExportServiceImpl {
    fn default() -> Self {
        Self::new()
    }
}

pub struct AuditServiceImpl {
    // db: sqlx::PgPool,
}

impl AuditServiceImpl {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl AuditService for AuditServiceImpl {
    async fn log(&self, _entry: AuditEntry) -> Result<(), SovereigntyError> {
        Ok(())
    }

    async fn query(
        &self,
        _filters: AuditFilters,
        _pagination: Pagination,
    ) -> Result<Vec<AuditEntry>, SovereigntyError> {
        Ok(Vec::new())
    }

    async fn export(
        &self,
        _user_id: Uuid,
        _date_range: DateRange,
        _format: ExportFormat,
    ) -> Result<String, SovereigntyError> {
        Ok(String::new())
    }
}

impl Default for AuditServiceImpl {
    fn default() -> Self {
        Self::new()
    }
}
