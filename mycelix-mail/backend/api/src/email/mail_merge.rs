// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Track CC: Mail Merge for Bulk Sending
// Personalized bulk email with CSV/data source integration

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use tokio::sync::mpsc;

use super::templates::{EmailTemplate, TemplateService, TemplateError};

/// Data source for mail merge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSource {
    /// CSV file content
    Csv(String),
    /// JSON array of objects
    Json(Vec<HashMap<String, String>>),
    /// Contact list reference
    ContactList(Uuid),
    /// Manual recipient list
    Manual(Vec<MergeRecipient>),
}

/// Individual recipient with merge data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeRecipient {
    pub email: String,
    pub name: Option<String>,
    pub variables: HashMap<String, String>,
}

/// Mail merge job configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MailMergeJob {
    pub id: Uuid,
    pub user_id: Uuid,
    pub name: String,
    pub template_id: Uuid,
    pub recipients: Vec<MergeRecipient>,
    pub global_variables: HashMap<String, String>,
    pub settings: MergeSettings,
    pub status: MergeJobStatus,
    pub progress: MergeProgress,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub errors: Vec<MergeError>,
}

/// Merge job settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeSettings {
    /// Delay between emails (ms) - rate limiting
    pub send_delay_ms: u64,
    /// Maximum concurrent sends
    pub max_concurrent: usize,
    /// Track opens
    pub track_opens: bool,
    /// Track clicks
    pub track_clicks: bool,
    /// Custom reply-to address
    pub reply_to: Option<String>,
    /// Schedule for later
    pub schedule_at: Option<DateTime<Utc>>,
    /// Send in batches with delay between
    pub batch_size: Option<usize>,
    pub batch_delay_minutes: Option<u32>,
    /// Test mode - send all to test address
    pub test_mode: bool,
    pub test_email: Option<String>,
    /// Unsubscribe link
    pub include_unsubscribe: bool,
    pub unsubscribe_url: Option<String>,
}

impl Default for MergeSettings {
    fn default() -> Self {
        Self {
            send_delay_ms: 100,
            max_concurrent: 10,
            track_opens: true,
            track_clicks: true,
            reply_to: None,
            schedule_at: None,
            batch_size: None,
            batch_delay_minutes: None,
            test_mode: false,
            test_email: None,
            include_unsubscribe: true,
            unsubscribe_url: None,
        }
    }
}

/// Job execution status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MergeJobStatus {
    Draft,
    Validating,
    Scheduled,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

/// Progress tracking
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MergeProgress {
    pub total_recipients: usize,
    pub sent: usize,
    pub failed: usize,
    pub pending: usize,
    pub opens: usize,
    pub clicks: usize,
    pub bounces: usize,
    pub unsubscribes: usize,
}

/// Individual send result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SendResult {
    pub recipient_email: String,
    pub success: bool,
    pub message_id: Option<String>,
    pub error: Option<String>,
    pub sent_at: DateTime<Utc>,
}

/// Merge error record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeError {
    pub recipient_email: String,
    pub error_type: MergeErrorType,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MergeErrorType {
    InvalidEmail,
    RenderFailed,
    SendFailed,
    RateLimited,
    Bounced,
    Unsubscribed,
}

/// Mail merge service
pub struct MailMergeService {
    jobs: HashMap<Uuid, MailMergeJob>,
    results: HashMap<Uuid, Vec<SendResult>>,
    template_service: TemplateService,
}

impl MailMergeService {
    pub fn new(template_service: TemplateService) -> Self {
        Self {
            jobs: HashMap::new(),
            results: HashMap::new(),
            template_service,
        }
    }

    /// Create a new mail merge job
    pub async fn create_job(
        &mut self,
        user_id: Uuid,
        name: String,
        template_id: Uuid,
        data_source: DataSource,
        global_variables: HashMap<String, String>,
        settings: MergeSettings,
    ) -> Result<MailMergeJob, MailMergeError> {
        // Validate template exists
        let _template = self.template_service.get_template(template_id)
            .ok_or(MailMergeError::TemplateNotFound)?;

        // Parse recipients from data source
        let recipients = self.parse_data_source(data_source).await?;

        if recipients.is_empty() {
            return Err(MailMergeError::NoRecipients);
        }

        let job = MailMergeJob {
            id: Uuid::new_v4(),
            user_id,
            name,
            template_id,
            recipients: recipients.clone(),
            global_variables,
            settings,
            status: MergeJobStatus::Draft,
            progress: MergeProgress {
                total_recipients: recipients.len(),
                pending: recipients.len(),
                ..Default::default()
            },
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            errors: Vec::new(),
        };

        self.jobs.insert(job.id, job.clone());

        Ok(job)
    }

    /// Parse data source into recipients
    async fn parse_data_source(&self, source: DataSource) -> Result<Vec<MergeRecipient>, MailMergeError> {
        match source {
            DataSource::Csv(content) => self.parse_csv(&content),
            DataSource::Json(data) => self.parse_json(data),
            DataSource::ContactList(_list_id) => {
                // In production: query contacts from database
                Ok(Vec::new())
            }
            DataSource::Manual(recipients) => Ok(recipients),
        }
    }

    /// Parse CSV content
    fn parse_csv(&self, content: &str) -> Result<Vec<MergeRecipient>, MailMergeError> {
        let mut recipients = Vec::new();
        let mut lines = content.lines();

        // Parse header
        let headers: Vec<String> = lines.next()
            .ok_or(MailMergeError::InvalidDataSource("Empty CSV".to_string()))?
            .split(',')
            .map(|s| s.trim().trim_matches('"').to_lowercase())
            .collect();

        // Find email column
        let email_idx = headers.iter()
            .position(|h| h == "email" || h == "e-mail" || h == "email_address")
            .ok_or(MailMergeError::InvalidDataSource("No email column found".to_string()))?;

        // Find name column (optional)
        let name_idx = headers.iter()
            .position(|h| h == "name" || h == "full_name" || h == "fullname");

        // Parse rows
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }

            let values: Vec<String> = self.parse_csv_line(line);

            if values.len() <= email_idx {
                continue;
            }

            let email = values[email_idx].trim().to_string();
            if !self.is_valid_email(&email) {
                continue;
            }

            let name = name_idx.and_then(|idx| values.get(idx).map(|s| s.trim().to_string()));

            // Build variables map
            let mut variables = HashMap::new();
            for (idx, header) in headers.iter().enumerate() {
                if idx != email_idx {
                    if let Some(value) = values.get(idx) {
                        variables.insert(header.clone(), value.trim().to_string());
                    }
                }
            }

            recipients.push(MergeRecipient {
                email,
                name,
                variables,
            });
        }

        Ok(recipients)
    }

    /// Parse a CSV line handling quoted values
    fn parse_csv_line(&self, line: &str) -> Vec<String> {
        let mut values = Vec::new();
        let mut current = String::new();
        let mut in_quotes = false;

        for ch in line.chars() {
            match ch {
                '"' => in_quotes = !in_quotes,
                ',' if !in_quotes => {
                    values.push(current.trim().trim_matches('"').to_string());
                    current = String::new();
                }
                _ => current.push(ch),
            }
        }

        values.push(current.trim().trim_matches('"').to_string());
        values
    }

    /// Parse JSON data
    fn parse_json(&self, data: Vec<HashMap<String, String>>) -> Result<Vec<MergeRecipient>, MailMergeError> {
        let mut recipients = Vec::new();

        for row in data {
            let email = row.get("email")
                .or_else(|| row.get("e-mail"))
                .or_else(|| row.get("email_address"))
                .ok_or_else(|| MailMergeError::InvalidDataSource("No email field".to_string()))?
                .clone();

            if !self.is_valid_email(&email) {
                continue;
            }

            let name = row.get("name")
                .or_else(|| row.get("full_name"))
                .cloned();

            let mut variables = row.clone();
            variables.remove("email");
            variables.remove("e-mail");
            variables.remove("email_address");

            recipients.push(MergeRecipient {
                email,
                name,
                variables,
            });
        }

        Ok(recipients)
    }

    /// Simple email validation
    fn is_valid_email(&self, email: &str) -> bool {
        let parts: Vec<&str> = email.split('@').collect();
        parts.len() == 2 && !parts[0].is_empty() && parts[1].contains('.')
    }

    /// Validate job before sending
    pub async fn validate_job(&mut self, job_id: Uuid) -> Result<ValidationResult, MailMergeError> {
        let job = self.jobs.get_mut(&job_id)
            .ok_or(MailMergeError::JobNotFound)?;

        job.status = MergeJobStatus::Validating;

        let template = self.template_service.get_template(job.template_id)
            .ok_or(MailMergeError::TemplateNotFound)?;

        let mut result = ValidationResult {
            valid: true,
            total_recipients: job.recipients.len(),
            valid_recipients: 0,
            invalid_emails: Vec::new(),
            missing_variables: Vec::new(),
            warnings: Vec::new(),
        };

        for recipient in &job.recipients {
            // Check email validity
            if !self.is_valid_email(&recipient.email) {
                result.invalid_emails.push(recipient.email.clone());
                continue;
            }

            // Check required variables
            let mut all_vars = job.global_variables.clone();
            all_vars.extend(recipient.variables.clone());

            for var in &template.variables {
                if var.required && !all_vars.contains_key(&var.name) && var.default_value.is_none() {
                    result.missing_variables.push(format!(
                        "{}: missing '{}'", recipient.email, var.name
                    ));
                }
            }

            result.valid_recipients += 1;
        }

        if !result.invalid_emails.is_empty() {
            result.warnings.push(format!(
                "{} recipients have invalid email addresses",
                result.invalid_emails.len()
            ));
        }

        if !result.missing_variables.is_empty() {
            result.valid = false;
        }

        job.status = MergeJobStatus::Draft;

        Ok(result)
    }

    /// Preview merge for a single recipient
    pub async fn preview(
        &self,
        job_id: Uuid,
        recipient_email: &str,
    ) -> Result<MergePreview, MailMergeError> {
        let job = self.jobs.get(&job_id)
            .ok_or(MailMergeError::JobNotFound)?;

        let recipient = job.recipients.iter()
            .find(|r| r.email == recipient_email)
            .ok_or(MailMergeError::RecipientNotFound)?;

        // Merge global and recipient variables
        let mut variables = job.global_variables.clone();
        variables.extend(recipient.variables.clone());
        variables.insert("email".to_string(), recipient.email.clone());
        if let Some(name) = &recipient.name {
            variables.insert("name".to_string(), name.clone());
        }

        let rendered = self.template_service.render(job.template_id, variables)
            .map_err(|e| MailMergeError::RenderFailed(e.to_string()))?;

        Ok(MergePreview {
            recipient_email: recipient.email.clone(),
            recipient_name: recipient.name.clone(),
            subject: rendered.subject,
            body_html: rendered.body_html,
            body_text: rendered.body_text,
        })
    }

    /// Start executing the mail merge job
    pub async fn start_job(&mut self, job_id: Uuid) -> Result<(), MailMergeError> {
        let job = self.jobs.get_mut(&job_id)
            .ok_or(MailMergeError::JobNotFound)?;

        if job.status != MergeJobStatus::Draft && job.status != MergeJobStatus::Paused {
            return Err(MailMergeError::InvalidJobState);
        }

        // Check if scheduled for later
        if let Some(schedule_at) = job.settings.schedule_at {
            if schedule_at > Utc::now() {
                job.status = MergeJobStatus::Scheduled;
                return Ok(());
            }
        }

        job.status = MergeJobStatus::Running;
        job.started_at = Some(Utc::now());

        // In production, this would spawn a background task
        // For now, simulate async processing
        self.execute_job(job_id).await
    }

    /// Execute job sending
    async fn execute_job(&mut self, job_id: Uuid) -> Result<(), MailMergeError> {
        let job = self.jobs.get(&job_id)
            .ok_or(MailMergeError::JobNotFound)?
            .clone();

        let batch_size = job.settings.batch_size.unwrap_or(job.recipients.len());
        let mut results = Vec::new();

        for (idx, recipient) in job.recipients.iter().enumerate() {
            // Check if job was cancelled
            if let Some(j) = self.jobs.get(&job_id) {
                if j.status == MergeJobStatus::Cancelled || j.status == MergeJobStatus::Paused {
                    break;
                }
            }

            // Build variables
            let mut variables = job.global_variables.clone();
            variables.extend(recipient.variables.clone());
            variables.insert("email".to_string(), recipient.email.clone());
            if let Some(name) = &recipient.name {
                variables.insert("name".to_string(), name.clone());
            }

            // Add unsubscribe link if enabled
            if job.settings.include_unsubscribe {
                let unsub_url = job.settings.unsubscribe_url.clone()
                    .unwrap_or_else(|| format!("/unsubscribe?email={}", recipient.email));
                variables.insert("unsubscribe_url".to_string(), unsub_url);
            }

            // Render and send
            let result = self.send_single(&job, recipient, &variables).await;
            results.push(result.clone());

            // Update progress
            if let Some(j) = self.jobs.get_mut(&job_id) {
                if result.success {
                    j.progress.sent += 1;
                } else {
                    j.progress.failed += 1;
                    j.errors.push(MergeError {
                        recipient_email: recipient.email.clone(),
                        error_type: MergeErrorType::SendFailed,
                        message: result.error.unwrap_or_default(),
                        timestamp: Utc::now(),
                    });
                }
                j.progress.pending = j.recipients.len() - j.progress.sent - j.progress.failed;
            }

            // Rate limiting delay
            if job.settings.send_delay_ms > 0 {
                tokio::time::sleep(tokio::time::Duration::from_millis(
                    job.settings.send_delay_ms
                )).await;
            }

            // Batch delay
            if let Some(batch_delay) = job.settings.batch_delay_minutes {
                if (idx + 1) % batch_size == 0 && idx + 1 < job.recipients.len() {
                    tokio::time::sleep(tokio::time::Duration::from_secs(
                        batch_delay as u64 * 60
                    )).await;
                }
            }
        }

        // Store results
        self.results.insert(job_id, results);

        // Update job status
        if let Some(j) = self.jobs.get_mut(&job_id) {
            if j.status == MergeJobStatus::Running {
                j.status = if j.progress.failed == 0 {
                    MergeJobStatus::Completed
                } else if j.progress.sent == 0 {
                    MergeJobStatus::Failed
                } else {
                    MergeJobStatus::Completed // Partial success
                };
                j.completed_at = Some(Utc::now());
            }
        }

        Ok(())
    }

    /// Send to a single recipient
    async fn send_single(
        &self,
        job: &MailMergeJob,
        recipient: &MergeRecipient,
        variables: &HashMap<String, String>,
    ) -> SendResult {
        // Render template
        let rendered = match self.template_service.render(job.template_id, variables.clone()) {
            Ok(r) => r,
            Err(e) => {
                return SendResult {
                    recipient_email: recipient.email.clone(),
                    success: false,
                    message_id: None,
                    error: Some(e.to_string()),
                    sent_at: Utc::now(),
                };
            }
        };

        // In test mode, redirect to test email
        let target_email = if job.settings.test_mode {
            job.settings.test_email.as_ref()
                .unwrap_or(&recipient.email)
                .clone()
        } else {
            recipient.email.clone()
        };

        // Simulate sending (in production: call SMTP service)
        let message_id = format!("<{}.{}@mycelix.mail>", Uuid::new_v4(), job.id);

        SendResult {
            recipient_email: target_email,
            success: true,
            message_id: Some(message_id),
            error: None,
            sent_at: Utc::now(),
        }
    }

    /// Pause a running job
    pub async fn pause_job(&mut self, job_id: Uuid) -> Result<(), MailMergeError> {
        let job = self.jobs.get_mut(&job_id)
            .ok_or(MailMergeError::JobNotFound)?;

        if job.status != MergeJobStatus::Running {
            return Err(MailMergeError::InvalidJobState);
        }

        job.status = MergeJobStatus::Paused;
        Ok(())
    }

    /// Cancel a job
    pub async fn cancel_job(&mut self, job_id: Uuid) -> Result<(), MailMergeError> {
        let job = self.jobs.get_mut(&job_id)
            .ok_or(MailMergeError::JobNotFound)?;

        job.status = MergeJobStatus::Cancelled;
        Ok(())
    }

    /// Get job by ID
    pub fn get_job(&self, job_id: Uuid) -> Option<&MailMergeJob> {
        self.jobs.get(&job_id)
    }

    /// List jobs for user
    pub fn list_jobs(&self, user_id: Uuid) -> Vec<&MailMergeJob> {
        self.jobs.values()
            .filter(|j| j.user_id == user_id)
            .collect()
    }

    /// Get send results for job
    pub fn get_results(&self, job_id: Uuid) -> Vec<&SendResult> {
        self.results.get(&job_id)
            .map(|r| r.iter().collect())
            .unwrap_or_default()
    }

    /// Export job report
    pub fn export_report(&self, job_id: Uuid) -> Result<JobReport, MailMergeError> {
        let job = self.jobs.get(&job_id)
            .ok_or(MailMergeError::JobNotFound)?;

        let results = self.results.get(&job_id)
            .cloned()
            .unwrap_or_default();

        Ok(JobReport {
            job_name: job.name.clone(),
            template_id: job.template_id,
            status: job.status.clone(),
            progress: job.progress.clone(),
            created_at: job.created_at,
            started_at: job.started_at,
            completed_at: job.completed_at,
            total_recipients: job.recipients.len(),
            successful_sends: results.iter().filter(|r| r.success).count(),
            failed_sends: results.iter().filter(|r| !r.success).count(),
            errors: job.errors.clone(),
            send_results: results,
        })
    }

    /// Track email open
    pub fn track_open(&mut self, job_id: Uuid, recipient_email: &str) {
        if let Some(job) = self.jobs.get_mut(&job_id) {
            if job.settings.track_opens {
                job.progress.opens += 1;
            }
        }
    }

    /// Track link click
    pub fn track_click(&mut self, job_id: Uuid, recipient_email: &str, link: &str) {
        if let Some(job) = self.jobs.get_mut(&job_id) {
            if job.settings.track_clicks {
                job.progress.clicks += 1;
            }
        }
    }

    /// Record bounce
    pub fn record_bounce(&mut self, job_id: Uuid, recipient_email: &str) {
        if let Some(job) = self.jobs.get_mut(&job_id) {
            job.progress.bounces += 1;
            job.errors.push(MergeError {
                recipient_email: recipient_email.to_string(),
                error_type: MergeErrorType::Bounced,
                message: "Email bounced".to_string(),
                timestamp: Utc::now(),
            });
        }
    }

    /// Record unsubscribe
    pub fn record_unsubscribe(&mut self, job_id: Uuid, recipient_email: &str) {
        if let Some(job) = self.jobs.get_mut(&job_id) {
            job.progress.unsubscribes += 1;
        }
    }
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub total_recipients: usize,
    pub valid_recipients: usize,
    pub invalid_emails: Vec<String>,
    pub missing_variables: Vec<String>,
    pub warnings: Vec<String>,
}

/// Merge preview
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergePreview {
    pub recipient_email: String,
    pub recipient_name: Option<String>,
    pub subject: String,
    pub body_html: Option<String>,
    pub body_text: String,
}

/// Job report for export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobReport {
    pub job_name: String,
    pub template_id: Uuid,
    pub status: MergeJobStatus,
    pub progress: MergeProgress,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub total_recipients: usize,
    pub successful_sends: usize,
    pub failed_sends: usize,
    pub errors: Vec<MergeError>,
    pub send_results: Vec<SendResult>,
}

/// Mail merge errors
#[derive(Debug, Clone)]
pub enum MailMergeError {
    JobNotFound,
    TemplateNotFound,
    RecipientNotFound,
    NoRecipients,
    InvalidDataSource(String),
    InvalidJobState,
    RenderFailed(String),
    SendFailed(String),
}

impl std::fmt::Display for MailMergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::JobNotFound => write!(f, "Mail merge job not found"),
            Self::TemplateNotFound => write!(f, "Email template not found"),
            Self::RecipientNotFound => write!(f, "Recipient not found"),
            Self::NoRecipients => write!(f, "No recipients in data source"),
            Self::InvalidDataSource(msg) => write!(f, "Invalid data source: {}", msg),
            Self::InvalidJobState => write!(f, "Invalid job state for this operation"),
            Self::RenderFailed(msg) => write!(f, "Template render failed: {}", msg),
            Self::SendFailed(msg) => write!(f, "Send failed: {}", msg),
        }
    }
}

impl std::error::Error for MailMergeError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_service() -> MailMergeService {
        MailMergeService::new(TemplateService::new())
    }

    #[tokio::test]
    async fn test_csv_parsing() {
        let service = create_test_service();

        let csv = r#"email,name,company
alice@example.com,Alice Smith,Acme Inc
bob@example.com,Bob Jones,Tech Corp
invalid-email,Bad Data,None
charlie@test.io,Charlie Brown,Peanuts"#;

        let recipients = service.parse_csv(csv).unwrap();

        assert_eq!(recipients.len(), 3);
        assert_eq!(recipients[0].email, "alice@example.com");
        assert_eq!(recipients[0].name, Some("Alice Smith".to_string()));
        assert_eq!(recipients[0].variables.get("company"), Some(&"Acme Inc".to_string()));
    }

    #[tokio::test]
    async fn test_create_job() {
        let mut service = create_test_service();
        let user_id = Uuid::new_v4();

        let template_id = *service.template_service.shared_templates.first().unwrap();

        let recipients = vec![
            MergeRecipient {
                email: "test@example.com".to_string(),
                name: Some("Test User".to_string()),
                variables: HashMap::new(),
            },
        ];

        let job = service.create_job(
            user_id,
            "Test Campaign".to_string(),
            template_id,
            DataSource::Manual(recipients),
            HashMap::new(),
            MergeSettings::default(),
        ).await.unwrap();

        assert_eq!(job.name, "Test Campaign");
        assert_eq!(job.progress.total_recipients, 1);
        assert_eq!(job.status, MergeJobStatus::Draft);
    }
}
