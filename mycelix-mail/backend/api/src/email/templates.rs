// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Email Templates
//!
//! Reusable email templates with variable substitution and mail merge

use chrono::{DateTime, Utc};
use handlebars::Handlebars;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use sqlx::PgPool;
use uuid::Uuid;

/// Email template
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct EmailTemplate {
    pub id: Uuid,
    pub user_id: Uuid,
    pub tenant_id: Option<Uuid>,
    pub name: String,
    pub description: Option<String>,
    pub subject: String,
    pub body_text: String,
    pub body_html: Option<String>,
    pub variables: JsonValue,
    pub category: Option<String>,
    pub is_shared: bool,
    pub use_count: i32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Variable definition for templates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateVariable {
    pub name: String,
    pub description: Option<String>,
    pub default_value: Option<String>,
    pub required: bool,
    pub variable_type: VariableType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum VariableType {
    Text,
    Number,
    Date,
    Boolean,
    Email,
    Url,
}

/// Rendered template result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderedTemplate {
    pub subject: String,
    pub body_text: String,
    pub body_html: Option<String>,
}

/// Mail merge recipient
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeRecipient {
    pub email: String,
    pub name: Option<String>,
    pub variables: JsonValue,
}

/// Mail merge job
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct MailMergeJob {
    pub id: Uuid,
    pub template_id: Uuid,
    pub user_id: Uuid,
    pub tenant_id: Option<Uuid>,
    pub total_recipients: i32,
    pub sent_count: i32,
    pub failed_count: i32,
    pub status: MailMergeStatus,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "mail_merge_status", rename_all = "lowercase")]
pub enum MailMergeStatus {
    Pending,
    Processing,
    Completed,
    Failed,
    Cancelled,
}

/// Template service
pub struct TemplateService {
    pool: PgPool,
    handlebars: Handlebars<'static>,
}

impl TemplateService {
    pub fn new(pool: PgPool) -> Self {
        let mut handlebars = Handlebars::new();
        handlebars.set_strict_mode(true);

        // Register custom helpers
        handlebars.register_helper("uppercase", Box::new(uppercase_helper));
        handlebars.register_helper("lowercase", Box::new(lowercase_helper));
        handlebars.register_helper("format_date", Box::new(format_date_helper));
        handlebars.register_helper("format_currency", Box::new(format_currency_helper));

        Self { pool, handlebars }
    }

    /// Create a new template
    pub async fn create_template(
        &self,
        user_id: Uuid,
        tenant_id: Option<Uuid>,
        name: &str,
        subject: &str,
        body_text: &str,
        body_html: Option<&str>,
        variables: Vec<TemplateVariable>,
    ) -> Result<EmailTemplate, TemplateError> {
        // Validate template syntax
        self.validate_template(subject, body_text, body_html)?;

        let id = Uuid::new_v4();
        let variables_json = serde_json::to_value(&variables)
            .map_err(|e| TemplateError::Validation(e.to_string()))?;

        let template = sqlx::query_as::<_, EmailTemplate>(
            r#"
            INSERT INTO email_templates (
                id, user_id, tenant_id, name, subject, body_text, body_html,
                variables, is_shared, use_count, created_at, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, false, 0, NOW(), NOW())
            RETURNING *
            "#,
        )
        .bind(id)
        .bind(user_id)
        .bind(tenant_id)
        .bind(name)
        .bind(subject)
        .bind(body_text)
        .bind(body_html)
        .bind(&variables_json)
        .fetch_one(&self.pool)
        .await
        .map_err(TemplateError::Database)?;

        Ok(template)
    }

    /// Validate template syntax
    fn validate_template(
        &self,
        subject: &str,
        body_text: &str,
        body_html: Option<&str>,
    ) -> Result<(), TemplateError> {
        // Try to compile templates
        self.handlebars
            .render_template(subject, &serde_json::json!({}))
            .map_err(|e| TemplateError::Validation(format!("Subject: {}", e)))?;

        self.handlebars
            .render_template(body_text, &serde_json::json!({}))
            .map_err(|e| TemplateError::Validation(format!("Body text: {}", e)))?;

        if let Some(html) = body_html {
            self.handlebars
                .render_template(html, &serde_json::json!({}))
                .map_err(|e| TemplateError::Validation(format!("Body HTML: {}", e)))?;
        }

        Ok(())
    }

    /// Get template by ID
    pub async fn get_template(
        &self,
        template_id: Uuid,
        user_id: Uuid,
    ) -> Result<Option<EmailTemplate>, sqlx::Error> {
        sqlx::query_as::<_, EmailTemplate>(
            r#"
            SELECT * FROM email_templates
            WHERE id = $1 AND (user_id = $2 OR is_shared = true)
            "#,
        )
        .bind(template_id)
        .bind(user_id)
        .fetch_optional(&self.pool)
        .await
    }

    /// List user's templates
    pub async fn list_templates(
        &self,
        user_id: Uuid,
        tenant_id: Option<Uuid>,
        category: Option<&str>,
    ) -> Result<Vec<EmailTemplate>, sqlx::Error> {
        let query = if let Some(cat) = category {
            sqlx::query_as::<_, EmailTemplate>(
                r#"
                SELECT * FROM email_templates
                WHERE (user_id = $1 OR is_shared = true)
                  AND ($2::uuid IS NULL OR tenant_id = $2)
                  AND category = $3
                ORDER BY use_count DESC, name ASC
                "#,
            )
            .bind(user_id)
            .bind(tenant_id)
            .bind(cat)
        } else {
            sqlx::query_as::<_, EmailTemplate>(
                r#"
                SELECT * FROM email_templates
                WHERE (user_id = $1 OR is_shared = true)
                  AND ($2::uuid IS NULL OR tenant_id = $2)
                ORDER BY use_count DESC, name ASC
                "#,
            )
            .bind(user_id)
            .bind(tenant_id)
        };

        query.fetch_all(&self.pool).await
    }

    /// Render template with variables
    pub fn render(
        &self,
        template: &EmailTemplate,
        variables: &JsonValue,
    ) -> Result<RenderedTemplate, TemplateError> {
        let subject = self
            .handlebars
            .render_template(&template.subject, variables)
            .map_err(|e| TemplateError::Render(e.to_string()))?;

        let body_text = self
            .handlebars
            .render_template(&template.body_text, variables)
            .map_err(|e| TemplateError::Render(e.to_string()))?;

        let body_html = if let Some(ref html) = template.body_html {
            Some(
                self.handlebars
                    .render_template(html, variables)
                    .map_err(|e| TemplateError::Render(e.to_string()))?,
            )
        } else {
            None
        };

        Ok(RenderedTemplate {
            subject,
            body_text,
            body_html,
        })
    }

    /// Start a mail merge job
    pub async fn start_mail_merge(
        &self,
        template_id: Uuid,
        user_id: Uuid,
        tenant_id: Option<Uuid>,
        recipients: Vec<MergeRecipient>,
    ) -> Result<MailMergeJob, TemplateError> {
        let job_id = Uuid::new_v4();

        let job = sqlx::query_as::<_, MailMergeJob>(
            r#"
            INSERT INTO mail_merge_jobs (
                id, template_id, user_id, tenant_id,
                total_recipients, sent_count, failed_count,
                status, created_at
            )
            VALUES ($1, $2, $3, $4, $5, 0, 0, 'pending', NOW())
            RETURNING *
            "#,
        )
        .bind(job_id)
        .bind(template_id)
        .bind(user_id)
        .bind(tenant_id)
        .bind(recipients.len() as i32)
        .fetch_one(&self.pool)
        .await
        .map_err(TemplateError::Database)?;

        // Store recipients
        for recipient in recipients {
            sqlx::query(
                r#"
                INSERT INTO mail_merge_recipients (
                    id, job_id, email, name, variables, status
                )
                VALUES ($1, $2, $3, $4, $5, 'pending')
                "#,
            )
            .bind(Uuid::new_v4())
            .bind(job_id)
            .bind(&recipient.email)
            .bind(&recipient.name)
            .bind(&recipient.variables)
            .execute(&self.pool)
            .await
            .map_err(TemplateError::Database)?;
        }

        Ok(job)
    }

    /// Increment template use count
    pub async fn increment_use_count(&self, template_id: Uuid) -> Result<(), sqlx::Error> {
        sqlx::query(
            "UPDATE email_templates SET use_count = use_count + 1 WHERE id = $1",
        )
        .bind(template_id)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Delete template
    pub async fn delete_template(
        &self,
        template_id: Uuid,
        user_id: Uuid,
    ) -> Result<bool, sqlx::Error> {
        let result = sqlx::query(
            "DELETE FROM email_templates WHERE id = $1 AND user_id = $2",
        )
        .bind(template_id)
        .bind(user_id)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected() > 0)
    }
}

// Handlebars helpers
fn uppercase_helper(
    h: &handlebars::Helper,
    _: &Handlebars,
    _: &handlebars::Context,
    _: &mut handlebars::RenderContext,
    out: &mut dyn handlebars::Output,
) -> handlebars::HelperResult {
    let param = h.param(0).and_then(|v| v.value().as_str()).unwrap_or("");
    out.write(&param.to_uppercase())?;
    Ok(())
}

fn lowercase_helper(
    h: &handlebars::Helper,
    _: &Handlebars,
    _: &handlebars::Context,
    _: &mut handlebars::RenderContext,
    out: &mut dyn handlebars::Output,
) -> handlebars::HelperResult {
    let param = h.param(0).and_then(|v| v.value().as_str()).unwrap_or("");
    out.write(&param.to_lowercase())?;
    Ok(())
}

fn format_date_helper(
    h: &handlebars::Helper,
    _: &Handlebars,
    _: &handlebars::Context,
    _: &mut handlebars::RenderContext,
    out: &mut dyn handlebars::Output,
) -> handlebars::HelperResult {
    let date = h.param(0).and_then(|v| v.value().as_str()).unwrap_or("");
    let format = h.param(1).and_then(|v| v.value().as_str()).unwrap_or("%B %d, %Y");

    if let Ok(dt) = DateTime::parse_from_rfc3339(date) {
        out.write(&dt.format(format).to_string())?;
    } else {
        out.write(date)?;
    }
    Ok(())
}

fn format_currency_helper(
    h: &handlebars::Helper,
    _: &Handlebars,
    _: &handlebars::Context,
    _: &mut handlebars::RenderContext,
    out: &mut dyn handlebars::Output,
) -> handlebars::HelperResult {
    let amount = h.param(0).and_then(|v| v.value().as_f64()).unwrap_or(0.0);
    let currency = h.param(1).and_then(|v| v.value().as_str()).unwrap_or("$");

    out.write(&format!("{}{:.2}", currency, amount))?;
    Ok(())
}

/// Template errors
#[derive(Debug)]
pub enum TemplateError {
    NotFound,
    Validation(String),
    Render(String),
    Database(sqlx::Error),
}

impl std::fmt::Display for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "Template not found"),
            Self::Validation(msg) => write!(f, "Template validation error: {}", msg),
            Self::Render(msg) => write!(f, "Template render error: {}", msg),
            Self::Database(e) => write!(f, "Database error: {}", e),
        }
    }
}

impl std::error::Error for TemplateError {}

/// Migration for templates
pub const TEMPLATE_MIGRATION: &str = r#"
CREATE TABLE IF NOT EXISTS email_templates (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    subject TEXT NOT NULL,
    body_text TEXT NOT NULL,
    body_html TEXT,
    variables JSONB NOT NULL DEFAULT '[]',
    category VARCHAR(100),
    is_shared BOOLEAN NOT NULL DEFAULT false,
    use_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_email_templates_user ON email_templates(user_id);
CREATE INDEX IF NOT EXISTS idx_email_templates_tenant ON email_templates(tenant_id);
CREATE INDEX IF NOT EXISTS idx_email_templates_category ON email_templates(category);

DO $$ BEGIN
    CREATE TYPE mail_merge_status AS ENUM ('pending', 'processing', 'completed', 'failed', 'cancelled');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

CREATE TABLE IF NOT EXISTS mail_merge_jobs (
    id UUID PRIMARY KEY,
    template_id UUID NOT NULL REFERENCES email_templates(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    total_recipients INTEGER NOT NULL,
    sent_count INTEGER NOT NULL DEFAULT 0,
    failed_count INTEGER NOT NULL DEFAULT 0,
    status mail_merge_status NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS mail_merge_recipients (
    id UUID PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES mail_merge_jobs(id) ON DELETE CASCADE,
    email VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    variables JSONB NOT NULL DEFAULT '{}',
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    error_message TEXT,
    sent_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_mail_merge_recipients_job ON mail_merge_recipients(job_id);
"#;
