// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Templates & Snippets Module
 *
 * Reusable email templates, text snippets, and merge field support
 */

use sqlx::PgPool;
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use regex::Regex;

#[derive(Debug, thiserror::Error)]
pub enum TemplateError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("Template not found")]
    NotFound,
    #[error("Invalid template syntax: {0}")]
    InvalidSyntax(String),
    #[error("Missing required field: {0}")]
    MissingField(String),
    #[error("Snippet trigger already exists")]
    DuplicateTrigger,
    #[error("Template limit exceeded")]
    LimitExceeded,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmailTemplate {
    pub id: Uuid,
    pub user_id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub subject: String,
    pub body_html: String,
    pub body_text: String,
    pub category: Option<String>,
    pub fields: Vec<TemplateField>,
    pub is_shared: bool,
    pub use_count: i32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TemplateField {
    pub name: String,
    pub field_type: FieldType,
    pub label: String,
    pub placeholder: Option<String>,
    pub default_value: Option<String>,
    pub required: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum FieldType {
    Text,
    LongText,
    Email,
    Date,
    Number,
    Select,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TemplateInput {
    pub name: String,
    pub description: Option<String>,
    pub subject: String,
    pub body_html: String,
    pub body_text: Option<String>,
    pub category: Option<String>,
    pub is_shared: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TextSnippet {
    pub id: Uuid,
    pub user_id: Uuid,
    pub trigger: String,
    pub expansion: String,
    pub description: Option<String>,
    pub category: Option<String>,
    pub use_count: i32,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SnippetInput {
    pub trigger: String,
    pub expansion: String,
    pub description: Option<String>,
    pub category: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MergeContext {
    pub recipient: Option<RecipientInfo>,
    pub sender: Option<SenderInfo>,
    pub custom_fields: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RecipientInfo {
    pub email: String,
    pub first_name: Option<String>,
    pub last_name: Option<String>,
    pub company: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SenderInfo {
    pub email: String,
    pub first_name: Option<String>,
    pub last_name: Option<String>,
    pub signature: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RenderedTemplate {
    pub subject: String,
    pub body_html: String,
    pub body_text: String,
    pub missing_fields: Vec<String>,
}

pub struct TemplateService {
    pool: PgPool,
}

impl TemplateService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn create_template(&self, user_id: Uuid, input: TemplateInput) -> Result<EmailTemplate, TemplateError> {
        let count: i64 = sqlx::query_scalar!("SELECT COUNT(*) FROM email_templates WHERE user_id = $1", user_id)
            .fetch_one(&self.pool).await?.unwrap_or(0);

        if count >= 100 { return Err(TemplateError::LimitExceeded); }

        let fields = self.extract_fields(&input.subject, &input.body_html)?;
        let body_text = input.body_text.unwrap_or_else(|| html2text::from_read(input.body_html.as_bytes(), 80));
        let id = Uuid::new_v4();
        let fields_json = serde_json::to_value(&fields).map_err(|e| TemplateError::InvalidSyntax(e.to_string()))?;

        sqlx::query!(
            r#"INSERT INTO email_templates (id, user_id, name, description, subject, body_html, body_text, category, fields, is_shared, use_count, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 0, NOW(), NOW())"#,
            id, user_id, input.name, input.description, input.subject, input.body_html, body_text,
            input.category, fields_json, input.is_shared.unwrap_or(false)
        ).execute(&self.pool).await?;

        self.get_template(user_id, id).await
    }

    pub async fn get_template(&self, user_id: Uuid, template_id: Uuid) -> Result<EmailTemplate, TemplateError> {
        let row = sqlx::query!(
            "SELECT id, user_id, name, description, subject, body_html, body_text, category, fields, is_shared, use_count, created_at, updated_at FROM email_templates WHERE id = $1 AND (user_id = $2 OR is_shared = true)",
            template_id, user_id
        ).fetch_optional(&self.pool).await?.ok_or(TemplateError::NotFound)?;

        let fields: Vec<TemplateField> = serde_json::from_value(row.fields).unwrap_or_default();

        Ok(EmailTemplate {
            id: row.id, user_id: row.user_id, name: row.name, description: row.description,
            subject: row.subject, body_html: row.body_html, body_text: row.body_text,
            category: row.category, fields, is_shared: row.is_shared, use_count: row.use_count,
            created_at: row.created_at, updated_at: row.updated_at,
        })
    }

    pub async fn list_templates(&self, user_id: Uuid, category: Option<&str>) -> Result<Vec<EmailTemplate>, TemplateError> {
        let rows = sqlx::query!(
            "SELECT id, user_id, name, description, subject, body_html, body_text, category, fields, is_shared, use_count, created_at, updated_at FROM email_templates WHERE (user_id = $1 OR is_shared = true) AND ($2::text IS NULL OR category = $2) ORDER BY use_count DESC",
            user_id, category
        ).fetch_all(&self.pool).await?;

        Ok(rows.into_iter().map(|row| {
            let fields: Vec<TemplateField> = serde_json::from_value(row.fields).unwrap_or_default();
            EmailTemplate {
                id: row.id, user_id: row.user_id, name: row.name, description: row.description,
                subject: row.subject, body_html: row.body_html, body_text: row.body_text,
                category: row.category, fields, is_shared: row.is_shared, use_count: row.use_count,
                created_at: row.created_at, updated_at: row.updated_at,
            }
        }).collect())
    }

    pub async fn render_template(&self, user_id: Uuid, template_id: Uuid, context: MergeContext) -> Result<RenderedTemplate, TemplateError> {
        let template = self.get_template(user_id, template_id).await?;
        let mut data = context.custom_fields.clone();

        if let Some(ref r) = context.recipient {
            data.insert("recipient.email".to_string(), r.email.clone());
            if let Some(ref v) = r.first_name { data.insert("recipient.firstName".to_string(), v.clone()); }
            if let Some(ref v) = r.last_name { data.insert("recipient.lastName".to_string(), v.clone()); }
            if let Some(ref v) = r.company { data.insert("recipient.company".to_string(), v.clone()); }
        }
        if let Some(ref s) = context.sender {
            data.insert("sender.email".to_string(), s.email.clone());
            if let Some(ref v) = s.first_name { data.insert("sender.firstName".to_string(), v.clone()); }
            if let Some(ref v) = s.last_name { data.insert("sender.lastName".to_string(), v.clone()); }
            if let Some(ref v) = s.signature { data.insert("sender.signature".to_string(), v.clone()); }
        }

        data.insert("date.today".to_string(), Utc::now().format("%B %d, %Y").to_string());

        let (subject, s_missing) = self.merge_fields(&template.subject, &data);
        let (body_html, h_missing) = self.merge_fields(&template.body_html, &data);
        let (body_text, t_missing) = self.merge_fields(&template.body_text, &data);

        let mut missing_fields: Vec<String> = [s_missing, h_missing, t_missing].concat();
        missing_fields.sort();
        missing_fields.dedup();

        for field in &template.fields {
            if field.required && missing_fields.contains(&field.name) {
                return Err(TemplateError::MissingField(field.name.clone()));
            }
        }

        sqlx::query!("UPDATE email_templates SET use_count = use_count + 1 WHERE id = $1", template_id)
            .execute(&self.pool).await?;

        Ok(RenderedTemplate { subject, body_html, body_text, missing_fields })
    }

    pub async fn delete_template(&self, user_id: Uuid, template_id: Uuid) -> Result<(), TemplateError> {
        let result = sqlx::query!("DELETE FROM email_templates WHERE id = $1 AND user_id = $2", template_id, user_id)
            .execute(&self.pool).await?;
        if result.rows_affected() == 0 { return Err(TemplateError::NotFound); }
        Ok(())
    }

    fn extract_fields(&self, subject: &str, body: &str) -> Result<Vec<TemplateField>, TemplateError> {
        let mut fields = Vec::new();
        let re = Regex::new(r"\{\{([a-zA-Z_][a-zA-Z0-9_.]*)\}\}").map_err(|e| TemplateError::InvalidSyntax(e.to_string()))?;

        for cap in re.captures_iter(&format!("{} {}", subject, body)) {
            let name = cap[1].to_string();
            if fields.iter().any(|f: &TemplateField| f.name == name) { continue; }
            let field_type = if name.contains("email") { FieldType::Email } else if name.contains("date") { FieldType::Date } else { FieldType::Text };
            let label = name.split('.').last().unwrap_or(&name).to_string();
            fields.push(TemplateField { name, field_type, label, placeholder: None, default_value: None, required: false });
        }
        Ok(fields)
    }

    fn merge_fields(&self, template: &str, data: &HashMap<String, String>) -> (String, Vec<String>) {
        let mut result = template.to_string();
        let mut missing = Vec::new();
        let re = Regex::new(r"\{\{([a-zA-Z_][a-zA-Z0-9_.]*)\}\}").unwrap();

        for cap in re.captures_iter(template) {
            let full = &cap[0];
            let name = &cap[1];
            if let Some(value) = data.get(name) { result = result.replace(full, value); }
            else { missing.push(name.to_string()); }
        }
        (result, missing)
    }
}

pub struct SnippetService {
    pool: PgPool,
}

impl SnippetService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn create_snippet(&self, user_id: Uuid, input: SnippetInput) -> Result<TextSnippet, TemplateError> {
        if input.trigger.len() > 20 { return Err(TemplateError::InvalidSyntax("Trigger must be 20 chars or less".to_string())); }

        let exists: bool = sqlx::query_scalar!("SELECT EXISTS(SELECT 1 FROM text_snippets WHERE user_id = $1 AND trigger = $2)", user_id, input.trigger)
            .fetch_one(&self.pool).await?.unwrap_or(false);
        if exists { return Err(TemplateError::DuplicateTrigger); }

        let id = Uuid::new_v4();
        sqlx::query!("INSERT INTO text_snippets (id, user_id, trigger, expansion, description, category, use_count, created_at) VALUES ($1, $2, $3, $4, $5, $6, 0, NOW())",
            id, user_id, input.trigger, input.expansion, input.description, input.category
        ).execute(&self.pool).await?;

        self.get_snippet(user_id, id).await
    }

    pub async fn get_snippet(&self, user_id: Uuid, snippet_id: Uuid) -> Result<TextSnippet, TemplateError> {
        let row = sqlx::query!("SELECT id, user_id, trigger, expansion, description, category, use_count, created_at FROM text_snippets WHERE id = $1 AND user_id = $2", snippet_id, user_id)
            .fetch_optional(&self.pool).await?.ok_or(TemplateError::NotFound)?;

        Ok(TextSnippet { id: row.id, user_id: row.user_id, trigger: row.trigger, expansion: row.expansion, description: row.description, category: row.category, use_count: row.use_count, created_at: row.created_at })
    }

    pub async fn list_snippets(&self, user_id: Uuid) -> Result<Vec<TextSnippet>, TemplateError> {
        let rows = sqlx::query!("SELECT id, user_id, trigger, expansion, description, category, use_count, created_at FROM text_snippets WHERE user_id = $1 ORDER BY use_count DESC", user_id)
            .fetch_all(&self.pool).await?;

        Ok(rows.into_iter().map(|r| TextSnippet { id: r.id, user_id: r.user_id, trigger: r.trigger, expansion: r.expansion, description: r.description, category: r.category, use_count: r.use_count, created_at: r.created_at }).collect())
    }

    pub async fn expand_trigger(&self, user_id: Uuid, trigger: &str) -> Result<Option<String>, TemplateError> {
        let snippet = sqlx::query!("SELECT id, expansion FROM text_snippets WHERE user_id = $1 AND trigger = $2", user_id, trigger)
            .fetch_optional(&self.pool).await?;

        if let Some(s) = snippet {
            sqlx::query!("UPDATE text_snippets SET use_count = use_count + 1 WHERE id = $1", s.id).execute(&self.pool).await?;
            Ok(Some(s.expansion))
        } else { Ok(None) }
    }

    pub async fn delete_snippet(&self, user_id: Uuid, snippet_id: Uuid) -> Result<(), TemplateError> {
        let result = sqlx::query!("DELETE FROM text_snippets WHERE id = $1 AND user_id = $2", snippet_id, user_id)
            .execute(&self.pool).await?;
        if result.rows_affected() == 0 { return Err(TemplateError::NotFound); }
        Ok(())
    }

    pub async fn get_triggers(&self, user_id: Uuid) -> Result<HashMap<String, String>, TemplateError> {
        let rows = sqlx::query!("SELECT trigger, expansion FROM text_snippets WHERE user_id = $1", user_id)
            .fetch_all(&self.pool).await?;
        Ok(rows.into_iter().map(|r| (r.trigger, r.expansion)).collect())
    }
}

pub struct SignatureService {
    pool: PgPool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Signature {
    pub id: Uuid,
    pub user_id: Uuid,
    pub name: String,
    pub content_html: String,
    pub content_text: String,
    pub is_default: bool,
    pub created_at: DateTime<Utc>,
}

impl SignatureService {
    pub fn new(pool: PgPool) -> Self { Self { pool } }

    pub async fn create_signature(&self, user_id: Uuid, name: String, content_html: String, is_default: bool) -> Result<Signature, TemplateError> {
        let id = Uuid::new_v4();
        let content_text = html2text::from_read(content_html.as_bytes(), 80);

        if is_default {
            sqlx::query!("UPDATE signatures SET is_default = false WHERE user_id = $1", user_id).execute(&self.pool).await?;
        }

        sqlx::query!("INSERT INTO signatures (id, user_id, name, content_html, content_text, is_default, created_at) VALUES ($1, $2, $3, $4, $5, $6, NOW())",
            id, user_id, name, content_html, content_text, is_default
        ).execute(&self.pool).await?;

        Ok(Signature { id, user_id, name, content_html, content_text, is_default, created_at: Utc::now() })
    }

    pub async fn list_signatures(&self, user_id: Uuid) -> Result<Vec<Signature>, TemplateError> {
        let rows = sqlx::query!("SELECT id, user_id, name, content_html, content_text, is_default, created_at FROM signatures WHERE user_id = $1 ORDER BY is_default DESC", user_id)
            .fetch_all(&self.pool).await?;

        Ok(rows.into_iter().map(|r| Signature { id: r.id, user_id: r.user_id, name: r.name, content_html: r.content_html, content_text: r.content_text, is_default: r.is_default, created_at: r.created_at }).collect())
    }

    pub async fn get_default_signature(&self, user_id: Uuid) -> Result<Option<Signature>, TemplateError> {
        let row = sqlx::query!("SELECT id, user_id, name, content_html, content_text, is_default, created_at FROM signatures WHERE user_id = $1 AND is_default = true", user_id)
            .fetch_optional(&self.pool).await?;

        Ok(row.map(|r| Signature { id: r.id, user_id: r.user_id, name: r.name, content_html: r.content_html, content_text: r.content_text, is_default: r.is_default, created_at: r.created_at }))
    }

    pub async fn delete_signature(&self, user_id: Uuid, signature_id: Uuid) -> Result<(), TemplateError> {
        let result = sqlx::query!("DELETE FROM signatures WHERE id = $1 AND user_id = $2", signature_id, user_id).execute(&self.pool).await?;
        if result.rows_affected() == 0 { return Err(TemplateError::NotFound); }
        Ok(())
    }
}
