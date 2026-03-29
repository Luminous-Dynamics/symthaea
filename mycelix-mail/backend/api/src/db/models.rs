// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Database models for Mycelix Mail
//!
//! SQLx-compatible models with FromRow derive

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;

// ============================================================================
// User Models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct User {
    pub id: Uuid,
    pub email: String,
    pub display_name: String,
    pub avatar_url: Option<String>,
    pub holochain_agent_id: Option<String>,
    pub public_key: Option<Vec<u8>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_login_at: Option<DateTime<Utc>>,
    pub is_active: bool,
    pub email_verified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateUser {
    pub email: String,
    pub display_name: String,
    pub avatar_url: Option<String>,
    pub holochain_agent_id: Option<String>,
    pub public_key: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UpdateUser {
    pub display_name: Option<String>,
    pub avatar_url: Option<String>,
    pub holochain_agent_id: Option<String>,
    pub public_key: Option<Vec<u8>>,
}

// ============================================================================
// Email Models
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize, sqlx::Type, PartialEq)]
#[sqlx(type_name = "email_status", rename_all = "lowercase")]
pub enum EmailStatus {
    Draft,
    Queued,
    Sent,
    Delivered,
    Failed,
    Bounced,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Email {
    pub id: Uuid,
    pub user_id: Uuid,
    pub thread_id: Option<Uuid>,
    pub message_id: String,
    pub in_reply_to: Option<String>,
    pub subject: String,
    pub body_text: Option<String>,
    pub body_html: Option<String>,
    pub from_address: String,
    pub to_addresses: Vec<String>,
    pub cc_addresses: Vec<String>,
    pub bcc_addresses: Vec<String>,
    pub status: EmailStatus,
    pub is_read: bool,
    pub is_starred: bool,
    pub is_archived: bool,
    pub is_deleted: bool,
    pub trust_score: Option<f64>,
    pub spam_score: Option<f64>,
    pub labels: Vec<String>,
    pub sent_at: Option<DateTime<Utc>>,
    pub received_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateEmail {
    pub user_id: Uuid,
    pub thread_id: Option<Uuid>,
    pub in_reply_to: Option<String>,
    pub subject: String,
    pub body_text: Option<String>,
    pub body_html: Option<String>,
    pub from_address: String,
    pub to_addresses: Vec<String>,
    pub cc_addresses: Vec<String>,
    pub bcc_addresses: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UpdateEmail {
    pub subject: Option<String>,
    pub body_text: Option<String>,
    pub body_html: Option<String>,
    pub is_read: Option<bool>,
    pub is_starred: Option<bool>,
    pub is_archived: Option<bool>,
    pub labels: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmailFilter {
    pub user_id: Option<Uuid>,
    pub thread_id: Option<Uuid>,
    pub is_read: Option<bool>,
    pub is_starred: Option<bool>,
    pub is_archived: Option<bool>,
    pub is_deleted: Option<bool>,
    pub labels: Option<Vec<String>>,
    pub from_address: Option<String>,
    pub search: Option<String>,
    pub min_trust_score: Option<f64>,
    pub since: Option<DateTime<Utc>>,
    pub until: Option<DateTime<Utc>>,
}

// ============================================================================
// Attachment Models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Attachment {
    pub id: Uuid,
    pub email_id: Uuid,
    pub filename: String,
    pub content_type: String,
    pub size_bytes: i64,
    pub storage_path: String,
    pub checksum: String,
    pub is_inline: bool,
    pub content_id: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateAttachment {
    pub email_id: Uuid,
    pub filename: String,
    pub content_type: String,
    pub size_bytes: i64,
    pub storage_path: String,
    pub checksum: String,
    pub is_inline: bool,
    pub content_id: Option<String>,
}

// ============================================================================
// Contact Models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Contact {
    pub id: Uuid,
    pub user_id: Uuid,
    pub email: String,
    pub display_name: Option<String>,
    pub avatar_url: Option<String>,
    pub holochain_agent_id: Option<String>,
    pub trust_score: f64,
    pub interaction_count: i32,
    pub last_interaction_at: Option<DateTime<Utc>>,
    pub is_blocked: bool,
    pub notes: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateContact {
    pub user_id: Uuid,
    pub email: String,
    pub display_name: Option<String>,
    pub holochain_agent_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UpdateContact {
    pub display_name: Option<String>,
    pub avatar_url: Option<String>,
    pub holochain_agent_id: Option<String>,
    pub trust_score: Option<f64>,
    pub is_blocked: Option<bool>,
    pub notes: Option<String>,
}

// ============================================================================
// Trust Attestation Models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct TrustAttestation {
    pub id: Uuid,
    pub from_user_id: Uuid,
    pub to_agent_id: String,
    pub trust_level: f64,
    pub context: String,
    pub signature: Vec<u8>,
    pub holochain_hash: Option<String>,
    pub expires_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub revoked_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateTrustAttestation {
    pub from_user_id: Uuid,
    pub to_agent_id: String,
    pub trust_level: f64,
    pub context: String,
    pub signature: Vec<u8>,
    pub holochain_hash: Option<String>,
    pub expires_at: Option<DateTime<Utc>>,
}

// ============================================================================
// Label Models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Label {
    pub id: Uuid,
    pub user_id: Uuid,
    pub name: String,
    pub color: String,
    pub is_system: bool,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateLabel {
    pub user_id: Uuid,
    pub name: String,
    pub color: String,
}

// ============================================================================
// Thread Models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Thread {
    pub id: Uuid,
    pub user_id: Uuid,
    pub subject: String,
    pub participant_addresses: Vec<String>,
    pub message_count: i32,
    pub unread_count: i32,
    pub last_message_at: DateTime<Utc>,
    pub is_archived: bool,
    pub is_deleted: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// ============================================================================
// Workflow Models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Workflow {
    pub id: Uuid,
    pub user_id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub trigger_type: String,
    pub trigger_config: serde_json::Value,
    pub actions: serde_json::Value,
    pub is_active: bool,
    pub run_count: i32,
    pub last_run_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateWorkflow {
    pub user_id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub trigger_type: String,
    pub trigger_config: serde_json::Value,
    pub actions: serde_json::Value,
}

// ============================================================================
// Pagination
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pagination {
    pub page: u32,
    pub per_page: u32,
}

impl Default for Pagination {
    fn default() -> Self {
        Self {
            page: 1,
            per_page: 50,
        }
    }
}

impl Pagination {
    pub fn offset(&self) -> u32 {
        (self.page.saturating_sub(1)) * self.per_page
    }

    pub fn limit(&self) -> u32 {
        self.per_page.min(100) // Cap at 100
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginatedResult<T> {
    pub items: Vec<T>,
    pub total: i64,
    pub page: u32,
    pub per_page: u32,
    pub total_pages: u32,
}

impl<T> PaginatedResult<T> {
    pub fn new(items: Vec<T>, total: i64, pagination: &Pagination) -> Self {
        let total_pages = ((total as f64) / (pagination.per_page as f64)).ceil() as u32;
        Self {
            items,
            total,
            page: pagination.page,
            per_page: pagination.per_page,
            total_pages,
        }
    }
}
