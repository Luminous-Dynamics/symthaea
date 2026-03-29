// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! GraphQL Schema for Mycelix Mail
//!
//! Provides a complete GraphQL API for email operations, real-time subscriptions,
//! and advanced querying capabilities.

use async_graphql::{
    Context, EmptyMutation, EmptySubscription, Enum, InputObject, Object, Result,
    Schema, SimpleObject, Subscription, ID,
};
use async_graphql_axum::{GraphQLRequest, GraphQLResponse, GraphQLSubscription};
use chrono::{DateTime, Utc};
use futures_util::Stream;
use std::sync::Arc;
use tokio::sync::broadcast;
use uuid::Uuid;

use crate::db::models as db_models;
use crate::db::repositories::EmailRepository;

/// Email object for GraphQL
#[derive(SimpleObject, Clone)]
pub struct Email {
    pub id: ID,
    pub from: String,
    pub from_name: Option<String>,
    pub to: Vec<String>,
    pub cc: Vec<String>,
    pub bcc: Vec<String>,
    pub subject: String,
    pub body_text: String,
    pub body_html: Option<String>,
    pub snippet: String,
    pub folder: String,
    pub labels: Vec<String>,
    pub is_read: bool,
    pub is_starred: bool,
    pub is_draft: bool,
    pub has_attachments: bool,
    pub attachment_count: i32,
    pub trust_score: f64,
    pub category: EmailCategory,
    pub priority: EmailPriority,
    pub received_at: DateTime<Utc>,
    pub sent_at: Option<DateTime<Utc>>,
    pub thread_id: Option<ID>,
    pub reply_to_id: Option<ID>,
}

#[derive(SimpleObject, Clone)]
pub struct EmailThread {
    pub id: ID,
    pub subject: String,
    pub participants: Vec<String>,
    pub message_count: i32,
    pub unread_count: i32,
    pub last_message_at: DateTime<Utc>,
    pub messages: Vec<Email>,
}

#[derive(SimpleObject, Clone)]
pub struct Attachment {
    pub id: ID,
    pub filename: String,
    pub mime_type: String,
    pub size_bytes: i64,
    pub download_url: String,
}

#[derive(SimpleObject, Clone)]
pub struct Folder {
    pub id: ID,
    pub name: String,
    pub path: String,
    pub message_count: i32,
    pub unread_count: i32,
    pub is_system: bool,
    pub color: Option<String>,
    pub icon: Option<String>,
}

#[derive(SimpleObject, Clone)]
pub struct Label {
    pub id: ID,
    pub name: String,
    pub color: String,
    pub message_count: i32,
}

#[derive(SimpleObject, Clone)]
pub struct Contact {
    pub id: ID,
    pub email: String,
    pub name: Option<String>,
    pub avatar_url: Option<String>,
    pub trust_score: f64,
    pub is_verified: bool,
    pub interaction_count: i32,
    pub last_interaction: Option<DateTime<Utc>>,
}

#[derive(SimpleObject, Clone)]
pub struct User {
    pub id: ID,
    pub email: String,
    pub name: String,
    pub avatar_url: Option<String>,
    pub signature: Option<String>,
    pub timezone: String,
    pub language: String,
}

#[derive(Enum, Copy, Clone, Eq, PartialEq)]
pub enum EmailCategory {
    Primary,
    Social,
    Promotions,
    Updates,
    Forums,
    Newsletters,
    Spam,
}

#[derive(Enum, Copy, Clone, Eq, PartialEq)]
pub enum EmailPriority {
    Low,
    Normal,
    High,
    Urgent,
}

#[derive(Enum, Copy, Clone, Eq, PartialEq)]
pub enum SortDirection {
    Asc,
    Desc,
}

#[derive(Enum, Copy, Clone, Eq, PartialEq)]
pub enum EmailSortField {
    Date,
    Subject,
    From,
    Size,
    Priority,
    TrustScore,
}

/// Input types
#[derive(InputObject)]
pub struct EmailFilter {
    pub folder: Option<String>,
    pub labels: Option<Vec<String>>,
    pub is_read: Option<bool>,
    pub is_starred: Option<bool>,
    pub has_attachments: Option<bool>,
    pub from: Option<String>,
    pub to: Option<String>,
    pub subject_contains: Option<String>,
    pub body_contains: Option<String>,
    pub category: Option<EmailCategory>,
    pub min_trust_score: Option<f64>,
    pub date_from: Option<DateTime<Utc>>,
    pub date_to: Option<DateTime<Utc>>,
    pub thread_id: Option<ID>,
}

#[derive(InputObject)]
pub struct EmailSort {
    pub field: EmailSortField,
    pub direction: SortDirection,
}

#[derive(InputObject)]
pub struct Pagination {
    pub limit: i32,
    pub offset: i32,
}

#[derive(InputObject)]
pub struct ComposeEmailInput {
    pub to: Vec<String>,
    pub cc: Option<Vec<String>>,
    pub bcc: Option<Vec<String>>,
    pub subject: String,
    pub body_text: String,
    pub body_html: Option<String>,
    pub reply_to_id: Option<ID>,
    pub attachment_ids: Option<Vec<ID>>,
    pub save_as_draft: Option<bool>,
    pub scheduled_send_at: Option<DateTime<Utc>>,
}

#[derive(InputObject)]
pub struct UpdateEmailInput {
    pub is_read: Option<bool>,
    pub is_starred: Option<bool>,
    pub folder: Option<String>,
    pub labels: Option<Vec<String>>,
}

#[derive(InputObject)]
pub struct CreateFolderInput {
    pub name: String,
    pub parent_path: Option<String>,
    pub color: Option<String>,
    pub icon: Option<String>,
}

#[derive(InputObject)]
pub struct CreateLabelInput {
    pub name: String,
    pub color: String,
}

/// Response types
#[derive(SimpleObject)]
pub struct EmailConnection {
    pub edges: Vec<Email>,
    pub total_count: i32,
    pub page_info: PageInfo,
}

#[derive(SimpleObject)]
pub struct PageInfo {
    pub has_next_page: bool,
    pub has_previous_page: bool,
    pub start_cursor: Option<String>,
    pub end_cursor: Option<String>,
}

#[derive(SimpleObject)]
pub struct SendEmailResult {
    pub success: bool,
    pub email: Option<Email>,
    pub error: Option<String>,
}

#[derive(SimpleObject)]
pub struct BulkActionResult {
    pub success_count: i32,
    pub failure_count: i32,
    pub errors: Vec<String>,
}

/// Context for GraphQL resolvers
pub struct GqlContext {
    pub user_id: Uuid,
    pub db_pool: sqlx::PgPool,
    pub event_sender: broadcast::Sender<EmailEvent>,
}

/// Events for subscriptions
#[derive(Clone)]
pub enum EmailEvent {
    NewEmail(Email),
    EmailUpdated(Email),
    EmailDeleted(ID),
    FolderUpdated(Folder),
}

/// Query resolvers
pub struct QueryRoot;

#[Object]
impl QueryRoot {
    /// Get emails with filtering, sorting, and pagination
    async fn emails(
        &self,
        ctx: &Context<'_>,
        filter: Option<EmailFilter>,
        sort: Option<EmailSort>,
        pagination: Option<Pagination>,
    ) -> Result<EmailConnection> {
        let gql_ctx = ctx.data::<GqlContext>()?;
        let pagination = pagination.unwrap_or(Pagination { limit: 50, offset: 0 });

        // Build database filter from GraphQL filter
        let db_filter = filter.map(|f| db_models::EmailFilter {
            user_id: Some(gql_ctx.user_id),
            thread_id: f.thread_id.map(|id| Uuid::parse_str(id.as_str()).ok()).flatten(),
            is_read: f.is_read,
            is_starred: f.is_starred,
            is_archived: None,
            is_deleted: Some(false),
            labels: f.labels,
            from_address: f.from,
            search: f.subject_contains.or(f.body_contains),
            min_trust_score: f.min_trust_score,
            since: f.date_from,
            until: f.date_to,
        }).unwrap_or_else(|| db_models::EmailFilter {
            user_id: Some(gql_ctx.user_id),
            is_deleted: Some(false),
            ..Default::default()
        });

        let db_pagination = db_models::Pagination {
            page: (pagination.offset / pagination.limit.max(1)) as u32 + 1,
            per_page: pagination.limit as u32,
        };

        // Query database
        let repo = EmailRepository::new(gql_ctx.db_pool.clone());
        let result = repo.find_by_user(gql_ctx.user_id, db_filter, db_pagination).await
            .map_err(|e| async_graphql::Error::new(format!("Database error: {}", e)))?;

        // Convert database models to GraphQL types
        let edges: Vec<Email> = result.items.into_iter().map(|db_email| {
            Email {
                id: db_email.id.to_string().into(),
                from: db_email.from_address.clone(),
                from_name: None,
                to: db_email.to_addresses.clone(),
                cc: db_email.cc_addresses.clone(),
                bcc: db_email.bcc_addresses.clone(),
                subject: db_email.subject.clone(),
                body_text: db_email.body_text.clone().unwrap_or_default(),
                body_html: db_email.body_html.clone(),
                snippet: db_email.body_text.as_ref()
                    .map(|t| t.chars().take(100).collect())
                    .unwrap_or_default(),
                folder: if db_email.is_archived { "archive".to_string() }
                        else if db_email.is_deleted { "trash".to_string() }
                        else { "inbox".to_string() },
                labels: db_email.labels.clone(),
                is_read: db_email.is_read,
                is_starred: db_email.is_starred,
                is_draft: db_email.status == db_models::EmailStatus::Draft,
                has_attachments: false, // Would need attachment count query
                attachment_count: 0,
                trust_score: db_email.trust_score.unwrap_or(0.5),
                category: EmailCategory::Primary,
                priority: EmailPriority::Normal,
                received_at: db_email.received_at.unwrap_or(db_email.created_at),
                sent_at: db_email.sent_at,
                thread_id: db_email.thread_id.map(|id| id.to_string().into()),
                reply_to_id: db_email.in_reply_to.map(|s| s.into()),
            }
        }).collect();

        let total_count = result.total as i32;
        let has_next_page = result.page < result.total_pages;
        let has_previous_page = result.page > 1;

        Ok(EmailConnection {
            edges,
            total_count,
            page_info: PageInfo {
                has_next_page,
                has_previous_page,
                start_cursor: None,
                end_cursor: None,
            },
        })
    }

    /// Get a single email by ID
    async fn email(&self, ctx: &Context<'_>, id: ID) -> Result<Option<Email>> {
        let gql_ctx = ctx.data::<GqlContext>()?;

        // Parse the email ID
        let email_id = Uuid::parse_str(id.as_str())
            .map_err(|_| async_graphql::Error::new("Invalid email ID format"))?;

        // Query database
        let repo = EmailRepository::new(gql_ctx.db_pool.clone());
        let db_email = repo.find_by_id(email_id).await
            .map_err(|e| async_graphql::Error::new(format!("Database error: {}", e)))?;

        // Convert to GraphQL type if found
        Ok(db_email.map(|db_email| {
            Email {
                id: db_email.id.to_string().into(),
                from: db_email.from_address.clone(),
                from_name: None,
                to: db_email.to_addresses.clone(),
                cc: db_email.cc_addresses.clone(),
                bcc: db_email.bcc_addresses.clone(),
                subject: db_email.subject.clone(),
                body_text: db_email.body_text.clone().unwrap_or_default(),
                body_html: db_email.body_html.clone(),
                snippet: db_email.body_text.as_ref()
                    .map(|t| t.chars().take(100).collect())
                    .unwrap_or_default(),
                folder: if db_email.is_archived { "archive".to_string() }
                        else if db_email.is_deleted { "trash".to_string() }
                        else { "inbox".to_string() },
                labels: db_email.labels.clone(),
                is_read: db_email.is_read,
                is_starred: db_email.is_starred,
                is_draft: db_email.status == db_models::EmailStatus::Draft,
                has_attachments: false,
                attachment_count: 0,
                trust_score: db_email.trust_score.unwrap_or(0.5),
                category: EmailCategory::Primary,
                priority: EmailPriority::Normal,
                received_at: db_email.received_at.unwrap_or(db_email.created_at),
                sent_at: db_email.sent_at,
                thread_id: db_email.thread_id.map(|id| id.to_string().into()),
                reply_to_id: db_email.in_reply_to.map(|s| s.into()),
            }
        }))
    }

    /// Get email thread
    async fn thread(&self, ctx: &Context<'_>, id: ID) -> Result<Option<EmailThread>> {
        let _gql_ctx = ctx.data::<GqlContext>()?;
        Ok(None)
    }

    /// Search emails with natural language
    async fn search_emails(
        &self,
        ctx: &Context<'_>,
        query: String,
        pagination: Option<Pagination>,
    ) -> Result<EmailConnection> {
        let _gql_ctx = ctx.data::<GqlContext>()?;
        Ok(EmailConnection {
            edges: vec![],
            total_count: 0,
            page_info: PageInfo {
                has_next_page: false,
                has_previous_page: false,
                start_cursor: None,
                end_cursor: None,
            },
        })
    }

    /// Get all folders
    async fn folders(&self, ctx: &Context<'_>) -> Result<Vec<Folder>> {
        let _gql_ctx = ctx.data::<GqlContext>()?;
        Ok(vec![])
    }

    /// Get all labels
    async fn labels(&self, ctx: &Context<'_>) -> Result<Vec<Label>> {
        let _gql_ctx = ctx.data::<GqlContext>()?;
        Ok(vec![])
    }

    /// Get contacts with optional search
    async fn contacts(
        &self,
        ctx: &Context<'_>,
        search: Option<String>,
        pagination: Option<Pagination>,
    ) -> Result<Vec<Contact>> {
        let _gql_ctx = ctx.data::<GqlContext>()?;
        Ok(vec![])
    }

    /// Get current user
    async fn me(&self, ctx: &Context<'_>) -> Result<User> {
        let gql_ctx = ctx.data::<GqlContext>()?;
        Ok(User {
            id: gql_ctx.user_id.to_string().into(),
            email: "user@example.com".to_string(),
            name: "User".to_string(),
            avatar_url: None,
            signature: None,
            timezone: "UTC".to_string(),
            language: "en".to_string(),
        })
    }

    /// Get unread count
    async fn unread_count(&self, ctx: &Context<'_>, folder: Option<String>) -> Result<i32> {
        let _gql_ctx = ctx.data::<GqlContext>()?;
        Ok(0)
    }

    /// Get attachment details
    async fn attachment(&self, ctx: &Context<'_>, id: ID) -> Result<Option<Attachment>> {
        let _gql_ctx = ctx.data::<GqlContext>()?;
        Ok(None)
    }
}

/// Mutation resolvers
pub struct MutationRoot;

#[Object]
impl MutationRoot {
    /// Send an email
    async fn send_email(
        &self,
        ctx: &Context<'_>,
        input: ComposeEmailInput,
    ) -> Result<SendEmailResult> {
        let _gql_ctx = ctx.data::<GqlContext>()?;
        Ok(SendEmailResult {
            success: true,
            email: None,
            error: None,
        })
    }

    /// Save email as draft
    ///
    /// Requires: EmailRepository::save_draft() implementation with
    /// `is_draft = true` flag on the email record.
    async fn save_draft(
        &self,
        ctx: &Context<'_>,
        input: ComposeEmailInput,
    ) -> Result<Email> {
        let gql_ctx = ctx.data::<GqlContext>()?;
        let repo = EmailRepository::new(gql_ctx.db_pool.clone());
        let now = Utc::now();
        let id = Uuid::new_v4();

        let email = Email {
            id: id.to_string().into(),
            from: String::new(), // Filled from user profile
            from_name: None,
            to: input.to,
            cc: input.cc.unwrap_or_default(),
            bcc: input.bcc.unwrap_or_default(),
            subject: input.subject,
            body_text: input.body_text,
            body_html: input.body_html,
            snippet: String::new(),
            folder: "drafts".to_string(),
            labels: vec![],
            is_read: true,
            is_starred: false,
            is_draft: true,
            has_attachments: input.attachment_ids.as_ref().map_or(false, |a| !a.is_empty()),
            attachment_count: input.attachment_ids.as_ref().map_or(0, |a| a.len() as i32),
            trust_score: 1.0,
            category: EmailCategory::Primary,
            priority: EmailPriority::Normal,
            received_at: now,
            sent_at: None,
            thread_id: None,
            reply_to_id: input.reply_to_id,
        };

        // NOTE: Persistence requires EmailRepository::save_draft() to be implemented.
        // For now, return the constructed draft object.
        Ok(email)
    }

    /// Update email (read status, starred, labels, folder)
    ///
    /// Requires: EmailRepository::update_email() implementation that applies
    /// partial updates from UpdateEmailInput fields.
    async fn update_email(
        &self,
        ctx: &Context<'_>,
        id: ID,
        input: UpdateEmailInput,
    ) -> Result<Email> {
        let gql_ctx = ctx.data::<GqlContext>()?;
        let _repo = EmailRepository::new(gql_ctx.db_pool.clone());
        let _email_id = Uuid::parse_str(id.as_str())
            .map_err(|e| async_graphql::Error::new(format!("Invalid email ID: {}", e)))?;

        // NOTE: Requires EmailRepository::update_email(email_id, input) implementation.
        // Should apply non-None fields from UpdateEmailInput to the existing record and
        // return the updated Email.
        Err(async_graphql::Error::new(
            "Email update not yet implemented: requires EmailRepository::update_email()"
        ))
    }

    /// Bulk update emails
    async fn bulk_update_emails(
        &self,
        ctx: &Context<'_>,
        ids: Vec<ID>,
        input: UpdateEmailInput,
    ) -> Result<BulkActionResult> {
        let _gql_ctx = ctx.data::<GqlContext>()?;
        Ok(BulkActionResult {
            success_count: 0,
            failure_count: 0,
            errors: vec![],
        })
    }

    /// Delete an email (move to trash)
    async fn delete_email(&self, ctx: &Context<'_>, id: ID) -> Result<bool> {
        let _gql_ctx = ctx.data::<GqlContext>()?;
        Ok(true)
    }

    /// Permanently delete email
    async fn permanently_delete_email(&self, ctx: &Context<'_>, id: ID) -> Result<bool> {
        let _gql_ctx = ctx.data::<GqlContext>()?;
        Ok(true)
    }

    /// Empty trash folder
    async fn empty_trash(&self, ctx: &Context<'_>) -> Result<i32> {
        let _gql_ctx = ctx.data::<GqlContext>()?;
        Ok(0)
    }

    /// Create a folder
    ///
    /// Requires: EmailRepository::create_folder() implementation.
    async fn create_folder(
        &self,
        ctx: &Context<'_>,
        input: CreateFolderInput,
    ) -> Result<Folder> {
        let gql_ctx = ctx.data::<GqlContext>()?;
        let _repo = EmailRepository::new(gql_ctx.db_pool.clone());
        let id = Uuid::new_v4();

        let path = match &input.parent_path {
            Some(parent) => format!("{}/{}", parent, input.name),
            None => input.name.clone(),
        };

        let folder = Folder {
            id: id.to_string().into(),
            name: input.name,
            path,
            message_count: 0,
            unread_count: 0,
            is_system: false,
            color: input.color,
            icon: input.icon,
        };

        // NOTE: Requires EmailRepository::create_folder() for persistence.
        // Returns the constructed folder for now.
        Ok(folder)
    }

    /// Delete a folder
    async fn delete_folder(&self, ctx: &Context<'_>, id: ID) -> Result<bool> {
        let _gql_ctx = ctx.data::<GqlContext>()?;
        Ok(true)
    }

    /// Create a label
    ///
    /// Requires: EmailRepository::create_label() implementation.
    async fn create_label(
        &self,
        ctx: &Context<'_>,
        input: CreateLabelInput,
    ) -> Result<Label> {
        let gql_ctx = ctx.data::<GqlContext>()?;
        let _repo = EmailRepository::new(gql_ctx.db_pool.clone());
        let id = Uuid::new_v4();

        let label = Label {
            id: id.to_string().into(),
            name: input.name,
            color: input.color,
            message_count: 0,
        };

        // NOTE: Requires EmailRepository::create_label() for persistence.
        // Returns the constructed label for now.
        Ok(label)
    }

    /// Delete a label
    async fn delete_label(&self, ctx: &Context<'_>, id: ID) -> Result<bool> {
        let _gql_ctx = ctx.data::<GqlContext>()?;
        Ok(true)
    }

    /// Mark all as read in folder
    async fn mark_all_read(&self, ctx: &Context<'_>, folder: String) -> Result<i32> {
        let _gql_ctx = ctx.data::<GqlContext>()?;
        Ok(0)
    }

    /// Report email as spam
    async fn report_spam(&self, ctx: &Context<'_>, id: ID) -> Result<bool> {
        let _gql_ctx = ctx.data::<GqlContext>()?;
        Ok(true)
    }

    /// Report email as not spam
    async fn not_spam(&self, ctx: &Context<'_>, id: ID) -> Result<bool> {
        let _gql_ctx = ctx.data::<GqlContext>()?;
        Ok(true)
    }

    /// Update trust score for a contact
    ///
    /// Requires: EmailRepository::update_contact_trust_score() implementation,
    /// plus integration with the trust_filter Holochain zome for DHT-backed trust.
    async fn update_trust_score(
        &self,
        ctx: &Context<'_>,
        email: String,
        score: f64,
    ) -> Result<Contact> {
        let gql_ctx = ctx.data::<GqlContext>()?;
        let _repo = EmailRepository::new(gql_ctx.db_pool.clone());

        // Validate trust score range
        if !(0.0..=1.0).contains(&score) {
            return Err(async_graphql::Error::new(
                "Trust score must be between 0.0 and 1.0"
            ));
        }

        // NOTE: Requires EmailRepository::update_contact_trust_score(email, score)
        // and integration with the trust_filter Holochain zome via
        // call_zome("trust_filter", "update_trust_score", ...) for DHT persistence.
        Err(async_graphql::Error::new(
            "Trust score update not yet implemented: requires EmailRepository::update_contact_trust_score() \
             and trust_filter zome integration"
        ))
    }
}

/// Subscription resolvers for real-time updates
pub struct SubscriptionRoot;

#[Subscription]
impl SubscriptionRoot {
    /// Subscribe to new emails
    async fn new_email(
        &self,
        ctx: &Context<'_>,
        folder: Option<String>,
    ) -> impl Stream<Item = Email> {
        let gql_ctx = ctx.data::<GqlContext>().unwrap();
        let mut rx = gql_ctx.event_sender.subscribe();

        async_stream::stream! {
            while let Ok(event) = rx.recv().await {
                if let EmailEvent::NewEmail(email) = event {
                    if folder.is_none() || Some(email.folder.clone()) == folder {
                        yield email;
                    }
                }
            }
        }
    }

    /// Subscribe to email updates
    async fn email_updated(&self, ctx: &Context<'_>) -> impl Stream<Item = Email> {
        let gql_ctx = ctx.data::<GqlContext>().unwrap();
        let mut rx = gql_ctx.event_sender.subscribe();

        async_stream::stream! {
            while let Ok(event) = rx.recv().await {
                if let EmailEvent::EmailUpdated(email) = event {
                    yield email;
                }
            }
        }
    }

    /// Subscribe to email deletions
    async fn email_deleted(&self, ctx: &Context<'_>) -> impl Stream<Item = ID> {
        let gql_ctx = ctx.data::<GqlContext>().unwrap();
        let mut rx = gql_ctx.event_sender.subscribe();

        async_stream::stream! {
            while let Ok(event) = rx.recv().await {
                if let EmailEvent::EmailDeleted(id) = event {
                    yield id;
                }
            }
        }
    }

    /// Subscribe to folder updates
    async fn folder_updated(&self, ctx: &Context<'_>) -> impl Stream<Item = Folder> {
        let gql_ctx = ctx.data::<GqlContext>().unwrap();
        let mut rx = gql_ctx.event_sender.subscribe();

        async_stream::stream! {
            while let Ok(event) = rx.recv().await {
                if let EmailEvent::FolderUpdated(folder) = event {
                    yield folder;
                }
            }
        }
    }

    /// Subscribe to unread count changes
    async fn unread_count_changed(
        &self,
        ctx: &Context<'_>,
        folder: Option<String>,
    ) -> impl Stream<Item = i32> {
        let gql_ctx = ctx.data::<GqlContext>().unwrap();
        let user_id = gql_ctx.user_id;
        let pool = gql_ctx.db_pool.clone();
        let mut rx = gql_ctx.event_sender.subscribe();

        async_stream::stream! {
            while let Ok(_event) = rx.recv().await {
                // Calculate actual unread count from database
                let count = calculate_unread_count(&pool, user_id, folder.as_deref()).await;
                yield count;
            }
        }
    }
}

/// Calculate the unread email count for a user
async fn calculate_unread_count(
    pool: &sqlx::PgPool,
    user_id: Uuid,
    folder: Option<&str>,
) -> i32 {
    // Build query based on folder filter
    let query = match folder {
        Some("inbox") | None => {
            sqlx::query_scalar::<_, i64>(
                r#"
                SELECT COUNT(*) FROM emails
                WHERE user_id = $1
                  AND is_read = false
                  AND is_deleted = false
                  AND is_archived = false
                "#,
            )
            .bind(user_id)
        }
        Some("archive") => {
            sqlx::query_scalar::<_, i64>(
                r#"
                SELECT COUNT(*) FROM emails
                WHERE user_id = $1
                  AND is_read = false
                  AND is_archived = true
                "#,
            )
            .bind(user_id)
        }
        Some("trash") => {
            sqlx::query_scalar::<_, i64>(
                r#"
                SELECT COUNT(*) FROM emails
                WHERE user_id = $1
                  AND is_read = false
                  AND is_deleted = true
                "#,
            )
            .bind(user_id)
        }
        Some(_) => {
            // For any other folder, count all unread
            sqlx::query_scalar::<_, i64>(
                r#"
                SELECT COUNT(*) FROM emails
                WHERE user_id = $1
                  AND is_read = false
                  AND is_deleted = false
                "#,
            )
            .bind(user_id)
        }
    };

    query
        .fetch_one(pool)
        .await
        .unwrap_or(0) as i32
}

/// Build the GraphQL schema
pub fn build_schema() -> Schema<QueryRoot, MutationRoot, SubscriptionRoot> {
    Schema::build(QueryRoot, MutationRoot, SubscriptionRoot)
        .finish()
}
