// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Advanced Privacy & Control Module
//!
//! Provides email expiration, message recall, tracking pixel detection,
//! read receipt control, anonymous mode, and link proxying.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use url::Url;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum PrivacyError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("Email not found: {0}")]
    EmailNotFound(Uuid),
    #[error("Cannot recall email: {0}")]
    RecallFailed(String),
    #[error("Feature not available: {0}")]
    FeatureUnavailable(String),
    #[error("Invalid URL: {0}")]
    InvalidUrl(String),
}

// ============================================================================
// Email Expiration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpiringEmail {
    pub email_id: Uuid,
    pub expires_at: DateTime<Utc>,
    pub expiration_type: ExpirationType,
    pub created_at: DateTime<Utc>,
    pub notified_recipient: bool,
    pub view_count: i32,
    pub max_views: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpirationType {
    TimeBased { hours: i32 },
    ViewBased { max_views: i32 },
    OnRead,
    Custom { condition: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetExpirationInput {
    pub expiration_type: ExpirationType,
    pub notify_recipient: bool,
    pub delete_attachments: bool,
    pub leave_placeholder: bool,
}

pub struct ExpirationService {
    pool: PgPool,
}

impl ExpirationService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn set_expiration(
        &self,
        email_id: Uuid,
        user_id: Uuid,
        input: SetExpirationInput,
    ) -> Result<ExpiringEmail, PrivacyError> {
        // Verify email ownership
        let email = sqlx::query!(
            "SELECT id FROM emails WHERE id = $1 AND sender_id = $2",
            email_id,
            user_id
        )
        .fetch_optional(&self.pool)
        .await?
        .ok_or(PrivacyError::EmailNotFound(email_id))?;

        let expires_at = match &input.expiration_type {
            ExpirationType::TimeBased { hours } => {
                Utc::now() + Duration::hours(*hours as i64)
            }
            ExpirationType::ViewBased { .. } => {
                // Set far future, will expire on view count
                Utc::now() + Duration::days(365)
            }
            ExpirationType::OnRead => {
                Utc::now() + Duration::days(365)
            }
            ExpirationType::Custom { .. } => {
                Utc::now() + Duration::days(30)
            }
        };

        let max_views = match &input.expiration_type {
            ExpirationType::ViewBased { max_views } => Some(*max_views),
            ExpirationType::OnRead => Some(1),
            _ => None,
        };

        sqlx::query!(
            r#"
            INSERT INTO expiring_emails (email_id, expires_at, expiration_type,
                                        max_views, notify_recipient, delete_attachments,
                                        leave_placeholder, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            ON CONFLICT (email_id) DO UPDATE SET
                expires_at = EXCLUDED.expires_at,
                expiration_type = EXCLUDED.expiration_type,
                max_views = EXCLUDED.max_views
            "#,
            email_id,
            expires_at,
            serde_json::to_string(&input.expiration_type).unwrap(),
            max_views,
            input.notify_recipient,
            input.delete_attachments,
            input.leave_placeholder
        )
        .execute(&self.pool)
        .await?;

        self.get_expiration(email_id).await
    }

    pub async fn get_expiration(&self, email_id: Uuid) -> Result<ExpiringEmail, PrivacyError> {
        let row = sqlx::query!(
            r#"
            SELECT email_id, expires_at, expiration_type, created_at,
                   notified_recipient, view_count, max_views
            FROM expiring_emails
            WHERE email_id = $1
            "#,
            email_id
        )
        .fetch_optional(&self.pool)
        .await?
        .ok_or(PrivacyError::EmailNotFound(email_id))?;

        Ok(ExpiringEmail {
            email_id: row.email_id,
            expires_at: row.expires_at,
            expiration_type: serde_json::from_str(&row.expiration_type)
                .unwrap_or(ExpirationType::TimeBased { hours: 24 }),
            created_at: row.created_at,
            notified_recipient: row.notified_recipient,
            view_count: row.view_count,
            max_views: row.max_views,
        })
    }

    pub async fn record_view(&self, email_id: Uuid) -> Result<bool, PrivacyError> {
        let expiring = sqlx::query!(
            r#"
            UPDATE expiring_emails
            SET view_count = view_count + 1
            WHERE email_id = $1
            RETURNING view_count, max_views, expiration_type
            "#,
            email_id
        )
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = expiring {
            let should_expire = match row.max_views {
                Some(max) => row.view_count >= max,
                None => false,
            };

            if should_expire {
                self.expire_email(email_id).await?;
                return Ok(true);
            }
        }

        Ok(false)
    }

    pub async fn expire_email(&self, email_id: Uuid) -> Result<(), PrivacyError> {
        let config = sqlx::query!(
            r#"
            SELECT delete_attachments, leave_placeholder
            FROM expiring_emails
            WHERE email_id = $1
            "#,
            email_id
        )
        .fetch_optional(&self.pool)
        .await?;

        let (delete_attachments, leave_placeholder) = config
            .map(|c| (c.delete_attachments, c.leave_placeholder))
            .unwrap_or((true, true));

        if delete_attachments {
            sqlx::query!(
                "DELETE FROM attachments WHERE email_id = $1",
                email_id
            )
            .execute(&self.pool)
            .await?;
        }

        if leave_placeholder {
            sqlx::query!(
                r#"
                UPDATE emails
                SET body_text = '[This message has expired]',
                    body_html = '<p style="color: gray; font-style: italic;">[This message has expired]</p>',
                    is_expired = true
                WHERE id = $1
                "#,
                email_id
            )
            .execute(&self.pool)
            .await?;
        } else {
            sqlx::query!("DELETE FROM emails WHERE id = $1", email_id)
                .execute(&self.pool)
                .await?;
        }

        Ok(())
    }

    pub async fn process_expired_emails(&self) -> Result<i32, PrivacyError> {
        let expired = sqlx::query_scalar!(
            r#"
            SELECT email_id
            FROM expiring_emails
            WHERE expires_at <= NOW() AND NOT processed
            "#
        )
        .fetch_all(&self.pool)
        .await?;

        let count = expired.len() as i32;

        for email_id in expired {
            self.expire_email(email_id).await?;
            sqlx::query!(
                "UPDATE expiring_emails SET processed = true WHERE email_id = $1",
                email_id
            )
            .execute(&self.pool)
            .await?;
        }

        Ok(count)
    }

    pub async fn cancel_expiration(&self, email_id: Uuid, user_id: Uuid) -> Result<(), PrivacyError> {
        let result = sqlx::query!(
            r#"
            DELETE FROM expiring_emails
            WHERE email_id = $1
            AND email_id IN (SELECT id FROM emails WHERE sender_id = $2)
            "#,
            email_id,
            user_id
        )
        .execute(&self.pool)
        .await?;

        if result.rows_affected() == 0 {
            return Err(PrivacyError::EmailNotFound(email_id));
        }

        Ok(())
    }
}

// ============================================================================
// Message Recall
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallRequest {
    pub id: Uuid,
    pub email_id: Uuid,
    pub requested_by: Uuid,
    pub requested_at: DateTime<Utc>,
    pub status: RecallStatus,
    pub recipient_statuses: Vec<RecipientRecallStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecallStatus {
    Pending,
    PartialSuccess,
    Success,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecipientRecallStatus {
    pub recipient_email: String,
    pub status: SingleRecallStatus,
    pub processed_at: Option<DateTime<Utc>>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SingleRecallStatus {
    Pending,
    Recalled,
    AlreadyRead,
    NotSupported,
    Failed,
}

pub struct RecallService {
    pool: PgPool,
}

impl RecallService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn can_recall(&self, email_id: Uuid, user_id: Uuid) -> Result<RecallEligibility, PrivacyError> {
        let email = sqlx::query!(
            r#"
            SELECT id, sent_at, recipients
            FROM emails
            WHERE id = $1 AND sender_id = $2 AND sent_at IS NOT NULL
            "#,
            email_id,
            user_id
        )
        .fetch_optional(&self.pool)
        .await?
        .ok_or(PrivacyError::EmailNotFound(email_id))?;

        let sent_at = email.sent_at.unwrap();
        let age_hours = (Utc::now() - sent_at).num_hours();

        // Check if within recall window (typically 1 hour for most providers)
        if age_hours > 1 {
            return Ok(RecallEligibility {
                can_recall: false,
                reason: Some("Email was sent more than 1 hour ago".to_string()),
                supported_recipients: vec![],
                unsupported_recipients: vec![],
            });
        }

        // Check recipient support (internal recipients are recallable)
        let recipients: Vec<String> = serde_json::from_value(email.recipients).unwrap_or_default();

        let mut supported = Vec::new();
        let mut unsupported = Vec::new();

        for recipient in recipients {
            if self.supports_recall(&recipient).await {
                supported.push(recipient);
            } else {
                unsupported.push(recipient);
            }
        }

        Ok(RecallEligibility {
            can_recall: !supported.is_empty(),
            reason: if supported.is_empty() {
                Some("No recipients support message recall".to_string())
            } else {
                None
            },
            supported_recipients: supported,
            unsupported_recipients: unsupported,
        })
    }

    async fn supports_recall(&self, email: &str) -> bool {
        // Check if recipient is internal or uses a supporting provider
        let domain = email.split('@').last().unwrap_or("");

        // Internal emails are always recallable
        let is_internal = sqlx::query_scalar!(
            "SELECT EXISTS(SELECT 1 FROM users WHERE email = $1)",
            email
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(Some(false))
        .unwrap_or(false);

        is_internal
    }

    pub async fn request_recall(&self, email_id: Uuid, user_id: Uuid) -> Result<RecallRequest, PrivacyError> {
        let eligibility = self.can_recall(email_id, user_id).await?;

        if !eligibility.can_recall {
            return Err(PrivacyError::RecallFailed(
                eligibility.reason.unwrap_or("Cannot recall this email".to_string()),
            ));
        }

        let request_id = Uuid::new_v4();

        sqlx::query!(
            r#"
            INSERT INTO recall_requests (id, email_id, requested_by, status, requested_at)
            VALUES ($1, $2, $3, $4, NOW())
            "#,
            request_id,
            email_id,
            user_id,
            serde_json::to_string(&RecallStatus::Pending).unwrap()
        )
        .execute(&self.pool)
        .await?;

        // Process supported recipients
        for recipient in &eligibility.supported_recipients {
            self.process_recall_for_recipient(request_id, email_id, recipient).await?;
        }

        // Mark unsupported recipients
        for recipient in &eligibility.unsupported_recipients {
            sqlx::query!(
                r#"
                INSERT INTO recall_recipient_status (request_id, recipient_email, status, reason)
                VALUES ($1, $2, $3, 'External recipient does not support recall')
                "#,
                request_id,
                recipient,
                serde_json::to_string(&SingleRecallStatus::NotSupported).unwrap()
            )
            .execute(&self.pool)
            .await?;
        }

        self.get_recall_request(request_id).await
    }

    async fn process_recall_for_recipient(
        &self,
        request_id: Uuid,
        email_id: Uuid,
        recipient: &str,
    ) -> Result<(), PrivacyError> {
        // Check if recipient has read the email
        let was_read = sqlx::query_scalar!(
            r#"
            SELECT is_read
            FROM email_recipients
            WHERE email_id = $1 AND recipient_email = $2
            "#,
            email_id,
            recipient
        )
        .fetch_optional(&self.pool)
        .await?
        .flatten()
        .unwrap_or(false);

        let status = if was_read {
            SingleRecallStatus::AlreadyRead
        } else {
            // Delete from recipient's inbox
            sqlx::query!(
                r#"
                UPDATE email_recipients
                SET is_deleted = true, deleted_at = NOW(), deleted_reason = 'recalled'
                WHERE email_id = $1 AND recipient_email = $2
                "#,
                email_id,
                recipient
            )
            .execute(&self.pool)
            .await?;

            SingleRecallStatus::Recalled
        };

        sqlx::query!(
            r#"
            INSERT INTO recall_recipient_status (request_id, recipient_email, status, processed_at)
            VALUES ($1, $2, $3, NOW())
            "#,
            request_id,
            recipient,
            serde_json::to_string(&status).unwrap()
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_recall_request(&self, request_id: Uuid) -> Result<RecallRequest, PrivacyError> {
        let request = sqlx::query!(
            r#"
            SELECT id, email_id, requested_by, requested_at, status
            FROM recall_requests
            WHERE id = $1
            "#,
            request_id
        )
        .fetch_one(&self.pool)
        .await?;

        let recipient_statuses = sqlx::query!(
            r#"
            SELECT recipient_email, status, processed_at, reason
            FROM recall_recipient_status
            WHERE request_id = $1
            "#,
            request_id
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|row| RecipientRecallStatus {
            recipient_email: row.recipient_email,
            status: serde_json::from_str(&row.status).unwrap_or(SingleRecallStatus::Pending),
            processed_at: row.processed_at,
            reason: row.reason,
        })
        .collect();

        Ok(RecallRequest {
            id: request.id,
            email_id: request.email_id,
            requested_by: request.requested_by,
            requested_at: request.requested_at,
            status: serde_json::from_str(&request.status).unwrap_or(RecallStatus::Pending),
            recipient_statuses,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallEligibility {
    pub can_recall: bool,
    pub reason: Option<String>,
    pub supported_recipients: Vec<String>,
    pub unsupported_recipients: Vec<String>,
}

// ============================================================================
// Tracking Pixel Detection
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingDetectionResult {
    pub email_id: Uuid,
    pub tracking_pixels: Vec<DetectedTracker>,
    pub tracking_links: Vec<DetectedTracker>,
    pub total_trackers: i32,
    pub risk_level: RiskLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedTracker {
    pub tracker_type: TrackerType,
    pub url: String,
    pub domain: String,
    pub company: Option<String>,
    pub purpose: String,
    pub blocked: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrackerType {
    Pixel,
    Link,
    Beacon,
    WebBug,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    None,
    Low,
    Medium,
    High,
}

pub struct TrackingDetectionService {
    pool: PgPool,
    known_trackers: Vec<TrackerSignature>,
}

#[derive(Debug, Clone)]
struct TrackerSignature {
    domain_pattern: String,
    company: String,
    tracker_type: TrackerType,
}

impl TrackingDetectionService {
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            known_trackers: Self::load_tracker_signatures(),
        }
    }

    fn load_tracker_signatures() -> Vec<TrackerSignature> {
        vec![
            TrackerSignature {
                domain_pattern: "mailchimp.com".to_string(),
                company: "Mailchimp".to_string(),
                tracker_type: TrackerType::Pixel,
            },
            TrackerSignature {
                domain_pattern: "sendgrid.net".to_string(),
                company: "SendGrid".to_string(),
                tracker_type: TrackerType::Pixel,
            },
            TrackerSignature {
                domain_pattern: "hubspot.com".to_string(),
                company: "HubSpot".to_string(),
                tracker_type: TrackerType::Pixel,
            },
            TrackerSignature {
                domain_pattern: "mailtrack.io".to_string(),
                company: "Mailtrack".to_string(),
                tracker_type: TrackerType::Pixel,
            },
            TrackerSignature {
                domain_pattern: "getnotify.com".to_string(),
                company: "GetNotify".to_string(),
                tracker_type: TrackerType::Pixel,
            },
            TrackerSignature {
                domain_pattern: "yesware.com".to_string(),
                company: "Yesware".to_string(),
                tracker_type: TrackerType::Pixel,
            },
            TrackerSignature {
                domain_pattern: "streak.com".to_string(),
                company: "Streak".to_string(),
                tracker_type: TrackerType::Pixel,
            },
            TrackerSignature {
                domain_pattern: "doubleclick.net".to_string(),
                company: "Google".to_string(),
                tracker_type: TrackerType::Link,
            },
            TrackerSignature {
                domain_pattern: "click.".to_string(),
                company: "Unknown".to_string(),
                tracker_type: TrackerType::Link,
            },
            TrackerSignature {
                domain_pattern: "track.".to_string(),
                company: "Unknown".to_string(),
                tracker_type: TrackerType::Link,
            },
        ]
    }

    pub async fn scan_email(&self, email_id: Uuid) -> Result<TrackingDetectionResult, PrivacyError> {
        let email = sqlx::query!(
            "SELECT body_html FROM emails WHERE id = $1",
            email_id
        )
        .fetch_optional(&self.pool)
        .await?
        .ok_or(PrivacyError::EmailNotFound(email_id))?;

        let html = email.body_html.unwrap_or_default();
        let mut detected_trackers = Vec::new();

        // Detect tracking pixels (1x1 images, hidden images)
        let pixel_pattern = regex::Regex::new(
            r#"<img[^>]*(?:width\s*=\s*["']?1["']?|height\s*=\s*["']?1["']?|style\s*=\s*["'][^"']*display\s*:\s*none)[^>]*src\s*=\s*["']([^"']+)["']"#
        ).unwrap();

        for cap in pixel_pattern.captures_iter(&html) {
            if let Some(url) = cap.get(1) {
                if let Some(tracker) = self.identify_tracker(url.as_str(), TrackerType::Pixel) {
                    detected_trackers.push(tracker);
                }
            }
        }

        // Detect tracking links
        let link_pattern = regex::Regex::new(r#"href\s*=\s*["']([^"']+)["']"#).unwrap();

        for cap in link_pattern.captures_iter(&html) {
            if let Some(url) = cap.get(1) {
                if let Some(tracker) = self.identify_tracker(url.as_str(), TrackerType::Link) {
                    detected_trackers.push(tracker);
                }
            }
        }

        let total = detected_trackers.len() as i32;
        let risk_level = match total {
            0 => RiskLevel::None,
            1..=2 => RiskLevel::Low,
            3..=5 => RiskLevel::Medium,
            _ => RiskLevel::High,
        };

        // Store detection results
        sqlx::query!(
            r#"
            INSERT INTO tracking_detections (email_id, trackers, risk_level, scanned_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (email_id) DO UPDATE SET
                trackers = EXCLUDED.trackers,
                risk_level = EXCLUDED.risk_level,
                scanned_at = NOW()
            "#,
            email_id,
            serde_json::to_value(&detected_trackers).unwrap(),
            serde_json::to_string(&risk_level).unwrap()
        )
        .execute(&self.pool)
        .await?;

        let tracking_pixels: Vec<_> = detected_trackers
            .iter()
            .filter(|t| matches!(t.tracker_type, TrackerType::Pixel))
            .cloned()
            .collect();

        let tracking_links: Vec<_> = detected_trackers
            .iter()
            .filter(|t| matches!(t.tracker_type, TrackerType::Link))
            .cloned()
            .collect();

        Ok(TrackingDetectionResult {
            email_id,
            tracking_pixels,
            tracking_links,
            total_trackers: total,
            risk_level,
        })
    }

    fn identify_tracker(&self, url: &str, expected_type: TrackerType) -> Option<DetectedTracker> {
        let parsed = Url::parse(url).ok()?;
        let domain = parsed.host_str()?.to_lowercase();

        for sig in &self.known_trackers {
            if domain.contains(&sig.domain_pattern) {
                return Some(DetectedTracker {
                    tracker_type: sig.tracker_type.clone(),
                    url: url.to_string(),
                    domain: domain.clone(),
                    company: Some(sig.company.clone()),
                    purpose: "Email open/click tracking".to_string(),
                    blocked: false,
                });
            }
        }

        // Heuristic detection
        if domain.starts_with("track.") || domain.starts_with("click.") || domain.contains("pixel") {
            return Some(DetectedTracker {
                tracker_type: expected_type,
                url: url.to_string(),
                domain,
                company: None,
                purpose: "Suspected tracking".to_string(),
                blocked: false,
            });
        }

        None
    }

    pub async fn strip_trackers(&self, email_id: Uuid) -> Result<String, PrivacyError> {
        let email = sqlx::query!(
            "SELECT body_html FROM emails WHERE id = $1",
            email_id
        )
        .fetch_one(&self.pool)
        .await?;

        let mut html = email.body_html.unwrap_or_default();

        // Remove tracking pixels
        let pixel_pattern = regex::Regex::new(
            r#"<img[^>]*(?:width\s*=\s*["']?1["']?|height\s*=\s*["']?1["']?)[^>]*/?\s*>"#
        ).unwrap();
        html = pixel_pattern.replace_all(&html, "").to_string();

        // Proxy tracking links through our proxy
        // (In production, rewrite URLs to go through privacy proxy)

        Ok(html)
    }
}

// ============================================================================
// Read Receipt Control
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadReceiptSettings {
    pub block_outgoing: bool,
    pub block_incoming_requests: bool,
    pub auto_deny_mdn: bool,
    pub whitelist: Vec<String>,
    pub blacklist: Vec<String>,
}

pub struct ReadReceiptService {
    pool: PgPool,
}

impl ReadReceiptService {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn get_settings(&self, user_id: Uuid) -> Result<ReadReceiptSettings, PrivacyError> {
        let settings = sqlx::query!(
            r#"
            SELECT block_outgoing, block_incoming_requests, auto_deny_mdn,
                   whitelist as "whitelist: Vec<String>", blacklist as "blacklist: Vec<String>"
            FROM read_receipt_settings
            WHERE user_id = $1
            "#,
            user_id
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(settings
            .map(|s| ReadReceiptSettings {
                block_outgoing: s.block_outgoing,
                block_incoming_requests: s.block_incoming_requests,
                auto_deny_mdn: s.auto_deny_mdn,
                whitelist: s.whitelist,
                blacklist: s.blacklist,
            })
            .unwrap_or(ReadReceiptSettings {
                block_outgoing: false,
                block_incoming_requests: true,
                auto_deny_mdn: true,
                whitelist: vec![],
                blacklist: vec![],
            }))
    }

    pub async fn update_settings(
        &self,
        user_id: Uuid,
        settings: ReadReceiptSettings,
    ) -> Result<(), PrivacyError> {
        sqlx::query!(
            r#"
            INSERT INTO read_receipt_settings (user_id, block_outgoing, block_incoming_requests,
                                              auto_deny_mdn, whitelist, blacklist)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (user_id) DO UPDATE SET
                block_outgoing = EXCLUDED.block_outgoing,
                block_incoming_requests = EXCLUDED.block_incoming_requests,
                auto_deny_mdn = EXCLUDED.auto_deny_mdn,
                whitelist = EXCLUDED.whitelist,
                blacklist = EXCLUDED.blacklist
            "#,
            user_id,
            settings.block_outgoing,
            settings.block_incoming_requests,
            settings.auto_deny_mdn,
            &settings.whitelist,
            &settings.blacklist
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn should_send_receipt(
        &self,
        user_id: Uuid,
        sender_email: &str,
    ) -> Result<bool, PrivacyError> {
        let settings = self.get_settings(user_id).await?;

        if settings.auto_deny_mdn {
            return Ok(false);
        }

        // Check whitelist
        if !settings.whitelist.is_empty() {
            let whitelisted = settings.whitelist.iter().any(|w| {
                sender_email.ends_with(w) || sender_email == w
            });
            if !whitelisted {
                return Ok(false);
            }
        }

        // Check blacklist
        let blacklisted = settings.blacklist.iter().any(|b| {
            sender_email.ends_with(b) || sender_email == b
        });

        Ok(!blacklisted)
    }

    pub async fn strip_mdn_request(&self, headers: &mut Vec<(String, String)>) {
        headers.retain(|(k, _)| {
            !k.eq_ignore_ascii_case("Disposition-Notification-To") &&
            !k.eq_ignore_ascii_case("Return-Receipt-To") &&
            !k.eq_ignore_ascii_case("X-Confirm-Reading-To")
        });
    }
}

// ============================================================================
// Anonymous Mode
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnonymousIdentity {
    pub id: Uuid,
    pub user_id: Uuid,
    pub alias_email: String,
    pub display_name: Option<String>,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub email_count: i32,
    pub is_active: bool,
}

pub struct AnonymousService {
    pool: PgPool,
    relay_domain: String,
}

impl AnonymousService {
    pub fn new(pool: PgPool, relay_domain: String) -> Self {
        Self { pool, relay_domain }
    }

    pub async fn create_identity(
        &self,
        user_id: Uuid,
        display_name: Option<&str>,
        expires_in_days: Option<i32>,
    ) -> Result<AnonymousIdentity, PrivacyError> {
        let alias = generate_anonymous_alias();
        let alias_email = format!("{}@{}", alias, self.relay_domain);

        let expires_at = expires_in_days.map(|d| Utc::now() + Duration::days(d as i64));

        let identity = sqlx::query_as!(
            AnonymousIdentity,
            r#"
            INSERT INTO anonymous_identities (user_id, alias_email, display_name,
                                             expires_at, created_at, is_active)
            VALUES ($1, $2, $3, $4, NOW(), true)
            RETURNING id, user_id, alias_email, display_name, created_at,
                      expires_at, email_count, is_active
            "#,
            user_id,
            alias_email,
            display_name,
            expires_at
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(identity)
    }

    pub async fn get_identities(&self, user_id: Uuid) -> Result<Vec<AnonymousIdentity>, PrivacyError> {
        let identities = sqlx::query_as!(
            AnonymousIdentity,
            r#"
            SELECT id, user_id, alias_email, display_name, created_at,
                   expires_at, email_count, is_active
            FROM anonymous_identities
            WHERE user_id = $1
            ORDER BY created_at DESC
            "#,
            user_id
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(identities)
    }

    pub async fn deactivate_identity(&self, identity_id: Uuid, user_id: Uuid) -> Result<(), PrivacyError> {
        sqlx::query!(
            r#"
            UPDATE anonymous_identities
            SET is_active = false
            WHERE id = $1 AND user_id = $2
            "#,
            identity_id,
            user_id
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn route_incoming(
        &self,
        alias_email: &str,
    ) -> Result<Option<Uuid>, PrivacyError> {
        let user_id = sqlx::query_scalar!(
            r#"
            SELECT user_id
            FROM anonymous_identities
            WHERE alias_email = $1 AND is_active = true
            AND (expires_at IS NULL OR expires_at > NOW())
            "#,
            alias_email
        )
        .fetch_optional(&self.pool)
        .await?
        .flatten();

        Ok(user_id)
    }

    pub async fn send_anonymously(
        &self,
        identity_id: Uuid,
        user_id: Uuid,
        recipient: &str,
        subject: &str,
        body: &str,
    ) -> Result<Uuid, PrivacyError> {
        let identity = sqlx::query!(
            r#"
            SELECT alias_email, display_name, is_active, expires_at
            FROM anonymous_identities
            WHERE id = $1 AND user_id = $2
            "#,
            identity_id,
            user_id
        )
        .fetch_optional(&self.pool)
        .await?
        .ok_or(PrivacyError::FeatureUnavailable(
            "Anonymous identity not found".to_string(),
        ))?;

        if !identity.is_active {
            return Err(PrivacyError::FeatureUnavailable(
                "Identity is deactivated".to_string(),
            ));
        }

        if let Some(expires) = identity.expires_at {
            if expires < Utc::now() {
                return Err(PrivacyError::FeatureUnavailable(
                    "Identity has expired".to_string(),
                ));
            }
        }

        // Create email with anonymous sender
        let email_id = Uuid::new_v4();
        let sender_name = identity.display_name.unwrap_or_else(|| "Anonymous".to_string());

        sqlx::query!(
            r#"
            INSERT INTO emails (id, sender_id, sender_email, sender_name, recipient_email,
                               subject, body_text, is_anonymous, anonymous_identity_id, sent_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, true, $8, NOW())
            "#,
            email_id,
            user_id,
            identity.alias_email,
            sender_name,
            recipient,
            subject,
            body,
            identity_id
        )
        .execute(&self.pool)
        .await?;

        // Increment counter
        sqlx::query!(
            "UPDATE anonymous_identities SET email_count = email_count + 1 WHERE id = $1",
            identity_id
        )
        .execute(&self.pool)
        .await?;

        Ok(email_id)
    }
}

fn generate_anonymous_alias() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Generate a random alias like "anon_x7k9m2"
    let chars: String = (0..6)
        .map(|_| {
            let idx = rng.gen_range(0..36);
            if idx < 10 {
                (b'0' + idx) as char
            } else {
                (b'a' + idx - 10) as char
            }
        })
        .collect();

    format!("anon_{}", chars)
}

// ============================================================================
// Link Proxying
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxiedLink {
    pub id: Uuid,
    pub original_url: String,
    pub proxy_url: String,
    pub email_id: Uuid,
    pub created_at: DateTime<Utc>,
    pub click_count: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkProxySettings {
    pub enabled: bool,
    pub strip_utm_params: bool,
    pub block_known_trackers: bool,
    pub warn_on_redirect: bool,
    pub whitelist_domains: Vec<String>,
}

pub struct LinkProxyService {
    pool: PgPool,
    proxy_base_url: String,
}

impl LinkProxyService {
    pub fn new(pool: PgPool, proxy_base_url: String) -> Self {
        Self { pool, proxy_base_url }
    }

    pub async fn get_settings(&self, user_id: Uuid) -> Result<LinkProxySettings, PrivacyError> {
        let settings = sqlx::query!(
            r#"
            SELECT enabled, strip_utm_params, block_known_trackers,
                   warn_on_redirect, whitelist_domains as "whitelist_domains: Vec<String>"
            FROM link_proxy_settings
            WHERE user_id = $1
            "#,
            user_id
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(settings
            .map(|s| LinkProxySettings {
                enabled: s.enabled,
                strip_utm_params: s.strip_utm_params,
                block_known_trackers: s.block_known_trackers,
                warn_on_redirect: s.warn_on_redirect,
                whitelist_domains: s.whitelist_domains,
            })
            .unwrap_or(LinkProxySettings {
                enabled: true,
                strip_utm_params: true,
                block_known_trackers: true,
                warn_on_redirect: true,
                whitelist_domains: vec![],
            }))
    }

    pub async fn proxy_links_in_email(
        &self,
        email_id: Uuid,
        html: &str,
    ) -> Result<String, PrivacyError> {
        let link_pattern = regex::Regex::new(r#"href\s*=\s*["']([^"']+)["']"#).unwrap();

        let mut result = html.to_string();

        for cap in link_pattern.captures_iter(html) {
            if let Some(url_match) = cap.get(1) {
                let original_url = url_match.as_str();

                // Skip mailto:, tel:, and other non-http links
                if !original_url.starts_with("http") {
                    continue;
                }

                let proxied = self.create_proxy_link(email_id, original_url).await?;
                result = result.replace(original_url, &proxied.proxy_url);
            }
        }

        Ok(result)
    }

    async fn create_proxy_link(
        &self,
        email_id: Uuid,
        original_url: &str,
    ) -> Result<ProxiedLink, PrivacyError> {
        let link_id = Uuid::new_v4();
        let proxy_url = format!("{}/p/{}", self.proxy_base_url, link_id);

        let link = sqlx::query_as!(
            ProxiedLink,
            r#"
            INSERT INTO proxied_links (id, email_id, original_url, proxy_url, created_at)
            VALUES ($1, $2, $3, $4, NOW())
            RETURNING id, original_url, proxy_url, email_id, created_at, click_count
            "#,
            link_id,
            email_id,
            original_url,
            proxy_url
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(link)
    }

    pub async fn resolve_proxy(&self, link_id: Uuid) -> Result<String, PrivacyError> {
        let link = sqlx::query!(
            r#"
            UPDATE proxied_links
            SET click_count = click_count + 1
            WHERE id = $1
            RETURNING original_url
            "#,
            link_id
        )
        .fetch_optional(&self.pool)
        .await?
        .ok_or(PrivacyError::InvalidUrl("Proxy link not found".to_string()))?;

        // Clean URL before returning
        let cleaned = self.clean_url(&link.original_url)?;

        Ok(cleaned)
    }

    fn clean_url(&self, url: &str) -> Result<String, PrivacyError> {
        let mut parsed = Url::parse(url)
            .map_err(|e| PrivacyError::InvalidUrl(e.to_string()))?;

        // Remove tracking parameters
        let tracking_params = [
            "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
            "fbclid", "gclid", "msclkid", "mc_eid", "mc_cid",
            "ref", "source", "tracking_id",
        ];

        let query_pairs: Vec<(String, String)> = parsed
            .query_pairs()
            .filter(|(key, _)| !tracking_params.contains(&key.as_ref()))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();

        if query_pairs.is_empty() {
            parsed.set_query(None);
        } else {
            let query: String = query_pairs
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join("&");
            parsed.set_query(Some(&query));
        }

        Ok(parsed.to_string())
    }
}
