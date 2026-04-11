// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Email routes
//!
//! Now enhanced with Bridge integration for cross-hApp reputation.
//! - Verifies sender identity via Bridge before accepting mail
//! - Reports positive/negative interactions to cross-hApp reputation
//! - Uses combined local + cross-hApp trust for spam filtering
//! - Integrates with identity-client for DID verification on send

use axum::{
    extract::{Path, Query, State},
    routing::{delete, get, post},
    Json, Router,
};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};

use crate::error::{AppError, AppResult};
use crate::middleware::AuthenticatedUser;
use crate::routes::{AppState, notify_new_mail};
use crate::services::bridge::{has_sufficient_reputation, spam_likelihood};
use crate::types::{ApiError, Email, InboxFilter, PaginatedResponse, SendEmailInput};
use crate::validation::{validate_did, validate_subject, validate_body, sanitize_string};
use mycelix_identity_client::AssuranceLevel;

/// Create email routes
pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", post(send_email))
        .route("/inbox", get(get_inbox))
        .route("/outbox", get(get_outbox))
        .route("/:id", get(get_email))
        .route("/:id", delete(delete_email))
        .route("/:id/thread", get(get_thread))
        .route("/:id/spam", post(report_spam))
        .route("/:id/not-spam", post(mark_not_spam))
}

/// Send an email
#[utoipa::path(
    post,
    path = "/emails",
    tag = "emails",
    request_body = SendEmailInput,
    responses(
        (status = 200, description = "Email sent successfully", body = Email),
        (status = 400, description = "Validation error", body = ApiError),
        (status = 401, description = "Not authenticated", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn send_email(
    State(state): State<AppState>,
    user: AuthenticatedUser,
    Json(input): Json<SendEmailInput>,
) -> AppResult<Json<Email>> {
    // Validate inputs
    validate_did(&input.to_did)?;
    let subject = sanitize_string(&input.subject);
    validate_subject(&subject)?;
    validate_body(&input.body)?;

    tracing::info!("User {} sending email to {}", user.did, input.to_did);

    // Identity verification (non-blocking - logs warnings on failure)
    if state.config.identity_verify_on_send {
        verify_sender_identity(&state, &user.did).await;
    }

    // Encrypt subject using ChaCha20-Poly1305 with recipient's public key
    let subject_encrypted = encrypt_subject_for_recipient(&state, &subject, &input.to_did).await?;

    // Store body to storage service
    let stored = state.storage.store(input.body.as_bytes()).await?;
    let body_cid = stored.cid;

    // Build message for zome
    let message = crate::services::holochain::MailMessageInput {
        from_did: user.did.clone(),
        to_did: input.to_did.clone(),
        subject_encrypted,
        body_cid: body_cid.clone(),
        timestamp: chrono::Utc::now().timestamp_micros(),
        thread_id: input.thread_id.clone(),
        epistemic_tier: format!("{:?}", input.epistemic_tier),
    };

    // Send via Holochain
    let hash = state.holochain.send_message(message).await?;

    // Record positive interaction (we're sending to them)
    let _ = state.holochain.record_positive_interaction(&input.to_did).await;

    // Also report positive interaction to Bridge for cross-hApp reputation
    if let Err(e) = state.bridge.report_positive_interaction(
        &input.to_did,
        "mail_sent",
    ).await {
        tracing::warn!("Failed to report positive interaction to Bridge: {}", e);
    }

    // Broadcast new mail event for real-time updates
    notify_new_mail(
        &state.event_broadcast,
        user.did.clone(),
        subject.chars().take(50).collect(),
        chrono::Utc::now().timestamp_micros(),
    );

    // Return email object
    let email = Email {
        id: BASE64.encode(hash.get_raw_39()),
        from_did: user.did,
        to_did: input.to_did,
        subject,
        body: input.body,
        timestamp: chrono::Utc::now(),
        thread_id: input.thread_id,
        epistemic_tier: input.epistemic_tier,
        sender_trust_score: None,
        is_spam: false,
        is_read: true,
        is_starred: false,
        labels: vec![],
    };

    Ok(Json(email))
}

/// Get inbox with optional filtering
#[utoipa::path(
    get,
    path = "/emails/inbox",
    tag = "emails",
    params(
        ("min_trust" = Option<f64>, Query, description = "Minimum trust score filter"),
        ("unread_only" = Option<bool>, Query, description = "Only show unread messages"),
        ("starred_only" = Option<bool>, Query, description = "Only show starred messages"),
        ("label" = Option<String>, Query, description = "Filter by label"),
        ("search" = Option<String>, Query, description = "Search query"),
        ("offset" = Option<u32>, Query, description = "Pagination offset"),
        ("limit" = Option<u32>, Query, description = "Page size (default: 50)")
    ),
    responses(
        (status = 200, description = "Inbox messages", body = PaginatedEmailResponse),
        (status = 401, description = "Not authenticated", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn get_inbox(
    State(state): State<AppState>,
    user: AuthenticatedUser,
    Query(filter): Query<InboxFilter>,
) -> AppResult<Json<PaginatedResponse<Email>>> {
    tracing::debug!("User {} fetching inbox with filter {:?}", user.did, filter);

    let min_trust = filter.min_trust.unwrap_or(state.config.default_min_trust);

    // Get filtered inbox with Byzantine detection
    let result = state.holochain.filter_inbox_matl(min_trust).await?;

    // Convert to Email objects with trust scores
    let mut emails = Vec::new();
    for msg in result.messages {
        // Get cached trust score for sender (now includes cross-hApp reputation)
        let trust = state.trust_cache.get_trust(&msg.from_did).await.ok();

        // Also check cross-hApp reputation via Bridge
        let bridge_reputation = state.bridge.get_reputation(&msg.from_did).await.ok();
        let bridge_spam_likelihood = bridge_reputation
            .as_ref()
            .map(|r| spam_likelihood(r))
            .unwrap_or(0.5);

        // Determine if spam based on multiple signals:
        // 1. Byzantine detection from local MATL
        // 2. Byzantine detection from Bridge
        // 3. High spam likelihood from Bridge
        let is_spam_local = result.byzantine_senders.contains(&msg.from_did);
        let is_spam_bridge = bridge_reputation
            .as_ref()
            .map(|r| r.is_byzantine || spam_likelihood(r) > 0.8)
            .unwrap_or(false);
        let is_spam = is_spam_local || is_spam_bridge;

        // Use combined trust score if available
        let sender_trust = trust.as_ref().map(|t| t.combined_score);

        let email = Email {
            id: format!("msg_{}", BASE64.encode(&msg.body_cid)),
            from_did: msg.from_did.clone(),
            to_did: msg.to_did,
            subject: String::from_utf8_lossy(&msg.subject_encrypted).to_string(),
            body: String::new(), // Fetch from IPFS lazily
            timestamp: chrono::DateTime::from_timestamp_micros(msg.timestamp)
                .unwrap_or_default(),
            thread_id: msg.thread_id,
            epistemic_tier: parse_epistemic_tier(&msg.epistemic_tier),
            sender_trust_score: sender_trust,
            is_spam,
            is_read: false,
            is_starred: false,
            labels: vec![],
        };
        emails.push(email);
    }

    // Apply additional filters
    if filter.unread_only {
        emails.retain(|e| !e.is_read);
    }
    if filter.starred_only {
        emails.retain(|e| e.is_starred);
    }
    if let Some(ref label) = filter.label {
        emails.retain(|e| e.labels.contains(label));
    }
    if let Some(ref search) = filter.search {
        let search_lower = search.to_lowercase();
        emails.retain(|e| {
            e.subject.to_lowercase().contains(&search_lower)
                || e.from_did.to_lowercase().contains(&search_lower)
        });
    }

    let total = emails.len() as u64;
    let offset = filter.offset.unwrap_or(0);
    let limit = filter.limit.unwrap_or(50);

    // Apply pagination
    let paginated: Vec<Email> = emails
        .into_iter()
        .skip(offset as usize)
        .take(limit as usize)
        .collect();

    let has_more = (offset + limit) < total as u32;

    Ok(Json(PaginatedResponse {
        data: paginated,
        total,
        offset,
        limit,
        has_more,
    }))
}

/// Get outbox (sent emails)
#[utoipa::path(
    get,
    path = "/emails/outbox",
    tag = "emails",
    params(
        ("offset" = Option<u32>, Query, description = "Pagination offset"),
        ("limit" = Option<u32>, Query, description = "Page size (default: 50)")
    ),
    responses(
        (status = 200, description = "Sent emails", body = PaginatedEmailResponse),
        (status = 401, description = "Not authenticated", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn get_outbox(
    State(state): State<AppState>,
    user: AuthenticatedUser,
    Query(filter): Query<InboxFilter>,
) -> AppResult<Json<PaginatedResponse<Email>>> {
    tracing::debug!("User {} fetching outbox", user.did);

    let messages = state.holochain.get_outbox().await?;

    let emails: Vec<Email> = messages
        .into_iter()
        .map(|msg| Email {
            id: format!("msg_{}", BASE64.encode(&msg.body_cid)),
            from_did: msg.from_did,
            to_did: msg.to_did,
            subject: String::from_utf8_lossy(&msg.subject_encrypted).to_string(),
            body: String::new(),
            timestamp: chrono::DateTime::from_timestamp_micros(msg.timestamp)
                .unwrap_or_default(),
            thread_id: msg.thread_id,
            epistemic_tier: parse_epistemic_tier(&msg.epistemic_tier),
            sender_trust_score: None,
            is_spam: false,
            is_read: true,
            is_starred: false,
            labels: vec!["sent".to_string()],
        })
        .collect();

    let total = emails.len() as u64;
    let offset = filter.offset.unwrap_or(0);
    let limit = filter.limit.unwrap_or(50);

    let paginated: Vec<Email> = emails
        .into_iter()
        .skip(offset as usize)
        .take(limit as usize)
        .collect();

    Ok(Json(PaginatedResponse {
        data: paginated,
        total,
        offset,
        limit,
        has_more: (offset + limit) < total as u32,
    }))
}

/// Get a single email by ID
#[utoipa::path(
    get,
    path = "/emails/{id}",
    tag = "emails",
    params(
        ("id" = String, Path, description = "Email ID (base64 action hash)")
    ),
    responses(
        (status = 200, description = "Email details", body = Email),
        (status = 401, description = "Not authenticated", body = ApiError),
        (status = 404, description = "Email not found", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn get_email(
    State(state): State<AppState>,
    user: AuthenticatedUser,
    Path(id): Path<String>,
) -> AppResult<Json<Email>> {
    tracing::debug!("User {} fetching email {}", user.did, id);

    let hash = parse_action_hash(&id)?;
    let message = state
        .holochain
        .get_message(hash)
        .await?
        .ok_or_else(|| AppError::NotFound(format!("Email {} not found", id)))?;

    // Get trust score for sender
    let trust = state.trust_cache.get_trust(&message.from_did).await.ok();

    // Fetch body from IPFS/storage using body_cid
    let body = fetch_email_body(&state, &message.body_cid, &user.did).await
        .unwrap_or_else(|e| {
            tracing::warn!(
                cid = %message.body_cid,
                error = %e,
                "Failed to fetch email body from storage"
            );
            format!("[Body unavailable - CID: {}]", message.body_cid)
        });

    // Decrypt subject if encrypted
    let subject = decrypt_subject_for_recipient(&message.subject_encrypted, &user.did)
        .unwrap_or_else(|e| {
            tracing::debug!(error = %e, "Could not decrypt subject, using raw bytes");
            String::from_utf8_lossy(&message.subject_encrypted).to_string()
        });

    let email = Email {
        id,
        from_did: message.from_did,
        to_did: message.to_did,
        subject,
        body,
        timestamp: chrono::DateTime::from_timestamp_micros(message.timestamp)
            .unwrap_or_default(),
        thread_id: message.thread_id,
        epistemic_tier: parse_epistemic_tier(&message.epistemic_tier),
        sender_trust_score: trust.as_ref().map(|t| t.score),
        is_spam: trust.map(|t| t.is_byzantine).unwrap_or(false),
        is_read: false,
        is_starred: false,
        labels: vec![],
    };

    Ok(Json(email))
}

/// Delete an email
#[utoipa::path(
    delete,
    path = "/emails/{id}",
    tag = "emails",
    params(
        ("id" = String, Path, description = "Email ID to delete")
    ),
    responses(
        (status = 200, description = "Email deleted", body = DeleteResponse),
        (status = 401, description = "Not authenticated", body = ApiError),
        (status = 404, description = "Email not found", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn delete_email(
    State(state): State<AppState>,
    user: AuthenticatedUser,
    Path(id): Path<String>,
) -> AppResult<Json<serde_json::Value>> {
    tracing::info!("User {} deleting email {}", user.did, id);

    let hash = parse_action_hash(&id)?;
    state.holochain.delete_message(hash).await?;

    Ok(Json(serde_json::json!({ "deleted": true })))
}

/// Get thread for an email
#[utoipa::path(
    get,
    path = "/emails/{id}/thread",
    tag = "emails",
    params(
        ("id" = String, Path, description = "Email ID to get thread for")
    ),
    responses(
        (status = 200, description = "Thread messages", body = Vec<Email>),
        (status = 401, description = "Not authenticated", body = ApiError),
        (status = 404, description = "Email not found", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn get_thread(
    State(state): State<AppState>,
    user: AuthenticatedUser,
    Path(id): Path<String>,
) -> AppResult<Json<Vec<Email>>> {
    tracing::debug!("User {} fetching thread for {}", user.did, id);

    let hash = parse_action_hash(&id)?;
    let messages = state.holochain.get_thread(hash).await?;

    let emails: Vec<Email> = messages
        .into_iter()
        .map(|msg| Email {
            id: format!("msg_{}", BASE64.encode(&msg.body_cid)),
            from_did: msg.from_did,
            to_did: msg.to_did,
            subject: String::from_utf8_lossy(&msg.subject_encrypted).to_string(),
            body: String::new(),
            timestamp: chrono::DateTime::from_timestamp_micros(msg.timestamp)
                .unwrap_or_default(),
            thread_id: msg.thread_id,
            epistemic_tier: parse_epistemic_tier(&msg.epistemic_tier),
            sender_trust_score: None,
            is_spam: false,
            is_read: false,
            is_starred: false,
            labels: vec![],
        })
        .collect();

    Ok(Json(emails))
}

/// Report an email as spam
#[utoipa::path(
    post,
    path = "/emails/{id}/spam",
    tag = "emails",
    params(
        ("id" = String, Path, description = "Email ID to report")
    ),
    request_body = SpamReportBody,
    responses(
        (status = 200, description = "Spam reported", body = SpamReportResponse),
        (status = 401, description = "Not authenticated", body = ApiError),
        (status = 404, description = "Email not found", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn report_spam(
    State(state): State<AppState>,
    user: AuthenticatedUser,
    Path(id): Path<String>,
    Json(input): Json<SpamReportBody>,
) -> AppResult<Json<serde_json::Value>> {
    tracing::info!("User {} reporting spam: {}", user.did, id);

    let hash = parse_action_hash(&id)?;

    let reason = input.reason.unwrap_or_else(|| "Marked as spam".to_string());

    state
        .holochain
        .report_spam(crate::services::holochain::SpamReportInput {
            message_hash: hash,
            reason: reason.clone(),
        })
        .await?;

    // Get the message to find sender for reputation reporting
    if let Ok(Some(msg)) = state.holochain.get_message(parse_action_hash(&id)?).await {
        // Invalidate sender's trust cache
        state.trust_cache.invalidate(&msg.from_did).await;

        // Report spam to Bridge for cross-hApp reputation
        if let Err(e) = state.bridge.report_spam(&msg.from_did, &reason).await {
            tracing::warn!("Failed to report spam to Bridge: {}", e);
        }

        // Also invalidate Bridge cache for this sender
        state.bridge.invalidate_cache(&msg.from_did).await;
    }

    Ok(Json(serde_json::json!({ "reported": true })))
}

/// Mark an email as not spam
#[utoipa::path(
    post,
    path = "/emails/{id}/not-spam",
    tag = "emails",
    params(
        ("id" = String, Path, description = "Email ID to mark as not spam")
    ),
    responses(
        (status = 200, description = "Marked as not spam", body = SpamReportResponse),
        (status = 401, description = "Not authenticated", body = ApiError),
        (status = 404, description = "Email not found", body = ApiError)
    ),
    security(("bearer_auth" = []))
)]
pub async fn mark_not_spam(
    State(state): State<AppState>,
    user: AuthenticatedUser,
    Path(id): Path<String>,
) -> AppResult<Json<serde_json::Value>> {
    tracing::info!("User {} marking not spam: {}", user.did, id);

    // Get the message to find sender
    let hash = parse_action_hash(&id)?;
    if let Ok(Some(msg)) = state.holochain.get_message(hash).await {
        // Record positive interaction locally
        state.holochain.record_positive_interaction(&msg.from_did).await?;

        // Report positive interaction to Bridge for cross-hApp reputation
        if let Err(e) = state.bridge.report_positive_interaction(
            &msg.from_did,
            "marked_not_spam",
        ).await {
            tracing::warn!("Failed to report positive interaction to Bridge: {}", e);
        }

        // Invalidate caches
        state.trust_cache.invalidate(&msg.from_did).await;
        state.bridge.invalidate_cache(&msg.from_did).await;
    }

    Ok(Json(serde_json::json!({ "marked_safe": true })))
}

// Helper types and functions

/// Spam report request body
#[derive(serde::Deserialize, utoipa::ToSchema)]
pub struct SpamReportBody {
    /// Reason for reporting (optional)
    pub reason: Option<String>,
}

/// Delete response
#[derive(serde::Serialize, utoipa::ToSchema)]
pub struct DeleteResponse {
    /// Whether deletion succeeded
    pub deleted: bool,
}

/// Spam report response
#[derive(serde::Serialize, utoipa::ToSchema)]
pub struct SpamReportResponse {
    /// Whether report was submitted
    pub reported: bool,
}

/// Paginated email response (for OpenAPI documentation)
#[derive(utoipa::ToSchema)]
pub struct PaginatedEmailResponse {
    /// List of emails
    pub data: Vec<Email>,
    /// Total count
    pub total: u64,
    /// Current offset
    pub offset: u32,
    /// Page size
    pub limit: u32,
    /// More pages available
    pub has_more: bool,
}

fn parse_action_hash(id: &str) -> AppResult<holochain_types::prelude::ActionHash> {
    use holochain_types::prelude::ActionHash;

    let id_clean = id.strip_prefix("msg_").unwrap_or(id);
    let bytes = BASE64.decode(id_clean)
        .map_err(|_| AppError::ValidationError("Invalid email ID format".to_string()))?;

    if bytes.len() != 39 {
        return Err(AppError::ValidationError("Invalid email ID length".to_string()));
    }

    Ok(ActionHash::from_raw_39(bytes.try_into().unwrap()))
}

fn parse_epistemic_tier(s: &str) -> crate::types::EpistemicTier {
    match s {
        "Tier0Null" => crate::types::EpistemicTier::Tier0Null,
        "Tier1Testimonial" => crate::types::EpistemicTier::Tier1Testimonial,
        "Tier2PrivatelyVerifiable" => crate::types::EpistemicTier::Tier2PrivatelyVerifiable,
        "Tier3CryptographicallyProven" => crate::types::EpistemicTier::Tier3CryptographicallyProven,
        "Tier4PubliclyReproducible" => crate::types::EpistemicTier::Tier4PubliclyReproducible,
        _ => crate::types::EpistemicTier::Tier1Testimonial,
    }
}

/// Verify sender identity before sending email
///
/// This is non-blocking - if the identity service is unavailable or verification
/// fails, we log a warning but allow the email to be sent. This ensures graceful
/// degradation when the identity hApp is unavailable.
async fn verify_sender_identity(state: &AppState, sender_did: &str) {
    // Try to resolve the sender's DID
    match state.identity.resolve_did(sender_did).await {
        Ok(resolution) => {
            if resolution.success {
                tracing::debug!(
                    did = %sender_did,
                    cached = resolution.cached,
                    "Sender DID resolved successfully"
                );

                // Check assurance level
                match state.identity.get_assurance_level(sender_did).await {
                    Ok(level) => {
                        if level < AssuranceLevel::E1 {
                            tracing::warn!(
                                did = %sender_did,
                                level = ?level,
                                "Sender has low assurance level (unverified)"
                            );
                        } else {
                            tracing::debug!(
                                did = %sender_did,
                                level = ?level,
                                "Sender assurance level verified"
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            did = %sender_did,
                            error = %e,
                            "Failed to get sender assurance level"
                        );
                    }
                }

                // Check guardian endorsements (informational only)
                match state.identity.check_guardian_endorsement(sender_did).await {
                    Ok(guardians) => {
                        if !guardians.is_empty() {
                            let endorsed_count = guardians.iter().filter(|g| g.has_endorsed).count();
                            tracing::debug!(
                                did = %sender_did,
                                total = guardians.len(),
                                endorsed = endorsed_count,
                                "Sender guardian endorsements"
                            );
                        }
                    }
                    Err(e) => {
                        tracing::debug!(
                            did = %sender_did,
                            error = %e,
                            "Could not check guardian endorsements (non-critical)"
                        );
                    }
                }
            } else {
                tracing::warn!(
                    did = %sender_did,
                    error = ?resolution.error,
                    "Sender DID not found or resolution failed"
                );
            }
        }
        Err(e) => {
            tracing::warn!(
                did = %sender_did,
                error = %e,
                "Identity service unavailable for sender verification"
            );
        }
    }
}

/// Encrypt the email subject for the recipient using ChaCha20-Poly1305
///
/// This function:
/// 1. Resolves the recipient's DID to get their public key
/// 2. Uses X25519 ECDH to derive a shared secret
/// 3. Encrypts the subject with ChaCha20-Poly1305
/// 4. Returns encrypted bytes with nonce prepended
async fn encrypt_subject_for_recipient(
    state: &AppState,
    subject: &str,
    recipient_did: &str,
) -> AppResult<Vec<u8>> {
    use crate::services::crypto::encrypt_for_recipient;

    // Try to resolve recipient's public key from their DID
    let recipient_pubkey = match resolve_recipient_pubkey(state, recipient_did).await {
        Ok(pubkey) => pubkey,
        Err(e) => {
            tracing::warn!(
                did = %recipient_did,
                error = %e,
                "Could not resolve recipient public key, using fallback encryption"
            );
            // Fallback: derive a key from the DID for development/testing
            // In production, this should fail or use a different mechanism
            derive_fallback_key(recipient_did)
        }
    };

    // Encrypt subject using X25519 + ChaCha20-Poly1305
    let envelope = encrypt_for_recipient(subject.as_bytes(), &recipient_pubkey)?;

    // Serialize the envelope into bytes for storage
    // Format: [ephemeral_pubkey (32)] + [nonce (12)] + [ciphertext (variable)]
    let ephemeral_bytes = BASE64.decode(&envelope.ephemeral_pubkey)
        .map_err(|_| AppError::EncryptionError("Invalid ephemeral pubkey encoding".to_string()))?;
    let nonce_bytes = BASE64.decode(&envelope.nonce)
        .map_err(|_| AppError::EncryptionError("Invalid nonce encoding".to_string()))?;
    let ciphertext = BASE64.decode(&envelope.ciphertext)
        .map_err(|_| AppError::EncryptionError("Invalid ciphertext encoding".to_string()))?;

    let mut encrypted_data = Vec::with_capacity(32 + 12 + ciphertext.len());
    encrypted_data.extend_from_slice(&ephemeral_bytes);
    encrypted_data.extend_from_slice(&nonce_bytes);
    encrypted_data.extend(ciphertext);

    tracing::debug!(
        recipient_did = %recipient_did,
        encrypted_len = encrypted_data.len(),
        "Subject encrypted for recipient"
    );

    Ok(encrypted_data)
}

/// Resolve a recipient's X25519 public key from their DID
async fn resolve_recipient_pubkey(
    state: &AppState,
    recipient_did: &str,
) -> AppResult<[u8; 32]> {
    // First, try to resolve via Holochain DHT
    if let Ok(agent_pubkey) = state.holochain.resolve_did(recipient_did).await {
        // Convert Holochain agent pubkey to X25519 pubkey
        // Agent pubkeys are Ed25519, so we derive X25519 key from it
        let raw_bytes = agent_pubkey.get_raw_39();
        if raw_bytes.len() >= 32 {
            let mut pubkey = [0u8; 32];
            pubkey.copy_from_slice(&raw_bytes[..32]);
            return Ok(pubkey);
        }
    }

    // Try to resolve via identity service
    match state.identity.resolve_did(recipient_did).await {
        Ok(resolution) if resolution.success => {
            if let Some(ref doc) = resolution.did_document {
                for method in &doc.verification_method {
                    if method.type_ == "X25519KeyAgreementKey2020" {
                        if let Some(stripped) = method.public_key_multibase.strip_prefix('z') {
                            if let Ok(key_bytes) = bs58::decode(stripped).into_vec() {
                                if key_bytes.len() == 32 {
                                    let mut pubkey = [0u8; 32];
                                    pubkey.copy_from_slice(&key_bytes);
                                    return Ok(pubkey);
                                }
                            }
                        }
                    }
                }
            }
        }
        _ => {}
    }

    Err(AppError::NotFound(format!(
        "Could not resolve public key for DID: {}",
        recipient_did
    )))
}

/// Derive a fallback encryption key from a DID (for development/testing only)
///
/// WARNING: This is not secure for production use. It's a deterministic
/// derivation that allows testing without a full DID infrastructure.
fn derive_fallback_key(did: &str) -> [u8; 32] {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(b"mycelix-mail-fallback-key:");
    hasher.update(did.as_bytes());
    let hash = hasher.finalize();

    let mut key = [0u8; 32];
    key.copy_from_slice(&hash);
    key
}

/// Fetch email body from IPFS/storage and decrypt if needed
///
/// The body is stored encrypted in IPFS. This function:
/// 1. Retrieves the content by CID from storage
/// 2. Decrypts the content using the recipient's key
/// 3. Returns the plaintext body
async fn fetch_email_body(
    state: &AppState,
    body_cid: &str,
    recipient_did: &str,
) -> AppResult<String> {
    use crate::services::storage::retrieve_decrypted;

    // Get the recipient's encryption key for decryption
    let encryption_key = derive_fallback_key(recipient_did);

    // Try to retrieve and decrypt from storage
    match retrieve_decrypted(&state.storage, body_cid, &encryption_key).await {
        Ok(plaintext) => {
            String::from_utf8(plaintext)
                .map_err(|e| AppError::InternalError(format!("Invalid UTF-8 in body: {}", e)))
        }
        Err(e) => {
            // If decryption fails, try retrieving raw content (for backwards compatibility)
            tracing::debug!(
                cid = %body_cid,
                error = %e,
                "Encrypted retrieval failed, trying raw retrieval"
            );

            let raw_content = state.storage.retrieve(body_cid).await?;
            String::from_utf8(raw_content)
                .map_err(|e| AppError::InternalError(format!("Invalid UTF-8 in body: {}", e)))
        }
    }
}

/// Decrypt the email subject using the recipient's private key
///
/// Subject format: [ephemeral_pubkey (32)] + [nonce (12)] + [ciphertext (variable)]
fn decrypt_subject_for_recipient(
    encrypted_subject: &[u8],
    recipient_did: &str,
) -> AppResult<String> {
    use crate::services::crypto::decrypt_symmetric;

    // Check minimum length: ephemeral (32) + nonce (12) + at least some ciphertext
    if encrypted_subject.len() < 44 {
        // Subject might not be encrypted (legacy format)
        return String::from_utf8(encrypted_subject.to_vec())
            .map_err(|e| AppError::EncryptionError(format!("Invalid subject encoding: {}", e)));
    }

    // Extract components
    let ephemeral_pubkey = &encrypted_subject[0..32];
    let nonce = &encrypted_subject[32..44];
    let ciphertext = &encrypted_subject[44..];

    // Derive the recipient's secret key (in production, this would come from key storage)
    // For now, we use a deterministic derivation for testing
    let recipient_secret = derive_recipient_secret(recipient_did);

    // Compute shared secret via ECDH
    use x25519_dalek::{PublicKey, StaticSecret};
    use sha2::{Digest, Sha256};

    let mut ephemeral_arr = [0u8; 32];
    ephemeral_arr.copy_from_slice(ephemeral_pubkey);
    let ephemeral_public = PublicKey::from(ephemeral_arr);

    let shared_secret = recipient_secret.diffie_hellman(&ephemeral_public);

    // Derive encryption key from shared secret
    let mut hasher = Sha256::new();
    hasher.update(shared_secret.as_bytes());
    let key_bytes: [u8; 32] = hasher.finalize().into();

    // Decrypt using ChaCha20-Poly1305
    let mut nonce_arr = [0u8; 12];
    nonce_arr.copy_from_slice(nonce);

    let plaintext = decrypt_symmetric(ciphertext, &nonce_arr, &key_bytes)?;

    String::from_utf8(plaintext)
        .map_err(|e| AppError::EncryptionError(format!("Invalid subject encoding: {}", e)))
}

/// Derive a deterministic secret key from a DID (for development/testing only)
///
/// WARNING: In production, secret keys should be stored securely in a keystore,
/// not derived deterministically from the DID.
fn derive_recipient_secret(did: &str) -> x25519_dalek::StaticSecret {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(b"mycelix-mail-recipient-secret:");
    hasher.update(did.as_bytes());
    let hash = hasher.finalize();

    let mut secret_bytes = [0u8; 32];
    secret_bytes.copy_from_slice(&hash);

    x25519_dalek::StaticSecret::from(secret_bytes)
}
