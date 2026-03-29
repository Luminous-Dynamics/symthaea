// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared types for the Mycelix-Mail backend
//!
//! These types mirror the integrity zome types but are optimized for JSON serialization
//! and frontend consumption. They convert to/from the Holochain types.

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use utoipa::ToSchema;

/// Epistemic tiers from Mycelix Epistemic Charter v2.0
/// Mirrors `mycelix_mail_integrity::EpistemicTier`
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum EpistemicTier {
    /// Tier 0: Null - No epistemic claim
    Tier0Null,
    /// Tier 1: Testimonial - Based on personal testimony
    Tier1Testimonial,
    /// Tier 2: Privately Verifiable - Can be verified by recipient
    Tier2PrivatelyVerifiable,
    /// Tier 3: Cryptographically Proven - Verified via cryptographic proof
    Tier3CryptographicallyProven,
    /// Tier 4: Publicly Reproducible - Independently verifiable by anyone
    Tier4PubliclyReproducible,
}

impl Default for EpistemicTier {
    fn default() -> Self {
        Self::Tier1Testimonial
    }
}

/// Email message as returned to the frontend
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[schema(example = json!({
    "id": "uhCkkX7...",
    "from_did": "did:key:z6Mk...",
    "to_did": "did:key:z6Mk...",
    "subject": "Hello from Mycelix!",
    "body": "This is a secure message.",
    "timestamp": "2025-01-15T10:30:00Z",
    "epistemic_tier": "tier1_testimonial",
    "is_spam": false,
    "is_read": false,
    "is_starred": false
}))]
pub struct Email {
    /// Action hash of the message (base64 encoded)
    pub id: String,
    /// Sender DID
    pub from_did: String,
    /// Recipient DID
    pub to_did: String,
    /// Decrypted subject (decryption happens in frontend or backend service)
    pub subject: String,
    /// Message body (fetched from IPFS via CID)
    pub body: String,
    /// ISO 8601 timestamp
    pub timestamp: DateTime<Utc>,
    /// Thread ID for conversation grouping
    pub thread_id: Option<String>,
    /// Epistemic classification
    pub epistemic_tier: EpistemicTier,
    /// Sender's trust score (cached)
    pub sender_trust_score: Option<f64>,
    /// Whether marked as spam
    pub is_spam: bool,
    /// Whether message has been read
    pub is_read: bool,
    /// Whether message is starred
    pub is_starred: bool,
    /// Labels applied to this message
    pub labels: Vec<String>,
}

/// Input for sending a new email
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[schema(example = json!({
    "to_did": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
    "subject": "Hello!",
    "body": "This is my secure message content.",
    "epistemic_tier": "tier1_testimonial"
}))]
pub struct SendEmailInput {
    /// Recipient DID
    pub to_did: String,
    /// Email subject line
    pub subject: String,
    /// Email body content
    pub body: String,
    /// Optional thread ID for replies
    pub thread_id: Option<String>,
    /// Epistemic classification (defaults to Tier1Testimonial)
    #[serde(default)]
    pub epistemic_tier: EpistemicTier,
}

/// Trust score information from MATL (Mycelix Adaptive Trust Layer)
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[schema(example = json!({
    "did": "did:key:z6Mk...",
    "score": 0.85,
    "last_updated": "2025-01-15T10:30:00Z",
    "source": "matl",
    "is_byzantine": false,
    "interaction_count": 42
}))]
pub struct TrustScoreInfo {
    /// The DID this trust score applies to
    pub did: String,
    /// Trust score (0.0 - 1.0)
    pub score: f64,
    /// When the score was last updated
    pub last_updated: DateTime<Utc>,
    /// Source of the trust score (e.g., "matl", "local", "cross-happ")
    pub source: String,
    /// Whether this user is flagged as potentially Byzantine
    pub is_byzantine: bool,
    /// Number of interactions (for confidence calculation)
    pub interaction_count: u64,
}

/// Contact entry
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Contact {
    /// Unique contact ID
    pub id: String,
    /// Display name
    pub name: String,
    /// Contact's DID
    pub did: String,
    /// Optional email alias
    pub email_alias: Option<String>,
    /// Optional notes
    pub notes: Option<String>,
    /// When the contact was added
    pub added_at: DateTime<Utc>,
    /// Trust score for this contact (cached)
    pub trust_score: Option<f64>,
}

/// Input for adding a contact
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[schema(example = json!({
    "name": "Alice",
    "did": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
    "email_alias": "alice@mycelix.local"
}))]
pub struct AddContactInput {
    /// Display name for the contact
    pub name: String,
    /// Contact's DID
    pub did: String,
    /// Optional email alias
    pub email_alias: Option<String>,
    /// Optional notes about the contact
    pub notes: Option<String>,
}

/// DID registration input
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct RegisterDidInput {
    /// DID to register (e.g., "did:key:z6Mk...")
    pub did: String,
}

/// DID resolution result
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DidResolution {
    /// The resolved DID
    pub did: String,
    /// Agent public key (Base64 encoded)
    pub agent_pub_key: String,
    /// Whether this is a local agent
    pub is_local: bool,
}

/// Spam report input
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[schema(example = json!({
    "message_id": "uhCkkX7...",
    "reason": "Unsolicited commercial content"
}))]
pub struct SpamReportInput {
    /// ID of the message being reported
    pub message_id: String,
    /// Reason for the spam report
    pub reason: String,
}

/// Login request
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[schema(example = json!({
    "did": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
    "password": "secure_password_123"
}))]
pub struct LoginRequest {
    /// User's DID
    pub did: String,
    /// User's password
    pub password: String,
}

/// Successful login/registration response
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct LoginResponse {
    /// JWT authentication token
    pub token: String,
    /// User's DID
    pub did: String,
    /// User's Holochain agent public key (base64 encoded)
    pub agent_pub_key: String,
    /// Token expiration time
    pub expires_at: DateTime<Utc>,
}

/// Registration request
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[schema(example = json!({
    "did": "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
    "password": "secure_password_123",
    "display_name": "Alice"
}))]
pub struct RegisterRequest {
    /// DID to register (will be stored on DHT)
    pub did: String,
    /// Password for local authentication
    pub password: String,
    /// Optional display name
    pub display_name: Option<String>,
}

/// User session stored in JWT claims
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct UserClaims {
    /// Subject (user's DID)
    pub sub: String,
    /// Agent public key
    pub agent_pub_key: String,
    /// Expiration timestamp
    pub exp: i64,
    /// Issued at timestamp
    pub iat: i64,
}

/// Filter options for inbox queries
#[derive(Debug, Clone, Serialize, Deserialize, Default, ToSchema)]
pub struct InboxFilter {
    /// Minimum trust score for sender (spam filtering)
    pub min_trust: Option<f64>,
    /// Only unread messages
    #[serde(default)]
    pub unread_only: bool,
    /// Only starred messages
    #[serde(default)]
    pub starred_only: bool,
    /// Filter by label
    pub label: Option<String>,
    /// Search query (searches subject and sender)
    pub search: Option<String>,
    /// Pagination: offset (default: 0)
    pub offset: Option<u32>,
    /// Pagination: limit (default: 50)
    pub limit: Option<u32>,
}

/// Paginated response wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginatedResponse<T> {
    /// List of items for the current page
    pub data: Vec<T>,
    /// Total number of items
    pub total: u64,
    /// Current offset
    pub offset: u32,
    /// Items per page
    pub limit: u32,
    /// Whether more items exist
    pub has_more: bool,
}

/// API error response
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[schema(example = json!({
    "code": "validation_error",
    "message": "Invalid DID format"
}))]
pub struct ApiError {
    /// Error code (e.g., "validation_error", "not_found", "unauthorized")
    pub code: String,
    /// Human-readable error message
    pub message: String,
    /// Optional additional details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

impl ApiError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            details: None,
        }
    }

    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = Some(details);
        self
    }
}

/// Health check response
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[schema(example = json!({
    "status": "healthy",
    "version": "0.1.0",
    "holochain_connected": true,
    "bridge_connected": true,
    "bridge_mode": "stub",
    "uptime_seconds": 3600
}))]
pub struct HealthResponse {
    /// Service status ("healthy", "degraded", "unhealthy")
    pub status: String,
    /// Backend version
    pub version: String,
    /// Whether connected to Holochain conductor
    pub holochain_connected: bool,
    /// Whether connected to Bridge for cross-hApp reputation
    pub bridge_connected: bool,
    /// Bridge connection mode (connected, stub, fallback, disconnected)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bridge_mode: Option<String>,
    /// Server uptime in seconds
    pub uptime_seconds: u64,
}

/// WebSocket message types for real-time updates
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(tag = "type", content = "payload")]
pub enum WsMessage {
    /// New email received
    NewEmail(Email),
    /// Trust score updated
    TrustUpdate(TrustScoreInfo),
    /// Email status changed (read, starred, etc.)
    EmailStatusChanged {
        /// Email ID
        id: String,
        /// Map of changed fields
        changes: serde_json::Value,
    },
    /// Connection established
    Connected {
        /// Unique session identifier
        session_id: String,
    },
    /// Ping for keepalive
    Ping,
    /// Pong response
    Pong,
    /// Error message
    Error(ApiError),
}
