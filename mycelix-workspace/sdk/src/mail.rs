// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mycelix Mail Module
//!
//! Client-side types and helpers for interacting with the Mycelix Mail cluster.
//!
//! Provides:
//! - Typed request/response wrappers for mail zome extern functions
//! - `MailClient` for ergonomic cross-cluster communication
//! - PQC-aware crypto suite enums matching the integrity zome's validation
//!
//! The mail cluster consists of 12 zomes:
//! - **messages**: Core encrypted email send/receive/threading
//! - **contacts**: Address book management
//! - **profiles**: User profile and display settings
//! - **trust**: Trust scoring for spam filtering
//! - **keys**: Post-quantum key management (Ed25519, Dilithium3, X25519, Kyber1024)
//! - **capabilities**: Capability-based access control for shared mailboxes
//! - **sync**: Offline-first CRDT synchronization
//! - **federation**: Cross-cell mail delivery and routing
//! - **search**: Local full-text search indexing
//! - **backup**: Encrypted backup and restore
//! - **scheduler**: Delayed and recurring email sends
//! - **audit**: Compliance audit logging
//!
//! ## Example
//!
//! ```rust
//! use mycelix_sdk::mail::{MailClient, SendEmailRequest, CryptoSuite};
//!
//! let client = MailClient::new("did:mycelix:alice");
//! assert_eq!(MailClient::role_name(), "mail");
//!
//! let req = SendEmailRequest {
//!     recipient_did: "did:mycelix:bob".to_string(),
//!     encrypted_subject: vec![0u8; 64],
//!     encrypted_body: vec![0u8; 256],
//!     nonce: [0u8; 24],
//!     crypto_suite: CryptoSuite::Ed25519X25519,
//!     signature: vec![0u8; 64],
//!     ephemeral_public_key: vec![0u8; 32],
//!     message_id: "msg-001".to_string(),
//!     thread_id: None,
//! };
//! let (fn_name, _payload) = MailClient::send_email_request(&req);
//! assert_eq!(fn_name, "send_email");
//! ```

use serde::{Deserialize, Serialize};

// =============================================================================
// MailClient
// =============================================================================

/// Client for mycelix-mail cluster operations.
///
/// Lightweight struct that builds serialized request payloads for dispatching
/// via `CallTargetCell::OtherRole("mail")` in Holochain, or for use in SDK
/// tests and simulations.
#[derive(Debug, Clone)]
pub struct MailClient {
    /// Agent decentralized identifier
    pub agent_did: String,
}

impl MailClient {
    /// Create a new mail client for the given agent DID.
    pub fn new(agent_did: impl Into<String>) -> Self {
        Self {
            agent_did: agent_did.into(),
        }
    }

    /// Target role name for `CallTargetCell::OtherRole`.
    pub fn role_name() -> &'static str {
        "mail"
    }

    /// Build a send_email request payload.
    pub fn send_email_request(req: &SendEmailRequest) -> (&'static str, Vec<u8>) {
        (
            "send_email",
            serde_json::to_vec(req).expect("SendEmailRequest serialization"),
        )
    }

    /// Build a get_inbox request payload.
    pub fn get_inbox_request(query: &EmailQuery) -> (&'static str, Vec<u8>) {
        (
            "get_inbox",
            serde_json::to_vec(query).expect("EmailQuery serialization"),
        )
    }

    /// Build a get_sent request payload.
    pub fn get_sent_request(query: &EmailQuery) -> (&'static str, Vec<u8>) {
        (
            "get_sent",
            serde_json::to_vec(query).expect("EmailQuery serialization"),
        )
    }

    /// Build a mark_as_read request payload.
    pub fn mark_as_read_request(req: &MarkAsReadRequest) -> (&'static str, Vec<u8>) {
        (
            "mark_as_read",
            serde_json::to_vec(req).expect("MarkAsReadRequest serialization"),
        )
    }

    /// Build a search request payload.
    pub fn search_request(query: &SearchQuery) -> (&'static str, Vec<u8>) {
        (
            "search_emails",
            serde_json::to_vec(query).expect("SearchQuery serialization"),
        )
    }

    /// Build a create_folder request payload.
    pub fn create_folder_request(name: &str) -> (&'static str, Vec<u8>) {
        (
            "create_folder",
            serde_json::to_vec(&serde_json::json!({"name": name})).expect("folder serialization"),
        )
    }

    /// Build a federation send (cross-network delivery) request payload.
    pub fn federated_send_request(req: &FederatedSendRequest) -> (&'static str, Vec<u8>) {
        (
            "federated_send",
            serde_json::to_vec(req).expect("FederatedSendRequest serialization"),
        )
    }
}

// =============================================================================
// Crypto Suite (mirrors integrity zome validation)
// =============================================================================

/// Cryptographic suite for email encryption and signing.
///
/// Must match the integrity zome's validation rules for signature and key lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CryptoSuite {
    /// Ed25519 signing + X25519 key exchange (classical)
    Ed25519X25519,
    /// Dilithium3 signing + Kyber1024 KEM (post-quantum, NIST standard)
    Dilithium3Kyber1024,
    /// Dilithium2 signing + Kyber768 KEM (post-quantum, smaller)
    Dilithium2Kyber768,
    /// Ed25519 + X25519 + Dilithium3 + Kyber1024 (hybrid classical+PQ)
    HybridClassicalPQ,
}

impl CryptoSuite {
    /// Expected signature length in bytes for this suite.
    pub fn expected_signature_len(&self) -> usize {
        match self {
            Self::Ed25519X25519 => 64,
            Self::Dilithium3Kyber1024 => 3293,
            Self::Dilithium2Kyber768 => 2420,
            Self::HybridClassicalPQ => 64 + 3293, // Ed25519 + Dilithium3
        }
    }

    /// Expected ephemeral public key length in bytes.
    pub fn expected_ephemeral_key_len(&self) -> usize {
        match self {
            Self::Ed25519X25519 => 32,            // X25519
            Self::Dilithium3Kyber1024 => 1568,    // Kyber1024
            Self::Dilithium2Kyber768 => 1088,     // Kyber768
            Self::HybridClassicalPQ => 32 + 1568, // X25519 + Kyber1024
        }
    }
}

// =============================================================================
// Request / Response Types
// =============================================================================

/// Request to send an encrypted email.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SendEmailRequest {
    /// Recipient's DID
    pub recipient_did: String,
    /// Encrypted subject bytes
    pub encrypted_subject: Vec<u8>,
    /// Encrypted body bytes
    pub encrypted_body: Vec<u8>,
    /// Encryption nonce (24 bytes for XChaCha20-Poly1305)
    pub nonce: [u8; 24],
    /// Crypto suite used for signing and key exchange
    pub crypto_suite: CryptoSuite,
    /// Signature over canonical email content
    pub signature: Vec<u8>,
    /// Ephemeral public key for key exchange
    pub ephemeral_public_key: Vec<u8>,
    /// Unique message identifier
    pub message_id: String,
    /// Thread ID for replies (None for new conversations)
    pub thread_id: Option<String>,
}

/// Query parameters for inbox/sent listing.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct EmailQuery {
    /// Filter by read status
    pub is_read: Option<bool>,
    /// Filter by starred status
    pub is_starred: Option<bool>,
    /// Filter by folder name
    pub folder: Option<String>,
    /// Maximum results to return
    pub limit: Option<u32>,
    /// Cursor for pagination (action hash of last item)
    pub cursor: Option<String>,
}

/// Request to mark an email as read.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MarkAsReadRequest {
    /// Action hash of the email to mark
    pub email_hash: String,
    /// Whether to send a read receipt to the sender
    pub send_receipt: bool,
}

/// Full-text search query.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchQuery {
    /// Search terms (control characters will be stripped by the search zome)
    pub query: String,
    /// Maximum results
    pub limit: Option<u32>,
    /// Search scope
    pub scope: Option<SearchScope>,
}

/// Scope for email search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchScope {
    /// Search inbox only
    Inbox,
    /// Search sent only
    Sent,
    /// Search all emails
    All,
}

/// Request for cross-network federated delivery.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FederatedSendRequest {
    /// Target network ID
    pub target_network_id: String,
    /// Encrypted email payload
    pub encrypted_payload: Vec<u8>,
    /// Routing priority (0-100, validated by federation zome)
    pub priority: u8,
}

/// Summary of an email in inbox/sent listings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmailSummary {
    /// Action hash of the email entry
    pub hash: String,
    /// Sender DID
    pub from_did: String,
    /// Recipient DID
    pub to_did: String,
    /// Encrypted subject bytes (decrypt client-side)
    pub encrypted_subject: Vec<u8>,
    /// Timestamp in microseconds
    pub timestamp_us: i64,
    /// Whether the email has been read
    pub is_read: bool,
    /// Whether the email is starred
    pub is_starred: bool,
    /// Thread ID if part of a conversation
    pub thread_id: Option<String>,
    /// Crypto suite used
    pub crypto_suite: CryptoSuite,
}

/// Response from inbox/sent queries.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmailListResponse {
    /// Email summaries matching the query
    pub emails: Vec<EmailSummary>,
    /// Cursor for next page (None if no more results)
    pub next_cursor: Option<String>,
    /// Total count (if available)
    pub total: Option<u64>,
}

// =============================================================================
// Attachment Types
// =============================================================================

/// Maximum attachment chunk size (10 MB), matching integrity validation.
pub const MAX_ATTACHMENT_CHUNK_SIZE: usize = 10 * 1024 * 1024;

/// Maximum number of chunks per attachment.
pub const MAX_ATTACHMENT_CHUNKS: u32 = 1000;

/// Attachment metadata for chunked encrypted file transfer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AttachmentMeta {
    /// Original filename
    pub filename: String,
    /// MIME type
    pub mime_type: String,
    /// Total size in bytes
    pub total_size: u64,
    /// Number of chunks
    pub chunk_count: u32,
    /// SHA-256 hash of the complete plaintext (32 bytes)
    pub content_hash: Vec<u8>,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mail_client_role_name() {
        assert_eq!(MailClient::role_name(), "mail");
    }

    #[test]
    fn test_mail_client_new() {
        let client = MailClient::new("did:mycelix:alice");
        assert_eq!(client.agent_did, "did:mycelix:alice");
    }

    #[test]
    fn test_send_email_request_serializes() {
        let req = SendEmailRequest {
            recipient_did: "did:mycelix:bob".to_string(),
            encrypted_subject: vec![1, 2, 3],
            encrypted_body: vec![4, 5, 6],
            nonce: [0u8; 24],
            crypto_suite: CryptoSuite::Ed25519X25519,
            signature: vec![0u8; 64],
            ephemeral_public_key: vec![0u8; 32],
            message_id: "msg-001".to_string(),
            thread_id: None,
        };
        let (fn_name, payload) = MailClient::send_email_request(&req);
        assert_eq!(fn_name, "send_email");
        assert!(!payload.is_empty());
        // Round-trip
        let decoded: SendEmailRequest = serde_json::from_slice(&payload).unwrap();
        assert_eq!(decoded, req);
    }

    #[test]
    fn test_crypto_suite_signature_lengths() {
        assert_eq!(CryptoSuite::Ed25519X25519.expected_signature_len(), 64);
        assert_eq!(
            CryptoSuite::Dilithium3Kyber1024.expected_signature_len(),
            3293
        );
        assert_eq!(
            CryptoSuite::Dilithium2Kyber768.expected_signature_len(),
            2420
        );
        assert_eq!(
            CryptoSuite::HybridClassicalPQ.expected_signature_len(),
            64 + 3293
        );
    }

    #[test]
    fn test_crypto_suite_ephemeral_key_lengths() {
        assert_eq!(CryptoSuite::Ed25519X25519.expected_ephemeral_key_len(), 32);
        assert_eq!(
            CryptoSuite::Dilithium3Kyber1024.expected_ephemeral_key_len(),
            1568
        );
        assert_eq!(
            CryptoSuite::Dilithium2Kyber768.expected_ephemeral_key_len(),
            1088
        );
        assert_eq!(
            CryptoSuite::HybridClassicalPQ.expected_ephemeral_key_len(),
            32 + 1568
        );
    }

    #[test]
    fn test_email_query_default() {
        let query = EmailQuery::default();
        assert!(query.is_read.is_none());
        assert!(query.limit.is_none());
        assert!(query.cursor.is_none());
    }

    #[test]
    fn test_get_inbox_request_serializes() {
        let query = EmailQuery {
            is_read: Some(false),
            limit: Some(50),
            ..Default::default()
        };
        let (fn_name, payload) = MailClient::get_inbox_request(&query);
        assert_eq!(fn_name, "get_inbox");
        let decoded: EmailQuery = serde_json::from_slice(&payload).unwrap();
        assert_eq!(decoded.is_read, Some(false));
        assert_eq!(decoded.limit, Some(50));
    }

    #[test]
    fn test_federated_send_priority_field() {
        let req = FederatedSendRequest {
            target_network_id: "net-001".to_string(),
            encrypted_payload: vec![0u8; 128],
            priority: 100,
        };
        let (fn_name, payload) = MailClient::federated_send_request(&req);
        assert_eq!(fn_name, "federated_send");
        let decoded: FederatedSendRequest = serde_json::from_slice(&payload).unwrap();
        assert_eq!(decoded.priority, 100);
    }

    #[test]
    fn test_attachment_meta_constants() {
        assert_eq!(MAX_ATTACHMENT_CHUNK_SIZE, 10 * 1024 * 1024);
        assert_eq!(MAX_ATTACHMENT_CHUNKS, 1000);
    }
}
