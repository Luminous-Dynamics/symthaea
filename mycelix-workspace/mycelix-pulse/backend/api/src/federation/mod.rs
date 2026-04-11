// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Federation Protocol
//!
//! Cross-instance communication for trust sharing and identity verification

pub mod protocol;
pub mod discovery;
pub mod identity;
pub mod trust_sync;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ============================================================================
// Federation Types
// ============================================================================

/// A federated instance in the network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedInstance {
    pub id: Uuid,
    pub domain: String,
    pub display_name: String,
    pub public_key: String,
    pub protocol_version: String,
    pub capabilities: Vec<FederationCapability>,
    pub status: InstanceStatus,
    pub last_seen: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum InstanceStatus {
    Active,
    Inactive,
    Suspended,
    Blocked,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FederationCapability {
    TrustSync,
    IdentityVerification,
    MessageRelay,
    KeyDiscovery,
    ReputationSharing,
}

/// Federated identity linking local user to remote instances
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedIdentity {
    pub id: Uuid,
    pub local_user_id: Uuid,
    pub remote_instance: String,
    pub remote_user_id: String,
    pub verification_status: VerificationStatus,
    pub verified_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum VerificationStatus {
    Pending,
    Verified,
    Failed,
    Revoked,
}

// ============================================================================
// Federation Messages
// ============================================================================

/// Protocol messages for inter-instance communication
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
pub enum FederationMessage {
    /// Handshake to establish connection
    Hello(HelloPayload),
    /// Response to handshake
    HelloAck(HelloAckPayload),
    /// Request trust attestations for a user
    TrustQuery(TrustQueryPayload),
    /// Response with trust attestations
    TrustResponse(TrustResponsePayload),
    /// Push trust update
    TrustUpdate(TrustUpdatePayload),
    /// Identity verification request
    VerifyIdentity(VerifyIdentityPayload),
    /// Identity verification response
    IdentityVerified(IdentityVerifiedPayload),
    /// Key discovery request
    KeyQuery(KeyQueryPayload),
    /// Key discovery response
    KeyResponse(KeyResponsePayload),
    /// Reputation query
    ReputationQuery(ReputationQueryPayload),
    /// Reputation response
    ReputationResponse(ReputationResponsePayload),
    /// Error response
    Error(ErrorPayload),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelloPayload {
    pub instance_id: Uuid,
    pub domain: String,
    pub public_key: String,
    pub protocol_version: String,
    pub capabilities: Vec<FederationCapability>,
    pub timestamp: DateTime<Utc>,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelloAckPayload {
    pub instance_id: Uuid,
    pub accepted: bool,
    pub capabilities: Vec<FederationCapability>,
    pub challenge: Option<String>,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustQueryPayload {
    pub requester: String,
    pub target_user: String,
    pub context: Option<String>,
    pub depth: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustResponsePayload {
    pub target_user: String,
    pub attestations: Vec<FederatedAttestation>,
    pub aggregate_score: Option<f64>,
    pub path: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedAttestation {
    pub from_user: String,
    pub from_instance: String,
    pub trust_level: f64,
    pub context: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustUpdatePayload {
    pub attestation: FederatedAttestation,
    pub action: TrustAction,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TrustAction {
    Create,
    Update,
    Revoke,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyIdentityPayload {
    pub local_user: String,
    pub remote_user: String,
    pub proof_type: ProofType,
    pub proof_data: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ProofType {
    SignedChallenge,
    DnsRecord,
    WellKnown,
    CrossSign,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityVerifiedPayload {
    pub local_user: String,
    pub remote_user: String,
    pub verified: bool,
    pub certificate: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyQueryPayload {
    pub user_id: String,
    pub key_type: KeyType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum KeyType {
    Signing,
    Encryption,
    All,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyResponsePayload {
    pub user_id: String,
    pub keys: Vec<PublicKeyInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicKeyInfo {
    pub key_id: String,
    pub key_type: String,
    pub public_key: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub revoked: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationQueryPayload {
    pub user_id: String,
    pub aspects: Vec<ReputationAspect>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ReputationAspect {
    Overall,
    Responsiveness,
    Reliability,
    SpamReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationResponsePayload {
    pub user_id: String,
    pub scores: Vec<ReputationScore>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationScore {
    pub aspect: ReputationAspect,
    pub score: f64,
    pub confidence: f64,
    pub sample_size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPayload {
    pub code: String,
    pub message: String,
    pub details: Option<serde_json::Value>,
}

// ============================================================================
// Federation Configuration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationConfig {
    /// This instance's domain
    pub domain: String,
    /// Instance display name
    pub display_name: String,
    /// Whether federation is enabled
    pub enabled: bool,
    /// Allowed instances (empty = allow all)
    pub allowed_instances: Vec<String>,
    /// Blocked instances
    pub blocked_instances: Vec<String>,
    /// Whether to auto-accept federation requests
    pub auto_accept: bool,
    /// Maximum trust depth to follow
    pub max_trust_depth: u8,
    /// Trust decay for federated attestations
    pub federated_trust_decay: f64,
}

impl Default for FederationConfig {
    fn default() -> Self {
        Self {
            domain: "localhost".to_string(),
            display_name: "Mycelix Mail Instance".to_string(),
            enabled: true,
            allowed_instances: vec![],
            blocked_instances: vec![],
            auto_accept: false,
            max_trust_depth: 3,
            federated_trust_decay: 0.1,
        }
    }
}
