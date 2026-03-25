// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Federation Integrity Zome
//!
//! Cross-cell federation for Mycelix Mail enabling:
//! - Inter-organization email routing
//! - Multi-DNA communication bridges
//! - External protocol gateways (SMTP, Matrix, etc.)
//! - Decentralized domain routing

use hdi::prelude::*;

/// Federated network identity
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FederatedNetwork {
    /// Unique network identifier
    pub network_id: String,
    /// DNA hash for this network
    pub dna_hash: DnaHash,
    /// Network display name
    pub name: String,
    /// Network domain (e.g., "org.example.mail")
    pub domain: String,
    /// Network type
    pub network_type: NetworkType,
    /// Public key for network-level operations
    pub network_pubkey: Vec<u8>,
    /// Bootstrap nodes for discovery
    pub bootstrap_nodes: Vec<String>,
    /// When registered
    pub registered_at: Timestamp,
    /// Whether active
    pub is_active: bool,
    /// Network capabilities
    pub capabilities: NetworkCapabilities,
    /// Trust requirements for joining
    pub trust_requirements: TrustRequirements,
}

/// Type of federated network
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum NetworkType {
    /// Standard Holochain-native network
    HolochainNative,
    /// Bridge to SMTP servers
    SmtpBridge,
    /// Bridge to Matrix/Element
    MatrixBridge,
    /// Bridge to ActivityPub (Mastodon, etc.)
    ActivityPubBridge,
    /// Bridge to XMPP/Jabber
    XmppBridge,
    /// Custom protocol bridge
    CustomBridge { protocol: String },
    /// Organization-specific private network
    OrganizationPrivate,
    /// Public open network
    PublicOpen,
}

/// Network capabilities
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Default)]
pub struct NetworkCapabilities {
    /// Supports end-to-end encryption
    pub supports_e2e: bool,
    /// Supports post-quantum crypto
    pub supports_pqc: bool,
    /// Supports attachments
    pub supports_attachments: bool,
    /// Maximum attachment size (bytes)
    pub max_attachment_size: Option<u64>,
    /// Supports read receipts
    pub supports_read_receipts: bool,
    /// Supports threading
    pub supports_threading: bool,
    /// Supports rich text
    pub supports_rich_text: bool,
    /// Supports reactions
    pub supports_reactions: bool,
    /// Rate limits
    pub rate_limit_per_hour: Option<u32>,
    /// Custom capabilities
    pub custom: Vec<String>,
}

/// Trust requirements for network federation
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Default)]
pub struct TrustRequirements {
    /// Minimum trust score required
    pub min_trust_score: Option<f64>,
    /// Require trust attestation from network admin
    pub require_admin_attestation: bool,
    /// Require trust attestation from N existing members
    pub require_member_attestations: Option<u32>,
    /// Require specific trust categories
    pub required_categories: Vec<String>,
    /// Automatic approval for known contacts
    pub auto_approve_known_contacts: bool,
    /// Quarantine period for new senders (seconds)
    pub quarantine_period: Option<u64>,
}

/// Federation route for email delivery
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FederationRoute {
    /// Route ID
    pub route_id: String,
    /// Source network
    pub source_network: String,
    /// Source domain pattern (e.g., "*@myorg.example")
    pub source_pattern: String,
    /// Destination network
    pub dest_network: String,
    /// Destination domain pattern
    pub dest_pattern: String,
    /// Route priority (higher = preferred)
    pub priority: u32,
    /// Route type
    pub route_type: RouteType,
    /// Whether route is active
    pub is_active: bool,
    /// Created at
    pub created_at: Timestamp,
    /// Last used
    pub last_used: Option<Timestamp>,
    /// Success count
    pub success_count: u64,
    /// Failure count
    pub failure_count: u64,
}

/// Type of federation route
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum RouteType {
    /// Direct Holochain-to-Holochain
    DirectHolochain,
    /// Via bridge agent
    BridgeAgent { bridge_agent: AgentPubKey },
    /// Via relay server
    RelayServer { relay_url: String },
    /// Via gateway service
    Gateway { gateway_id: String },
    /// Multipath (try multiple routes)
    Multipath { routes: Vec<String> },
}

/// Federated message envelope
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FederatedEnvelope {
    /// Envelope ID
    pub envelope_id: String,
    /// Source network
    pub source_network: String,
    /// Source agent
    pub source_agent: String,
    /// Destination network
    pub dest_network: String,
    /// Destination agent (or domain for lookup)
    pub dest_agent: String,
    /// Encrypted payload
    pub encrypted_payload: Vec<u8>,
    /// Encryption metadata
    pub encryption_meta: EnvelopeEncryption,
    /// Envelope signature
    pub signature: Vec<u8>,
    /// Timestamp
    pub timestamp: Timestamp,
    /// TTL in seconds
    pub ttl: u64,
    /// Hop count (for loop prevention)
    pub hop_count: u8,
    /// Max hops allowed
    pub max_hops: u8,
    /// Previous hops (network IDs)
    pub previous_hops: Vec<String>,
    /// Priority
    pub priority: EnvelopePriority,
    /// Delivery status
    pub status: DeliveryStatus,
}

/// Encryption metadata for envelope
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EnvelopeEncryption {
    /// Encryption scheme
    pub scheme: EncryptionScheme,
    /// Ephemeral public key
    pub ephemeral_pubkey: Vec<u8>,
    /// Key exchange metadata
    pub key_exchange: Vec<u8>,
    /// Nonce
    pub nonce: Vec<u8>,
}

/// Supported encryption schemes
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum EncryptionScheme {
    /// X25519 + ChaCha20-Poly1305
    X25519ChaCha20,
    /// Kyber + AES-GCM (post-quantum)
    KyberAesGcm,
    /// Hybrid X25519 + Kyber
    HybridX25519Kyber,
    /// No encryption (for bridge compatibility)
    Plaintext,
    /// Custom scheme
    Custom(String),
}

/// Envelope priority
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum EnvelopePriority {
    Low,
    Normal,
    High,
    Urgent,
}

/// Delivery status
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum DeliveryStatus {
    /// Queued for delivery
    Queued,
    /// In transit
    InTransit { current_hop: String },
    /// Delivered to destination network
    Delivered { delivered_at: Timestamp },
    /// Delivered to final recipient
    DeliveredToRecipient { delivered_at: Timestamp },
    /// Failed
    Failed { error: String, failed_at: Timestamp },
    /// Bounced
    Bounced { reason: String, bounced_at: Timestamp },
    /// Expired
    Expired,
}

/// Bridge agent registration
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BridgeAgent {
    /// Agent public key
    pub agent: AgentPubKey,
    /// Bridge type
    pub bridge_type: NetworkType,
    /// Networks this bridge connects
    pub connected_networks: Vec<String>,
    /// Bridge endpoint (if applicable)
    pub endpoint: Option<String>,
    /// Registered at
    pub registered_at: Timestamp,
    /// Last heartbeat
    pub last_heartbeat: Option<Timestamp>,
    /// Whether active
    pub is_active: bool,
    /// Capacity (messages per hour)
    pub capacity: u32,
    /// Current load
    pub current_load: u32,
    /// Trust score
    pub trust_score: f64,
}

/// Domain registry entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DomainRegistration {
    /// Domain name
    pub domain: String,
    /// Owning network
    pub network_id: String,
    /// DNS verification record
    pub dns_verification: Option<String>,
    /// Admin agent
    pub admin_agent: AgentPubKey,
    /// Registered at
    pub registered_at: Timestamp,
    /// Expires at
    pub expires_at: Option<Timestamp>,
    /// Whether verified
    pub is_verified: bool,
    /// MX-equivalent routing info
    pub mail_routing: Vec<MailRouting>,
}

/// Mail routing entry (similar to MX record)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MailRouting {
    /// Priority (lower = preferred)
    pub priority: u16,
    /// Target network
    pub target_network: String,
    /// Target agent or wildcard
    pub target_agent: Option<AgentPubKey>,
    /// Weight for load balancing
    pub weight: u16,
}

/// Federation peer status
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FederationPeer {
    /// Peer network ID
    pub network_id: String,
    /// Peer DNA hash
    pub dna_hash: DnaHash,
    /// Discovery method
    pub discovered_via: DiscoveryMethod,
    /// Last seen
    pub last_seen: Timestamp,
    /// Latency (ms)
    pub latency_ms: u32,
    /// Messages exchanged
    pub messages_exchanged: u64,
    /// Trust level
    pub trust_level: PeerTrustLevel,
    /// Whether blocked
    pub is_blocked: bool,
    /// Block reason
    pub block_reason: Option<String>,
}

/// How peer was discovered
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum DiscoveryMethod {
    /// Manual configuration
    Manual,
    /// DHT discovery
    DhtDiscovery,
    /// Bootstrap node
    Bootstrap,
    /// Referral from other peer
    Referral { from: String },
    /// DNS lookup
    DnsLookup,
}

/// Trust level for federated peer
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum PeerTrustLevel {
    /// Unknown/new peer
    Unknown,
    /// Verified via cryptographic proof
    Verified,
    /// Trusted by admin
    AdminTrusted,
    /// Trusted by web of trust
    WebOfTrust { score: f64 },
    /// Blocked/untrusted
    Blocked,
}

/// Federation audit log
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FederationAuditLog {
    /// Log ID
    pub log_id: String,
    /// Action type
    pub action: FederationAction,
    /// Source network
    pub source_network: String,
    /// Destination network
    pub dest_network: Option<String>,
    /// Envelope ID (if applicable)
    pub envelope_id: Option<String>,
    /// Actor agent
    pub actor: Option<AgentPubKey>,
    /// Timestamp
    pub timestamp: Timestamp,
    /// Success
    pub success: bool,
    /// Details
    pub details: Option<String>,
}

/// Federation actions for audit
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum FederationAction {
    /// Network registered
    NetworkRegistered,
    /// Network deregistered
    NetworkDeregistered,
    /// Route created
    RouteCreated,
    /// Route updated
    RouteUpdated,
    /// Envelope sent
    EnvelopeSent,
    /// Envelope received
    EnvelopeReceived,
    /// Envelope relayed
    EnvelopeRelayed,
    /// Envelope delivered
    EnvelopeDelivered,
    /// Delivery failed
    DeliveryFailed,
    /// Bridge connected
    BridgeConnected,
    /// Bridge disconnected
    BridgeDisconnected,
    /// Peer discovered
    PeerDiscovered,
    /// Peer blocked
    PeerBlocked,
    /// Domain registered
    DomainRegistered,
    /// Domain verified
    DomainVerified,
}

/// Link types
#[hdk_link_types]
pub enum LinkTypes {
    /// Network ID -> network entry
    NetworkIdToNetwork,
    /// Domain -> domain registration
    DomainToRegistration,
    /// Network -> routes
    NetworkToRoutes,
    /// Network -> bridge agents
    NetworkToBridges,
    /// Network -> peers
    NetworkToPeers,
    /// Agent -> owned networks
    AgentToOwnedNetworks,
    /// Agent -> bridge registrations
    AgentToBridgeRegistrations,
    /// Envelope -> audit logs
    EnvelopeToAuditLogs,
    /// Network -> audit logs
    NetworkToAuditLogs,
}

/// Entry types
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(required_validations = 2)]
    FederatedNetwork(FederatedNetwork),
    #[entry_type(required_validations = 2)]
    FederationRoute(FederationRoute),
    #[entry_type(required_validations = 1)]
    FederatedEnvelope(FederatedEnvelope),
    #[entry_type(required_validations = 2)]
    BridgeAgent(BridgeAgent),
    #[entry_type(required_validations = 2)]
    DomainRegistration(DomainRegistration),
    #[entry_type(required_validations = 1)]
    FederationPeer(FederationPeer),
    #[entry_type(required_validations = 1)]
    FederationAuditLog(FederationAuditLog),
}

// ==================== VALIDATION ====================

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => {
                validate_create_entry(app_entry, action)
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_entry(
    entry: EntryTypes,
    action: Create,
) -> ExternResult<ValidateCallbackResult> {
    match entry {
        EntryTypes::FederatedNetwork(network) => validate_network(&network, &action),
        EntryTypes::FederationRoute(route) => validate_route(&route, &action),
        EntryTypes::FederatedEnvelope(envelope) => validate_envelope(&envelope, &action),
        EntryTypes::DomainRegistration(domain) => validate_domain(&domain, &action),
        EntryTypes::BridgeAgent(bridge) => validate_bridge_agent(&bridge, &action),
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_network(
    network: &FederatedNetwork,
    _action: &Create,
) -> ExternResult<ValidateCallbackResult> {
    // Network ID must not be empty
    if network.network_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Network ID cannot be empty".to_string(),
        ));
    }

    // Network ID length cap
    if network.network_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Network ID cannot exceed 256 characters".to_string(),
        ));
    }

    // Domain must not be empty
    if network.domain.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Network domain cannot be empty".to_string(),
        ));
    }

    // Domain length cap
    if network.domain.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Network domain cannot exceed 256 characters".to_string(),
        ));
    }

    // Name must not be empty
    if network.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Network name cannot be empty".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_route(
    route: &FederationRoute,
    _action: &Create,
) -> ExternResult<ValidateCallbackResult> {
    // Route ID must not be empty
    if route.route_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Route ID cannot be empty".to_string(),
        ));
    }

    // Priority cap
    if route.priority > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Route priority cannot exceed 100".to_string(),
        ));
    }

    // Pattern length caps
    if route.source_pattern.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Source pattern cannot exceed 256 characters".to_string(),
        ));
    }
    if route.dest_pattern.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Destination pattern cannot exceed 256 characters".to_string(),
        ));
    }

    // Source and dest must be different
    if route.source_network == route.dest_network
        && route.source_pattern == route.dest_pattern {
        return Ok(ValidateCallbackResult::Invalid(
            "Route source and destination cannot be identical".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_envelope(
    envelope: &FederatedEnvelope,
    _action: &Create,
) -> ExternResult<ValidateCallbackResult> {
    // Envelope ID must not be empty
    if envelope.envelope_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Envelope ID cannot be empty".to_string(),
        ));
    }

    // TTL cap (7 days)
    if envelope.ttl > 604800 {
        return Ok(ValidateCallbackResult::Invalid(
            "TTL cannot exceed 604800 seconds (7 days)".to_string(),
        ));
    }

    // Max hops cap
    if envelope.max_hops > 20 {
        return Ok(ValidateCallbackResult::Invalid(
            "Max hops cannot exceed 20".to_string(),
        ));
    }

    // Hop count must not exceed max
    if envelope.hop_count > envelope.max_hops {
        return Ok(ValidateCallbackResult::Invalid(
            "Envelope hop count exceeds maximum".to_string(),
        ));
    }

    // Loop detection: check for duplicate network IDs in previous hops
    {
        let mut seen = std::collections::HashSet::new();
        for hop in &envelope.previous_hops {
            if !seen.insert(hop) {
                return Ok(ValidateCallbackResult::Invalid(
                    "Loop detected: duplicate network in previous hops".to_string(),
                ));
            }
        }
    }

    // Must have encrypted payload or be plaintext bridge
    if envelope.encrypted_payload.is_empty()
        && envelope.encryption_meta.scheme != EncryptionScheme::Plaintext {
        return Ok(ValidateCallbackResult::Invalid(
            "Envelope must have encrypted payload".to_string(),
        ));
    }

    // Must have signature
    if envelope.signature.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Envelope must be signed".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_bridge_agent(
    bridge: &BridgeAgent,
    action: &Create,
) -> ExternResult<ValidateCallbackResult> {
    // Agent must be the author
    if bridge.agent != action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Bridge agent must match action author".to_string(),
        ));
    }

    // Trust score must be finite and in range
    if !bridge.trust_score.is_finite() || bridge.trust_score < 0.0 || bridge.trust_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Bridge trust score must be a finite number between 0.0 and 1.0".to_string(),
        ));
    }

    // Must connect at least one network
    if bridge.connected_networks.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Bridge must connect at least one network".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_domain(
    domain: &DomainRegistration,
    action: &Create,
) -> ExternResult<ValidateCallbackResult> {
    // Domain must not be empty
    if domain.domain.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Domain cannot be empty".to_string(),
        ));
    }

    // Admin must be author
    if domain.admin_agent != action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Domain admin must match author".to_string(),
        ));
    }

    // Must have at least one mail routing entry
    if domain.mail_routing.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Domain must have mail routing".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
