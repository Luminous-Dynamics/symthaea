//! Authenticated Iroh gossip transport for [`crate::SwarmMessage`].
//!
//! The socket signs every origin envelope with the Iroh endpoint key. Because
//! gossip messages can be relayed, `Message::delivered_from` is retained only
//! as the immediate forwarding neighbor; the original author is established by
//! the envelope signature.
//!
//! ## Router integration
//!
//! `iroh-gossip` does not accept incoming connections by itself. Register the
//! clone returned by [`TelepathicSocket::gossip_protocol`] on the application's
//! shared `iroh::protocol::Router` under the same gossip ALPN before calling
//! [`TelepathicSocket::run`]. This crate intentionally does not own the router,
//! allowing one endpoint to serve multiple Luminous protocols.

use crate::{DeliveryClass, MessageValidationError, SwarmMessage};
use bincode::Options;
use futures::StreamExt;
use iroh::{
    Endpoint, EndpointAddr, EndpointId, SecretKey, Signature, address_lookup::memory::MemoryLookup,
};
use iroh_gossip::{TopicId, api::GossipSender, net::Gossip};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{Mutex, RwLock, mpsc, watch};
use uuid::Uuid;

/// Version of the signed Symthaea swarm envelope.
pub const WIRE_PROTOCOL_VERSION: u16 = 2;
/// Version of the independently evolvable signed invitation schema.
pub const INVITE_PROTOCOL_VERSION: u16 = 1;
/// Hard cap applied before deserialization and broadcast.
pub const MAX_WIRE_MESSAGE_BYTES: usize = 5 * 1024 * 1024;
/// Maximum number of addresses carried in an invitation.
pub const MAX_BOOTSTRAP_PEERS: usize = 32;
/// Invitations should remain compact enough for QR, CLI, or rendezvous exchange.
pub const MAX_INVITE_BYTES: usize = 64 * 1024;
/// Maximum accepted lifetime for a signed session invitation.
pub const MAX_INVITE_TTL_MS: u64 = 7 * 24 * 60 * 60 * 1_000;
/// Default lifetime for a host invitation.
pub const DEFAULT_INVITE_TTL_MS: u64 = 24 * 60 * 60 * 1_000;
/// Recommended maximum wait for a relay-backed, internet-dialable address.
pub const DEFAULT_ONLINE_WAIT: Duration = Duration::from_secs(15);
/// Maximum lifetime accepted for any signed gossip envelope.
pub const MAX_ENVELOPE_TTL_MS: u64 = 60 * 60 * 1_000;
/// Allowed wall-clock disagreement when validating issue and expiry times.
pub const MAX_CLOCK_SKEW_MS: u64 = 2 * 60 * 1_000;
/// Default freshness window for state-like gossip.
pub const BEST_EFFORT_TTL_MS: u64 = 30 * 1_000;
/// Default freshness window for proof, law, aid, and weight-update messages.
pub const DURABLE_TTL_MS: u64 = 10 * 60 * 1_000;
/// Number of message IDs retained for duplicate suppression.
pub const DEFAULT_REPLAY_WINDOW: usize = 16_384;
/// Number of sequence values retained per signed process session.
pub const DEFAULT_SEQUENCE_WINDOW: u64 = 1_024;
/// Maximum number of author/session sequence windows retained at once.
pub const MAX_SEQUENCE_SESSIONS: usize = 16_384;
/// Maximum accepted gossip messages from one immediate neighbor per second.
pub const MAX_PEER_MESSAGES_PER_SECOND: u64 = 256;
/// Maximum accepted gossip bytes from one immediate neighbor per second.
pub const MAX_PEER_BYTES_PER_SECOND: u64 = 16 * 1024 * 1024;
/// Maximum number of immediate-neighbor rate buckets retained at once.
pub const MAX_RATE_LIMIT_PEERS: usize = 4_096;
/// Maximum number of endpoint/application identity bindings retained or loaded.
pub const MAX_IDENTITY_BINDINGS: usize = 16_384;
/// Maximum encoded size of a persisted identity book.
pub const MAX_IDENTITY_BOOK_BYTES: usize = 2 * 1024 * 1024;
/// Schema version for persisted identity bindings.
pub const IDENTITY_BOOK_VERSION: u16 = 1;

/// Signed origin envelope carried inside Iroh gossip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmEnvelope {
    pub protocol_version: u16,
    pub message_id: Uuid,
    /// Cryptographic origin identity, not merely the immediate gossip neighbor.
    pub author: EndpointId,
    /// Unique signed process/session identifier. Sequence numbers restart only
    /// when this value changes.
    pub session_id: Uuid,
    /// Monotonic counter local to the author session.
    pub sequence: u64,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    pub payload: SwarmMessage,
    pub signature: Signature,
}

#[derive(Serialize)]
struct SigningView<'a> {
    protocol_version: u16,
    author: EndpointId,
    session_id: Uuid,
    sequence: u64,
    issued_at_ms: u64,
    expires_at_ms: u64,
    payload: &'a SwarmMessage,
}

impl SwarmEnvelope {
    pub fn sign(
        payload: SwarmMessage,
        secret_key: &SecretKey,
        session_id: Uuid,
        sequence: u64,
        issued_at_ms: u64,
        ttl_ms: u64,
    ) -> Result<Self, EnvelopeError> {
        payload.validate().map_err(EnvelopeError::InvalidPayload)?;
        if ttl_ms == 0 || ttl_ms > MAX_ENVELOPE_TTL_MS {
            return Err(EnvelopeError::InvalidTtl { ttl_ms });
        }
        let expires_at_ms = issued_at_ms
            .checked_add(ttl_ms)
            .ok_or(EnvelopeError::TimestampOverflow)?;
        let author = secret_key.public();
        if session_id.is_nil() {
            return Err(EnvelopeError::NilSessionId);
        }
        if sequence == 0 {
            return Err(EnvelopeError::ZeroSequence);
        }
        let signature = secret_key.sign(&signing_bytes(
            WIRE_PROTOCOL_VERSION,
            author,
            session_id,
            sequence,
            issued_at_ms,
            expires_at_ms,
            &payload,
        )?);
        let message_id = message_id_from_signature(&signature);

        Ok(Self {
            protocol_version: WIRE_PROTOCOL_VERSION,
            message_id,
            author,
            session_id,
            sequence,
            issued_at_ms,
            expires_at_ms,
            payload,
            signature,
        })
    }

    pub fn verify_at(&self, now_ms: u64) -> Result<(), EnvelopeError> {
        if self.protocol_version != WIRE_PROTOCOL_VERSION {
            return Err(EnvelopeError::UnsupportedVersion {
                received: self.protocol_version,
            });
        }
        if self.session_id.is_nil() {
            return Err(EnvelopeError::NilSessionId);
        }
        if self.sequence == 0 {
            return Err(EnvelopeError::ZeroSequence);
        }
        if self.expires_at_ms < self.issued_at_ms {
            return Err(EnvelopeError::InvalidTtl { ttl_ms: 0 });
        }
        let ttl_ms = self.expires_at_ms - self.issued_at_ms;
        if ttl_ms == 0 || ttl_ms > MAX_ENVELOPE_TTL_MS {
            return Err(EnvelopeError::InvalidTtl { ttl_ms });
        }
        if self.issued_at_ms > now_ms.saturating_add(MAX_CLOCK_SKEW_MS) {
            return Err(EnvelopeError::IssuedInFuture {
                issued_at_ms: self.issued_at_ms,
                now_ms,
            });
        }
        if now_ms > self.expires_at_ms.saturating_add(MAX_CLOCK_SKEW_MS) {
            return Err(EnvelopeError::Expired {
                expires_at_ms: self.expires_at_ms,
                now_ms,
            });
        }
        self.payload
            .validate()
            .map_err(EnvelopeError::InvalidPayload)?;

        let bytes = signing_bytes(
            self.protocol_version,
            self.author,
            self.session_id,
            self.sequence,
            self.issued_at_ms,
            self.expires_at_ms,
            &self.payload,
        )?;
        self.author
            .verify(&bytes, &self.signature)
            .map_err(|_| EnvelopeError::InvalidSignature)?;
        if self.message_id != message_id_from_signature(&self.signature) {
            return Err(EnvelopeError::InvalidMessageId);
        }
        Ok(())
    }
}

fn signing_bytes(
    protocol_version: u16,
    author: EndpointId,
    session_id: Uuid,
    sequence: u64,
    issued_at_ms: u64,
    expires_at_ms: u64,
    payload: &SwarmMessage,
) -> Result<Vec<u8>, EnvelopeError> {
    encode_bounded(
        &SigningView {
            protocol_version,
            author,
            session_id,
            sequence,
            issued_at_ms,
            expires_at_ms,
            payload,
        },
        MAX_WIRE_MESSAGE_BYTES,
    )
    .map_err(EnvelopeError::Codec)
}

pub(crate) fn message_id_from_signature(signature: &Signature) -> Uuid {
    let signature_bytes = signature.to_bytes();
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&signature_bytes[..16]);
    // Mark as an RFC 9562 version-8 (application-defined) UUID. The 128-bit
    // value is derived from the verified deterministic Ed25519 signature.
    bytes[6] = (bytes[6] & 0x0f) | 0x80;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Uuid::from_bytes(bytes)
}

static SESSION_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Derive a process-session identifier from endpoint-authenticated entropy.
///
/// The value is not a credential, but deriving it from a fresh signed context
/// prevents predictable time/PID-only identifiers and makes accidental reuse
/// across process restarts negligibly likely.
pub(crate) fn new_session_id(secret_key: &SecretKey) -> Uuid {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let counter = SESSION_COUNTER.fetch_add(1, Ordering::Relaxed);
    let process_id = u64::from(std::process::id());

    let mut context = Vec::with_capacity(32 + 32 + 16 + 8 + 8);
    context.extend_from_slice(b"symthaea-swarm/session/v1\0");
    context.extend_from_slice(secret_key.public().as_bytes());
    context.extend_from_slice(&nanos.to_le_bytes());
    context.extend_from_slice(&counter.to_le_bytes());
    context.extend_from_slice(&process_id.to_le_bytes());
    message_id_from_signature(&secret_key.sign(&context))
}

/// Portable, signed session invitation. Endpoint addresses are retained so a
/// caller's address-lookup layer can import them; gossip itself joins by
/// endpoint ID. The issuer signature prevents a rendezvous channel from
/// silently rewriting the topic or bootstrap set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmInvite {
    pub protocol_version: u16,
    pub topic: [u8; 32],
    pub issuer: EndpointId,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    pub bootstrap: Vec<EndpointAddr>,
    pub signature: Signature,
}

#[derive(Serialize)]
struct InviteSigningView<'a> {
    protocol_version: u16,
    topic: [u8; 32],
    issuer: EndpointId,
    issued_at_ms: u64,
    expires_at_ms: u64,
    bootstrap: &'a [EndpointAddr],
}

impl SwarmInvite {
    pub fn sign(
        topic: [u8; 32],
        bootstrap: Vec<EndpointAddr>,
        secret_key: &SecretKey,
        issued_at_ms: u64,
        ttl_ms: u64,
    ) -> Result<Self, InviteError> {
        if ttl_ms == 0 || ttl_ms > MAX_INVITE_TTL_MS {
            return Err(InviteError::InvalidTtl { ttl_ms });
        }
        let expires_at_ms = issued_at_ms
            .checked_add(ttl_ms)
            .ok_or(InviteError::TimestampOverflow)?;
        let issuer = secret_key.public();
        let signature = secret_key.sign(&invite_signing_bytes(
            INVITE_PROTOCOL_VERSION,
            topic,
            issuer,
            issued_at_ms,
            expires_at_ms,
            &bootstrap,
        )?);
        let invite = Self {
            protocol_version: INVITE_PROTOCOL_VERSION,
            topic,
            issuer,
            issued_at_ms,
            expires_at_ms,
            bootstrap,
            signature,
        };
        invite.verify_at(issued_at_ms)?;
        Ok(invite)
    }

    pub fn host(
        topic: [u8; 32],
        host: EndpointAddr,
        secret_key: &SecretKey,
        issued_at_ms: u64,
    ) -> Result<Self, InviteError> {
        Self::sign(
            topic,
            vec![host],
            secret_key,
            issued_at_ms,
            DEFAULT_INVITE_TTL_MS,
        )
    }

    pub fn verify_now(&self) -> Result<(), InviteError> {
        self.verify_at(system_time_ms().map_err(InviteError::Clock)?)
    }

    pub fn verify_at(&self, now_ms: u64) -> Result<(), InviteError> {
        if self.protocol_version != INVITE_PROTOCOL_VERSION {
            return Err(InviteError::UnsupportedVersion {
                received: self.protocol_version,
            });
        }
        if self.bootstrap.is_empty() {
            return Err(InviteError::EmptyBootstrap);
        }
        if self.bootstrap.len() > MAX_BOOTSTRAP_PEERS {
            return Err(InviteError::TooManyBootstrapPeers {
                count: self.bootstrap.len(),
            });
        }
        let mut ids = HashSet::new();
        if self.bootstrap.iter().any(|peer| !ids.insert(peer.id)) {
            return Err(InviteError::DuplicateBootstrapPeer);
        }
        if !ids.contains(&self.issuer) {
            return Err(InviteError::IssuerNotBootstrap);
        }
        if self.expires_at_ms < self.issued_at_ms {
            return Err(InviteError::InvalidTtl { ttl_ms: 0 });
        }
        let ttl_ms = self.expires_at_ms - self.issued_at_ms;
        if ttl_ms == 0 || ttl_ms > MAX_INVITE_TTL_MS {
            return Err(InviteError::InvalidTtl { ttl_ms });
        }
        if self.issued_at_ms > now_ms.saturating_add(MAX_CLOCK_SKEW_MS) {
            return Err(InviteError::IssuedInFuture {
                issued_at_ms: self.issued_at_ms,
                now_ms,
            });
        }
        if now_ms > self.expires_at_ms.saturating_add(MAX_CLOCK_SKEW_MS) {
            return Err(InviteError::Expired {
                expires_at_ms: self.expires_at_ms,
                now_ms,
            });
        }
        let bytes = invite_signing_bytes(
            self.protocol_version,
            self.topic,
            self.issuer,
            self.issued_at_ms,
            self.expires_at_ms,
            &self.bootstrap,
        )?;
        self.issuer
            .verify(&bytes, &self.signature)
            .map_err(|_| InviteError::InvalidSignature)
    }

    pub fn bootstrap_ids(&self) -> Vec<EndpointId> {
        self.bootstrap.iter().map(|peer| peer.id).collect()
    }

    /// Import the invitation's out-of-band endpoint addresses into the same
    /// [`MemoryLookup`] that was installed on the local [`Endpoint`] builder.
    ///
    /// Importing only endpoint IDs is insufficient when no DNS/pkarr lookup is
    /// available: Iroh also needs at least one usable relay or direct address.
    pub fn import_addresses(&self, address_lookup: &MemoryLookup) -> Result<(), InviteError> {
        self.verify_now()?;
        for endpoint_addr in &self.bootstrap {
            address_lookup.add_endpoint_info(endpoint_addr.clone());
        }
        Ok(())
    }

    pub fn encode(&self) -> Result<Vec<u8>, InviteError> {
        self.verify_now()?;
        let bytes = encode_bounded(self, MAX_INVITE_BYTES).map_err(InviteError::Codec)?;
        if bytes.len() > MAX_INVITE_BYTES {
            return Err(InviteError::TooLarge { size: bytes.len() });
        }
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, InviteError> {
        Self::decode_at(bytes, system_time_ms().map_err(InviteError::Clock)?)
    }

    pub fn decode_at(bytes: &[u8], now_ms: u64) -> Result<Self, InviteError> {
        if bytes.len() > MAX_INVITE_BYTES {
            return Err(InviteError::TooLarge { size: bytes.len() });
        }
        let invite: Self = decode_bounded(bytes, MAX_INVITE_BYTES).map_err(InviteError::Codec)?;
        invite.verify_at(now_ms)?;
        Ok(invite)
    }
}

fn invite_signing_bytes(
    protocol_version: u16,
    topic: [u8; 32],
    issuer: EndpointId,
    issued_at_ms: u64,
    expires_at_ms: u64,
    bootstrap: &[EndpointAddr],
) -> Result<Vec<u8>, InviteError> {
    encode_bounded(
        &InviteSigningView {
            protocol_version,
            topic,
            issuer,
            issued_at_ms,
            expires_at_ms,
            bootstrap,
        },
        MAX_INVITE_BYTES,
    )
    .map_err(InviteError::Codec)
}

#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum InviteError {
    #[error("unsupported invite protocol version {received}")]
    UnsupportedVersion { received: u16 },
    #[error("invite must contain at least one bootstrap endpoint")]
    EmptyBootstrap,
    #[error("invite issuer is not present in the bootstrap endpoint set")]
    IssuerNotBootstrap,
    #[error("invite contains {count} bootstrap peers; maximum is {MAX_BOOTSTRAP_PEERS}")]
    TooManyBootstrapPeers { count: usize },
    #[error("invite contains the same bootstrap endpoint more than once")]
    DuplicateBootstrapPeer,
    #[error("invalid invite TTL: {ttl_ms} ms")]
    InvalidTtl { ttl_ms: u64 },
    #[error("invite timestamp arithmetic overflow")]
    TimestampOverflow,
    #[error("invite issue time {issued_at_ms} is too far ahead of local time {now_ms}")]
    IssuedInFuture { issued_at_ms: u64, now_ms: u64 },
    #[error("invite expired at {expires_at_ms}; local time is {now_ms}")]
    Expired { expires_at_ms: u64, now_ms: u64 },
    #[error("invalid invite signature")]
    InvalidSignature,
    #[error("invite is too large: {size} bytes")]
    TooLarge { size: usize },
    #[error("invite codec error: {0}")]
    Codec(String),
    #[error("system clock error: {0}")]
    Clock(String),
}

#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum EnvelopeError {
    #[error("unsupported swarm protocol version {received}")]
    UnsupportedVersion { received: u16 },
    #[error("invalid envelope TTL: {ttl_ms} ms")]
    InvalidTtl { ttl_ms: u64 },
    #[error("timestamp arithmetic overflow")]
    TimestampOverflow,
    #[error("session ID must not be nil")]
    NilSessionId,
    #[error("sequence number zero is reserved")]
    ZeroSequence,
    #[error("envelope issue time {issued_at_ms} is too far ahead of local time {now_ms}")]
    IssuedInFuture { issued_at_ms: u64, now_ms: u64 },
    #[error("envelope expired at {expires_at_ms}; local time is {now_ms}")]
    Expired { expires_at_ms: u64, now_ms: u64 },
    #[error("message ID does not match the verified envelope signature")]
    InvalidMessageId,
    #[error("invalid envelope signature")]
    InvalidSignature,
    #[error("invalid payload: {0}")]
    InvalidPayload(MessageValidationError),
    #[error("envelope codec error: {0}")]
    Codec(String),
}

/// Policy used when an authenticated endpoint claims an application UUID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityPolicy {
    /// Persist the first valid one-to-one mapping and reject later conflicts.
    TrustOnFirstUse,
    /// Accept only mappings explicitly loaded or enrolled by the application.
    PinnedOnly,
}

impl Default for IdentityPolicy {
    fn default() -> Self {
        Self::TrustOnFirstUse
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdentityBinding {
    pub endpoint: EndpointId,
    pub node_id: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IdentityBookSnapshot {
    version: u16,
    bindings: Vec<IdentityBinding>,
}

/// One-to-one binding between application UUIDs and signed Iroh endpoint
/// identities. The book can operate as TOFU or as a fail-closed pinned ledger.
#[derive(Debug, Default, Clone)]
pub struct IdentityBook {
    endpoint_to_node: HashMap<EndpointId, Uuid>,
    node_to_endpoint: HashMap<Uuid, EndpointId>,
}

impl IdentityBook {
    pub fn observe(
        &mut self,
        endpoint: EndpointId,
        node_id: Uuid,
    ) -> Result<(), IdentityBindingError> {
        self.verify_or_observe(endpoint, node_id, IdentityPolicy::TrustOnFirstUse)
    }

    pub fn verify_or_observe(
        &mut self,
        endpoint: EndpointId,
        node_id: Uuid,
        policy: IdentityPolicy,
    ) -> Result<(), IdentityBindingError> {
        if node_id.is_nil() {
            return Err(IdentityBindingError::NilNodeId);
        }
        if let Some(existing) = self.endpoint_to_node.get(&endpoint) {
            if *existing != node_id {
                return Err(IdentityBindingError::EndpointChangedNode {
                    endpoint,
                    expected: *existing,
                    claimed: node_id,
                });
            }
        }
        if let Some(existing) = self.node_to_endpoint.get(&node_id) {
            if *existing != endpoint {
                return Err(IdentityBindingError::NodeChangedEndpoint {
                    node_id,
                    expected: *existing,
                    claimed: endpoint,
                });
            }
        }

        let known = self.endpoint_to_node.contains_key(&endpoint)
            && self.node_to_endpoint.contains_key(&node_id);
        if !known && policy == IdentityPolicy::PinnedOnly {
            return Err(IdentityBindingError::UnknownIdentity { endpoint, node_id });
        }
        if !known && self.len() >= MAX_IDENTITY_BINDINGS {
            return Err(IdentityBindingError::CapacityExceeded);
        }
        self.endpoint_to_node.insert(endpoint, node_id);
        self.node_to_endpoint.insert(node_id, endpoint);
        Ok(())
    }

    /// Explicitly add a binding regardless of the runtime verification policy.
    pub fn enroll(
        &mut self,
        endpoint: EndpointId,
        node_id: Uuid,
    ) -> Result<(), IdentityBindingError> {
        self.observe(endpoint, node_id)
    }

    pub fn endpoint_for(&self, node_id: Uuid) -> Option<EndpointId> {
        self.node_to_endpoint.get(&node_id).copied()
    }

    pub fn node_for(&self, endpoint: EndpointId) -> Option<Uuid> {
        self.endpoint_to_node.get(&endpoint).copied()
    }

    pub fn bindings(&self) -> Vec<IdentityBinding> {
        let mut bindings = self
            .endpoint_to_node
            .iter()
            .map(|(endpoint, node_id)| IdentityBinding {
                endpoint: *endpoint,
                node_id: *node_id,
            })
            .collect::<Vec<_>>();
        bindings.sort_by_key(|binding| binding.endpoint.to_string());
        bindings
    }

    pub fn from_bindings(
        bindings: impl IntoIterator<Item = IdentityBinding>,
    ) -> Result<Self, IdentityBookError> {
        let mut book = Self::default();
        for binding in bindings {
            book.enroll(binding.endpoint, binding.node_id)
                .map_err(IdentityBookError::Binding)?;
        }
        Ok(book)
    }

    pub fn encode(&self) -> Result<Vec<u8>, IdentityBookError> {
        let snapshot = IdentityBookSnapshot {
            version: IDENTITY_BOOK_VERSION,
            bindings: self.bindings(),
        };
        encode_bounded(&snapshot, MAX_IDENTITY_BOOK_BYTES).map_err(IdentityBookError::Codec)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, IdentityBookError> {
        if bytes.len() > MAX_IDENTITY_BOOK_BYTES {
            return Err(IdentityBookError::TooLarge { size: bytes.len() });
        }
        let snapshot: IdentityBookSnapshot =
            decode_bounded(bytes, MAX_IDENTITY_BOOK_BYTES).map_err(IdentityBookError::Codec)?;
        if snapshot.version != IDENTITY_BOOK_VERSION {
            return Err(IdentityBookError::UnsupportedVersion {
                received: snapshot.version,
            });
        }
        if snapshot.bindings.len() > MAX_IDENTITY_BINDINGS {
            return Err(IdentityBookError::TooManyBindings {
                count: snapshot.bindings.len(),
            });
        }
        Self::from_bindings(snapshot.bindings)
    }

    pub fn len(&self) -> usize {
        self.endpoint_to_node.len()
    }

    pub fn is_empty(&self) -> bool {
        self.endpoint_to_node.is_empty()
    }
}

#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum IdentityBindingError {
    #[error("nil application node IDs are not accepted on the network")]
    NilNodeId,
    #[error("identity book reached its maximum capacity")]
    CapacityExceeded,
    #[error("endpoint {endpoint} and node {node_id} are not enrolled")]
    UnknownIdentity { endpoint: EndpointId, node_id: Uuid },
    #[error("endpoint {endpoint} was bound to {expected}, but claimed {claimed}")]
    EndpointChangedNode {
        endpoint: EndpointId,
        expected: Uuid,
        claimed: Uuid,
    },
    #[error("node {node_id} was bound to {expected}, but claimed endpoint {claimed}")]
    NodeChangedEndpoint {
        node_id: Uuid,
        expected: EndpointId,
        claimed: EndpointId,
    },
}

#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum IdentityBookError {
    #[error("unsupported identity-book version {received}")]
    UnsupportedVersion { received: u16 },
    #[error("identity book contains {count} bindings; maximum is {MAX_IDENTITY_BINDINGS}")]
    TooManyBindings { count: usize },
    #[error("identity book is too large: {size} bytes")]
    TooLarge { size: usize },
    #[error("identity binding is invalid: {0}")]
    Binding(IdentityBindingError),
    #[error("identity-book codec error: {0}")]
    Codec(String),
}

#[derive(Debug)]
pub(crate) struct ReplayWindow {
    capacity: usize,
    order: VecDeque<Uuid>,
    seen: HashSet<Uuid>,
}

impl ReplayWindow {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            order: VecDeque::new(),
            seen: HashSet::new(),
        }
    }

    /// Returns true only for a newly observed ID.
    pub(crate) fn insert(&mut self, message_id: Uuid) -> bool {
        if !self.seen.insert(message_id) {
            return false;
        }
        self.order.push_back(message_id);
        while self.order.len() > self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.seen.remove(&oldest);
            }
        }
        true
    }
}

#[derive(Debug)]
struct SequenceWindow {
    width: u64,
    highest: u64,
    seen: HashSet<u64>,
}

impl SequenceWindow {
    fn new(width: u64) -> Self {
        Self {
            width: width.max(1),
            highest: 0,
            seen: HashSet::new(),
        }
    }

    /// Accept a sequence once while allowing bounded network reordering.
    fn insert(&mut self, sequence: u64) -> bool {
        if sequence == 0 {
            return false;
        }
        if self.highest > sequence && self.highest - sequence >= self.width {
            return false;
        }
        if !self.seen.insert(sequence) {
            return false;
        }
        self.highest = self.highest.max(sequence);
        let minimum = self.highest.saturating_sub(self.width - 1);
        self.seen.retain(|value| *value >= minimum);
        true
    }
}

#[derive(Debug)]
pub(crate) struct AuthorSequenceWindows {
    capacity: usize,
    width: u64,
    order: VecDeque<(EndpointId, Uuid)>,
    windows: HashMap<(EndpointId, Uuid), SequenceWindow>,
}

impl AuthorSequenceWindows {
    pub(crate) fn new(capacity: usize, width: u64) -> Self {
        Self {
            capacity: capacity.max(1),
            width: width.max(1),
            order: VecDeque::new(),
            windows: HashMap::new(),
        }
    }

    pub(crate) fn insert(&mut self, author: EndpointId, session_id: Uuid, sequence: u64) -> bool {
        let key = (author, session_id);
        if !self.windows.contains_key(&key) {
            while self.windows.len() >= self.capacity {
                if let Some(oldest) = self.order.pop_front() {
                    self.windows.remove(&oldest);
                } else {
                    break;
                }
            }
            self.order.push_back(key);
            self.windows.insert(key, SequenceWindow::new(self.width));
        }
        self.windows
            .get_mut(&key)
            .map(|window| window.insert(sequence))
            .unwrap_or(false)
    }
}

/// Explicit truth boundary for the current transport implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwarmTransportCapabilities {
    pub signed_origins: bool,
    pub expiring_invites: bool,
    pub gossip_broadcast: bool,
    pub authenticated_direct_streams: bool,
    pub unreliable_datagrams: bool,
    pub end_to_end_acknowledgements: bool,
}

impl SwarmTransportCapabilities {
    pub const CONTROL_PLANE_V2: Self = Self {
        signed_origins: true,
        expiring_invites: true,
        gossip_broadcast: true,
        authenticated_direct_streams: false,
        unreliable_datagrams: false,
        end_to_end_acknowledgements: false,
    };
}

/// Local acceptance receipt for an outbound gossip message.
///
/// This proves that the signed envelope was accepted by the local Iroh gossip
/// actor. It is not proof that any remote peer received, persisted, or applied
/// the message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BroadcastReceipt {
    pub author: EndpointId,
    pub session_id: Uuid,
    pub sequence: u64,
    pub message_id: Uuid,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    pub encoded_bytes: usize,
}

/// Message delivered to the application after signature, expiry, replay, and
/// identity-binding checks.
#[derive(Debug, Clone)]
pub struct AuthenticatedSwarmMessage {
    pub author: EndpointId,
    /// Immediate gossip neighbor. This is not necessarily the original author.
    pub delivered_from: EndpointId,
    pub message_id: Uuid,
    pub session_id: Uuid,
    pub sequence: u64,
    pub issued_at_ms: u64,
    pub payload: SwarmMessage,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SocketState {
    Created,
    Subscribing,
    Active { neighbors: usize },
    Stopping,
    Stopped,
    Failed(String),
}

#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum RejectionReason {
    #[error("wire message is too large: {size} bytes")]
    MessageTooLarge { size: usize },
    #[error("immediate neighbor exceeded the receive rate limit with a {size}-byte message")]
    RateLimited { size: usize },
    #[error("envelope decode failed: {detail}")]
    Decode { detail: String },
    #[error("invalid signed envelope: {0}")]
    InvalidEnvelope(EnvelopeError),
    #[error("sequence {sequence} in session {session_id} was replayed or is too old")]
    SequenceReplay { session_id: Uuid, sequence: u64 },
    #[error("application identity binding failed: {0}")]
    IdentityBinding(IdentityBindingError),
}

#[derive(Debug, Clone)]
pub enum SwarmNetworkEvent {
    NeighborUp(EndpointId),
    NeighborDown(EndpointId),
    Message(AuthenticatedSwarmMessage),
    Lagged,
    Rejected {
        delivered_from: EndpointId,
        reason: RejectionReason,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateLimitConfig {
    pub max_peers: usize,
    pub window: Duration,
    pub max_messages_per_peer: u64,
    pub max_bytes_per_peer: u64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_peers: MAX_RATE_LIMIT_PEERS,
            window: Duration::from_secs(1),
            max_messages_per_peer: MAX_PEER_MESSAGES_PER_SECOND,
            max_bytes_per_peer: MAX_PEER_BYTES_PER_SECOND,
        }
    }
}

impl RateLimitConfig {
    pub(crate) fn validate(self) -> Result<Self, SwarmNetworkError> {
        if self.max_peers == 0 {
            return Err(SwarmNetworkError::InvalidRateLimitConfig(
                "max_peers must be greater than zero",
            ));
        }
        if self.window.is_zero() {
            return Err(SwarmNetworkError::InvalidRateLimitConfig(
                "window must be greater than zero",
            ));
        }
        if self.max_messages_per_peer == 0 {
            return Err(SwarmNetworkError::InvalidRateLimitConfig(
                "max_messages_per_peer must be greater than zero",
            ));
        }
        if self.max_bytes_per_peer == 0 {
            return Err(SwarmNetworkError::InvalidRateLimitConfig(
                "max_bytes_per_peer must be greater than zero",
            ));
        }
        Ok(self)
    }
}

#[derive(Debug)]
struct PeerRateState {
    window_started: Instant,
    messages: u64,
    bytes: u64,
}

#[derive(Debug)]
pub(crate) struct PeerRateLimiter {
    capacity: usize,
    window: Duration,
    max_messages: u64,
    max_bytes: u64,
    peers: HashMap<EndpointId, PeerRateState>,
}

impl PeerRateLimiter {
    pub(crate) fn new(config: RateLimitConfig) -> Self {
        Self {
            capacity: config.max_peers,
            window: config.window,
            max_messages: config.max_messages_per_peer,
            max_bytes: config.max_bytes_per_peer,
            peers: HashMap::new(),
        }
    }

    pub(crate) fn allow(&mut self, peer: EndpointId, size: usize, now: Instant) -> bool {
        if !self.peers.contains_key(&peer) {
            self.peers
                .retain(|_, state| now.duration_since(state.window_started) < self.window);
            if self.peers.len() >= self.capacity {
                return false;
            }
            self.peers.insert(
                peer,
                PeerRateState {
                    window_started: now,
                    messages: 0,
                    bytes: 0,
                },
            );
        }

        let Some(state) = self.peers.get_mut(&peer) else {
            return false;
        };
        if now.duration_since(state.window_started) >= self.window {
            state.window_started = now;
            state.messages = 0;
            state.bytes = 0;
        }
        let next_messages = state.messages.saturating_add(1);
        let next_bytes = state.bytes.saturating_add(size as u64);
        if next_messages > self.max_messages || next_bytes > self.max_bytes {
            return false;
        }
        state.messages = next_messages;
        state.bytes = next_bytes;
        true
    }
}

#[derive(Debug, Default)]
struct SocketMetrics {
    received: AtomicU64,
    bytes_received: AtomicU64,
    accepted: AtomicU64,
    rejected: AtomicU64,
    duplicate: AtomicU64,
    sequence_replay: AtomicU64,
    rate_limited: AtomicU64,
    sent: AtomicU64,
    bytes_sent: AtomicU64,
    send_errors: AtomicU64,
    durable_queue_full: AtomicU64,
    best_effort_dropped: AtomicU64,
    gossip_lagged: AtomicU64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SocketMetricsSnapshot {
    pub received: u64,
    pub bytes_received: u64,
    pub accepted: u64,
    pub rejected: u64,
    pub duplicate: u64,
    pub sequence_replay: u64,
    pub rate_limited: u64,
    pub sent: u64,
    pub bytes_sent: u64,
    pub send_errors: u64,
    pub durable_queue_full: u64,
    pub best_effort_dropped: u64,
    pub gossip_lagged: u64,
}

impl SocketMetrics {
    fn snapshot(&self) -> SocketMetricsSnapshot {
        SocketMetricsSnapshot {
            received: self.received.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            accepted: self.accepted.load(Ordering::Relaxed),
            rejected: self.rejected.load(Ordering::Relaxed),
            duplicate: self.duplicate.load(Ordering::Relaxed),
            sequence_replay: self.sequence_replay.load(Ordering::Relaxed),
            rate_limited: self.rate_limited.load(Ordering::Relaxed),
            sent: self.sent.load(Ordering::Relaxed),
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            send_errors: self.send_errors.load(Ordering::Relaxed),
            durable_queue_full: self.durable_queue_full.load(Ordering::Relaxed),
            best_effort_dropped: self.best_effort_dropped.load(Ordering::Relaxed),
            gossip_lagged: self.gossip_lagged.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone)]
enum InboundSink {
    Legacy(mpsc::Sender<SwarmMessage>),
    Events(mpsc::Sender<SwarmNetworkEvent>),
}

/// Iroh-backed, signed gossip socket.
#[derive(Clone)]
pub struct TelepathicSocket {
    endpoint: Endpoint,
    gossip: Gossip,
    topic_id: TopicId,
    topic_raw: [u8; 32],
    inbound: InboundSink,
    bootstrap: Arc<RwLock<Vec<EndpointId>>>,
    sender: Arc<Mutex<Option<GossipSender>>>,
    state_tx: watch::Sender<SocketState>,
    shutdown_tx: watch::Sender<bool>,
    config_gate: Arc<Mutex<()>>,
    run_claimed: Arc<AtomicBool>,
    session_id: Uuid,
    sequence: Arc<AtomicU64>,
    replay: Arc<Mutex<ReplayWindow>>,
    sequence_windows: Arc<Mutex<AuthorSequenceWindows>>,
    identities: Arc<Mutex<IdentityBook>>,
    identity_policy: Arc<RwLock<IdentityPolicy>>,
    rate_limiter: Arc<Mutex<PeerRateLimiter>>,
    metrics: Arc<SocketMetrics>,
}

impl TelepathicSocket {
    /// Legacy constructor retaining the old message-only receive API.
    ///
    /// Messages are still signature-verified, but the application does not
    /// receive author or lifecycle metadata. New code should use
    /// [`Self::new_authenticated`].
    #[deprecated(note = "use TelepathicSocket::new_authenticated to retain authenticated metadata")]
    pub async fn new(
        endpoint: Endpoint,
        topic_raw: [u8; 32],
        inbound_tx: mpsc::Sender<SwarmMessage>,
    ) -> Result<Self, SwarmNetworkError> {
        Ok(Self::build(
            endpoint,
            topic_raw,
            InboundSink::Legacy(inbound_tx),
        ))
    }

    pub async fn new_authenticated(
        endpoint: Endpoint,
        topic_raw: [u8; 32],
        inbound_tx: mpsc::Sender<SwarmNetworkEvent>,
    ) -> Result<Self, SwarmNetworkError> {
        Ok(Self::build(
            endpoint,
            topic_raw,
            InboundSink::Events(inbound_tx),
        ))
    }

    /// Construct a socket from an invitation and import its endpoint addresses.
    ///
    /// `address_lookup` must be the same [`MemoryLookup`] clone installed with
    /// `Endpoint::builder(...).address_lookup(address_lookup.clone())` before
    /// the endpoint was bound.
    pub async fn from_invite_authenticated(
        endpoint: Endpoint,
        address_lookup: &MemoryLookup,
        invite: SwarmInvite,
        inbound_tx: mpsc::Sender<SwarmNetworkEvent>,
    ) -> Result<Self, SwarmNetworkError> {
        invite
            .import_addresses(address_lookup)
            .map_err(SwarmNetworkError::InvalidInvite)?;
        let bootstrap = invite.bootstrap_ids();
        let socket = Self::build(endpoint, invite.topic, InboundSink::Events(inbound_tx));
        socket.set_bootstrap_peers(bootstrap).await?;
        Ok(socket)
    }

    fn build(endpoint: Endpoint, topic_raw: [u8; 32], inbound: InboundSink) -> Self {
        let gossip = Gossip::builder()
            .max_message_size(MAX_WIRE_MESSAGE_BYTES)
            .spawn(endpoint.clone());
        let (state_tx, _) = watch::channel(SocketState::Created);
        let (shutdown_tx, _) = watch::channel(false);
        let session_id = new_session_id(endpoint.secret_key());
        Self {
            endpoint,
            gossip,
            topic_id: TopicId::from(topic_raw),
            topic_raw,
            inbound,
            bootstrap: Arc::new(RwLock::new(Vec::new())),
            sender: Arc::new(Mutex::new(None)),
            state_tx,
            shutdown_tx,
            config_gate: Arc::new(Mutex::new(())),
            run_claimed: Arc::new(AtomicBool::new(false)),
            session_id,
            sequence: Arc::new(AtomicU64::new(1)),
            replay: Arc::new(Mutex::new(ReplayWindow::new(DEFAULT_REPLAY_WINDOW))),
            sequence_windows: Arc::new(Mutex::new(AuthorSequenceWindows::new(
                MAX_SEQUENCE_SESSIONS,
                DEFAULT_SEQUENCE_WINDOW,
            ))),
            identities: Arc::new(Mutex::new(IdentityBook::default())),
            identity_policy: Arc::new(RwLock::new(IdentityPolicy::default())),
            rate_limiter: Arc::new(Mutex::new(PeerRateLimiter::new(RateLimitConfig::default()))),
            metrics: Arc::new(SocketMetrics::default()),
        }
    }

    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    pub fn node_id(&self) -> EndpointId {
        self.endpoint.id()
    }

    pub fn topic_id(&self) -> TopicId {
        self.topic_id
    }

    pub fn session_id(&self) -> Uuid {
        self.session_id
    }

    /// Clone this handler into the application's shared Iroh router.
    pub fn gossip_protocol(&self) -> Gossip {
        self.gossip.clone()
    }

    pub fn state(&self) -> watch::Receiver<SocketState> {
        self.state_tx.subscribe()
    }

    pub fn current_state(&self) -> SocketState {
        self.state_tx.borrow().clone()
    }

    pub fn metrics(&self) -> SocketMetricsSnapshot {
        self.metrics.snapshot()
    }

    pub const fn capabilities(&self) -> SwarmTransportCapabilities {
        SwarmTransportCapabilities::CONTROL_PLANE_V2
    }

    pub async fn wait_for_neighbors(
        &self,
        minimum: usize,
        timeout: Duration,
    ) -> Result<(), SwarmNetworkError> {
        if minimum == 0 {
            return Ok(());
        }
        let mut state_rx = self.state();
        let wait = async {
            loop {
                let state = state_rx.borrow().clone();
                match state {
                    SocketState::Active { neighbors } if neighbors >= minimum => return Ok(()),
                    SocketState::Failed(reason) => {
                        return Err(SwarmNetworkError::SocketFailed(reason));
                    }
                    SocketState::Stopped | SocketState::Stopping => {
                        return Err(SwarmNetworkError::SocketStopped);
                    }
                    _ => {}
                }
                state_rx
                    .changed()
                    .await
                    .map_err(|_| SwarmNetworkError::StateChannelClosed)?;
            }
        };
        tokio::time::timeout(timeout, wait)
            .await
            .map_err(|_| SwarmNetworkError::NeighborWaitTimeout { minimum })?
    }

    pub async fn identity_book(&self) -> IdentityBook {
        self.identities.lock().await.clone()
    }

    pub async fn identity_policy(&self) -> IdentityPolicy {
        *self.identity_policy.read().await
    }

    pub async fn set_identity_policy(
        &self,
        policy: IdentityPolicy,
    ) -> Result<(), SwarmNetworkError> {
        let _config_guard = self.config_gate.lock().await;
        self.ensure_configurable()?;
        *self.identity_policy.write().await = policy;
        Ok(())
    }

    pub async fn replace_identity_book(&self, book: IdentityBook) -> Result<(), SwarmNetworkError> {
        let _config_guard = self.config_gate.lock().await;
        self.ensure_configurable()?;
        *self.identities.lock().await = book;
        Ok(())
    }

    pub async fn set_rate_limit_config(
        &self,
        config: RateLimitConfig,
    ) -> Result<(), SwarmNetworkError> {
        let _config_guard = self.config_gate.lock().await;
        self.ensure_configurable()?;
        let config = config.validate()?;
        *self.rate_limiter.lock().await = PeerRateLimiter::new(config);
        Ok(())
    }

    fn ensure_configurable(&self) -> Result<(), SwarmNetworkError> {
        if self.run_claimed.load(Ordering::Acquire)
            || !matches!(self.current_state(), SocketState::Created)
        {
            return Err(SwarmNetworkError::ConfigurationLocked);
        }
        Ok(())
    }

    pub async fn set_bootstrap_peers(
        &self,
        peers: impl IntoIterator<Item = EndpointId>,
    ) -> Result<(), SwarmNetworkError> {
        let _config_guard = self.config_gate.lock().await;
        self.ensure_configurable()?;
        let mut unique = HashSet::new();
        let peers = peers
            .into_iter()
            .filter(|peer| *peer != self.node_id())
            .filter(|peer| unique.insert(*peer))
            .collect::<Vec<_>>();
        if peers.len() > MAX_BOOTSTRAP_PEERS {
            return Err(SwarmNetworkError::TooManyBootstrapPeers { count: peers.len() });
        }
        *self.bootstrap.write().await = peers;
        Ok(())
    }

    /// Create a signed host invitation containing this endpoint's currently
    /// known relay/direct addresses.
    pub fn invite(&self) -> Result<SwarmInvite, SwarmNetworkError> {
        let now_ms = unix_time_ms()?;
        SwarmInvite::host(
            self.topic_raw,
            self.endpoint.addr(),
            self.endpoint.secret_key(),
            now_ms,
        )
        .map_err(SwarmNetworkError::InvalidInvite)
    }

    /// Wait for Iroh's initial online determination before producing an
    /// internet-oriented invitation. Local-only applications may use
    /// [`Self::invite`] immediately instead.
    pub async fn invite_online(&self, timeout: Duration) -> Result<SwarmInvite, SwarmNetworkError> {
        tokio::time::timeout(timeout, self.endpoint.online())
            .await
            .map_err(|_| SwarmNetworkError::EndpointOnlineTimeout)?;
        self.invite()
    }

    pub fn shutdown(&self) {
        self.shutdown_tx.send_replace(true);
    }

    pub async fn run(self) -> Result<(), SwarmNetworkError> {
        {
            let _config_guard = self.config_gate.lock().await;
            if self
                .run_claimed
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                return Err(SwarmNetworkError::AlreadyStarted);
            }
        }

        let result = self.run_inner().await;
        self.set_state(SocketState::Stopping);
        *self.sender.lock().await = None;
        if let Err(error) = self.gossip.shutdown().await {
            tracing::warn!(%error, "iroh gossip shutdown reported an error");
        }
        match &result {
            Ok(()) => self.set_state(SocketState::Stopped),
            Err(error) => self.set_state(SocketState::Failed(error.to_string())),
        }
        result
    }

    async fn run_inner(&self) -> Result<(), SwarmNetworkError> {
        if *self.shutdown_tx.borrow() {
            return Ok(());
        }
        self.set_state(SocketState::Subscribing);
        let bootstrap = self.bootstrap.read().await.clone();
        let topic = self
            .gossip
            .subscribe(self.topic_id, bootstrap)
            .await
            .map_err(|error| SwarmNetworkError::Gossip(error.to_string()))?;
        let (sender, mut receiver) = topic.split();
        *self.sender.lock().await = Some(sender);
        self.set_state(SocketState::Active { neighbors: 0 });

        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let mut neighbors = HashSet::new();
        loop {
            tokio::select! {
                changed = shutdown_rx.changed() => {
                    if changed.is_err() || *shutdown_rx.borrow() {
                        return Ok(());
                    }
                }
                event = receiver.next() => {
                    let Some(event) = event else {
                        return Ok(());
                    };
                    let event = event
                        .map_err(|error| SwarmNetworkError::Gossip(error.to_string()))?;
                    match event {
                        iroh_gossip::api::Event::NeighborUp(peer) => {
                            neighbors.insert(peer);
                            self.set_state(SocketState::Active { neighbors: neighbors.len() });
                            self.dispatch_event(
                                SwarmNetworkEvent::NeighborUp(peer),
                                DeliveryClass::Durable,
                            ).await?;
                        }
                        iroh_gossip::api::Event::NeighborDown(peer) => {
                            neighbors.remove(&peer);
                            self.set_state(SocketState::Active { neighbors: neighbors.len() });
                            self.dispatch_event(
                                SwarmNetworkEvent::NeighborDown(peer),
                                DeliveryClass::Durable,
                            ).await?;
                        }
                        iroh_gossip::api::Event::Received(message) => {
                            if let Err(error) = self.handle_received(message).await {
                                if matches!(
                                    &error,
                                    SwarmNetworkError::InboundClosed
                                        | SwarmNetworkError::DurableQueueFull
                                ) {
                                    return Err(error);
                                }
                                tracing::warn!(%error, "rejected swarm gossip message");
                            }
                        }
                        iroh_gossip::api::Event::Lagged => {
                            self.metrics.gossip_lagged.fetch_add(1, Ordering::Relaxed);
                            self.dispatch_event(
                                SwarmNetworkEvent::Lagged,
                                DeliveryClass::Durable,
                            ).await?;
                        }
                    }
                }
            }
        }
    }

    pub async fn broadcast(&self, message: SwarmMessage) -> Result<(), SwarmNetworkError> {
        self.broadcast_tracked(message).await.map(|_| ())
    }

    pub async fn broadcast_tracked(
        &self,
        message: SwarmMessage,
    ) -> Result<BroadcastReceipt, SwarmNetworkError> {
        message
            .validate()
            .map_err(SwarmNetworkError::InvalidMessage)?;
        let now_ms = unix_time_ms()?;
        let ttl_ms = match message.delivery_class() {
            DeliveryClass::BestEffort => BEST_EFFORT_TTL_MS,
            DeliveryClass::Durable => DURABLE_TTL_MS,
        };
        let sequence = self
            .sequence
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .map_err(|_| SwarmNetworkError::SequenceExhausted)?;
        let envelope = SwarmEnvelope::sign(
            message,
            self.endpoint.secret_key(),
            self.session_id,
            sequence,
            now_ms,
            ttl_ms,
        )
        .map_err(SwarmNetworkError::InvalidEnvelope)?;
        let receipt = BroadcastReceipt {
            author: envelope.author,
            session_id: envelope.session_id,
            sequence: envelope.sequence,
            message_id: envelope.message_id,
            issued_at_ms: envelope.issued_at_ms,
            expires_at_ms: envelope.expires_at_ms,
            encoded_bytes: 0,
        };
        let content =
            encode_bounded(&envelope, MAX_WIRE_MESSAGE_BYTES).map_err(SwarmNetworkError::Codec)?;
        if content.len() > MAX_WIRE_MESSAGE_BYTES {
            return Err(SwarmNetworkError::MessageTooLarge {
                size: content.len(),
            });
        }

        let content_len = content.len();
        let mut guard = self.sender.lock().await;
        let sender = guard.as_mut().ok_or(SwarmNetworkError::NotRunning)?;
        match sender.broadcast(content.into()).await {
            Ok(()) => {
                self.metrics.sent.fetch_add(1, Ordering::Relaxed);
                self.metrics
                    .bytes_sent
                    .fetch_add(content_len as u64, Ordering::Relaxed);
                Ok(BroadcastReceipt {
                    encoded_bytes: content_len,
                    ..receipt
                })
            }
            Err(error) => {
                self.metrics.send_errors.fetch_add(1, Ordering::Relaxed);
                Err(SwarmNetworkError::Gossip(error.to_string()))
            }
        }
    }

    async fn handle_received(
        &self,
        message: iroh_gossip::api::Message,
    ) -> Result<(), SwarmNetworkError> {
        self.metrics.received.fetch_add(1, Ordering::Relaxed);
        self.metrics
            .bytes_received
            .fetch_add(message.content.len() as u64, Ordering::Relaxed);
        let delivered_from = message.delivered_from;
        if !self.rate_limiter.lock().await.allow(
            delivered_from,
            message.content.len(),
            Instant::now(),
        ) {
            self.metrics.rate_limited.fetch_add(1, Ordering::Relaxed);
            return self
                .reject(
                    delivered_from,
                    RejectionReason::RateLimited {
                        size: message.content.len(),
                    },
                )
                .await;
        }
        if message.content.len() > MAX_WIRE_MESSAGE_BYTES {
            return self
                .reject(
                    delivered_from,
                    RejectionReason::MessageTooLarge {
                        size: message.content.len(),
                    },
                )
                .await;
        }

        let envelope: SwarmEnvelope = match decode_bounded(&message.content, MAX_WIRE_MESSAGE_BYTES)
        {
            Ok(envelope) => envelope,
            Err(error) => {
                return self
                    .reject(delivered_from, RejectionReason::Decode { detail: error })
                    .await;
            }
        };
        let now_ms = unix_time_ms()?;
        if let Err(error) = envelope.verify_at(now_ms) {
            return self
                .reject(delivered_from, RejectionReason::InvalidEnvelope(error))
                .await;
        }

        if !self.replay.lock().await.insert(envelope.message_id) {
            self.metrics.duplicate.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        if !self.sequence_windows.lock().await.insert(
            envelope.author,
            envelope.session_id,
            envelope.sequence,
        ) {
            self.metrics.sequence_replay.fetch_add(1, Ordering::Relaxed);
            return self
                .reject(
                    delivered_from,
                    RejectionReason::SequenceReplay {
                        session_id: envelope.session_id,
                        sequence: envelope.sequence,
                    },
                )
                .await;
        }

        let identity_policy = *self.identity_policy.read().await;
        if let Err(error) = self.identities.lock().await.verify_or_observe(
            envelope.author,
            envelope.payload.claimed_node_id(),
            identity_policy,
        ) {
            return self
                .reject(delivered_from, RejectionReason::IdentityBinding(error))
                .await;
        }

        let delivery_class = envelope.payload.delivery_class();
        let authenticated = AuthenticatedSwarmMessage {
            author: envelope.author,
            delivered_from,
            message_id: envelope.message_id,
            session_id: envelope.session_id,
            sequence: envelope.sequence,
            issued_at_ms: envelope.issued_at_ms,
            payload: envelope.payload,
        };
        self.metrics.accepted.fetch_add(1, Ordering::Relaxed);
        self.dispatch_event(SwarmNetworkEvent::Message(authenticated), delivery_class)
            .await
    }

    async fn reject(
        &self,
        delivered_from: EndpointId,
        reason: RejectionReason,
    ) -> Result<(), SwarmNetworkError> {
        self.metrics.rejected.fetch_add(1, Ordering::Relaxed);
        self.dispatch_event(
            SwarmNetworkEvent::Rejected {
                delivered_from,
                reason,
            },
            DeliveryClass::BestEffort,
        )
        .await
    }

    async fn dispatch_event(
        &self,
        event: SwarmNetworkEvent,
        delivery_class: DeliveryClass,
    ) -> Result<(), SwarmNetworkError> {
        match (&self.inbound, event) {
            (InboundSink::Legacy(sender), SwarmNetworkEvent::Message(message)) => {
                self.send_with_policy(sender, message.payload, delivery_class)
                    .await
            }
            (InboundSink::Legacy(_), _) => Ok(()),
            (InboundSink::Events(sender), event) => {
                self.send_with_policy(sender, event, delivery_class).await
            }
        }
    }

    async fn send_with_policy<T: Send + 'static>(
        &self,
        sender: &mpsc::Sender<T>,
        value: T,
        delivery_class: DeliveryClass,
    ) -> Result<(), SwarmNetworkError> {
        match delivery_class {
            DeliveryClass::Durable => match sender.try_send(value) {
                Ok(()) => Ok(()),
                Err(mpsc::error::TrySendError::Full(_)) => {
                    self.metrics
                        .durable_queue_full
                        .fetch_add(1, Ordering::Relaxed);
                    Err(SwarmNetworkError::DurableQueueFull)
                }
                Err(mpsc::error::TrySendError::Closed(_)) => Err(SwarmNetworkError::InboundClosed),
            },
            DeliveryClass::BestEffort => match sender.try_send(value) {
                Ok(()) => Ok(()),
                Err(mpsc::error::TrySendError::Full(_)) => {
                    self.metrics
                        .best_effort_dropped
                        .fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
                Err(mpsc::error::TrySendError::Closed(_)) => Err(SwarmNetworkError::InboundClosed),
            },
        }
    }

    fn set_state(&self, state: SocketState) {
        self.state_tx.send_replace(state);
    }
}

pub(crate) fn encode_bounded<T: Serialize>(value: &T, limit: usize) -> Result<Vec<u8>, String> {
    bincode::DefaultOptions::new()
        .with_limit(limit as u64)
        .serialize(value)
        .map_err(|error| error.to_string())
}

pub(crate) fn decode_bounded<T: DeserializeOwned>(bytes: &[u8], limit: usize) -> Result<T, String> {
    bincode::DefaultOptions::new()
        .with_limit(limit as u64)
        .reject_trailing_bytes()
        .deserialize(bytes)
        .map_err(|error| error.to_string())
}

pub(crate) fn system_time_ms() -> Result<u64, String> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?;
    u64::try_from(duration.as_millis())
        .map_err(|_| "system time does not fit in u64 milliseconds".into())
}

fn unix_time_ms() -> Result<u64, SwarmNetworkError> {
    system_time_ms().map_err(SwarmNetworkError::Clock)
}

#[derive(Debug, thiserror::Error)]
pub enum SwarmNetworkError {
    #[error("telepathic socket is not running")]
    NotRunning,
    #[error("telepathic socket run loop has already been started")]
    AlreadyStarted,
    #[error("socket configuration is locked after startup begins")]
    ConfigurationLocked,
    #[error("the local signed-message sequence is exhausted; create a new socket session")]
    SequenceExhausted,
    #[error("invalid rate-limit configuration: {0}")]
    InvalidRateLimitConfig(&'static str),
    #[error("endpoint did not become internet-dialable before the timeout")]
    EndpointOnlineTimeout,
    #[error("timed out waiting for at least {minimum} gossip neighbors")]
    NeighborWaitTimeout { minimum: usize },
    #[error("socket stopped before the requested condition was met")]
    SocketStopped,
    #[error("socket failed before the requested condition was met: {0}")]
    SocketFailed(String),
    #[error("socket state channel closed unexpectedly")]
    StateChannelClosed,
    #[error("durable local delivery queue is full")]
    DurableQueueFull,
    #[error("local inbound channel is closed")]
    InboundClosed,
    #[error("wire message is too large: {size} bytes")]
    MessageTooLarge { size: usize },
    #[error("bootstrap peer count {count} exceeds {MAX_BOOTSTRAP_PEERS}")]
    TooManyBootstrapPeers { count: usize },
    #[error("invalid message: {0}")]
    InvalidMessage(MessageValidationError),
    #[error("invalid envelope: {0}")]
    InvalidEnvelope(EnvelopeError),
    #[error("invalid invite: {0}")]
    InvalidInvite(InviteError),
    #[error("codec error: {0}")]
    Codec(String),
    #[error("iroh gossip error: {0}")]
    Gossip(String),
    #[error("system clock error: {0}")]
    Clock(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LawGossipMsg;

    fn message(node_id: Uuid) -> SwarmMessage {
        SwarmMessage::LawGossip(LawGossipMsg {
            node_id,
            law_id: "safe-torque".into(),
            smtlib2: "(assert (< robot_torque 0.9))".into(),
            proposing_phi: 0.6,
            timestamp: 10,
        })
    }

    #[test]
    fn signed_envelope_detects_payload_tampering() {
        let key = SecretKey::from_bytes(&[7u8; 32]);
        let now = 1_000_000;
        let mut envelope = SwarmEnvelope::sign(
            message(Uuid::from_u128(1)),
            &key,
            Uuid::from_u128(9),
            1,
            now,
            1_000,
        )
        .unwrap();
        assert!(envelope.verify_at(now).is_ok());
        if let SwarmMessage::LawGossip(law) = &mut envelope.payload {
            law.proposing_phi = 0.9;
        }
        assert_eq!(
            envelope.verify_at(now),
            Err(EnvelopeError::InvalidSignature)
        );
    }

    #[test]
    fn envelope_rejects_noncanonical_message_id() {
        let key = SecretKey::from_bytes(&[7u8; 32]);
        let now = 1_000_000;
        let mut envelope = SwarmEnvelope::sign(
            message(Uuid::from_u128(1)),
            &key,
            Uuid::from_u128(9),
            1,
            now,
            1_000,
        )
        .unwrap();
        envelope.message_id = Uuid::from_u128(99);
        assert_eq!(
            envelope.verify_at(now),
            Err(EnvelopeError::InvalidMessageId)
        );
    }

    #[test]
    fn expired_envelope_is_rejected() {
        let key = SecretKey::from_bytes(&[8u8; 32]);
        let envelope = SwarmEnvelope::sign(
            message(Uuid::from_u128(1)),
            &key,
            Uuid::from_u128(9),
            1,
            100,
            10,
        )
        .unwrap();
        assert!(matches!(
            envelope.verify_at(100 + 10 + MAX_CLOCK_SKEW_MS + 1),
            Err(EnvelopeError::Expired { .. })
        ));
    }

    #[test]
    fn replay_window_rejects_duplicate_ids() {
        let mut window = ReplayWindow::new(2);
        let first = Uuid::from_u128(1);
        assert!(window.insert(first));
        assert!(!window.insert(first));
        assert!(window.insert(Uuid::from_u128(2)));
        assert!(window.insert(Uuid::from_u128(3)));
        assert!(window.insert(first));
    }

    #[test]
    fn identity_book_prevents_uuid_takeover() {
        let first = SecretKey::from_bytes(&[1u8; 32]).public();
        let second = SecretKey::from_bytes(&[2u8; 32]).public();
        let node = Uuid::from_u128(42);
        let mut book = IdentityBook::default();
        assert!(book.observe(first, node).is_ok());
        assert!(matches!(
            book.observe(second, node),
            Err(IdentityBindingError::NodeChangedEndpoint { .. })
        ));
    }

    #[test]
    fn pinned_identity_policy_rejects_unknown_nodes() {
        let endpoint = SecretKey::from_bytes(&[12u8; 32]).public();
        let node_id = Uuid::from_u128(55);
        let mut book = IdentityBook::default();
        assert!(matches!(
            book.verify_or_observe(endpoint, node_id, IdentityPolicy::PinnedOnly),
            Err(IdentityBindingError::UnknownIdentity { .. })
        ));
        book.enroll(endpoint, node_id).unwrap();
        assert!(
            book.verify_or_observe(endpoint, node_id, IdentityPolicy::PinnedOnly)
                .is_ok()
        );
    }

    #[test]
    fn identity_book_round_trips_without_weakening_bindings() {
        let endpoint = SecretKey::from_bytes(&[13u8; 32]).public();
        let node_id = Uuid::from_u128(56);
        let mut book = IdentityBook::default();
        book.enroll(endpoint, node_id).unwrap();
        let encoded = book.encode().unwrap();
        let decoded = IdentityBook::decode(&encoded).unwrap();
        assert_eq!(decoded.endpoint_for(node_id), Some(endpoint));
        assert_eq!(decoded.node_for(endpoint), Some(node_id));
    }

    #[test]
    fn sequence_window_allows_reordering_but_rejects_replays() {
        let mut window = SequenceWindow::new(4);
        assert!(window.insert(10));
        assert!(window.insert(8));
        assert!(!window.insert(8));
        assert!(!window.insert(6));
        assert!(window.insert(11));
    }

    #[test]
    fn a_new_signed_session_can_restart_sequence_numbers() {
        let author = SecretKey::from_bytes(&[4u8; 32]).public();
        let mut windows = AuthorSequenceWindows::new(4, 8);
        assert!(windows.insert(author, Uuid::from_u128(1), 1));
        assert!(!windows.insert(author, Uuid::from_u128(1), 1));
        assert!(windows.insert(author, Uuid::from_u128(2), 1));
    }

    #[test]
    fn peer_rate_limiter_enforces_message_and_byte_budgets() {
        let peer = SecretKey::from_bytes(&[11u8; 32]).public();
        let start = Instant::now();
        let mut limiter = PeerRateLimiter::new(RateLimitConfig {
            max_peers: 2,
            window: Duration::from_secs(1),
            max_messages_per_peer: 2,
            max_bytes_per_peer: 10,
        });
        assert!(limiter.allow(peer, 4, start));
        assert!(limiter.allow(peer, 6, start));
        assert!(!limiter.allow(peer, 1, start));
        assert!(limiter.allow(peer, 10, start + Duration::from_secs(1)));
    }

    #[test]
    fn signed_invite_round_trips_and_imports_bootstrap_addresses() {
        let key = SecretKey::from_bytes(&[3u8; 32]);
        let first = key.public();
        let addr = EndpointAddr::new(first);
        let invite = SwarmInvite::host([9u8; 32], addr, &key, 1_000).unwrap();
        let encoded = encode_bounded(&invite, MAX_INVITE_BYTES).unwrap();
        let decoded = SwarmInvite::decode_at(&encoded, 1_000).unwrap();
        assert_eq!(decoded.topic, [9u8; 32]);
        assert_eq!(decoded.issuer, first);
        assert_eq!(decoded.bootstrap_ids(), vec![first]);

        let address_lookup = MemoryLookup::new();
        for endpoint_addr in &decoded.bootstrap {
            address_lookup.add_endpoint_info(endpoint_addr.clone());
        }
        assert!(address_lookup.get_endpoint_info(first).is_some());

        let mut with_trailing_data = encoded;
        with_trailing_data.push(0);
        assert!(SwarmInvite::decode_at(&with_trailing_data, 1_000).is_err());
    }

    #[test]
    fn invite_signature_detects_topic_tampering() {
        let key = SecretKey::from_bytes(&[5u8; 32]);
        let mut invite =
            SwarmInvite::host([1u8; 32], EndpointAddr::new(key.public()), &key, 1_000).unwrap();
        invite.topic = [2u8; 32];
        assert_eq!(invite.verify_at(1_000), Err(InviteError::InvalidSignature));
    }

    #[test]
    fn expired_invite_is_rejected() {
        let key = SecretKey::from_bytes(&[6u8; 32]);
        let invite = SwarmInvite::sign(
            [1u8; 32],
            vec![EndpointAddr::new(key.public())],
            &key,
            100,
            10,
        )
        .unwrap();
        assert!(matches!(
            invite.verify_at(100 + 10 + MAX_CLOCK_SKEW_MS + 1),
            Err(InviteError::Expired { .. })
        ));
    }

    #[test]
    fn process_session_ids_are_endpoint_authenticated_and_unique() {
        let key = SecretKey::from_bytes(&[17u8; 32]);
        let first = new_session_id(&key);
        let second = new_session_id(&key);
        assert!(!first.is_nil());
        assert!(!second.is_nil());
        assert_ne!(first, second);
    }
}
