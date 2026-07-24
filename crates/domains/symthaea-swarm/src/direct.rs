//! Authenticated direct Iroh transport for latency-sensitive Luminous traffic.
//!
//! [`crate::networking::TelepathicSocket`] remains the many-to-many gossip
//! control plane. This module adds a separate ALPN for peer-to-peer reliable
//! streams and unreliable datagrams. Direct packets are signed by the endpoint
//! key even though the QUIC connection is authenticated; retaining a signed
//! envelope makes captured packets self-authenticating and lets adapters bind
//! durable evidence to an endpoint identity.
//!
//! ## Delivery truth
//!
//! * [`DirectTransport::send_reliable`] returns only after the remote protocol
//!   has validated the packet and queued it to its local application channel.
//!   It does not prove that domain logic applied or persisted the message.
//! * [`DirectTransport::send_datagram`] reports local QUIC acceptance only.
//!   Datagrams may be lost, duplicated, or reordered.
//! * The direct protocol does not discover peers. Import a signed swarm invite,
//!   exchange an [`EndpointAddr`], or use another rendezvous mechanism first.

use crate::{
    enrollment::{DirectEnrollmentBook, DirectEnrollmentError, DirectPeerRolloverProof},
    networking::{
        AuthorSequenceWindows, DEFAULT_REPLAY_WINDOW, DEFAULT_SEQUENCE_WINDOW, MAX_CLOCK_SKEW_MS,
        MAX_SEQUENCE_SESSIONS, PeerRateLimiter, RateLimitConfig, ReplayWindow, decode_bounded,
        encode_bounded, message_id_from_signature, new_session_id, system_time_ms,
    },
};
use iroh::{
    Endpoint, EndpointAddr, EndpointId, SecretKey, Signature,
    endpoint::{Connection, RecvStream, SendStream},
    protocol::{AcceptError, ProtocolHandler},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};
use tokio::sync::{Mutex, RwLock, Semaphore, mpsc, watch};
use uuid::Uuid;

/// ALPN registered in the shared Iroh router for direct Luminous traffic.
pub const DIRECT_ALPN: &[u8] = b"luminous/direct/2";
/// Signed direct-envelope schema version.
pub const DIRECT_PROTOCOL_VERSION: u16 = 2;
/// Maximum payload accepted on a reliable stream.
pub const MAX_DIRECT_RELIABLE_PAYLOAD_BYTES: usize = 4 * 1024 * 1024;
/// Absolute payload cap for a datagram. The active QUIC path usually imposes a
/// much smaller limit and is checked before every send.
pub const MAX_DIRECT_DATAGRAM_PAYLOAD_BYTES: usize = 48 * 1024;
/// Maximum encoded reliable request, including envelope and signature.
pub const MAX_DIRECT_RELIABLE_FRAME_BYTES: usize = MAX_DIRECT_RELIABLE_PAYLOAD_BYTES + 16 * 1024;
/// Maximum encoded datagram, including envelope and signature.
pub const MAX_DIRECT_DATAGRAM_FRAME_BYTES: usize = MAX_DIRECT_DATAGRAM_PAYLOAD_BYTES + 16 * 1024;
/// Maximum encoded acknowledgement or rejection response.
pub const MAX_DIRECT_ACK_BYTES: usize = 4 * 1024;
/// Longest accepted direct packet lifetime.
pub const MAX_DIRECT_TTL_MS: u64 = 10 * 60 * 1_000;
/// Freshness window for reliable control/state messages.
pub const DEFAULT_DIRECT_RELIABLE_TTL_MS: u64 = 2 * 60 * 1_000;
/// Freshness window for latency-sensitive datagrams.
pub const DEFAULT_DIRECT_DATAGRAM_TTL_MS: u64 = 5 * 1_000;
/// Default timeout for dialing and acknowledged reliable delivery.
pub const DEFAULT_DIRECT_OPERATION_TIMEOUT: Duration = Duration::from_secs(10);
/// Default maximum number of simultaneously retained direct peers.
pub const DEFAULT_MAX_DIRECT_PEERS: usize = 1_024;
/// Default maximum number of concurrently processed reliable streams per peer.
pub const DEFAULT_MAX_CONCURRENT_STREAMS_PER_PEER: usize = 128;
/// Default maximum datagrams accepted from one direct peer per second.
pub const DEFAULT_MAX_DATAGRAMS_PER_PEER_PER_SECOND: u64 = 2_048;
/// Default maximum datagram bytes accepted from one direct peer per second.
pub const DEFAULT_MAX_DATAGRAM_BYTES_PER_PEER_PER_SECOND: u64 = 16 * 1024 * 1024;
/// Default number of completed or in-flight idempotent operations retained.
pub const DEFAULT_MAX_IDEMPOTENT_OPERATIONS: usize = 65_536;

/// Stable numeric lane identifier. The transport does not interpret lane data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DirectLane(pub u16);

impl DirectLane {
    pub const CONTROL: Self = Self(1);
    pub const PLAYER_INPUT: Self = Self(2);
    pub const STATE_SNAPSHOT: Self = Self(3);
    pub const TELEMETRY: Self = Self(4);
    pub const ROBOTICS: Self = Self(5);
    pub const ASSET_TRANSFER: Self = Self(6);
}

/// Delivery primitive selected for a signed direct envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DirectDelivery {
    Reliable,
    Datagram,
}

/// Endpoint-signed packet carried by either a direct stream or datagram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectEnvelope {
    pub protocol_version: u16,
    pub message_id: Uuid,
    pub author: EndpointId,
    pub session_id: Uuid,
    pub sequence: u64,
    /// Stable caller-supplied operation identity for duplicate-safe reliable retries.
    pub operation_id: Option<Uuid>,
    pub lane: DirectLane,
    pub delivery: DirectDelivery,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    pub payload: Vec<u8>,
    pub signature: Signature,
}

#[derive(Serialize)]
struct DirectSigningView<'a> {
    protocol_version: u16,
    author: EndpointId,
    session_id: Uuid,
    sequence: u64,
    operation_id: Option<Uuid>,
    lane: DirectLane,
    delivery: DirectDelivery,
    issued_at_ms: u64,
    expires_at_ms: u64,
    payload: &'a [u8],
}

impl DirectEnvelope {
    #[allow(clippy::too_many_arguments)]
    pub fn sign(
        payload: Vec<u8>,
        lane: DirectLane,
        delivery: DirectDelivery,
        secret_key: &SecretKey,
        session_id: Uuid,
        sequence: u64,
        issued_at_ms: u64,
        ttl_ms: u64,
    ) -> Result<Self, DirectEnvelopeError> {
        Self::sign_with_operation(
            payload,
            lane,
            delivery,
            secret_key,
            session_id,
            sequence,
            issued_at_ms,
            ttl_ms,
            None,
        )
    }

    /// Sign a packet with a stable operation ID. Operation IDs are accepted only
    /// for reliable delivery and let callers retry after an ambiguous timeout
    /// without enqueueing the domain action more than once.
    #[allow(clippy::too_many_arguments)]
    pub fn sign_with_operation(
        payload: Vec<u8>,
        lane: DirectLane,
        delivery: DirectDelivery,
        secret_key: &SecretKey,
        session_id: Uuid,
        sequence: u64,
        issued_at_ms: u64,
        ttl_ms: u64,
        operation_id: Option<Uuid>,
    ) -> Result<Self, DirectEnvelopeError> {
        validate_direct_payload(delivery, payload.len())?;
        if operation_id == Some(Uuid::nil()) {
            return Err(DirectEnvelopeError::NilOperationId);
        }
        if operation_id.is_some() && delivery != DirectDelivery::Reliable {
            return Err(DirectEnvelopeError::OperationIdRequiresReliable);
        }
        if session_id.is_nil() {
            return Err(DirectEnvelopeError::NilSessionId);
        }
        if sequence == 0 {
            return Err(DirectEnvelopeError::ZeroSequence);
        }
        if ttl_ms == 0 || ttl_ms > MAX_DIRECT_TTL_MS {
            return Err(DirectEnvelopeError::InvalidTtl { ttl_ms });
        }
        let expires_at_ms = issued_at_ms
            .checked_add(ttl_ms)
            .ok_or(DirectEnvelopeError::TimestampOverflow)?;
        let author = secret_key.public();
        let signature = secret_key.sign(&direct_signing_bytes(
            DIRECT_PROTOCOL_VERSION,
            author,
            session_id,
            sequence,
            operation_id,
            lane,
            delivery,
            issued_at_ms,
            expires_at_ms,
            &payload,
        )?);
        let message_id = message_id_from_signature(&signature);
        Ok(Self {
            protocol_version: DIRECT_PROTOCOL_VERSION,
            message_id,
            author,
            session_id,
            sequence,
            operation_id,
            lane,
            delivery,
            issued_at_ms,
            expires_at_ms,
            payload,
            signature,
        })
    }

    pub fn verify_at(&self, now_ms: u64) -> Result<(), DirectEnvelopeError> {
        if self.protocol_version != DIRECT_PROTOCOL_VERSION {
            return Err(DirectEnvelopeError::UnsupportedVersion {
                received: self.protocol_version,
            });
        }
        validate_direct_payload(self.delivery, self.payload.len())?;
        if self.session_id.is_nil() {
            return Err(DirectEnvelopeError::NilSessionId);
        }
        if self.sequence == 0 {
            return Err(DirectEnvelopeError::ZeroSequence);
        }
        if self.operation_id == Some(Uuid::nil()) {
            return Err(DirectEnvelopeError::NilOperationId);
        }
        if self.operation_id.is_some() && self.delivery != DirectDelivery::Reliable {
            return Err(DirectEnvelopeError::OperationIdRequiresReliable);
        }
        if self.expires_at_ms < self.issued_at_ms {
            return Err(DirectEnvelopeError::InvalidTtl { ttl_ms: 0 });
        }
        let ttl_ms = self.expires_at_ms - self.issued_at_ms;
        if ttl_ms == 0 || ttl_ms > MAX_DIRECT_TTL_MS {
            return Err(DirectEnvelopeError::InvalidTtl { ttl_ms });
        }
        if self.issued_at_ms > now_ms.saturating_add(MAX_CLOCK_SKEW_MS) {
            return Err(DirectEnvelopeError::IssuedInFuture {
                issued_at_ms: self.issued_at_ms,
                now_ms,
            });
        }
        if now_ms > self.expires_at_ms.saturating_add(MAX_CLOCK_SKEW_MS) {
            return Err(DirectEnvelopeError::Expired {
                expires_at_ms: self.expires_at_ms,
                now_ms,
            });
        }
        let bytes = direct_signing_bytes(
            self.protocol_version,
            self.author,
            self.session_id,
            self.sequence,
            self.operation_id,
            self.lane,
            self.delivery,
            self.issued_at_ms,
            self.expires_at_ms,
            &self.payload,
        )?;
        self.author
            .verify(&bytes, &self.signature)
            .map_err(|_| DirectEnvelopeError::InvalidSignature)?;
        if self.message_id != message_id_from_signature(&self.signature) {
            return Err(DirectEnvelopeError::InvalidMessageId);
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn direct_signing_bytes(
    protocol_version: u16,
    author: EndpointId,
    session_id: Uuid,
    sequence: u64,
    operation_id: Option<Uuid>,
    lane: DirectLane,
    delivery: DirectDelivery,
    issued_at_ms: u64,
    expires_at_ms: u64,
    payload: &[u8],
) -> Result<Vec<u8>, DirectEnvelopeError> {
    encode_bounded(
        &DirectSigningView {
            protocol_version,
            author,
            session_id,
            sequence,
            operation_id,
            lane,
            delivery,
            issued_at_ms,
            expires_at_ms,
            payload,
        },
        MAX_DIRECT_RELIABLE_FRAME_BYTES,
    )
    .map_err(DirectEnvelopeError::Codec)
}

fn validate_direct_payload(
    delivery: DirectDelivery,
    size: usize,
) -> Result<(), DirectEnvelopeError> {
    let maximum = match delivery {
        DirectDelivery::Reliable => MAX_DIRECT_RELIABLE_PAYLOAD_BYTES,
        DirectDelivery::Datagram => MAX_DIRECT_DATAGRAM_PAYLOAD_BYTES,
    };
    if size > maximum {
        return Err(DirectEnvelopeError::PayloadTooLarge {
            delivery,
            size,
            maximum,
        });
    }
    Ok(())
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum DirectEnvelopeError {
    #[error("unsupported direct protocol version {received}")]
    UnsupportedVersion { received: u16 },
    #[error("direct session ID must not be nil")]
    NilSessionId,
    #[error("direct sequence number zero is reserved")]
    ZeroSequence,
    #[error("direct operation ID must not be nil")]
    NilOperationId,
    #[error("operation IDs are supported only for reliable delivery")]
    OperationIdRequiresReliable,
    #[error("invalid direct packet TTL: {ttl_ms} ms")]
    InvalidTtl { ttl_ms: u64 },
    #[error("direct timestamp arithmetic overflow")]
    TimestampOverflow,
    #[error("direct packet issue time {issued_at_ms} is too far ahead of local time {now_ms}")]
    IssuedInFuture { issued_at_ms: u64, now_ms: u64 },
    #[error("direct packet expired at {expires_at_ms}; local time is {now_ms}")]
    Expired { expires_at_ms: u64, now_ms: u64 },
    #[error("{delivery:?} payload is too large: {size} bytes; maximum is {maximum}")]
    PayloadTooLarge {
        delivery: DirectDelivery,
        size: usize,
        maximum: usize,
    },
    #[error("direct message ID does not match the verified signature")]
    InvalidMessageId,
    #[error("invalid direct packet signature")]
    InvalidSignature,
    #[error("direct packet codec error: {0}")]
    Codec(String),
}

/// Connection direction used for deterministic duplicate resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectConnectionOrigin {
    Incoming,
    Outgoing,
}

/// Admission policy for authenticated Iroh peers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectPeerPolicy {
    /// Any endpoint with a valid Iroh handshake may connect.
    AnyAuthenticated,
    /// Only explicitly enrolled endpoint IDs may connect.
    PinnedOnly,
}

impl Default for DirectPeerPolicy {
    fn default() -> Self {
        Self::AnyAuthenticated
    }
}

/// Validated direct message queued to the local application.
#[derive(Debug, Clone)]
pub struct AuthenticatedDirectMessage {
    pub author: EndpointId,
    pub message_id: Uuid,
    pub session_id: Uuid,
    pub sequence: u64,
    pub operation_id: Option<Uuid>,
    pub lane: DirectLane,
    pub delivery: DirectDelivery,
    pub issued_at_ms: u64,
    pub payload: Vec<u8>,
}

impl From<DirectEnvelope> for AuthenticatedDirectMessage {
    fn from(envelope: DirectEnvelope) -> Self {
        Self {
            author: envelope.author,
            message_id: envelope.message_id,
            session_id: envelope.session_id,
            sequence: envelope.sequence,
            operation_id: envelope.operation_id,
            lane: envelope.lane,
            delivery: envelope.delivery,
            issued_at_ms: envelope.issued_at_ms,
            payload: envelope.payload,
        }
    }
}

/// Receipt for a packet accepted by the local direct transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectSendReceipt {
    pub peer: EndpointId,
    pub message_id: Uuid,
    pub session_id: Uuid,
    pub sequence: u64,
    pub operation_id: Option<Uuid>,
    pub lane: DirectLane,
    pub delivery: DirectDelivery,
    pub encoded_bytes: usize,
}

/// Result of an acknowledged reliable stream send.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReliableDeliveryReceipt {
    pub local: DirectSendReceipt,
    /// The remote protocol validated and queued the message. Domain processing
    /// and persistence may still happen later.
    pub remote_queue_accepted: bool,
    /// True when the remote endpoint had already accepted this operation ID.
    pub remote_duplicate: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum ReliableResponse {
    Accepted {
        message_id: Uuid,
        duplicate: bool,
    },
    Rejected {
        message_id: Option<Uuid>,
        reason: ReliableRejectCode,
    },
}

/// Stable rejection codes returned over an authenticated reliable stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReliableRejectCode {
    Overloaded,
    FrameTooLarge,
    DecodeFailed,
    InvalidEnvelope,
    AuthorMismatch,
    DeliveryMismatch,
    Replay,
    OperationInProgress,
    OperationConflict,
    ApplicationQueueFull,
    ApplicationQueueClosed,
}

/// Application-facing direct transport event.
#[derive(Debug, Clone)]
pub enum DirectEvent {
    PeerConnected {
        peer: EndpointId,
        origin: DirectConnectionOrigin,
    },
    PeerReplaced {
        peer: EndpointId,
        retained_origin: DirectConnectionOrigin,
    },
    PeerDisconnected {
        peer: EndpointId,
        reason: String,
    },
    Reliable(AuthenticatedDirectMessage),
    Datagram(AuthenticatedDirectMessage),
    Rejected {
        peer: EndpointId,
        delivery: DirectDelivery,
        reason: DirectRejectReason,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirectRejectReason {
    RateLimited {
        size: usize,
    },
    FrameTooLarge {
        size: usize,
        maximum: usize,
    },
    Decode(String),
    InvalidEnvelope(DirectEnvelopeError),
    AuthorMismatch {
        authenticated_peer: EndpointId,
        claimed_author: EndpointId,
    },
    DeliveryMismatch {
        expected: DirectDelivery,
        received: DirectDelivery,
    },
    Replay {
        session_id: Uuid,
        sequence: u64,
    },
    OperationInProgress {
        operation_id: Uuid,
    },
    OperationConflict {
        operation_id: Uuid,
    },
    ApplicationQueueFull,
    ApplicationQueueClosed,
}

/// Runtime configuration for the direct data plane.
#[derive(Debug, Clone)]
pub struct DirectTransportConfig {
    pub max_peers: usize,
    pub max_concurrent_streams_per_peer: usize,
    pub operation_timeout: Duration,
    pub reliable_ttl_ms: u64,
    pub datagram_ttl_ms: u64,
    pub max_datagram_peers: usize,
    pub datagram_rate_window: Duration,
    pub max_datagrams_per_peer: u64,
    pub max_datagram_bytes_per_peer: u64,
    pub max_idempotent_operations: usize,
}

impl Default for DirectTransportConfig {
    fn default() -> Self {
        Self {
            max_peers: DEFAULT_MAX_DIRECT_PEERS,
            max_concurrent_streams_per_peer: DEFAULT_MAX_CONCURRENT_STREAMS_PER_PEER,
            operation_timeout: DEFAULT_DIRECT_OPERATION_TIMEOUT,
            reliable_ttl_ms: DEFAULT_DIRECT_RELIABLE_TTL_MS,
            datagram_ttl_ms: DEFAULT_DIRECT_DATAGRAM_TTL_MS,
            max_datagram_peers: DEFAULT_MAX_DIRECT_PEERS,
            datagram_rate_window: Duration::from_secs(1),
            max_datagrams_per_peer: DEFAULT_MAX_DATAGRAMS_PER_PEER_PER_SECOND,
            max_datagram_bytes_per_peer: DEFAULT_MAX_DATAGRAM_BYTES_PER_PEER_PER_SECOND,
            max_idempotent_operations: DEFAULT_MAX_IDEMPOTENT_OPERATIONS,
        }
    }
}

impl DirectTransportConfig {
    fn validate(&self) -> Result<(), DirectTransportError> {
        if self.max_peers == 0 {
            return Err(DirectTransportError::InvalidConfig(
                "max_peers must be greater than zero",
            ));
        }
        if self.max_concurrent_streams_per_peer == 0 {
            return Err(DirectTransportError::InvalidConfig(
                "max_concurrent_streams_per_peer must be greater than zero",
            ));
        }
        if self.operation_timeout.is_zero() {
            return Err(DirectTransportError::InvalidConfig(
                "operation_timeout must be greater than zero",
            ));
        }
        for ttl_ms in [self.reliable_ttl_ms, self.datagram_ttl_ms] {
            if ttl_ms == 0 || ttl_ms > MAX_DIRECT_TTL_MS {
                return Err(DirectTransportError::InvalidConfig(
                    "direct TTLs must be within the protocol maximum",
                ));
            }
        }
        if self.max_idempotent_operations == 0 {
            return Err(DirectTransportError::InvalidConfig(
                "max_idempotent_operations must be greater than zero",
            ));
        }
        RateLimitConfig {
            max_peers: self.max_datagram_peers,
            window: self.datagram_rate_window,
            max_messages_per_peer: self.max_datagrams_per_peer,
            max_bytes_per_peer: self.max_datagram_bytes_per_peer,
        }
        .validate()
        .map_err(|_| DirectTransportError::InvalidConfig("invalid datagram rate limit"))?;
        Ok(())
    }
}

/// Bounded reconnect policy for a maintained direct peer session.
#[derive(Debug, Clone)]
pub struct ReconnectPolicy {
    pub initial_backoff: Duration,
    pub maximum_backoff: Duration,
    pub health_probe_interval: Duration,
    /// Maximum consecutive failed dials. `None` retries until cancelled or the
    /// parent transport shuts down.
    pub maximum_attempts: Option<u32>,
}

impl Default for ReconnectPolicy {
    fn default() -> Self {
        Self {
            initial_backoff: Duration::from_millis(250),
            maximum_backoff: Duration::from_secs(30),
            health_probe_interval: Duration::from_millis(500),
            maximum_attempts: None,
        }
    }
}

impl ReconnectPolicy {
    fn validate(&self) -> Result<(), DirectTransportError> {
        if self.initial_backoff.is_zero() {
            return Err(DirectTransportError::InvalidReconnectPolicy(
                "initial_backoff must be greater than zero",
            ));
        }
        if self.maximum_backoff < self.initial_backoff {
            return Err(DirectTransportError::InvalidReconnectPolicy(
                "maximum_backoff must not be smaller than initial_backoff",
            ));
        }
        if self.health_probe_interval.is_zero() {
            return Err(DirectTransportError::InvalidReconnectPolicy(
                "health_probe_interval must be greater than zero",
            ));
        }
        if self.maximum_attempts == Some(0) {
            return Err(DirectTransportError::InvalidReconnectPolicy(
                "maximum_attempts must be greater than zero when present",
            ));
        }
        Ok(())
    }

    fn delay_for(&self, attempt: u32, entropy: u64) -> Duration {
        let initial_ms = u64::try_from(self.initial_backoff.as_millis()).unwrap_or(u64::MAX);
        let maximum_ms = u64::try_from(self.maximum_backoff.as_millis()).unwrap_or(u64::MAX);
        let shift = attempt.saturating_sub(1).min(31);
        let base_ms = initial_ms.saturating_mul(1u64 << shift).min(maximum_ms);
        // Add up to 20% deterministic per-session jitter. This avoids a new RNG
        // dependency while preventing synchronized reconnect storms.
        let jitter_max = base_ms / 5;
        let mixed = entropy
            .rotate_left(attempt % 63)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let jitter = if jitter_max == 0 {
            0
        } else {
            mixed % jitter_max.saturating_add(1)
        };
        Duration::from_millis(base_ms.saturating_add(jitter).min(maximum_ms))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaintainedPeerState {
    Starting,
    Connecting {
        attempt: u32,
    },
    Connected,
    BackingOff {
        attempt: u32,
        delay: Duration,
        last_error: String,
    },
    Stopped,
    Failed(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerConnectionSnapshot {
    pub peer: EndpointId,
    pub origin: DirectConnectionOrigin,
    pub stable_id: usize,
    pub datagrams_supported: bool,
    pub maximum_datagram_size: Option<usize>,
    pub datagram_send_buffer_space: usize,
}

#[derive(Debug, Clone)]
pub struct DirectHealthSnapshot {
    pub local_endpoint: EndpointId,
    pub session_id: Uuid,
    pub peers: Vec<PeerConnectionSnapshot>,
    pub peer_policy: DirectPeerPolicy,
    pub metrics: DirectMetricsSnapshot,
    pub capabilities: DirectTransportCapabilities,
}

/// Explicit capability boundary for the direct implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectTransportCapabilities {
    pub endpoint_authenticated_connections: bool,
    pub signed_packets: bool,
    pub reliable_streams: bool,
    pub unreliable_datagrams: bool,
    pub remote_queue_acknowledgements: bool,
    pub domain_apply_acknowledgements: bool,
    pub peer_discovery: bool,
}

impl DirectTransportCapabilities {
    pub const DATA_PLANE_V1: Self = Self {
        endpoint_authenticated_connections: true,
        signed_packets: true,
        reliable_streams: true,
        unreliable_datagrams: true,
        remote_queue_acknowledgements: true,
        domain_apply_acknowledgements: false,
        peer_discovery: false,
    };
}

#[derive(Debug, Default)]
struct DirectMetrics {
    connections_opened: AtomicU64,
    connections_replaced: AtomicU64,
    connections_closed: AtomicU64,
    reliable_sent: AtomicU64,
    reliable_accepted: AtomicU64,
    reliable_rejected: AtomicU64,
    datagrams_sent: AtomicU64,
    datagrams_received: AtomicU64,
    datagrams_dropped: AtomicU64,
    bytes_sent: AtomicU64,
    bytes_received: AtomicU64,
    invalid_packets: AtomicU64,
    replayed_packets: AtomicU64,
    rate_limited: AtomicU64,
    event_queue_full: AtomicU64,
    idempotent_duplicates: AtomicU64,
    idempotent_in_progress: AtomicU64,
    idempotent_conflicts: AtomicU64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DirectMetricsSnapshot {
    pub connections_opened: u64,
    pub connections_replaced: u64,
    pub connections_closed: u64,
    pub reliable_sent: u64,
    pub reliable_accepted: u64,
    pub reliable_rejected: u64,
    pub datagrams_sent: u64,
    pub datagrams_received: u64,
    pub datagrams_dropped: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub invalid_packets: u64,
    pub replayed_packets: u64,
    pub rate_limited: u64,
    pub event_queue_full: u64,
    pub idempotent_duplicates: u64,
    pub idempotent_in_progress: u64,
    pub idempotent_conflicts: u64,
}

impl DirectMetrics {
    fn snapshot(&self) -> DirectMetricsSnapshot {
        DirectMetricsSnapshot {
            connections_opened: self.connections_opened.load(Ordering::Relaxed),
            connections_replaced: self.connections_replaced.load(Ordering::Relaxed),
            connections_closed: self.connections_closed.load(Ordering::Relaxed),
            reliable_sent: self.reliable_sent.load(Ordering::Relaxed),
            reliable_accepted: self.reliable_accepted.load(Ordering::Relaxed),
            reliable_rejected: self.reliable_rejected.load(Ordering::Relaxed),
            datagrams_sent: self.datagrams_sent.load(Ordering::Relaxed),
            datagrams_received: self.datagrams_received.load(Ordering::Relaxed),
            datagrams_dropped: self.datagrams_dropped.load(Ordering::Relaxed),
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            invalid_packets: self.invalid_packets.load(Ordering::Relaxed),
            replayed_packets: self.replayed_packets.load(Ordering::Relaxed),
            rate_limited: self.rate_limited.load(Ordering::Relaxed),
            event_queue_full: self.event_queue_full.load(Ordering::Relaxed),
            idempotent_duplicates: self.idempotent_duplicates.load(Ordering::Relaxed),
            idempotent_in_progress: self.idempotent_in_progress.load(Ordering::Relaxed),
            idempotent_conflicts: self.idempotent_conflicts.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OperationState {
    Pending,
    Accepted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OperationRecord {
    state: OperationState,
    fingerprint: [u8; 32],
}

#[derive(Debug)]
struct OperationWindow {
    capacity: usize,
    order: VecDeque<(EndpointId, Uuid)>,
    states: HashMap<(EndpointId, Uuid), OperationRecord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OperationClaim {
    New,
    Pending,
    Accepted,
    Conflict,
}

impl OperationWindow {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            order: VecDeque::with_capacity(capacity.min(4_096)),
            states: HashMap::new(),
        }
    }

    fn claim(
        &mut self,
        author: EndpointId,
        operation_id: Uuid,
        fingerprint: [u8; 32],
    ) -> OperationClaim {
        let key = (author, operation_id);
        match self.states.get(&key).copied() {
            Some(record) if record.fingerprint != fingerprint => OperationClaim::Conflict,
            Some(OperationRecord {
                state: OperationState::Pending,
                ..
            }) => OperationClaim::Pending,
            Some(OperationRecord {
                state: OperationState::Accepted,
                ..
            }) => OperationClaim::Accepted,
            None => {
                while self.states.len() >= self.capacity {
                    let Some(oldest) = self.order.pop_front() else {
                        break;
                    };
                    if self.states.get(&oldest).map(|record| record.state)
                        == Some(OperationState::Accepted)
                    {
                        self.states.remove(&oldest);
                    }
                }
                if self.states.len() >= self.capacity {
                    return OperationClaim::Pending;
                }
                self.states.insert(
                    key,
                    OperationRecord {
                        state: OperationState::Pending,
                        fingerprint,
                    },
                );
                OperationClaim::New
            }
        }
    }

    fn accept(&mut self, author: EndpointId, operation_id: Uuid) {
        let key = (author, operation_id);
        if let Some(record) = self.states.get_mut(&key) {
            record.state = OperationState::Accepted;
            self.order.push_back(key);
        }
    }

    fn release(&mut self, author: EndpointId, operation_id: Uuid) {
        let key = (author, operation_id);
        if self.states.get(&key).map(|record| record.state) == Some(OperationState::Pending) {
            self.states.remove(&key);
        }
    }
}

fn operation_fingerprint(envelope: &DirectEnvelope) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"luminous-direct-idempotency-v1");
    hasher.update(&envelope.protocol_version.to_le_bytes());
    hasher.update(&envelope.lane.0.to_le_bytes());
    hasher.update(&(envelope.payload.len() as u64).to_le_bytes());
    hasher.update(&envelope.payload);
    *hasher.finalize().as_bytes()
}

#[derive(Debug, Clone)]
struct ManagedConnection {
    connection: Connection,
    origin: DirectConnectionOrigin,
}

struct DirectInner {
    endpoint: Endpoint,
    event_tx: mpsc::Sender<DirectEvent>,
    config: DirectTransportConfig,
    connections: RwLock<HashMap<EndpointId, ManagedConnection>>,
    dial_gate: Mutex<()>,
    peer_policy: RwLock<DirectPeerPolicy>,
    pinned_peers: RwLock<HashSet<EndpointId>>,
    shutdown_tx: watch::Sender<bool>,
    session_id: Uuid,
    sequence: AtomicU64,
    replay: Mutex<ReplayWindow>,
    sequence_windows: Mutex<AuthorSequenceWindows>,
    datagram_rate_limiter: Mutex<PeerRateLimiter>,
    operations: Mutex<OperationWindow>,
    metrics: DirectMetrics,
}

/// Cloneable direct transport handle and router protocol source.
#[derive(Clone)]
pub struct DirectTransport {
    inner: Arc<DirectInner>,
}

impl fmt::Debug for DirectTransport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DirectTransport")
            .field("endpoint", &self.inner.endpoint.id())
            .field("session_id", &self.inner.session_id)
            .finish_non_exhaustive()
    }
}

impl DirectTransport {
    pub fn new(
        endpoint: Endpoint,
        event_tx: mpsc::Sender<DirectEvent>,
    ) -> Result<Self, DirectTransportError> {
        Self::with_config(endpoint, event_tx, DirectTransportConfig::default())
    }

    pub fn with_config(
        endpoint: Endpoint,
        event_tx: mpsc::Sender<DirectEvent>,
        config: DirectTransportConfig,
    ) -> Result<Self, DirectTransportError> {
        config.validate()?;
        let (shutdown_tx, _) = watch::channel(false);
        let session_id = new_session_id(endpoint.secret_key());
        let max_idempotent_operations = config.max_idempotent_operations;
        let rate_config = RateLimitConfig {
            max_peers: config.max_datagram_peers,
            window: config.datagram_rate_window,
            max_messages_per_peer: config.max_datagrams_per_peer,
            max_bytes_per_peer: config.max_datagram_bytes_per_peer,
        };
        Ok(Self {
            inner: Arc::new(DirectInner {
                endpoint,
                event_tx,
                config,
                connections: RwLock::new(HashMap::new()),
                dial_gate: Mutex::new(()),
                peer_policy: RwLock::new(DirectPeerPolicy::default()),
                pinned_peers: RwLock::new(HashSet::new()),
                shutdown_tx,
                session_id,
                sequence: AtomicU64::new(1),
                replay: Mutex::new(ReplayWindow::new(DEFAULT_REPLAY_WINDOW)),
                sequence_windows: Mutex::new(AuthorSequenceWindows::new(
                    MAX_SEQUENCE_SESSIONS,
                    DEFAULT_SEQUENCE_WINDOW,
                )),
                datagram_rate_limiter: Mutex::new(PeerRateLimiter::new(rate_config)),
                operations: Mutex::new(OperationWindow::new(max_idempotent_operations)),
                metrics: DirectMetrics::default(),
            }),
        })
    }

    pub fn endpoint(&self) -> &Endpoint {
        &self.inner.endpoint
    }

    pub fn node_id(&self) -> EndpointId {
        self.inner.endpoint.id()
    }

    pub fn session_id(&self) -> Uuid {
        self.inner.session_id
    }

    pub const fn capabilities(&self) -> DirectTransportCapabilities {
        DirectTransportCapabilities::DATA_PLANE_V1
    }

    pub fn protocol_handler(&self) -> DirectProtocol {
        DirectProtocol {
            transport: self.clone(),
        }
    }

    pub fn metrics(&self) -> DirectMetricsSnapshot {
        self.inner.metrics.snapshot()
    }

    pub async fn peer_snapshots(&self) -> Vec<PeerConnectionSnapshot> {
        self.inner
            .connections
            .read()
            .await
            .iter()
            .filter_map(|(peer, managed)| {
                if managed.connection.close_reason().is_some() {
                    return None;
                }
                let maximum_datagram_size = managed.connection.max_datagram_size();
                Some(PeerConnectionSnapshot {
                    peer: *peer,
                    origin: managed.origin,
                    stable_id: managed.connection.stable_id(),
                    datagrams_supported: maximum_datagram_size.is_some(),
                    maximum_datagram_size,
                    datagram_send_buffer_space: managed.connection.datagram_send_buffer_space(),
                })
            })
            .collect()
    }

    pub async fn health_snapshot(&self) -> DirectHealthSnapshot {
        DirectHealthSnapshot {
            local_endpoint: self.node_id(),
            session_id: self.session_id(),
            peers: self.peer_snapshots().await,
            peer_policy: self.peer_policy().await,
            metrics: self.metrics(),
            capabilities: self.capabilities(),
        }
    }

    pub async fn peer_policy(&self) -> DirectPeerPolicy {
        *self.inner.peer_policy.read().await
    }

    pub async fn set_peer_policy(&self, policy: DirectPeerPolicy) {
        *self.inner.peer_policy.write().await = policy;
        if policy == DirectPeerPolicy::PinnedOnly {
            let pinned = self.inner.pinned_peers.read().await.clone();
            let peers = self.connected_peers().await;
            for peer in peers {
                if !pinned.contains(&peer) {
                    self.close_peer(peer, b"pinned-only admission enabled")
                        .await;
                }
            }
        }
    }

    pub async fn enroll_peer(&self, peer: EndpointId) -> Result<(), DirectTransportError> {
        if peer == self.node_id() {
            return Err(DirectTransportError::SelfConnection);
        }
        let mut pinned = self.inner.pinned_peers.write().await;
        if !pinned.contains(&peer) && pinned.len() >= self.inner.config.max_peers {
            return Err(DirectTransportError::PeerCapacityReached {
                maximum: self.inner.config.max_peers,
            });
        }
        pinned.insert(peer);
        Ok(())
    }

    pub async fn remove_peer(&self, peer: EndpointId) {
        self.inner.pinned_peers.write().await.remove(&peer);
        if self.peer_policy().await == DirectPeerPolicy::PinnedOnly {
            self.close_peer(peer, b"peer enrollment removed").await;
        }
    }

    /// Snapshot the pinned endpoint allowlist in a versioned persistence format.
    pub async fn peer_enrollments(&self) -> Result<DirectEnrollmentBook, DirectTransportError> {
        let peers = self
            .inner
            .pinned_peers
            .read()
            .await
            .iter()
            .copied()
            .collect::<Vec<_>>();
        DirectEnrollmentBook::from_peers(peers).map_err(DirectTransportError::Enrollment)
    }

    /// Atomically replace the pinned allowlist. In pinned-only mode, active
    /// connections absent from the replacement book are closed immediately.
    pub async fn replace_peer_enrollments(
        &self,
        book: DirectEnrollmentBook,
    ) -> Result<(), DirectTransportError> {
        book.validate().map_err(DirectTransportError::Enrollment)?;
        if book.peers.len() > self.inner.config.max_peers {
            return Err(DirectTransportError::PeerCapacityReached {
                maximum: self.inner.config.max_peers,
            });
        }
        let replacement = book.peers.iter().copied().collect::<HashSet<_>>();
        *self.inner.pinned_peers.write().await = replacement.clone();
        if self.peer_policy().await == DirectPeerPolicy::PinnedOnly {
            for peer in self.connected_peers().await {
                if !replacement.contains(&peer) {
                    self.close_peer(peer, b"peer enrollment book replaced")
                        .await;
                }
            }
        }
        Ok(())
    }

    /// Apply a dual-signed endpoint-key rollover and close the superseded
    /// connection. Both keys must consent, preventing an operator database edit
    /// from silently reassigning a trusted peer identity.
    pub async fn apply_peer_rollover(
        &self,
        proof: &DirectPeerRolloverProof,
    ) -> Result<(), DirectTransportError> {
        proof
            .verify_now()
            .map_err(DirectTransportError::Enrollment)?;
        {
            let mut pinned = self.inner.pinned_peers.write().await;
            if !pinned.contains(&proof.old_peer) {
                return Err(DirectTransportError::Enrollment(
                    DirectEnrollmentError::OldPeerNotEnrolled,
                ));
            }
            if pinned.contains(&proof.new_peer) {
                return Err(DirectTransportError::Enrollment(
                    DirectEnrollmentError::NewPeerAlreadyEnrolled,
                ));
            }
            pinned.remove(&proof.old_peer);
            pinned.insert(proof.new_peer);
        }
        self.close_peer(proof.old_peer, b"endpoint identity rolled over")
            .await;
        Ok(())
    }

    pub async fn connected_peers(&self) -> Vec<EndpointId> {
        self.inner
            .connections
            .read()
            .await
            .iter()
            .filter_map(|(peer, managed)| {
                managed.connection.close_reason().is_none().then_some(*peer)
            })
            .collect()
    }

    pub async fn is_connected(&self, peer: EndpointId) -> bool {
        self.connection(peer).await.is_some()
    }

    /// Dial and register a direct connection. Concurrent local dial attempts are
    /// serialized to avoid same-origin duplicate connections.
    pub async fn connect(&self, address: EndpointAddr) -> Result<EndpointId, DirectTransportError> {
        let peer = address.id;
        self.ensure_peer_allowed(peer).await?;
        if peer == self.node_id() {
            return Err(DirectTransportError::SelfConnection);
        }
        if self.connection(peer).await.is_some() {
            return Ok(peer);
        }

        let _dial_guard = self.inner.dial_gate.lock().await;
        if self.connection(peer).await.is_some() {
            return Ok(peer);
        }
        let connection = tokio::time::timeout(
            self.inner.config.operation_timeout,
            self.inner.endpoint.connect(address, DIRECT_ALPN),
        )
        .await
        .map_err(|_| DirectTransportError::OperationTimeout {
            operation: "connect",
        })?
        .map_err(|error| DirectTransportError::Iroh(error.to_string()))?;
        if connection.remote_id() != peer {
            connection.close(1u32.into(), b"unexpected endpoint identity");
            return Err(DirectTransportError::RemoteIdentityMismatch {
                expected: peer,
                received: connection.remote_id(),
            });
        }
        if self
            .register_connection(connection.clone(), DirectConnectionOrigin::Outgoing)
            .await?
        {
            let transport = self.clone();
            tokio::spawn(async move {
                transport.run_connection(connection).await;
            });
        }
        Ok(peer)
    }

    /// Spawn a bounded reconnect loop for one endpoint address. The task stops
    /// on cancellation, transport shutdown, non-retryable policy failures, or
    /// after `maximum_attempts` consecutive failed dials.
    pub async fn maintain_connection(
        &self,
        address: EndpointAddr,
        policy: ReconnectPolicy,
    ) -> Result<MaintainedPeer, DirectTransportError> {
        policy.validate()?;
        let peer = address.id;
        if peer == self.node_id() {
            return Err(DirectTransportError::SelfConnection);
        }
        self.ensure_peer_allowed(peer).await?;

        let (cancel_tx, _) = watch::channel(false);
        let (state_tx, state_rx) = watch::channel(MaintainedPeerState::Starting);
        let transport = self.clone();
        let task_cancel = cancel_tx.subscribe();
        let entropy = maintained_peer_entropy(self.session_id(), peer);
        let join = tokio::spawn(async move {
            run_maintained_peer(transport, address, policy, entropy, task_cancel, state_tx).await;
        });
        Ok(MaintainedPeer {
            peer,
            transport: self.clone(),
            cancel_tx,
            state_rx,
            task: Mutex::new(Some(join)),
        })
    }

    /// Send a reliable packet and wait for remote validation plus application
    /// queue admission.
    pub async fn send_reliable(
        &self,
        peer: EndpointId,
        lane: DirectLane,
        payload: Vec<u8>,
    ) -> Result<ReliableDeliveryReceipt, DirectTransportError> {
        self.send_reliable_inner(peer, lane, None, payload).await
    }

    /// Retry-safe reliable delivery. Reusing `operation_id` after an ambiguous
    /// timeout returns a duplicate acknowledgement without queueing the action
    /// a second time at the remote endpoint.
    pub async fn send_reliable_idempotent(
        &self,
        peer: EndpointId,
        lane: DirectLane,
        operation_id: Uuid,
        payload: Vec<u8>,
    ) -> Result<ReliableDeliveryReceipt, DirectTransportError> {
        if operation_id.is_nil() {
            return Err(DirectTransportError::InvalidEnvelope(
                DirectEnvelopeError::NilOperationId,
            ));
        }
        self.send_reliable_inner(peer, lane, Some(operation_id), payload)
            .await
    }

    async fn send_reliable_inner(
        &self,
        peer: EndpointId,
        lane: DirectLane,
        operation_id: Option<Uuid>,
        payload: Vec<u8>,
    ) -> Result<ReliableDeliveryReceipt, DirectTransportError> {
        let envelope = self.sign_packet_with_operation(
            lane,
            DirectDelivery::Reliable,
            payload,
            self.inner.config.reliable_ttl_ms,
            operation_id,
        )?;
        let encoded = encode_bounded(&envelope, MAX_DIRECT_RELIABLE_FRAME_BYTES)
            .map_err(DirectTransportError::Codec)?;
        if encoded.len() > MAX_DIRECT_RELIABLE_FRAME_BYTES {
            return Err(DirectTransportError::FrameTooLarge {
                size: encoded.len(),
                maximum: MAX_DIRECT_RELIABLE_FRAME_BYTES,
            });
        }
        let connection = self
            .connection(peer)
            .await
            .ok_or(DirectTransportError::NotConnected { peer })?;
        let receipt = DirectSendReceipt {
            peer,
            message_id: envelope.message_id,
            session_id: envelope.session_id,
            sequence: envelope.sequence,
            operation_id: envelope.operation_id,
            lane,
            delivery: DirectDelivery::Reliable,
            encoded_bytes: encoded.len(),
        };

        let operation = async {
            let (mut send, mut recv) = connection
                .open_bi()
                .await
                .map_err(|error| DirectTransportError::Iroh(error.to_string()))?;
            send.write_all(&encoded)
                .await
                .map_err(|error| DirectTransportError::Iroh(error.to_string()))?;
            send.finish()
                .map_err(|error| DirectTransportError::Iroh(error.to_string()))?;
            let response_bytes = recv
                .read_to_end(MAX_DIRECT_ACK_BYTES)
                .await
                .map_err(|error| DirectTransportError::Iroh(error.to_string()))?;
            let response: ReliableResponse = decode_bounded(&response_bytes, MAX_DIRECT_ACK_BYTES)
                .map_err(DirectTransportError::Codec)?;
            match response {
                ReliableResponse::Accepted {
                    message_id,
                    duplicate,
                } if message_id == envelope.message_id => Ok(duplicate),
                ReliableResponse::Accepted { message_id, .. } => {
                    Err(DirectTransportError::AckMessageMismatch {
                        expected: envelope.message_id,
                        received: message_id,
                    })
                }
                ReliableResponse::Rejected { reason, .. } => {
                    Err(DirectTransportError::RemoteRejected { reason })
                }
            }
        };

        match tokio::time::timeout(self.inner.config.operation_timeout, operation).await {
            Ok(Ok(remote_duplicate)) => {
                self.inner
                    .metrics
                    .reliable_sent
                    .fetch_add(1, Ordering::Relaxed);
                self.inner
                    .metrics
                    .bytes_sent
                    .fetch_add(encoded.len() as u64, Ordering::Relaxed);
                Ok(ReliableDeliveryReceipt {
                    local: receipt,
                    remote_queue_accepted: true,
                    remote_duplicate,
                })
            }
            Ok(Err(error)) => {
                self.inner
                    .metrics
                    .reliable_rejected
                    .fetch_add(1, Ordering::Relaxed);
                Err(error)
            }
            Err(_) => {
                self.inner
                    .metrics
                    .reliable_rejected
                    .fetch_add(1, Ordering::Relaxed);
                Err(DirectTransportError::OperationTimeout {
                    operation: "reliable delivery acknowledgement",
                })
            }
        }
    }

    /// Queue an unreliable packet on the current QUIC path.
    pub async fn send_datagram(
        &self,
        peer: EndpointId,
        lane: DirectLane,
        payload: Vec<u8>,
    ) -> Result<DirectSendReceipt, DirectTransportError> {
        let envelope = self.sign_packet(
            lane,
            DirectDelivery::Datagram,
            payload,
            self.inner.config.datagram_ttl_ms,
        )?;
        let encoded = encode_bounded(&envelope, MAX_DIRECT_DATAGRAM_FRAME_BYTES)
            .map_err(DirectTransportError::Codec)?;
        if encoded.len() > MAX_DIRECT_DATAGRAM_FRAME_BYTES {
            return Err(DirectTransportError::FrameTooLarge {
                size: encoded.len(),
                maximum: MAX_DIRECT_DATAGRAM_FRAME_BYTES,
            });
        }
        let connection = self
            .connection(peer)
            .await
            .ok_or(DirectTransportError::NotConnected { peer })?;
        let path_maximum = connection
            .max_datagram_size()
            .ok_or(DirectTransportError::DatagramsUnsupported { peer })?;
        if encoded.len() > path_maximum {
            return Err(DirectTransportError::DatagramExceedsPathMtu {
                size: encoded.len(),
                maximum: path_maximum,
            });
        }
        connection
            .send_datagram(encoded.clone().into())
            .map_err(|error| DirectTransportError::Iroh(error.to_string()))?;
        self.inner
            .metrics
            .datagrams_sent
            .fetch_add(1, Ordering::Relaxed);
        self.inner
            .metrics
            .bytes_sent
            .fetch_add(encoded.len() as u64, Ordering::Relaxed);
        Ok(DirectSendReceipt {
            peer,
            message_id: envelope.message_id,
            session_id: envelope.session_id,
            sequence: envelope.sequence,
            operation_id: envelope.operation_id,
            lane,
            delivery: DirectDelivery::Datagram,
            encoded_bytes: encoded.len(),
        })
    }

    pub async fn close_peer(&self, peer: EndpointId, reason: &[u8]) {
        if let Some(managed) = self.inner.connections.write().await.remove(&peer) {
            managed.connection.close(0u32.into(), reason);
            self.inner
                .metrics
                .connections_closed
                .fetch_add(1, Ordering::Relaxed);
            self.emit_event(DirectEvent::PeerDisconnected {
                peer,
                reason: String::from_utf8_lossy(reason).into_owned(),
            });
        }
    }

    pub async fn shutdown(&self) {
        self.inner.shutdown_tx.send_replace(true);
        let connections = self
            .inner
            .connections
            .write()
            .await
            .drain()
            .map(|(_, managed)| managed.connection)
            .collect::<Vec<_>>();
        self.inner
            .metrics
            .connections_closed
            .fetch_add(connections.len() as u64, Ordering::Relaxed);
        for connection in connections {
            connection.close(0u32.into(), b"direct transport shutdown");
        }
    }

    fn sign_packet(
        &self,
        lane: DirectLane,
        delivery: DirectDelivery,
        payload: Vec<u8>,
        ttl_ms: u64,
    ) -> Result<DirectEnvelope, DirectTransportError> {
        self.sign_packet_with_operation(lane, delivery, payload, ttl_ms, None)
    }

    fn sign_packet_with_operation(
        &self,
        lane: DirectLane,
        delivery: DirectDelivery,
        payload: Vec<u8>,
        ttl_ms: u64,
        operation_id: Option<Uuid>,
    ) -> Result<DirectEnvelope, DirectTransportError> {
        let sequence = self
            .inner
            .sequence
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .map_err(|_| DirectTransportError::SequenceExhausted)?;
        let now_ms = system_time_ms().map_err(DirectTransportError::Clock)?;
        DirectEnvelope::sign_with_operation(
            payload,
            lane,
            delivery,
            self.inner.endpoint.secret_key(),
            self.inner.session_id,
            sequence,
            now_ms,
            ttl_ms,
            operation_id,
        )
        .map_err(DirectTransportError::InvalidEnvelope)
    }

    async fn connection(&self, peer: EndpointId) -> Option<Connection> {
        self.inner
            .connections
            .read()
            .await
            .get(&peer)
            .filter(|managed| managed.connection.close_reason().is_none())
            .map(|managed| managed.connection.clone())
    }

    async fn ensure_peer_allowed(&self, peer: EndpointId) -> Result<(), DirectTransportError> {
        match *self.inner.peer_policy.read().await {
            DirectPeerPolicy::AnyAuthenticated => Ok(()),
            DirectPeerPolicy::PinnedOnly => {
                if self.inner.pinned_peers.read().await.contains(&peer) {
                    Ok(())
                } else {
                    Err(DirectTransportError::PeerNotEnrolled { peer })
                }
            }
        }
    }

    async fn register_connection(
        &self,
        candidate: Connection,
        origin: DirectConnectionOrigin,
    ) -> Result<bool, DirectTransportError> {
        let peer = candidate.remote_id();
        if peer == self.node_id() {
            candidate.close(1u32.into(), b"self connection rejected");
            return Err(DirectTransportError::SelfConnection);
        }
        if let Err(error) = self.ensure_peer_allowed(peer).await {
            candidate.close(1u32.into(), b"peer not enrolled");
            return Err(error);
        }

        let mut replaced = None;
        let accepted = {
            let mut connections = self.inner.connections.write().await;
            if !connections.contains_key(&peer) && connections.len() >= self.inner.config.max_peers
            {
                candidate.close(1u32.into(), b"direct peer capacity reached");
                return Err(DirectTransportError::PeerCapacityReached {
                    maximum: self.inner.config.max_peers,
                });
            }

            match connections.get(&peer) {
                None => {
                    connections.insert(
                        peer,
                        ManagedConnection {
                            connection: candidate.clone(),
                            origin,
                        },
                    );
                    true
                }
                Some(existing) if existing.connection.close_reason().is_some() => {
                    replaced = connections.insert(
                        peer,
                        ManagedConnection {
                            connection: candidate.clone(),
                            origin,
                        },
                    );
                    true
                }
                Some(existing) => {
                    let prefer_outgoing = self.node_id().as_bytes() < peer.as_bytes();
                    let candidate_preferred =
                        matches!(origin, DirectConnectionOrigin::Outgoing) == prefer_outgoing;
                    let existing_preferred =
                        matches!(existing.origin, DirectConnectionOrigin::Outgoing)
                            == prefer_outgoing;
                    if candidate_preferred && !existing_preferred {
                        replaced = connections.insert(
                            peer,
                            ManagedConnection {
                                connection: candidate.clone(),
                                origin,
                            },
                        );
                        true
                    } else {
                        candidate.close(0u32.into(), b"duplicate direct connection");
                        false
                    }
                }
            }
        };

        if let Some(previous) = replaced {
            previous
                .connection
                .close(0u32.into(), b"superseded direct connection");
            self.inner
                .metrics
                .connections_replaced
                .fetch_add(1, Ordering::Relaxed);
            self.emit_event(DirectEvent::PeerReplaced {
                peer,
                retained_origin: origin,
            });
        } else if accepted {
            self.inner
                .metrics
                .connections_opened
                .fetch_add(1, Ordering::Relaxed);
            self.emit_event(DirectEvent::PeerConnected { peer, origin });
        }
        Ok(accepted)
    }

    async fn accept_connection(&self, connection: Connection) -> Result<(), DirectTransportError> {
        if self
            .register_connection(connection.clone(), DirectConnectionOrigin::Incoming)
            .await?
        {
            self.run_connection(connection).await;
        }
        Ok(())
    }

    async fn run_connection(&self, connection: Connection) {
        let peer = connection.remote_id();
        let stable_id = connection.stable_id();
        let stream_limit = Arc::new(Semaphore::new(
            self.inner.config.max_concurrent_streams_per_peer,
        ));
        let mut shutdown_rx = self.inner.shutdown_tx.subscribe();
        let disconnect_reason = loop {
            tokio::select! {
                changed = shutdown_rx.changed() => {
                    if changed.is_err() || *shutdown_rx.borrow() {
                        break "local shutdown".to_string();
                    }
                }
                accepted = connection.accept_bi() => {
                    match accepted {
                        Ok((send, recv)) => {
                            match Arc::clone(&stream_limit).try_acquire_owned() {
                                Ok(permit) => {
                                    let transport = self.clone();
                                    tokio::spawn(async move {
                                        let _permit = permit;
                                        transport.handle_reliable_stream(peer, send, recv).await;
                                    });
                                }
                                Err(_) => {
                                    self.inner
                                        .metrics
                                        .reliable_rejected
                                        .fetch_add(1, Ordering::Relaxed);
                                    drop(send);
                                    drop(recv);
                                    connection.close(1u32.into(), b"reliable stream capacity exceeded");
                                    break "reliable stream capacity exceeded".to_string();
                                }
                            }
                        }
                        Err(error) => break format!("reliable stream accept failed: {error}"),
                    }
                }
                datagram = connection.read_datagram() => {
                    match datagram {
                        Ok(bytes) => self.handle_datagram(peer, &bytes).await,
                        Err(error) => break format!("datagram receive failed: {error}"),
                    }
                }
                reason = connection.closed() => {
                    break reason.to_string();
                }
            }
        };

        if self.remove_connection_if_current(peer, stable_id).await {
            self.inner
                .metrics
                .connections_closed
                .fetch_add(1, Ordering::Relaxed);
            self.emit_event(DirectEvent::PeerDisconnected {
                peer,
                reason: disconnect_reason,
            });
        }
    }

    async fn remove_connection_if_current(&self, peer: EndpointId, stable_id: usize) -> bool {
        let mut connections = self.inner.connections.write().await;
        let remove = connections
            .get(&peer)
            .map(|managed| managed.connection.stable_id() == stable_id)
            .unwrap_or(false);
        if remove {
            connections.remove(&peer);
        }
        remove
    }

    async fn handle_reliable_stream(
        &self,
        peer: EndpointId,
        send: SendStream,
        mut recv: RecvStream,
    ) {
        let bytes = match recv.read_to_end(MAX_DIRECT_RELIABLE_FRAME_BYTES).await {
            Ok(bytes) => bytes,
            Err(_) => {
                self.inner
                    .metrics
                    .invalid_packets
                    .fetch_add(1, Ordering::Relaxed);
                let _ = send_reliable_response(
                    send,
                    ReliableResponse::Rejected {
                        message_id: None,
                        reason: ReliableRejectCode::FrameTooLarge,
                    },
                )
                .await;
                self.emit_rejection(
                    peer,
                    DirectDelivery::Reliable,
                    DirectRejectReason::FrameTooLarge {
                        size: MAX_DIRECT_RELIABLE_FRAME_BYTES.saturating_add(1),
                        maximum: MAX_DIRECT_RELIABLE_FRAME_BYTES,
                    },
                );
                return;
            }
        };
        self.inner
            .metrics
            .bytes_received
            .fetch_add(bytes.len() as u64, Ordering::Relaxed);
        let envelope: DirectEnvelope = match decode_bounded(&bytes, MAX_DIRECT_RELIABLE_FRAME_BYTES)
        {
            Ok(envelope) => envelope,
            Err(error) => {
                self.inner
                    .metrics
                    .invalid_packets
                    .fetch_add(1, Ordering::Relaxed);
                let _ = send_reliable_response(
                    send,
                    ReliableResponse::Rejected {
                        message_id: None,
                        reason: ReliableRejectCode::DecodeFailed,
                    },
                )
                .await;
                self.emit_rejection(
                    peer,
                    DirectDelivery::Reliable,
                    DirectRejectReason::Decode(error),
                );
                return;
            }
        };
        let message_id = envelope.message_id;
        match self
            .validate_inbound_envelope(peer, DirectDelivery::Reliable, &envelope)
            .await
        {
            Ok(()) => {}
            Err((code, reason)) => {
                self.inner
                    .metrics
                    .invalid_packets
                    .fetch_add(1, Ordering::Relaxed);
                let _ = send_reliable_response(
                    send,
                    ReliableResponse::Rejected {
                        message_id: Some(message_id),
                        reason: code,
                    },
                )
                .await;
                self.emit_rejection(peer, DirectDelivery::Reliable, reason);
                return;
            }
        }

        let operation_id = envelope.operation_id;
        let author = envelope.author;
        let fingerprint = operation_id.map(|_| operation_fingerprint(&envelope));
        let event = DirectEvent::Reliable(envelope.into());

        let send_result =
            if let (Some(operation_id), Some(fingerprint)) = (operation_id, fingerprint) {
                let mut operations = self.inner.operations.lock().await;
                match operations.claim(author, operation_id, fingerprint) {
                    OperationClaim::Accepted => {
                        drop(operations);
                        self.inner
                            .metrics
                            .idempotent_duplicates
                            .fetch_add(1, Ordering::Relaxed);
                        let _ = send_reliable_response(
                            send,
                            ReliableResponse::Accepted {
                                message_id,
                                duplicate: true,
                            },
                        )
                        .await;
                        return;
                    }
                    OperationClaim::Pending => {
                        drop(operations);
                        self.inner
                            .metrics
                            .idempotent_in_progress
                            .fetch_add(1, Ordering::Relaxed);
                        self.inner
                            .metrics
                            .reliable_rejected
                            .fetch_add(1, Ordering::Relaxed);
                        let _ = send_reliable_response(
                            send,
                            ReliableResponse::Rejected {
                                message_id: Some(message_id),
                                reason: ReliableRejectCode::OperationInProgress,
                            },
                        )
                        .await;
                        self.emit_rejection(
                            peer,
                            DirectDelivery::Reliable,
                            DirectRejectReason::OperationInProgress { operation_id },
                        );
                        return;
                    }
                    OperationClaim::Conflict => {
                        drop(operations);
                        self.inner
                            .metrics
                            .idempotent_conflicts
                            .fetch_add(1, Ordering::Relaxed);
                        self.inner
                            .metrics
                            .reliable_rejected
                            .fetch_add(1, Ordering::Relaxed);
                        let _ = send_reliable_response(
                            send,
                            ReliableResponse::Rejected {
                                message_id: Some(message_id),
                                reason: ReliableRejectCode::OperationConflict,
                            },
                        )
                        .await;
                        self.emit_rejection(
                            peer,
                            DirectDelivery::Reliable,
                            DirectRejectReason::OperationConflict { operation_id },
                        );
                        return;
                    }
                    OperationClaim::New => {
                        let result = self.inner.event_tx.try_send(event);
                        match &result {
                            Ok(()) => operations.accept(author, operation_id),
                            Err(_) => operations.release(author, operation_id),
                        }
                        result
                    }
                }
            } else {
                self.inner.event_tx.try_send(event)
            };

        match send_result {
            Ok(()) => {
                self.inner
                    .metrics
                    .reliable_accepted
                    .fetch_add(1, Ordering::Relaxed);
                let _ = send_reliable_response(
                    send,
                    ReliableResponse::Accepted {
                        message_id,
                        duplicate: false,
                    },
                )
                .await;
            }
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.inner
                    .metrics
                    .event_queue_full
                    .fetch_add(1, Ordering::Relaxed);
                self.inner
                    .metrics
                    .reliable_rejected
                    .fetch_add(1, Ordering::Relaxed);
                let _ = send_reliable_response(
                    send,
                    ReliableResponse::Rejected {
                        message_id: Some(message_id),
                        reason: ReliableRejectCode::ApplicationQueueFull,
                    },
                )
                .await;
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                self.inner
                    .metrics
                    .reliable_rejected
                    .fetch_add(1, Ordering::Relaxed);
                let _ = send_reliable_response(
                    send,
                    ReliableResponse::Rejected {
                        message_id: Some(message_id),
                        reason: ReliableRejectCode::ApplicationQueueClosed,
                    },
                )
                .await;
            }
        }
    }

    async fn handle_datagram(&self, peer: EndpointId, bytes: &[u8]) {
        self.inner
            .metrics
            .datagrams_received
            .fetch_add(1, Ordering::Relaxed);
        self.inner
            .metrics
            .bytes_received
            .fetch_add(bytes.len() as u64, Ordering::Relaxed);
        if !self.inner.datagram_rate_limiter.lock().await.allow(
            peer,
            bytes.len(),
            std::time::Instant::now(),
        ) {
            self.inner
                .metrics
                .rate_limited
                .fetch_add(1, Ordering::Relaxed);
            self.emit_rejection(
                peer,
                DirectDelivery::Datagram,
                DirectRejectReason::RateLimited { size: bytes.len() },
            );
            return;
        }
        if bytes.len() > MAX_DIRECT_DATAGRAM_FRAME_BYTES {
            self.inner
                .metrics
                .invalid_packets
                .fetch_add(1, Ordering::Relaxed);
            self.emit_rejection(
                peer,
                DirectDelivery::Datagram,
                DirectRejectReason::FrameTooLarge {
                    size: bytes.len(),
                    maximum: MAX_DIRECT_DATAGRAM_FRAME_BYTES,
                },
            );
            return;
        }
        let envelope: DirectEnvelope = match decode_bounded(bytes, MAX_DIRECT_DATAGRAM_FRAME_BYTES)
        {
            Ok(envelope) => envelope,
            Err(error) => {
                self.inner
                    .metrics
                    .invalid_packets
                    .fetch_add(1, Ordering::Relaxed);
                self.emit_rejection(
                    peer,
                    DirectDelivery::Datagram,
                    DirectRejectReason::Decode(error),
                );
                return;
            }
        };
        if let Err((_, reason)) = self
            .validate_inbound_envelope(peer, DirectDelivery::Datagram, &envelope)
            .await
        {
            self.inner
                .metrics
                .invalid_packets
                .fetch_add(1, Ordering::Relaxed);
            self.emit_rejection(peer, DirectDelivery::Datagram, reason);
            return;
        }
        match self
            .inner
            .event_tx
            .try_send(DirectEvent::Datagram(envelope.into()))
        {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.inner
                    .metrics
                    .datagrams_dropped
                    .fetch_add(1, Ordering::Relaxed);
                self.inner
                    .metrics
                    .event_queue_full
                    .fetch_add(1, Ordering::Relaxed);
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                self.inner
                    .metrics
                    .datagrams_dropped
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    async fn validate_inbound_envelope(
        &self,
        peer: EndpointId,
        expected_delivery: DirectDelivery,
        envelope: &DirectEnvelope,
    ) -> Result<(), (ReliableRejectCode, DirectRejectReason)> {
        let now_ms = match system_time_ms() {
            Ok(now_ms) => now_ms,
            Err(error) => {
                return Err((
                    ReliableRejectCode::InvalidEnvelope,
                    DirectRejectReason::Decode(error),
                ));
            }
        };
        if let Err(error) = envelope.verify_at(now_ms) {
            return Err((
                ReliableRejectCode::InvalidEnvelope,
                DirectRejectReason::InvalidEnvelope(error),
            ));
        }
        if envelope.author != peer {
            return Err((
                ReliableRejectCode::AuthorMismatch,
                DirectRejectReason::AuthorMismatch {
                    authenticated_peer: peer,
                    claimed_author: envelope.author,
                },
            ));
        }
        if envelope.delivery != expected_delivery {
            return Err((
                ReliableRejectCode::DeliveryMismatch,
                DirectRejectReason::DeliveryMismatch {
                    expected: expected_delivery,
                    received: envelope.delivery,
                },
            ));
        }
        if !self.inner.replay.lock().await.insert(envelope.message_id) {
            self.inner
                .metrics
                .replayed_packets
                .fetch_add(1, Ordering::Relaxed);
            return Err((
                ReliableRejectCode::Replay,
                DirectRejectReason::Replay {
                    session_id: envelope.session_id,
                    sequence: envelope.sequence,
                },
            ));
        }
        if !self.inner.sequence_windows.lock().await.insert(
            envelope.author,
            envelope.session_id,
            envelope.sequence,
        ) {
            self.inner
                .metrics
                .replayed_packets
                .fetch_add(1, Ordering::Relaxed);
            return Err((
                ReliableRejectCode::Replay,
                DirectRejectReason::Replay {
                    session_id: envelope.session_id,
                    sequence: envelope.sequence,
                },
            ));
        }
        Ok(())
    }

    fn is_retryable_connect_error(error: &DirectTransportError) -> bool {
        !matches!(
            error,
            DirectTransportError::InvalidConfig(_)
                | DirectTransportError::InvalidReconnectPolicy(_)
                | DirectTransportError::SelfConnection
                | DirectTransportError::PeerNotEnrolled { .. }
                | DirectTransportError::PeerCapacityReached { .. }
                | DirectTransportError::RemoteIdentityMismatch { .. }
                | DirectTransportError::SequenceExhausted
        )
    }

    fn emit_rejection(
        &self,
        peer: EndpointId,
        delivery: DirectDelivery,
        reason: DirectRejectReason,
    ) {
        self.emit_event(DirectEvent::Rejected {
            peer,
            delivery,
            reason,
        });
    }

    fn emit_event(&self, event: DirectEvent) {
        if let Err(mpsc::error::TrySendError::Full(_)) = self.inner.event_tx.try_send(event) {
            self.inner
                .metrics
                .event_queue_full
                .fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Handle for a reconnecting peer task.
pub struct MaintainedPeer {
    peer: EndpointId,
    transport: DirectTransport,
    cancel_tx: watch::Sender<bool>,
    state_rx: watch::Receiver<MaintainedPeerState>,
    task: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl fmt::Debug for MaintainedPeer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MaintainedPeer")
            .field("peer", &self.peer)
            .field("state", &self.current_state())
            .finish_non_exhaustive()
    }
}

impl MaintainedPeer {
    pub fn peer(&self) -> EndpointId {
        self.peer
    }

    pub fn state(&self) -> watch::Receiver<MaintainedPeerState> {
        self.state_rx.clone()
    }

    pub fn current_state(&self) -> MaintainedPeerState {
        self.state_rx.borrow().clone()
    }

    /// Stop reconnecting but leave an established shared connection intact.
    pub async fn stop(&self) {
        self.cancel_tx.send_replace(true);
        let task = { self.task.lock().await.take() };
        if let Some(task) = task {
            let _ = task.await;
        }
    }

    /// Stop reconnecting and close the peer's current connection.
    pub async fn stop_and_close(&self, reason: &[u8]) {
        self.stop().await;
        self.transport.close_peer(self.peer, reason).await;
    }
}

async fn run_maintained_peer(
    transport: DirectTransport,
    address: EndpointAddr,
    policy: ReconnectPolicy,
    entropy: u64,
    mut cancel_rx: watch::Receiver<bool>,
    state_tx: watch::Sender<MaintainedPeerState>,
) {
    let peer = address.id;
    let mut shutdown_rx = transport.inner.shutdown_tx.subscribe();
    let mut failed_attempts = 0u32;

    loop {
        if *cancel_rx.borrow() || *shutdown_rx.borrow() {
            state_tx.send_replace(MaintainedPeerState::Stopped);
            return;
        }

        if transport.is_connected(peer).await {
            failed_attempts = 0;
            state_tx.send_replace(MaintainedPeerState::Connected);
            tokio::select! {
                changed = cancel_rx.changed() => {
                    if changed.is_err() || *cancel_rx.borrow() {
                        state_tx.send_replace(MaintainedPeerState::Stopped);
                        return;
                    }
                }
                changed = shutdown_rx.changed() => {
                    if changed.is_err() || *shutdown_rx.borrow() {
                        state_tx.send_replace(MaintainedPeerState::Stopped);
                        return;
                    }
                }
                _ = tokio::time::sleep(policy.health_probe_interval) => {}
            }
            continue;
        }

        let attempt = failed_attempts.saturating_add(1);
        state_tx.send_replace(MaintainedPeerState::Connecting { attempt });
        match transport.connect(address.clone()).await {
            Ok(_) => {
                failed_attempts = 0;
                state_tx.send_replace(MaintainedPeerState::Connected);
            }
            Err(error) => {
                failed_attempts = attempt;
                if !DirectTransport::is_retryable_connect_error(&error) {
                    state_tx.send_replace(MaintainedPeerState::Failed(error.to_string()));
                    return;
                }
                if policy
                    .maximum_attempts
                    .is_some_and(|maximum| failed_attempts >= maximum)
                {
                    state_tx.send_replace(MaintainedPeerState::Failed(error.to_string()));
                    return;
                }
                let delay = policy.delay_for(failed_attempts, entropy);
                state_tx.send_replace(MaintainedPeerState::BackingOff {
                    attempt: failed_attempts,
                    delay,
                    last_error: error.to_string(),
                });
                tokio::select! {
                    changed = cancel_rx.changed() => {
                        if changed.is_err() || *cancel_rx.borrow() {
                            state_tx.send_replace(MaintainedPeerState::Stopped);
                            return;
                        }
                    }
                    changed = shutdown_rx.changed() => {
                        if changed.is_err() || *shutdown_rx.borrow() {
                            state_tx.send_replace(MaintainedPeerState::Stopped);
                            return;
                        }
                    }
                    _ = tokio::time::sleep(delay) => {}
                }
            }
        }
    }
}

fn maintained_peer_entropy(session_id: Uuid, peer: EndpointId) -> u64 {
    let session = session_id.as_u128();
    let mut peer_prefix = [0u8; 8];
    peer_prefix.copy_from_slice(&peer.as_bytes()[..8]);
    (session as u64) ^ ((session >> 64) as u64) ^ u64::from_le_bytes(peer_prefix)
}

/// Cloneable protocol handler registered under [`DIRECT_ALPN`].
#[derive(Clone)]
pub struct DirectProtocol {
    transport: DirectTransport,
}

impl fmt::Debug for DirectProtocol {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DirectProtocol")
            .field("endpoint", &self.transport.node_id())
            .finish()
    }
}

impl ProtocolHandler for DirectProtocol {
    async fn accept(&self, connection: Connection) -> Result<(), AcceptError> {
        self.transport
            .accept_connection(connection)
            .await
            .map_err(|error| AcceptError::from_err(error))
    }

    async fn shutdown(&self) {
        self.transport.shutdown().await;
    }
}

async fn send_reliable_response(
    mut send: SendStream,
    response: ReliableResponse,
) -> Result<(), DirectTransportError> {
    let bytes =
        encode_bounded(&response, MAX_DIRECT_ACK_BYTES).map_err(DirectTransportError::Codec)?;
    send.write_all(&bytes)
        .await
        .map_err(|error| DirectTransportError::Iroh(error.to_string()))?;
    send.finish()
        .map_err(|error| DirectTransportError::Iroh(error.to_string()))?;
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum DirectTransportError {
    #[error("invalid direct transport configuration: {0}")]
    InvalidConfig(&'static str),
    #[error("invalid reconnect policy: {0}")]
    InvalidReconnectPolicy(&'static str),
    #[error("cannot connect the direct transport to its own endpoint")]
    SelfConnection,
    #[error("peer {peer} is not enrolled for pinned-only direct transport")]
    PeerNotEnrolled { peer: EndpointId },
    #[error("direct peer capacity reached; maximum is {maximum}")]
    PeerCapacityReached { maximum: usize },
    #[error("direct peer {peer} is not connected")]
    NotConnected { peer: EndpointId },
    #[error("direct datagrams are unsupported by peer {peer}")]
    DatagramsUnsupported { peer: EndpointId },
    #[error("datagram is {size} bytes but the active path maximum is {maximum}")]
    DatagramExceedsPathMtu { size: usize, maximum: usize },
    #[error("direct frame is {size} bytes; maximum is {maximum}")]
    FrameTooLarge { size: usize, maximum: usize },
    #[error("remote endpoint identity mismatch: expected {expected}, received {received}")]
    RemoteIdentityMismatch {
        expected: EndpointId,
        received: EndpointId,
    },
    #[error("remote direct protocol rejected the message: {reason:?}")]
    RemoteRejected { reason: ReliableRejectCode },
    #[error("reliable acknowledgement referenced {received}, expected {expected}")]
    AckMessageMismatch { expected: Uuid, received: Uuid },
    #[error("direct operation timed out: {operation}")]
    OperationTimeout { operation: &'static str },
    #[error("the local direct sequence is exhausted; create a new transport session")]
    SequenceExhausted,
    #[error("invalid direct envelope: {0}")]
    InvalidEnvelope(DirectEnvelopeError),
    #[error("direct peer enrollment failed: {0}")]
    Enrollment(DirectEnrollmentError),
    #[error("direct codec error: {0}")]
    Codec(String),
    #[error("Iroh direct transport error: {0}")]
    Iroh(String),
    #[error("system clock error: {0}")]
    Clock(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_envelope_detects_payload_tampering() {
        let key = SecretKey::from_bytes(&[21u8; 32]);
        let now = 1_000_000;
        let mut envelope = DirectEnvelope::sign(
            b"input-17".to_vec(),
            DirectLane::PLAYER_INPUT,
            DirectDelivery::Datagram,
            &key,
            Uuid::from_u128(7),
            1,
            now,
            1_000,
        )
        .unwrap();
        assert!(envelope.verify_at(now).is_ok());
        envelope.payload[0] ^= 1;
        assert_eq!(
            envelope.verify_at(now),
            Err(DirectEnvelopeError::InvalidSignature)
        );
    }

    #[test]
    fn direct_delivery_is_signature_bound() {
        let key = SecretKey::from_bytes(&[22u8; 32]);
        let now = 1_000_000;
        let mut envelope = DirectEnvelope::sign(
            b"control".to_vec(),
            DirectLane::CONTROL,
            DirectDelivery::Reliable,
            &key,
            Uuid::from_u128(8),
            1,
            now,
            1_000,
        )
        .unwrap();
        envelope.delivery = DirectDelivery::Datagram;
        assert_eq!(
            envelope.verify_at(now),
            Err(DirectEnvelopeError::InvalidSignature)
        );
    }

    #[test]
    fn datagram_payloads_are_bounded_before_signing() {
        let key = SecretKey::from_bytes(&[23u8; 32]);
        let result = DirectEnvelope::sign(
            vec![0; MAX_DIRECT_DATAGRAM_PAYLOAD_BYTES + 1],
            DirectLane::STATE_SNAPSHOT,
            DirectDelivery::Datagram,
            &key,
            Uuid::from_u128(9),
            1,
            1_000,
            1_000,
        );
        assert!(matches!(
            result,
            Err(DirectEnvelopeError::PayloadTooLarge {
                delivery: DirectDelivery::Datagram,
                ..
            })
        ));
    }

    #[test]
    fn reconnect_backoff_is_bounded_and_jittered() {
        let policy = ReconnectPolicy {
            initial_backoff: Duration::from_millis(100),
            maximum_backoff: Duration::from_secs(2),
            health_probe_interval: Duration::from_millis(100),
            maximum_attempts: Some(5),
        };
        let first = policy.delay_for(1, 7);
        let later = policy.delay_for(20, 7);
        assert!(first >= Duration::from_millis(100));
        assert!(first <= Duration::from_millis(120));
        assert!(later <= Duration::from_secs(2));
    }

    #[test]
    fn invalid_reconnect_policy_is_rejected() {
        let policy = ReconnectPolicy {
            initial_backoff: Duration::ZERO,
            ..ReconnectPolicy::default()
        };
        assert!(matches!(
            policy.validate(),
            Err(DirectTransportError::InvalidReconnectPolicy(_))
        ));
    }

    #[test]
    fn direct_capabilities_do_not_overclaim_domain_application() {
        let capabilities = DirectTransportCapabilities::DATA_PLANE_V1;
        assert!(capabilities.reliable_streams);
        assert!(capabilities.unreliable_datagrams);
        assert!(capabilities.remote_queue_acknowledgements);
        assert!(!capabilities.domain_apply_acknowledgements);
        assert!(!capabilities.peer_discovery);
    }
    #[test]
    fn operation_id_is_signature_bound_and_reliable_only() {
        let key = SecretKey::from_bytes(&[24u8; 32]);
        let operation_id = Uuid::from_u128(44);
        let now = 1_000_000;
        let mut envelope = DirectEnvelope::sign_with_operation(
            b"apply-once".to_vec(),
            DirectLane::CONTROL,
            DirectDelivery::Reliable,
            &key,
            Uuid::from_u128(10),
            1,
            now,
            1_000,
            Some(operation_id),
        )
        .unwrap();
        assert!(envelope.verify_at(now).is_ok());
        envelope.operation_id = Some(Uuid::from_u128(45));
        assert_eq!(
            envelope.verify_at(now),
            Err(DirectEnvelopeError::InvalidSignature)
        );

        assert!(matches!(
            DirectEnvelope::sign_with_operation(
                Vec::new(),
                DirectLane::PLAYER_INPUT,
                DirectDelivery::Datagram,
                &key,
                Uuid::from_u128(10),
                2,
                now,
                1_000,
                Some(operation_id),
            ),
            Err(DirectEnvelopeError::OperationIdRequiresReliable)
        ));
    }

    #[test]
    fn operation_window_reserves_then_deduplicates() {
        let key = SecretKey::from_bytes(&[25u8; 32]);
        let author = key.public();
        let operation_id = Uuid::from_u128(99);
        let fingerprint = [7u8; 32];
        let mut window = OperationWindow::new(2);
        assert_eq!(
            window.claim(author, operation_id, fingerprint),
            OperationClaim::New
        );
        assert_eq!(
            window.claim(author, operation_id, fingerprint),
            OperationClaim::Pending
        );
        window.accept(author, operation_id);
        assert_eq!(
            window.claim(author, operation_id, fingerprint),
            OperationClaim::Accepted
        );
        assert_eq!(
            window.claim(author, operation_id, [8u8; 32]),
            OperationClaim::Conflict
        );
    }
}
