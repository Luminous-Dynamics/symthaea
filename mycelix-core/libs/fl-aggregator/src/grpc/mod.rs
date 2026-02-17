//! gRPC API for Federated Learning Aggregator.
//!
//! High-performance binary protocol for FL coordination with support for:
//! - Node registration and session management
//! - Unified payload submission (dense, sparse, hyper, binary, quantized)
//! - Real-time round event streaming
//! - Byzantine-resistant aggregation
//!
//! ## Feature Flag
//!
//! This module requires the `grpc` feature flag:
//!
//! ```toml
//! [dependencies]
//! fl-aggregator = { version = "0.1", features = ["grpc"] }
//! ```
//!
//! ## Required Dependencies
//!
//! When the `grpc` feature is enabled, the following dependencies are used:
//! - `tonic` - gRPC framework for Rust
//! - `prost` - Protocol Buffers implementation
//! - `prost-types` - Well-known Protocol Buffer types
//! - `tokio-stream` - Async stream utilities
//!
//! ## Service Definition
//!
//! The service implements the FederatedLearning gRPC service:
//!
//! ```protobuf
//! service FederatedLearning {
//!   rpc RegisterNode(RegisterNodeRequest) returns (RegisterNodeResponse);
//!   rpc Submit(SubmitRequest) returns (SubmitResponse);
//!   rpc GetStatus(GetStatusRequest) returns (GetStatusResponse);
//!   rpc GetAggregated(GetAggregatedRequest) returns (GetAggregatedResponse);
//!   rpc WatchRounds(WatchRoundsRequest) returns (stream RoundEvent);
//! }
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use fl_aggregator::grpc::{FederatedLearningServer, ServiceState, GrpcServiceConfig};
//! use fl_aggregator::{AggregatorConfig, AsyncAggregator};
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() {
//!     let config = AggregatorConfig::default();
//!     let aggregator = AsyncAggregator::new(config);
//!
//!     let state = Arc::new(ServiceState::new(aggregator));
//!     let server = FederatedLearningServer::new(state, GrpcServiceConfig::default());
//!
//!     // Use with tonic transport
//!     // Server::builder().add_service(server.into_service()).serve(addr).await;
//! }
//! ```

pub mod conversions;
pub mod service;

// Re-export primary types
pub use conversions::*;
pub use service::*;

// ============================================================================
// Protocol Buffer Stub Types
// ============================================================================
//
// These types mirror the proto schema in proto/mycelix_fl.proto.
// They use prost derive macros for serialization but don't require
// proto compilation (build.rs), making the module structure compile
// independently.
//
// For full proto compilation, add to build.rs:
// ```rust
// fn main() {
//     tonic_build::configure()
//         .build_server(true)
//         .build_client(true)
//         .compile(&["proto/mycelix_fl.proto"], &["proto/"])
//         .unwrap();
// }
// ```
// ============================================================================

use std::collections::HashMap;

/// Payload type enumeration (matches proto PayloadType).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, prost::Enumeration)]
#[repr(i32)]
pub enum GrpcPayloadType {
    Unspecified = 0,
    DenseGradient = 1,
    HyperEncoded = 2,
    BinaryHypervector = 3,
    SparseGradient = 4,
    QuantizedGradient = 5,
}

/// Round state enumeration (matches proto RoundState).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, prost::Enumeration)]
#[repr(i32)]
pub enum GrpcRoundState {
    Unspecified = 0,
    Pending = 1,
    Collecting = 2,
    Aggregating = 3,
    Complete = 4,
}

/// Signature algorithm enumeration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, prost::Enumeration)]
#[repr(i32)]
pub enum GrpcSignatureAlgorithm {
    Unspecified = 0,
    Dilithium5 = 1,
    Ed25519 = 2,
    EcdsaP256 = 3,
}

/// Round event type enumeration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, prost::Enumeration)]
#[repr(i32)]
pub enum GrpcRoundEventType {
    Unspecified = 0,
    RoundStarted = 1,
    SubmissionReceived = 2,
    RoundCompleted = 3,
    ByzantineDetected = 4,
    NodeHealed = 5,
}

// ============================================================================
// Node Identity & Authentication
// ============================================================================

/// Node identity (DID-based).
#[derive(Clone, prost::Message)]
pub struct NodeIdentity {
    /// DID (e.g., "did:mycelix:node123")
    #[prost(string, tag = "1")]
    pub did: String,

    /// Dilithium5 public key
    #[prost(bytes, tag = "2")]
    pub public_key: Vec<u8>,

    /// E0-E4 identity assurance level
    #[prost(uint32, tag = "3")]
    pub assurance_level: u32,
}

/// Cryptographic signature.
#[derive(Clone, prost::Message)]
pub struct Signature {
    /// Signature bytes
    #[prost(bytes, tag = "1")]
    pub data: Vec<u8>,

    /// Algorithm used
    #[prost(enumeration = "GrpcSignatureAlgorithm", tag = "2")]
    pub algorithm: i32,

    /// Signing timestamp
    #[prost(int64, tag = "3")]
    pub timestamp: i64,

    /// 32-byte nonce for replay prevention
    #[prost(bytes, tag = "4")]
    pub nonce: Vec<u8>,
}

/// Node capabilities.
#[derive(Clone, prost::Message)]
pub struct NodeCapabilities {
    /// Supported payload types
    #[prost(enumeration = "GrpcPayloadType", repeated, tag = "1")]
    pub supported_payloads: Vec<i32>,

    /// Whether streaming is supported
    #[prost(bool, tag = "2")]
    pub supports_streaming: bool,

    /// Whether proofs are supported
    #[prost(bool, tag = "3")]
    pub supports_proofs: bool,

    /// Maximum gradient dimension
    #[prost(uint64, tag = "4")]
    pub max_gradient_dimension: u64,

    /// Framework (pytorch, tensorflow, jax, etc.)
    #[prost(string, tag = "5")]
    pub framework: String,
}

// ============================================================================
// Payload Types
// ============================================================================

/// Dense gradient (traditional FL).
#[derive(Clone, prost::Message)]
pub struct GrpcDenseGradient {
    /// Gradient values
    #[prost(float, repeated, packed = "true", tag = "1")]
    pub values: Vec<f32>,

    /// Expected dimension
    #[prost(uint64, tag = "2")]
    pub dimension: u64,
}

/// HyperFeel-encoded gradient.
#[derive(Clone, prost::Message)]
pub struct GrpcHyperEncodedGradient {
    /// 16,384 x i8 = 16KB hypervector
    #[prost(bytes, tag = "1")]
    pub hypervector: Vec<u8>,

    /// Dimension (always 16384 for HyperFeel)
    #[prost(uint32, tag = "2")]
    pub dimension: u32,

    /// Encoding metadata
    #[prost(message, optional, tag = "3")]
    pub encoding: Option<EncodingMetadata>,

    /// Phi (integrated information) metrics
    #[prost(message, optional, tag = "4")]
    pub phi: Option<PhiMetrics>,
}

/// Binary hypervector (Symthaea native).
#[derive(Clone, prost::Message)]
pub struct GrpcBinaryHypervector {
    /// 16,384 bits = 2KB
    #[prost(bytes, tag = "1")]
    pub data: Vec<u8>,

    /// Bit dimension
    #[prost(uint32, tag = "2")]
    pub dimension: u32,
}

/// Sparse gradient.
#[derive(Clone, prost::Message)]
pub struct GrpcSparseGradient {
    /// Indices of non-zero values
    #[prost(uint32, repeated, packed = "true", tag = "1")]
    pub indices: Vec<u32>,

    /// Values at those indices
    #[prost(float, repeated, packed = "true", tag = "2")]
    pub values: Vec<f32>,

    /// Full dimension
    #[prost(uint64, tag = "3")]
    pub full_dimension: u64,

    /// Sparsity ratio
    #[prost(float, tag = "4")]
    pub sparsity_ratio: f32,
}

/// Quantized gradient.
#[derive(Clone, prost::Message)]
pub struct GrpcQuantizedGradient {
    /// Quantized values
    #[prost(bytes, tag = "1")]
    pub data: Vec<u8>,

    /// Bits per value (1, 2, 4, or 8)
    #[prost(uint32, tag = "2")]
    pub bits_per_value: u32,

    /// Dequantization scale
    #[prost(float, tag = "3")]
    pub scale: f32,

    /// Dequantization offset
    #[prost(float, tag = "4")]
    pub zero_point: f32,

    /// Dimension
    #[prost(uint64, tag = "5")]
    pub dimension: u64,
}

/// Encoding metadata for HyperFeel.
#[derive(Clone, prost::Message)]
pub struct EncodingMetadata {
    /// Encoder version (e.g., "hyperfeel-v2")
    #[prost(string, tag = "1")]
    pub encoder_version: String,

    /// Quantization bits
    #[prost(uint32, tag = "2")]
    pub quantize_bits: u32,

    /// Original size in bytes
    #[prost(uint64, tag = "3")]
    pub original_size: u64,

    /// Compression ratio
    #[prost(float, tag = "4")]
    pub compression_ratio: f32,

    /// Random projection seed
    #[prost(int64, tag = "5")]
    pub projection_seed: i64,

    /// Causal encoding enabled
    #[prost(bool, tag = "6")]
    pub use_causal: bool,

    /// Temporal encoding enabled
    #[prost(bool, tag = "7")]
    pub use_temporal: bool,
}

/// Phi (integrated information) metrics.
#[derive(Clone, prost::Message)]
pub struct PhiMetrics {
    /// Phi before training step
    #[prost(float, tag = "1")]
    pub phi_before: f32,

    /// Phi after training step
    #[prost(float, tag = "2")]
    pub phi_after: f32,

    /// Phi gain
    #[prost(float, tag = "3")]
    pub phi_gain: f32,

    /// Epistemic confidence
    #[prost(float, tag = "4")]
    pub epistemic_confidence: f32,

    /// Layer-wise breakdown
    #[prost(map = "string, float", tag = "5")]
    pub components: HashMap<String, f32>,
}

/// Unified payload envelope.
#[derive(Clone, prost::Message)]
pub struct GrpcPayload {
    /// Payload data (oneof)
    #[prost(oneof = "GrpcPayloadData", tags = "1, 2, 3, 4, 5")]
    pub data: Option<GrpcPayloadData>,
}

/// Oneof for payload types.
#[derive(Clone, prost::Oneof)]
pub enum GrpcPayloadData {
    #[prost(message, tag = "1")]
    Dense(GrpcDenseGradient),
    #[prost(message, tag = "2")]
    Hyper(GrpcHyperEncodedGradient),
    #[prost(message, tag = "3")]
    Binary(GrpcBinaryHypervector),
    #[prost(message, tag = "4")]
    Sparse(GrpcSparseGradient),
    #[prost(message, tag = "5")]
    Quantized(GrpcQuantizedGradient),
}

// ============================================================================
// Request/Response Messages
// ============================================================================

/// Register node request.
#[derive(Clone, prost::Message)]
pub struct RegisterNodeRequest {
    /// Node identity
    #[prost(message, optional, tag = "1")]
    pub node: Option<NodeIdentity>,

    /// Node capabilities
    #[prost(message, optional, tag = "2")]
    pub capabilities: Option<NodeCapabilities>,
}

/// Register node response.
#[derive(Clone, prost::Message)]
pub struct RegisterNodeResponse {
    /// Whether registration was accepted
    #[prost(bool, tag = "1")]
    pub accepted: bool,

    /// Error message if not accepted
    #[prost(string, tag = "2")]
    pub error_message: String,

    /// Session token for subsequent requests
    #[prost(string, tag = "3")]
    pub session_token: String,

    /// Current round number
    #[prost(uint64, tag = "4")]
    pub current_round: u64,
}

/// Submit request.
#[derive(Clone, prost::Message)]
pub struct SubmitRequest {
    /// Node identity
    #[prost(message, optional, tag = "1")]
    pub node: Option<NodeIdentity>,

    /// Round number
    #[prost(uint64, tag = "2")]
    pub round: u64,

    /// Payload
    #[prost(message, optional, tag = "3")]
    pub payload: Option<GrpcPayload>,

    /// Signature
    #[prost(message, optional, tag = "4")]
    pub signature: Option<Signature>,

    /// Proof of Gradient Quality
    #[prost(bytes, tag = "5")]
    pub pogq_proof: Vec<u8>,

    /// zkSTARK validity proof
    #[prost(bytes, tag = "6")]
    pub zkstark_proof: Vec<u8>,

    /// Timestamp
    #[prost(int64, tag = "7")]
    pub timestamp: i64,

    /// Additional metadata
    #[prost(map = "string, string", tag = "8")]
    pub metadata: HashMap<String, String>,
}

/// Submit response.
#[derive(Clone, prost::Message)]
pub struct SubmitResponse {
    /// Whether submission was accepted
    #[prost(bool, tag = "1")]
    pub accepted: bool,

    /// Error message if not accepted
    #[prost(string, tag = "2")]
    pub error_message: String,

    /// Round number
    #[prost(uint64, tag = "3")]
    pub round: u64,

    /// Submissions received this round
    #[prost(uint32, tag = "4")]
    pub submissions_received: u32,

    /// Submissions expected this round
    #[prost(uint32, tag = "5")]
    pub submissions_expected: u32,

    /// Whether round is complete
    #[prost(bool, tag = "6")]
    pub round_complete: bool,
}

/// Get status request.
#[derive(Clone, prost::Message)]
pub struct GetStatusRequest {}

/// Get status response.
#[derive(Clone, prost::Message)]
pub struct GetStatusResponse {
    /// Current round number
    #[prost(uint64, tag = "1")]
    pub current_round: u64,

    /// Round state
    #[prost(enumeration = "GrpcRoundState", tag = "2")]
    pub round_state: i32,

    /// Registered nodes
    #[prost(uint32, tag = "3")]
    pub registered_nodes: u32,

    /// Submissions this round
    #[prost(uint32, tag = "4")]
    pub submissions_this_round: u32,

    /// Expected submissions this round
    #[prost(uint32, tag = "5")]
    pub expected_this_round: u32,

    /// Round start time (unix timestamp)
    #[prost(int64, tag = "6")]
    pub round_start_time: i64,

    /// Aggregation algorithm
    #[prost(string, tag = "7")]
    pub aggregation_algorithm: String,
}

/// Get aggregated request.
#[derive(Clone, prost::Message)]
pub struct GetAggregatedRequest {
    /// Round number (0 = latest)
    #[prost(uint64, tag = "1")]
    pub round: u64,

    /// Preferred output format
    #[prost(enumeration = "GrpcPayloadType", tag = "2")]
    pub preferred_format: i32,

    /// Include proofs
    #[prost(bool, tag = "3")]
    pub include_proofs: bool,
}

/// Aggregation metadata.
#[derive(Clone, prost::Message)]
pub struct AggregationMetadata {
    /// Algorithm used
    #[prost(string, tag = "1")]
    pub algorithm: String,

    /// Number of participants
    #[prost(uint32, tag = "2")]
    pub participants: u32,

    /// Number excluded
    #[prost(uint32, tag = "3")]
    pub excluded: u32,

    /// Byzantine ratio
    #[prost(float, tag = "4")]
    pub byzantine_ratio: f32,

    /// Aggregation time in ms
    #[prost(int64, tag = "5")]
    pub aggregation_time_ms: i64,

    /// Shapley values (node contribution scores)
    #[prost(map = "string, float", tag = "6")]
    pub shapley_values: HashMap<String, f32>,
}

/// Get aggregated response.
#[derive(Clone, prost::Message)]
pub struct GetAggregatedResponse {
    /// Round number
    #[prost(uint64, tag = "1")]
    pub round: u64,

    /// Aggregated payload
    #[prost(message, optional, tag = "2")]
    pub aggregated: Option<GrpcPayload>,

    /// Aggregation metadata
    #[prost(message, optional, tag = "3")]
    pub metadata: Option<AggregationMetadata>,

    /// Byzantine nodes detected
    #[prost(string, repeated, tag = "4")]
    pub byzantine_nodes: Vec<String>,

    /// Nodes that were healed
    #[prost(string, repeated, tag = "5")]
    pub healed_nodes: Vec<String>,

    /// Aggregation proof
    #[prost(bytes, tag = "6")]
    pub aggregation_proof: Vec<u8>,
}

/// Watch rounds request.
#[derive(Clone, prost::Message)]
pub struct WatchRoundsRequest {
    /// Start round (0 = current)
    #[prost(uint64, tag = "1")]
    pub start_round: u64,

    /// Filter by event type
    #[prost(string, repeated, tag = "2")]
    pub event_types: Vec<String>,
}

/// Round started details.
#[derive(Clone, prost::Message)]
pub struct RoundStartedDetails {
    /// Expected nodes
    #[prost(uint32, tag = "1")]
    pub expected_nodes: u32,

    /// Deadline timestamp
    #[prost(int64, tag = "2")]
    pub deadline: i64,
}

/// Submission received details.
#[derive(Clone, prost::Message)]
pub struct SubmissionReceivedDetails {
    /// Node DID
    #[prost(string, tag = "1")]
    pub node_did: String,

    /// Payload type
    #[prost(enumeration = "GrpcPayloadType", tag = "2")]
    pub payload_type: i32,

    /// Size in bytes
    #[prost(uint64, tag = "3")]
    pub size_bytes: u64,
}

/// Round completed details.
#[derive(Clone, prost::Message)]
pub struct RoundCompletedDetails {
    /// Participants
    #[prost(uint32, tag = "1")]
    pub participants: u32,

    /// Byzantine count
    #[prost(uint32, tag = "2")]
    pub byzantine_count: u32,

    /// Aggregation time in ms
    #[prost(float, tag = "3")]
    pub aggregation_time_ms: f32,
}

/// Byzantine detected details.
#[derive(Clone, prost::Message)]
pub struct ByzantineDetectedDetails {
    /// Node DID
    #[prost(string, tag = "1")]
    pub node_did: String,

    /// Reason
    #[prost(string, tag = "2")]
    pub reason: String,

    /// Confidence
    #[prost(float, tag = "3")]
    pub confidence: f32,
}

/// Round event.
#[derive(Clone, prost::Message)]
pub struct RoundEvent {
    /// Round number
    #[prost(uint64, tag = "1")]
    pub round: u64,

    /// Event type
    #[prost(enumeration = "GrpcRoundEventType", tag = "2")]
    pub event_type: i32,

    /// Timestamp
    #[prost(int64, tag = "3")]
    pub timestamp: i64,

    /// Event details (oneof)
    #[prost(oneof = "RoundEventDetails", tags = "4, 5, 6, 7")]
    pub details: Option<RoundEventDetails>,
}

/// Oneof for round event details.
#[derive(Clone, prost::Oneof)]
pub enum RoundEventDetails {
    #[prost(message, tag = "4")]
    Started(RoundStartedDetails),
    #[prost(message, tag = "5")]
    Submission(SubmissionReceivedDetails),
    #[prost(message, tag = "6")]
    Completed(RoundCompletedDetails),
    #[prost(message, tag = "7")]
    Byzantine(ByzantineDetectedDetails),
}

/// Heartbeat request.
#[derive(Clone, prost::Message)]
pub struct HeartbeatRequest {
    /// Session token
    #[prost(string, tag = "1")]
    pub session_token: String,

    /// Timestamp
    #[prost(int64, tag = "2")]
    pub timestamp: i64,
}

/// Heartbeat response.
#[derive(Clone, prost::Message)]
pub struct HeartbeatResponse {
    /// Whether heartbeat was accepted
    #[prost(bool, tag = "1")]
    pub ok: bool,

    /// Current round
    #[prost(uint64, tag = "2")]
    pub current_round: u64,

    /// Round state
    #[prost(enumeration = "GrpcRoundState", tag = "3")]
    pub round_state: i32,
}

/// Unregister node request.
#[derive(Clone, prost::Message)]
pub struct UnregisterNodeRequest {
    /// Session token
    #[prost(string, tag = "1")]
    pub session_token: String,
}

/// Unregister node response.
#[derive(Clone, prost::Message)]
pub struct UnregisterNodeResponse {
    /// Whether unregistration was successful
    #[prost(bool, tag = "1")]
    pub ok: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    #[test]
    fn test_payload_type_values() {
        assert_eq!(GrpcPayloadType::Unspecified as i32, 0);
        assert_eq!(GrpcPayloadType::DenseGradient as i32, 1);
        assert_eq!(GrpcPayloadType::HyperEncoded as i32, 2);
        assert_eq!(GrpcPayloadType::BinaryHypervector as i32, 3);
    }

    #[test]
    fn test_round_state_values() {
        assert_eq!(GrpcRoundState::Pending as i32, 1);
        assert_eq!(GrpcRoundState::Collecting as i32, 2);
        assert_eq!(GrpcRoundState::Complete as i32, 4);
    }

    #[test]
    fn test_node_identity_encoding() {
        let identity = NodeIdentity {
            did: "did:mycelix:node123".to_string(),
            public_key: vec![1, 2, 3, 4],
            assurance_level: 2,
        };

        let encoded = identity.encode_to_vec();
        assert!(!encoded.is_empty());

        let decoded = NodeIdentity::decode(encoded.as_slice()).unwrap();
        assert_eq!(decoded.did, "did:mycelix:node123");
        assert_eq!(decoded.assurance_level, 2);
    }

    #[test]
    fn test_dense_gradient_encoding() {
        let gradient = GrpcDenseGradient {
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            dimension: 5,
        };

        let encoded = gradient.encode_to_vec();
        let decoded = GrpcDenseGradient::decode(encoded.as_slice()).unwrap();

        assert_eq!(decoded.values.len(), 5);
        assert_eq!(decoded.dimension, 5);
        assert!((decoded.values[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_submit_request_encoding() {
        let request = SubmitRequest {
            node: Some(NodeIdentity {
                did: "did:mycelix:test".to_string(),
                ..Default::default()
            }),
            round: 42,
            payload: Some(GrpcPayload {
                data: Some(GrpcPayloadData::Dense(GrpcDenseGradient {
                    values: vec![1.0, 2.0],
                    dimension: 2,
                })),
            }),
            timestamp: 1234567890,
            ..Default::default()
        };

        let encoded = request.encode_to_vec();
        let decoded = SubmitRequest::decode(encoded.as_slice()).unwrap();

        assert_eq!(decoded.round, 42);
        assert!(decoded.node.is_some());
        assert!(decoded.payload.is_some());
    }

    #[test]
    fn test_round_event_encoding() {
        let event = RoundEvent {
            round: 10,
            event_type: GrpcRoundEventType::RoundCompleted as i32,
            timestamp: 1234567890,
            details: Some(RoundEventDetails::Completed(RoundCompletedDetails {
                participants: 5,
                byzantine_count: 1,
                aggregation_time_ms: 150.5,
            })),
        };

        let encoded = event.encode_to_vec();
        let decoded = RoundEvent::decode(encoded.as_slice()).unwrap();

        assert_eq!(decoded.round, 10);
        assert!(matches!(decoded.details, Some(RoundEventDetails::Completed(_))));
    }

    #[test]
    fn test_get_status_response() {
        let response = GetStatusResponse {
            current_round: 5,
            round_state: GrpcRoundState::Collecting as i32,
            registered_nodes: 10,
            submissions_this_round: 7,
            expected_this_round: 10,
            round_start_time: 1234567890,
            aggregation_algorithm: "krum".to_string(),
        };

        let encoded = response.encode_to_vec();
        let decoded = GetStatusResponse::decode(encoded.as_slice()).unwrap();

        assert_eq!(decoded.current_round, 5);
        assert_eq!(decoded.round_state, GrpcRoundState::Collecting as i32);
        assert_eq!(decoded.aggregation_algorithm, "krum");
    }
}
