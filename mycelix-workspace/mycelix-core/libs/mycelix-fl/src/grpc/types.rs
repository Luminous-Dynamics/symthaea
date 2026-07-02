// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hand-written protobuf message types for the FL gRPC service.
//!
//! These mirror `proto/fl_service.proto` but avoid requiring protoc at
//! build time. Each struct derives `prost::Message` for wire-compatible
//! serialization.

/// Request to register a node with the coordinator.
#[derive(Clone, PartialEq, prost::Message)]
pub struct RegisterNodeRequest {
    /// Unique node identifier.
    #[prost(string, tag = "1")]
    pub node_id: String,
    /// Optional Ed25519 public key (32 bytes).
    #[prost(bytes = "vec", tag = "2")]
    pub public_key: Vec<u8>,
    /// Optional DID identifier.
    #[prost(string, tag = "3")]
    pub did: String,
}

/// Response to a node registration request.
#[derive(Clone, PartialEq, prost::Message)]
pub struct RegisterNodeResponse {
    /// Whether registration succeeded.
    #[prost(bool, tag = "1")]
    pub success: bool,
    /// Human-readable status message.
    #[prost(string, tag = "2")]
    pub message: String,
    /// Current round number (so the node can sync).
    #[prost(uint64, tag = "3")]
    pub current_round: u64,
}

/// Submit a gradient for the current round.
#[derive(Clone, PartialEq, prost::Message)]
pub struct SubmitGradientRequest {
    /// Node identifier (must be registered).
    #[prost(string, tag = "1")]
    pub node_id: String,
    /// Flattened gradient values.
    #[prost(float, repeated, tag = "2")]
    pub values: Vec<f32>,
    /// Round number this gradient belongs to.
    #[prost(uint64, tag = "3")]
    pub round: u64,
}

/// Response to a gradient submission.
#[derive(Clone, PartialEq, prost::Message)]
pub struct SubmitGradientResponse {
    /// Whether the gradient was accepted.
    #[prost(bool, tag = "1")]
    pub accepted: bool,
    /// Human-readable status message.
    #[prost(string, tag = "2")]
    pub message: String,
}

/// Request the status of a round.
#[derive(Clone, PartialEq, prost::Message)]
pub struct GetRoundStatusRequest {
    /// Round number to query. 0 means the current round.
    #[prost(uint64, tag = "1")]
    pub round: u64,
}

/// Status of an FL round.
#[derive(Clone, PartialEq, prost::Message)]
pub struct GetRoundStatusResponse {
    /// Round number.
    #[prost(uint64, tag = "1")]
    pub round: u64,
    /// Status string: "pending", "collecting", "aggregating", "complete", "failed".
    #[prost(string, tag = "2")]
    pub status: String,
    /// Number of nodes that have submitted gradients.
    #[prost(uint32, tag = "3")]
    pub submitted_nodes: u32,
    /// Number of nodes required before aggregation can proceed.
    #[prost(uint32, tag = "4")]
    pub required_nodes: u32,
}

/// Request the aggregation result for a completed round.
#[derive(Clone, PartialEq, prost::Message)]
pub struct GetRoundResultRequest {
    /// Round number to retrieve.
    #[prost(uint64, tag = "1")]
    pub round: u64,
}

/// Aggregation result for a completed round.
#[derive(Clone, PartialEq, prost::Message)]
pub struct GetRoundResultResponse {
    /// Round number.
    #[prost(uint64, tag = "1")]
    pub round: u64,
    /// Whether aggregation succeeded.
    #[prost(bool, tag = "2")]
    pub success: bool,
    /// Aggregated gradient vector.
    #[prost(float, repeated, tag = "3")]
    pub aggregated_gradient: Vec<f32>,
    /// Node IDs included in the aggregation.
    #[prost(string, repeated, tag = "4")]
    pub included_nodes: Vec<String>,
    /// Node IDs excluded by the defense.
    #[prost(string, repeated, tag = "5")]
    pub excluded_nodes: Vec<String>,
    /// Per-node quality scores.
    #[prost(message, repeated, tag = "6")]
    pub scores: Vec<NodeScore>,
}

/// Quality score for a single node.
#[derive(Clone, PartialEq, prost::Message)]
pub struct NodeScore {
    /// Node identifier.
    #[prost(string, tag = "1")]
    pub node_id: String,
    /// Quality score assigned by the defense/PoGQ.
    #[prost(double, tag = "2")]
    pub score: f64,
}

/// Health check request (empty).
#[derive(Clone, PartialEq, prost::Message)]
pub struct HealthCheckRequest {}

/// Health check response.
#[derive(Clone, PartialEq, prost::Message)]
pub struct HealthCheckResponse {
    /// Whether the coordinator is healthy.
    #[prost(bool, tag = "1")]
    pub healthy: bool,
    /// Uptime in seconds.
    #[prost(uint64, tag = "2")]
    pub uptime_secs: u64,
    /// Number of registered nodes.
    #[prost(uint32, tag = "3")]
    pub registered_nodes: u32,
    /// Current round number (0 if no round active).
    #[prost(uint64, tag = "4")]
    pub current_round: u64,
}
