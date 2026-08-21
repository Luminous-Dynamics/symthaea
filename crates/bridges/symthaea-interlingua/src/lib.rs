// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Symthaea Cognitive Interchange Protocol (SCIP).
//!
//! This bridge keeps [`symthaea_communication::GroundedConceptGraph`] as the
//! canonical semantic object while providing negotiated machine-oriented
//! representations for HDC systems, LLM adapters, and future cognitive peers.
//!
//! The protocol is deliberately evidence-preserving: an HDC vector is an
//! associative projection of grounded meaning, never proof of that meaning by
//! itself.

#![forbid(unsafe_code)]

pub mod delta;
pub mod graph_delta;
pub mod hdc;
pub mod metrics;
pub mod negotiation;
pub mod planner;
pub mod protocol;
pub mod text_fallback;
pub mod wire;

pub use delta::{HdcDeltaEntry, SparseHdcDelta};
pub use graph_delta::{GraphDelta, canonical_edge_bytes, edge_semantic_hash};
pub use hdc::{
    GroundedHdcCodec, ProjectionVerification, SCIP_HDC_ALGEBRA_V1, SCIP_HDC_ATOM_DERIVATION_V1,
    SCIP_HDC_NAMESPACE_V1, profile_fingerprint,
};
pub use metrics::{DeltaMetrics, ProjectionMetrics, measure_delta, measure_projection};
pub use negotiation::{NegotiatedSession, PeerCapabilities, negotiate};
pub use planner::{
    ProjectionAttachment, ProjectionCandidate, ProjectionPolicy, SemanticTransferMode, TransferPlan,
    TransferPlanningInput, TransferPolicy, plan_transfer,
};
pub use protocol::{
    CognitiveEnvelope, GROUNDED_GRAPH_SCHEMA_V1, HdcPayload, HdcProfile, InterchangeError,
    InterchangePayload, InterchangeRepresentation, ProtocolVersion, SCIP_CONTENT_HASH_HEX_LEN,
    SCIP_PROTOCOL_ID, SCIP_V1, ScipLimits, SemanticProfile, SemanticReference,
    StructuredJsonPayload, canonical_graph_bytes, canonical_graph_bytes_with_limits,
    canonicalize_graph, canonicalize_graph_with_limits, graph_semantic_hash, validate_graph,
    validate_graph_with_limits,
};
pub use text_fallback::{LlmFallbackMode, LlmFallbackPacket, LlmTextFallback};
pub use wire::{
    HdcWireEncoding, HdcWireError, HdcWirePacket, MAX_HDC_WIRE_DIMENSION, WireFidelity,
    WireSelection, WireSelectionPolicy, select_wire_encoding,
};
