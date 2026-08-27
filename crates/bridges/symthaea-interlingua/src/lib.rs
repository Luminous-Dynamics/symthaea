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
pub mod hdc;
pub mod metrics;
pub mod negotiation;
pub mod protocol;
pub mod text_fallback;

pub use delta::{HdcDeltaEntry, SparseHdcDelta};
pub use hdc::{
    GroundedHdcCodec, ProjectionVerification, SCIP_HDC_ALGEBRA_V1, SCIP_HDC_ATOM_DERIVATION_V1,
    SCIP_HDC_NAMESPACE_V1, profile_fingerprint,
};
pub use metrics::{DeltaMetrics, ProjectionMetrics, measure_delta, measure_projection};
pub use negotiation::{NegotiatedSession, PeerCapabilities, negotiate};
pub use protocol::{
    CognitiveEnvelope, GROUNDED_GRAPH_SCHEMA_V1, HdcPayload, HdcProfile, InterchangeError,
    InterchangePayload, InterchangeRepresentation, ProtocolVersion, SCIP_PROTOCOL_ID, SCIP_V1,
    SemanticProfile, SemanticReference, canonical_graph_bytes, graph_semantic_hash, validate_graph,
};
pub use text_fallback::{LlmFallbackMode, LlmFallbackPacket, LlmTextFallback};
