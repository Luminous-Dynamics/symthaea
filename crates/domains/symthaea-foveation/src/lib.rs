// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Foveation bridge: dual-stream vision architecture for Symthaea.
//!
//! Connects the fast dorsal stream (VisionManifold surprise/saliency) to a detailed
//! asynchronous ventral stream while preserving capture provenance and recording what
//! semantic backend/operation actually executed.

#![deny(unsafe_code)]

pub mod channel;
pub mod crop;
pub mod evidence;
pub mod manager;
pub mod types;
pub mod ventral;

pub use channel::FoveationChannel;
pub use evidence::{StructuredFoveationEvidence, StructuredFoveationEvidenceError};
pub use manager::FoveationManager;
pub use types::{
    FoveationConfig, FoveationRequest, FoveationResult, FoveationTelemetry, FrameBuffer,
    FrameObservationError, RecognizedContent, RoutingStrategy, SalientRegion, VentralExecutionKind,
    VentralExecutionReceipt, VentralOperation,
};
