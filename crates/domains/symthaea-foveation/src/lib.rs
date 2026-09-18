// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Foveation bridge: dual-stream vision architecture for Symthaea.
//!
//! Connects the fast dorsal stream (VisionManifold surprise/saliency) to
//! the detailed ventral stream (SigLIP, OCR, VQA) via a background thread.
//! The result: Symthaea keeps her high-rate cognitive loop intact while gaining
//! the ability to "read" and "recognize" objects on demand.
//!
//! VIS-001R additionally carries capture-owned visual provenance through the asynchronous
//! boundary and records requested routing separately from the semantic operation/backend that
//! actually executed.
//!
//! # Architecture
//!
//! ```text
//! observed frame + provenance
//!             ↓
//! SurpriseMap → FoveationManager → FoveationChannel (background)
//!                  ↑ priority queue         ↓ crop + dispatch
//!              on_saliency()         VentralPipeline
//!                                         ↓
//!                                   FoveationResult
//!                           (HV + content + execution receipt)
//!                                         ↓
//!                         StructuredFoveationEvidence
//!                                         ↓
//!                                   drain_results() → GWT
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use symthaea_foveation::{FoveationManager, FoveationConfig, FrameBuffer};
//!
//! let mut mgr = FoveationManager::new(FoveationConfig::default(), 8);
//! // Compatibility path: no typed observation provenance is invented.
//! mgr.on_frame(frame_buffer);
//! mgr.on_saliency(&salient_patches);
//! mgr.tick(now_us);
//! let results = mgr.drain_results();
//! ```

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
