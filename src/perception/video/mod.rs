// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Video Input Module - Real-time visual perception for Symthaea
//!
//! This module provides video capture capabilities for temporal perception,
//! enabling Symthaea to process continuous visual streams and detect temporal
//! patterns (rhythms, motion, gestures) that static image analysis cannot capture.
//!
//! ## Architecture
//!
//! ```text
//! VideoSource (trait)
//!     ├── WebcamSource   → Real webcam via nokhwa
//!     ├── FileSource     → Video file playback
//!     └── MockSource     → Synthetic test data
//!           │
//!           ▼
//!     Frame (RGB + timestamp)
//!           │
//!           ▼
//!     TemporalFeatures (brightness, edges, motion)
//!           │
//!           ▼
//!     HDC Projection (16,384D)
//!           │
//!           ▼
//!     HierarchicalLTC (temporal dynamics)
//! ```
//!
//! ## Key Innovation
//!
//! By feeding temporal features into the LTC network, Symthaea can learn
//! to recognize *when* things happen, not just *what* they are. This enables:
//!
//! - Rhythm detection (60 BPM vs 120 BPM)
//! - Gesture recognition (timing matters)
//! - Anomaly detection (unusual temporal patterns)
//! - Predictive perception (anticipating what comes next)

pub mod frame;
pub mod optical_flow;
pub mod rhythm_detector;
pub mod source;
pub mod temporal_features;

pub use frame::{Frame, FrameFormat};
pub use optical_flow::{MotionVector, OpticalFlow};
pub use rhythm_detector::{LtcRhythmConfig, LtcRhythmDetector, RhythmDetection};
pub use source::{MockPattern, MockVideoSource, VideoSource, VideoSourceConfig, WebcamSource};
pub use temporal_features::{TemporalFeatureExtractor, TemporalFeatures};
