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

pub mod source;
pub mod frame;
pub mod optical_flow;
pub mod temporal_features;
pub mod ltc_rhythm;

pub use source::{VideoSource, VideoSourceConfig, WebcamSource, MockVideoSource, MockPattern};
pub use frame::{Frame, FrameFormat};
pub use optical_flow::{OpticalFlow, MotionVector};
pub use temporal_features::{TemporalFeatures, TemporalFeatureExtractor};
pub use ltc_rhythm::{LtcRhythmDetector, LtcRhythmConfig, RhythmDetection, RhythmPrototype};
