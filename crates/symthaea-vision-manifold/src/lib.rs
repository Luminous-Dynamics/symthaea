#![deny(unsafe_code)]
#![allow(clippy::needless_range_loop)]

//! Symthaea Vision Manifold: Patch-based HDC video encoding with CfC temporal dynamics.
//!
//! Encodes video frames into 16,384-dimensional holographic hypervectors using the
//! bind-bundle paradigm from Hyperdimensional Computing, then tracks scene state via
//! Closed-form Continuous-time (CfC) neurons with O(1) temporal prediction.
//!
//! # Architecture
//!
//! ```text
//! pixels → PatchHdcEncoder → ContinuousHV → VisionManifold (CfC) → prediction + surprise
//!                                                 ↕
//!                                          SurpriseMap (per-patch free energy)
//! ```
//!
//! # Key properties
//!
//! - **Holographic encoding**: Each frame is a superposition of position-bound patch
//!   appearances. Similar frames produce similar HVs; spatial layout is preserved.
//! - **O(1) temporal prediction**: The CfC closed-form solution
//!   `state' = x_inf + (state - x_inf) · exp(-dt/τ)` has cost independent of dt.
//! - **Surprise-driven attention**: Per-patch prediction error identifies regions
//!   of unexpected change (active inference foraging).
//! - **TemporalPredictor**: Implements the shared trait for cross-domain integration.

pub mod attention;
pub mod bridge;
pub mod camera;
pub mod encoder;
pub mod manifold;
pub mod training;
pub mod types;

pub use attention::SurpriseMap;
pub use bridge::VisionBridge;
pub use camera::{CameraManifold, CapturedFrame, MockCameraSource};
#[cfg(feature = "camera")]
pub use camera::CameraSource;
pub use encoder::{MultiScaleEncoder, PatchHdcEncoder};
pub use manifold::VisionManifold;
pub use training::{BpttResult, ManifoldTrainer};
pub use types::{
    AttentionMap, LearningConfig, MultiScaleConfig, PatchGrid, TrainingConfig, TrainingMethod,
    VisionConfig, VisionTelemetry,
};
