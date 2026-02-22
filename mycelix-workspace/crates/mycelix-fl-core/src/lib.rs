//! mycelix-fl-core: Universal Federated Learning Core
//!
//! A lightweight, dependency-minimal crate containing the canonical FL types,
//! aggregation algorithms, Byzantine detection, differential privacy, and
//! a unified pipeline that chains them all together.
//!
//! # Design Decisions
//!
//! - **f32 precision**: Sufficient for ML gradients, matches Symthaea and Holochain.
//!   Conversion utilities are provided for f64-based SDKs.
//! - **No heavy dependencies**: Only serde, rand, thiserror. Can be used by both
//!   Symthaea (burn ecosystem) and Mycelix SDK (holochain ecosystem) without conflicts.
//! - **Pipeline pattern**: The `UnifiedPipeline` chains all capabilities in the
//!   correct order: validate -> DP -> gate -> detect -> trim -> aggregate.
//!
//! # Validated Byzantine Tolerance
//!
//! - 34% with trimmed-mean (classical BFT limit ~33%)
//! - Up to 45% when reputation disparity is sufficient (hybrid BFT)
//! - 45% does NOT converge when all nodes have equal reputation

pub mod adaptive_defense;
pub mod aggregation;
pub mod byzantine;
pub mod consciousness_plugin;
pub mod convert;
pub mod hybrid_bft;
pub mod meta_learning;
pub mod pipeline;
pub mod plugins;
pub mod privacy;
pub mod types;

#[cfg(feature = "replay")]
pub mod replay_detection;

#[cfg(feature = "shapley")]
pub mod shapley;

#[cfg(feature = "ensemble")]
pub mod ensemble_defense;

#[cfg(feature = "compression")]
pub mod compression;

#[cfg(feature = "holochain")]
pub mod holochain_bridge;

pub use adaptive_defense::*;
pub use aggregation::*;
pub use byzantine::*;
pub use consciousness_plugin::*;
pub use convert::*;
pub use hybrid_bft::*;
pub use pipeline::*;
pub use privacy::*;
pub use types::*;

#[cfg(feature = "replay")]
pub use replay_detection::*;

#[cfg(feature = "shapley")]
pub use shapley::*;

#[cfg(feature = "ensemble")]
pub use ensemble_defense::*;

#[cfg(feature = "compression")]
pub use compression::*;
