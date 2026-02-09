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

pub mod types;
pub mod aggregation;
pub mod byzantine;
pub mod hybrid_bft;
pub mod privacy;
pub mod pipeline;
pub mod convert;

pub use types::*;
pub use aggregation::*;
pub use byzantine::*;
pub use hybrid_bft::*;
pub use privacy::*;
pub use pipeline::*;
pub use convert::*;
