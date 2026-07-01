//! mycelix-fl-core stub
//!
//! Minimal stub providing the types, traits, and functions that symthaea imports.
//! This allows symthaea to compile without the full mycelix-workspace dependency.

pub mod hybrid_bft;
pub mod pipeline;
pub mod privacy;
pub mod types;

// Re-export everything at crate root (matching real crate's `pub use X::*` pattern)
pub use hybrid_bft::*;
pub use pipeline::{PipelineResult, UnifiedPipeline};
pub use privacy::*;
pub use types::*;
