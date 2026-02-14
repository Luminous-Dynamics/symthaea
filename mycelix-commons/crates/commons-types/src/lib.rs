//! Mycelix Commons Shared Types & Utilities
//!
//! Common functionality for all domain zomes in the Commons cluster:
//! - Batch query operations (solving N+1 query patterns)
//! - Anchor utilities for consistent indexing
//! - Bridge types for cross-domain communication
//! - Geographic types shared across domains

pub mod batch;
pub mod anchors;
pub mod bridge_types;
pub mod geo;

pub use batch::*;
pub use anchors::*;
pub use bridge_types::*;
pub use geo::*;
