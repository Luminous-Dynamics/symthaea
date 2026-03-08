//! Bridge types for cross-domain communication within the Civic cluster
//!
//! Re-exports shared bridge entry types from `mycelix-bridge-entry-types`
//! with cluster-specific aliases for API compatibility.

pub use mycelix_bridge_common::BridgeHealth;
pub use mycelix_bridge_entry_types::{BridgeEventEntry, BridgeQueryEntry};

/// A cross-domain query within the Civic cluster.
/// Type alias for the shared `BridgeQueryEntry`.
pub type CivicQuery = BridgeQueryEntry;

/// A cross-domain event broadcast within the Civic cluster.
/// Type alias for the shared `BridgeEventEntry`.
pub type CivicEvent = BridgeEventEntry;
