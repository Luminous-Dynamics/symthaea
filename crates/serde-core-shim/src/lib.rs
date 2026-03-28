// Local shim: re-exports serde as serde_core for bitflags 2.10+ compatibility
// with holochain's pinned serde =1.0.219.
pub use serde::*;
