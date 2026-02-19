//! Pagination types for bounded query results.
//!
//! Re-exports the canonical pagination primitives from `mycelix-bridge-common`
//! so that domain zomes within the Civic cluster can import them from a
//! single, familiar crate.

pub use mycelix_bridge_common::{PaginationInput, PaginatedResponse, paginate_links};
