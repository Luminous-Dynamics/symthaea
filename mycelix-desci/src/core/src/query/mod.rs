//! Query System
//!
//! Provides efficient searching and filtering of epistemic claims with indexing

pub mod filter;
pub mod index;
pub mod engine;

pub use engine::QueryEngine;
pub use filter::{QueryFilter, SortBy, SortOrder};
pub use index::ClaimIndex;
