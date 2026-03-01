//! Shared mirror types for fabrication sweettests.
//!
//! These types mirror the Rust coordinator structs but are defined locally
//! since sweettests can't import from WASM crates.

use holochain::prelude::*;
use std::path::PathBuf;

// =============================================================================
// Pagination types (mirrors fabrication_common)
// =============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PaginationInput {
    pub offset: u32,
    pub limit: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HashPaginationInput {
    pub hash: ActionHash,
    pub pagination: Option<PaginationInput>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AgentPaginationInput {
    pub pagination: Option<PaginationInput>,
}

// =============================================================================
// Common design input (used by multiple test files)
// =============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateDesignInput {
    pub title: String,
    pub description: String,
    pub category: String,
    pub license: String,
    pub safety_class: String,
    pub files: Vec<String>,
    pub tags: Vec<String>,
    pub material_compatibility: Vec<String>,
    pub parametric_schema: Option<String>,
}

// =============================================================================
// DNA path helper
// =============================================================================

pub fn fabrication_dna_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // sweettest/ -> tests/
    path.pop(); // tests/ -> fabrication/
    path.push("workdir");
    path.push("fabrication.dna");
    path
}
