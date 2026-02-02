#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(not(feature = "prover"), allow(unused_imports))]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! # Mycelix ZK Tax SDK
//!
//! Zero-knowledge proofs for privacy-preserving tax compliance.
//!
//! ## Overview
//!
//! This SDK enables users to prove they fall within a specific tax bracket
//! **without revealing their actual income**. This is useful for:
//!
//! - Privacy-preserving income verification
//! - Compliance attestations without data exposure
//! - Decentralized identity income credentials
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use mycelix_zk_tax::{TaxBracketProver, Jurisdiction, FilingStatus};
//!
//! // Create a prover
//! let prover = TaxBracketProver::new();
//!
//! // Generate a proof (income is NEVER revealed)
//! let proof = prover.prove(
//!     85_000,                      // Private income
//!     Jurisdiction::US,
//!     FilingStatus::Single,
//!     2024,
//! ).unwrap();
//!
//! // The proof shows bracket 2 (22%) without revealing $85,000
//! assert_eq!(proof.bracket_index, 2);
//!
//! // Anyone can verify
//! assert!(proof.verify().is_ok());
//! ```
//!
//! ## Features
//!
//! - `std` (default): Standard library support
//! - `alloc`: Enables no_std with alloc (for embedded/WASM)
//! - `prover` (default): Full proving capability with RISC Zero zkVM
//! - `verifier-only`: Lightweight, verification only
//! - `wasm`: WebAssembly support for browsers
//! - `cli`: Command-line tool
//! - `server`: Axum HTTP server support
//! - `anchoring`: Blockchain proof anchoring (Ethereum, Solana, Bitcoin)
//! - `entity`: Business entity taxation (S-Corp, C-Corp, LLC, etc.)
//! - `service`: Rate limiting, caching, and metrics
//! - `aggregation`: Batch proof aggregation with Merkle trees
//! - `tracing-support`: Observability with tracing
//! - `full`: All features enabled
//!
//! ## no_std Support
//!
//! This crate supports `no_std` environments with the `alloc` feature:
//!
//! ```toml
//! [dependencies]
//! mycelix-zk-tax = { version = "0.1", default-features = false, features = ["alloc"] }
//! ```
//!
//! ## Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`api`] | REST API types and OpenAPI spec generation |
//! | [`brackets`] | Tax bracket definitions by jurisdiction |
//! | [`proof`] | Proof types (range, batch, composite, etc.) |
//! | [`prover`] | RISC Zero zkVM proof generation |
//! | [`credentials`] | W3C Verifiable Credentials support |
//! | [`anchoring`] | Blockchain anchoring for proof immutability |
//! | [`entity`] | Business entity tax calculations |
//! | [`service`] | Rate limiting, caching, metrics |
//! | [`aggregation`] | Merkle tree proof aggregation |

#[cfg(not(feature = "std"))]
extern crate alloc;

// Re-export alloc types for internal use
#[cfg(not(feature = "std"))]
pub(crate) mod std_compat {
    pub use alloc::string::{String, ToString};
    pub use alloc::vec::Vec;
    pub use alloc::vec;
    pub use alloc::format;
    pub use alloc::boxed::Box;
    pub use core::fmt;
    pub use core::result::Result;
    pub use core::option::Option;
}

#[cfg(feature = "std")]
#[allow(unused_imports)]
pub(crate) mod std_compat {
    pub use std::string::{String, ToString};
    pub use std::vec::Vec;
    pub use std::vec;
    pub use std::format;
    pub use std::boxed::Box;
    pub use std::fmt;
    pub use std::result::Result;
    pub use std::option::Option;
}

// =============================================================================
// Core Modules (always available)
// =============================================================================

pub mod api;
pub mod brackets;
pub mod cache;
pub mod credentials;
pub mod error;
pub mod jurisdiction;
pub mod proof;
pub mod subnational;
pub mod types;

// =============================================================================
// Feature-Gated Modules
// =============================================================================

#[cfg(feature = "prover")]
#[cfg_attr(docsrs, doc(cfg(feature = "prover")))]
pub mod prover;

#[cfg(feature = "wasm")]
#[cfg_attr(docsrs, doc(cfg(feature = "wasm")))]
pub mod wasm;

/// Blockchain anchoring for proof immutability.
///
/// Supports Ethereum, Solana, and Bitcoin networks.
#[cfg_attr(docsrs, doc(cfg(feature = "anchoring")))]
pub mod anchoring;

/// Business entity taxation support.
///
/// Supports S-Corp, C-Corp, LLC, partnerships, and more.
#[cfg_attr(docsrs, doc(cfg(feature = "entity")))]
pub mod entity;

/// Service layer with rate limiting, caching, and metrics.
#[cfg_attr(docsrs, doc(cfg(feature = "service")))]
pub mod service;

/// Proof aggregation using Merkle trees.
///
/// Efficiently batch-verify multiple proofs.
#[cfg_attr(docsrs, doc(cfg(feature = "aggregation")))]
pub mod aggregation;

/// Tracing and observability support.
///
/// Provides structured logging using the `tracing` crate.
/// Enable with the `tracing-support` feature.
#[cfg(feature = "tracing-support")]
#[cfg_attr(docsrs, doc(cfg(feature = "tracing-support")))]
pub mod tracing_support;

// Also export core types even without full feature
#[cfg(not(feature = "tracing-support"))]
mod tracing_support;

// Re-exports for convenience
pub use brackets::TaxBracket;
pub use cache::{
    CacheConfig, CacheEntry, CacheStats, ProofCache, ProofCacheKey,
    cache_proof, clear_global_cache, get_cached, get_or_generate_cached,
    global_cache, global_stats,
};
pub use error::{Error, Result};
pub use jurisdiction::{FilingStatus, Jurisdiction};
pub use proof::{
    BatchProof, BatchProofBuilder, IncomeRangeProof, RangeProofBuilder,
    TaxBracketProof, TaxBracketReceipt, YearProofSummary, prove_three_year_history,
    // Effective tax rate proofs
    EffectiveTaxRateProof, EffectiveTaxRateBuilder,
    // Cross-jurisdiction proofs
    CrossJurisdictionProof, CrossJurisdictionProofBuilder, JurisdictionBracketInfo,
    // Deduction proofs
    DeductionProof, DeductionProofBuilder, DeductionCategory, DeductionCategorySummary,
    // Composite proofs
    CompositeProof, CompositeProofBuilder,
    // Proof chaining
    ProofChain, ProofChainLink,
    // Combined federal + state proofs
    CombinedFederalStateProof, CombinedProofBuilder as CombinedFederalStateProofBuilder,
    // Income stability proofs
    IncomeStabilityProof,
    // Time-bound proofs
    TimeBoundProof,
    // Merkle proof aggregation
    MerkleProofTree, MerkleProofPath,
    // Selective disclosure
    SelectiveDisclosureProof, SelectiveDisclosureBuilder, DisclosableField,
    // Proof compression
    CompressedProof, CompressedProofType,
};
pub use types::{BracketCommitment, TaxYear};

#[cfg(feature = "prover")]
pub use prover::TaxBracketProver;

// Re-exports from new modules
pub use aggregation::{AggregatedProof, ProofAggregator, AggregationSummary};
pub use anchoring::{ProofAnchor, AnchorConfig, Network, AnchorMerkleTree};
pub use entity::{EntityType, EntityTaxInfo, BusinessTaxProof};
pub use service::{ServiceConfig, RateLimiter, ProofCacheService, MetricsCollector};

#[cfg(feature = "prover")]
pub use entity::BusinessProver;

#[cfg(feature = "prover")]
pub use service::ProofService;

/// SDK version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Supported tax years (2020-2025)
pub const SUPPORTED_YEARS: &[u32] = &[2020, 2021, 2022, 2023, 2024, 2025];

/// Check if a tax year is supported
pub fn is_year_supported(year: u32) -> bool {
    SUPPORTED_YEARS.contains(&year)
}

// =============================================================================
// Prelude for convenient imports
// =============================================================================

/// Convenient re-exports for common usage patterns.
///
/// ```rust,ignore
/// use mycelix_zk_tax::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{
        TaxBracket, Jurisdiction, FilingStatus,
        TaxBracketProof, Error, Result,
        BracketCommitment, TaxYear,
    };

    #[cfg(feature = "prover")]
    pub use crate::TaxBracketProver;

    pub use crate::aggregation::ProofAggregator;
    pub use crate::anchoring::Network;
    pub use crate::entity::EntityType;

    #[cfg(feature = "tracing-support")]
    pub use crate::tracing_support::{
        TracingConfig, LogLevel, OperationMetrics,
        init_tracing, init_default_tracing, try_init_tracing,
    };
}
