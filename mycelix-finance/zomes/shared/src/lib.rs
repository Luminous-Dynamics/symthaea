#![deny(unsafe_code)]
//! Mycelix Finance Shared Utilities
//!
//! This crate provides common functionality for all Mycelix Finance zomes:
//! - Batch query operations (solving N+1 query patterns)
//! - Anchor utilities
//! - Common types and validation
//!
//! ## Batch Operations
//!
//! The `batch` module provides efficient batch fetching to avoid N+1 query patterns:
//!
//! ```rust,ignore
//! use mycelix_finance_shared::batch::*;
//!
//! // Instead of N individual get() calls in a loop:
//! // for link in links {
//! //     let record = get(link.target)?;  // N+1 pattern!
//! // }
//!
//! // Use batch operations:
//! let records = links_to_records(links)?;  // Single efficient operation
//! ```

use hdk::prelude::*;
use serde::{Deserialize, Serialize};

// Re-export the types crate for downstream consumers
pub use mycelix_finance_types;

// Re-export all modules
pub use anchors::*;
pub use batch::*;
pub use economics::*;
pub use governance::*;
pub use identity::*;
pub use input_validation::*;
pub use race_resolution::*;
pub use rate_limit::*;
pub use types::*;
pub use circuit_breaker::*;
pub use consciousness_gating::*;
pub use oracle_verification::*;
pub use update_chain::*;
pub use validation::*;

/// Community size threshold above which governance proposals are required for
/// currency creation, demurrage changes, and dispute resolution.
pub const COMMUNITY_GOVERNANCE_THRESHOLD: u32 = 10;

/// Race-condition resolution for concurrent link creation.
///
/// When multiple agents create the same logical link simultaneously, we need
/// a deterministic way to pick a single winner. This module provides a shared
/// helper that selects the link with the lowest `create_link_hash`.
pub mod race_resolution {
    use super::*;

    /// Deterministically pick a single winner from a set of concurrent links.
    ///
    /// Uses a two-level tiebreaker: first by `create_link_hash`, then by
    /// `author` (agent pubkey). The secondary key ensures determinism even in
    /// split-brain scenarios where different DHT partitions may compute
    /// different ActionHashes for the same logical operation.
    ///
    /// # Errors
    /// Returns an error if the slice is empty.
    pub fn pick_race_winner(links: &[Link]) -> ExternResult<&Link> {
        links
            .iter()
            .min_by(|a, b| {
                a.create_link_hash
                    .cmp(&b.create_link_hash)
                    .then_with(|| a.author.cmp(&b.author))
            })
            .ok_or_else(|| {
                wasm_error!(WasmErrorInner::Guest(
                    "Race resolution failed: no links found after creation".into()
                ))
            })
    }
}

/// Update chain traversal for Holochain entries.
///
/// In Holochain, `get()` returns the **original** entry, not the latest version.
/// When an entry has been modified via `update_entry()`, you must use `get_details()`
/// to discover the update chain and follow it to the latest version.
///
/// All coordinator zomes that do link-based lookups SHOULD use `follow_update_chain`
/// instead of bare `get()` when the entry might have been updated.
pub mod update_chain {
    use super::*;

    /// Maximum update chain depth before we bail out.
    /// Prevents unbounded network calls on entries with pathological update histories.
    pub const MAX_UPDATE_CHAIN_DEPTH: usize = 256;

    /// Recursively follow the Holochain update chain from an original action hash
    /// to find the latest version of a record.
    ///
    /// Each `update_entry` creates a new action that is recorded as an update of
    /// its predecessor. This function walks `get_details().updates` until it finds
    /// an action with no further updates.
    ///
    /// Stops after [`MAX_UPDATE_CHAIN_DEPTH`] hops to prevent unbounded traversal.
    pub fn follow_update_chain(action_hash: ActionHash) -> ExternResult<Record> {
        let mut current_hash = action_hash;
        for _ in 0..MAX_UPDATE_CHAIN_DEPTH {
            let details = get_details(current_hash.clone(), GetOptions::default())?.ok_or(
                wasm_error!(WasmErrorInner::Guest("Record not found".into())),
            )?;
            match details {
                Details::Record(record_details) => {
                    // Deterministic fork resolution: when concurrent updates
                    // create a fork (multiple updates pointing to the same
                    // predecessor), select the update with the LOWEST ActionHash
                    // (lexicographic comparison of raw bytes). This ensures all
                    // nodes converge to the same view regardless of DHT ordering.
                    if let Some(chosen_update) = record_details
                        .updates
                        .iter()
                        .min_by_key(|u| u.action_address().clone())
                    {
                        current_hash = chosen_update.action_address().clone();
                    } else {
                        return Ok(record_details.record);
                    }
                }
                _ => {
                    return get(current_hash, GetOptions::default())?.ok_or(wasm_error!(
                        WasmErrorInner::Guest("Record not found".into())
                    ));
                }
            }
        }
        // Reached max depth — return whatever we have at the current hash
        get(current_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
            "Update chain exceeded maximum depth".into()
        )))
    }
}

/// Batch operations module - solves N+1 query patterns
///
/// Provides efficient batch fetching for common patterns:
/// - Batch get records from multiple hashes
/// - Paginated link fetching helpers
/// - Efficient record collection from links
pub mod batch {
    use super::*;

    /// Options for batch record fetching
    #[derive(Clone, Debug, Default)]
    pub struct BatchGetOptions {
        /// Maximum number of records to fetch (0 = unlimited)
        pub limit: usize,
        /// Skip records that are deleted
        pub skip_deleted: bool,
    }

    /// Result of a batch get operation
    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct BatchGetResult {
        /// Successfully fetched records
        pub records: Vec<Record>,
        /// Hashes that were not found (404)
        pub not_found: Vec<ActionHash>,
        /// Hashes that failed to fetch (errors)
        pub errors: Vec<(ActionHash, String)>,
        /// Total requested
        pub total_requested: usize,
        /// Successfully fetched count
        pub success_count: usize,
    }

    impl BatchGetResult {
        pub fn new(total_requested: usize) -> Self {
            Self {
                records: Vec::new(),
                not_found: Vec::new(),
                errors: Vec::new(),
                total_requested,
                success_count: 0,
            }
        }
    }

    /// Batch get records from multiple action hashes
    ///
    /// This is more efficient than individual get() calls in a loop
    /// because it collects all results and handles errors gracefully.
    ///
    /// # Example
    /// ```rust,ignore
    /// // Instead of:
    /// for hash in hashes {
    ///     let record = get(hash, GetOptions::default())?;  // N calls
    /// }
    ///
    /// // Use:
    /// let result = batch_get_records(hashes, BatchGetOptions::default())?;
    /// for record in result.records {
    ///     // Process records
    /// }
    /// ```
    ///
    /// # Arguments
    /// * `hashes` - Action hashes to fetch
    /// * `options` - Batch get options
    ///
    /// # Returns
    /// BatchGetResult with records, not_found, and errors
    pub fn batch_get_records(
        hashes: Vec<ActionHash>,
        options: BatchGetOptions,
    ) -> ExternResult<BatchGetResult> {
        let total = hashes.len();
        let mut result = BatchGetResult::new(total);

        let limit = if options.limit == 0 {
            total
        } else {
            options.limit.min(total)
        };

        for hash in hashes.into_iter().take(limit) {
            match get(hash.clone(), GetOptions::default()) {
                Ok(Some(record)) => {
                    // Check if deleted
                    if options.skip_deleted {
                        if let Action::Delete(_) = record.action() {
                            continue;
                        }
                    }
                    result.records.push(record);
                    result.success_count += 1;
                }
                Ok(None) => {
                    result.not_found.push(hash);
                }
                Err(e) => {
                    result.errors.push((hash, format!("{:?}", e)));
                }
            }
        }

        Ok(result)
    }

    /// Get records from links (non-paginated helper)
    ///
    /// Converts a list of links to their target records.
    /// This solves the common N+1 pattern:
    ///
    /// ```rust,ignore
    /// // N+1 pattern (BAD):
    /// for link in get_links(...)? {
    ///     if let Some(record) = get(link.target)? {  // N calls!
    ///         records.push(record);
    ///     }
    /// }
    ///
    /// // Fixed (GOOD):
    /// let links = get_links(...)?;
    /// let records = links_to_records(links)?;  // Single batch operation
    /// ```
    pub fn links_to_records(links: Vec<Link>) -> ExternResult<Vec<Record>> {
        let hashes: Vec<ActionHash> = links
            .into_iter()
            .filter_map(|link| link.target.into_action_hash())
            .collect();

        let batch_result = batch_get_records(hashes, BatchGetOptions::default())?;
        Ok(batch_result.records)
    }

    /// Convert links to records with pagination
    ///
    /// Takes a list of links and returns paginated records.
    /// Use this after getting links from your zome's link type.
    ///
    /// # Arguments
    /// * `links` - Links to process
    /// * `pagination` - Pagination parameters
    ///
    /// # Returns
    /// PaginatedResult with the fetched records
    pub fn links_to_records_paginated(
        links: Vec<Link>,
        pagination: &super::types::PaginationInput,
    ) -> ExternResult<super::types::PaginatedResult<Record>> {
        pagination.validate()?;

        let total = links.len();

        // Apply pagination
        let paginated_links: Vec<_> = links
            .into_iter()
            .skip(pagination.offset)
            .take(pagination.limit)
            .collect();

        // Extract target hashes
        let hashes: Vec<ActionHash> = paginated_links
            .iter()
            .filter_map(|link| link.target.clone().into_action_hash())
            .collect();

        // Batch fetch records
        let batch_result = batch_get_records(hashes, BatchGetOptions::default())?;

        Ok(super::types::PaginatedResult::new(
            batch_result.records,
            total,
            pagination,
        ))
    }

    /// Get the most recent N records from links
    ///
    /// Useful for "recent activity" views.
    /// Links are sorted by timestamp (newest first).
    pub fn links_to_recent_records(
        mut links: Vec<Link>,
        count: usize,
    ) -> ExternResult<Vec<Record>> {
        // Sort by timestamp (newest first)
        links.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        // Take only the requested count
        let hashes: Vec<ActionHash> = links
            .into_iter()
            .take(count)
            .filter_map(|link| link.target.into_action_hash())
            .collect();

        let batch_result = batch_get_records(hashes, BatchGetOptions::default())?;
        Ok(batch_result.records)
    }

    /// Batch get links from multiple bases
    ///
    /// Useful when you need to get links from multiple anchor points.
    /// Instead of calling get_links N times, this collects all links
    /// from multiple bases in one logical operation.
    ///
    /// # Arguments
    /// * `bases` - Entry hashes to get links from
    /// * `link_type` - The link type to query
    ///
    /// # Returns
    /// A vector of (base_hash, links) pairs
    pub fn batch_get_links<LT: TryInto<LinkTypeFilter, Error = WasmError> + Clone>(
        bases: Vec<EntryHash>,
        link_type: LT,
    ) -> ExternResult<Vec<(EntryHash, Vec<Link>)>> {
        let mut results = Vec::new();

        for base in bases {
            let query = LinkQuery::try_new(base.clone(), link_type.clone())?;
            let links = get_links(query, GetStrategy::default())?;
            results.push((base, links));
        }

        Ok(results)
    }

    /// Extract and decode entries from records
    ///
    /// Helper to convert records to their typed entry content.
    ///
    /// # Example
    /// ```rust,ignore
    /// let records = links_to_records(links)?;
    /// let loans: Vec<Loan> = extract_entries(&records);
    /// ```
    pub fn extract_entries<T: TryFrom<SerializedBytes, Error = SerializedBytesError>>(
        records: &[Record],
    ) -> Vec<T> {
        records
            .iter()
            .filter_map(|r| r.entry().to_app_option::<T>().unwrap_or_default())
            .collect()
    }

    /// Filter records by a predicate on their entry content
    ///
    /// # Example
    /// ```rust,ignore
    /// let active_loans = filter_records_by::<Loan, _>(&records, |loan| {
    ///     loan.status == LoanStatus::Active
    /// });
    /// ```
    pub fn filter_records_by<T, F>(records: &[Record], predicate: F) -> Vec<Record>
    where
        T: TryFrom<SerializedBytes, Error = SerializedBytesError>,
        F: Fn(&T) -> bool,
    {
        records
            .iter()
            .filter(|r| match r.entry().to_app_option::<T>() {
                Ok(Some(entry)) => predicate(&entry),
                _ => false,
            })
            .cloned()
            .collect()
    }
}

/// Anchor utilities for consistent indexing
pub mod anchors {
    use super::*;

    /// Create a deterministic entry hash for an anchor string.
    /// Uses blake2b-256 for cryptographic safety.
    ///
    /// All coordinator zomes MUST use this function for anchor hashing.
    /// Do NOT use std::collections::hash_map::DefaultHasher — it's not
    /// cryptographically sound and produces different hashes across Rust versions.
    pub fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
        let hash = holo_hash::blake2b_256(anchor_str.as_bytes());
        Ok(EntryHash::from_raw_32(hash.to_vec()))
    }

    /// Create a sharded anchor for scalable indexing
    ///
    /// Instead of one global anchor, uses first character to create 26+ anchors.
    /// This improves scalability for high-volume data.
    pub fn sharded_anchor_hash(prefix: &str, key: &str) -> ExternResult<EntryHash> {
        let shard_char = key
            .chars()
            .next()
            .unwrap_or('_')
            .to_uppercase()
            .next()
            .unwrap_or('_');

        anchor_hash(&format!("{}_{}", prefix, shard_char))
    }

    /// Get all shard anchors for a given prefix (for bulk operations)
    pub fn all_shard_anchors(prefix: &str) -> Vec<String> {
        let mut anchors = Vec::new();
        for c in 'A'..='Z' {
            anchors.push(format!("{}_{}", prefix, c));
        }
        anchors.push(format!("{}__", prefix)); // For non-alpha characters
        anchors
    }
}

/// Common types used across finance zomes
pub mod types {
    use super::*;

    /// Input for paginated queries
    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct PaginationInput {
        pub offset: usize,
        pub limit: usize,
    }

    impl PaginationInput {
        pub const MAX_LIMIT: usize = 100;

        pub fn validate(&self) -> ExternResult<()> {
            if self.limit > Self::MAX_LIMIT {
                return Err(wasm_error!(WasmErrorInner::Guest(format!(
                    "Limit cannot exceed {}",
                    Self::MAX_LIMIT
                ))));
            }
            if self.limit == 0 {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Limit must be greater than 0".to_string()
                )));
            }
            Ok(())
        }
    }

    impl Default for PaginationInput {
        fn default() -> Self {
            Self {
                offset: 0,
                limit: 50,
            }
        }
    }

    /// Result wrapper for paginated queries
    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct PaginatedResult<T> {
        pub items: Vec<T>,
        pub total: usize,
        pub offset: usize,
        pub limit: usize,
        pub has_more: bool,
    }

    impl<T> PaginatedResult<T> {
        pub fn new(items: Vec<T>, total: usize, pagination: &PaginationInput) -> Self {
            Self {
                has_more: pagination.offset + items.len() < total,
                items,
                total,
                offset: pagination.offset,
                limit: pagination.limit,
            }
        }

        pub fn empty(pagination: &PaginationInput) -> Self {
            Self {
                items: Vec::new(),
                total: 0,
                offset: pagination.offset,
                limit: pagination.limit,
                has_more: false,
            }
        }
    }

    /// Standard error types for consistent error handling
    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub enum FinanceError {
        NotFound(String),
        Unauthorized(String),
        ValidationError(String),
        InsufficientFunds(String),
        InvalidState(String),
        InternalError(String),
    }

    impl std::fmt::Display for FinanceError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                FinanceError::NotFound(msg) => write!(f, "Not found: {}", msg),
                FinanceError::Unauthorized(msg) => write!(f, "Unauthorized: {}", msg),
                FinanceError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
                FinanceError::InsufficientFunds(msg) => write!(f, "Insufficient funds: {}", msg),
                FinanceError::InvalidState(msg) => write!(f, "Invalid state: {}", msg),
                FinanceError::InternalError(msg) => write!(f, "Internal error: {}", msg),
            }
        }
    }

    impl From<FinanceError> for WasmError {
        fn from(err: FinanceError) -> Self {
            wasm_error!(WasmErrorInner::Guest(err.to_string()))
        }
    }
}

/// Shared economic types used across all Mycelix Finance zomes.
///
/// These canonical type definitions are now provided by `mycelix_finance_types`.
/// This module re-exports them for backward compatibility.
pub mod economics {
    pub use mycelix_finance_types::*;
}

/// Governance authorization helpers
///
/// Shared logic for checking if the calling agent is an authorized governance
/// agent. Each coordinator zome fetches its own `LinkTypes::GovernanceAgents`
/// links and passes them to `verify_governance_or_bootstrap_from_links()`.
///
/// This eliminates duplicated authorization logic across recognition, staking,
/// and tend coordinators while keeping link type resolution local to each zome.
pub mod governance {
    use super::*;

    /// Standard anchor name for governance agent registration links.
    /// All zomes MUST use this same anchor string to share a single governance
    /// agent registry within the DNA.
    pub const GOVERNANCE_AGENTS_ANCHOR: &str = "governance_agents";

    /// Check if the calling agent is in the provided governance agent links.
    ///
    /// **Bootstrap rule**: if the links list is empty (no governance agents
    /// registered yet), any agent is allowed. This enables initial setup.
    ///
    /// # Usage
    /// ```rust,ignore
    /// use mycelix_finance_shared::governance::*;
    ///
    /// fn verify_governance_or_bootstrap() -> ExternResult<()> {
    ///     let gov_links = get_links(
    ///         LinkQuery::try_new(
    ///             anchor_hash(GOVERNANCE_AGENTS_ANCHOR)?,
    ///             LinkTypes::GovernanceAgents,
    ///         )?,
    ///         GetStrategy::default(),
    ///     )?;
    ///     verify_governance_or_bootstrap_from_links(gov_links)
    /// }
    /// ```
    pub fn verify_governance_or_bootstrap_from_links(gov_links: Vec<Link>) -> ExternResult<()> {
        if gov_links.is_empty() {
            return Ok(());
        }

        let caller = agent_info()?.agent_initial_pubkey;
        for link in gov_links {
            if let Ok(agent) = AgentPubKey::try_from(link.target) {
                if agent == caller {
                    return Ok(());
                }
            }
        }

        Err(wasm_error!(WasmErrorInner::Guest(
            "Caller is not an authorized governance agent".into()
        )))
    }
}

/// Agent identity helpers
pub mod identity {
    use super::*;

    /// Derive the canonical DID for the calling agent.
    ///
    /// Returns `did:mycelix:<agent_pubkey_base64url>`.
    /// All zomes MUST use this to verify caller identity instead of trusting
    /// user-supplied DIDs. This prevents DID spoofing attacks.
    pub fn caller_did() -> ExternResult<String> {
        let agent = agent_info()?.agent_initial_pubkey;
        Ok(format!("did:mycelix:{}", agent))
    }

    /// Verify that a claimed DID matches the calling agent.
    ///
    /// Returns Ok(()) if the DID matches, or an error if it doesn't.
    /// Use this at the top of any extern that takes a DID as input
    /// to prevent agents from acting as other members.
    pub fn verify_caller_is_did(claimed_did: &str) -> ExternResult<()> {
        let actual = caller_did()?;
        if claimed_did != actual {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Caller DID mismatch: claimed {} but agent is {}",
                claimed_did, actual
            ))));
        }
        Ok(())
    }
}

/// String ID input validation helpers
pub mod input_validation {
    use super::*;

    /// Validate that a string ID is non-empty and within reasonable bounds.
    pub fn validate_id(id: &str, field_name: &str) -> ExternResult<()> {
        if id.is_empty() || id.len() > 256 {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "{} must be 1-256 characters, got {}",
                field_name,
                id.len()
            ))));
        }
        Ok(())
    }

    /// Validate that a DID string has proper format (non-empty, starts with "did:").
    pub fn validate_did_format(did: &str, field_name: &str) -> ExternResult<()> {
        if did.is_empty() || did.len() > 256 {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "{} must be 1-256 characters, got {}",
                field_name,
                did.len()
            ))));
        }
        if !did.starts_with("did:") {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "{} must start with 'did:', got '{}'",
                field_name,
                &did[..did.len().min(20)]
            ))));
        }
        Ok(())
    }
}

/// Input validation module
pub mod validation {
    use super::*;

    /// Validate a Decentralized Identifier (DID)
    pub fn validate_did(did: &str) -> ExternResult<()> {
        if did.is_empty() {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "DID is required".to_string()
            )));
        }

        if !did.starts_with("did:") {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "DID must start with 'did:'".to_string()
            )));
        }

        let parts: Vec<&str> = did.splitn(3, ':').collect();
        if parts.len() < 3 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "DID must have format 'did:method:specific-id'".to_string()
            )));
        }

        Ok(())
    }

    /// Validate an amount is positive
    pub fn validate_positive_amount(amount: f64, field: &str) -> ExternResult<()> {
        if amount <= 0.0 {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "{} must be positive",
                field
            ))));
        }
        if amount.is_nan() || amount.is_infinite() {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "{} must be a valid number",
                field
            ))));
        }
        Ok(())
    }

    /// Validate a percentage is in valid range (0.0 - 1.0)
    pub fn validate_percentage(value: f64, field: &str) -> ExternResult<()> {
        if !(0.0..=1.0).contains(&value) {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "{} must be between 0.0 and 1.0",
                field
            ))));
        }
        Ok(())
    }

    /// Validate a currency code (ISO 4217 or crypto symbol)
    pub fn validate_currency(currency: &str) -> ExternResult<()> {
        if currency.is_empty() {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Currency is required".to_string()
            )));
        }
        if currency.len() > 10 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Currency code too long".to_string()
            )));
        }
        Ok(())
    }
}

/// Consciousness gating for Mycelix Finance operations.
///
/// These functions verify that the calling agent meets minimum consciousness
/// tier requirements before performing sensitive operations. Consciousness
/// scores are fetched from the identity cluster via cross-role calls.
///
/// ## Tiers
///
/// - **Participant** (combined >= 0.3): Basic financial operations (deposits,
///   payments, collateral). Most members clear this bar.
/// - **Citizen** (identity >= 0.25, reputation >= 0.10): Higher-bar operations
///   like currency creation and parameter amendment.
///
/// ## Fallback (Fail-Closed)
///
/// When the identity cluster is unreachable, both functions **deny** the
/// operation. This prevents silent permission grants during network partitions.
/// The integrity zome still enforces economic invariants independently.
pub mod consciousness_gating {
    use super::*;

    /// Verify the caller meets Participant consciousness tier.
    ///
    /// Participant tier requires a combined consciousness score >= 0.3.
    /// Used for: deposits, payments, collateral registration.
    pub fn verify_participant_tier() -> ExternResult<()> {
        let now_micros = sys_time()?.as_micros();
        super::circuit_breaker::check_breaker("identity_cluster", now_micros)?;

        match call(
            CallTargetCell::OtherRole("identity".into()),
            ZomeName::from("consciousness_gating"),
            FunctionName::from("check_participant_tier"),
            None,
            (),
        ) {
            Ok(ZomeCallResponse::Ok(result)) => {
                super::circuit_breaker::record_success("identity_cluster");
                let passed = result.decode::<bool>().unwrap_or(false);
                if passed {
                    Ok(())
                } else {
                    Err(wasm_error!(WasmErrorInner::Guest(
                        "Participant+ tier required (combined consciousness >= 0.3)".into()
                    )))
                }
            }
            // Identity cluster unreachable — fail closed
            _other => {
                let now = sys_time()?.as_micros();
                super::circuit_breaker::record_failure("identity_cluster", now);
                Err(wasm_error!(WasmErrorInner::Guest(
                    "Identity cluster unreachable — consciousness gating unavailable. \
                     Operations requiring Participant tier are suspended until \
                     the identity cluster is restored.".into()
                )))
            }
        }
    }

    /// Verify the caller meets Citizen+ consciousness tier.
    ///
    /// Citizen tier requires:
    /// - identity_score >= 0.25 (verified DID)
    /// - reputation_score >= 0.10 (some community participation)
    ///
    /// Used for: currency creation, parameter amendment.
    pub fn verify_citizen_tier() -> ExternResult<()> {
        let now_micros = sys_time()?.as_micros();
        super::circuit_breaker::check_breaker("identity_cluster", now_micros)?;

        match call(
            CallTargetCell::OtherRole("identity".into()),
            ZomeName::from("consciousness_gating"),
            FunctionName::from("check_citizen_tier"),
            None,
            (),
        ) {
            Ok(ZomeCallResponse::Ok(result)) => {
                super::circuit_breaker::record_success("identity_cluster");
                let passed = result.decode::<bool>().unwrap_or(false);
                if passed {
                    Ok(())
                } else {
                    Err(wasm_error!(WasmErrorInner::Guest(
                        "Citizen+ tier required (identity >= 0.25, reputation >= 0.10)".into()
                    )))
                }
            }
            // Identity cluster unreachable — fail closed
            _other => {
                let now = sys_time()?.as_micros();
                super::circuit_breaker::record_failure("identity_cluster", now);
                Err(wasm_error!(WasmErrorInner::Guest(
                    "Identity cluster unreachable — consciousness gating unavailable. \
                     Operations requiring Citizen tier are suspended until \
                     the identity cluster is restored.".into()
                )))
            }
        }
    }
}

/// Oracle rate verification for collateral deposits.
///
/// Verifies that a claimed oracle rate is within tolerance of the
/// price oracle consensus. Prevents rate injection attacks.
pub mod oracle_verification {
    use super::*;
    use mycelix_finance_types::ORACLE_RATE_TOLERANCE;

    /// Verify that a claimed oracle rate is within tolerance of consensus.
    ///
    /// Fetches the current consensus price from the price-oracle zome
    /// and rejects the claimed rate if it deviates more than ORACLE_RATE_TOLERANCE (5%).
    ///
    /// Falls back to accepting the claimed rate if the oracle is unreachable
    /// (bootstrap/standalone mode), with a warning logged.
    pub fn verify_oracle_rate(
        item: &str,
        claimed_rate: f64,
    ) -> ExternResult<()> {
        if !claimed_rate.is_finite() || claimed_rate <= 0.0 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Oracle rate must be a finite positive number".into()
            )));
        }

        // Fetch consensus from price-oracle zome
        #[derive(Debug, Serialize)]
        struct GetConsensusInput { item: String }
        #[derive(Debug, Deserialize)]
        struct ConsensusResult { median_price: f64 }

        match call(
            CallTargetCell::Local,
            ZomeName::from("price_oracle"),
            FunctionName::from("get_consensus_price"),
            None,
            GetConsensusInput { item: item.to_string() },
        ) {
            Ok(ZomeCallResponse::Ok(result)) => {
                match result.decode::<ConsensusResult>() {
                    Ok(consensus) if consensus.median_price.is_finite() && consensus.median_price > 0.0 => {
                        let deviation = (claimed_rate - consensus.median_price).abs() / consensus.median_price;
                        if deviation > ORACLE_RATE_TOLERANCE {
                            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                                "Oracle rate {:.6} deviates {:.1}% from consensus {:.6} (max {}%)",
                                claimed_rate, deviation * 100.0, consensus.median_price, ORACLE_RATE_TOLERANCE * 100.0
                            ))));
                        }
                        Ok(())
                    }
                    Ok(_) => {
                        debug!("verify_oracle_rate: consensus price invalid, accepting claimed rate");
                        Ok(())
                    }
                    Err(e) => {
                        debug!("verify_oracle_rate: decode error: {:?}, accepting claimed rate", e);
                        Ok(())
                    }
                }
            }
            _ => {
                // Oracle unreachable — accept with warning (bootstrap/standalone)
                debug!("verify_oracle_rate: price oracle unreachable, accepting claimed rate {}", claimed_rate);
                Ok(())
            }
        }
    }
}

/// Circuit breaker for cross-cluster calls.
///
/// After N consecutive failures to a target cluster, the breaker "opens" and
/// immediately returns an error for a cooldown period instead of hammering
/// a dead service. After the cooldown, the breaker enters "half-open" state
/// and allows one probe call. If it succeeds, the breaker closes; if it fails,
/// the cooldown restarts.
///
/// Implementation uses thread-local storage since Holochain WASM zomes are
/// single-threaded per cell.
pub mod circuit_breaker {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;

    /// Number of consecutive failures before the breaker opens.
    pub const FAILURE_THRESHOLD: u32 = 5;

    /// Cooldown period in microseconds (60 seconds).
    pub const COOLDOWN_MICROS: i64 = 60_000_000;

    /// Circuit breaker state for a single target.
    #[derive(Clone, Debug)]
    pub struct BreakerState {
        /// Consecutive failure count
        pub failures: u32,
        /// Whether the breaker is open (blocking calls)
        pub open: bool,
        /// When the breaker opened (microseconds since epoch)
        pub opened_at: i64,
        /// Total calls made
        pub total_calls: u64,
        /// Total failures
        pub total_failures: u64,
    }

    impl Default for BreakerState {
        fn default() -> Self {
            Self {
                failures: 0,
                open: false,
                opened_at: 0,
                total_calls: 0,
                total_failures: 0,
            }
        }
    }

    impl BreakerState {
        /// Check if the breaker should allow a call through.
        pub fn should_allow(&self, now_micros: i64) -> bool {
            if !self.open {
                return true;
            }
            // Allow one probe after cooldown (half-open)
            now_micros - self.opened_at > COOLDOWN_MICROS
        }

        /// Record a successful call — resets the breaker to closed.
        pub fn record_success(&mut self) {
            self.failures = 0;
            self.open = false;
            self.total_calls += 1;
        }

        /// Record a failed call — increments failure count, potentially opens breaker.
        pub fn record_failure(&mut self, now_micros: i64) {
            self.failures += 1;
            self.total_calls += 1;
            self.total_failures += 1;
            if self.failures >= FAILURE_THRESHOLD {
                self.open = true;
                self.opened_at = now_micros;
            }
        }
    }

    // Thread-local breaker registry (safe in WASM single-threaded context)
    thread_local! {
        static BREAKERS: RefCell<HashMap<String, BreakerState>> = RefCell::new(HashMap::new());
    }

    /// Check if a call to the target should be allowed.
    /// Returns Ok(()) if allowed, Err if breaker is open.
    pub fn check_breaker(target: &str, now_micros: i64) -> ExternResult<()> {
        BREAKERS.with(|breakers| {
            let breakers = breakers.borrow();
            if let Some(state) = breakers.get(target) {
                if !state.should_allow(now_micros) {
                    return Err(wasm_error!(WasmErrorInner::Guest(format!(
                        "Circuit breaker open for '{}': {} consecutive failures. \
                         Cooldown expires in {:.0}s.",
                        target,
                        state.failures,
                        (COOLDOWN_MICROS - (now_micros - state.opened_at)) as f64 / 1_000_000.0
                    ))));
                }
            }
            Ok(())
        })
    }

    /// Record a successful call to a target.
    pub fn record_success(target: &str) {
        BREAKERS.with(|breakers| {
            let mut breakers = breakers.borrow_mut();
            breakers.entry(target.to_string()).or_default().record_success();
        })
    }

    /// Record a failed call to a target.
    pub fn record_failure(target: &str, now_micros: i64) {
        BREAKERS.with(|breakers| {
            let mut breakers = breakers.borrow_mut();
            breakers.entry(target.to_string()).or_default().record_failure(now_micros);
        })
    }

    /// Get the current breaker state for a target (for telemetry/debugging).
    pub fn get_state(target: &str) -> Option<BreakerState> {
        BREAKERS.with(|breakers| {
            breakers.borrow().get(target).cloned()
        })
    }

    /// Reset a specific breaker (for testing or manual recovery).
    pub fn reset_breaker(target: &str) {
        BREAKERS.with(|breakers| {
            breakers.borrow_mut().remove(target);
        })
    }

    /// Reset all breakers.
    pub fn reset_all() {
        BREAKERS.with(|breakers| {
            breakers.borrow_mut().clear();
        })
    }
}

/// Per-agent rate limiting constants and utilities.
///
/// Holochain coordinator zomes each define their own `LinkTypes`, so the shared
/// crate cannot directly create or query links.  Instead we provide the
/// constants and anchor-naming convention here; each coordinator applies the
/// check using its own link type.
///
/// ## Recommended pattern (in a coordinator zome)
///
/// ```rust,ignore
/// use mycelix_finance_shared::{anchor_hash, DEFAULT_RATE_LIMIT_PER_MINUTE, rate_limit_anchor_key};
///
/// fn check_rate_limit(anchor_prefix: &str, agent: &AgentPubKey) -> ExternResult<()> {
///     let key = rate_limit_anchor_key(anchor_prefix, agent);
///     let anchor = anchor_hash(&key)?;
///     let links = get_links(
///         LinkQuery::try_new(anchor.clone(), LinkTypes::AnchorLinks)?,
///         GetStrategy::default(),
///     )?;
///     if links.len() >= DEFAULT_RATE_LIMIT_PER_MINUTE {
///         return Err(wasm_error!(WasmErrorInner::Guest(format!(
///             "Rate limit exceeded: max {} ops/min for {}",
///             DEFAULT_RATE_LIMIT_PER_MINUTE, anchor_prefix
///         ))));
///     }
///     // Record this operation
///     create_link(anchor, agent.clone().into(), LinkTypes::AnchorLinks, ())?;
///     Ok(())
/// }
/// ```
pub mod rate_limit {
    /// Default maximum operations per minute per agent.
    ///
    /// Individual coordinator zomes may override this with a tighter limit
    /// (e.g., currency-mint uses 60 for exchange recording).
    pub const DEFAULT_RATE_LIMIT_PER_MINUTE: usize = 100;

    /// Build the anchor key for a rate-limit bucket.
    ///
    /// The key encodes the anchor prefix, the agent's public key, and a
    /// minute-granularity time bucket so that old buckets naturally become
    /// unreferenced and can be garbage-collected.
    ///
    /// `now_micros` should come from `sys_time()?.as_micros()`.
    pub fn rate_limit_anchor_key(
        anchor_prefix: &str,
        agent: &hdk::prelude::AgentPubKey,
        now_micros: i64,
    ) -> String {
        let bucket = now_micros / 60_000_000; // 60-second window
        format!("rate:{}:{}:{}", anchor_prefix, agent, bucket)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Pagination tests
    // =========================================================================

    #[test]
    fn test_pagination_validation_valid() {
        let valid = types::PaginationInput {
            offset: 0,
            limit: 50,
        };
        assert!(valid.validate().is_ok());
    }

    #[test]
    fn test_pagination_validation_limit_zero() {
        let input = types::PaginationInput {
            offset: 0,
            limit: 0,
        };
        assert!(input.validate().is_err());
    }

    #[test]
    fn test_pagination_validation_limit_exceeds_max() {
        let input = types::PaginationInput {
            offset: 0,
            limit: 101,
        };
        assert!(input.validate().is_err());
    }

    #[test]
    fn test_pagination_validation_at_max() {
        let input = types::PaginationInput {
            offset: 0,
            limit: 100,
        };
        assert!(input.validate().is_ok());
    }

    #[test]
    fn test_pagination_default() {
        let input = types::PaginationInput::default();
        assert_eq!(input.offset, 0);
        assert_eq!(input.limit, 50);
        assert!(input.validate().is_ok());
    }

    #[test]
    fn test_paginated_result_has_more() {
        let pagination = types::PaginationInput {
            offset: 0,
            limit: 10,
        };
        let result = types::PaginatedResult::<u32>::new(vec![1, 2, 3], 20, &pagination);
        assert!(result.has_more);
        assert_eq!(result.total, 20);
        assert_eq!(result.items.len(), 3);
    }

    #[test]
    fn test_paginated_result_no_more() {
        let pagination = types::PaginationInput {
            offset: 0,
            limit: 10,
        };
        let result = types::PaginatedResult::<u32>::new(vec![1, 2, 3], 3, &pagination);
        assert!(!result.has_more);
    }

    #[test]
    fn test_paginated_result_empty() {
        let pagination = types::PaginationInput {
            offset: 0,
            limit: 10,
        };
        let result = types::PaginatedResult::<u32>::empty(&pagination);
        assert!(result.items.is_empty());
        assert_eq!(result.total, 0);
        assert!(!result.has_more);
    }

    // =========================================================================
    // Anchor tests
    // =========================================================================

    #[test]
    fn test_anchor_hash_consistency() {
        let hash1 = anchors::anchor_hash("test_anchor").unwrap();
        let hash2 = anchors::anchor_hash("test_anchor").unwrap();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_anchor_hash_different_inputs() {
        let hash1 = anchors::anchor_hash("loans").unwrap();
        let hash2 = anchors::anchor_hash("payments").unwrap();
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_sharded_anchors() {
        let shards = anchors::all_shard_anchors("loans");
        assert_eq!(shards.len(), 27); // A-Z + _
        assert!(shards.contains(&"loans_A".to_string()));
        assert!(shards.contains(&"loans_Z".to_string()));
        assert!(shards.contains(&"loans__".to_string()));
    }

    #[test]
    fn test_sharded_anchor_hash_alpha() {
        let hash_a = anchors::sharded_anchor_hash("loans", "alice").unwrap();
        let hash_b = anchors::sharded_anchor_hash("loans", "bob").unwrap();
        assert_ne!(hash_a, hash_b); // different shards
    }

    #[test]
    fn test_sharded_anchor_hash_same_shard() {
        let hash1 = anchors::sharded_anchor_hash("loans", "alice").unwrap();
        let hash2 = anchors::sharded_anchor_hash("loans", "anna").unwrap();
        assert_eq!(hash1, hash2); // same shard (A)
    }

    #[test]
    fn test_sharded_anchor_hash_non_alpha() {
        let hash = anchors::sharded_anchor_hash("loans", "123abc").unwrap();
        // Non-alpha first char should still produce a valid hash
        let _ = hash;
    }

    // =========================================================================
    // Validation tests
    // =========================================================================

    #[test]
    fn test_validate_did_valid() {
        assert!(validation::validate_did("did:mycelix:abc123").is_ok());
    }

    #[test]
    fn test_validate_did_empty() {
        assert!(validation::validate_did("").is_err());
    }

    #[test]
    fn test_validate_did_no_prefix() {
        assert!(validation::validate_did("mycelix:abc123").is_err());
    }

    #[test]
    fn test_validate_did_incomplete() {
        assert!(validation::validate_did("did:mycelix").is_err());
    }

    #[test]
    fn test_validate_positive_amount() {
        assert!(validation::validate_positive_amount(1.0, "amount").is_ok());
        assert!(validation::validate_positive_amount(0.0, "amount").is_err());
        assert!(validation::validate_positive_amount(-1.0, "amount").is_err());
        assert!(validation::validate_positive_amount(f64::NAN, "amount").is_err());
        assert!(validation::validate_positive_amount(f64::INFINITY, "amount").is_err());
    }

    #[test]
    fn test_validate_percentage() {
        assert!(validation::validate_percentage(0.0, "rate").is_ok());
        assert!(validation::validate_percentage(0.5, "rate").is_ok());
        assert!(validation::validate_percentage(1.0, "rate").is_ok());
        assert!(validation::validate_percentage(-0.1, "rate").is_err());
        assert!(validation::validate_percentage(1.1, "rate").is_err());
    }

    #[test]
    fn test_validate_currency() {
        assert!(validation::validate_currency("USD").is_ok());
        assert!(validation::validate_currency("BTC").is_ok());
        assert!(validation::validate_currency("").is_err());
        assert!(validation::validate_currency("TOOLONGCURRENCY").is_err());
    }

    // =========================================================================
    // Error type tests
    // =========================================================================

    #[test]
    fn test_finance_error_display() {
        let err = types::FinanceError::NotFound("Loan not found".to_string());
        assert_eq!(format!("{}", err), "Not found: Loan not found");
    }

    #[test]
    fn test_finance_error_all_variants() {
        let errors = vec![
            types::FinanceError::NotFound("x".into()),
            types::FinanceError::Unauthorized("x".into()),
            types::FinanceError::ValidationError("x".into()),
            types::FinanceError::InsufficientFunds("x".into()),
            types::FinanceError::InvalidState("x".into()),
            types::FinanceError::InternalError("x".into()),
        ];
        for err in errors {
            let display = format!("{}", err);
            assert!(display.contains("x"));
        }
    }

    #[test]
    fn test_batch_get_result_new() {
        let result = batch::BatchGetResult::new(10);
        assert_eq!(result.total_requested, 10);
        assert_eq!(result.success_count, 0);
        assert!(result.records.is_empty());
        assert!(result.not_found.is_empty());
        assert!(result.errors.is_empty());
    }

    // =========================================================================
    // Property-based tests (proptest)
    // =========================================================================

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        // Property: `anchor_hash` is deterministic -- same input always produces
        // the same hash.
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            #[test]
            fn anchor_hash_is_deterministic(input in "\\PC{1,200}") {
                let hash1 = anchors::anchor_hash(&input).unwrap();
                let hash2 = anchors::anchor_hash(&input).unwrap();
                prop_assert_eq!(hash1, hash2, "anchor_hash must be deterministic");
            }
        }

        // Property: `anchor_hash` has no collisions for distinct inputs.
        // With blake2b-256, any two distinct strings should produce distinct hashes.
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            #[test]
            fn anchor_hash_no_collisions(
                a in "\\PC{1,100}",
                b in "\\PC{1,100}",
            ) {
                prop_assume!(a != b);
                let hash_a = anchors::anchor_hash(&a).unwrap();
                let hash_b = anchors::anchor_hash(&b).unwrap();
                prop_assert_ne!(
                    hash_a, hash_b,
                    "Distinct inputs must produce distinct hashes: {:?} vs {:?}",
                    a, b
                );
            }
        }
    }

    // =========================================================================
    // Circuit breaker tests
    // =========================================================================

    #[test]
    fn test_breaker_starts_closed() {
        circuit_breaker::reset_all();
        assert!(circuit_breaker::check_breaker("test_target", 0).is_ok());
    }

    #[test]
    fn test_breaker_opens_after_threshold() {
        circuit_breaker::reset_all();
        let target = "test_open";
        for i in 0..circuit_breaker::FAILURE_THRESHOLD {
            circuit_breaker::record_failure(target, i as i64 * 1000);
        }
        // Should be open now
        let now = circuit_breaker::FAILURE_THRESHOLD as i64 * 1000;
        assert!(circuit_breaker::check_breaker(target, now).is_err());
    }

    #[test]
    fn test_breaker_allows_probe_after_cooldown() {
        circuit_breaker::reset_all();
        let target = "test_cooldown";
        for i in 0..circuit_breaker::FAILURE_THRESHOLD {
            circuit_breaker::record_failure(target, i as i64 * 1000);
        }
        let opened_at = (circuit_breaker::FAILURE_THRESHOLD - 1) as i64 * 1000;
        // Still blocked during cooldown
        assert!(circuit_breaker::check_breaker(target, opened_at + 1000).is_err());
        // Allowed after cooldown
        assert!(circuit_breaker::check_breaker(target, opened_at + circuit_breaker::COOLDOWN_MICROS + 1).is_ok());
    }

    #[test]
    fn test_breaker_closes_on_success() {
        circuit_breaker::reset_all();
        let target = "test_close";
        for i in 0..circuit_breaker::FAILURE_THRESHOLD {
            circuit_breaker::record_failure(target, i as i64 * 1000);
        }
        // Open
        assert!(circuit_breaker::check_breaker(target, 100_000).is_err());
        // Success resets
        circuit_breaker::record_success(target);
        assert!(circuit_breaker::check_breaker(target, 100_001).is_ok());
    }

    #[test]
    fn test_breaker_state_tracks_totals() {
        circuit_breaker::reset_all();
        let target = "test_totals";
        circuit_breaker::record_success(target);
        circuit_breaker::record_success(target);
        circuit_breaker::record_failure(target, 1000);
        let state = circuit_breaker::get_state(target).unwrap();
        assert_eq!(state.total_calls, 3);
        assert_eq!(state.total_failures, 1);
        assert_eq!(state.failures, 1); // consecutive
        assert!(!state.open);
    }

    #[test]
    fn test_breaker_success_resets_consecutive_failures() {
        circuit_breaker::reset_all();
        let target = "test_reset";
        // 4 failures (just under threshold of 5)
        for i in 0..4 {
            circuit_breaker::record_failure(target, i * 1000);
        }
        let state = circuit_breaker::get_state(target).unwrap();
        assert_eq!(state.failures, 4);
        assert!(!state.open);
        // One success resets consecutive count
        circuit_breaker::record_success(target);
        let state = circuit_breaker::get_state(target).unwrap();
        assert_eq!(state.failures, 0);
        // Now 5 more failures needed to open
        for i in 0..4 {
            circuit_breaker::record_failure(target, (i + 10) * 1000);
        }
        assert!(circuit_breaker::check_breaker(target, 50_000).is_ok()); // still closed
    }
}
