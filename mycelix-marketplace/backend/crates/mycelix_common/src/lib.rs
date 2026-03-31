// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Marketplace Common Utilities
//!
//! Shared functionality across all Mycelix zomes to reduce code duplication
//! and ensure consistent behavior.

use hdk::prelude::*;

/// Bridge integration for cross-app reputation sharing
///
/// The Bridge zome enables Mycelix Marketplace to participate in the broader
/// Holochain reputation ecosystem, allowing reputation to be portable across
/// applications.
pub mod bridge {
    use super::*;

    /// Bridge zome name - this should match the external Bridge DNA configuration
    pub const BRIDGE_ZOME_NAME: &str = "bridge";

    /// Weight for local reputation in combined score (60%)
    pub const LOCAL_REPUTATION_WEIGHT: f64 = 0.6;

    /// Weight for cross-app reputation in combined score (40%)
    pub const CROSS_APP_REPUTATION_WEIGHT: f64 = 0.4;

    /// Default reputation value when Bridge is unavailable or identity not found
    pub const DEFAULT_CROSS_APP_REPUTATION: f64 = 0.5;

    /// Cross-app reputation data from Bridge zome
    #[derive(Serialize, Deserialize, Debug, Clone)]
    pub struct CrossAppReputation {
        /// Decentralized identifier
        pub did: String,
        /// Aggregated reputation score across apps (0.0 - 1.0)
        pub score: f64,
        /// Number of apps contributing to this score
        pub app_count: u32,
        /// Total transactions across all apps
        pub total_transactions: u64,
        /// Timestamp of last update
        pub updated_at: u64,
    }

    /// Identity info from Bridge zome
    #[derive(Serialize, Deserialize, Debug, Clone)]
    pub struct BridgeIdentity {
        /// Decentralized identifier
        pub did: String,
        /// Whether identity is verified
        pub verified: bool,
        /// Associated agent public key
        pub agent_pubkey: AgentPubKey,
        /// Verification level (0 = unverified, 1 = email, 2 = phone, 3 = full KYC)
        pub verification_level: u8,
    }

    /// Input for reporting reputation to Bridge
    #[derive(Serialize, Deserialize, Debug, Clone)]
    pub struct ReportReputationInput {
        /// DID of the agent being reported
        pub did: String,
        /// Transaction outcome (positive contribution to reputation)
        pub positive: bool,
        /// Transaction value in cents
        pub value_cents: u64,
        /// Application identifier
        pub app_id: String,
        /// Additional context
        pub context: Option<String>,
    }

    /// Result of a Bridge call attempt
    #[derive(Debug, Clone)]
    pub enum BridgeResult<T> {
        /// Successfully received data from Bridge
        Success(T),
        /// Bridge is unavailable (not installed, offline, etc.)
        Unavailable,
        /// Error communicating with Bridge
        Error(String),
    }

    impl<T> BridgeResult<T> {
        /// Returns true if the Bridge call was successful
        pub fn is_success(&self) -> bool {
            matches!(self, BridgeResult::Success(_))
        }

        /// Returns the value if successful, or a default value otherwise
        pub fn unwrap_or(self, default: T) -> T {
            match self {
                BridgeResult::Success(v) => v,
                _ => default,
            }
        }

        /// Returns the value if successful, or computes it from a closure
        pub fn unwrap_or_else<F: FnOnce() -> T>(self, f: F) -> T {
            match self {
                BridgeResult::Success(v) => v,
                _ => f(),
            }
        }
    }

    /// Get cross-app reputation from Bridge zome
    ///
    /// This calls the Bridge zome's get_reputation function to retrieve
    /// aggregated reputation data for a given DID across all participating apps.
    ///
    /// # Arguments
    /// * `did` - Decentralized identifier of the agent
    ///
    /// # Returns
    /// * `BridgeResult<CrossAppReputation>` - The reputation data or unavailable status
    pub fn get_cross_app_reputation(did: &str) -> BridgeResult<CrossAppReputation> {
        let input = did.to_string();

        match call_bridge_zome::<String, CrossAppReputation>("get_reputation", input) {
            Ok(rep) => BridgeResult::Success(rep),
            Err(e) => {
                let error_msg = format!("{:?}", e);
                // Check if error indicates Bridge is not available vs actual error
                if error_msg.contains("not found")
                    || error_msg.contains("No DNA")
                    || error_msg.contains("unavailable")
                {
                    BridgeResult::Unavailable
                } else {
                    BridgeResult::Error(error_msg)
                }
            }
        }
    }

    /// Report a reputation event to Bridge zome
    ///
    /// After a transaction completes, this reports the outcome to the Bridge
    /// so it can be aggregated with reputation from other apps.
    ///
    /// # Arguments
    /// * `input` - The reputation report details
    ///
    /// # Returns
    /// * `BridgeResult<()>` - Success or unavailable/error status
    pub fn report_reputation(input: ReportReputationInput) -> BridgeResult<()> {
        match call_bridge_zome::<ReportReputationInput, ()>("report_reputation", input) {
            Ok(_) => BridgeResult::Success(()),
            Err(e) => {
                let error_msg = format!("{:?}", e);
                if error_msg.contains("not found")
                    || error_msg.contains("No DNA")
                    || error_msg.contains("unavailable")
                {
                    BridgeResult::Unavailable
                } else {
                    BridgeResult::Error(error_msg)
                }
            }
        }
    }

    /// Query identity information from Bridge zome
    ///
    /// This verifies a seller's identity through the Bridge's cross-app
    /// identity system.
    ///
    /// # Arguments
    /// * `agent` - The agent public key to look up
    ///
    /// # Returns
    /// * `BridgeResult<BridgeIdentity>` - The identity info or unavailable status
    pub fn query_identity(agent: &AgentPubKey) -> BridgeResult<BridgeIdentity> {
        match call_bridge_zome::<AgentPubKey, BridgeIdentity>("query_identity", agent.clone()) {
            Ok(identity) => BridgeResult::Success(identity),
            Err(e) => {
                let error_msg = format!("{:?}", e);
                if error_msg.contains("not found")
                    || error_msg.contains("No DNA")
                    || error_msg.contains("unavailable")
                    || error_msg.contains("Identity not found")
                {
                    BridgeResult::Unavailable
                } else {
                    BridgeResult::Error(error_msg)
                }
            }
        }
    }

    /// Compute combined reputation score using local and cross-app data
    ///
    /// This implements the weighted average formula:
    /// combined = (local_rep * 0.6) + (cross_app_rep * 0.4)
    ///
    /// If cross-app reputation is unavailable, falls back to local-only.
    ///
    /// # Arguments
    /// * `local_reputation` - The local MATL score (0.0 - 1.0)
    /// * `did` - Decentralized identifier for cross-app lookup
    ///
    /// # Returns
    /// * `(f64, bool)` - Combined score and whether cross-app data was used
    pub fn compute_combined_reputation(local_reputation: f64, did: &str) -> (f64, bool) {
        match get_cross_app_reputation(did) {
            BridgeResult::Success(cross_app) => {
                let combined = (local_reputation * LOCAL_REPUTATION_WEIGHT)
                    + (cross_app.score * CROSS_APP_REPUTATION_WEIGHT);
                (combined.clamp(0.0, 1.0), true)
            }
            _ => {
                // Bridge unavailable - use local reputation only
                // Note: We could also blend with default (0.5) but pure local is more conservative
                (local_reputation, false)
            }
        }
    }

    /// Internal helper to call Bridge zome functions
    fn call_bridge_zome<I, O>(function_name: &str, input: I) -> ExternResult<O>
    where
        I: serde::Serialize + std::fmt::Debug,
        O: serde::de::DeserializeOwned + std::fmt::Debug,
    {
        // Use call() for inter-zome calls within the same DNA
        // For cross-DNA calls, we'd use call_remote() with the bridge DNA hash
        // This implementation assumes Bridge is installed as a zome in the same DNA
        // or is accessible via a bridged connection
        let response = call(
            CallTargetCell::Local,
            ZomeName::from(BRIDGE_ZOME_NAME),
            FunctionName::from(function_name),
            None,
            input,
        )?;

        match response {
            ZomeCallResponse::Ok(extern_io) => extern_io.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to decode Bridge response: {:?}",
                    e
                )))
            }),
            ZomeCallResponse::Unauthorized(_, _, _, _) => Err(wasm_error!(WasmErrorInner::Guest(
                "Bridge call unauthorized".into()
            ))),
            ZomeCallResponse::NetworkError(e) => Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Bridge network error: {}",
                e
            )))),
            ZomeCallResponse::CountersigningSession(e) => {
                Err(wasm_error!(WasmErrorInner::Guest(format!(
                    "Bridge countersigning error: {}",
                    e
                ))))
            }
            ZomeCallResponse::AuthenticationFailed(_, _) => Err(wasm_error!(WasmErrorInner::Guest(
                "Bridge call authentication failed".into()
            ))),
        }
    }

    /// Convert an AgentPubKey to a DID string
    ///
    /// This creates a deterministic DID from the agent's public key.
    /// Format: did:holo:<base64_pubkey>
    pub fn agent_to_did(agent: &AgentPubKey) -> String {
        format!("did:holo:{}", agent)
    }
}

/// Error handling utilities
pub mod error_handling {
    use super::*;

    /// Centralized error conversion for to_app_option()
    ///
    /// This eliminates the repetitive pattern of:
    /// ```rust
    /// record.entry()
    ///     .to_app_option()
    ///     .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialization error: {:?}", e))))?
    ///     .ok_or(wasm_error!(WasmErrorInner::Guest("Could not deserialize entry".into())))?
    /// ```
    pub fn deserialize_entry<T>(record: &Record) -> ExternResult<T>
    where
        T: TryFrom<SerializedBytes, Error = SerializedBytesError>,
    {
        record
            .entry()
            .to_app_option()
            .map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Deserialization error: {:?}",
                    e
                )))
            })?
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Could not deserialize entry".into()
            )))
    }

    /// Deserialize entry from an Option<Record>
    ///
    /// Returns Ok(None) if record is None, Err if deserialization fails
    pub fn deserialize_optional_entry<T>(record: Option<Record>) -> ExternResult<Option<T>>
    where
        T: TryFrom<SerializedBytes, Error = SerializedBytesError>,
    {
        match record {
            Some(record) => deserialize_entry(&record).map(Some),
            None => Ok(None),
        }
    }
}

/// Remote call utilities
pub mod remote_calls {
    use super::*;

    /// Type-safe remote call wrapper
    ///
    /// Eliminates the repetitive pattern of:
    /// ```rust
    /// let current_agent = agent_info()?.agent_initial_pubkey;
    /// let response = call_remote(
    ///     current_agent.clone(),
    ///     ZomeName::from("zome_name"),
    ///     FunctionName::from("function_name"),
    ///     None,
    ///     input,
    /// )?;
    /// let result: MyType = match response {
    ///     ZomeCallResponse::Ok(extern_io) => extern_io.decode()
    ///         .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Failed to decode: {:?}", e))))?,
    ///     _ => return Err(wasm_error!(WasmErrorInner::Guest("Remote call failed".into()))),
    /// };
    /// ```
    pub fn call_zome<I, O>(zome_name: &str, function_name: &str, input: I) -> ExternResult<O>
    where
        I: serde::Serialize + std::fmt::Debug,
        O: serde::de::DeserializeOwned + std::fmt::Debug,
    {
        let current_agent = agent_info()?.agent_initial_pubkey;
        let response = call_remote(
            current_agent,
            ZomeName::from(zome_name),
            FunctionName::from(function_name),
            None,
            input,
        )?;

        match response {
            ZomeCallResponse::Ok(extern_io) => extern_io.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to decode response from {}.{}: {:?}",
                    zome_name, function_name, e
                )))
            }),
            _ => Err(wasm_error!(WasmErrorInner::Guest(
                format!("Remote call to {}.{} failed", zome_name, function_name).into()
            ))),
        }
    }

    /// Call remote zome without expecting a return value
    ///
    /// Useful for fire-and-forget operations like updating MATL scores
    pub fn call_zome_void<I>(zome_name: &str, function_name: &str, input: I) -> ExternResult<()>
    where
        I: serde::Serialize + std::fmt::Debug,
    {
        let current_agent = agent_info()?.agent_initial_pubkey;
        call_remote(
            current_agent,
            ZomeName::from(zome_name),
            FunctionName::from(function_name),
            None,
            input,
        )?;
        Ok(())
    }
}

/// Link query utilities
pub mod link_queries {
    use super::*;

    /// Get all links with the standard Local strategy
    ///
    /// Eliminates the repetitive pattern of:
    /// ```rust
    /// let links = get_links(
    ///     LinkQuery::try_new(base, LinkTypes::X)?,
    ///     GetStrategy::Local,
    /// )?;
    /// ```
    pub fn get_links_local(
        base: impl Into<AnyLinkableHash>,
        link_type: impl TryInto<LinkTypeFilter, Error = WasmError>,
    ) -> ExternResult<Vec<Link>> {
        get_links(LinkQuery::try_new(base, link_type)?, GetStrategy::Local)
    }

    /// Get all links and deserialize their targets as entries
    pub fn get_linked_entries<T>(
        base: impl Into<AnyLinkableHash>,
        link_type: impl TryInto<LinkTypeFilter, Error = WasmError>,
    ) -> ExternResult<Vec<T>>
    where
        T: TryFrom<SerializedBytes, Error = SerializedBytesError>,
    {
        let links = get_links_local(base, link_type)?;
        let mut entries = Vec::new();

        for link in links {
            if let Some(action_hash) = link.target.into_action_hash() {
                if let Some(record) = get(action_hash, GetOptions::default())? {
                    entries.push(crate::error_handling::deserialize_entry(&record)?);
                }
            }
        }

        Ok(entries)
    }
}

/// Validation utilities
pub mod validation {
    use super::*;

    /// Verify that the caller is one of the allowed agents
    pub fn verify_caller_is_one_of(allowed: &[AgentPubKey]) -> ExternResult<AgentPubKey> {
        let agent_info = agent_info()?;
        let caller = agent_info.agent_initial_pubkey;

        if allowed.contains(&caller) {
            Ok(caller)
        } else {
            Err(wasm_error!(WasmErrorInner::Guest(
                "Unauthorized: caller not in allowed list".into()
            )))
        }
    }

    /// Verify that the caller is the expected agent
    pub fn verify_caller_is(expected: &AgentPubKey) -> ExternResult<()> {
        let agent_info = agent_info()?;
        let caller = agent_info.agent_initial_pubkey;

        if caller == *expected {
            Ok(())
        } else {
            Err(wasm_error!(WasmErrorInner::Guest(
                "Unauthorized: caller is not the expected agent".into()
            )))
        }
    }
}

/// Time utilities
pub mod time {
    use super::*;

    /// Get current timestamp as u64 microseconds
    ///
    /// Eliminates the repetitive pattern of:
    /// ```rust
    /// sys_time()?.as_micros() as u64
    /// ```
    pub fn now_micros() -> ExternResult<u64> {
        Ok(sys_time()?.as_micros() as u64)
    }

    /// Get current timestamp
    pub fn now() -> ExternResult<Timestamp> {
        sys_time()
    }
}

/// Common result types and error enums
pub mod types {
    use super::*;

    /// Standard result type for Mycelix zomes
    pub type MResult<T> = Result<T, MError>;

    /// Common error types across Mycelix marketplace
    #[derive(Debug, Serialize, Deserialize)]
    pub enum MError {
        /// Resource not found
        NotFound(String),
        /// Caller not authorized for this operation
        Unauthorized(String),
        /// Invalid input provided
        InvalidInput(String),
        /// Rate limit exceeded
        RateLimited { retry_after: u64 },
        /// Insufficient MATL score for operation
        InsufficientMATL { have: f64, need: f64 },
        /// Internal error
        Internal(String),
    }

    impl From<MError> for WasmError {
        fn from(e: MError) -> Self {
            wasm_error!(WasmErrorInner::Guest(format!("{:?}", e)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These are unit tests for non-HDK dependent functions
    // Integration tests would require a Holochain conductor

    #[test]
    fn test_error_types() {
        use types::MError;

        let error = MError::NotFound("test".into());
        assert!(matches!(error, MError::NotFound(_)));

        let error = MError::InsufficientMATL {
            have: 0.5,
            need: 0.7,
        };
        assert!(matches!(error, MError::InsufficientMATL { .. }));
    }

    // ─── MError Variant Coverage ────────────────────────────────────

    #[test]
    fn test_error_unauthorized() {
        use types::MError;
        let error = MError::Unauthorized("forbidden".into());
        assert!(matches!(error, MError::Unauthorized(_)));
    }

    #[test]
    fn test_error_invalid_input() {
        use types::MError;
        let error = MError::InvalidInput("bad field".into());
        assert!(matches!(error, MError::InvalidInput(_)));
    }

    #[test]
    fn test_error_rate_limited() {
        use types::MError;
        let error = MError::RateLimited { retry_after: 60 };
        match error {
            MError::RateLimited { retry_after } => assert_eq!(retry_after, 60),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_error_internal() {
        use types::MError;
        let error = MError::Internal("panic".into());
        assert!(matches!(error, MError::Internal(_)));
    }

    #[test]
    fn test_error_all_variants_debug() {
        use types::MError;
        let errors: Vec<MError> = vec![
            MError::NotFound("item".into()),
            MError::Unauthorized("no auth".into()),
            MError::InvalidInput("bad".into()),
            MError::RateLimited { retry_after: 30 },
            MError::InsufficientMATL { have: 0.3, need: 0.5 },
            MError::Internal("oops".into()),
        ];
        for error in &errors {
            let debug_str = format!("{:?}", error);
            assert!(!debug_str.is_empty());
        }
    }

    // ─── Bridge Constants ───────────────────────────────────────────

    #[test]
    fn test_bridge_weights_sum_to_one() {
        let sum = bridge::LOCAL_REPUTATION_WEIGHT + bridge::CROSS_APP_REPUTATION_WEIGHT;
        assert!((sum - 1.0).abs() < 1e-10, "weights should sum to 1.0");
    }

    #[test]
    fn test_default_cross_app_reputation_neutral() {
        assert_eq!(bridge::DEFAULT_CROSS_APP_REPUTATION, 0.5);
    }

    #[test]
    fn test_bridge_zome_name() {
        assert_eq!(bridge::BRIDGE_ZOME_NAME, "bridge");
    }

    // ─── BridgeResult Tests ─────────────────────────────────────────

    #[test]
    fn test_bridge_result_is_success() {
        let result: bridge::BridgeResult<i32> = bridge::BridgeResult::Success(42);
        assert!(result.is_success());
    }

    #[test]
    fn test_bridge_result_unavailable_not_success() {
        let result: bridge::BridgeResult<i32> = bridge::BridgeResult::Unavailable;
        assert!(!result.is_success());
    }

    #[test]
    fn test_bridge_result_error_not_success() {
        let result: bridge::BridgeResult<i32> = bridge::BridgeResult::Error("fail".into());
        assert!(!result.is_success());
    }

    #[test]
    fn test_bridge_result_unwrap_or_success() {
        let result: bridge::BridgeResult<i32> = bridge::BridgeResult::Success(42);
        assert_eq!(result.unwrap_or(0), 42);
    }

    #[test]
    fn test_bridge_result_unwrap_or_unavailable() {
        let result: bridge::BridgeResult<i32> = bridge::BridgeResult::Unavailable;
        assert_eq!(result.unwrap_or(99), 99);
    }

    #[test]
    fn test_bridge_result_unwrap_or_error() {
        let result: bridge::BridgeResult<i32> = bridge::BridgeResult::Error("fail".into());
        assert_eq!(result.unwrap_or(-1), -1);
    }

    #[test]
    fn test_bridge_result_unwrap_or_else_success() {
        let result: bridge::BridgeResult<i32> = bridge::BridgeResult::Success(42);
        assert_eq!(result.unwrap_or_else(|| 0), 42);
    }

    #[test]
    fn test_bridge_result_unwrap_or_else_unavailable() {
        let result: bridge::BridgeResult<i32> = bridge::BridgeResult::Unavailable;
        assert_eq!(result.unwrap_or_else(|| 99), 99);
    }

    // ─── CrossAppReputation Construction ────────────────────────────

    #[test]
    fn test_cross_app_reputation_construction() {
        let rep = bridge::CrossAppReputation {
            did: "did:holo:abc123".into(),
            score: 0.85,
            app_count: 5,
            total_transactions: 150,
            updated_at: 1700000000,
        };
        assert_eq!(rep.did, "did:holo:abc123");
        assert_eq!(rep.score, 0.85);
        assert_eq!(rep.app_count, 5);
        assert_eq!(rep.total_transactions, 150);
    }

    // ─── ReportReputationInput Construction ─────────────────────────

    #[test]
    fn test_report_reputation_input_construction() {
        let input = bridge::ReportReputationInput {
            did: "did:holo:seller".into(),
            positive: true,
            value_cents: 5000,
            app_id: "marketplace".into(),
            context: Some("Successful purchase".into()),
        };
        assert!(input.positive);
        assert_eq!(input.value_cents, 5000);
        assert_eq!(input.app_id, "marketplace");
    }

    #[test]
    fn test_report_reputation_input_negative() {
        let input = bridge::ReportReputationInput {
            did: "did:holo:scammer".into(),
            positive: false,
            value_cents: 9999,
            app_id: "marketplace".into(),
            context: None,
        };
        assert!(!input.positive);
        assert!(input.context.is_none());
    }
}
